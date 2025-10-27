#!/usr/bin/env python
"""
run_batch.py
Run any training script multiple times from the CLI.
- Choose the script: --script cnn_training.py (or another .py)
- Choose parameter strategy: --mode grid | random
- Grid:   --grid batch_size=8,16,32 --grid lr=1e-3,1e-4 --grid epochs=5
- Random: --rand batch_size=choice(8,16,32,64,128) --rand lr=log10(-4,-1) --rand epochs=int(3,7)

Optional:
- Concurrency: --jobs 4
- Extra fixed args: --extra "--unify_experiments false --device cpu"

NEW (Schedulers + LR tracing):
- Forward a learning-rate scheduler to your training script:
    --scheduler {none,step,exp,cosine,onecycle,plateau}
    --lr_step_size 10 --lr_gamma 0.1 --t_max 5 --max_lr 0.1 --pct_start 0.3 --plateau_patience 3
- Ask training to log LR trace files here (one CSV per run):
    --lr_log_root path/to/lr_traces
  After runs finish, we parse those CSVs and write:
    lr_traces/<exp_id>.csv (produced by training)
    lr_traces/exp_<exp_id>.csv (alternate name)
    lr_traces/exp_<exp_id>_lr_stats.json  {"lr_min":..., "lr_max":...}

Examples
--------
Grid (30 runs) with cosine schedule and LR logging:
  python run_batch.py --script cnn_training.py \
    --mode grid \
    --grid batch_size=8,16,32,64,128 \
    --grid lr=1e-1,5e-2,1e-2,1e-3,5e-4,1e-4 \
    --grid epochs=5 \
    --scheduler cosine --t_max 5 \
    --lr_log_root ./lr_traces \
    --jobs 2

Random (100 runs) with StepLR:
  python run_batch.py --script cnn_training.py \
    --mode random --n 100 \
    --rand batch_size=choice(8,16,32,64,128) \
    --rand lr=log10(-4,-1) \
    --rand epochs=int(3,7) \
    --rand seed=int(0,100000) \
    --scheduler step --lr_step_size 5 --lr_gamma 0.5 \
    --lr_log_root ./lr_traces \
    --jobs 4
"""
from __future__ import annotations
import argparse, itertools, json, math, os, random, shlex, subprocess, sys, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------- parsing utils ----------------

def parse_grid_spec(items):
    """
    items like ["batch_size=8,16,32","lr=1e-3,1e-4","epochs=5"]
    -> {"batch_size":[8,16,32], "lr":[0.001,0.0001], "epochs":[5]}
    """
    grid = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"--grid expects key=val1,val2; got: {spec}")
        k, v = spec.split("=", 1)
        vals = []
        for tok in v.split(","):
            tok = tok.strip()
            try:
                vals.append(int(tok))
            except ValueError:
                try:
                    vals.append(float(tok))
                except ValueError:
                    vals.append(tok)
        grid[k] = vals
    return grid


def parse_rand_spec(items):
    """
    items like ["batch_size=choice(8,16,32)","lr=log10(-4,-1)","epochs=int(3,7)"]
    returns dict of callables: {"batch_size": <fn()>, "lr": <fn()> ...}
    Supported:
      choice(a,b,c)
      uniform(a,b)     -> float
      int(a,b)         -> int in [a,b]
      log10(a,b)       -> 10**uniform(a,b)  (a,b are exponents)
      fixed(value)     -> literal (string/number)
    """
    generators = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"--rand expects key=generator(...); got: {spec}")
        k, expr = spec.split("=", 1)
        expr = expr.strip()

        def gen_choice(args):  # args: "8,16,32"
            opts = [try_num(x) for x in args.split(",")]
            return lambda: random.choice(opts)

        def gen_uniform(args):
            a, b = [float(x) for x in args.split(",")]
            return lambda: random.uniform(a, b)

        def gen_int(args):
            a, b = [int(float(x)) for x in args.split(",")]
            return lambda: random.randint(a, b)

        def gen_log10(args):
            a, b = [float(x) for x in args.split(",")]
            return lambda: 10 ** random.uniform(a, b)

        def gen_fixed(args):
            return lambda: try_num(args)

        if expr.startswith("choice(") and expr.endswith(")"):
            fn = gen_choice(expr[7:-1])
        elif expr.startswith("uniform(") and expr.endswith(")"):
            fn = gen_uniform(expr[8:-1])
        elif expr.startswith("int(") and expr.endswith(")"):
            fn = gen_int(expr[4:-1])
        elif expr.startswith("log10(") and expr.endswith(")"):
            fn = gen_log10(expr[6:-1])
        elif expr.startswith("fixed(") and expr.endswith(")"):
            fn = gen_fixed(expr[6:-1])
        else:
            # allow bare literal e.g., epochs=5
            fn = gen_fixed(expr)
        generators[k] = fn
    return generators


def try_num(x):
    x = x.strip()
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


# ---------------- command builder ----------------

def build_cmd(script: Path, params: dict, extra: str | None, run_id: int, args) -> list[str]:
    """
    Build the training command. Also forwards LR scheduler flags and LR log hints.
    We will pass BOTH --run_id (for backward compat) AND --exp_id (for LR logging).
    """
    cmd = [sys.executable, str(script)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]

    # maintain original behavior
    cmd += ["--run_id", str(run_id)]

    # ---- Forward scheduler options (if enabled) ----
    if args.scheduler and args.scheduler != "none":
        cmd += ["--scheduler", args.scheduler,
                "--lr_step_size", str(args.lr_step_size),
                "--lr_gamma", str(args.lr_gamma)]
        if args.t_max is not None:
            cmd += ["--t_max", str(args.t_max)]
        if args.max_lr is not None:
            cmd += ["--max_lr", str(args.max_lr)]
        cmd += ["--pct_start", str(args.pct_start),
                "--plateau_patience", str(args.plateau_patience)]

    # ---- Pass LR logging hints to training (optional, but needed to compute lr_min/lr_max) ----
    if args.lr_log_root:
        lr_dir = Path(args.lr_log_root)
        lr_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--lr_log_dir", str(lr_dir), "--exp_id", str(run_id)]

    if extra:
        cmd += shlex.split(extra)
    return cmd


# ---------------- runner + summary ----------------

def run_one(cmd):
    print("→", " ".join(shlex.quote(c) for c in cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
        return {"cmd": cmd, "returncode": 0}
    except subprocess.CalledProcessError as e:
        return {"cmd": cmd, "returncode": e.returncode}


def compute_lr_stats_for_run(exp_id: int, lr_log_root: Path):
    """
    Look for LR trace CSV (either <id>.csv or exp_<id>.csv) inside lr_log_root,
    compute min/max, and write exp_<id>_lr_stats.json in the same directory.
    """
    candidates = [lr_log_root / f"{exp_id}.csv", lr_log_root / f"exp_{exp_id}.csv"]
    trace = None
    for p in candidates:
        if p.exists():
            trace = p
            break
    if trace is None:
        return False

    lrs = []
    try:
        with open(trace, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                if "lr" in r and r["lr"] not in ("", None):
                    lrs.append(float(r["lr"]))
                else:
                    for k in ("learning_rate", "lr_value", "LR"):
                        if k in r and r[k] not in ("", None):
                            lrs.append(float(r[k]))
                            break
        if not lrs:
            return False
        stats = {"lr_min": float(min(lrs)), "lr_max": float(max(lrs))}
        out = lr_log_root / f"exp_{exp_id}_lr_stats.json"
        out.write_text(json.dumps(stats, indent=2))
        print(f"✓ wrote {out}  (min={stats['lr_min']}, max={stats['lr_max']})")
        return True
    except Exception as e:
        print(f"[warn] failed to parse LR trace for exp {exp_id}: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser(description="Run any training script many times.")
    p.add_argument("--script", required=True, help="Path to the training .py file")
    p.add_argument("--mode", choices=["grid", "random"], default="grid")
    p.add_argument("--grid", action="append", default=[],
                   help="key=val1,val2 (repeatable). Used in --mode grid")
    p.add_argument("--rand", action="append", default=[],
                   help="key=generator(...) (repeatable). Used in --mode random")
    p.add_argument("--n", type=int, default=None, help="Number of runs (random mode)")
    p.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    p.add_argument("--extra", type=str, default="", help="Extra fixed CLI args passed to the script")
    p.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")

    # ===== Learning-rate scheduling (forwarded to training script) =====
    p.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "step", "exp", "cosine", "onecycle", "plateau"],
                   help="Learning rate scheduler to use inside the training script")
    p.add_argument("--lr_step_size", type=int, default=10, help="StepLR: epochs between drops")
    p.add_argument("--lr_gamma", type=float, default=0.1, help="Decay factor for Step/Exp/Plateau")
    p.add_argument("--t_max", type=int, default=None, help="CosineAnnealingLR: T_max (epochs)")
    p.add_argument("--max_lr", type=float, default=None, help="OneCycleLR: max learning rate")
    p.add_argument("--pct_start", type=float, default=0.3, help="OneCycleLR: warmup percentage")
    p.add_argument("--plateau_patience", type=int, default=3, help="ReduceLROnPlateau: patience (epochs)")
    p.add_argument("--lr_log_root", type=str, default=None,
                   help="If set, training script should write LR trace CSVs here; we'll compute lr_min/lr_max")

    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    script = Path(args.script)
    if not script.exists():
        raise SystemExit(f"Script not found: {script}")

    commands: list[tuple[int, list[str]]] = []

    if args.mode == "grid":
        grid = parse_grid_spec(args.grid)
        if not grid:
            raise SystemExit("Grid mode requires at least one --grid key=vals")
        keys = list(grid.keys())
        for run_id, combo in enumerate(itertools.product(*(grid[k] for k in keys)), start=1):
            params = {k: v for k, v in zip(keys, combo)}
            if "seed" not in params:
                params["seed"] = run_id
            cmd = build_cmd(script, params, args.extra, run_id, args)
            commands.append((run_id, cmd))

    else:  # random
        gens = parse_rand_spec(args.rand)
        if not gens:
            raise SystemExit("Random mode requires at least one --rand key=generator(...)")
        n = args.n or 50
        for run_id in range(1, n + 1):
            params = {k: fn() for k, fn in gens.items()}
            if "seed" not in params:
                params["seed"] = run_id
            cmd = build_cmd(script, params, args.extra, run_id, args)
            commands.append((run_id, cmd))

    # run with optional concurrency, collect results
    results = []
    if args.jobs and args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(run_one, cmd) for _, cmd in commands]
            for f in as_completed(futures):
                results.append(f.result())
    else:
        for _, cmd in commands:
            results.append(run_one(cmd))

    failed = [r for r in results if r["returncode"] != 0]

    print("\n=== Batch summary ===")
    print(f"Total runs: {len(results)}  |  Succeeded: {len(results) - len(failed)}  |  Failed: {len(failed)}")
    for r in failed:
        print("FAILED:", " ".join(r["cmd"]), "rc=", r["returncode"])

    # ===== Post-process LR traces into lr_min/lr_max =====
    if args.lr_log_root:
        lr_root = Path(args.lr_log_root)
        for run_id, _ in commands:
            compute_lr_stats_for_run(run_id, lr_root)

    # Optionally exit non-zero if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

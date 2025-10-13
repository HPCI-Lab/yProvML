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

Examples
--------
Grid (30 runs):
  python run_batch.py --script cnn_training.py \
    --mode grid \
    --grid batch_size=8,16,32,64,128 \
    --grid lr=1e-1,5e-2,1e-2,1e-3,5e-4,1e-4 \
    --grid epochs=5 \
    --jobs 2

Random (100 runs):
  python run_batch.py --script cnn_training.py \
    --mode random --n 100 \
    --rand batch_size=choice(8,16,32,64,128) \
    --rand lr=log10(-4,-1) \
    --rand epochs=int(3,7) \
    --rand seed=int(0,100000) \
    --jobs 4
"""
from __future__ import annotations
import argparse, itertools, json, math, os, random, shlex, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

def build_cmd(script, params: dict, extra: str | None, run_id: int):
    cmd = [sys.executable, str(script)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--run_id", str(run_id)]
    if extra:
        cmd += shlex.split(extra)
    return cmd

# ---------------- fault-tolerant runner + summary ----------------
def run_one(cmd):
    import subprocess, shlex
    print("→", " ".join(shlex.quote(c) for c in cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
        return {"cmd": cmd, "returncode": 0}
    except subprocess.CalledProcessError as e:
        return {"cmd": cmd, "returncode": e.returncode}

def main():
    p = argparse.ArgumentParser(description="Run any training script many times.")
    p.add_argument("--script", required=True, help="Path to the training .py file")
    p.add_argument("--mode", choices=["grid","random"], default="grid")
    p.add_argument("--grid", action="append", default=[],
                   help="key=val1,val2 (repeatable). Used in --mode grid")
    p.add_argument("--rand", action="append", default=[],
                   help="key=generator(...) (repeatable). Used in --mode random")
    p.add_argument("--n", type=int, default=None, help="Number of runs (random mode)")
    p.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    p.add_argument("--extra", type=str, default="", help="Extra fixed CLI args passed to the script")
    p.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    script = Path(args.script)
    if not script.exists():
        raise SystemExit(f"Script not found: {script}")

    commands = []

    if args.mode == "grid":
        grid = parse_grid_spec(args.grid)
        if not grid:
            raise SystemExit("Grid mode requires at least one --grid key=vals")
        keys = list(grid.keys())
        for run_id, combo in enumerate(itertools.product(*(grid[k] for k in keys)), start=1):
            params = {k: v for k, v in zip(keys, combo)}
            # default seed if not provided in grid
            if "seed" not in params:
                params["seed"] = run_id
            commands.append(build_cmd(script, params, args.extra, run_id))

    else:  # random
        gens = parse_rand_spec(args.rand)
        if not gens:
            raise SystemExit("Random mode requires at least one --rand key=generator(...)")
        n = args.n or 50
        for run_id in range(1, n+1):
            params = {k: fn() for k, fn in gens.items()}
            if "seed" not in params:
                params["seed"] = run_id
            commands.append(build_cmd(script, params, args.extra, run_id))

    # run with optional concurrency, collect results, and summarize
    results = []
    if args.jobs and args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(run_one, cmd) for cmd in commands]
            for f in as_completed(futures):
                results.append(f.result())
    else:
        for cmd in commands:
            results.append(run_one(cmd))

    failed = [r for r in results if r["returncode"] != 0]

    print("\n=== Batch summary ===")
    print(f"Total runs: {len(results)}  |  Succeeded: {len(results)-len(failed)}  |  Failed: {len(failed)}")
    for r in failed:
        print("FAILED:", " ".join(r["cmd"]), "rc=", r["returncode"])

    # Optionally exit non-zero if any failed
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()

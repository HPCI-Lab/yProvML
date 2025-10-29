import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc
except ImportError:
    nc = None


def choose_data_var(ds, prefer: Optional[str] = None) -> Optional[str]:
    """
    Return the best candidate metric variable name.
    Priority:
      1) --var if provided and numeric
      2) 'values' if numeric
      3) first numeric var that isn't obvious metadata
    """
    if prefer and prefer in ds.variables:
        v = ds.variables[prefer]
        if np.issubdtype(v.dtype, np.number):
            return prefer

    if "values" in ds.variables:
        v = ds.variables["values"]
        if np.issubdtype(v.dtype, np.number):
            return "values"

    bad = {"timestamps", "epochs", "lengths", "_FillValue"}
    for name, v in ds.variables.items():
        try:
            if name in bad:
                continue
            if np.issubdtype(v.dtype, np.number):
                return name
        except Exception:
            continue
    return None


def split_by_lengths_1d(values_1d: np.ndarray, lengths_1d: np.ndarray) -> List[np.ndarray]:
    chunks = []
    start = 0
    for L in lengths_1d:
        L = int(L)
        end = start + max(L, 0)
        chunks.append(values_1d[start:end])
        start = end
    return chunks


def last_finite(arr: np.ndarray) -> float:
    mask = np.isfinite(arr)
    if not mask.any():
        return np.nan
    return float(arr[np.where(mask)[0][-1]])


def read_metric_last_values(
    nc_path: Path,
    prefer_var: Optional[str] = None,
    debug: bool = False,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Return:
      per_exp_last: {exp: last_value}
      per_exp_epochs: {exp: epochs_count}
    Supports:
      - 2-D (exp,time) or (time,exp)
      - 1-D time with companion lengths(exp)
      - 1-D time with NO lengths  → treated as a single experiment (exp=0)
      - 0-D scalar                → treated as a single experiment (exp=0)
    """
    per_exp_last: Dict[int, float] = {}
    per_exp_epochs: Dict[int, int] = {}

    with nc.Dataset(nc_path, "r") as ds:
        if debug:
            print(f"[debug] {nc_path}")
            print("  dims:", list(dict(ds.dimensions).keys()))
            print("  vars:", list(ds.variables.keys()))

        vname = choose_data_var(ds, prefer=prefer_var)
        if vname is None:
            if debug:
                print("  (no numeric data var found)")
            return per_exp_last, per_exp_epochs

        v = ds.variables[vname]
        vals = np.array(v)

        # Case A: 2-D → normalize to (exp, time)
        if vals.ndim == 2:
            exp_dim_len = None
            if "exp" in ds.dimensions:
                exp_dim_len = len(ds.dimensions["exp"])
            if exp_dim_len is not None:
                if vals.shape[0] == exp_dim_len:
                    vt = vals
                elif vals.shape[1] == exp_dim_len:
                    vt = vals.T
                else:
                    vt = vals  # fallback: assume first axis is exp
            else:
                vt = vals

            exp_n = vt.shape[0]
            for e in range(exp_n):
                row = vt[e, :]
                per_exp_last[e] = last_finite(row)
                per_exp_epochs[e] = int(np.isfinite(row).sum())
            return per_exp_last, per_exp_epochs

        # Case B: 1-D with lengths(exp) companion
        if vals.ndim == 1 and "lengths" in ds.variables:
            lengths = np.array(ds.variables["lengths"]).astype(int)
            chunks = split_by_lengths_1d(vals, lengths)
            for e, chunk in enumerate(chunks):
                per_exp_last[e] = last_finite(chunk)
                per_exp_epochs[e] = int(np.isfinite(chunk).sum())
            return per_exp_last, per_exp_epochs

        # Case C: 1-D without lengths → treat as a single experiment
        if vals.ndim == 1:
            per_exp_last[0] = last_finite(vals)
            per_exp_epochs[0] = int(np.isfinite(vals).sum())
            return per_exp_last, per_exp_epochs

        # Case D: 0-D scalar → single experiment
        if vals.ndim == 0:
            val = float(vals) if np.isfinite(vals) else np.nan
            per_exp_last[0] = val
            per_exp_epochs[0] = 1 if np.isfinite(vals) else 0
            return per_exp_last, per_exp_epochs

        if debug:
            print(f"  (unsupported layout: {vals.ndim}D)")
        return per_exp_last, per_exp_epochs


def metric_name_from_filename(path: Path) -> str:
    name = path.stem
    if "_Context" in name:
        name = name.split("_Context", 1)[0]
    return name


def exp_id_from_path(path: Path, pattern: Optional[str], debug: bool = False) -> Optional[int]:
    """
    Extract an experiment id from the file path using a regex pattern that must
    have a capturing group for the integer id. Example default matches 'mnist_exp_17'.
    """
    text = str(path)
    if not pattern:
        pattern = r"mnist_exp_(\d+)"
    m = re.search(pattern, text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    if debug:
        print(f"[debug] Could not extract exp id from path with pattern {pattern!r}: {text}")
    return None


def build_runs_csv_from_yprov(input_path: str, out_csv: str, prefer_var: Optional[str] = None,
                              exp_pattern: Optional[str] = None, debug: bool = False):
    if nc is None:
        raise RuntimeError("netCDF4 is required. `pip install netCDF4`")

    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Collect candidate *.nc files
    nc_files = list(root.rglob("*.nc")) if root.is_dir() else [root]
    # Prefer unified GR0 files if present
    gr0 = [p for p in nc_files if p.name.endswith("_GR0.nc")]
    if gr0:
        nc_files = gr0

    if not nc_files:
        raise RuntimeError(f"No NetCDF metric files found under: {input_path}")

    per_exp_rows: Dict[int, Dict[str, float]] = {}
    per_exp_epochs_global: Dict[int, int] = {}

    for f in sorted(nc_files):
        try:
            metric_col = metric_name_from_filename(f)
            per_exp_last, per_exp_epochs = read_metric_last_values(
                f, prefer_var=prefer_var, debug=debug
            )

            if not per_exp_last:
                if debug:
                    print(f"[debug] Skipping {f.name}: no readable metric series")
                continue

            # If this file looks like "single experiment per file" (e==0 only),
            # try to assign a global exp id from the path (e.g., mnist_exp_17)
            path_exp = exp_id_from_path(f, exp_pattern, debug=debug)

            for e, val in per_exp_last.items():
                exp_key = path_exp if path_exp is not None else e
                if exp_key not in per_exp_rows:
                    per_exp_rows[exp_key] = {"exp": exp_key}
                per_exp_rows[exp_key][metric_col] = val

            for e, ep in per_exp_epochs.items():
                exp_key = path_exp if path_exp is not None else e
                prev = per_exp_epochs_global.get(exp_key, 0)
                if ep > prev:
                    per_exp_epochs_global[exp_key] = ep

        except Exception as ex:
            if debug:
                print(f"[debug] Error reading {f}: {ex}")
            continue

    if not per_exp_rows:
        raise RuntimeError(
            "Found NetCDFs, but no valid numeric data to aggregate.\n"
            "Try using --var <variable_name> and/or --debug to inspect files."
        )

    # Attach epochs if inferred
    for e, row in per_exp_rows.items():
        if per_exp_epochs_global:
            row["epochs"] = per_exp_epochs_global.get(e, np.nan)

    df = (
        pd.DataFrame([per_exp_rows[k] for k in sorted(per_exp_rows.keys())])
        .sort_values("exp")
        .reset_index(drop=True)
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote merged runs CSV → {out_csv}")
    print(f"Columns: {list(df.columns)}  |  Rows: {len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build runs_unified.csv from NetCDF metrics (supports 2-D, 1-D+lengths, and 1-D/0-D single-file experiments)."
    )
    ap.add_argument("--input", required=True, help="Path to metrics dir or prov dir")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--var", default=None, help="Force metric variable name (overrides auto-detect)")
    ap.add_argument("--exp-pattern", default=None,
                    help=r"Regex with a capturing group for exp id (default: mnist_exp_(\\d+))")
    ap.add_argument("--debug", action="store_true", help="Print per-file dim/var summaries for troubleshooting")
    args = ap.parse_args()
    build_runs_csv_from_yprov(args.input, args.out,
                              prefer_var=args.var,
                              exp_pattern=args.exp_pattern,
                              debug=args.debug)

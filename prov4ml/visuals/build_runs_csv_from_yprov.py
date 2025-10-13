import os
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc
except ImportError:
    nc = None


def choose_data_var(ds) -> Optional[str]:
    """
    Return the best candidate metric variable name.
    Prefer 'values' if present; else first numeric var that isn't obvious metadata.
    """
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
    """
    Split a 1-D time array into per-exp chunks using lengths(exp).
    """
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


def read_metric_last_values(nc_path: Path, debug: bool = False) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Return:
      per_exp_last: {exp: last_value}
      per_exp_epochs: {exp: epochs_count}
    Works with:
      - 2-D (exp,time)
      - 1-D time with companion lengths(exp)
    """
    per_exp_last: Dict[int, float] = {}
    per_exp_epochs: Dict[int, int] = {}

    with nc.Dataset(nc_path, "r") as ds:
        if debug:
            print(f"[debug] {nc_path.name}")
            print("  dims:", dict(ds.dimensions).keys())
            print("  vars:", list(ds.variables.keys()))

        vname = choose_data_var(ds)
        if vname is None:
            if debug:
                print("  (no numeric data var found)")
            return per_exp_last, per_exp_epochs

        v = ds.variables[vname]
        vals = np.array(v)

        # Case A: 2-D (exp, time) or (time, exp) – normalize to (exp, time)
        if vals.ndim == 2:
            # Guess which axis is exp: prefer axis whose length equals dim 'exp' if present
            exp_dim_len = None
            if "exp" in ds.dimensions:
                exp_dim_len = len(ds.dimensions["exp"])
            if exp_dim_len is not None:
                if vals.shape[0] == exp_dim_len:
                    vt = vals  # (exp, time)
                elif vals.shape[1] == exp_dim_len:
                    vt = vals.T  # (exp, time)
                else:
                    # fall back: assume first axis is exp
                    vt = vals
            else:
                vt = vals

            exp_n = vt.shape[0]
            for e in range(exp_n):
                row = vt[e, :]
                per_exp_last[e] = last_finite(row)
                per_exp_epochs[e] = int(np.isfinite(row).sum())
            return per_exp_last, per_exp_epochs

        # Case B: 1-D time with lengths(exp)
        if vals.ndim == 1 and "lengths" in ds.variables:
            lengths = np.array(ds.variables["lengths"]).astype(int)
            chunks = split_by_lengths_1d(vals, lengths)
            for e, chunk in enumerate(chunks):
                per_exp_last[e] = last_finite(chunk)
                per_exp_epochs[e] = int(np.isfinite(chunk).sum())
            return per_exp_last, per_exp_epochs

        # Unknown layout
        if debug:
            print(f"  (unsupported layout: {vals.ndim}D; missing lengths for 1-D)")
        return per_exp_last, per_exp_epochs


def metric_name_from_filename(path: Path) -> str:
    name = path.stem
    if "_Context" in name:
        name = name.split("_Context", 1)[0]
    return name


def build_runs_csv_from_yprov(input_path: str, out_csv: str, debug: bool = False):
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

    # Aggregate across files
    per_exp_rows: Dict[int, Dict[str, float]] = {}
    per_exp_epochs_global: Dict[int, int] = {}

    for f in sorted(nc_files):
        try:
            metric_col = metric_name_from_filename(f)
            per_exp_last, per_exp_epochs = read_metric_last_values(f, debug=debug)
            if not per_exp_last:
                if debug:
                    print(f"[debug] Skipping {f.name}: no readable metric series")
                continue

            # Merge metric column
            for e, val in per_exp_last.items():
                if e not in per_exp_rows:
                    per_exp_rows[e] = {"exp": e}
                per_exp_rows[e][metric_col] = val

            # Keep the first epochs info we can read (or update if longer)
            for e, ep in per_exp_epochs.items():
                prev = per_exp_epochs_global.get(e, 0)
                if ep > prev:
                    per_exp_epochs_global[e] = ep

        except Exception as ex:
            if debug:
                print(f"[debug] Error reading {f}: {ex}")
            continue

    if not per_exp_rows:
        # print a hint for debugging
        raise RuntimeError("Found NetCDFs, but no valid (exp,time) data variables to aggregate.")

    # Attach epochs if inferred
    for e, row in per_exp_rows.items():
        if per_exp_epochs_global:
            row["epochs"] = per_exp_epochs_global.get(e, np.nan)

    # Build DataFrame
    df = pd.DataFrame([per_exp_rows[k] for k in sorted(per_exp_rows.keys())]).sort_values("exp").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote merged runs CSV → {out_csv}")
    print(f"Columns: {list(df.columns)}  |  Rows: {len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build runs_unified.csv from unified NetCDF metrics (handles 2-D and 1-D+lengths).")
    ap.add_argument("--input", required=True, help="Path to metrics dir or prov dir (e.g., prov/experiment_name_0/metrics or prov)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--debug", action="store_true", help="Print per-file dim/var summaries for troubleshooting")
    args = ap.parse_args()
    build_runs_csv_from_yprov(args.input, args.out, debug=args.debug)

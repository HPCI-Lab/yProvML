# examples/prepare_runs/prepare_runs.py
"""
Dynamic unifier → CSV → visualize.

Usage examples
--------------
# 1) Point to a single NetCDF/Zarr file
python prepare_runs.py --input ../outputs/prov/unified/CE_train_Context.TRAINING_GR0.nc

# 2) Point to a directory (it will find ALL .nc/.zarr inside)
python prepare_runs.py --input ../outputs/prov/unified --outdir ../outputs

# 3) Only include certain metrics (name or prefix)
python prepare_runs.py --input ../outputs/prov/unified --include CE_train ACC_val
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import xarray as xr

# If your visualize_runs is elsewhere, adjust this import:
from visualize_runs import visualize_runs


def _metric_name(ds: xr.Dataset, path: Path) -> str:
    """Try dataset attr '_name', else derive from filename (strip '_Context...')."""
    m = (ds.attrs.get("_name") or "").strip()
    if m:
        return m
    return re.sub(r"_Context.*", "", path.stem)


def _context_name(ds: xr.Dataset, path: Path) -> str:
    """Try dataset attr '_context', else parse from filename."""
    c = (ds.attrs.get("_context") or "").strip()
    if c:
        return c
    m = re.search(r"_Context\.([A-Za-z0-9_.-]+)", path.stem)
    return m.group(1) if m else ""


def _extract_summary_from_ds(ds: xr.Dataset, metric: str, context: str) -> pd.DataFrame:
    """
    Expect dims: exp (E), time (T)
    Vars: values(exp,time), epochs(exp,time), timestamps(exp,time), lengths(exp)
    Produces one row per exp with the final valid value.
    """
    if not {"values", "epochs", "timestamps"} <= set(ds.variables):
        raise ValueError(f"Dataset missing required variables: {set(ds.variables)}")

    values = ds["values"].values
    epochs = ds["epochs"].values
    timestamps = ds["timestamps"].values
    if "lengths" in ds.variables:
        lengths = ds["lengths"].values
    else:
        lengths = np.full(values.shape[0], values.shape[1], dtype=int)

    E, T = values.shape
    rows = []
    for e in range(E):
        L = int(lengths[e]) if not (isinstance(lengths[e], float) and np.isnan(lengths[e])) else T
        L = max(0, min(L, T))
        if L == 0:
            rows.append({"exp": e, f"{metric}": np.nan, "_context": context})
            continue

        v = float(values[e, L - 1]) if not np.isnan(values[e, L - 1]) else np.nan
        rows.append({"exp": e, f"{metric}": v, "_context": context})

    return pd.DataFrame(rows).sort_values("exp").reset_index(drop=True)


def _gather_metric_files(input_path: Path) -> List[Path]:
    """If directory → find all .nc/.zarr recursively; if file → return [file]."""
    if input_path.is_file():
        if input_path.suffix not in (".nc", ".zarr"):
            raise SystemExit(f"Unsupported file type: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        files = list(input_path.rglob("*.nc")) + list(input_path.rglob("*.zarr"))
        if not files:
            raise SystemExit(f"No .nc or .zarr files found under: {input_path}")
        return files
    raise SystemExit(f"Input path not found: {input_path}")


def build_runs_csv_dynamic(
    input_path: str | Path,
    output_csv: str | Path,
    include_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build a wide per-exp CSV by merging final values from ALL metric files found.
    If include_metrics is provided, it filters metrics by name/prefix.
    """
    input_path = Path(input_path)
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = _gather_metric_files(input_path)

    summary_frames = []
    seen_metrics = set()

    for p in files:
        try:
            ds = xr.open_dataset(p) if p.suffix == ".nc" else xr.open_zarr(p)
        except Exception as e:
            print(f"[warn] Could not open {p}: {e}")
            continue

        try:
            metric = _metric_name(ds, p)
            context = _context_name(ds, p)
        except Exception as e:
            print(f"[warn] Could not derive names from {p}: {e}")
            ds.close()
            continue

        if include_metrics:
            if not any(metric == m or metric.startswith(m) for m in include_metrics):
                ds.close()
                continue

        try:
            summary = _extract_summary_from_ds(ds, metric, context)
        except Exception as e:
            print(f"[warn] Skipping {p.name}: {e}")
            ds.close()
            continue
        finally:
            ds.close()

        # reduce to exp + metric column
        slim = summary[["exp", metric]].copy()
        # if metric name already seen, disambiguate with trailing index
        base = metric
        k = 2
        while metric in seen_metrics and metric in slim.columns:
            metric = f"{base}_{k}"
            slim.rename(columns={base: metric}, inplace=True)
            k += 1
        seen_metrics.add(metric)

        summary_frames.append(slim)

    if not summary_frames:
        raise SystemExit("No valid metric summaries produced. Check your inputs or filters.")

    # outer-merge all summaries on 'exp'
    runs = summary_frames[0]
    for s in summary_frames[1:]:
        runs = runs.merge(s, on="exp", how="outer")

    runs = runs.sort_values("exp").reset_index(drop=True)
    runs.to_csv(out_csv, index=False)
    print(f"✅ Wrote merged runs CSV → {out_csv}")
    return runs


def main():
    ap = argparse.ArgumentParser(description="Dynamic unified metrics → CSV → visualize")
    ap.add_argument("--input", required=True,
                    help="Path to a single .nc/.zarr file OR a directory containing them")
    ap.add_argument("--outdir", default=".",
                    help="Where to write runs_unified.csv and figures (default: current dir)")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Optional metric names or prefixes to include (e.g., CE_train ACC_val)")
    ap.add_argument("--no_viz", action="store_true", help="Only build CSV, skip visualization")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs_csv = outdir / "runs_unified.csv"

    # 1) Build CSV dynamically from file or directory
    runs_df = build_runs_csv_dynamic(args.input, runs_csv, include_metrics=args.include)

    # 2) Visualize (if requested)
    if not args.no_viz:
        print("Generating visualizations…")
        visualize_runs(pd.read_csv(runs_csv), save_dir=args.outdir)
        print("✅ Visualizations complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
pareto_plus.py
- Labels Pareto-optimal points on the plot (accuracy vs cost/emissions).
- Clusters the Pareto-optimal runs (KMeans) and colors them on the plot.
- Exports CSVs for the Pareto front and the cluster report.

Usage (example):
  python prov4ml/analysis/pareto_plus.py \
    --runs_csv prov4ml/visuals/outputs/analysis/runs_enriched.csv \
    --outdir   prov4ml/visuals/outputs/analysis \
    --accuracy_col ACC_val \
    --cost_cols emissions energy_consumed train_epoch_time_ms \
    --n_clusters 3
"""
from __future__ import annotations
import argparse, os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def pick_id_col(df: pd.DataFrame) -> str:
    for c in ["exp", "run_id", "id"]:
        if c in df.columns:
            return c
    return None  # will use index


def nondominated_front(df: pd.DataFrame, acc_col: str, cost_col: str) -> pd.DataFrame:
    """
    Pareto: maximize accuracy, minimize cost.
    Keep points for which NO other point has (acc >=) and (cost <=) with one strict.
    """
    arr_acc = df[acc_col].to_numpy()
    arr_cost = df[cost_col].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        # strictly better if higher acc and lower/equal cost OR equal/higher acc and strictly lower cost
        dom = (arr_acc >= arr_acc[i]) & (arr_cost <= arr_cost[i]) & (
            (arr_acc > arr_acc[i]) | (arr_cost < arr_cost[i])
        )
        # if any other j dominates i, drop i
        if np.any(dom & (np.arange(len(df)) != i)):
            keep[i] = False
    return df.loc[keep]


def annotate_points(ax, xs, ys, labels, max_labels=40, fontsize=8):
    """
    Annotate up to max_labels points to avoid unreadable plots.
    """
    n = len(xs)
    if n == 0:
        return
    idx = np.arange(n)
    # If too many, pick a spread set
    if n > max_labels:
        sel = np.linspace(0, n - 1, max_labels).astype(int)
    else:
        sel = idx
    for i in sel:
        ax.annotate(str(labels[i]), (xs[i], ys[i]), xytext=(5, 3), textcoords="offset points", fontsize=fontsize, alpha=0.9)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def cluster_front(df_front: pd.DataFrame, acc_col: str, cost_col: str, n_clusters: int = 3):
    if not SKLEARN_OK or len(df_front) < max(2, n_clusters):
        return None, None
    X = df_front[[acc_col, cost_col]].to_numpy()
    km = KMeans(n_clusters=n_clusters, n_init="auto" if hasattr(KMeans(), "n_init") else 10, random_state=42)
    labels = km.fit_predict(X)
    return km, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--accuracy_col", default="ACC_val")
    ap.add_argument("--cost_cols", nargs="+", default=["emissions", "energy_consumed", "train_epoch_time_ms"])
    ap.add_argument("--n_clusters", type=int, default=3)
    ap.add_argument("--max_labels", type=int, default=40, help="Max point labels per plot")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.runs_csv)

    # Basic filters
    if args.accuracy_col not in df.columns:
        raise SystemExit(f"accuracy_col '{args.accuracy_col}' not found in {args.runs_csv}")

    id_col = pick_id_col(df)
    if id_col is None:
        df["_id_for_plot"] = df.index
        id_col = "_id_for_plot"

    # Try each cost col in order
    for cost_col in args.cost_cols:
        if cost_col not in df.columns:
            print(f"[skip] cost col '{cost_col}' not in CSV.")
            continue
        sub = df[[id_col, args.accuracy_col, cost_col] + [c for c in df.columns if c.startswith("param_")]].copy()
        sub = sub.dropna(subset=[args.accuracy_col, cost_col])
        # Need variation
        if len(sub) < 2 or sub[args.accuracy_col].nunique() < 2 or sub[cost_col].nunique() < 2:
            print(f"[skip] Not enough variation for '{cost_col}'.")
            continue

        # Pareto front
        front = nondominated_front(sub, args.accuracy_col, cost_col).sort_values([cost_col, args.accuracy_col], ascending=[True, False])
        if len(front) == 0:
            print(f"[warn] empty front for '{cost_col}'.")
            continue

        # Save front CSV
        fcsv = outdir / f"pareto_front_{safe_name(args.accuracy_col)}_vs_{safe_name(cost_col)}.csv"
        front.to_csv(fcsv, index=False)
        print(f"[write] {fcsv}")

        # Plot all + front (labeled)
        plt.figure(figsize=(7.2, 5.0))
        ax = plt.gca()
        # all points
        ax.scatter(sub[cost_col], sub[args.accuracy_col], s=30, alpha=0.25, label="All runs")
        # front points
        fx = front[cost_col].to_numpy()
        fy = front[args.accuracy_col].to_numpy()
        flabels = front[id_col].astype(str).to_list()
        scat = ax.scatter(fx, fy, s=55, marker="o", edgecolors="black", linewidths=0.8, alpha=0.95, label="Pareto front")
        annotate_points(ax, fx, fy, flabels, max_labels=args.max_labels)

        ax.set_xlabel(cost_col)
        ax.set_ylabel(args.accuracy_col)
        ax.set_title(f"Pareto: {args.accuracy_col} vs {cost_col}\n(labeled front)")
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        fpng = outdir / f"pareto_labeled_{safe_name(args.accuracy_col)}_vs_{safe_name(cost_col)}.png"
        plt.savefig(fpng, dpi=200)
        plt.close()
        print(f"[write] {fpng}")

        # Cluster the front
        km, clabels = cluster_front(front, args.accuracy_col, cost_col, n_clusters=args.n_clusters)
        if clabels is None:
            print(f"[info] skip clustering for '{cost_col}' (scikit-learn missing or too few points).")
            continue

        front2 = front.copy()
        front2["cluster"] = clabels
        fcsv2 = outdir / f"pareto_front_clusters_{safe_name(args.accuracy_col)}_vs_{safe_name(cost_col)}.csv"
        front2.to_csv(fcsv2, index=False)
        print(f"[write] {fcsv2}")

        # Cluster summary
        grp = front2.groupby("cluster")[[args.accuracy_col, cost_col]].agg(["count", "mean", "min", "max"])
        fcsv3 = outdir / f"pareto_front_cluster_report_{safe_name(args.accuracy_col)}_vs_{safe_name(cost_col)}.csv"
        grp.to_csv(fcsv3)
        print(f"[write] {fcsv3}")

        # Plot front colored by cluster (with labels)
        plt.figure(figsize=(7.2, 5.0))
        ax = plt.gca()
        for k in sorted(front2["cluster"].unique()):
            d = front2[front2["cluster"] == k]
            ax.scatter(d[cost_col], d[args.accuracy_col], s=65, alpha=0.95, label=f"cluster {k}", edgecolors="black", linewidths=0.6)
            # label each point (cap at max_labels per plot)
            annotate_points(ax, d[cost_col].to_numpy(), d[args.accuracy_col].to_numpy(), d[id_col].astype(str).to_list(), max_labels=args.max_labels)

        ax.set_xlabel(cost_col)
        ax.set_ylabel(args.accuracy_col)
        ax.set_title(f"Pareto front clusters: {args.accuracy_col} vs {cost_col}")
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        fpng2 = outdir / f"pareto_front_clusters_{safe_name(args.accuracy_col)}_vs_{safe_name(cost_col)}.png"
        plt.savefig(fpng2, dpi=200)
        plt.close()
        print(f"[write] {fpng2}")


if __name__ == "__main__":
    main()

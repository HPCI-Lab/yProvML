#!/usr/bin/env python
"""
visualize_clusters.py
Visualize clustering results from yProv4ML analysis.

- Loads `runs_enriched.csv` and `cluster_report.csv`
- Plots cluster membership over accuracy/emission space
- Shows parameter distributions per cluster
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_csv", required=True,
                        help="Path to runs_enriched.csv (with params + metrics)")
    parser.add_argument("--cluster_csv", required=True,
                        help="Path to cluster_report.csv (produced by yProv4ML)")
    parser.add_argument("--outdir", required=True,
                        help="Output folder for visualizations")
    parser.add_argument("--acc_col", default="ACC_val")
    parser.add_argument("--loss_col", default="MSE_val")
    parser.add_argument("--emission_col", default="emissions")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.runs_csv)
    clusters = pd.read_csv(args.cluster_csv)

    # --- Merge on run_id or exp ---
    key = "exp" if "exp" in df.columns else "run_id"
    if key not in clusters.columns and "cluster" in clusters.columns:
        df["cluster"] = clusters["cluster"]
    elif "cluster" in clusters.columns:
        df = df.merge(clusters[["cluster"]], left_on=key, right_index=True, how="left")
    else:
        raise SystemExit("No cluster column found in cluster_report.csv")

    if "cluster" not in df.columns:
        raise SystemExit("Cluster column not merged correctly")

    print(f"[info] Found {df['cluster'].nunique()} clusters in the data")

    # --- Scatter plot: Accuracy vs Emissions colored by cluster ---
    if args.acc_col in df.columns and args.emission_col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x=args.emission_col, y=args.acc_col,
                        hue="cluster", palette="tab10", s=80, edgecolor="black")
        plt.title(f"Accuracy vs Emissions by Cluster")
        plt.xlabel(args.emission_col)
        plt.ylabel(args.acc_col)
        plt.tight_layout()
        plt.savefig(outdir / "clusters_acc_vs_emissions.png", dpi=200)
        plt.close()
        print(f"[write] clusters_acc_vs_emissions.png")

    # --- Scatter plot: Accuracy vs Loss colored by cluster ---
    if args.acc_col in df.columns and args.loss_col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x=args.loss_col, y=args.acc_col,
                        hue="cluster", palette="tab10", s=80, edgecolor="black")
        plt.title(f"Accuracy vs Loss by Cluster")
        plt.xlabel(args.loss_col)
        plt.ylabel(args.acc_col)
        plt.tight_layout()
        plt.savefig(outdir / "clusters_acc_vs_loss.png", dpi=200)
        plt.close()
        print(f"[write] clusters_acc_vs_loss.png")

    # --- Parameter distribution per cluster ---
    param_cols = [c for c in df.columns if c.startswith("param_")]
    if param_cols:
        for col in param_cols:
            plt.figure(figsize=(7, 4))
            sns.countplot(data=df, x=col, hue="cluster", palette="tab10")
            plt.title(f"Parameter Distribution: {col}")
            plt.tight_layout()
            fname = outdir / f"param_distribution_{col}.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"[write] {fname}")

    # --- Cluster summary heatmap (mean per cluster) ---
    summary = df.groupby("cluster")[[args.acc_col, args.loss_col, args.emission_col]].mean()
    plt.figure(figsize=(6, 3))
    sns.heatmap(summary, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Cluster-wise Mean Metrics")
    plt.tight_layout()
    plt.savefig(outdir / "cluster_means_heatmap.png", dpi=200)
    plt.close()
    print(f"[write] cluster_means_heatmap.png")


if __name__ == "__main__":
    main()

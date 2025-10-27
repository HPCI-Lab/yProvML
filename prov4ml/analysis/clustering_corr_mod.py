from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except Exception:
    StandardScaler = None
    KMeans = None

def cluster_outputs(runs: pd.DataFrame, outdir: Path, n_clusters: int = 3):
    out_cols = [c for c in runs.columns if any(k in c.lower() for k in
                 ["acc","loss","emission","co2","carbon","energy","cost"])]
    usable = [c for c in out_cols if c in runs.columns and np.issubdtype(runs[c].dtype, np.number)]
    if len(usable) < 2: return
    X = runs[usable].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1)
    if X.shape[1] < 2: return
    X = X.fillna(X.median(numeric_only=True))
    try:
        Xs = StandardScaler().fit_transform(X.values) if StandardScaler else X.values
        Z = linkage(Xs, method="ward")
        plt.figure(figsize=(10,6))
        labels = [f"exp{int(e)}" if "exp" in runs.columns and not pd.isna(e) else "?" for e in runs.get("exp", pd.Series(range(len(runs))))]
        dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
        plt.title("Dendrogram (outputs: acc/loss/emissions/energy/cost)")
        plt.tight_layout()
        plt.savefig(outdir / "dendrogram_outputs.png", dpi=300)
        plt.close()
        labs = fcluster(Z, t=n_clusters, criterion='maxclust')
        runs = runs.copy()
        runs["out_cluster"] = labs
    except Exception:
        if KMeans is None or StandardScaler is None: return
        Xs = StandardScaler().fit_transform(X.values)
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        labs = km.fit_predict(Xs)
        runs = runs.copy()
        runs["out_cluster"] = labs + 1
    report = runs.groupby("out_cluster").median(numeric_only=True).reset_index()
    report.to_csv(outdir / "cluster_report.csv", index=False)

def export_correlations(runs: pd.DataFrame, outdir: Path):
    num_cols = runs.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2: return
    corr = runs[num_cols].corr(method="spearman")
    corr.to_csv(outdir / "correlations_spearman.csv", index=True)
    plt.figure(figsize=(10,8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Spearman ρ")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(num_cols)), num_cols, fontsize=7)
    plt.title("Correlation matrix (Spearman)")
    plt.tight_layout()
    plt.savefig(outdir / "correlations_spearman.png", dpi=300)
    plt.close()

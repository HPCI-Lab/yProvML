# prov4ml/visuals/visualize_runs.py
import os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# SHAP is optional; we fail gracefully if missing
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Parameters we’re allowed to use for PCA/Clustering (if present)
PARAM_COLS_CANDIDATES = [
    "param_batch_size", "param_epochs", "param_lr", "param_seed", "epochs"
]

def _available_params(df: pd.DataFrame):
    return [
        c for c in PARAM_COLS_CANDIDATES
        if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]

# Ignore seed for annotations unless it's the only thing that differs
IGNORE_FOR_ANNOT = {"param_seed"}
TOP_K_CHANGES = 2  # show up to 2 params per merge label

def _cluster_summary(df, idxs, cols):
    """Return per-cluster median values for readability."""
    sub = df.loc[idxs, cols]
    return sub.median(numeric_only=True)

def _format_change(name, left_val, right_val):
    def is_intlike(x):
        try:
            return float(x).is_integer()
        except Exception:
            return False
    if is_intlike(left_val) and is_intlike(right_val):
        return f"{name} {int(left_val)}→{int(right_val)}"
    diff = float(right_val) - float(left_val)
    return f"{name} Δ{diff:.4g} (L={float(left_val):.4g}, R={float(right_val):.4g})"

def _top_param_changes(df, left_idx, right_idx, param_cols):
    """Compute top-K changing params between two clusters."""
    left = _cluster_summary(df, left_idx, param_cols)
    right = _cluster_summary(df, right_idx, param_cols)
    diffs = (left - right).abs().sort_values(ascending=False)
    diffs = diffs[diffs > 0]  # non-zero only

    if diffs.empty:
        return ["no-change"]

    non_seed = [p for p in diffs.index if p not in IGNORE_FOR_ANNOT]
    chosen = non_seed[:TOP_K_CHANGES] if non_seed else list(diffs.index[:TOP_K_CHANGES])

    labels = []
    for p in chosen:
        labels.append(_format_change(p, left[p], right[p]))
    return labels


def visualize_runs(df: pd.DataFrame, save_dir: str = "prov4ml/visuals/outputs", n_clusters: int = 3):
    os.makedirs(save_dir, exist_ok=True)
    print("→ Running PCA + Dendrogram (params-only); accuracy shown as labels (not used for PCA)")
    print(f"→ Saving figures to: {save_dir}")

    # --- pick parameter columns actually present ---
    param_cols = _available_params(df)
    if not param_cols:
        raise ValueError(
            "No usable parameter columns found. "
            f"Looked for any of: {PARAM_COLS_CANDIDATES}"
        )

    # Accuracy column (optional; shown on labels but NOT used for PCA/clustering)
    acc_col = "ACC_val" if "ACC_val" in df.columns else None

    # --- features for PCA/Clustering = parameters only ---
    X = df[param_cols].copy()
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_scaled)
    df = df.copy()
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]

    # --- Only show legend entries for parameters that actually vary ---
    def _varies(col):
        vals = pd.unique(df[col].dropna())
        return len(vals) > 1

    varying_bs = "param_batch_size" in df.columns and _varies("param_batch_size")

    # ===== PCA PLOT =====
    plt.figure(figsize=(7.8, 6.2))

    # color by accuracy if present; otherwise fall back to param_lr (then first param)
    color_col = (
        "ACC_val"
        if ("ACC_val" in df.columns and pd.api.types.is_numeric_dtype(df["ACC_val"]))
        else ("param_lr" if "param_lr" in param_cols else param_cols[0])
    )

    # Marker by batch size (fallback to a single marker)
    marker_col = "param_batch_size" if "param_batch_size" in param_cols else None
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    if marker_col:
        unique_bs = sorted(df[marker_col].unique())
        marker_map = {bs: marker_cycle[i % len(marker_cycle)] for i, bs in enumerate(unique_bs)}
    else:
        unique_bs, marker_map = [], {}

    # Draw grouped by marker so we get legend entries
    last_scatter = None
    for bs_val in (unique_bs if marker_col else [None]):
        mask = (df[marker_col] == bs_val) if marker_col else np.ones(len(df), dtype=bool)
        last_scatter = plt.scatter(
            df.loc[mask, "PC1"], df.loc[mask, "PC2"],
            c=df.loc[mask, color_col],
            cmap="viridis",
            s=70,
            marker=marker_map.get(bs_val, "o"),
            edgecolor="k",
            linewidths=0.5,
            alpha=0.9,
            label=(f"batch_size={bs_val}" if marker_col else None),
        )

    # colorbar for chosen color_col
    if last_scatter is not None:
        cbar = plt.colorbar(last_scatter)
        cbar.set_label(color_col)

    # Legend for batch_size markers
    marker_handles = []
    if varying_bs and marker_col:
        for bs in unique_bs:
            marker_handles.append(
                mlines.Line2D([], [], color="black", marker=marker_map[bs],
                              linestyle="None", markersize=8, label=f"batch_size={bs}")
            )

    handles = []
    if marker_handles:
        handles.extend(marker_handles)
    if handles:
        plt.legend(handles=handles, loc="best", frameon=True)

    # point labels: show exp + ACC (if available)
    for _, row in df.iterrows():
        label = f"exp{int(row['exp'])}"
        if acc_col is not None and pd.notna(row[acc_col]):
            label += f" | acc={row[acc_col]:.3f}"
        # (epochs removed from legend per your request; we keep it out of labels too)
        plt.text(row["PC1"] + 0.02, row["PC2"], label, fontsize=8)

    plt.title("PCA of Experiments (params only)\n(Accuracy in labels; not used for PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    # Caption with the params actually used in PCA
    used_params = ", ".join(param_cols)
    caption = f"PCA features: {used_params}"
    plt.gcf().text(
        0.99, 0.02,
        textwrap.fill(caption, width=80),
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.6)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_runs.png"), dpi=300)
    plt.close()
    print("✅ Saved pca_runs.png")

    # ===== DENDROGRAM =====
    Z = linkage(X_scaled, method="ward")

    plt.figure(figsize=(9, 5))
    dn = dendrogram(
        Z,
        labels=[f"exp{int(e)}" for e in df["exp"]],
        leaf_rotation=90,
        leaf_font_size=9,
        above_threshold_color="gray",
    )
    plt.title("Dendrogram of Experiment Similarities (params only)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dendrogram_runs.png"), dpi=300)
    print("✅ Saved dendrogram_runs.png")

    # --- Annotate each merge with top-K parameter changes ---
    n = len(df)
    clusters = {i: [i] for i in range(n)}  # start with individual observations

    for i, (left, right, height, _count) in enumerate(Z):
        left = int(left); right = int(right)
        left_members  = clusters[left]  if left  >= n else [left]
        right_members = clusters[right] if right >= n else [right]
        clusters[n + i] = left_members + right_members

        changes = _top_param_changes(df, left_members, right_members, param_cols)
        label = "\n".join(changes)  # multi-line if showing 2 params

        # place annotation at the middle of this merge segment
        x_coords = dn["icoord"][i]
        y_coords = dn["dcoord"][i]
        x_mid = (x_coords[1] + x_coords[2]) / 2.0
        y_mid = y_coords[1]

        plt.annotate(
            label,
            xy=(x_mid, y_mid),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
        )

    plt.savefig(os.path.join(save_dir, "dendrogram_runs_annotated.png"), dpi=300)
    plt.close()
    print("✅ Saved dendrogram_runs_annotated.png")

    # ===== SUPERVISED CLUSTER EXPLANATION (Surrogate model + SHAP) =====
    print(f"→ Explaining cluster membership with a surrogate (RandomForest) + SHAP; n_clusters={n_clusters}")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    df["cluster"] = labels
        # ===== CLUSTER SUMMARY REPORT =====
    print("→ Building cluster_report.csv")
    cluster_report = []
    grouped = df.groupby("cluster")
    for cluster_id, sub in grouped:
        row = {"cluster": cluster_id, "n_experiments": len(sub)}
        # Median or mean for key numeric columns
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.number):
                row[f"median_{c}"] = np.median(sub[c].dropna())
                row[f"mean_{c}"] = np.mean(sub[c].dropna())
        cluster_report.append(row)

    report_df = pd.DataFrame(cluster_report).sort_values("cluster")
    csv_path = os.path.join(save_dir, "cluster_report.csv")
    report_df.to_csv(csv_path, index=False)
    print(f"✅ Saved cluster_report.csv → {csv_path}")


    X_params = df[param_cols].values
    y_clusters = df["cluster"].values

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=0,
        class_weight="balanced"
    )
    clf.fit(X_params, y_clusters)

    # Export a simple text summary (feature importances) even if SHAP is missing
    importances = pd.Series(clf.feature_importances_, index=param_cols).sort_values(ascending=False)
    importances.to_csv(os.path.join(save_dir, "surrogate_feature_importances.csv"))
    print("✅ Saved surrogate_feature_importances.csv")

    if not _HAS_SHAP:
        print("[warn] SHAP not installed; skipping SHAP explanations. pip install shap")
        return

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_params)   # list: one array per class
        # Global importance: mean(|SHAP|) aggregated across classes
        import numpy as _np
        abs_per_class = [_np.abs(sv) for sv in shap_values]  # [C x N x F]
        mean_abs = _np.mean(_np.stack([a.mean(axis=0) for a in abs_per_class], axis=0), axis=0)  # [F]
        feature_names = param_cols

        # (a) Bar plot (global)
        plt.figure(figsize=(7.5, 5))
        order = _np.argsort(mean_abs)[::-1]
        plt.bar([feature_names[i] for i in order], mean_abs[order])
        plt.xticks(rotation=30, ha="right")
        plt.title(f"SHAP global importance for cluster membership (n_clusters={n_clusters})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_clusters_bar.png"), dpi=300)
        plt.close()

        # (b) Beeswarm for the largest cluster
        largest_label = pd.Series(labels).value_counts().idxmax()
        unique_sorted = sorted(pd.unique(labels))
        class_index = unique_sorted.index(largest_label)

        shap.summary_plot(
            shap_values[class_index],
            features=X_params,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP beeswarm (cluster={largest_label})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_beeswarm_largest_cluster.png"), dpi=300)
        plt.close()

        # (c) Dependence plots for top 3 features
        top3 = [feature_names[i] for i in order[:3]]
        for feat in top3:
            idx = feature_names.index(feat)
            shap.dependence_plot(
                idx,
                shap_values[class_index],
                X_params,
                feature_names=feature_names,
                show=False
            )
            safe = feat.replace("/", "_")
            plt.title(f"SHAP dependence for '{feat}' (cluster={largest_label})")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"shap_dependence_{safe}.png"), dpi=300)
            plt.close()

        print("✅ Saved SHAP cluster explanations → "
              f"{save_dir}/shap_clusters_bar.png, shap_beeswarm_largest_cluster.png, shap_dependence_*.png")
    except Exception as e:
        print(f"[warn] SHAP cluster explanations failed: {e}")

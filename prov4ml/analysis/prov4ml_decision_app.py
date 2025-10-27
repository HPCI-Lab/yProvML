
# prov4ml_decision_app.py
# Streamlit Decision-Support Dashboard for yProv4ML experiments
#
# Run:
#   streamlit run prov4ml_decision_app.py
#
# Features
# - Load enriched CSV (defaults to prov4ml/visuals/outputs/analysis/runs_enriched.csv)
# - Auto-detect id column, param columns (param_*), numeric metrics
# - Choose accuracy metric and cost metric
# - Compute & plot Pareto front (maximize accuracy, minimize cost) with labels
# - KMeans clustering (both full dataset and Pareto-only), visualized
# - Descriptive analytics (summary, correlations)
# - Prescriptive recommendations (top trade-offs, suggested parameter ranges & next experiments)
#
# Notes
# - Requires: streamlit, pandas, numpy, scikit-learn, plotly
# - We use Plotly for interactivity (hover tooltips, selection)
# - Column name defaults match your earlier runs: ACC_val, emissions, energy_consumed, train_epoch_time_ms

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import plotly.express as px
import plotly.graph_objects as go

try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ----------------------------- helpers -----------------------------
def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["exp", "run_id", "id", "experiment", "run"]:
        if c in df.columns:
            return c
    return None

def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def param_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("param_")]

def candidate_accuracy_cols(df: pd.DataFrame) -> List[str]:
    cands = [c for c in df.columns if re.search(r"(acc|accuracy)", c, re.I)]
    # Put common defaults first
    if "ACC_val" in cands:
        cands.remove("ACC_val")
        cands = ["ACC_val"] + cands
    return cands or [c for c in numeric_cols(df) if "loss" not in c.lower()][:5]

def candidate_cost_cols(df: pd.DataFrame) -> List[str]:
    keys = ["emission", "co2", "energy", "time", "latency", "cost", "power"]
    cands = [c for c in df.columns if any(k in c.lower() for k in keys)]
    # promote known names first
    order = ["emissions", "emissions_gCO2eq", "energy_consumed", "energy_J", "train_epoch_time_ms"]
    ranked = [c for c in order if c in df.columns]
    ranked += [c for c in cands if c not in ranked]
    return ranked

def nondominated_front(df: pd.DataFrame, acc_col: str, cost_col: str) -> pd.DataFrame:
    """Maximize accuracy, minimize cost."""
    arr_acc = df[acc_col].to_numpy()
    arr_cost = df[cost_col].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]: 
            continue
        dom = (arr_acc >= arr_acc[i]) & (arr_cost <= arr_cost[i]) & (
            (arr_acc > arr_acc[i]) | (arr_cost < arr_cost[i])
        )
        if np.any(dom & (np.arange(len(df)) != i)):
            keep[i] = False
    return df.loc[keep]

def summarize_clusters(df: pd.DataFrame, cluster_col: str, metrics: List[str]) -> pd.DataFrame:
    if cluster_col not in df.columns:
        return pd.DataFrame()
    agg = {}
    for m in metrics:
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
            agg[m] = ["count", "mean", "min", "max"]
    if not agg:
        return pd.DataFrame()
    return df.groupby(cluster_col).agg(agg)

def kmeans_cluster(df: pd.DataFrame, cols: List[str], n_clusters: int, random_state: int = 42):
    if not SKLEARN_OK or len(df) < max(2, n_clusters):
        return None, None
    X = df[cols].to_numpy()
    model = KMeans(n_clusters=n_clusters, n_init="auto" if hasattr(KMeans(), "n_init") else 10, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels

def suggest_parameter_ranges(df: pd.DataFrame, good_mask: np.ndarray, param_columns: List[str]) -> Dict[str, Tuple[float, float]]:
    """Given a mask of 'good' rows, return min/max ranges for each param across those rows."""
    recs = {}
    good_df = df.loc[good_mask]
    for p in param_columns:
        if p in good_df.columns:
            series = good_df[p].dropna()
            if len(series) == 0: 
                continue
            try:
                lo, hi = float(series.min()), float(series.max())
            except Exception:
                # handle categorical params by mode
                top = series.value_counts().index.tolist()[:3]
                recs[p] = ("categorical", top)
            else:
                recs[p] = (lo, hi)
    return recs

def make_tooltip_text(row: pd.Series, id_col: Optional[str], param_columns: List[str], acc_col: str, cost_col: str) -> str:
    rid = str(row[id_col]) if id_col and id_col in row else "(idx " + str(row.name) + ")"
    params = ", ".join([f"{p}={row[p]}" for p in param_columns if p in row])
    return f"id={rid}<br>{acc_col}={row[acc_col]:.4g}<br>{cost_col}={row[cost_col]:.4g}<br>{params}"

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="yProv4ML Decision Support", layout="wide")
st.title("yProv4ML • Decision-Support Dashboard")
st.write("Interactive analysis for **multi-objective model selection** (performance vs sustainability).")

default_path = Path("prov4ml/visuals/outputs/analysis/runs_enriched.csv")
st.sidebar.header("1) Data")
path_mode = st.sidebar.radio("Load data from:", ["Default path", "Upload CSV"])

df = None
if path_mode == "Default path":
    if default_path.exists():
        df = pd.read_csv(default_path)
        st.sidebar.success(f"Loaded: {default_path}")
    else:
        st.sidebar.error(f"Default not found: {default_path}")
else:
    up = st.sidebar.file_uploader("Upload runs_enriched.csv", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.sidebar.success("CSV uploaded")

if df is None:
    st.stop()

# Detect basic structure
id_col = detect_id_column(df)
pcols = param_cols(df)
ncols = numeric_cols(df)
acc_cands = candidate_accuracy_cols(df)
cost_cands = candidate_cost_cols(df)

st.sidebar.header("2) Metrics")
acc_col = st.sidebar.selectbox("Accuracy metric (maximize)", options=acc_cands or ncols, index=0 if acc_cands else 0, key="acc_col")
cost_col = st.sidebar.selectbox("Cost/Emission metric (minimize)", options=cost_cands or ncols, index=0, key="cost_col")

# Filter nulls
work = df.dropna(subset=[acc_col, cost_col]).copy()
if len(work) < 2:
    st.warning("Not enough rows with both selected metrics.")
    st.stop()

st.sidebar.header("3) Clustering")
clustering_mode = st.sidebar.radio("Cluster on:", ["Full dataset", "Pareto-only"])
n_clusters = st.sidebar.slider("Number of clusters (KMeans)", 2, 8, 3)
annotate = st.sidebar.checkbox("Label points on plots", value=True)

# ----------------------------- Descriptive analytics -----------------------------
st.subheader("Descriptive analytics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows (valid)", len(work))
with col2:
    st.metric(f"{acc_col} (mean)", f"{work[acc_col].mean():.4g}")
with col3:
    st.metric(f"{cost_col} (mean)", f"{work[cost_col].mean():.4g}")

with st.expander("Correlation (Spearman) among numeric columns"):
    corr = work[ncols].corr(method="spearman")
    st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None))

# ----------------------------- Pareto analysis -----------------------------
st.subheader("Pareto analysis")
front = nondominated_front(work[[acc_col, cost_col] + pcols + ([id_col] if id_col else [])], acc_col, cost_col)
st.write(f"Found **{len(front)}** Pareto-optimal runs (maximize {acc_col}, minimize {cost_col}).")

# Scatter: all vs front
def plot_scatter_all_vs_front(df_all, df_front, acc_col, cost_col, id_col, pcols, title):
    fig = go.Figure()
    # all
    hover_all = df_all.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
    fig.add_trace(go.Scatter(
        x=df_all[cost_col], y=df_all[acc_col],
        mode="markers", name="All runs",
        opacity=0.35, marker=dict(size=9, line=dict(width=0.5, color="black")),
        text=hover_all, hoverinfo="text"
    ))
    # front
    hover_front = df_front.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
    fig.add_trace(go.Scatter(
        x=df_front[cost_col], y=df_front[acc_col],
        mode="markers+text" if annotate else "markers",
        name="Pareto front", marker=dict(size=12, line=dict(width=1, color="black")),
        text=[str(r[id_col]) if id_col and (id_col in r) else str(i) for i, r in df_front.iterrows()] if annotate else None,
        textposition="top center",
        hovertext=hover_front, hoverinfo="text"
    ))
    fig.update_layout(title=title, xaxis_title=cost_col, yaxis_title=acc_col, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

st.plotly_chart(plot_scatter_all_vs_front(work, front, acc_col, cost_col, id_col, pcols, "All runs vs Pareto front"), use_container_width=True)

# ----------------------------- Clustering -----------------------------
st.subheader("Clustering")
cluster_df = front if clustering_mode == "Pareto-only" else work
cluster_cols = [acc_col, cost_col]
labels = None
if SKLEARN_OK and len(cluster_df) >= n_clusters:
    model, labels = kmeans_cluster(cluster_df, cluster_cols, n_clusters=n_clusters)
    if labels is not None:
        cluster_df = cluster_df.copy()
        cluster_df["cluster"] = labels
        st.write(f"**KMeans clusters** on {clustering_mode} set: K={n_clusters}")
        # NEW: make clusters categorical and pick a discrete palette
        cluster_df["cluster_label"] = cluster_df["cluster"].astype(str)  # "0","1","2", ...
        PALETTE = px.colors.qualitative.Set2 
        # Scatter by cluster
        hover = cluster_df.apply(lambda r: make_tooltip_text(r, id_col, pcols, acc_col, cost_col), axis=1)
        fig = px.scatter(
            cluster_df,
            x=cost_col, y=acc_col,
            color="cluster_label",
            color_discrete_sequence=PALETTE,
            hover_name=id_col if id_col else None,
            hover_data=pcols,
            title=f"Clusters on {clustering_mode.lower()} set"
        )
        fig.update_layout(legend_title_text="cluster")
        if annotate and id_col:
            fig.update_traces(text=cluster_df[id_col].astype(str), textposition="top center", selector=dict(mode="markers"))
        fig.update_layout(title=f"Clusters on {clustering_mode.lower()} set", xaxis_title=cost_col, yaxis_title=acc_col)
        st.plotly_chart(fig, use_container_width=True)

        # Parameter distributions
        if pcols:
            with st.expander("Parameter distributions by cluster"):
                for p in pcols:
                    ct = cluster_df.groupby(["cluster_label", p]).size().reset_index(name="count")
                    figp = px.bar(
                        ct,
                        x=p, y="count",
                        color="cluster_label",
                        barmode="group",
                        color_discrete_sequence=PALETTE,
                        title=f"Param distribution: {p}"
                    )
                    figp.update_layout(legend_title_text="cluster")
                    st.plotly_chart(figp, use_container_width=True)


        # Cluster-level summary
        with st.expander("Cluster-wise summary (mean/min/max/count)"):
            summary = summarize_clusters(cluster_df, "cluster", [acc_col, cost_col])
            st.dataframe(summary)
    else:
        st.info("Clustering skipped (insufficient rows).")
else:
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed; clustering disabled. pip install scikit-learn")
    else:
        st.info("Not enough rows for clustering with current selection.")

# ----------------------------- Prescriptive analytics -----------------------------
st.subheader("Prescriptive analytics")

# Heuristic: "good" = (accuracy >= top quantile) & (cost <= bottom quantile)
q_acc = st.slider("Top-quantile for accuracy (good ≥)", 0.50, 0.95, 0.80, 0.05)
q_cost = st.slider("Bottom-quantile for cost (good ≤)", 0.05, 0.50, 0.25, 0.05)
thr_acc = work[acc_col].quantile(q_acc)
thr_cost = work[cost_col].quantile(q_cost)
good_mask = (work[acc_col] >= thr_acc) & (work[cost_col] <= thr_cost)

st.write(f"Thresholds → **{acc_col} ≥ {thr_acc:.4g}** and **{cost_col} ≤ {thr_cost:.4g}**. Selected **{int(good_mask.sum())}/{len(work)}** runs as *high-utility* candidates.")

if good_mask.any():
    good = work.loc[good_mask, [acc_col, cost_col] + ([id_col] if id_col else []) + pcols].copy()
    st.dataframe(good.sort_values([cost_col, acc_col], ascending=[True, False]).head(20))

    # Recommend parameter ranges
    recs = suggest_parameter_ranges(work, good_mask.values, pcols)
    if recs:
        st.markdown("**Recommended parameter ranges / choices** (from high-utility runs):")
        for p, v in recs.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "categorical":
                st.write(f"- `{p}`: prefer values **{v[1]}**")
            elif isinstance(v, tuple) and len(v) == 2:
                st.write(f"- `{p}`: try range **[{v[0]:.4g}, {v[1]:.4g}]**")

    # Next experiments around top Pareto points
    st.markdown("**Next experiments (around frontier)**")
    # pick top 5 from front
    top_front = front.sort_values([cost_col, acc_col], ascending=[True, False]).head(5)
    def neighbor_grid(val, scale=0.5, n=3):
        # propose multiplicative neighbors (for lr-like) and +/- 1 step for integers
        cands = []
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return cands
        try:
            v = float(val)
            cands = sorted(set([v, v*(1-scale), v*(1+scale)]))
        except Exception:
            # categorical — return the same for now
            cands = [val]
        return cands[:n]

    for idx, row in top_front.iterrows():
        rid = row[id_col] if id_col and id_col in row else idx
        st.write(f"- Front run **{rid}**: {acc_col}={row[acc_col]:.4g}, {cost_col}={row[cost_col]:.4g}")
        # propose local neighborhood for params
        proposals = {}
        for p in pcols:
            val = row.get(p, None)
            if val is None: 
                continue
            if pd.api.types.is_integer_dtype(type(val)) or (isinstance(val, (int, np.integer))):
                proposals[p] = [int(val)]
            else:
                proposals[p] = neighbor_grid(val, scale=0.5)
        st.code(json.dumps(proposals, indent=2))
else:
    st.info("No runs meet both thresholds. Try adjusting the sliders or switch cost metric.")

st.caption("Built for human-in-the-loop, multi-objective model selection with yProv4ML.")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def _heatmap(df: pd.DataFrame, x_param: str, y_param: str, value_col: str, out_path: Path, title: str) -> bool:
    try:
        H = (df.groupby([x_param, y_param])[value_col]
                .mean()
                .reset_index()
                .pivot(index=y_param, columns=x_param, values=value_col))
    except Exception:
        return False
    if H.empty: return False
    plt.figure(figsize=(8,6))
    plt.imshow(H.values, aspect="auto")
    plt.colorbar(label=value_col)
    plt.xticks(range(H.shape[1]), [str(x) for x in H.columns], rotation=45, ha="right")
    plt.yticks(range(H.shape[0]), [str(y) for y in H.index])
    plt.xlabel(x_param); plt.ylabel(y_param); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    return True

def heatmaps_lr_batch(runs: pd.DataFrame, outdir: Path, acc_col: str|None, cost_col: str|None,
                      x_param: str="param_lr", y_param: str="param_batch_size"):
    if x_param in runs.columns and y_param in runs.columns:
        if acc_col:
            _heatmap(runs, x_param, y_param, acc_col, outdir / "heatmap_lr_batch_acc.png",
                     f"Mean {acc_col} over ({x_param}, {y_param})")
        if cost_col:
            _heatmap(runs, x_param, y_param, cost_col, outdir / f"heatmap_lr_batch_{cost_col}.png",
                     f"Mean {cost_col} over ({x_param}, {y_param})")

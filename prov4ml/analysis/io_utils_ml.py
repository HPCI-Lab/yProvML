from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

def ensure_outdir(outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def load_runs_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _find_best_column(candidates: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def choose_columns(runs: pd.DataFrame,
                   accuracy_col: Optional[str]=None,
                   emissions_cols: Optional[List[str]]=None,
                   cost_priority: Optional[List[str]]=None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    acc = accuracy_col or _find_best_column(["ACC_val","accuracy","ACC","val_acc","val_accuracy"], runs)
    if emissions_cols:
        em = _find_best_column(emissions_cols, runs)
    else:
        em_candidates = [c for c in runs.columns if c.lower().endswith("_cumsum") or
                         "emission" in c.lower() or "co2" in c.lower() or "carbon" in c.lower() or c.lower().endswith("_total")]
        em = _find_best_column(em_candidates, runs)
    cost_priority = cost_priority or ["cost_total_usd","cost_gpu_time_usd","cost_energy_usd","energy_kwh","wall_time_s","gpu_hours"]
    cost = _find_best_column(cost_priority, runs)
    return acc, em, cost

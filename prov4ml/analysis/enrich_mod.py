from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _load_lr_trace(p: Path):
    try:
        df = pd.read_csv(p)
        if 'lr' not in df.columns:
            for alt in ['learning_rate','lr_value','LR']:
                if alt in df.columns:
                    df = df.rename(columns={alt:'lr'}); break
        if 'lr' not in df.columns:
            return None
        if 'epoch' not in df.columns and 'step' not in df.columns:
            df['step'] = np.arange(len(df))
        return df
    except Exception:
        return None

def merge_lr_min_max(runs: pd.DataFrame, lr_trace_dir: Optional[str]) -> pd.DataFrame:
    runs = runs.copy()
    runs['lr_min'] = np.nan
    runs['lr_max'] = np.nan
    if not lr_trace_dir:
        if 'param_lr' in runs.columns:
            runs['lr_min'] = runs['param_lr'].astype(float)
            runs['lr_max'] = runs['param_lr'].astype(float)
        return runs
    lr_dir = Path(lr_trace_dir)
    if not lr_dir.exists():
        return runs
    lr_min_map: Dict[int,float] = {}
    lr_max_map: Dict[int,float] = {}
    for _, row in runs.iterrows():
        exp = row.get('exp', None)
        if pd.isna(exp): continue
        try: exp_str = str(int(exp))
        except Exception: exp_str = str(exp)
        for name in [f"{exp_str}.csv", f"exp_{exp_str}.csv"]:
            p = lr_dir / name
            if p.exists():
                tdf = _load_lr_trace(p)
                if tdf is None: break
                lr_min_map[int(exp)] = float(np.nanmin(tdf['lr'].values.astype(float)))
                lr_max_map[int(exp)] = float(np.nanmax(tdf['lr'].values.astype(float)))
                break
    if 'exp' in runs.columns:
        runs['lr_min'] = runs['exp'].map(lr_min_map)
        runs['lr_max'] = runs['exp'].map(lr_max_map)
    if 'param_lr' in runs.columns:
        runs['lr_min'] = runs['lr_min'].fillna(runs['param_lr'].astype(float))
        runs['lr_max'] = runs['lr_max'].fillna(runs['param_lr'].astype(float))
    return runs

def merge_compute_json(runs: pd.DataFrame, compute_dirs: Optional[List[str]]) -> pd.DataFrame:
    if not compute_dirs: return runs
    runs = runs.copy()
    # collect compute.json files
    paths = []
    for d in compute_dirs:
        dpath = Path(d)
        if not dpath.exists(): continue
        paths.extend(list(dpath.rglob("compute.json")))
    if not paths: return runs
    # index by exp id present in path pieces
    matched = {}
    for cf in paths:
        parts = [part for part in cf.parts]
        cands = []
        for p in parts:
            if p.isdigit(): cands.append(int(p))
            for pre in ["exp_","exp"]:
                if p.startswith(pre) and p[len(pre):].isdigit():
                    cands.append(int(p[len(pre):]))
        for cid in set(cands):
            matched.setdefault(cid, []).append(cf)
    cmap: Dict[int,dict] = {}
    if 'exp' not in runs.columns: return runs
    for exp in runs['exp'].dropna().unique():
        try: expi = int(exp)
        except Exception: continue
        files = matched.get(expi, [])
        if not files: continue
        files = sorted(files, key=lambda p: len(str(p)))
        try:
            cmap[expi] = json.loads(files[0].read_text())
        except Exception:
            pass
    cols = ["wall_time_s","gpu_count","gpu_hours","gpu_avg_power_w","energy_kwh","cost_energy_usd","cost_gpu_time_usd","cost_total_usd"]
    for c in cols:
        if c not in runs.columns:
            runs[c] = np.nan
    for idx, row in runs.iterrows():
        exp = row.get('exp', None)
        if pd.isna(exp): continue
        try: expi = int(exp)
        except Exception: continue
        data = cmap.get(expi)
        if not data: continue
        for c in cols:
            if c in data and data[c] is not None:
                runs.at[idx, c] = _safe_float(data[c])
    return runs

def add_emissions_cumsum(runs: pd.DataFrame, emissions_cols: List[str]) -> pd.DataFrame:
    runs = runs.copy()
    existing = [c for c in emissions_cols if c in runs.columns]
    if existing: return runs
    prefixes = []
    for c in runs.columns:
        cl = c.lower()
        if cl.startswith("co2_step_") or cl.startswith("emissions_step_"):
            pref = c.split("_step_")[0]
            if pref not in prefixes: prefixes.append(pref)
    for pref in prefixes:
        cols = [c for c in runs.columns if c.startswith(pref + "_step_")]
        def _k(x):
            try: return int(x.split("_step_")[-1])
            except Exception: return 10**9
        cols = sorted(cols, key=_k)
        if not cols: continue
        vals = runs[cols].astype(float).values
        runs[pref + "_cumsum"] = np.nansum(vals, axis=1)
    return runs

def add_co2_from_energy(runs: pd.DataFrame, kg_per_kwh: float) -> pd.DataFrame:
    runs = runs.copy()
    if "energy_kwh" in runs.columns and "CO2_from_energy_kg" not in runs.columns:
        runs["CO2_from_energy_kg"] = runs["energy_kwh"].astype(float) * float(kg_per_kwh)
    return runs

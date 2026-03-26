# src/params_utils.py

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

ALLOWED_STAGE = {"1", "2", "both"}

@dataclass
class ParamPack:
    df: pd.DataFrame
    names: List[str]
    name_to_i: Dict[str, int]
    x0_full: np.ndarray
    lb_full: np.ndarray
    ub_full: np.ndarray
    vary_full: np.ndarray
    stage_full: List[str]

def read_params_csv(path: str) -> ParamPack:
    df = pd.read_csv(path)

    required = ["name", "value", "vary", "lower", "upper", "stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in params CSV: {missing}")

    df["name"] = df["name"].astype(str).str.strip()
    df["stage"] = df["stage"].astype(str).str.strip().str.lower()
    df["vary"] = df["vary"].astype(int)

    bad_stage = sorted(set(df["stage"]) - ALLOWED_STAGE)
    if bad_stage:
        raise ValueError(f"Invalid stage values: {bad_stage}")

    if df["name"].duplicated().any():
        dup = df.loc[df["name"].duplicated(), "name"].tolist()
        raise ValueError(f"Duplicate parameter names: {dup}")

    names = df["name"].tolist()
    name_to_i = {n: i for i, n in enumerate(names)}

    x0 = df["value"].astype(float).to_numpy()
    lb = df["lower"].astype(float).to_numpy()
    ub = df["upper"].astype(float).to_numpy()
    vary = df["vary"].to_numpy().astype(bool)
    stage = df["stage"].tolist()

    if np.any(lb >= ub):
        bad = np.where(lb >= ub)[0].tolist()
        raise ValueError(f"Found lower >= upper for rows: {bad}")

    x0 = np.clip(x0, lb, ub)

    return ParamPack(
        df=df,
        names=names,
        name_to_i=name_to_i,
        x0_full=x0,
        lb_full=lb,
        ub_full=ub,
        vary_full=vary,
        stage_full=stage,
    )

def stage_free_mask(pack: ParamPack, stage: str) -> np.ndarray:
    if stage not in {"1", "2"}:
        raise ValueError("stage must be '1' or '2'")

    active = np.array([(s == stage or s == "both") for s in pack.stage_full], dtype=bool)
    free_mask = active & pack.vary_full
    return free_mask

def extract_free(pack: ParamPack, free_mask: np.ndarray):
    x0_free = pack.x0_full[free_mask]
    lb_free = pack.lb_full[free_mask]
    ub_free = pack.ub_full[free_mask]
    return x0_free, lb_free, ub_free

def inject_free(pack: ParamPack, free_mask: np.ndarray, x_free: np.ndarray) -> np.ndarray:
    x_full = pack.x0_full.copy()
    x_full[free_mask] = x_free
    return x_full

def write_params_csv(pack: ParamPack, out_path: str, x_full: np.ndarray) -> None:
    df_out = pack.df.copy()
    df_out["value"] = x_full
    df_out.to_csv(out_path, index=False)

def unpack_params(x_full: np.ndarray, name_to_i: Dict[str, int]) -> Dict[str, float]:
    return {k: float(x_full[i]) for k, i in name_to_i.items()}
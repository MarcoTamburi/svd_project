# src/fit3.py
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

from params_utils import read_params_csv, stage_free_mask, extract_free, inject_free, write_params_csv
from io_utils import load_T_and_Vprime

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) #Converte il contenuto JSON in un dict Python.

def residuals_stub(x_full, T, V_prime):
    # per ora residuo finto: zero, giusto per testare la pipeline
    return np.zeros(V_prime.size, dtype=float)

def run_fit3(config_path: str, run_metadata: dict):
    cfg = load_config(config_path)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # copia config usato
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    pack = read_params_csv(cfg["params_csv"])
    T, V_prime = load_T_and_Vprime(
    cfg["data"]["T_path"],
    cfg["data"]["V_prime_path"],
    )

    method = cfg.get("fit", {}).get("method", "trf")
    max_nfev = cfg.get("fit", {}).get("max_nfev", None)

    # stage 1
    free_mask_1 = stage_free_mask(pack, "1")
    x0_1, lb_1, ub_1 = extract_free(pack, free_mask_1)

    def fun1(x_free):
        x_full = inject_free(pack, free_mask_1, x_free)
        return residuals_stub(x_full, T, V_prime)

    res1 = least_squares(fun1, x0_1, bounds=(lb_1, ub_1), method=method, max_nfev=max_nfev)
    x_full_1 = inject_free(pack, free_mask_1, res1.x)
    write_params_csv(pack, str(out_dir / "params_after_stage1.csv"), x_full_1)

    pack.x0_full = x_full_1 # aggiorno pack con i risultati stage 1, così stage 2 parte da lì

    # stage 2
    free_mask_2 = stage_free_mask(pack, "2")
    x0_2, lb_2, ub_2 = extract_free(pack, free_mask_2)

    def fun2(x_free):
        x_full = inject_free(pack, free_mask_2, x_free)
        return residuals_stub(x_full, T, V_prime)

    res2 = least_squares(fun2, x0_2, bounds=(lb_2, ub_2), method=method, max_nfev=max_nfev)
    x_full_2 = inject_free(pack, free_mask_2, res2.x)
    write_params_csv(pack, str(out_dir / "params_final.csv"), x_full_2)

    summary = {
        "run_timestamp": run_metadata["run_timestamp"],
        "config_path": run_metadata["config_path"],
        "n_components": run_metadata["n_components"],
        "stage1_success": bool(res1.success),
        "stage1_cost": float(res1.cost),
        "stage2_success": bool(res2.success),
        "stage2_cost": float(res2.cost),
    }

    with open(out_dir / "fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return x_full_2, summary
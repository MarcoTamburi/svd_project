# src/fit3.py
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

from params_utils import (
    read_params_csv,
    stage_free_mask,
    extract_free,
    inject_free,
    write_params_csv,
    update_pack_values,
)
from io_utils import load_fit3_inputs


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def residuals_stub(x_full, T, V_prime):
    # per ora residuo finto: zero, giusto per testare la pipeline
    return np.zeros(V_prime.size, dtype=float)

def estimate_edge_coefficients(U_prime, spectral_matrix):
    # spectral_matrix shape attesa: (n_lambda, n_temperatures)
    # U_prime shape attesa: (n_lambda, 3)

    if U_prime.ndim != 2:
        raise ValueError(f"U_prime deve essere 2D, shape trovata: {U_prime.shape}")

    if spectral_matrix.ndim != 2:
        raise ValueError(f"spectral_matrix deve essere 2D, shape trovata: {spectral_matrix.shape}")

    if U_prime.shape[0] != spectral_matrix.shape[0]:
        raise ValueError(
            f"Incompatibilità tra U_prime e spectral_matrix: "
            f"{U_prime.shape[0]} vs {spectral_matrix.shape[0]} righe"
        )

    if U_prime.shape[1] != 3:
        raise ValueError(
            f"Per fit3 U_prime deve avere 3 colonne, trovate: {U_prime.shape[1]}"
        )

    reg = LinearRegression()

    # folded = prima colonna della matrice spettrale
    spttr_F = spectral_matrix[:, 0]
    reg.fit(U_prime, spttr_F)
    C11, C21, C31 = reg.coef_
    predicted_F = reg.predict(U_prime)

    # unfolded = ultima colonna della matrice spettrale
    spttr_U = spectral_matrix[:, -1]
    reg.fit(U_prime, spttr_U)
    C13, C23, C33 = reg.coef_
    predicted_U = reg.predict(U_prime)

    coeffs = {
        "C11": float(C11),
        "C21": float(C21),
        "C31": float(C31),
        "C13": float(C13),
        "C23": float(C23),
        "C33": float(C33),
    }

    debug_data = {
        "folded_original": spttr_F,
        "folded_predicted": predicted_F,
        "unfolded_original": spttr_U,
        "unfolded_predicted": predicted_U,
    }

    return coeffs, debug_data

def run_fit3(config_path: str, run_metadata: dict):
    cfg = load_config(config_path)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # salva config usata
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
   
    pack = read_params_csv(cfg["params_csv"])

    # nuovo loader coerente coi file reali
    T, V_prime, U_prime, spectral_matrix, wavelengths = load_fit3_inputs(
        cfg["data"]["spectra_matrix_path"],
        cfg["data"]["V_prime_path"],
        cfg["data"]["U_prime_path"],
    )
    preprocess_coeffs, preprocess_debug = estimate_edge_coefficients(
        U_prime,
        spectral_matrix
    )

    with open(out_dir / "preprocess_coeffs.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_coeffs, f, indent=2)

    pack = update_pack_values(pack, preprocess_coeffs)

    write_params_csv(pack, str(out_dir / "params_after_preprocess.csv"), pack.x0_full)

    method = cfg.get("fit", {}).get("method", "trf")
    max_nfev = cfg.get("fit", {}).get("max_nfev", None)

    # stage 1
    free_mask_1 = stage_free_mask(pack, "1")
    x0_1, lb_1, ub_1 = extract_free(pack, free_mask_1)

    def fun1(x_free):
        x_full = inject_free(pack, free_mask_1, x_free)
        return residuals_stub(x_full, T, V_prime)

    res1 = least_squares(
        fun1,
        x0_1,
        bounds=(lb_1, ub_1),
        method=method,
        max_nfev=max_nfev
    )

    x_full_1 = inject_free(pack, free_mask_1, res1.x)
    write_params_csv(pack, str(out_dir / "params_after_stage1.csv"), x_full_1)
    
    # aggiorno pack con i risultati stage 1, così stage 2 parte da lì
    pack.x0_full = x_full_1

    # se params_utils usa anche il dataframe interno, meglio tenerlo allineato
    if "value" in pack.df.columns:
        pack.df["value"] = x_full_1

    # stage 2
    free_mask_2 = stage_free_mask(pack, "2")
    x0_2, lb_2, ub_2 = extract_free(pack, free_mask_2)

    def fun2(x_free):
        x_full = inject_free(pack, free_mask_2, x_free)
        return residuals_stub(x_full, T, V_prime)

    res2 = least_squares(
        fun2,
        x0_2,
        bounds=(lb_2, ub_2),
        method=method,
        max_nfev=max_nfev
    )

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
        "T_shape": list(T.shape),
        "V_prime_shape": list(V_prime.shape),
        "U_prime_shape": list(U_prime.shape),
        "spectral_matrix_shape": list(spectral_matrix.shape),
        "wavelengths_shape": list(wavelengths.shape),
        "preprocess_coeffs": preprocess_coeffs,
    }

    with open(out_dir / "fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return x_full_2, summary
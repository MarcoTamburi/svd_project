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
from model_fit3 import (
    predict_vprime_from_params,
    residuals_fit3,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def save_preprocessing_outputs(out_dir, wavelengths, preprocess_debug):
    import pandas as pd
    from matplotlib import pyplot as plt

    folded_original = preprocess_debug["folded_original"]
    folded_predicted = preprocess_debug["folded_predicted"]
    unfolded_original = preprocess_debug["unfolded_original"]
    unfolded_predicted = preprocess_debug["unfolded_predicted"]

    folded_df = pd.DataFrame({
        "wavelength": wavelengths,
        "original": folded_original,
        "predicted": folded_predicted,
        "residual": folded_original - folded_predicted,
    })
    folded_df.to_csv(out_dir / "preprocess_folded_data.csv", index=False)

    unfolded_df = pd.DataFrame({
        "wavelength": wavelengths,
        "original": unfolded_original,
        "predicted": unfolded_predicted,
        "residual": unfolded_original - unfolded_predicted,
    })
    unfolded_df.to_csv(out_dir / "preprocess_unfolded_data.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, folded_original[::-1], label="original folded spectrum", marker="o")
    plt.plot(wavelengths, folded_predicted[::-1], label="predicted folded spectrum")
    plt.title("Preprocessing - Folded spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "preprocess_folded_fit.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, unfolded_original[::-1], label="original unfolded spectrum", marker="o")
    plt.plot(wavelengths, unfolded_predicted[::-1], label="predicted unfolded spectrum")
    plt.title("Preprocessing - Unfolded spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "preprocess_unfolded_fit.png", dpi=300)
    plt.close()


def save_final_fit_plots(out_dir, T, V_prime, x_full, pack):
    from matplotlib import pyplot as plt

    _, _, f_pred = predict_vprime_from_params(T, x_full, pack)

    labels = ["V1_prime", "V2_prime", "V3_prime"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axs[i].plot(T - 273.15, V_prime[i], "o", label="Experimental data")
        axs[i].plot(T - 273.15, f_pred[i], "-", label="Global fit")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[2].set_xlabel("Temperature (°C)")
    plt.suptitle("Final global fit")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "final_global_fit.png", dpi=300)
    plt.close()

    return f_pred


def save_final_fit_data(out_dir, T, V_prime, f_pred):
    import pandas as pd

    df = pd.DataFrame({
        "T_kelvin": T,
        "T_celsius": T - 273.15,
        "V1_exp": V_prime[0],
        "V1_fit": f_pred[0],
        "V1_resid": V_prime[0] - f_pred[0],
        "V2_exp": V_prime[1],
        "V2_fit": f_pred[1],
        "V2_resid": V_prime[1] - f_pred[1],
        "V3_exp": V_prime[2],
        "V3_fit": f_pred[2],
        "V3_resid": V_prime[2] - f_pred[2],
    })

    df.to_csv(out_dir / "final_fit_curves.csv", index=False)


def run_fit3(config_path: str, run_metadata: dict):
    cfg = load_config(config_path)

    debug_enabled = cfg.get("debug", {}).get("enabled", False)
    save_preprocess_plots_flag = cfg.get("plots", {}).get("save_preprocess", True)
    save_final_fit_plots_flag = cfg.get("plots", {}).get("save_final_fit", True)

    base_out_dir = Path(cfg["output_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"fit3_{run_metadata['run_timestamp'].replace(':', '-').replace(' ', '_')}"
    out_dir = base_out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    pack = read_params_csv(cfg["params_csv"])

    T, V_prime, U_prime, spectral_matrix, wavelengths = load_fit3_inputs(
        cfg["data"]["spectra_matrix_path"],
        cfg["data"]["V_prime_path"],
        cfg["data"]["U_prime_path"],
    )

    preprocess_coeffs, preprocess_debug = estimate_edge_coefficients(
        U_prime,
        spectral_matrix
    )

    T = T + 273.15  # converti da Celsius a Kelvin

    if save_preprocess_plots_flag:
        save_preprocessing_outputs(out_dir, wavelengths, preprocess_debug)

    with open(out_dir / "preprocess_coeffs.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_coeffs, f, indent=2)

    pack = update_pack_values(pack, preprocess_coeffs)

    write_params_csv(pack, str(out_dir / "params_after_preprocess.csv"), pack.x0_full)

    method = cfg.get("fit", {}).get("method", "trf")
    max_nfev = cfg.get("fit", {}).get("max_nfev", None)

    free_mask_1 = stage_free_mask(pack, "1")
    x0_1, lb_1, ub_1 = extract_free(pack, free_mask_1)

    stage1_names = [name for name, is_free in zip(pack.df["name"], free_mask_1) if is_free]

    if debug_enabled:
        print("Stage 1 free params:", stage1_names)
        print("Stage 1 x0:", x0_1)

    def fun1(x_free):
        x_full = inject_free(pack, free_mask_1, x_free)
        return residuals_fit3(x_full, T, V_prime, pack)

    def debug_stage1_sensitivity(x_ref, eps_rel=1e-4):
        r0 = fun1(x_ref)
        n0 = np.linalg.norm(r0)
        print("Stage 1 residual norm at x0:", n0)

        for i, name in enumerate(stage1_names):
            x_try = x_ref.copy()

            step = eps_rel * max(1.0, abs(x_ref[i]))
            x_try[i] += step

            r_try = fun1(x_try)
            n_try = np.linalg.norm(r_try)

            print(
                f"{name}: x0={x_ref[i]:.6g}, step={step:.6g}, "
                f"norm_before={n0:.6g}, norm_after={n_try:.6g}, "
                f"delta={n_try - n0:.6g}"
            )

    if debug_enabled:
        print("\nDebug: Sensitivity of stage 1 residuals to initial parameters")
        debug_stage1_sensitivity(x0_1)

    res1 = least_squares(
        fun1,
        x0_1,
        bounds=(lb_1, ub_1),
        method=method,
        max_nfev=max_nfev
    )

    if debug_enabled:
        print("Stage 1 result x:", res1.x)
        print("Stage 1 cost:", res1.cost)

    x_full_1 = inject_free(pack, free_mask_1, res1.x)
    write_params_csv(pack, str(out_dir / "params_after_stage1.csv"), x_full_1)

    pack.x0_full = x_full_1

    if "value" in pack.df.columns:
        pack.df["value"] = x_full_1

    free_mask_2 = stage_free_mask(pack, "2")
    x0_2, lb_2, ub_2 = extract_free(pack, free_mask_2)

    def fun2(x_free):
        x_full = inject_free(pack, free_mask_2, x_free)
        return residuals_fit3(x_full, T, V_prime, pack)

    res2 = least_squares(
        fun2,
        x0_2,
        bounds=(lb_2, ub_2),
        method=method,
        max_nfev=max_nfev
    )

    stage2_names = [name for name, is_free in zip(pack.df["name"], free_mask_2) if is_free]

    if debug_enabled:
        print("\nStage 2 free params:", stage2_names)
        print("Stage 2 x0:", x0_2)
        print("Stage 2 result x:", res2.x)

    x_full_2 = inject_free(pack, free_mask_2, res2.x)

    if save_final_fit_plots_flag:
        f_pred_final = save_final_fit_plots(out_dir, T, V_prime, x_full_2, pack)
    else:
        _, _, f_pred_final = predict_vprime_from_params(T, x_full_2, pack)

    final_params = {
        name: float(x_full_2[idx]) for name, idx in pack.name_to_i.items()
    }

    save_final_fit_data(out_dir, T, V_prime, f_pred_final)

    residuals_final = V_prime - f_pred_final
    chi2 = float(np.sum(residuals_final ** 2))
    dof = V_prime.size - len(res2.x)
    chi2_red = float(chi2 / dof) if dof > 0 else None

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
        "output_dir": str(out_dir),
        "final_params": final_params,
        "chi2": chi2,
        "chi2_reduced": chi2_red,
    }

    with open(out_dir / "fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return x_full_2, summary
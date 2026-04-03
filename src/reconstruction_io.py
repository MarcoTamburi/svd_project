from pathlib import Path
import json

from fit3 import load_config
from io_utils import load_fit3_inputs
from params_utils import read_params_csv


def find_latest_fit3_run(results_dir):
    """
    Trova la run fit3 più recente dentro results/fit3_run.

    Parameters
    ----------
    results_dir : str or Path
        Cartella che contiene le run, ad esempio:
        results/fit3_run

    Returns
    -------
    Path
        Path della cartella run più recente.
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory non trovata: {results_dir}")

    run_dirs = [
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name.startswith("fit3_")
    ]

    if not run_dirs:
        raise FileNotFoundError(
            f"Nessuna cartella di run trovata in: {results_dir}"
        )

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest_run


def load_completed_fit3_run(run_dir):
    """
    Carica una run già completata del fit3 senza rieseguire il fit.
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory non trovata: {run_dir}")

    config_used_path = run_dir / "config_used.json"
    params_final_path = run_dir / "params_final.csv"
    reconstruction_dir = run_dir / "reconstruction"

    if not config_used_path.exists():
        raise FileNotFoundError(f"config_used.json non trovato in: {run_dir}")

    if not params_final_path.exists():
        raise FileNotFoundError(f"params_final.csv non trovato in: {run_dir}")

    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_used_path))
    pack = read_params_csv(str(params_final_path))

    T, V_prime, U_prime, spectral_matrix, wavelengths = load_fit3_inputs(
        cfg["data"]["spectra_matrix_path"],
        cfg["data"]["V_prime_path"],
        cfg["data"]["U_prime_path"],
    )

    T = T + 273.15
    x_final = pack.x0_full.copy()

    return {
        "run_dir": run_dir,
        "reconstruction_dir": reconstruction_dir,
        "cfg": cfg,
        "pack": pack,
        "x_final": x_final,
        "T": T,
        "V_prime": V_prime,
        "U_prime": U_prime,
        "spectral_matrix": spectral_matrix,
        "wavelengths": wavelengths,
    }


def load_latest_completed_fit3_run(results_dir):
    """
    Trova e carica automaticamente la run fit3 più recente.
    """
    latest_run_dir = find_latest_fit3_run(results_dir)
    return load_completed_fit3_run(latest_run_dir)

# src/io_utils.py

import pandas as pd
import numpy as np

def load_T_and_Vprime(T_path: str, V_prime_path: str):
    # --- leggo file con T nella prima riga ---
    df = pd.read_csv(T_path, sep="\t", header=None)

    # prendo i nomi delle colonne, salto la prima ("Wavelength")
    columns = df.columns.tolist()
    T = np.array([float(c) for c in columns[1:]])

    # --- leggo V_prime ---
    V_prime = pd.read_csv(V_prime_path, sep="\t", header=None).to_numpy(dtype=float)

    return T, V_prime
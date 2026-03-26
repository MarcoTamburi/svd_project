# src/io_utils.py
import pandas as pd

def load_T_and_Vprime(T_path: str, T_column: str, V_prime_path: str):
    dfT = pd.read_csv(T_path)
    if T_column not in dfT.columns:
        raise ValueError(f"Colonna '{T_column}' non trovata in {T_path}. Colonne: {list(dfT.columns)}")
    T = dfT[T_column].to_numpy(dtype=float)

    V_prime = pd.read_csv(V_prime_path).to_numpy(dtype=float)
    return T, V_prime
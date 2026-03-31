import numpy as np

R = 1.987  # gas constant in cal/(mol*K)


def calc_M_2p(T, Tm1, Tm2, dH1, dH2):
    A = np.exp(-dH1 / R * (1 / Tm1 - 1 / T))
    B = np.exp(-dH2 / R * (1 / Tm2 - 1 / T))

    denom = 1 + A + (A * B)

    M1 = 1 / denom
    M2 = A / denom
    M3 = (A * B) / denom

    return np.stack([M1, M2, M3], axis=0)  # shape (3, len(T))


def build_C_matrix(x_full, pack):
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    C = np.array([
        [get("C11"), get("C12"), get("C13")],
        [get("C21"), get("C22"), get("C23")],
        [get("C31"), get("C32"), get("C33")]
    ], dtype=float)

    return C


def predict_vprime_from_params(T, x_full, pack):
    def get(name):
        return float(x_full[pack.name_to_i[name]])

    Tm1 = get("Tm1")
    Tm2 = get("Tm2")
    dH1 = get("dH1")
    dH2 = get("dH2")

    C = build_C_matrix(x_full, pack)
    M = calc_M_2p(T, Tm1, Tm2, dH1, dH2)
    V_pred = C @ M

    return C, M, V_pred


def residuals_fit3(x_full, T, V_prime, pack):
    _, _, f_pred = predict_vprime_from_params(T, x_full, pack)
    resid = (V_prime - f_pred).flatten()

    if np.any(np.isnan(resid)) or np.any(np.isinf(resid)):
        return np.full(V_prime.size, 1e12, dtype=float)

    return resid
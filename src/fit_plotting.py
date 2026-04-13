import pandas as pd
from matplotlib import pyplot as plt


def save_vprime_fit_plots(
    out_dir,
    T,
    V_prime,
    x_full,
    pack,
    predict_fn,
    plot_filename,
    title,
):
    _, _, f_pred = predict_fn(T, x_full, pack)

    n_components = V_prime.shape[0]
    labels = [f"V{i+1}_prime" for i in range(n_components)]

    fig, axs = plt.subplots(n_components, 1, figsize=(10, 2.6 * n_components), sharex=True)

    if n_components == 1:
        axs = [axs]

    for i in range(n_components):
        axs[i].plot(T - 273.15, V_prime[i], "o", label="Experimental data")
        axs[i].plot(T - 273.15, f_pred[i], "-", label="Model fit")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Temperature (°C)")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / plot_filename, dpi=300)
    plt.close()

    return f_pred


def save_vprime_fit_data(out_dir, T, V_prime, f_pred, csv_filename):
    data = {
        "T_kelvin": T,
        "T_celsius": T - 273.15,
    }

    n_components = V_prime.shape[0]

    for i in range(n_components):
        idx = i + 1
        data[f"V{idx}_exp"] = V_prime[i]
        data[f"V{idx}_fit"] = f_pred[i]
        data[f"V{idx}_resid"] = V_prime[i] - f_pred[i]

    df = pd.DataFrame(data)
    df.to_csv(out_dir / csv_filename, index=False)


def save_stage1_fit_outputs(out_dir, T, V_prime, x_full, pack, predict_fn):
    f_pred = save_vprime_fit_plots(
        out_dir=out_dir,
        T=T,
        V_prime=V_prime,
        x_full=x_full,
        pack=pack,
        predict_fn=predict_fn,
        plot_filename="stage1_global_fit.png",
        title="Stage 1 global fit",
    )

    save_vprime_fit_data(
        out_dir=out_dir,
        T=T,
        V_prime=V_prime,
        f_pred=f_pred,
        csv_filename="stage1_fit_curves.csv",
    )

    return f_pred


def save_final_fit_outputs(out_dir, T, V_prime, x_full, pack, predict_fn):
    f_pred = save_vprime_fit_plots(
        out_dir=out_dir,
        T=T,
        V_prime=V_prime,
        x_full=x_full,
        pack=pack,
        predict_fn=predict_fn,
        plot_filename="final_global_fit.png",
        title="Final global fit",
    )

    save_vprime_fit_data(
        out_dir=out_dir,
        T=T,
        V_prime=V_prime,
        f_pred=f_pred,
        csv_filename="final_fit_curves.csv",
    )

    return f_pred
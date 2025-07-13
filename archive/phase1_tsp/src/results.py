import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["grid.color"] = "#cccccc"
plt.rcParams["axes.xmargin"] = 0


def plot_scatter(plot_df, const_cut_off=None):
    n_plots = len(plot_df)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(9, 3.1 * n_rows))

    if n_rows == 1:
        axs = axs.reshape(1, -1)

    axs = axs.flatten()

    for i, result in plot_df.iterrows():
        ax = axs[i]

        ax.scatter(
            result["y_test_not_censored"],
            result["y_pred"],
            alpha=0.5,
            edgecolors="k",
            lw=0.2,
            s=3,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim(0.01, 320)
        ax.set_ylim(0.01, 320)
        ax.plot([0.01, 300], [0.01, 300], "k--", alpha=0.75, zorder=0)
        ax.set_title(f'{result["name"]} (RMSE={result["rmse"]:.2f})', fontsize=11)
        if const_cut_off is not None:
            ax.axhline(y=const_cut_off, color="red", linestyle="--")

    for i in range(n_plots, len(axs)):
        axs[i].set_visible(False)

    for row in range(n_rows):
        left_idx = row * n_cols
        if left_idx < n_plots:
            axs[left_idx].set_ylabel("Predicted Runtime")

    bottom_row_start = (n_rows - 1) * n_cols
    for col in range(n_cols):
        bottom_idx = bottom_row_start + col
        if bottom_idx < n_plots:
            axs[bottom_idx].set_xlabel("Actual Runtime")

    plt.subplots_adjust(wspace=0.33, hspace=0.34)
    return fig, axs


def plot_line(result_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))

    plot_df = (
        result_df.groupby(["name", "solver_number"], sort=False)["rmse"]
        .mean()
        .reset_index()
    )

    for name, group in plot_df.groupby("name", sort=False):
        plt.plot(
            group["solver_number"],
            group["rmse"],
            "o-",
            label=name,
            linewidth=1.5,
            markersize=4,
        )

    plt.xscale("log")
    plt.xlabel("Number of solvers (configurations)")
    plt.ylabel("RMSE (on logarithmized predictions)")
    plt.legend(loc="best", frameon=True)
    plt.xticks(plot_df["solver_number"].unique(), plot_df["solver_number"].unique())
    plt.title("RMSE vs Number of Solvers")
    plt.ylim(0)
    return fig, ax


def wilcoxon_df(
    result_df,
    model_info_list=None,
):
    from scipy.stats import wilcoxon

    result_agg = (
        result_df.groupby(["solver_number", "name"], sort=False)["rmse"]
        .mean()
        .reset_index()
    )

    frames = []
    for solver_number, group in result_agg.groupby("solver_number"):
        group = group.copy()
        best_rmse_i = group["rmse"].argmin()
        best_rsme_model = group.iloc[best_rmse_i]["name"]
        group["p"] = np.nan

        result_solver_number = result_df.loc[
            lambda x: x["solver_number"] == solver_number
        ].pivot_table(index="random_state", columns="name", values="rmse")
        y = result_solver_number[best_rsme_model].to_numpy()

        for i, row in group.iterrows():
            if row["name"] == best_rsme_model:
                continue

            x = result_solver_number[row["name"]].to_numpy()
            stat, p = wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="greater",
                correction=True,
                method="auto",
            )
            group.at[i, "p"] = p
        frames.append(group)

    result_agg = pd.concat(frames)
    p_values = result_agg.pivot_table(index="name", columns="solver_number", values="p")
    result_agg = result_agg.pivot_table(
        index="name", columns="solver_number", values="rmse"
    )

    if model_info_list is not None:
        idx = [x["name"] for x in model_info_list]
        p_values = p_values.loc[idx]
        result_agg = result_agg.loc[idx]

    mask = p_values.isna() | (p_values > 0.05 / (p_values.shape[0] - 1))

    def highlight_mask(x):
        return ["font-weight: bold" if mask.loc[x.name, col] else "" for col in x.index]

    styled_result = (
        result_agg.style.apply(highlight_mask, axis=1)
        .format(precision=3)
        .highlight_min(axis=0)
    )
    return styled_result

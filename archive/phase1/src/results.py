import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["grid.color"] = "#cccccc"
plt.rcParams["axes.xmargin"] = 0


def plot_scatter(plot_df, const_cut_off=None):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
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
        ax.set_title(f'{result["name"]} (RMSE={result["rmse"]:.2f})')
        if const_cut_off is not None:
            ax.axhline(y=const_cut_off, color="red", linestyle="--")

    axs[0].set_ylabel("Predicted Runtime")
    axs[3].set_ylabel("Predicted Runtime")
    axs[3].set_xlabel("Actual Runtime")
    axs[4].set_xlabel("Actual Runtime")
    axs[5].set_xlabel("Actual Runtime")
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
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

# primary imports
import argparse
import os
import re
import sys

# third-party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scienceplots

# local imports setup
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import create_df


# ---- plot settings ----
plt.style.use(["science", "grid"])

TITLE_SIZE = 30
AXIS_SIZE = 26
TICK_SIZE = 24
LEGEND_SIZE = 22
LINEWIDTH = 3

COV_LO, COV_HI = 0.75, 0.95
SAVE = True

GCACI_COLOR_MAP = mpl.cm.Reds
OTHER_COLOR = "#0700C8"
MARGINAL_COLOR = "#2CA02C"


def is_gcaci(method: str) -> bool:
    return "gcaci" in method


def is_up_ocp(method: str) -> bool:
    return "up-ocp" in method


def parse_lr(method: str) -> float:
    m = re.search(r"lr=([0-9]*\.?[0-9]+)", method)
    if m is None:
        return 1
    return float(m.group(1))


def make_color_and_label_functions(df):
    def color_for_method(method: str):
        if is_gcaci(method):
            return GCACI_COLOR_MAP(0.65)
        elif is_up_ocp(method):
            return MARGINAL_COLOR
        return OTHER_COLOR

    def label_for_method(method: str):
        if is_gcaci(method):
            return r"GCACI ($\eta=1.0$)"
        elif is_up_ocp(method):
            return r"UP-OCP"
        return r"POGO"

    return color_for_method, label_for_method


def filter_coverage_band(g):
    in_band = (
        (g["avg_worst_group_coverage"] >= COV_LO)
        & (g["avg_worst_group_coverage"] <= COV_HI)
    )
    return g[in_band].copy()


def style_coverage_axis(ax, title, xlabel):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim([COV_LO - 0.01, COV_HI + 0.01])

    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_SIZE)
    ax.set_ylabel("Lowest Observed Group Coverage", fontsize=AXIS_SIZE)

    ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE, pad=10)
    ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE, pad=10)
    ax.grid(True, linestyle="-", linewidth=1, alpha=0.5)


def save_figure(fig, suffix, stock):
    outdir = os.path.join("figures", stock)
    os.makedirs(outdir, exist_ok=True)

    fig.savefig(
        os.path.join(outdir, f"{suffix}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def plot_figure(
    df,
    x_col,
    title,
    xlabel,
    suffix,
    color_for_method,
    label_for_method,
    stock,
    save=True,
    log_x=False,
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for method, g in df.groupby("method"):
        g = filter_coverage_band(g)

        if g.empty:
            continue

        ax.errorbar(
            g[x_col].to_numpy(),
            g["avg_worst_group_coverage"].to_numpy(),
            fmt="o-",
            color=color_for_method(method),
            linewidth=LINEWIDTH,
            markersize=5,
            capsize=3,
            elinewidth=1,
            label=label_for_method(method),
        )

    if log_x:
        ax.set_xscale("log")

    style_coverage_axis(ax, title=title, xlabel=xlabel)
    # ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()

    if save:
        save_figure(fig, suffix, stock)

    plt.close(fig)
    return fig, ax


def plot_observed_vs_desired(
    df,
    color_for_method,
    label_for_method,
    stock,
    save=True,
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for method, g in df.groupby("method"):
        x = (1 - g["alpha"].to_numpy())
        y = g["avg_worst_group_coverage"].to_numpy()

        ax.plot(
            x,
            y,
            "-o",
            color=color_for_method(method),
            linewidth=LINEWIDTH,
            markersize=5,
            label=label_for_method(method),
        )

    grid = np.linspace(0.75, 0.95, 400)
    ax.plot(grid, grid, "k--", linewidth=4)

    tol = 0.03
    grey = "0.55"
    ax.plot(
        grid,
        np.clip(grid - tol, 0, 1),
        ":",
        color=grey,
        linewidth=4,
    )
    ax.plot(
        grid,
        np.clip(grid + tol, 0, 1),
        ":",
        color=grey,
        linewidth=4,
    )

    ax.set_xlim([0.75 - 1e-2, 0.95+ 1e-2])
    ax.set_ylim([0.75 - 1e-2, 0.95+ 1e-2])

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, pad=10)
    ax.grid(True, linestyle="-", linewidth=1, alpha=0.5)

    ax.set_xlabel("Target Coverage", fontsize=AXIS_SIZE)
    ax.set_ylabel("Lowest Observed Group Coverage", fontsize=AXIS_SIZE)
    ax.set_title("Observed vs. Target Coverage", fontsize=TITLE_SIZE)

    # ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()

    if save:
        save_figure(fig, "cov_vs_dcov", stock)

    plt.close(fig)
    return fig, ax


def generate_figs(stock):
    df = create_df(stock)

    if df.empty:
        print(f"No data found for {stock}. Skipping.")
        return

    color_for_method, label_for_method = make_color_and_label_functions(df)

    plot_figure(
        df=df,
        x_col="avg_set_size_mean",
        title="Group Coverage vs. Radius Length",
        xlabel="Average Radius Length",
        suffix="cov_vs_rad",
        color_for_method=color_for_method,
        label_for_method=label_for_method,
        stock=stock,
        save=SAVE,
        log_x=True,
    )

    plot_figure(
        df=df,
        x_col="avg_longest_group_error_seq",
        title="Group Coverage vs. Adaptivity",
        xlabel="Max Consecutive Miscovers (Across All Groups)",
        suffix="cov_vs_adapt",
        color_for_method=color_for_method,
        label_for_method=label_for_method,
        stock=stock,
        save=SAVE,
    )

    plot_observed_vs_desired(
        df=df,
        color_for_method=color_for_method,
        label_for_method=label_for_method,
        stock=stock,
        save=SAVE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures for a given stock.")
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Stock name, e.g. AAPL, MSFT, DAL",
    )
    args = parser.parse_args()

    generate_figs(args.stock)
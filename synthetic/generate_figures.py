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
LINEWIDTH = 3

COV_LO, COV_HI = 0.75, 0.95
DEFAULT_NUM_GROUPS = 100
SAVE = True

DGP_NAMES = [
    "bounded_no_changepoint",
    "bounded_changepoint",
    "unbounded_slow",
    "unbounded_moderate",
    "unbounded_extreme",
]

GCACI_COLOR_MAP = mpl.cm.Reds
OTHER_COLOR = "#0700C8"


def is_gcaci(method: str) -> bool:
    return "gcaci" in method


def parse_lr(method: str) -> float:
    m = re.search(r"lr=([0-9]*\.?[0-9]+)", method)
    if m is None:
        raise ValueError(f"Could not parse learning rate from method name: {method}")
    return float(m.group(1))


def make_color_and_label_functions(df):
    methods = sorted(df["method"].unique())
    gcaci_methods = [m for m in methods if is_gcaci(m)]

    if gcaci_methods:
        lrs = np.array([parse_lr(m) for m in gcaci_methods], dtype=float)
        norm = mpl.colors.Normalize(vmin=float(lrs.min()), vmax=float(lrs.max()))
    else:
        norm = None

    def color_for_method(method: str):
        if is_gcaci(method):
            u = norm(parse_lr(method))
            u = 0.30 + 0.60 * u
            return GCACI_COLOR_MAP(u)
        return OTHER_COLOR

    def label_for_method(method: str):
        if is_gcaci(method):
            return f"GC-ACI lr={parse_lr(method):g}"
        return method

    return color_for_method, label_for_method


def filter_coverage_band(g):
    in_band = (
        (g["avg_worst_group_coverage"] >= COV_LO)
        & (g["avg_worst_group_coverage"] <= COV_HI)
    )
    return g[in_band].copy()


def style_coverage_axis(ax, title, xlabel):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim([COV_LO - 1e-2, COV_HI + 1e-2])

    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_SIZE)
    ax.set_ylabel("Lowest Observed Group Coverage", fontsize=AXIS_SIZE)

    ax.tick_params(axis="x", which="both", labelsize=TICK_SIZE, pad=10)
    ax.tick_params(axis="y", which="major", labelsize=TICK_SIZE, pad=10)
    ax.grid(True, linestyle="-", linewidth=1, alpha=0.5)


def save_figure(fig, num_groups, dgp_name, suffix):
    outdir = os.path.join("figures", f"k={num_groups}")
    os.makedirs(outdir, exist_ok=True)

    fig.savefig(
        os.path.join(outdir, f"{dgp_name}_{suffix}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def plot_errorbar_figure(
    df,
    num_groups,
    dgp_name,
    x_col,
    xerr_col,
    title,
    xlabel,
    suffix,
    color_for_method,
    label_for_method,
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
            xerr=g[xerr_col].to_numpy(),
            yerr=g["avg_worst_group_coverage_sem"].to_numpy(),
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
        ax.xaxis.set_minor_locator(mtick.LogLocator(base=10, subs=(5.0,)))
        ax.xaxis.set_minor_formatter(mtick.NullFormatter())

    style_coverage_axis(ax, title=title, xlabel=xlabel)
    fig.tight_layout()

    if save:
        save_figure(fig, num_groups, dgp_name, suffix)

    plt.close(fig)
    return fig, ax


def plot_observed_vs_desired(
    df,
    num_groups,
    dgp_name,
    color_for_method,
    label_for_method,
    save=True,
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for method, g in df.groupby("method"):
        x = (1-g['alpha'].to_numpy())
        y = g['avg_worst_group_coverage'].to_numpy()

        ax.plot(
            x,
            y,
            "-o",
            color=color_for_method(method),
            linewidth=LINEWIDTH,
            markersize=5,
            label=label_for_method(method),
        )

    grid = np.linspace(0.75-1e-2, 0.95+1e-2, 400)
    ax.plot(grid, grid, "k--", linewidth=4, label="Observed = Target")

    tol = 0.03
    grey = "0.55"
    ax.plot(
        grid,
        np.clip(grid - tol, 0, 1),
        ":",
        color=grey,
        linewidth=4,
        label="±3% tolerance",
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

    fig.tight_layout()

    if save:
        save_figure(fig, num_groups, dgp_name, "cov_vs_dcov")

    plt.close(fig)
    return fig, ax


def generate_figures_for_dgp(num_groups, dgp_name):
    print(f"Generating figures for k={num_groups}/{dgp_name}")

    results_name = f"k={num_groups}/{dgp_name}"
    df = create_df(results_name)

    # Isolate methods to plot
    df = df[df["method"] != "up-ocp"]

    if df.empty:
        print(f"No data found for {results_name}. Skipping.")
        return

    color_for_method, label_for_method = make_color_and_label_functions(df)

    plot_errorbar_figure(
        df=df,
        num_groups=num_groups,
        dgp_name=dgp_name,
        x_col="avg_set_size_mean",
        xerr_col="avg_set_size_mean_sem",
        title="Group Coverage vs. Radius Length",
        xlabel="Average Radius Length",
        suffix="cov_vs_rad",
        color_for_method=color_for_method,
        label_for_method=label_for_method,
        save=SAVE,
        log_x=True,
    )

    plot_errorbar_figure(
        df=df,
        num_groups=num_groups,
        dgp_name=dgp_name,
        x_col="avg_longest_group_error_seq",
        xerr_col="avg_longest_group_error_seq_sem",
        title="Group Coverage vs. Adaptivity",
        xlabel="Max Consecutive Miscovers (Across All Groups)",
        suffix="cov_vs_adapt",
        color_for_method=color_for_method,
        label_for_method=label_for_method,
        save=SAVE,
    )

    plot_observed_vs_desired(
        df=df,
        num_groups=num_groups,
        dgp_name=dgp_name,
        color_for_method=color_for_method,
        label_for_method=label_for_method,
        save=SAVE,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots for all DGPs.")

    parser.add_argument(
        "--num-groups",
        type=int,
        default=DEFAULT_NUM_GROUPS,
        help=f"Number of groups. Default: {DEFAULT_NUM_GROUPS}",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    num_groups = args.num_groups

    for dgp_name in DGP_NAMES:
        generate_figures_for_dgp(num_groups, dgp_name)


if __name__ == "__main__":
    main()
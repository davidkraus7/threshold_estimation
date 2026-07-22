"""E1 (Figure 1): shape of the profile criterion near gamma0, both designs.

One simulated path per (design, T). Each column is a design; the top two rows
show the profile log-likelihood per observation against gamma at T = 500 and
T = 5000, and the bottom row shows the population density of the threshold
variable q over the same gamma axis, so the reader can line up the density jump
with the kink in the criterion.

Expected: for delta = 0.4 a sharply kinked V with steps at the order statistics
of q and a density that jumps down by the factor x at gamma0; for delta = 0 a
smooth shallow parabola and no density break. The jump is the identifying
discontinuity, so E1 shows where the information about gamma0 comes from.

Notes on faithfulness:
  - The criterion is evaluated on a dense uniform gamma grid, not only at order
    statistics, so the plotted curve is the true continuous profile and its
    visible maximum is the estimate. Sampling only at order statistics (an
    earlier version) left the plotted peak and gamma_hat visibly misaligned
    wherever the sample had a wide gap between order statistics.
  - The four criterion panels share a y-axis, so the endogenous V being far
    deeper and sharper than the exogenous dish -- a headline of the project --
    is visible directly rather than hidden by per-panel autoscaling.

Run: python simulations/experiments/e1_criterion.py

Output: results/e1_criterion.parquet, results/e1_criterion.pdf
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # no display needed; keeps overnight runs headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from threshold_estimation import analytics, config, dgp, likelihood  # noqa: E402

WINDOW = 0.75  # half-width in gamma either side of gamma0
DENSE_POINTS = 261  # uniform criterion evaluations across the window
DENSITY_T = 100_000  # independent sample for the population density panels
EXPERIMENT = "e1"

TRUTH_COLOUR = "0.35"
ESTIMATE_COLOUR = "#c0392b"
CURVE_COLOUR = "#1f4e79"


def criterion_curve(design: config.Design, T: int):
    """The true profile criterion over the window, and its maximiser.

    The plotting grid is every order statistic in the window (so the curve is
    the actual profile, steps and all) unioned with a uniform grid (so wide
    inter-order-statistic gaps are still drawn). gamma_hat is the argmax of this
    same curve, so the marked estimate sits at the plotted peak by construction
    -- there is no separate estimate() whose grid could disagree with the plot.
    """
    z = config.standard_shocks(EXPERIMENT, 0, n=T)
    pi, q = dgp.simulate(design, T, z)

    order_in = q[np.abs(q - design.gamma0) <= WINDOW]
    uniform = np.linspace(design.gamma0 - WINDOW, design.gamma0 + WINDOW, DENSE_POINTS)
    grid = np.unique(np.concatenate([order_in, uniform]))

    # Warm-start each optimisation from the previous grid point's solution
    # (grid is sorted, so neighbours are close). Cold-starting every point makes
    # a T = 5000 panel take minutes; warm-starting brings it to seconds.
    profile = np.empty(grid.size)
    start = None
    for i, g in enumerate(grid):
        value, theta = likelihood.profile_loglik(pi, q, g, start)
        profile[i] = value
        if np.isfinite(value):
            start = theta[likelihood._FREE_IDX]
    profile[~np.isfinite(profile)] = np.nan  # incoherent evaluations -> gaps in the curve
    gamma_hat = float(grid[np.nanargmax(profile)])
    return grid, profile / T, gamma_hat


def density_T_label() -> str:
    """DENSITY_T as 10^k when it is a power of ten, else with thousands commas."""
    k = round(np.log10(DENSITY_T))
    return f"10^{{{k}}}" if 10**k == DENSITY_T else f"{DENSITY_T:,}"


def compute_curves() -> dict:
    curves = {}
    for design in config.DESIGNS:
        for T in config.T_E1:
            print(f"  {design.name}, T = {T} ...", flush=True)
            curves[(design.name, T)] = criterion_curve(design, T)
    return curves


def save_curves(curves: dict) -> None:
    frames = [
        pd.DataFrame({"design": name, "T": T, "gamma": grid,
                      "profile_per_obs": prof, "gamma_hat": ghat})
        for (name, T), (grid, prof, ghat) in curves.items()
    ]
    data = pd.concat(frames, ignore_index=True)
    data["git_sha"] = config.git_sha()
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    data.to_parquet(config.RESULTS / "e1_criterion.parquet")


def load_curves() -> dict:
    data = pd.read_parquet(config.RESULTS / "e1_criterion.parquet")
    curves = {}
    for (name, T), panel in data.groupby(["design", "T"]):
        panel = panel.sort_values("gamma")
        curves[(name, T)] = (panel["gamma"].to_numpy(),
                             panel["profile_per_obs"].to_numpy(),
                             float(panel["gamma_hat"].iloc[0]))
    return curves


def compute_densities() -> dict:
    densities = {}
    for design in config.DESIGNS:
        big = config.standard_shocks("e1_density", 0, n=DENSITY_T)
        densities[design.name] = dgp.simulate(design, DENSITY_T, big)[1]
    return densities


def make_figure(curves: dict, densities: dict) -> None:
    # Shared y-limits across the four criterion panels: centre each curve at 0
    # and use a common floor, so depth is comparable across designs and T.
    floor = min(np.nanmin(prof - np.nanmax(prof)) for _, prof, _ in curves.values())
    ylim = (1.05 * floor, -0.03 * floor)

    fig, axes = plt.subplots(
        3, 2, figsize=(10.0, 8.3), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 3.0, 1.5]},
    )
    fig.subplots_adjust(top=0.815, bottom=0.07, left=0.09, right=0.98,
                        hspace=0.28, wspace=0.15)

    col_centres = (0.31, 0.75)
    for col, design in enumerate(config.DESIGNS):
        # Design identity as a figure-level column heading, clear of the legend.
        fig.text(col_centres[col], 0.855,
                 f"{design.name}   ($\\delta={design.delta:g}$,  $x={design.x:.3f}$)",
                 ha="center", fontsize=11.5, fontweight="bold")
        for row, T in enumerate(config.T_E1):
            ax = axes[row, col]
            grid, prof, ghat = curves[(design.name, T)]
            centred = prof - np.nanmax(prof)
            ax.plot(grid, centred, lw=1.0, color=CURVE_COLOUR)
            ax.axvline(design.gamma0, color=TRUTH_COLOUR, lw=1.0, ls="--")
            ax.axvline(ghat, color=ESTIMATE_COLOUR, lw=1.2)
            ax.set_ylim(ylim)
            # T label top-left, gamma_hat readout bottom-right: opposite corners,
            # both clear because every curve peaks near the centre.
            ax.annotate(f"$T = {T}$", xy=(0.03, 0.92), xycoords="axes fraction",
                        ha="left", va="top", fontsize=9.5, fontweight="bold", color="0.25")
            ax.annotate(
                f"$\\hat\\gamma = {ghat:+.4f}$\n$\\hat\\gamma-\\gamma_0 = {ghat - design.gamma0:+.4f}$",
                xy=(0.975, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8.5, color=ESTIMATE_COLOUR,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6),
            )
            if col == 0:
                ax.set_ylabel(r"$T^{-1}\ell_T(\gamma)$, centred", fontsize=9)

        # Density panel for this design, on the shared gamma axis.
        ax = axes[2, col]
        q_big = densities[design.name]
        ax.hist(q_big[np.abs(q_big - design.gamma0) <= WINDOW], bins=60,
                density=True, color="0.7", edgecolor="none")
        ax.axvline(design.gamma0, color=TRUTH_COLOUR, lw=1.0, ls="--")
        f_minus, f_plus = analytics.onesided_densities(design)
        ax.annotate(
            f"density of $q$   ($T={density_T_label()}$;  $f_+/f_- = {f_plus / f_minus:.2f} = x$)",
            xy=(0.03, 0.90), xycoords="axes fraction", ha="left", va="top",
            fontsize=8.5, color="0.25",
        )
        ax.set_xlabel(r"$\gamma$")
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("density", fontsize=9)

    axes[0, 0].set_xlim(-WINDOW, WINDOW)

    fig.text(0.5, 0.965, r"E1: profile criterion and the density of $q_t$ near $\gamma_0$",
             ha="center", fontsize=13, fontweight="bold")
    handles = [
        plt.Line2D([], [], color=CURVE_COLOUR, lw=1.6, label="profile criterion"),
        plt.Line2D([], [], color=TRUTH_COLOUR, ls="--", lw=1.3, label=r"$\gamma_0$ (truth)"),
        plt.Line2D([], [], color=ESTIMATE_COLOUR, lw=1.7, label=r"$\hat\gamma$ (estimate)"),
    ]
    legend = fig.legend(
        handles=handles, loc="center", ncol=3, fontsize=10.5,
        frameon=True, fancybox=True, framealpha=1.0, edgecolor="0.8",
        borderpad=0.8, columnspacing=3.0, handlelength=2.6, handletextpad=0.9,
        bbox_to_anchor=(0.5, 0.918),
    )
    legend.get_frame().set_linewidth(0.6)

    fig.savefig(config.RESULTS / "e1_criterion.pdf")
    print(f"\nwrote {config.RESULTS / 'e1_criterion.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", action="store_true",
                        help="rebuild the figure from the cached parquet (same data, no recompute)")
    args = parser.parse_args()

    if args.replot:
        curves = load_curves()
        print("replotting from cached e1_criterion.parquet")
    else:
        curves = compute_curves()
        save_curves(curves)

    make_figure(curves, compute_densities())

    for design in config.DESIGNS:
        f_minus, f_plus = analytics.onesided_densities(design)
        _, _, ghat = curves[(design.name, config.T_E1[-1])]
        print(f"  {design.name}: density jump f+/f- = {f_plus / f_minus:.4f}"
              f" (= |x| = {abs(design.x):.4f}),  gamma_hat(T={config.T_E1[-1]}) = {ghat:+.5f}")


if __name__ == "__main__":
    main()

"""E1 (Figure 1): shape of the profile criterion near gamma0, both designs.

One simulated path per (design, T). Plots the profile log-likelihood per
observation against gamma in a window around gamma0, with an inset histogram of
q_t over the same window.

Expected: for delta = 0.4 a V-shaped criterion with visible steps at the order
statistics of q, and a visible density break in the histogram; for delta = 0 a
smooth parabola and no break. The break is the identifying discontinuity, so E1
shows where the information about gamma0 actually comes from.

Run: python simulations/experiments/e1_criterion.py

Output: results/e1_criterion.parquet, results/e1_criterion.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed; keeps overnight runs headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from threshold_estimation import analytics, config, dgp, estimation  # noqa: E402

WINDOW = 0.75  # half-width in gamma either side of gamma0
MAX_POINTS = 150  # criterion evaluations per panel
EXPERIMENT = "e1"


def profile_near_threshold(design: config.Design, T: int):
    """Profile log-likelihood per observation, on order statistics near gamma0."""
    z = config.standard_shocks(EXPERIMENT, 0, n=T)
    pi, q = dgp.simulate(design, T, z)

    inside = np.unique(q[np.abs(q - design.gamma0) <= WINDOW])
    if inside.size > MAX_POINTS:
        picks = np.unique(np.linspace(0, inside.size - 1, MAX_POINTS).round().astype(int))
        inside = inside[picks]

    est = estimation.estimate(pi, q, grid=inside, refine=False)
    frame = pd.DataFrame(
        {
            "design": design.name,
            "delta": design.delta,
            "T": T,
            "gamma": est.grid,
            "profile_per_obs": est.profile / T,
            "gamma_hat": est.gamma_hat,
        }
    )
    return frame, q


def main() -> None:
    frames, samples = [], {}
    for design in config.DESIGNS:
        for T in config.T_E1:
            print(f"  {design.name}, T = {T} ...", flush=True)
            frame, q = profile_near_threshold(design, T)
            frames.append(frame)
            samples[(design.name, T)] = q
        big = config.standard_shocks("e1_density", 0, n=100_000)
        samples[(design.name, "density")] = dgp.simulate(design, 100_000, big)[1]

    data = pd.concat(frames, ignore_index=True)
    data["git_sha"] = config.git_sha()
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    data.to_parquet(config.RESULTS / "e1_criterion.parquet")

    fig, axes = plt.subplots(len(config.T_E1), len(config.DESIGNS),
                             figsize=(9.5, 6.4), sharex=True)
    for row, T in enumerate(config.T_E1):
        for col, design in enumerate(config.DESIGNS):
            ax = axes[row, col]
            panel = data[(data["design"] == design.name) & (data["T"] == T)]
            centred = panel["profile_per_obs"] - panel["profile_per_obs"].max()
            ax.plot(panel["gamma"], centred, lw=0.9, color="#1f4e79")
            ax.axvline(design.gamma0, color="0.35", lw=0.8, ls="--")
            ax.axvline(panel["gamma_hat"].iloc[0], color="#c0392b", lw=0.8)
            ax.set_title(
                f"{design.name}  (delta = {design.delta:g},  x = {design.x:.3f}),  T = {T}",
                fontsize=9,
            )
            if col == 0:
                ax.set_ylabel(r"$T^{-1}\ell_T(\gamma)$, centred")
            if row == len(config.T_E1) - 1:
                ax.set_xlabel(r"$\gamma$")

            # The inset illustrates the population density of q, not this path:
            # at T = 500 a histogram is far too noisy to show a factor-x break,
            # so draw a large independent sample for it.
            inset = ax.inset_axes((0.06, 0.08, 0.36, 0.30))
            q_big = samples[(design.name, "density")]
            inset.hist(q_big[np.abs(q_big - design.gamma0) <= WINDOW], bins=30,
                       color="0.65", edgecolor="none")
            inset.axvline(design.gamma0, color="0.15", lw=0.9, ls="--")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(
                f"density of $q$  (jump {abs(design.x):.2f}, $T=10^5$)", fontsize=6
            )

    fig.suptitle(
        r"E1: profile criterion and the density of $q_t$ near $\gamma_0$", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(config.RESULTS / "e1_criterion.pdf")
    print(f"\nwrote {config.RESULTS / 'e1_criterion.pdf'}")

    for design in config.DESIGNS:
        f_minus, f_plus = analytics.onesided_densities(design)
        print(
            f"  {design.name}: predicted density jump f+/f- = {f_plus / f_minus:.4f}"
            f"  (= |x| = {abs(design.x):.4f})"
        )


if __name__ == "__main__":
    main()

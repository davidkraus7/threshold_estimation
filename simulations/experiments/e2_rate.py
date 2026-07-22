"""E2 (Figure 2, Table 1): RMSE of gamma_hat against T. The headline exhibit.

Regresses log2 RMSE(gamma_hat) on log2 T for both designs. The conjecture is a
slope near -1 for the endogenous design (T-rate, Chan 1993 style) and near -1/2
for the exogenous one (the continuous-design rate of Chan-Tsay 1998 and Hansen
2017). Common random numbers across designs, so any difference is attributable
to delta alone.

Run:
    python simulations/experiments/e2_rate.py --quick    # ~10 min, R = 100
    python simulations/experiments/e2_rate.py            # full, R = 1000
    python simulations/experiments/e2_rate.py --figures-only

Outputs: results/e2_rate[_quick].parquet, .pdf, _table1.csv

Do not lower --grid-points far or pass --no-refine: the threshold grid is much
coarser than the 1/T error a superconsistent estimator achieves, so without
refinement the RMSE measures the grid and the slope collapses toward -1/2 for
BOTH designs, which looks exactly like a negative result.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from threshold_estimation import config, dgp, estimation  # noqa: E402

EXPERIMENT = "e2"

QUICK = {"T_grid": (250, 500, 1000, 2000), "reps": 100, "grid_points": 96}
# Grid 96 is the validated setting (refinement resolves gamma below its own RMSE;
# see check_grid.py). T grid is powers of two through 8192 for the rerun -- clean
# doublings a referee reads as sample sizes, and one past the saved run's 4000.
FULL = {"T_grid": (256, 512, 1024, 2048, 4096, 8192), "reps": config.R_MAIN, "grid_points": 96}


def run_one(task) -> dict:
    """One replication. Module level so it can be sent to a worker process."""
    design, T, rep, grid_points, refine = task
    z = config.standard_shocks(EXPERIMENT, rep, n=T)
    pi, q = dgp.simulate(design, T, z)
    grid = estimation.gamma_grid(q, max_points=grid_points)
    est = estimation.estimate(pi, q, grid=grid, refine=refine)
    return {
        "design": design.name,
        "delta": design.delta,
        "x": design.x,
        "T": T,
        "rep": rep,
        "gamma_hat": est.gamma_hat,
        "gamma_error": est.gamma_hat - design.gamma0,
        "c1_hat": est.theta_hat[0],
        "kappa1_hat": est.theta_hat[1],
        "kappa2_hat": est.theta_hat[2],
        "c2_hat": est.theta_hat[3],
        "delta_hat": est.theta_hat[4],
        "kink_hat": est.theta_hat[2] - est.theta_hat[1],
        "kink_error": (est.theta_hat[2] - est.theta_hat[1]) - (design.kappa2 - design.kappa1),
        "loglik": est.loglik,
    }


def simulate_all(preset: dict, workers: int, refine: bool) -> pd.DataFrame:
    tasks = [
        (design, T, rep, preset["grid_points"], refine)
        for design in config.DESIGNS
        for T in preset["T_grid"]
        for rep in range(preset["reps"])
    ]
    # Longest first: cost is roughly linear in T, and starting the big tasks
    # early stops the run ending with idle workers waiting on one straggler.
    tasks.sort(key=lambda t: -t[1])
    print(f"{len(tasks)} replications on {workers} workers")

    rows, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(run_one, tasks, chunksize=1):
            rows.append(row)
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(
                    f"  {done:6d}/{len(tasks)}  {elapsed / 60:5.1f} min elapsed,"
                    f"  {(len(tasks) - done) / rate / 60:5.1f} min left",
                    flush=True,
                )
    print(f"finished in {(time.time() - t0) / 60:.1f} min")
    return pd.DataFrame(rows)


def summarise(data: pd.DataFrame) -> pd.DataFrame:
    """Table 1: bias, SD and RMSE with Monte Carlo standard errors."""
    out = []
    for (design, T), cell in data.groupby(["design", "T"]):
        R = len(cell)
        row = {"design": design, "T": T, "R": R}
        for label, errors in (("gamma", cell["gamma_error"]), ("kink", cell["kink_error"])):
            rmse = float(np.sqrt(np.mean(errors**2)))
            row |= {
                f"{label}_bias": errors.mean(),
                f"{label}_bias_se": errors.std(ddof=1) / np.sqrt(R),
                f"{label}_sd": errors.std(ddof=1),
                f"{label}_rmse": rmse,
                f"{label}_rmse_se": rmse / np.sqrt(2 * R),
            }
        out.append(row)
    return pd.DataFrame(out).sort_values(["design", "T"]).reset_index(drop=True)


COLOURS = {"endogenous": "#1f4e79", "exogenous": "#c0392b"}
# Conjectured asymptotic rate per design, for the per-series reference guides.
CONJECTURE = {"endogenous": (-1.0, r"slope $-1$"), "exogenous": (-0.5, r"slope $-1/2$")}


def slope_and_se(data: pd.DataFrame, n_boot: int = 2000) -> dict[str, tuple[float, float]]:
    """Slope of log2 RMSE(gamma_hat) on log2 T, with a bootstrap s.e.

    The s.e. resamples replications within each (design, T) cell and refits,
    so it reflects the Monte Carlo error in the RMSE points -- the honest way
    to report "-0.94 (0.02)" rather than a bare "-0.94".
    """
    rng = np.random.default_rng(0)
    out = {}
    for design, cell in data.groupby("design"):
        by_T = {T: g["gamma_error"].to_numpy() for T, g in cell.groupby("T")}
        Ts = sorted(by_T)
        x = np.log2(np.array(Ts, float))
        rms = lambda e: np.sqrt(np.mean(e**2))  # noqa: E731
        y = np.array([np.log2(rms(by_T[T])) for T in Ts])
        slope = float(np.polyfit(x, y, 1)[0])
        boots = np.empty(n_boot)
        for b in range(n_boot):
            yb = np.array([np.log2(rms(by_T[T][rng.integers(0, by_T[T].size, by_T[T].size)]))
                           for T in Ts])
            boots[b] = np.polyfit(x, yb, 1)[0]
        out[design] = (slope, float(boots.std(ddof=1)))
    return out


def make_figure(table: pd.DataFrame, slopes: dict, path) -> None:
    """Two stacked panels sharing the T axis:

      A  log2 RMSE vs log2 T, with MC error bars, per-series rate guides, and
         slopes (with bootstrap s.e.) in the legend.
      B  T * RMSE vs T -- flat means T-consistent, rising means slower than T;
         a far more legible rate diagnostic than eyeballing guide lines.
    """
    Ts = np.array(sorted(table["T"].unique()), float)
    x = np.log2(Ts)

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(7.8, 8.6), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.55], "hspace": 0.13},
    )
    fig.subplots_adjust(top=0.92, bottom=0.09, left=0.115, right=0.955)

    # ---- Panel A: log2 RMSE ------------------------------------------------
    for design, cell in table.sort_values("T").groupby("design"):
        c = COLOURS[design]
        delta = 0.4 if design == "endogenous" else 0.0
        lr = np.log2(cell["gamma_rmse"].to_numpy())
        se_log = (cell["gamma_rmse_se"] / cell["gamma_rmse"] / np.log(2)).to_numpy()
        slope, sse = slopes[design]
        ax.errorbar(np.log2(cell["T"]), lr, yerr=se_log, fmt="o-", color=c,
                    lw=1.7, ms=5.5, capsize=3, elinewidth=1.1, zorder=3,
                    label=f"{design} switching of $q$  ($\\delta={delta:g}$):"
                          f"   slope ${slope:+.2f}$  (s.e. {sse:.2f})")
        # Rate guide anchored at THIS series' own first point, so it never
        # implies one series beats another's rate.
        ref, txt = CONJECTURE[design]
        ax.plot(x, lr[0] + ref * (x - x[0]), ls=":", color=c, alpha=0.6, lw=1.2, zorder=1)
        gy = lr[0] + ref * (x[-1] - x[0])
        ax.annotate(txt, (x[-1], gy), color=c, alpha=0.85, fontsize=8.5,
                    xytext=(-3, -7 if design == "endogenous" else 7),
                    textcoords="offset points",
                    ha="right", va="top" if design == "endogenous" else "bottom")
    ax.set_ylabel(r"$\log_2 \mathrm{RMSE}(\hat\gamma)$", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.margins(x=0.08)

    # ---- Panel B: T * RMSE -------------------------------------------------
    for design, cell in table.sort_values("T").groupby("design"):
        c = COLOURS[design]
        tr = (cell["T"] * cell["gamma_rmse"]).to_numpy()
        tre = (cell["T"] * cell["gamma_rmse_se"]).to_numpy()
        axd.errorbar(np.log2(cell["T"]), tr, yerr=tre, fmt="o-", color=c,
                     lw=1.7, ms=5.5, capsize=3, elinewidth=1.1)
    axd.set_ylabel(r"$T \cdot \mathrm{RMSE}(\hat\gamma)$", fontsize=11)
    axd.set_xticks(x)
    axd.set_xticklabels([f"{int(t)}" for t in Ts])
    axd.set_xlabel(r"sample size $T$  (spaced by $\log_2 T$)", fontsize=11)

    fig.suptitle(r"E2: convergence rate of $\hat\gamma$ under endogenous vs exogenous switching",
                 fontsize=13, fontweight="bold", y=0.965)
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="R = 100, T up to 2000")
    parser.add_argument("--reps", type=int, help="override the replication count")
    parser.add_argument("--grid-points", type=int, help="override the threshold grid size")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--no-refine", action="store_true",
                        help="diagnostic only; floors the RMSE at the grid resolution")
    parser.add_argument("--figures-only", action="store_true",
                        help="rebuild the figure and table from the cached parquet")
    args = parser.parse_args()

    preset = dict(QUICK if args.quick else FULL)
    if args.reps:
        preset["reps"] = args.reps
    if args.grid_points:
        preset["grid_points"] = args.grid_points

    suffix = "_quick" if args.quick else ""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    cache = config.RESULTS / f"e2_rate{suffix}.parquet"

    if args.figures_only:
        data = pd.read_parquet(cache)
        print(f"loaded {len(data)} replications from {cache.name}"
              f" (git {data['git_sha'].iloc[0]})")
    else:
        print(f"E2 {'quick' if args.quick else 'full'}: T = {preset['T_grid']},"
              f" R = {preset['reps']}, grid = {preset['grid_points']} points")
        data = simulate_all(preset, args.workers, not args.no_refine)
        data["git_sha"] = config.git_sha()
        data["estimator"] = "argmax order statistic"
        data["grid_points"] = preset["grid_points"]
        data.to_parquet(cache)

    table = summarise(data)
    slopes = slope_and_se(data)
    R = int(table["R"].min())
    table.to_csv(config.RESULTS / f"e2_rate{suffix}_table1.csv", index=False)
    make_figure(table, slopes, config.RESULTS / f"e2_rate{suffix}.pdf")
    if R < 25:
        print(f"\nWARNING: only {R} replications per cell -- slopes are noise, not results.")

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\nTable 1: RMSE of gamma_hat and of the regime contrast")
    print(
        table[["design", "T", "R", "gamma_bias", "gamma_sd", "gamma_rmse",
               "gamma_rmse_se", "kink_rmse"]].to_string(index=False, float_format="%.5f")
    )
    print("\nFitted slopes of log2 RMSE(gamma_hat) on log2 T   [slope (bootstrap s.e.)]:")
    for design, (slope, sse) in slopes.items():
        expected = -1.0 if design == "endogenous" else -0.5
        print(f"  {design:12s} {slope:+.3f} ({sse:.3f})   (conjectured {expected:+.1f})")
    print(f"\nwrote {cache.name}, e2_rate{suffix}.pdf, e2_rate{suffix}_table1.csv")


if __name__ == "__main__":
    main()

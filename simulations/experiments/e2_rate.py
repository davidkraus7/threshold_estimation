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
FULL = {"T_grid": config.T_GRID, "reps": config.R_MAIN, "grid_points": config.MAX_GRID_POINTS}


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


def fitted_slopes(table: pd.DataFrame) -> dict[str, float]:
    """Slope of log2 RMSE(gamma_hat) on log2 T, by design."""
    slopes = {}
    for design, cell in table.groupby("design"):
        slopes[design] = float(
            np.polyfit(np.log2(cell["T"]), np.log2(cell["gamma_rmse"]), 1)[0]
        )
    return slopes


def make_figure(table: pd.DataFrame, slopes: dict[str, float], path, provenance: str) -> None:
    """Provenance goes on the figure itself, so a stale or underpowered
    exhibit identifies itself instead of being mistaken for a finished one."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    colours = {"endogenous": "#1f4e79", "exogenous": "#c0392b"}
    for design, cell in table.groupby("design"):
        ax.plot(
            np.log2(cell["T"]), np.log2(cell["gamma_rmse"]),
            "o-", color=colours.get(design, None), lw=1.4, ms=5,
            label=f"{design} (fitted slope {slopes[design]:+.2f})",
        )

    anchor_x = np.log2(table["T"].min())
    for slope, style, text in ((-1.0, ":", "slope $-1$"), (-0.5, "--", r"slope $-1/2$")):
        anchor_y = np.log2(table["gamma_rmse"].max())
        xs = np.log2(np.array(sorted(table["T"].unique()), dtype=float))
        ax.plot(xs, anchor_y + slope * (xs - anchor_x), style, color="0.5", lw=1.0)
        ax.annotate(text, (xs[-1], anchor_y + slope * (xs[-1] - anchor_x)),
                    fontsize=8, color="0.4", va="center", ha="left")

    ax.set_xlabel(r"$\log_2 T$")
    ax.set_ylabel(r"$\log_2 \mathrm{RMSE}(\hat\gamma)$")
    ax.set_title("E2: convergence rate of the threshold estimator", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    R = int(table["R"].min())
    warning = "  ***  TOO FEW REPLICATIONS TO INTERPRET  ***" if R < 25 else ""
    fig.text(0.5, 0.015, provenance + warning, ha="center", fontsize=7,
             color="#c0392b" if warning else "0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
        data.to_parquet(cache)

    table = summarise(data)
    slopes = fitted_slopes(table)
    R = int(table["R"].min())
    provenance = (
        f"R = {R} replications per cell,  grid = {preset['grid_points']} points,"
        f"  git {data['git_sha'].iloc[0]}"
    )
    table.to_csv(config.RESULTS / f"e2_rate{suffix}_table1.csv", index=False)
    make_figure(table, slopes, config.RESULTS / f"e2_rate{suffix}.pdf", provenance)
    if R < 25:
        print(f"\nWARNING: only {R} replications per cell -- slopes are noise, not results.")

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\nTable 1: RMSE of gamma_hat and of the regime contrast")
    print(
        table[["design", "T", "R", "gamma_bias", "gamma_sd", "gamma_rmse",
               "gamma_rmse_se", "kink_rmse"]].to_string(index=False, float_format="%.5f")
    )
    print("\nFitted slopes of log2 RMSE(gamma_hat) on log2 T:")
    for design, slope in slopes.items():
        expected = -1.0 if design == "endogenous" else -0.5
        print(f"  {design:12s} {slope:+.3f}   (conjectured {expected:+.1f})")
    print(f"\nwrote {cache.name}, e2_rate{suffix}.pdf, e2_rate{suffix}_table1.csv")


if __name__ == "__main__":
    main()

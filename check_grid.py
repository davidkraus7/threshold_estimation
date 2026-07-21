"""Diagnostic: is the coarse threshold grid losing the peak as T grows?

    python check_grid.py                          # T = 500, 2000 vs exhaustive
    python check_grid.py --T 4000 --reference 256 # T = 4000, 96 vs 256 points

Hypothesis. The profile's peak narrows like 1/T while the coarse grid holds a
fixed number of points, so the coarse search increasingly lands in the wrong
basin and the estimated convergence rate flattens for numerical reasons.

Method. On identical paths, compare the coarse search against a reference
search. `--reference exhaustive` evaluates every trimmed order statistic (the
exact maximiser, but O(T) profile evaluations, so impractical much beyond
T = 2000); `--reference N` uses an N-point grid instead, which is enough to
show whether the coarse setting has stopped tracking a denser one.

Deletable once the grid configuration is settled.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "simulations"))

import numpy as np  # noqa: E402

from threshold_estimation import config, dgp, estimation  # noqa: E402

DESIGN = config.ENDOGENOUS  # the sharp-peaked design, where the risk lives
EXPERIMENT = "e2"  # same stream as E2, so these are the same paths


def one(task):
    T, rep, coarse_points, reference_points = task
    z = config.standard_shocks(EXPERIMENT, rep, n=T)
    pi, q = dgp.simulate(DESIGN, T, z)

    coarse = estimation.estimate(
        pi, q, grid=estimation.gamma_grid(q, max_points=coarse_points)
    )
    reference = estimation.estimate(
        pi, q, grid=estimation.gamma_grid(q, max_points=reference_points),
        refine=reference_points < 10**9,
    )
    return {
        "T": T,
        "rep": rep,
        "coarse": coarse.gamma_hat,
        "reference": reference.gamma_hat,
        "coarse_ll": coarse.loglik,
        "reference_ll": reference.loglik,
        "n_reference": reference.grid.size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[500, 2000])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--coarse", type=int, default=96)
    parser.add_argument("--reference", default="exhaustive",
                        help="'exhaustive' or a grid point count")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    reference_points = 10**9 if args.reference == "exhaustive" else int(args.reference)
    tasks = [(T, rep, args.coarse, reference_points)
             for T in args.T for rep in range(args.reps)]
    tasks.sort(key=lambda t: -t[0])

    print(f"design {DESIGN.name}, {len(tasks)} paths, "
          f"coarse = {args.coarse} points vs reference = {args.reference}\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, tasks, chunksize=1))
    print(f"finished in {(time.time() - t0) / 60:.1f} min\n")

    print(f"{'T':>6} {'ref pts':>8} {'RMSE coarse':>12} {'RMSE ref':>10} "
          f"{'inflation':>10} {'disagree':>9} {'coarse ll worse':>16}")
    for T in sorted(set(args.T)):
        cell = [r for r in rows if r["T"] == T]
        c = np.array([r["coarse"] for r in cell]) - DESIGN.gamma0
        e = np.array([r["reference"] for r in cell]) - DESIGN.gamma0
        rmse_c, rmse_e = np.sqrt(np.mean(c**2)), np.sqrt(np.mean(e**2))
        differ = np.mean([abs(r["coarse"] - r["reference"]) > 1e-9 for r in cell])
        worse = np.mean([r["coarse_ll"] < r["reference_ll"] - 1e-6 for r in cell])
        print(
            f"{T:>6} {int(np.mean([r['n_reference'] for r in cell])):>8} "
            f"{rmse_c:>12.5f} {rmse_e:>10.5f} {rmse_c / rmse_e:>9.2f}x "
            f"{differ:>8.0%} {worse:>15.0%}"
        )

    print("\nRead: 'coarse ll worse' is the share of paths where the coarse search found")
    print("a strictly lower likelihood -- direct evidence of a missed peak. If that share")
    print("and the RMSE inflation stay near zero and 1.0x, the coarse setting is safe.")


if __name__ == "__main__":
    main()

"""Diagnostic: how loose can the inner optimiser tolerance be?

Run: python check_tolerance.py

The Nelder-Mead tolerance in likelihood.TOLERANCE only has to be tight enough
to rank neighbouring grid points, whose profile values differ by O(1). If a
looser setting leaves gamma_hat unchanged across replications and designs, it
is free speed on every experiment in the suite.

Deletable once the tolerance is settled.
"""

from __future__ import annotations

import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "simulations"))

import numpy as np  # noqa: E402

from threshold_estimation import config, dgp, estimation, likelihood  # noqa: E402

T = 1000
REPS = 30
TOLERANCES = (1e-8, 1e-6, 1e-4, 1e-3)


def one(task):
    design, rep, tol = task
    likelihood.TOLERANCE = tol  # module-level, so each worker picks it up
    z = config.standard_shocks("e2", rep, n=T)
    pi, q = dgp.simulate(design, T, z)
    started = time.time()
    est = estimation.estimate(pi, q, grid=estimation.gamma_grid(q, max_points=96))
    return {
        "design": design.name,
        "rep": rep,
        "tol": tol,
        "gamma_hat": est.gamma_hat,
        "loglik": est.loglik,
        "seconds": time.time() - started,
    }


def main() -> None:
    tasks = [(d, r, tol) for d in config.DESIGNS for r in range(REPS) for tol in TOLERANCES]
    print(f"{len(tasks)} estimations at T = {T}\n")
    with ProcessPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(one, tasks, chunksize=1))

    baseline = {(r["design"], r["rep"]): r for r in rows if r["tol"] == TOLERANCES[0]}
    print(f"{'design':<12}{'tol':>8}{'mean s':>9}{'speedup':>9}"
          f"{'max |d gamma|':>15}{'max |d loglik|':>16}{'changed':>9}")
    for design in config.DESIGNS:
        base_time = np.mean([r["seconds"] for r in rows
                             if r["design"] == design.name and r["tol"] == TOLERANCES[0]])
        for tol in TOLERANCES:
            cell = [r for r in rows if r["design"] == design.name and r["tol"] == tol]
            d_gamma = [abs(r["gamma_hat"] - baseline[(r["design"], r["rep"])]["gamma_hat"])
                       for r in cell]
            d_ll = [abs(r["loglik"] - baseline[(r["design"], r["rep"])]["loglik"])
                    for r in cell]
            mean_s = np.mean([r["seconds"] for r in cell])
            print(
                f"{design.name:<12}{tol:>8.0e}{mean_s:>9.2f}{base_time / mean_s:>8.2f}x"
                f"{max(d_gamma):>15.2e}{max(d_ll):>16.2e}"
                f"{sum(g > 1e-12 for g in d_gamma):>6}/{len(cell)}"
            )

    print("\nRead: 'changed' counts replications whose gamma_hat moved at all relative")
    print("to the tightest tolerance. Zero changes plus a real speedup means free speed.")


if __name__ == "__main__":
    main()

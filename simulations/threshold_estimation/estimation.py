"""Threshold estimation: grid search over gamma, profiling out theta.

The profile likelihood jumps at each order statistic of q -- that is where the
regime counts in the Jacobian term change -- so it is maximised over a grid of
order statistics rather than by a smooth optimiser. The maximiser is an
interval, and gamma_hat is its midpoint, as in Yu (2012).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from . import likelihood
from .config import MAX_GRID_POINTS, TRIM


@dataclass(frozen=True)
class Estimate:
    gamma_hat: float
    theta_hat: np.ndarray  # ordered as likelihood.PARAM_NAMES
    loglik: float
    grid: np.ndarray
    profile: np.ndarray


def gamma_grid(
    q: np.ndarray, trim: float | None = None, max_points: int | None = None
) -> np.ndarray:
    """Candidate thresholds: order statistics of q, trimmed and thinned.

    Trimming to the central 1 - 2*trim keeps both regimes populated at every
    candidate; thinning caps the cost of one estimation.
    """
    trim = TRIM if trim is None else trim
    max_points = MAX_GRID_POINTS if max_points is None else max_points

    ordered = np.sort(q)
    lo = int(np.floor(trim * ordered.size))
    hi = int(np.ceil((1.0 - trim) * ordered.size))
    candidates = np.unique(ordered[lo:hi])

    if candidates.size > max_points:
        picks = np.unique(np.linspace(0, candidates.size - 1, max_points).round().astype(int))
        candidates = candidates[picks]
    return candidates


def _profile_over(pi: np.ndarray, q: np.ndarray, grid: np.ndarray):
    """Profile log-likelihood at every grid point, warm-starting along the way."""
    values = np.empty(grid.size)
    thetas = np.empty((grid.size, len(likelihood.PARAM_NAMES)))
    start = None
    for i, gamma in enumerate(grid):
        values[i], thetas[i] = likelihood.profile_loglik(pi, q, gamma, start)
        start = thetas[i][likelihood._FREE_IDX]
    return values, thetas


def estimate(
    pi: np.ndarray,
    q: np.ndarray,
    grid: np.ndarray | None = None,
    refine: bool = True,
    n_basins: int = 8,
) -> Estimate:
    """Maximise the profile likelihood over gamma.

    The profile jumps at every order statistic, so a thinned grid samples an
    unrepresentative value at each point and its argmax need not lie next to
    the true maximum. Refining around that one point therefore fixes resolution
    but not basin selection, and the failure gets worse as T grows and the peak
    narrows -- which flattens the estimated convergence rate for numerical
    reasons. Refining around the `n_basins` best coarse points instead costs a
    few times more than a single-basin refinement and far less than evaluating
    every order statistic.
    """
    grid = gamma_grid(q) if grid is None else np.asarray(grid, float)
    values, thetas = _profile_over(pi, q, grid)

    if refine and grid.size > 1:
        ranked = np.argsort(values)[::-1][:n_basins]
        windows = [
            q[(q >= grid[max(k - 1, 0)]) & (q <= grid[min(k + 1, grid.size - 1)])]
            for k in ranked
        ]
        fine = np.unique(np.concatenate(windows)) if windows else np.empty(0)
        fine = fine[~np.isin(fine, grid)]
        if fine.size:
            fine_values, fine_thetas = _profile_over(pi, q, fine)
            grid = np.concatenate([grid, fine])
            values = np.concatenate([values, fine_values])
            thetas = np.vstack([thetas, fine_thetas])
            order = np.argsort(grid)
            grid, values, thetas = grid[order], values[order], thetas[order]

    best = int(np.argmax(values))

    # The profile is flat between consecutive order statistics, so the argmax
    # is an interval; take its midpoint.
    gamma_star = grid[best]
    above = q[q > gamma_star]
    gamma_hat = float(0.5 * (gamma_star + above.min())) if above.size else float(gamma_star)

    return Estimate(
        gamma_hat=gamma_hat,
        theta_hat=thetas[best],
        loglik=float(values[best]),
        grid=grid,
        profile=values,
    )


def as_if_known_covariance(
    theta: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float, step: float = 1e-5
) -> np.ndarray:
    """Inverse observed information at fixed gamma, by central differences.

    Treating gamma as known is exactly the approximation E4 tests: whether
    inference that ignores the estimation of gamma is valid at realistic T.
    """
    k = theta.size
    hessian = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = step, step
            second = (
                likelihood.loglik(theta + ei + ej, pi, q, gamma)
                - likelihood.loglik(theta + ei - ej, pi, q, gamma)
                - likelihood.loglik(theta - ei + ej, pi, q, gamma)
                + likelihood.loglik(theta - ei - ej, pi, q, gamma)
            ) / (4.0 * step**2)
            hessian[i, j] = hessian[j, i] = second
    return np.linalg.inv(-hessian)


def wald_intervals(
    est: Estimate, pi: np.ndarray, q: np.ndarray, level: float = 0.95
) -> dict[str, tuple[float, float, float]]:
    """As-if-known confidence intervals, treating gamma_hat as known (E4).

    Returns {name: (estimate, lower, upper)} for each parameter and for the
    regime contrast kappa2 - kappa1, which is linear in theta and so needs no
    delta method. The impact-response interval E4 also calls for is a nonlinear
    function of theta; add it with a delta method when E4 is written.
    """
    theta, gamma = est.theta_hat, est.gamma_hat
    covariance = as_if_known_covariance(theta, pi, q, gamma)

    k = theta.size
    z = float(norm.ppf(0.5 + level / 2.0))
    out: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(likelihood.PARAM_NAMES):
        se = float(np.sqrt(covariance[i, i]))
        out[name] = (float(theta[i]), float(theta[i] - z * se), float(theta[i] + z * se))

    contrast = np.zeros(k)
    contrast[2], contrast[1] = 1.0, -1.0  # kappa2 - kappa1
    point = float(contrast @ theta)
    se = float(np.sqrt(contrast @ covariance @ contrast))
    out["kappa2_minus_kappa1"] = (point, point - z * se, point + z * se)
    return out

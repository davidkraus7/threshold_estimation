"""Gaussian log-likelihood, concentrated in the linear parameters.

For a candidate threshold gamma, with s_t(gamma) = 1 + 1{q_t > gamma},

    l_T(gamma, theta) = sum_t [ -log(2*pi) - log sigma1 - log sigma2
                                - e_1t(gamma)^2/(2 sigma1^2) - e_2t^2/(2 sigma2^2)
                                + log|1 - delta*kappa_{s_t(gamma)}| ]

where e_1t(gamma) = pi_t - c1 - kappa1*min(q_t-gamma, 0) - kappa2*max(q_t-gamma, 0)
and e_2t = q_t - c2 - delta*pi_t. The -log(2*pi) constant is included here so
this is the actual log-likelihood; simulation_plan.tex drops it, which shifts
the level by a constant but changes no difference, derivative or argmax.

Two things this file exists to get right, both of which converge without error
if you get them wrong:

1. The Jacobian term log|1 - delta*kappa_{s_t}| is what makes this a likelihood
   rather than a least-squares criterion, and it is the entire source of
   information about gamma0. The first two terms are continuous in gamma; only
   the regime counts jump. Drop it and the T-rate fails mechanically.

2. The kink regressors are rebuilt at every candidate gamma. Pinning them at
   gamma0 would let a conditional-mean jump leak in and make the estimator
   superconsistent for the wrong reason.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

PARAM_NAMES = ("c1", "kappa1", "kappa2", "c2", "delta", "sigma1", "sigma2")
CONCENTRATED = ("c1", "c2", "sigma1", "sigma2")  # closed form given the rest
FREE = ("kappa1", "kappa2", "delta")

_FREE_IDX = np.array([1, 2, 4])  # kappa1, kappa2, delta within PARAM_NAMES
_CONC_IDX = np.array([0, 3, 5, 6])  # c1, c2, sigma1, sigma2

_LOG_2PI = float(np.log(2.0 * np.pi))


def kink_regressors(q: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """(min(q - gamma, 0), max(q - gamma, 0)); rebuild at every gamma."""
    centred = q - gamma
    return np.minimum(centred, 0.0), np.maximum(centred, 0.0)


def residuals(
    theta: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Structural residuals (e_1t(gamma), e_2t)."""
    c1, kappa1, kappa2, c2, delta = theta[0], theta[1], theta[2], theta[3], theta[4]
    lo, hi = kink_regressors(q, gamma)
    e1 = pi - c1 - kappa1 * lo - kappa2 * hi
    e2 = q - c2 - delta * pi
    return e1, e2


def loglik(theta: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float) -> float:
    """Log-likelihood including the Jacobian term log|det Phi_0^(s_t)|.

    Returns -inf outside the coherent region (the two regime determinants
    differing in sign) and for non-positive sigmas, so an optimiser that
    wanders there is pushed back rather than returning a nan.
    """
    kappa1, kappa2, delta = theta[1], theta[2], theta[4]
    sigma1, sigma2 = theta[5], theta[6]
    if sigma1 <= 0.0 or sigma2 <= 0.0:
        return -np.inf

    det1, det2 = 1.0 - delta * kappa1, 1.0 - delta * kappa2
    if det1 * det2 <= 0.0:
        return -np.inf

    e1, e2 = residuals(theta, pi, q, gamma)
    T = pi.size
    n1 = int(np.count_nonzero(q <= gamma))
    n2 = T - n1

    gaussian = (
        -T * (_LOG_2PI + np.log(sigma1) + np.log(sigma2))
        - np.dot(e1, e1) / (2.0 * sigma1**2)
        - np.dot(e2, e2) / (2.0 * sigma2**2)
    )
    jacobian = n1 * np.log(abs(det1)) + n2 * np.log(abs(det2))
    return float(gaussian + jacobian)


def concentrate(
    free: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float
) -> np.ndarray:
    """Closed-form (c1, c2, sigma1, sigma2) given (kappa1, kappa2, delta) and gamma.

    The Jacobian term involves neither the intercepts nor the sigmas, so these
    four are exactly the Gaussian MLEs given the rest.
    """
    kappa1, kappa2, delta = free
    lo, hi = kink_regressors(q, gamma)

    fitted1 = kappa1 * lo + kappa2 * hi
    c1 = float(np.mean(pi - fitted1))
    e1 = pi - c1 - fitted1
    sigma1 = float(np.sqrt(np.mean(e1**2)))

    c2 = float(np.mean(q - delta * pi))
    e2 = q - c2 - delta * pi
    sigma2 = float(np.sqrt(np.mean(e2**2)))

    return np.array([c1, c2, sigma1, sigma2])


def assemble(free: np.ndarray, conc: np.ndarray) -> np.ndarray:
    """Build a full theta from the free and concentrated blocks."""
    theta = np.empty(len(PARAM_NAMES))
    theta[_FREE_IDX] = free
    theta[_CONC_IDX] = conc
    return theta


def _ols_start(pi: np.ndarray, q: np.ndarray, gamma: float) -> np.ndarray:
    """Data-driven starting values for (kappa1, kappa2, delta).

    Least squares on each equation separately, ignoring the Jacobian term.
    Biased -- that bias is the whole point of the model -- but a good start.
    """
    lo, hi = kink_regressors(q, gamma)
    X1 = np.column_stack([np.ones_like(pi), lo, hi])
    coefs1, *_ = np.linalg.lstsq(X1, pi, rcond=None)
    X2 = np.column_stack([np.ones_like(q), pi])
    coefs2, *_ = np.linalg.lstsq(X2, q, rcond=None)
    return np.array([coefs1[1], coefs1[2], coefs2[1]])


def profile_loglik(
    pi: np.ndarray, q: np.ndarray, gamma: float, start: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Maximise the log-likelihood over theta at fixed gamma.

    Optimises the three free parameters with Nelder-Mead, concentrating the
    other four in closed form at every evaluation. Derivative-free because the
    coherency boundary is a -inf barrier that gradient methods handle badly;
    the problem is only three-dimensional and warm-starting from the previous
    grid point keeps the iteration counts low.

    Returns
    -------
    (value, theta_hat) with theta_hat ordered as PARAM_NAMES.
    """
    free0 = _ols_start(pi, q, gamma) if start is None else np.asarray(start, float)

    def negative(free: np.ndarray) -> float:
        value = loglik(assemble(free, concentrate(free, pi, q, gamma)), pi, q, gamma)
        return np.inf if not np.isfinite(value) else -value

    result = minimize(negative, free0, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000})

    free_hat = result.x
    theta_hat = assemble(free_hat, concentrate(free_hat, pi, q, gamma))
    return loglik(theta_hat, pi, q, gamma), theta_hat

"""Gaussian log-likelihood, concentrated in the linear parameters."""

from __future__ import annotations

import numpy as np

PARAM_NAMES = ("c1", "kappa1", "kappa2", "c2", "delta", "sigma1", "sigma2")
CONCENTRATED = ("c1", "c2", "sigma1", "sigma2")  # closed form given the rest
FREE = ("kappa1", "kappa2", "delta")


def kink_regressors(q: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """(min(q - gamma, 0), max(q - gamma, 0)); rebuild at every gamma."""
    raise NotImplementedError


def residuals(
    theta: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Structural residuals (e_1t(gamma), e_2t)."""
    raise NotImplementedError


def loglik(theta: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float) -> float:
    """Log-likelihood including the Jacobian term log|det Phi_0^(s_t)|."""
    raise NotImplementedError


def concentrate(
    free: np.ndarray, pi: np.ndarray, q: np.ndarray, gamma: float
) -> np.ndarray:
    """Closed-form CONCENTRATED parameters given FREE ones and gamma."""
    raise NotImplementedError


def profile_loglik(
    pi: np.ndarray, q: np.ndarray, gamma: float, start: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Maximise over theta at fixed gamma; returns (value, theta_hat)."""
    raise NotImplementedError

"""Threshold estimation: grid search over gamma, profiling out theta."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    """Candidate thresholds: order statistics of q, trimmed and thinned."""
    raise NotImplementedError


def estimate(
    pi: np.ndarray, q: np.ndarray, grid: np.ndarray | None = None, refine: bool = True
) -> Estimate:
    """Maximise the profile likelihood over gamma."""
    raise NotImplementedError


def wald_intervals(est: Estimate, pi: np.ndarray, q: np.ndarray, level: float = 0.95):
    """As-if-known confidence intervals, treating gamma_hat as known (E4)."""
    raise NotImplementedError

"""Closed-form population quantities, for checking simulations against theory."""

from __future__ import annotations

import numpy as np

from .config import Design


def d(x: float | np.ndarray) -> float | np.ndarray:
    """d(x) = x log x - x + 1."""
    raise NotImplementedError


def branch_moments(design: Design) -> tuple[tuple[float, float], tuple[float, float]]:
    """Mean and sd of each regime branch of q_t: ((mu1, s1), (mu2, s2))."""
    raise NotImplementedError


def onesided_densities(design: Design) -> tuple[float, float]:
    """One-sided densities of q_t at gamma0: (f_minus, f_plus)."""
    raise NotImplementedError


def kl_slopes(design: Design) -> tuple[float, float]:
    """One-sided slopes (b_minus, b_plus) of the population criterion at gamma0."""
    raise NotImplementedError


def lower_regime_share(design: Design) -> float:
    """P(q_t <= gamma0)."""
    raise NotImplementedError

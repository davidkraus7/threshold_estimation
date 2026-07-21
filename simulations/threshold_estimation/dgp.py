"""Simulating the model: branch solver with a per-observation coherency check."""

from __future__ import annotations

import numpy as np

from .config import Design


def branch_candidates(design: Design, eps: np.ndarray) -> np.ndarray:
    """Both regime-branch solutions for q_t, as a (T, 2) array."""
    raise NotImplementedError


def simulate(design: Design, T: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One path (pi, q) of length T from standardised shocks z."""
    raise NotImplementedError

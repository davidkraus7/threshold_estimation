"""Simulating the model: branch solver with a per-observation coherency check.

The system is simultaneous, so q_t cannot be generated recursively. Assuming
regime l holds, the outcome equation reads pi = c1 + kappa_l*(q - gamma0) +
eps_1, and substituting it into q = c2 + delta*pi + eps_2 gives the branch
candidate

    q^(l) = [c2 + delta*(c1 - kappa_l*gamma0 + eps_1) + eps_2] / (1 - delta*kappa_l).

Candidate 1 is realised iff q^(1) <= gamma0, candidate 2 iff q^(2) > gamma0.
Coherency guarantees exactly one holds at every draw; `simulate` verifies that
per observation rather than trusting the parameter-level check in Design.
"""

from __future__ import annotations

import numpy as np

from .config import Design


def branch_candidates(design: Design, eps: np.ndarray) -> np.ndarray:
    """Both regime-branch solutions for q_t, as a (T, 2) array.

    Parameters
    ----------
    design : Design
    eps : (T, 2) array of scaled innovations (eps_1t, eps_2t).
    """
    kappas = np.array([design.kappa1, design.kappa2])
    dets = np.array(design.det_phi)
    numerators = (
        design.c2
        + design.delta * (design.c1 - kappas * design.gamma0 + eps[:, [0]])
        + eps[:, [1]]
    )
    return numerators / dets


def simulate(design: Design, T: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One path (pi, q) of length T from standardised shocks z.

    Parameters
    ----------
    design : Design
    T : sample size.
    z : (n, 2) standardised innovations from config.standard_shocks, n >= T.
        Only the first T rows are used, so smaller samples are prefixes of
        larger ones. Scaling by (sigma1, sigma2) happens here, not in the draw.

    Raises
    ------
    AssertionError
        If any observation admits zero or two admissible branches, which would
        mean coherency has failed despite the parameter-level check.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"z must have shape (n, 2), got {z.shape}")
    if z.shape[0] < T:
        raise ValueError(f"z has only {z.shape[0]} rows, need T = {T}")

    eps = z[:T] * np.array([design.sigma1, design.sigma2])
    candidates = branch_candidates(design, eps)

    admissible = np.column_stack(
        [candidates[:, 0] <= design.gamma0, candidates[:, 1] > design.gamma0]
    )
    n_admissible = admissible.sum(axis=1)
    if not np.all(n_admissible == 1):
        bad = np.flatnonzero(n_admissible != 1)
        raise AssertionError(
            f"coherency violated at {bad.size} of {T} observations "
            f"(first at t={bad[0]}, with {n_admissible[bad[0]]} admissible branches); "
            f"design {design.name!r}, det Phi_0 = {design.det_phi}"
        )

    q = np.where(admissible[:, 0], candidates[:, 0], candidates[:, 1])

    # Build pi from the min/max form rather than from the selected branch, so
    # the outcome equation holds by construction and is exactly continuous at
    # gamma0 regardless of which branch was taken.
    lo = np.minimum(q - design.gamma0, 0.0)
    hi = np.maximum(q - design.gamma0, 0.0)
    pi = design.c1 + design.kappa1 * lo + design.kappa2 * hi + eps[:, 0]

    return pi, q

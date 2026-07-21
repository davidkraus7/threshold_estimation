"""Closed-form population quantities, for checking simulations against theory.

Nothing here simulates: every function is a short expression of a Design. They
exist so that tests and figures can assert against theory rather than against
numbers someone once observed.

Everything follows from one fact. Each branch of q_t is normal, and the
standardised argument at the threshold,

    u = (gamma0 - mu_l) / s_l
      = (gamma0 - c2 - delta*c1) * sign(1 - delta*kappa_l) / sqrt(v),
    v = delta^2*sigma1^2 + sigma2^2,

is free of kappa_l. Under coherency both determinants share a sign, so both
branches share the same u. Hence the regime share is Phi(u), the one-sided
densities are phi(u)/s_l, and their ratio is |x| -- the density jump at the
threshold that identifies gamma0.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .config import Design


def d(x: float) -> float:
    """d(x) = x*log(x) - x + 1. Non-negative, zero iff x == 1, quadratic near 1.

    The local quadratic behaviour is why information about gamma0 vanishes
    quadratically as the system approaches recursiveness.
    """
    if x <= 0.0:
        raise ValueError(f"d(x) needs x > 0, got {x}")
    return x * np.log(x) - x + 1.0


def branch_moments(design: Design) -> tuple[tuple[float, float], tuple[float, float]]:
    """Mean and sd of each regime branch of q_t: ((mu1, s1), (mu2, s2))."""
    root_v = np.sqrt(design.delta**2 * design.sigma1**2 + design.sigma2**2)
    moments = []
    for kappa, det in zip((design.kappa1, design.kappa2), design.det_phi):
        mu = (design.c2 + design.delta * (design.c1 - kappa * design.gamma0)) / det
        moments.append((mu, root_v / abs(det)))
    return moments[0], moments[1]


def _u(design: Design) -> float:
    """The standardised threshold argument, shared by both branches."""
    root_v = np.sqrt(design.delta**2 * design.sigma1**2 + design.sigma2**2)
    numerator = design.gamma0 - design.c2 - design.delta * design.c1
    return float(numerator * np.sign(design.det_phi[0]) / root_v)


def onesided_densities(design: Design) -> tuple[float, float]:
    """One-sided densities of q_t at gamma0: (f_minus, f_plus).

    Their ratio is |x|: the identifying discontinuity, and a McCrary-style
    observable signature of endogenous switching.
    """
    (_, s1), (_, s2) = branch_moments(design)
    ordinate = float(norm.pdf(_u(design)))
    return ordinate / s1, ordinate / s2


def kl_slopes(design: Design) -> tuple[float, float]:
    """One-sided slopes (b_minus, b_plus) of the population criterion at gamma0.

    For KL(gamma0 +/- eta) = b_pm * eta + O(eta^2),

        b_plus  = f_minus(gamma0) * d(x),
        b_minus = f_plus(gamma0)  * d(1/x).

    Note the crossing: the slope above gamma0 uses the density from below.
    Accounting only for the Jacobian term on the misassigned band gives a
    negative value when x < 1, which cannot be a divergence; the missing
    first-order piece is a residual cross term, nonzero exactly because eps_1
    and q_t are correlated through delta. Endogeneity is what makes the
    divergence positive, and at delta = 0 both slopes vanish with it.
    """
    f_minus, f_plus = onesided_densities(design)
    x = design.x
    return f_plus * d(1.0 / x), f_minus * d(x)


def lower_regime_share(design: Design) -> float:
    """P(q_t <= gamma0). Roughly 0.39 at the baseline; E6 pushes it to 0.05."""
    return float(norm.cdf(_u(design)))

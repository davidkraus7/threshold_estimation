"""Parameters and seed policy for the Monte Carlo suite.

Every number governing an experiment lives here, and no other module defines
one. The model and notation are in ../../CLAUDE.md; the experimental design is
in "02. docs/03. simulation plan/simulation_plan.tex".
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SIMULATIONS = Path(__file__).resolve().parents[1]
REPO = SIMULATIONS.parent
RESULTS = SIMULATIONS / "results"


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """A full parameterisation of the DGP. Defaults are the baseline design."""

    name: str
    delta: float
    gamma0: float = 0.0
    c1: float = 0.0
    c2: float = 0.3
    kappa1: float = 0.25
    kappa2: float = 1.5
    sigma1: float = 1.0
    sigma2: float = 1.0

    def __post_init__(self) -> None:
        d1, d2 = self.det_phi
        if d1 * d2 <= 0.0:
            # Coherency fails, so no unique solution for z_t exists. Raising
            # here means a bad design dies at construction, not 10,000
            # replications into an overnight run.
            raise ValueError(
                f"design {self.name!r} is incoherent: det Phi_0 = ({d1:.4g}, {d2:.4g}) "
                f"differ in sign; coherency needs delta < 1/max(kappa1, kappa2)."
            )

    @property
    def det_phi(self) -> tuple[float, float]:
        """(1 - delta*kappa1, 1 - delta*kappa2)."""
        return (1.0 - self.delta * self.kappa1, 1.0 - self.delta * self.kappa2)

    @property
    def x(self) -> float:
        """Determinant ratio: the factor by which the density of q_t jumps."""
        d1, d2 = self.det_phi
        return d2 / d1

    def vary(self, **kwargs) -> "Design":
        """Copy with fields replaced; `name` is suffixed if not supplied."""
        if "name" not in kwargs:
            tag = "_".join(f"{k}{v:g}" for k, v in sorted(kwargs.items()))
            kwargs["name"] = f"{self.name}_{tag}"
        return replace(self, **kwargs)


# At delta = 0.4: det Phi_0 = (0.9, 0.4), x = 0.444, lower regime share ~40%.
ENDOGENOUS = Design(name="endogenous", delta=0.4)

# delta = 0 is a pure regression kink: same mean nonlinearity, x = 1, no jump.
EXOGENOUS = Design(name="exogenous", delta=0.0)

DESIGNS = (ENDOGENOUS, EXOGENOUS)


# --------------------------------------------------------------------------
# Experiment parameters
# --------------------------------------------------------------------------

T_E1 = (500, 5000)
T_GRID = (250, 500, 1000, 2000, 4000)  # E2-E4
T_E5 = 1000
T_E6 = 1000
T_MAX = max((*T_E1, *T_GRID, T_E5, T_E6))

R_MAIN = 1000  # E2, E3, E4
R_E5 = 500
R_E6 = 500

DELTA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)  # E5; coherency needs < 1/kappa2
C2_GRID = (0.3, 1.1, 1.8)  # E6: lower-regime shares ~40%, 15%, 5%
KAPPA_GAP_GRID = (1.25, 0.5, 0.25)  # E6: kappa2 - kappa1, holding kappa1 = 0.25

TRIM = 0.10  # threshold grid spans the central 80% of the q order statistics
MAX_GRID_POINTS = 256


# --------------------------------------------------------------------------
# Seed policy
# --------------------------------------------------------------------------

MASTER_SEED = 20260721
"""Never change: every cached result depends on it implicitly."""


def _experiment_key(experiment: str) -> int:
    """Stable integer key for an experiment name.

    Deliberately not `hash()`, which is randomised per process by
    PYTHONHASHSEED and would give different streams on different days.
    """
    return int.from_bytes(
        hashlib.blake2b(experiment.encode("utf-8"), digest_size=8).digest(), "big"
    )


def replication_rng(experiment: str, rep: int) -> np.random.Generator:
    """Generator for one replication, keyed on (experiment, rep) only.

    The stream must not depend on delta, kappa or any other design parameter:
    that is what makes designs common-random-number comparable, so differences
    between them are attributable to the design alone. The guarantee holds only
    if shocks are drawn before anything design-dependent touches the generator.
    """
    return np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, _experiment_key(experiment), rep])
    )


def standard_shocks(experiment: str, rep: int, n: int = T_MAX) -> np.ndarray:
    """(n, 2) standard normal innovations. Scale by sigma in the DGP, not here.

    Returns the full T_MAX path by default; take the first T rows for sample
    size T so smaller samples are prefixes of larger ones. Nesting this way
    correlates replications across T and reduces noise in the E2 rate estimate.
    """
    return replication_rng(experiment, rep).standard_normal((n, 2))


def git_sha() -> str:
    """Short commit hash, suffixed '-dirty' if uncommitted; for stamping results."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha

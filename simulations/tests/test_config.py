"""Live tests for what is implemented: designs and the seed policy.

These guard the reproducibility contract, which every cached result depends on
silently. They should stay green from day one.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from threshold_estimation import config


class TestDesign:
    def test_baseline_determinant_ratio(self):
        assert config.ENDOGENOUS.det_phi == pytest.approx((0.9, 0.4))
        assert config.ENDOGENOUS.x == pytest.approx(0.4444, abs=1e-4)

    def test_exogenous_benchmark_has_no_density_jump(self):
        assert config.EXOGENOUS.x == pytest.approx(1.0)

    def test_incoherent_design_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="incoherent"):
            config.Design(name="bad", delta=0.9)  # 1 - 0.9*1.5 < 0

    def test_e5_delta_grid_is_coherent_throughout(self):
        for delta in config.DELTA_GRID:
            config.ENDOGENOUS.vary(delta=delta)


class TestSeedPolicy:
    def test_replications_are_reproducible(self):
        a = config.standard_shocks("e2", 7, n=50)
        b = config.standard_shocks("e2", 7, n=50)
        np.testing.assert_array_equal(a, b)

    def test_replications_and_experiments_are_independent(self):
        base = config.standard_shocks("e2", 7, n=50)
        assert not np.allclose(base, config.standard_shocks("e2", 8, n=50))
        assert not np.allclose(base, config.standard_shocks("e5", 7, n=50))

    def test_samples_are_nested_across_T(self):
        """The T = 250 sample must be a prefix of the T = 4000 sample."""
        full = config.standard_shocks("e2", 3, n=4000)
        np.testing.assert_array_equal(full[:250], config.standard_shocks("e2", 3, n=250))

    def test_experiment_key_survives_hash_randomisation(self):
        """The stream must not depend on PYTHONHASHSEED."""
        code = "from threshold_estimation import config; print(config._experiment_key('e2'))"
        keys = {
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=config.SIMULATIONS,
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            for seed in ("0", "1")
        }
        assert len(keys) == 1


def test_git_sha_is_reported():
    assert config.git_sha() != ""

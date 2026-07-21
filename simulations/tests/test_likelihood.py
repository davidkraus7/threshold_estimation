"""Bug classes: missing Jacobian term, and continuity not tracking gamma
(roadmap.tex:324). Both converge without error and corrupt the headline result.

To check once likelihood.py exists, per sanity check (i):
  - loglik minus the Gaussian part equals sum_t log|det Phi_0^(s_t)| exactly;
  - kink regressors are rebuilt at every candidate gamma, so the mean function
    is continuous at gamma and not only at gamma0;
  - the average score in theta vanishes at the truth;
  - concentrated parameters match a free numerical optimum.
"""

import pytest


@pytest.mark.skip(reason="awaiting likelihood.py")
def test_likelihood():
    raise NotImplementedError

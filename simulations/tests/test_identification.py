"""Identification and the shape of the population criterion.

To check once likelihood.py, estimation.py and analytics.py exist, per sanity
checks (ii) and (iv):
  - kappa1 == kappa2 leaves gamma unidentified: the profile is flat;
  - d(x) >= 0, zero iff x == 1, locally quadratic;
  - f_plus / f_minus == |x| at gamma0 (the identifying density jump);
  - [slow] the one-sided slopes of a long-sample criterion match kl_slopes.
    This one is the numerical proof of Lemma 1.
"""

import pytest


@pytest.mark.skip(reason="awaiting likelihood.py, estimation.py, analytics.py")
def test_identification():
    raise NotImplementedError

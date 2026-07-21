"""Bug class: coherency violations passing silently (roadmap.tex:324).

To check once dgp.py exists, per simulation_plan.tex sanity check (iii):
  - exactly one admissible branch at every observation;
  - simulated paths satisfy both structural equations to machine precision;
  - lower-regime share matches analytics.lower_regime_share;
  - at delta = 0 the solution collapses to the recursive one;
  - both designs see identical shocks at the same replication index.
"""

import pytest


@pytest.mark.skip(reason="awaiting dgp.py")
def test_dgp():
    raise NotImplementedError

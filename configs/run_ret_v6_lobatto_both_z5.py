"""Retirement-only A/B test (run A): Lobatto Z=5 on BOTH return and state axes.

Diff from canonical:
  - wealth_min:        0.05 -> 0.13
  - ret_lobatto_Z:     (None, 7, 7) -> (None, 5, 5)
  - state_lobatto_Z:   (None, 7, 7) -> (None, 5, 5)

Pair: configs/run_ret_v6_lobatto_ret_only_z5.py (state Lobatto disabled).
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_ret_v6_lobatto_both_z5"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.13,
    ret_lobatto_Z=(None, 5.0, 5.0),
    state_lobatto_Z=(None, 5.0, 5.0),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)

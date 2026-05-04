"""Retirement-only run v5: canonical with wealth_min=0.13 and Lobatto Z=5.

Diff from canonical:
  - wealth_min:        0.05 -> 0.13
  - ret_lobatto_Z:     (None, 7, 7) -> (None, 5, 5)
  - state_lobatto_Z:   (None, 7, 7) -> (None, 5, 5)

Retirement only (ages 67-99).
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_v5_z5"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.13,
    ret_lobatto_Z=(None, 5.0, 5.0),
    state_lobatto_Z=(None, 5.0, 5.0),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)

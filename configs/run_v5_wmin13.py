"""Retirement-only run v5 (wmin isolation): canonical with wealth_min=0.13.

Diff from canonical:
  - wealth_min:        0.05 -> 0.13

Lobatto Z stays at canonical (None, 7, 7) on both ret and state.
Pair with run_v5_z5.py to isolate wealth_min vs Lobatto-Z effects.

Retirement only (ages 67-99).
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_v5_wmin13"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    wealth_min=0.13,
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)

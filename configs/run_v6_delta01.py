"""Retirement-only run v6: canonical with bequest delta = 0.1.

Diff from canonical:
  - delta_bequest:  module default (0.005) -> 0.10

Pair: configs/run_v6_delta005.py (delta = 0.05).
Retirement only (ages 67-99).
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_v6_delta01"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC
solver_config = CANONICAL_SOLVER._replace(delta_bequest=0.1)
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)

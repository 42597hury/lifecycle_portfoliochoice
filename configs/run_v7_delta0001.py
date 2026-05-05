"""Retirement-only run v7: canonical with bequest delta = 0.001.

Diff from canonical:
  - delta_bequest:  -1 (module default 0.005) -> 0.001

Pair: configs/run_v7_delta0005.py (delta = 0.005).
Retirement only (ages 67-99).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v7_delta0001"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC
solver_config = CANONICAL_SOLVER._replace(delta_bequest=0.001)
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

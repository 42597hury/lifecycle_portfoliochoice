"""Retirement-only run v8: pure canonical (delta_bequest=0.001).

No overrides — inherits everything from canonical.
Retirement only (ages 67-99).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v8_canonical"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL

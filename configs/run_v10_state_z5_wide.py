"""Retirement-only run v10: canonical with axis-1 Lobatto restored at Z=5, wider state grid.

Diff from canonical:
  - state_lobatto_Z:  (None, 7, 7) -> (None, 5, 5)   (restore axis 1, Z=5)
  - state_n_stds:     (2.0, 2.25, 2.25) -> (2.93, 2.93, 2.93)

Return Lobatto canonical (None, 7, 7). Retirement only (ages 67-99).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v10_state_z5_wide"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    state_lobatto_Z=(None, 5.0, 5.0),
    state_n_stds=(2.93, 2.93, 2.93),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

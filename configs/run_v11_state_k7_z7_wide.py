"""Retirement-only run v11: bump state K=5→7 on Lobatto axes, wider state grid.

Diff from canonical:
  - n_state_quad_nodes:  (3, 5, 5) -> (3, 7, 7)        # K bump on Lobatto axes
  - state_lobatto_Z:     (None, 7, 7)  (canonical, unchanged)
  - state_n_stds:        (2.0, 2.25, 2.25) -> (2.93, 2.93, 2.93)

Return Lobatto canonical. Retirement only (ages 67-99).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v11_state_k7_z7_wide"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    n_state_quad_nodes=(3, 7, 7),
    state_lobatto_Z=(None, 7.0, 7.0),
    state_n_stds=(2.93, 2.93, 2.93),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

"""Retirement-only run v12: asymmetric state Lobatto Z (axis 1 = 4, axis 2 = 6.5).

Diff from canonical:
  - state_n_stds:     (2.0, 2.25, 2.25) -> (2.93, 2.93, 2.93)
  - state_lobatto_Z:  (None, 7.0, 7.0) -> (None, 4.0, 6.5)

Return Lobatto canonical (None, 7, 7). Retirement only (ages 67-99).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v12_state_z4_z6p5"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    state_n_stds=(2.93, 2.93, 2.93),
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=(None, 7.0, 7.0),
    n_state_quad_nodes=(3, 5, 5),
    state_lobatto_Z=(None, 4.0, 6.5),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

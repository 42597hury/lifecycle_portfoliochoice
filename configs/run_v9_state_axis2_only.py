"""Run v9: canonical with state Lobatto on axis 2 only (Z=5), wider state grid.

Diff from canonical:
  - state_lobatto_Z:  (None, 7, 7) -> (None, None, 5)
  - state_n_stds:     (2.0, 2.25, 2.25) -> (2.93, 2.93, 2.93)

Drops Lobatto on state axis 1 (spr) entirely; keeps state axis 2 (y_1) at
Z=5 instead of canonical Z=7. Return Lobatto unchanged.
state_n_stds widened to 2.93 per axis (~99% per-axis coverage).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v9_state_axis2_only"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    state_lobatto_Z=(None, None, 5.0),
    state_n_stds=(2.93, 2.93, 2.93),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

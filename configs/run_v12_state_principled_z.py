"""Retirement-only run v12: per-axis principled Lobatto Z at the wide state envelope.

Diff from canonical:
  - n_state_quad_nodes:  (3, 5, 5)        (canonical, unchanged — back from v11's K=7)
  - state_lobatto_Z:     (None, 7, 7) -> (None, 4.13, 6.51)   # per-axis principled
  - state_n_stds:        (2.0, 2.25, 2.25) -> (2.93, 2.93, 2.93)

Z is set to n_stds[k] * L_z[k,k] / L_state[k,k] per axis (§3.3 in
LOBATTO_CONFIG_TRACKER.md). At n_stds=2.93 with the current calibration
(L_z=(0.167, 0.0158, 0.0165), L_state=(0.167, 0.0112, 0.0074)) the principled
values are (_, 4.13, 6.51). v11 overshot axis 1 by 1.70x and used K=7 which
created Newton cliffs at fictional bankruptcy boundaries past the grid edge;
v12 reverts K to 5 and lands the tail node on the corner exactly.

Return Lobatto canonical. Retirement only (ages 67-99).
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v12_state_principled_z"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    state_lobatto_Z=(None, 4.13, 6.51),
    state_n_stds=(2.93, 2.93, 2.93),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

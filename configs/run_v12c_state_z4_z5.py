"""Retirement-only run v12c: asymmetric state Lobatto Z (axis 1 = 4, axis 2 = 5).

A/B sister of `run_v12_state_z4_z6p5.py`. The two configs differ in ONE knob —
state_lobatto_Z[2] (axis 2). Everything else identical.

  v12  : state_lobatto_Z = (None, 4.0, 6.5)   axis 2 at principled Z (6.51)
  v12c : state_lobatto_Z = (None, 4.0, 5.0)   axis 2 deliberately at v10-style undershoot

Both put axis 1 at its principled Z (4.13 → rounded to 4.0). The difference
isolates the axis-2 contribution: if v12 beats v12c, axis 2's principled-Z
calibration is doing real work; if v12 ≈ v12c, axis 1's calibration was
sufficient and axis 2 was a red herring.

Diff from canonical:
  - state_n_stds:    (2.0, 2.25, 2.25)  -> (2.93, 2.93, 2.93)
  - state_lobatto_Z: (None, 7.0, 7.0)   -> (None, 4.0, 5.0)

Return Lobatto canonical (None, 7, 7). Retirement only (ages 67-99).

See `docs/notes/LOBATTO_CONFIG_TRACKER.md` §3.3 for the principled-Z framework
and §5.2 for the empirical hypothesis this pair tests.
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
    CANONICAL_SOLVE_CONTROL,
)

BUNDLE_SUFFIX = "_v12c_state_z4_z5"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    state_n_stds=(2.93, 2.93, 2.93),
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=(None, 7.0, 7.0),
    n_state_quad_nodes=(3, 5, 5),
    state_lobatto_Z=(None, 4.0, 5.0),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = CANONICAL_SOLVE_CONTROL._replace(youngest_age_to_solve=67)

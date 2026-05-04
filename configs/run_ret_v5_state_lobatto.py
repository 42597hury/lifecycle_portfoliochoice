"""Retirement-only A/B test: STATE LOBATTO arm.

Pair this run with `run_ret_v5_state_gh.py` (the matched-K state GH arm).
The two arms are identical except for state_lobatto_Z. Same K=(3,5,5) on
both ret and state quadrature, same return Lobatto Z=5, same retirement-only
window. Only the state Lobatto toggle differs:

  Arm A (this file): state_lobatto_Z = (None, 5.0, 5.0)
  Arm B (sister):    state_lobatto_Z = None

Goal: isolate whether explicit Lobatto tail nodes on the state-quadrature
axes are doing the bankruptcy-detection work, or whether matched-K GH on
state would do the same job.

Cost-matched: both arms have 75 ret * 75 state = 5,625 cloud nodes per state.
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_ret_v5_state_lobatto"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    n_ret_nodes_1d=(3, 5, 5),
    ret_lobatto_Z=(None, 5.0, 5.0),
    n_state_quad_nodes=(3, 5, 5),
    state_lobatto_Z=(None, 5.0, 5.0),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)

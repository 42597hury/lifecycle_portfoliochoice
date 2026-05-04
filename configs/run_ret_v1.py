"""Retirement-only run: canonical + finer ret/state quadrature.

Solves ages 99→67 only (33 retirement periods).
Overrides: n_ret_nodes_1d=(3,5,7), n_state_quad_nodes=(3,4,7)
"""

from lifecycle.model import SolveControl
from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_ret_v1"

base_config = BASE_CONFIG
disc_config_template = CANONICAL_DISC._replace(
    n_ret_nodes_1d=(3, 5, 7),
    n_state_quad_nodes=(3, 4, 7),
)
solver_config = CANONICAL_SOLVER
SOLVE_CONTROL = SolveControl(youngest_age_to_solve=67)

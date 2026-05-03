"""Smoke-test config: tiniest viable solve for verifying the EC2 pipeline.

Inherits canonical economics + solver from `configs/_canonical.py`;
overrides discretization knobs hard for a 2-5 minute solve. The income/return
quadrature stays at its documented minimum (n=3 for income, (3,5,3) for
returns) to keep the solve numerically valid.

Bundle name will be:
    system_iv_full_var_unconstrained_principal_grid3x3x3_nz5_smoke
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_smoke"

base_config = BASE_CONFIG

disc_config_template = CANONICAL_DISC._replace(
    n_wealth=40,
    n_savings=40,
    state_grid_sizes=(3, 3, 3),
    n_z=5,
    n_state_quad_nodes=(2, 2, 2),
    n_ret_nodes_1d=(3, 5, 3),
)

solver_config = CANONICAL_SOLVER

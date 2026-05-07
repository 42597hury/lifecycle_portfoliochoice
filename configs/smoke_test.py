"""Smoke-test config: tiniest viable solve for verifying the EC2 pipeline.

Inherits canonical economics + solver from `configs/_canonical.py`;
overrides discretization knobs hard for a 2-5 minute solve. State vector is
4D ((dp, spr, rtb, y_1)); return block is (xr, xb).

Bundle name will be:
    system_iv_full_var_unconstrained_cholesky_grid3x3x3x3_nz5_smoke
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
    state_grid_sizes=(3, 3, 3, 3),
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=5,
    n_state_quad_nodes=(2, 2, 2, 2),
    state_lobatto_Z=None,
    n_ret_nodes_1d=(3, 3),
    ret_lobatto_Z=None,
)

solver_config = CANONICAL_SOLVER

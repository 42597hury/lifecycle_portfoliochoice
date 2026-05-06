"""Preflight smoke-test config for the sweep_main launch.

Inherits from configs._canonical so it exercises the EXACT same wealth_min,
state_n_stds, alpha cap, solver tuning, and base economics that the
production cells use. Only the discretization grids/quadrature are shrunk to
the absolute minimum so a single solve takes ~2-5 minutes including Numba
JIT warmup.

Bundle name will be:
    system_iv_full_var_unconstrained_principal_grid3x3x3_nz5_smoketest
"""

from configs._canonical import (
    PREDICTABILITY_SYSTEM,
    BASE_CONFIG,
    CANONICAL_DISC,
    CANONICAL_SOLVER,
)

BUNDLE_SUFFIX = "_smoketest"

base_config = BASE_CONFIG

# Tiniest viable discretization. Income/return quadrature kept at their
# moment-matching minimums (3 / (3,5,3)); state grid + wealth/savings shrunk
# hard. wealth_min, state_n_stds, alpha cap all inherited from canonical.
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

# Solver: identical to canonical (proves the exact same code path).
solver_config = CANONICAL_SOLVER

"""Tiny config for end-to-end CVC pre-commitment pipeline check.

Designed to solve in 2-5 minutes. Drops Lobatto (incompatible with K=2)
and reduces grids enough to keep the solve trivial. NOT a production gate;
this only verifies the simple+clamp ↔ ccv_log dispatch works end-to-end.

For the formal §7.1 gate use a production config (e.g. run_v8_canonical.py)
with --constrained.
"""

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig

PREDICTABILITY_SYSTEM = "IV"
BUNDLE_SUFFIX = "_precommitment_smoke"

base_config = dict(BASE_CONFIG)
base_config["constrained"] = True

disc_config_template = DiscretizationConfig(
    n_wealth=20,
    wealth_min=0.05,
    wealth_max=200.0,
    n_savings=20,
    state_grid_sizes=(3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=2.0,
    n_z=3,
    n_stds=2.0,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=2,
    n_state_quad_nodes=2,
    ret_lobatto_Z=None,
    state_lobatto_Z=None,
)

solver_config = CANONICAL_SOLVER._replace(
    delta_bequest=-1.0,
    alpha_min=-1.0,
    alpha_max=1.0,
)

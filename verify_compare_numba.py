"""Phase 4 — Numba side: solve the SAME tiny 6-age problem, save policies."""
import sys; sys.path.insert(0, ".")
import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, Precompute
from lifecycle.solver import run_lifecycle_solver

# Identical disc to the JAX side.
disc = DiscretizationConfig(
    n_wealth=20, wealth_min=0.13, wealth_max=200.0,
    n_savings=20,
    state_grid_sizes=(3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=5,
    n_eps_nodes=3,
    n_eta_nodes=3,
    n_ret_nodes_1d=(2, 3, 3),
    n_state_quad_nodes=(2, 3, 3),
)
tiny_base = dict(BASE_CONFIG)
tiny_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(tiny_base, var_config, verbose=False)
pc = Precompute(model, disc, verbose=False)

# Numba SolverConfig has alpha_min/max as the leverage cap. To compare against the
# uncapped JAX solver, set them very wide so the cap is non-binding everywhere.
sc = CANONICAL_SOLVER._replace(
    max_iter=100,
    max_iter_unconstrained=100,
    alpha_min=-1e30,
    alpha_max=+1e30,
)
print(f"Numba solving (n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w})...")
C, S, B, _ = run_lifecycle_solver(model, pc, solver_config=sc, verbose=0)

np.savez(
    "/tmp/numba_policies.npz",
    C=C, S=S, B=B,
    wealth_grid=np.asarray(pc.wealth_grid),
    z_grid=np.asarray(pc.z_grid),
    ages=np.asarray(pc.ages),
    state_grid=np.asarray(pc.state_grid),
)
print(f"Numba saved: C.shape={C.shape}  alpha_s [{S.min():.3f}, {S.max():.3f}]  alpha_b [{B.min():.3f}, {B.max():.3f}]")

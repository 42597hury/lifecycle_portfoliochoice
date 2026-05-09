"""Phase 4 — JAX side: solve a tiny 6-age problem, save policies for comparison.

Post real-yields pivot (2026-05-08): 3-axis state vector (cape, spr, y_1).
The Numba reference side of this comparison is the legacy 4-axis nominal
model and is no longer comparable bit-for-bit; this script remains useful
as a quick policy-shape smoke and for tracking JAX-vs-JAX regressions.
"""
import sys; sys.path.insert(0, ".")
import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

# Tiny 6-age config — same shape neighbourhood as verify/smoke.py.
disc = DiscretizationConfig(
    n_wealth=20, wealth_min=0.13, wealth_max=200.0,
    n_savings=20,
    state_grid_sizes=(3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=5,
    n_eps_nodes=3,
    n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 3),
)
tiny_base = dict(BASE_CONFIG)
tiny_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_real_full_var_config_hardcoded()
model = build_model(tiny_base, var_config, verbose=False)
pc = build_precompute(model, disc, verbose=False)

sc = CANONICAL_SOLVER._replace(max_iter=100)
print(f"JAX solving (n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w})...")
C, S, B, _ = run_lifecycle_solver(model, pc, sc, verbose=0)

np.savez(
    "/tmp/jax_policies.npz",
    C=C, S=S, B=B,
    wealth_grid=np.asarray(pc.wealth_grid),
    z_grid=np.asarray(pc.z_grid),
    ages=np.asarray(pc.ages),
    state_grid=np.asarray(pc.state_grid),
)
print(f"JAX saved: C.shape={C.shape}  alpha_s [{S.min():.3f}, {S.max():.3f}]  alpha_b [{B.min():.3f}, {B.max():.3f}]")

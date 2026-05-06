"""Phase 2 smoke: tiny config (3 ages × small grids) — exercises every kernel."""
import time
import numpy as np

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

# Small grids; 6-age window 60..65 covers terminal + retirement + work-to-retire + working.
tiny_disc = DiscretizationConfig(
    n_wealth=20, wealth_min=0.13, wealth_max=200.0,
    n_savings=20,
    # Post rtb-as-state: 4-D state vector (cy, spr, rtb, y_1).
    state_grid_sizes=(3, 3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=5,
    n_eps_nodes=3,
    n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 2, 3),
)

# Override the canonical age range so the smoke is fast.
tiny_base = dict(BASE_CONFIG)
tiny_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(tiny_base, var_config, verbose=False)
pc = build_precompute(model, tiny_disc, verbose=False)

print(f"Built model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}")
print(f"Devices: {len(__import__('jax').devices())}")

# max_iter=8000 (canonical) is overkill on a smooth FOC; 100 gives the same policy.
sc = CANONICAL_SOLVER._replace(max_iter=100, max_iter_unconstrained=100)

t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1)
print(f"\nSolve wall: {time.time() - t0:.1f}s")
print(f"C shape:    {C.shape}")
print(f"NaN check:  C={np.isnan(C).sum()}  S={np.isnan(S).sum()}  B={np.isnan(B).sum()}")
print(f"alpha_s range: [{S.min():.3f}, {S.max():.3f}]")
print(f"alpha_b range: [{B.min():.3f}, {B.max():.3f}]")
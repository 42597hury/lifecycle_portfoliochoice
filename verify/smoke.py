"""Phase 2 smoke: tiny config + solve_control window — exercises every kernel.

Coverage: full retirement (67..99 = 33 ages) + 5 working years (62..66) + the
work->retire boundary at age 66. Per-cell cost is kept low by tiny grids and
quadrature; the wider age window catches more kernel paths than a 6-age slice.
"""
import time
import numpy as np

# --- path bootstrap (verify/ subdir) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, SolveControl
from lifecycle.var import build_real_full_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

tiny_disc = DiscretizationConfig(
    n_wealth=12, wealth_min=0.13, wealth_max=200.0,
    n_savings=12,
    # Post real-yields pivot: 3-axis state vector (cape, spr, y_1).
    state_grid_sizes=(2, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.25),
    n_z=3,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    # Kept asymmetric on purpose: per-axis node-count handling can hide
    # "assumes uniform" bugs that a symmetric (2,2,2) would silently mask.
    n_state_quad_nodes=(2, 3, 3),
)

var_config = build_real_full_var_config_hardcoded()
model = build_model(BASE_CONFIG, var_config, verbose=False)
pc = build_precompute(model, tiny_disc, verbose=False)

print(f"Built model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}")
print(f"Devices: {len(__import__('jax').devices())}")

# max_iter=8000 (canonical) is overkill on a smooth FOC; under fori_loop
# every iter is wall cost, so keep this tight. 30 leaves ~5x margin over
# typical convergence (5-10 iters/cell with backward-age warm-start).
sc = CANONICAL_SOLVER._replace(max_iter=30, max_iter_unconstrained=30)

# Solve full retirement (67..99) + 5 working years (62..66) = 38 ages.
solve_control = SolveControl(youngest_age_to_solve=62)

t0 = time.time()
C, S, B, diag = run_lifecycle_solver(
    model, pc, sc, verbose=1, solve_control=solve_control,
)
print(f"\nSolve wall: {time.time() - t0:.1f}s")
print(f"C shape:    {C.shape}")
solved_mask = diag["solved_age_mask"]
C_s, S_s, B_s = C[solved_mask], S[solved_mask], B[solved_mask]
print(f"Ages solved: {int(solved_mask.sum())} / {pc.n_age}")
print(f"NaN check (solved):  C={np.isnan(C_s).sum()}  S={np.isnan(S_s).sum()}  B={np.isnan(B_s).sum()}")
print(f"alpha_s range: [{S_s.min():.3f}, {S_s.max():.3f}]")
print(f"alpha_b range: [{B_s.min():.3f}, {B_s.max():.3f}]")

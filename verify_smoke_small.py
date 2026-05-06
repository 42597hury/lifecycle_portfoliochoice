"""Reduced smoke (faster than verify_smoke.py) for chunking-fix gating.

Matches verify_smoke.py's structure (6-age window 60-65 covers terminal +
retire + work-to-retire boundary + working) but uses the smaller state
grid + quad config from verify_chunking.py so the run finishes in ~3-5
min on local CPU instead of ~25 min.

Same scope as a verify_smoke run for regression purposes: each of the
four kernel builders (terminal, retirement, working, boundary) gets
exercised at the default cell_vmap_chunks=1, with backward induction
across ages so per-age policies feed the next age's c_next.
"""
import os

# Force single-device CPU before any lifecycle import.
os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")

import time
import numpy as np

import jax  # noqa: E402

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.var import build_nominal_system1_var_config_hardcoded  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.solver import run_lifecycle_solver  # noqa: E402

assert len(jax.devices()) == 1, (
    f"Expected 1 device for vmap-only path; got {len(jax.devices())}. "
    "Set LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 before any lifecycle import."
)

# Smaller config than verify_smoke.py — same shape that verify_chunking uses.
small_disc = DiscretizationConfig(
    n_wealth=15, wealth_min=0.13, wealth_max=200.0,
    n_savings=15,
    state_grid_sizes=(2, 2, 2, 2),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=4,
    n_eps_nodes=2,
    n_eta_nodes=2,
    n_ret_nodes_1d=(2, 2),
    n_state_quad_nodes=(2, 2, 2, 2),
)

# Same age window as verify_smoke.py: 60..65 covers terminal + retire +
# work-to-retire boundary + working.
small_base = dict(BASE_CONFIG)
small_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(small_base, var_config, verbose=False)
pc = build_precompute(model, small_disc, verbose=False)

print(f"Devices: {jax.devices()}  (vmap-only path)")
print(f"Built model: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}")

sc = CANONICAL_SOLVER._replace(max_iter=50, max_iter_unconstrained=50)

t0 = time.time()
C, S, B, diag = run_lifecycle_solver(model, pc, sc, verbose=1)
wall = time.time() - t0

print(f"\nSolve wall: {wall:.1f}s")
print(f"C shape:    {C.shape}")
print(f"NaN check:  C={np.isnan(C).sum()}  S={np.isnan(S).sum()}  B={np.isnan(B).sum()}")
print(f"Inf check:  C={np.isinf(C).sum()}  S={np.isinf(S).sum()}  B={np.isinf(B).sum()}")
print(f"alpha_s range: [{S.min():.3f}, {S.max():.3f}]")
print(f"alpha_b range: [{B.min():.3f}, {B.max():.3f}]")
print(f"Status:     {diag['solve_status']}  ({diag['n_ages_solved']}/{pc.n_age} ages)")

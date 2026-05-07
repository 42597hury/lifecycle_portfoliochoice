"""Drill into per-savings-point n_iters within a single cell.

Monkey-patches `_solve_terminal_at_i_s` / etc. to capture the full
`n_iters_egm` array (shape ``(n_savings + 1,)`` per cell) before the
``jnp.max`` per-cell collapse. This tells us whether the per-cell max
aggregation is the misleading layer or whether Newton is genuinely
running to max_iter at every savings point.
"""
import os

os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")

import sys
import time
import numpy as np
import jax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute

import lifecycle.solver as sm

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

small_base = dict(BASE_CONFIG)
small_base.update(start_age=60, retire_age=63, terminal_age=65)

var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(small_base, var_config, verbose=False)
pc = build_precompute(model, small_disc, verbose=False)

MAX_ITER = 50
sc = CANONICAL_SOLVER._replace(max_iter=MAX_ITER, max_iter_unconstrained=MAX_ITER)

# Patch _egm_scan_cell to return n_iters_egm in full instead of max
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap

# Capture: for one specific cell at terminal, the full n_iters_egm.
# Easiest: directly invoke _solve_terminal_at_i_s for a single i_s.
# That requires constructing the inputs.

# Build the terminal kernel and call it normally; then we need access to
# the per-cell `n_iters_egm` BEFORE the max collapse.

# Approach: temporarily replace `_egm_scan_cell` with a wrapper that
# stashes the array via a side-channel. That's hard inside JIT, so
# instead, monkey-patch `_solve_terminal_at_i_s` directly to return the
# full `n_iters_egm` via `n_iters_max = n_iters_egm` (an array) — then
# build the terminal kernel with that patched function and inspect.

# Easiest of all: re-implement the per-i_s solve by replicating the
# terminal call signature and running it inline (small problem so this
# is fine).

# Build a single terminal solve for i_s=0 via the existing helper.
from lifecycle.solver import (
    _egm_scan_cell, terminal_foc_jac_ccv, _all_is_log_returns_numpy, _pc_to_jnp,
)
from collections import namedtuple

ModelParams = namedtuple("ModelParams", ["gamma", "beta", "b_bar", "delta", "rho"])
mp = ModelParams(
    gamma=jnp.float64(model.gamma),
    beta=jnp.float64(model.beta),
    b_bar=jnp.float64(model.b_bar),
    delta=jnp.float64(sc.delta_bequest),
    rho=jnp.float64(model.rho),
)
pcj = _pc_to_jnp(pc, sc.delta_bequest)

log_R_bill_np, log_x_s_np, log_x_b_np = _all_is_log_returns_numpy(pcj)
log_R_bill = jnp.asarray(log_R_bill_np)
log_x_s = jnp.asarray(log_x_s_np)
log_x_b = jnp.asarray(log_x_b_np)

s_grid = pcj.s_grid
weight_kv_kr = pcj.weight_kv_kr
A_per_state = pcj.annuity_factors
sigma2_xr = pcj.sigma2_xr
sigma2_xb = pcj.sigma2_xb
sigma_xrxb = pcj.sigma_xrxb

def per_is_egm_full(i_s):
    A_is = A_per_state[i_s]
    log_R_bill_i = log_R_bill[i_s]
    log_x_s_i = log_x_s[i_s]
    log_x_b_i = log_x_b[i_s]

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return terminal_foc_jac_ccv(
                a_s, a_b, s_val, A_is,
                log_R_bill_i, log_x_s_i, log_x_b_i, weight_kv_kr,
                sigma2_xr, sigma2_xb, sigma_xrxb,
                mp.gamma, mp.b_bar, mp.delta,
            )
        return foc_fn

    (x_egm, c_egm, a_s_egm, a_b_egm,
     n_iters_egm, n_backtrack_egm, _exit_code_egm) = _egm_scan_cell(
        foc_factory, s_grid,
        sc.init_alpha_s, sc.init_alpha_b,
        mp.gamma, mp.beta,
        sc.tol, sc.max_iter, sc.max_backtrack_iter,
        sc.line_search_max_step, sc.singular_det,
        sc.grad_step_size, sc.grad_denom_eps,
        sc.tiny_savings, sc.euler_inv_floor,
        sc.min_consumption, sc.egm_anchor,
        bool(sc.use_fori_newton),
    )
    return n_iters_egm, n_backtrack_egm

print("Per-savings-point Newton iters at TERMINAL age, shape (n_savings+1,):")
print(f"s_grid (n_savings={pc.n_s} points): "
      f"min={s_grid.min():.3e} max={s_grid.max():.3e}")
print()

for i_s in range(min(pc.N_state, 4)):
    ni, nb = per_is_egm_full(i_s)
    ni_np = np.asarray(ni)
    nb_np = np.asarray(nb)
    print(f"i_s={i_s}: n_iters_egm = {ni_np.tolist()}")
    print(f"        n_backtrack_egm = {nb_np.tolist()}")
    print(f"        max_iters={ni_np.max()}, n_at_max={int((ni_np[1:]==MAX_ITER).sum())}/{ni_np.size-1}")
    print()

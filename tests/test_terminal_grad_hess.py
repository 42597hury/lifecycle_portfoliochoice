"""
test_terminal_grad_hess.py -- Finite-difference verification of
_terminal_portfolio_grad and _terminal_portfolio_hess against
_terminal_portfolio_moment.

At several (alpha_s, alpha_b) points across the simplex:
  (a) Compare analytical gradient to central finite differences of the objective
  (b) Compare analytical Hessian to central finite differences of the gradient
  (c) Verify Hessian symmetry

Uses the same saved run as test_terminal_omega.py.
"""

import numpy as np
import json
from pathlib import Path

# ── Load saved run ───────────────────────────────────────────────────
RUN_DIR = Path("saved_runs/constrained_grid5x5x5_nz11")

with open(RUN_DIR / "metadata.json") as f:
    meta = json.load(f)

base_config = meta["run_config"]["base_config"]
disc_meta = meta["run_config"]["discretization_config"]

from model import DiscretizationConfig
from precompute import build_model, Precompute

var_raw = meta["run_config"]["var_config"]

def restore_ndarray(obj):
    if isinstance(obj, dict) and obj.get("kind") == "ndarray":
        return np.array(obj["values"], dtype=obj["dtype"])
    return obj

var_config = {}
for k, v in var_raw.items():
    if isinstance(v, dict) and v.get("kind") == "ndarray":
        var_config[k] = restore_ndarray(v)
    elif k == "residual_correlation":
        var_config[k] = restore_ndarray(v)
    else:
        var_config[k] = v

for key in ["Phi", "Omega", "z_bar"]:
    if key in var_config and not isinstance(var_config[key], np.ndarray):
        var_config[key] = np.array(var_config[key], dtype=float)
if "const" in var_config and not isinstance(var_config["const"], np.ndarray):
    var_config["const"] = np.array(var_config["const"], dtype=float)

model = build_model(base_config, var_config, verbose=False)

_n_ret_nodes_1d = disc_meta["n_ret_nodes_1d"]
if isinstance(_n_ret_nodes_1d, list):
    _n_ret_nodes_1d = tuple(_n_ret_nodes_1d)

disc_config = DiscretizationConfig(
    n_wealth=disc_meta["n_wealth"],
    n_savings=disc_meta["n_savings"],
    savings_max=disc_meta.get("savings_max"),
    state_grid_sizes=tuple(disc_meta["state_grid_sizes"]),
    n_z=disc_meta["n_z"],
    n_stds=disc_meta["n_stds"],
    n_eps_nodes=disc_meta["n_eps_nodes"],
    n_eta_nodes=disc_meta.get("n_eta_nodes", 3),
    n_ret_nodes_1d=_n_ret_nodes_1d,
)

pc = Precompute(model, disc_config, verbose=False)

from solver import (
    build_gross_return_arrays,
    _terminal_portfolio_moment,
    _terminal_portfolio_grad,
    _terminal_portfolio_hess,
    _terminal_prepare_scenarios,
)
from math import exp

gamma = model.gamma
N_state = pc.N_state

# ── Test points on the simplex ──────────────────────────────────────
# (alpha_s, alpha_b) with alpha_bill = 1 - alpha_s - alpha_b >= 0
test_alphas = [
    (0.3, 0.3),    # interior
    (0.1, 0.5),    # bond-heavy interior
    (0.6, 0.1),    # stock-heavy interior
    (0.05, 0.05),  # near all-bills corner
    (0.45, 0.45),  # near stock-bond edge
    (0.8, 0.1),    # high stock
    (0.1, 0.8),    # high bond
]

# ── Finite difference settings ──────────────────────────────────────
h = 1e-7  # step size for central differences

print("=" * 72)
print("TERMINAL PORTFOLIO GRADIENT & HESSIAN FINITE-DIFFERENCE TEST")
print("=" * 72)
print(f"  gamma={gamma}  N_state={N_state}  h={h}")
print()

all_pass = True
worst_grad_err = 0.0
worst_hess_err = 0.0

# Test at a spread of financial states
test_states = [0, N_state // 4, N_state // 2, 3 * N_state // 4, N_state - 1]

for i_s in test_states:
    # Prepare scenario data for this state
    R_bill = exp(pc.r_bill_grid[i_s])
    Rx_s, Rx_b = build_gross_return_arrays(pc.mu_r[i_s, :, :], pc.ret_nodes)
    scenario_weights, R_stock, R_bond, Rex_s, Rex_b = \
        _terminal_prepare_scenarios(pc.Pi_state[i_s, :], Rx_s, Rx_b,
                                     pc.ret_weights, R_bill)

    for (a_s, a_b) in test_alphas:
        # Skip if too close to simplex boundary for finite differences
        a_bill = 1.0 - a_s - a_b
        if a_bill < 2 * h or a_s < 2 * h or a_b < 2 * h:
            continue

        # ── (a) Gradient: central FD of objective ────────────────────
        analytical_grad = _terminal_portfolio_grad(
            a_s, a_b, R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma
        )

        # df/d(alpha_s)
        f_sp = _terminal_portfolio_moment(a_s + h, a_b, R_bill, scenario_weights, R_stock, R_bond, gamma)
        f_sm = _terminal_portfolio_moment(a_s - h, a_b, R_bill, scenario_weights, R_stock, R_bond, gamma)
        fd_grad_s = (f_sp - f_sm) / (2.0 * h)

        # df/d(alpha_b)
        f_bp = _terminal_portfolio_moment(a_s, a_b + h, R_bill, scenario_weights, R_stock, R_bond, gamma)
        f_bm = _terminal_portfolio_moment(a_s, a_b - h, R_bill, scenario_weights, R_stock, R_bond, gamma)
        fd_grad_b = (f_bp - f_bm) / (2.0 * h)

        fd_grad = np.array([fd_grad_s, fd_grad_b])

        grad_err = np.max(np.abs(analytical_grad - fd_grad) / (np.abs(fd_grad) + 1e-30))
        worst_grad_err = max(worst_grad_err, grad_err)

        if grad_err > 1e-5:
            print(f"  FAIL grad  i_s={i_s} alpha=({a_s:.2f},{a_b:.2f}) "
                  f"err={grad_err:.2e}")
            print(f"    analytical: {analytical_grad}")
            print(f"    FD:         {fd_grad}")
            all_pass = False

        # ── (b) Hessian: central FD of gradient ─────────────────────
        analytical_hess = _terminal_portfolio_hess(
            a_s, a_b, R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma
        )

        # d(grad)/d(alpha_s)
        g_sp = _terminal_portfolio_grad(a_s + h, a_b, R_bill, scenario_weights,
                                         R_stock, R_bond, Rex_s, Rex_b, gamma)
        g_sm = _terminal_portfolio_grad(a_s - h, a_b, R_bill, scenario_weights,
                                         R_stock, R_bond, Rex_s, Rex_b, gamma)
        fd_hess_col0 = (g_sp - g_sm) / (2.0 * h)

        # d(grad)/d(alpha_b)
        g_bp = _terminal_portfolio_grad(a_s, a_b + h, R_bill, scenario_weights,
                                         R_stock, R_bond, Rex_s, Rex_b, gamma)
        g_bm = _terminal_portfolio_grad(a_s, a_b - h, R_bill, scenario_weights,
                                         R_stock, R_bond, Rex_s, Rex_b, gamma)
        fd_hess_col1 = (g_bp - g_bm) / (2.0 * h)

        fd_hess = np.column_stack([fd_hess_col0, fd_hess_col1])

        hess_err = np.max(np.abs(analytical_hess - fd_hess) / (np.abs(fd_hess) + 1e-30))
        worst_hess_err = max(worst_hess_err, hess_err)

        if hess_err > 1e-4:
            print(f"  FAIL hess  i_s={i_s} alpha=({a_s:.2f},{a_b:.2f}) "
                  f"err={hess_err:.2e}")
            print(f"    analytical:\n{analytical_hess}")
            print(f"    FD:\n{fd_hess}")
            all_pass = False

        # ── (c) Hessian symmetry ─────────────────────────────────────
        sym_err = abs(analytical_hess[0, 1] - analytical_hess[1, 0])
        if sym_err > 1e-15:
            print(f"  FAIL symmetry  i_s={i_s} alpha=({a_s:.2f},{a_b:.2f}) "
                  f"|H[0,1]-H[1,0]|={sym_err:.2e}")
            all_pass = False

print()
print("-" * 72)
print(f"Worst gradient relative error:  {worst_grad_err:.2e}")
print(f"Worst Hessian  relative error:  {worst_hess_err:.2e}")
print(f"Gradient tolerance:  1e-5  (central FD with h={h})")
print(f"Hessian  tolerance:  1e-4  (FD of FD, lower accuracy expected)")
print()

if all_pass:
    print("ALL PASSED")
else:
    print("SOME TESTS FAILED")
    raise SystemExit(1)

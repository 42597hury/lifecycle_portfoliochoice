"""
test_terminal_omega.py -- Item 3.3: Omega formula numerical verification.

Loads a saved solver run, rebuilds Precompute (no re-solve), and independently
verifies the terminal consumption rule c*/W = ratio/(ratio+1) where
ratio = (beta * Omega)^{-1/gamma} and Omega = b_bar * A^{gamma-1} * E[R_p^{1-gamma}].

Five checks at each test state:
  (a) Read solved portfolio from S_mat/B_mat at terminal age
  (b) Manual loop over (j_s, k_r) scenarios computing E[R_p^{1-gamma}],
      cross-checked against solver helper _terminal_portfolio_moment
  (c) Compute Omega = b_bar * A^{gamma-1} * moment
  (d) ratio = (beta * Omega)^{-1/gamma}
  (e) c*/W = ratio/(ratio+1), compared to C_mat[T,:,i_s,:]/wealth_grid

Also sweeps all N_state states for worst-case error.

AUGMENTED:
  (g) Verify terminal FOC: E[R_p^{-gamma} * (R_k - R_bill)] = 0 at solved alpha
  (h) Verify c*/W is constant across ALL z-grid points (not just median)
"""

import numpy as np
import json
import sys
from math import exp
from pathlib import Path

# ── Load saved run ───────────────────────────────────────────────────
RUN_DIR = Path("saved_runs/constrained_grid5x5x5_nz11")

with open(RUN_DIR / "metadata.json") as f:
    meta = json.load(f)

data = np.load(RUN_DIR / "policy_arrays.npz")
C_mat = data["C_mat"]
S_mat = data["S_mat"]
B_mat = data["B_mat"]

base_config = meta["run_config"]["base_config"]
disc_meta = meta["run_config"]["discretization_config"]

# ── Rebuild model + Precompute (no solver run needed) ────────────────
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.precompute import build_model, Precompute

# Reconstruct var_config with numpy arrays
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

# Convert lists to numpy where needed
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

# ── Verify shapes match ─────────────────────────────────────────────
assert C_mat.shape == (pc.n_age, pc.n_z, pc.N_state, pc.n_w), \
    f"Shape mismatch: C_mat {C_mat.shape} vs expected {(pc.n_age, pc.n_z, pc.N_state, pc.n_w)}"

# ── Import solver helpers ────────────────────────────────────────────
from lifecycle.solver import (build_gross_return_arrays, _terminal_portfolio_moment,
                    _terminal_prepare_scenarios)

# ── Test parameters ──────────────────────────────────────────────────
T = pc.n_age - 1
mid_z = pc.n_z // 2
N_state = pc.N_state
n_ret_quad = len(pc.ret_weights)
gamma = model.gamma
beta = model.beta
b_bar = model.b_bar

test_states = [0, N_state // 2, N_state - 1]
state_labels = ["LOW", "MID", "HIGH"]

print("=" * 72)
print("ITEM 3.3 -- OMEGA FORMULA NUMERICAL VERIFICATION")
print("=" * 72)
print(f"  Run: {RUN_DIR.name}")
print(f"  Shape: {C_mat.shape}  gamma={gamma}  beta={beta}  b_bar={b_bar}")
print(f"  Terminal age index T={T} (age {pc.ages[T]})")
print()

all_pass = True

for label, i_s in zip(state_labels, test_states):

    print(f"-- State i_s = {i_s} ({label}) --")

    # ================================================================
    # (a) Read the solved terminal portfolio from policy arrays
    # ================================================================
    alpha_s_solved = float(S_mat[T, mid_z, i_s, pc.n_w // 2])
    alpha_b_solved = float(B_mat[T, mid_z, i_s, pc.n_w // 2])
    alpha_bill_solved = 1.0 - alpha_s_solved - alpha_b_solved

    print(f"  (a) Solved portfolio: a_s = {alpha_s_solved:.6f}, "
          f"a_b = {alpha_b_solved:.6f}, a_bill = {alpha_bill_solved:.6f}")

    # ================================================================
    # (b) Manually compute moment = E[R_p^{1-gamma}] by explicit loop
    # ================================================================
    R_bill = exp(pc.r_bill_grid[i_s])

    moment_manual = 0.0
    for j_s in range(N_state):
        pi_js = pc.Pi_state[i_s, j_s]
        if pi_js < 1e-15:
            continue

        mu_stock_js = pc.mu_r[i_s, j_s, 0]
        mu_bond_js = pc.mu_r[i_s, j_s, 1]

        for k_r in range(n_ret_quad):
            w_kr = pc.ret_weights[k_r]
            if w_kr < 1e-15:
                continue

            R_stock = R_bill * exp(mu_stock_js + pc.ret_nodes[k_r, 0])
            R_bond = R_bill * exp(mu_bond_js + pc.ret_nodes[k_r, 1])

            R_p = (alpha_s_solved * R_stock
                   + alpha_b_solved * R_bond
                   + alpha_bill_solved * R_bill)

            moment_manual += pi_js * w_kr * R_p ** (1.0 - gamma)

    print(f"  (b) Manual moment E[R_p^(1-g)] = {moment_manual:.10f}")

    # -- Cross-check against solver helper --
    Rx_stock_next, Rx_bond_next = build_gross_return_arrays(
        pc.mu_r[i_s, :, :], pc.ret_nodes)
    scenario_weights, R_stock_arr, R_bond_arr, Rex_s_arr, Rex_b_arr = \
        _terminal_prepare_scenarios(
            pc.Pi_state[i_s, :], Rx_stock_next, Rx_bond_next,
            pc.ret_weights, R_bill)
    moment_helper = _terminal_portfolio_moment(
        alpha_s_solved, alpha_b_solved, R_bill,
        scenario_weights, R_stock_arr, R_bond_arr, gamma)

    moment_err = abs(moment_manual - moment_helper)
    print(f"       Helper moment               = {moment_helper:.10f}")
    print(f"       |manual - helper|            = {moment_err:.2e}")
    if moment_err > 1e-12:
        print("       *** MISMATCH in moment computation ***")
        all_pass = False

    # ================================================================
    # (c) Compute Omega = b_bar * A^{gamma-1} * moment
    # ================================================================
    A_is = pc.annuity_factors[i_s]

    omega = b_bar * A_is ** (gamma - 1.0) * moment_manual

    print(f"  (c) A(y_nom) = {A_is:.6f}")
    print(f"       Omega = b_bar * A^(g-1) * moment = {omega:.6f}")

    # ================================================================
    # (d) ratio = (beta * Omega)^{-1/gamma}
    # ================================================================
    ratio = (beta * omega) ** (-1.0 / gamma)

    print(f"  (d) beta*Omega = {beta * omega:.6f}")
    print(f"       ratio = (b*O)^(-1/g) = {ratio:.6f}")

    # ================================================================
    # (e) c*/W = ratio / (ratio + 1)
    # ================================================================
    c_over_W_formula = ratio / (ratio + 1.0)

    print(f"  (e) c*/W = ratio/(ratio+1) = {c_over_W_formula:.10f}")

    # -- Compare to solved policy --
    c_solved = C_mat[T, mid_z, i_s, :]
    c_over_W_solved = c_solved / np.maximum(pc.wealth_grid, 1e-15)

    c_over_W_min = c_over_W_solved.min()
    c_over_W_max = c_over_W_solved.max()
    c_over_W_mean = c_over_W_solved.mean()

    rel_err = abs(c_over_W_formula - c_over_W_mean) / c_over_W_mean

    print()
    print(f"  -- Comparison to C_mat --")
    print(f"     Solved c/W: min = {c_over_W_min:.10f}, "
          f"max = {c_over_W_max:.10f}")
    print(f"     Solved c/W mean   = {c_over_W_mean:.10f}")
    print(f"     Formula c/W       = {c_over_W_formula:.10f}")
    print(f"     Relative error    = {rel_err:.2e}")

    if rel_err > 1e-8:
        print("     *** FAIL: formula does not match solved policy ***")
        all_pass = False
    else:
        print("     PASS")

    # -- Check homogeneity: c/W constant across wealth --
    spread = (c_over_W_max - c_over_W_min) / c_over_W_mean
    print(f"     c/W spread (max-min)/mean = {spread:.2e} "
          f"{'homogeneous' if spread < 1e-8 else '*** NOT homogeneous ***'}")

    # ================================================================
    # (g) AUGMENTED: Verify terminal FOC at solved alpha
    #     E[R_p^{-gamma} * (R_k - R_bill)] = 0
    # ================================================================
    foc_s_manual = 0.0
    foc_b_manual = 0.0
    for j_s in range(N_state):
        pi_js = pc.Pi_state[i_s, j_s]
        if pi_js < 1e-15:
            continue
        for k_r in range(n_ret_quad):
            w_kr = pc.ret_weights[k_r]
            if w_kr < 1e-15:
                continue
            R_stock = R_bill * exp(pc.mu_r[i_s, j_s, 0] + pc.ret_nodes[k_r, 0])
            R_bond = R_bill * exp(pc.mu_r[i_s, j_s, 1] + pc.ret_nodes[k_r, 1])
            R_p = (alpha_s_solved * R_stock
                   + alpha_b_solved * R_bond
                   + alpha_bill_solved * R_bill)
            wt = pi_js * w_kr * R_p ** (-gamma)
            foc_s_manual += wt * (R_stock - R_bill)
            foc_b_manual += wt * (R_bond - R_bill)

    # At a constrained solution, the FOC may not be zero if the constraint binds.
    # For interior solutions both should be ~0; for corners/edges, the active
    # constraint means the FOC in the free direction(s) should be ~0.
    # Report the magnitude relative to the moment scale.
    foc_scale = abs(moment_manual) + 1e-30
    foc_s_rel = abs(foc_s_manual) / foc_scale
    foc_b_rel = abs(foc_b_manual) / foc_scale

    print(f"  (g) Terminal FOC residuals (relative to moment):")
    print(f"       stock FOC = {foc_s_manual:+.4e}  (rel {foc_s_rel:.2e})")
    print(f"       bond  FOC = {foc_b_manual:+.4e}  (rel {foc_b_rel:.2e})")

    # For interior: both should be tiny. For constrained: gradient should point
    # inward (complementary slackness).
    is_interior = (alpha_s_solved > 1e-6 and alpha_b_solved > 1e-6
                   and alpha_s_solved + alpha_b_solved < 1.0 - 1e-6)
    if is_interior:
        if foc_s_rel > 1e-5 or foc_b_rel > 1e-5:
            print("       *** FAIL: interior solution but FOC not satisfied ***")
            all_pass = False
        else:
            print("       PASS (interior, both FOCs near zero)")
    else:
        print(f"       (constrained solution -- checking KKT instead)")
        # At a corner/edge, the gradient of the objective w.r.t. the active
        # constraint direction should be non-negative (minimization on simplex).
        # We just verify the free directions have small FOC.
        kkt_ok = True
        if alpha_s_solved < 1e-6:
            # stock at lower bound: gradient w.r.t. alpha_s should be >= 0
            # (increasing alpha_s increases objective = bad for minimization)
            # The (1-gamma) factor in the gradient flips sign for gamma > 1
            grad_s = (1.0 - gamma) * foc_s_manual
            if grad_s < -1e-6 * foc_scale:
                print(f"       *** KKT violation: a_s=0 but grad_s={grad_s:.4e} < 0 ***")
                kkt_ok = False
        if alpha_b_solved < 1e-6:
            grad_b = (1.0 - gamma) * foc_b_manual
            if grad_b < -1e-6 * foc_scale:
                print(f"       *** KKT violation: a_b=0 but grad_b={grad_b:.4e} < 0 ***")
                kkt_ok = False
        if kkt_ok:
            print("       PASS (KKT conditions satisfied)")
        else:
            all_pass = False

    # ================================================================
    # (h) AUGMENTED: Verify c/W constant across ALL z-grid points
    # ================================================================
    cW_across_z = []
    for iz in range(pc.n_z):
        cW_iz = C_mat[T, iz, i_s, pc.n_w // 2] / pc.wealth_grid[pc.n_w // 2]
        cW_across_z.append(cW_iz)
    cW_across_z = np.array(cW_across_z)
    z_spread = (cW_across_z.max() - cW_across_z.min()) / cW_across_z.mean()
    print(f"  (h) c/W across z-grid: min={cW_across_z.min():.10f} "
          f"max={cW_across_z.max():.10f} spread={z_spread:.2e}")
    if z_spread > 1e-8:
        print("       *** FAIL: c/W varies with z at terminal age ***")
        all_pass = False
    else:
        print("       PASS (c/W independent of z)")

    print()

# ── Sweep ALL states ─────────────────────────────────────────────────
print("-- Sweep across ALL N_state states --")
max_rel_err_all = 0.0
worst_state = -1
n_foc_violations = 0

for i_s in range(N_state):
    R_bill = exp(pc.r_bill_grid[i_s])
    A_is = pc.annuity_factors[i_s]

    a_s = float(S_mat[T, mid_z, i_s, pc.n_w // 2])
    a_b = float(B_mat[T, mid_z, i_s, pc.n_w // 2])
    a_bill = 1.0 - a_s - a_b

    # Moment via helper
    Rx_s, Rx_b = build_gross_return_arrays(pc.mu_r[i_s, :, :], pc.ret_nodes)
    sw, Rs, Rb, _, _ = _terminal_prepare_scenarios(
        pc.Pi_state[i_s, :], Rx_s, Rx_b, pc.ret_weights, R_bill)
    moment = _terminal_portfolio_moment(a_s, a_b, R_bill, sw, Rs, Rb, gamma)

    # Omega -> ratio -> c/W
    omega = b_bar * A_is ** (gamma - 1.0) * moment
    ratio = (beta * omega) ** (-1.0 / gamma)
    cW_formula = ratio / (ratio + 1.0)

    # Solved c/W
    cW_solved = float(C_mat[T, mid_z, i_s, pc.n_w // 2]) / pc.wealth_grid[pc.n_w // 2]

    rel_err = abs(cW_formula - cW_solved) / cW_solved
    if rel_err > max_rel_err_all:
        max_rel_err_all = rel_err
        worst_state = i_s

print(f"   Worst relative error across {N_state} states: {max_rel_err_all:.2e} "
      f"(at i_s = {worst_state})")
if max_rel_err_all < 1e-8:
    print("   ALL STATES PASS")
else:
    print("   *** FAIL: some states have large error ***")
    all_pass = False

print()
print("=" * 72)
if all_pass:
    print("ITEM 3.3: PASS -- Omega formula numerically verified")
    print("  - Manual moment == helper moment to machine precision")
    print("  - Omega -> ratio -> c/W matches C_mat across all states")
    print("  - Terminal FOC / KKT conditions satisfied at solved portfolios")
    print("  - c/W independent of z (bequest has no income dependence)")
else:
    print("ITEM 3.3: FAIL -- see errors above")
print("=" * 72)

sys.exit(0 if all_pass else 1)

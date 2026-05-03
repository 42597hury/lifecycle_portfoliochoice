"""
Test Suite: State Innovation Quadrature Validation

Tests arranged from cheapest to most expensive.
Stop on failure — a lower-tier failure means higher tiers will also fail.
"""

import numpy as np
import time
import json
import warnings
from pathlib import Path

from model import DiscretizationConfig, SolverConfig
from precompute import build_model, Precompute
from discretization import get_state_quadrature

warnings.filterwarnings("ignore")

SAVED_RUN = Path("saved_runs/constrained_grid5x5x5_nz11")

# ── Test harness ──
n_pass = 0
n_fail = 0


def check(name, condition, detail=""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  PASS  {name}")
    else:
        n_fail += 1
        print(f"  FAIL  {name}  -- {detail}")


def tier_header(tier, description):
    print(f"\n{'='*70}")
    print(f"  TIER {tier} -- {description}")
    print(f"{'='*70}")


# ── Setup: build once, reuse everywhere ──

def load_var_config():
    with open(SAVED_RUN / "metadata.json") as f:
        meta = json.load(f)
    rc = meta["run_config"]
    vc = rc["var_config"]
    for key in ("Phi", "Omega"):
        val = vc[key]
        vc[key] = np.array(val["values"] if isinstance(val, dict) and "values" in val else val, dtype=float)
    val = vc["z_bar"]
    vc["z_bar"] = np.array(val["values"] if isinstance(val, dict) and "values" in val else val, dtype=float)
    return rc["base_config"], vc


def build_once():
    """Build the primary model + Precompute used by most tests."""
    bc, vc = load_var_config()
    model = build_model(bc, vc, verbose=False)
    dc = DiscretizationConfig(
        state_grid_sizes=(5, 5, 5), n_state_quad_nodes=3,
        n_wealth=50, n_savings=50, n_z=7,
        n_eps_nodes=3, n_eta_nodes=3, n_ret_nodes_1d=2,
    )
    pc = Precompute(model, dc, verbose=False)
    return model, pc


def build_nonsquare():
    """Build a non-square grid variant for T2.11."""
    bc, vc = load_var_config()
    model = build_model(bc, vc, verbose=False)
    dc = DiscretizationConfig(
        state_grid_sizes=(5, 7, 3), n_state_quad_nodes=3,
        n_wealth=50, n_savings=50, n_z=7,
        n_eps_nodes=3, n_eta_nodes=3, n_ret_nodes_1d=2,
    )
    pc = Precompute(model, dc, verbose=False)
    return model, pc


# =====================================================================
# TIER 1 -- Pure Quadrature Mathematics (no Precompute needed)
# =====================================================================

def run_tier_1(model):
    tier_header(1, "Pure Quadrature Mathematics")

    # T1.1 Weights sum to one
    for K in [1, 2, 3, 4, 5]:
        _, w = get_state_quadrature(model, n_nodes=K)
        err = abs(w.sum() - 1.0)
        check(f"T1.1 K={K} weights sum", err < 1e-14, f"err={err:.2e}")

    # T1.2 All weights positive
    for K in [2, 3, 4]:
        _, w = get_state_quadrature(model, n_nodes=K)
        check(f"T1.2 K={K} all weights > 0", np.all(w > 0), f"min={w.min():.2e}")

    # T1.3 Mean is zero
    for K in [2, 3, 4]:
        v, w = get_state_quadrature(model, n_nodes=K)
        err = np.max(np.abs(w @ v))
        check(f"T1.3 K={K} mean zero", err < 1e-13, f"max|mean|={err:.2e}")

    # T1.4 Covariance equals Sigma_ss
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=float)
    for K in [2, 3, 4]:
        v, w = get_state_quadrature(model, n_nodes=K)
        cov = sum(w[k] * np.outer(v[k], v[k]) for k in range(len(w)))
        err = np.max(np.abs(cov - Sigma_ss))
        check(f"T1.4 K={K} covariance", err < 1e-12, f"max err={err:.2e}")

    # T1.5 Third moments zero (skewness = 0 for Gaussian)
    v, w = get_state_quadrature(model, n_nodes=3)
    max_err = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                moment = np.sum(w * v[:, i] * v[:, j] * v[:, k])
                max_err = max(max_err, abs(moment))
    check("T1.5 third moments zero", max_err < 1e-11, f"max={max_err:.2e}")

    # T1.6 Node count correct
    for K in [1, 2, 3, 4]:
        v, w = get_state_quadrature(model, n_nodes=K)
        expected = K ** model.n_state
        check(f"T1.6 K={K} node count", v.shape == (expected, model.n_state),
              f"got {v.shape}, expected ({expected}, {model.n_state})")


# =====================================================================
# TIER 2 -- Precompute Array Validation
# =====================================================================

def run_tier_2(model, pc, pc_nonsquare):
    tier_header(2, "Precompute Array Validation")

    # T2.1 Builds without error (already did, or we wouldn't be here)
    check("T2.1 Precompute builds", True)

    # T2.2 Array shapes
    n_sq = 27  # 3^3
    check("T2.2a v_nodes shape", pc.v_nodes.shape == (n_sq, 3))
    check("T2.2b v_weights shape", pc.v_weights.shape == (n_sq,))
    check("T2.2c M_v_nodes shape", pc.M_v_nodes.shape == (n_sq, 2))
    check("T2.2d const_r shape", pc.const_r.shape == (2,))
    check("T2.2e A_r shape", pc.A_r.shape == (2, 3))
    check("T2.2f exp_ret_stock shape", pc.exp_ret_stock.shape == (pc.n_ret_quad,))
    check("T2.2g exp_ret_bond shape", pc.exp_ret_bond.shape == (pc.n_ret_quad,))

    # T2.3 M_v_nodes consistency
    expected = pc.v_nodes @ np.asarray(model.M, dtype=float).T
    err = np.max(np.abs(pc.M_v_nodes - expected))
    check("T2.3 M_v_nodes = v_nodes @ M.T", err < 1e-14, f"err={err:.2e}")

    # T2.4 Return formula algebraic identity
    rng = np.random.default_rng(42)
    Phi_0_ret = np.asarray(model.Phi_0_ret, dtype=float)
    Phi_21 = np.asarray(model.Phi_21, dtype=float)
    M = np.asarray(model.M, dtype=float)
    max_err = 0.0
    for i in range(min(20, pc.N_state)):
        s_i = pc.state_grid[i]
        for _ in range(5):
            v = rng.standard_normal(3) * 0.01
            mu_1 = pc.const_r + pc.A_r @ s_i + M @ v
            mu_2 = Phi_0_ret + Phi_21 @ s_i + M @ v
            max_err = max(max_err, np.max(np.abs(mu_1 - mu_2)))
    check("T2.4 return formula identity", max_err < 1e-13, f"err={max_err:.2e}")

    # T2.5 Unconditional return mean consistency (P7)
    max_err = 0.0
    for i in range(pc.N_state):
        s_i = pc.state_grid[i]
        avg = pc.const_r + pc.A_r @ s_i + pc.v_weights @ pc.M_v_nodes
        target = Phi_0_ret + Phi_21 @ s_i
        max_err = max(max_err, np.max(np.abs(avg - target)))
    check("T2.5 return mean consistency (P7)", max_err < 1e-12, f"err={max_err:.2e}")

    # T2.6 Conditional state mean (P4)
    Phi_0 = np.asarray(model.Phi_0_state, dtype=float)
    Phi_11 = np.asarray(model.Phi_11, dtype=float)
    max_err = 0.0
    for i in range(pc.N_state):
        s_i = pc.state_grid[i]
        target = Phi_0 + Phi_11 @ s_i
        computed = sum(pc.v_weights[k] * (Phi_0 + Phi_11 @ s_i + pc.v_nodes[k])
                       for k in range(pc.n_state_quad))
        max_err = max(max_err, np.max(np.abs(computed - target)))
    check("T2.6 conditional state mean (P4)", max_err < 1e-12, f"err={max_err:.2e}")

    # T2.7 Conditional state covariance (P5)
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=float)
    max_err = 0.0
    for i in [0, pc.N_state // 4, pc.N_state // 2, 3 * pc.N_state // 4, pc.N_state - 1]:
        cov = sum(pc.v_weights[k] * np.outer(pc.v_nodes[k], pc.v_nodes[k])
                  for k in range(pc.n_state_quad))
        max_err = max(max_err, np.max(np.abs(cov - Sigma_ss)))
    check("T2.7 conditional state covariance (P5)", max_err < 1e-12, f"err={max_err:.2e}")

    # T2.8 State-return innovation covariance (P8)
    cov_sr = sum(pc.v_weights[k] * np.outer(pc.v_nodes[k], M @ pc.v_nodes[k])
                 for k in range(pc.n_state_quad))
    expected_sr = np.asarray(model.Sigma_rs, dtype=float).T
    err = np.max(np.abs(cov_sr - expected_sr))
    check("T2.8 state-return innovation cov (P8)", err < 1e-12, f"err={err:.2e}")

    # T2.9 Next-state/return covariance (P9)
    target_sr = Sigma_ss @ M.T
    err = np.max(np.abs(cov_sr - target_sr))
    check("T2.9 next-state/return cov (P9)", err < 1e-12, f"err={err:.2e}")

    # T2.10 Flat index ordering (5x5x5)
    N0, N1, N2 = pc.state_grid_sizes
    ok = True
    for j_s in range(pc.N_state):
        i0, i1, i2 = pc.state_indices[j_s]
        if j_s != i0 * N1 * N2 + i1 * N2 + i2:
            check("T2.10 flat index ordering", False, f"j_s={j_s}")
            ok = False
            break
    if ok:
        check("T2.10 flat index ordering", True)

    # T2.11 Flat index ordering non-square (5x7x3)
    N0ns, N1ns, N2ns = pc_nonsquare.state_grid_sizes
    ok = True
    for j_s in range(pc_nonsquare.N_state):
        i0, i1, i2 = pc_nonsquare.state_indices[j_s]
        if j_s != i0 * N1ns * N2ns + i1 * N2ns + i2:
            check("T2.11 flat index non-square", False, f"j_s={j_s}")
            ok = False
            break
    if ok:
        check("T2.11 flat index non-square", True)

    # T2.12 exp_ret precomputation
    ok = True
    for k in range(pc.n_ret_quad):
        if abs(pc.exp_ret_stock[k] - np.exp(pc.ret_nodes[k, 0])) > 1e-15 or \
           abs(pc.exp_ret_bond[k] - np.exp(pc.ret_nodes[k, 1])) > 1e-15:
            check("T2.12 exp_ret precomputation", False, f"k={k}")
            ok = False
            break
    if ok:
        check("T2.12 exp_ret precomputation", True)


# =====================================================================
# TIER 3 -- Component Tests (bracket, trilinear)
# =====================================================================

def run_tier_3(model, pc):
    tier_header(3, "Component Tests (bracket, trilinear)")

    # T3.1 Bracket helper
    from solver import bracket_state_3d
    grids_0 = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    grids_1 = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    grids_2 = np.array([-4.5, -4.0, -3.5, -3.0, -2.5])

    lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
        0.0, 0.02, -3.5, grids_0, grids_1, grids_2)
    check("T3.1a exact grid point",
          lo0 == 2 and abs(f0) < 1e-14 and lo1 == 2 and abs(f1) < 1e-14
          and lo2 == 2 and abs(f2) < 1e-14,
          f"lo=({lo0},{lo1},{lo2}), f=({f0},{f1},{f2})")

    lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
        -0.005, 0.015, -3.25, grids_0, grids_1, grids_2)
    check("T3.1b midpoint fracs",
          abs(f0 - 0.5) < 1e-14 and abs(f1 - 0.5) < 1e-14 and abs(f2 - 0.5) < 1e-14,
          f"f=({f0},{f1},{f2})")

    lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
        0.05, -0.01, -5.0, grids_0, grids_1, grids_2)
    check("T3.1c clamping",
          lo0 == 3 and abs(f0 - 1.0) < 1e-14 and lo1 == 0 and abs(f1) < 1e-14
          and lo2 == 0 and abs(f2) < 1e-14,
          f"lo=({lo0},{lo1},{lo2}), f=({f0},{f1},{f2})")

    # T3.3 Trilinear reproduces linear function exactly
    a, b = 2.0, np.array([0.3, -0.5, 0.1])
    c_fake = np.zeros((pc.n_z, pc.N_state, pc.n_w))
    for j_s in range(pc.N_state):
        c_fake[:, j_s, :] = a + np.dot(b, pc.state_grid[j_s])

    rng = np.random.default_rng(42)
    max_err = 0.0
    N1, N2 = pc.state_grid_sizes[1], pc.state_grid_sizes[2]
    for _ in range(100):
        s_test = np.array([rng.uniform(pc.state_grids[d][0], pc.state_grids[d][-1])
                           for d in range(3)])
        exact = a + np.dot(b, s_test)
        lo = [0, 0, 0]
        frac = [0.0, 0.0, 0.0]
        for d in range(3):
            g = pc.state_grids[d]
            idx = max(0, min(np.searchsorted(g, s_test[d]) - 1, len(g) - 2))
            lo[d] = idx
            frac[d] = max(0.0, min(1.0, (s_test[d] - g[idx]) / (g[idx + 1] - g[idx])))
        interp = 0.0
        for d0 in range(2):
            for d1 in range(2):
                for d2 in range(2):
                    w = ((1 - frac[0]) if d0 == 0 else frac[0]) * \
                        ((1 - frac[1]) if d1 == 0 else frac[1]) * \
                        ((1 - frac[2]) if d2 == 0 else frac[2])
                    j = (lo[0] + d0) * N1 * N2 + (lo[1] + d1) * N2 + (lo[2] + d2)
                    interp += w * c_fake[0, j, 0]
        max_err = max(max_err, abs(interp - exact))
    check("T3.3 trilinear exact for linear", max_err < 1e-12, f"err={max_err:.2e}")


# =====================================================================
# TIER 4 -- Integration Tests (one-period solver)
# =====================================================================

def run_tier_4(model, pc):
    tier_header(4, "Integration Tests (one-period solver)")

    from solver import (solve_terminal_age, solve_retirement_step,
                        solve_retirement_step_quad, solve_working_age_step,
                        solve_working_age_step_quad)
    sc = SolverConfig()
    Phi_0_state = np.ascontiguousarray(model.Phi_0_state)
    Phi_11_arr = np.ascontiguousarray(model.Phi_11)

    # T4.1 Terminal condition
    c_T, s_T, b_T, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors, pc.r_bill_grid,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=model.constrained)
    check("T4.1a terminal c finite", np.all(np.isfinite(c_T)))
    check("T4.1b terminal c positive", np.all(c_T > 0))
    check("T4.1c terminal alpha_s in [0,1]",
          np.all(s_T >= -1e-10) and np.all(s_T <= 1 + 1e-10))

    # T4.2 One retirement period comparison
    print("  (JIT compiling retirement solvers...)")
    psi = pc.survival_probs_2d[-2, :]
    pension = pc.pension_after_tax[-1, :]

    c_old, s_old, b_old, _, _ = solve_retirement_step(
        pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
        c_T, pension, pc.annuity_factors,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights, pc.r_bill_grid,
        model.gamma, psi, model.beta, model.b_bar,
        constrained=model.constrained, solver_config=sc)

    c_new, s_new, b_new, _, _ = solve_retirement_step_quad(
        pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
        c_T, pension, pc.annuity_factors, pc.r_bill_grid,
        pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
        pc.v_nodes, pc.v_weights, pc.M_v_nodes, pc.const_r, pc.A_r,
        Phi_0_state, Phi_11_arr,
        pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
        model.gamma, psi, model.beta, model.b_bar,
        constrained=model.constrained, solver_config=sc)

    mask = c_old > 0.001
    rel_c = np.abs(c_new[mask] - c_old[mask]) / np.maximum(c_old[mask], 1e-10)
    check("T4.2a retirement c max rel err < 10%", np.max(rel_c) < 0.10,
          f"max={np.max(rel_c):.4f}")
    check("T4.2b retirement c mean rel err < 5%", np.mean(rel_c) < 0.05,
          f"mean={np.mean(rel_c):.4f}")
    check("T4.2c quad c finite", np.all(np.isfinite(c_new)))
    check("T4.2d quad c positive", np.all(c_new > 0))

    # T4.3 One working-age period comparison
    print("  (JIT compiling working-age solvers...)")
    t_work = model.retire_age - model.start_age - 1
    log_det_next = pc.log_det_profile[min(t_work + 1, len(pc.log_det_profile) - 1)]

    c_old_w, _, _, _, _ = solve_working_age_step(
        pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
        c_T, log_det_next,
        pc.annuity_factors, model.rho, pc.eta_nodes, pc.eta_weights, pc.dz,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights, pc.r_bill_grid,
        pc.eps_nodes, pc.eps_weights,
        model.gamma, psi, model.beta, model.b_bar,
        constrained=model.constrained, solver_config=sc)

    c_new_w, _, _, _, _ = solve_working_age_step_quad(
        pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
        c_T, log_det_next,
        pc.annuity_factors, model.rho, pc.eta_nodes, pc.eta_weights, pc.dz,
        pc.r_bill_grid,
        pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
        pc.v_nodes, pc.v_weights, pc.M_v_nodes, pc.const_r, pc.A_r,
        Phi_0_state, Phi_11_arr,
        pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
        pc.eps_nodes, pc.eps_weights,
        model.gamma, psi, model.beta, model.b_bar,
        constrained=model.constrained, solver_config=sc)

    mask_w = c_old_w > 0.001
    rel_w = np.abs(c_new_w[mask_w] - c_old_w[mask_w]) / np.maximum(c_old_w[mask_w], 1e-10)
    check("T4.3a working c max rel err < 10%", np.max(rel_w) < 0.10,
          f"max={np.max(rel_w):.4f}")
    check("T4.3b working c mean rel err < 5%", np.mean(rel_w) < 0.05,
          f"mean={np.mean(rel_w):.4f}")
    check("T4.3c quad c finite", np.all(np.isfinite(c_new_w)))
    check("T4.3d quad c positive", np.all(c_new_w > 0))


# =====================================================================
# TIER 5 -- Full Lifecycle Tests
# =====================================================================

def run_tier_5(model, pc):
    tier_header(5, "Full Lifecycle Tests")

    from solver import run_lifecycle_solver

    C, S, B, diag = run_lifecycle_solver(
        model, pc, solver_config=SolverConfig(),
        use_state_quadrature=True, verbose=0)

    check("T5.1a no NaN in C", not np.any(np.isnan(C)))
    check("T5.1b no NaN in S", not np.any(np.isnan(S)))
    check("T5.1c no NaN in B", not np.any(np.isnan(B)))
    check("T5.1d no Inf in C", not np.any(np.isinf(C)))
    check("T5.1e C non-negative", np.all(C >= 0))

    # T5.2 Monotonicity in wealth
    n_age, n_z, N_state, n_w = C.shape
    total = n_age * n_z * N_state
    violations = 0
    worst = 0.0
    for t in range(n_age):
        for iz in range(n_z):
            for i_s in range(N_state):
                diffs = np.diff(C[t, iz, i_s, :])
                if np.any(diffs < -1e-8):
                    violations += 1
                    worst = max(worst, -diffs.min())
    frac = violations / total
    check("T5.2 wealth monotonicity", frac < 0.05,
          f"{violations}/{total} ({100*frac:.1f}%) violated, worst={worst:.2e}")

    # T5.3 Portfolio bounds
    check("T5.3a alpha_s >= 0", np.all(S >= -1e-6), f"min S={S.min():.6f}")
    check("T5.3b alpha_b >= 0", np.all(B >= -1e-6), f"min B={B.min():.6f}")
    check("T5.3c alpha_s+alpha_b <= 1", np.all(S + B <= 1 + 1e-6),
          f"max S+B={(S+B).max():.6f}")

    return C, S, B, diag


# =====================================================================
# TIER 6 -- Economic Sanity Tests
# =====================================================================

def run_tier_6(C, S, B, model, pc):
    tier_header(6, "Economic Sanity Tests")

    iz_med = C.shape[1] // 2
    is_med = C.shape[2] // 2
    iw_med = C.shape[3] // 2

    # T6.1 Stock share declines with age
    stock_young = S[5, iz_med, is_med, iw_med]
    stock_old = S[-5, iz_med, is_med, iw_med]
    check("T6.1 stock declines with age", stock_young > stock_old,
          f"young={stock_young:.3f}, old={stock_old:.3f}")

    # T6.3 c/W increases with age
    w_val = pc.wealth_grid[iw_med]
    if w_val > 0:
        cw_young = C[5, iz_med, is_med, iw_med] / w_val
        cw_old = C[-3, iz_med, is_med, iw_med] / w_val
        check("T6.3 c/W increases with age", cw_old > cw_young,
              f"young={cw_young:.4f}, old={cw_old:.4f}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    global n_pass, n_fail
    t0 = time.time()

    # Build once
    print("Building model and Precompute (5x5x5)...")
    t_build = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t_build:.1f}s")

    # TIER 1 -- only needs model, no pc rebuild
    run_tier_1(model)
    if n_fail > 0:
        print(f"\n  STOPPING: {n_fail} Tier 1 failure(s).")
        return

    # TIER 2 -- needs pc + nonsquare pc
    print("\nBuilding non-square Precompute (5x7x3)...")
    t_build = time.time()
    _, pc_ns = build_nonsquare()
    print(f"  Done in {time.time() - t_build:.1f}s")
    run_tier_2(model, pc, pc_ns)
    del pc_ns  # free memory
    if n_fail > 0:
        print(f"\n  STOPPING: {n_fail} failure(s) in Tiers 1-2.")
        return

    # TIER 3
    run_tier_3(model, pc)
    if n_fail > 0:
        print(f"\n  STOPPING: {n_fail} failure(s) in Tiers 1-3.")
        return

    # TIER 4
    run_tier_4(model, pc)
    if n_fail > 0:
        print(f"\n  STOPPING: {n_fail} failure(s) in Tiers 1-4.")
        return

    # TIER 5
    result = run_tier_5(model, pc)
    if result is not None and n_fail == 0:
        C, S, B, diag = result
        # TIER 6
        run_tier_6(C, S, B, model, pc)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS: {n_pass} passed, {n_fail} failed  ({elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

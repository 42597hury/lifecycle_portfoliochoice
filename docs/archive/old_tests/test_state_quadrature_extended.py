"""
Extended Tests: State Innovation Quadrature

Partitioned into independent test groups. Run one at a time:
    python test_state_quadrature_extended.py fast      # A,B,E — no solver, seconds
    python test_state_quadrature_extended.py foc_conv  # C1 — FOC K-convergence, ~1 min
    python test_state_quadrature_extended.py lifecycle  # D,G — 1 full solve with age output
    python test_state_quadrature_extended.py mc         # F3 — Monte Carlo cross-check, ~2 min
    python test_state_quadrature_extended.py policy_conv # C2 — policy K-convergence, ~15 min
    python test_state_quadrature_extended.py timing     # A4 — quad vs Markov at 7^3
    python test_state_quadrature_extended.py determinism # 2 full solves, compare bit-exact

Without arguments, prints this help.
"""

import sys
import numpy as np
import time
import json
import warnings
from pathlib import Path

from model import DiscretizationConfig, SolverConfig
from precompute import build_model, Precompute
from discretization import get_state_quadrature

warnings.filterwarnings("ignore")

_orig_print = print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    sys.stdout.flush()

SAVED_RUN = Path("saved_runs/constrained_grid5x5x5_nz11")

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


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


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
    bc, vc = load_var_config()
    model = build_model(bc, vc, verbose=False)
    dc = DiscretizationConfig(
        state_grid_sizes=(5, 5, 5), n_state_quad_nodes=3,
        n_wealth=50, n_savings=50, n_z=7,
        n_eps_nodes=3, n_eta_nodes=3, n_ret_nodes_1d=2,
    )
    pc = Precompute(model, dc, verbose=False)
    return model, pc


def summary():
    print(f"\n{'='*70}")
    print(f"  RESULTS: {n_pass} passed, {n_fail} failed")
    print(f"{'='*70}")


# =====================================================================
# "fast" — Numba, contiguity, boundary, stress (no solver)
# =====================================================================

def cmd_fast():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- A: Numba & arrays ---
    header("A — Numba Compilation & Array Checks")

    from solver import bracket_state_3d, _interp_z_wealth
    import numba

    g = np.array([0.0, 1.0, 2.0])
    try:
        bracket_state_3d(0.5, 0.5, 0.5, g, g, g)
        check("A1a bracket_state_3d nopython", True)
    except numba.core.errors.TypingError as e:
        check("A1a bracket_state_3d nopython", False, str(e)[:200])

    c_dummy = np.ones((7, 125, 50))
    try:
        _interp_z_wealth(c_dummy, 0, 1, 0.5, 5, 0.5, 1.0, 7, False, 1e-10)
        check("A1b _interp_z_wealth nopython", True)
    except numba.core.errors.TypingError as e:
        check("A1b _interp_z_wealth nopython", False, str(e)[:200])

    arrays_to_check = [
        'v_nodes', 'v_weights', 'M_v_nodes', 'const_r', 'A_r',
        'exp_ret_stock', 'exp_ret_bond', 'state_grid',
        'ret_nodes', 'ret_weights', 'wealth_grid',
    ]
    bad = [n for n in arrays_to_check if not getattr(pc, n).flags['C_CONTIGUOUS']]
    check("A3 all quadrature arrays C-contiguous", len(bad) == 0, f"non-contiguous: {bad}")

    for d in range(3):
        arr = pc.state_grids[d]
        check(f"A3c state_grids[{d}] contiguous", arr.flags['C_CONTIGUOUS'])

    for name in ['v_nodes', 'v_weights', 'M_v_nodes', 'const_r', 'A_r']:
        arr = getattr(pc, name)
        check(f"A3d {name} dtype float64", arr.dtype == np.float64, f"got {arr.dtype}")

    # --- B: Boundary hits ---
    header("B — Boundary Hit Analysis")

    Phi_0 = np.asarray(model.Phi_0_state, dtype=float)
    Phi_11 = np.asarray(model.Phi_11, dtype=float)

    n_hits = np.zeros(3)
    n_total = 0
    for i_s in range(pc.N_state):
        s_i = pc.state_grid[i_s]
        for k_v in range(pc.n_state_quad):
            s_next = Phi_0 + Phi_11 @ s_i + pc.v_nodes[k_v]
            for d in range(3):
                if s_next[d] < pc.state_grids[d][0] or s_next[d] > pc.state_grids[d][-1]:
                    n_hits[d] += 1
            n_total += 1

    any_hit = 0
    for i_s in range(pc.N_state):
        s_i = pc.state_grid[i_s]
        for k_v in range(pc.n_state_quad):
            s_next = Phi_0 + Phi_11 @ s_i + pc.v_nodes[k_v]
            for d in range(3):
                if s_next[d] < pc.state_grids[d][0] or s_next[d] > pc.state_grids[d][-1]:
                    any_hit += 1
                    break

    for d, name in enumerate(model.state_names):
        pct = 100 * n_hits[d] / n_total
        print(f"    {name}: {pct:.1f}% boundary hits")
        check(f"B3 {name} boundary < 25%", pct < 25.0, f"{pct:.1f}%")
    pct_any = 100 * any_hit / n_total
    print(f"    Any dimension: {pct_any:.1f}%")
    check("B3 overall boundary < 50%", pct_any < 50.0, f"{pct_any:.1f}%")

    # --- E: Stress tests ---
    header("E — Stress Tests (FOC at corners & tiny savings)")

    from solver import compute_foc_jac_retirement_quad, solve_terminal_age
    Phi_0_state = np.ascontiguousarray(model.Phi_0_state)
    Phi_11_arr = np.ascontiguousarray(model.Phi_11)

    print("  (JIT compiling terminal + retirement FOC...)")
    c_T, _, _, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors, pc.r_bill_grid,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=model.constrained)

    N1 = len(pc.state_grids[1])
    N2 = len(pc.state_grids[2])
    z_idx = pc.n_z // 2
    psi = pc.survival_probs_2d[-2, z_idx]
    c_next_slice = c_T[z_idx, :, :]

    for corner_name, i_s in [("min corner", 0), ("max corner", pc.N_state - 1)]:
        s_i = pc.state_grid[i_s]
        base_mu_r_i = pc.const_r + pc.A_r @ s_i
        R_bill = np.exp(pc.r_bill_grid[i_s])
        s_val = pc.wealth_grid[pc.n_w // 2] * 0.5

        foc_s, foc_b, J_ss, J_bb, J_sb, euler = compute_foc_jac_retirement_quad(
            0.3, 0.3, s_val, z_idx, i_s,
            pc.wealth_grid, c_next_slice,
            pc.pension_after_tax[-1, z_idx], pc.annuity_factors[i_s],
            pc.v_nodes, pc.v_weights, pc.M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11_arr, s_i,
            pc.state_grids[0], pc.state_grids[1], pc.state_grids[2], N1, N2,
            pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights, R_bill,
            model.gamma, psi, model.beta, model.b_bar,
        )
        all_finite = (np.isfinite(foc_s) and np.isfinite(foc_b) and
                      np.isfinite(J_ss) and np.isfinite(J_bb) and
                      np.isfinite(euler) and euler > 0)
        check(f"E2 {corner_name} FOC finite", all_finite,
              f"foc=({foc_s:.2e},{foc_b:.2e}) euler={euler:.2e}")

    # Tiny savings
    i_s_mid = pc.N_state // 2
    s_i = pc.state_grid[i_s_mid]
    base_mu_r_i = pc.const_r + pc.A_r @ s_i
    R_bill = np.exp(pc.r_bill_grid[i_s_mid])
    foc_s, foc_b, _, _, _, euler = compute_foc_jac_retirement_quad(
        0.3, 0.3, 1e-8, z_idx, i_s_mid,
        pc.wealth_grid, c_next_slice,
        pc.pension_after_tax[-1, z_idx], pc.annuity_factors[i_s_mid],
        pc.v_nodes, pc.v_weights, pc.M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11_arr, s_i,
        pc.state_grids[0], pc.state_grids[1], pc.state_grids[2], N1, N2,
        pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights, R_bill,
        model.gamma, psi, model.beta, model.b_bar,
    )
    check("E3 tiny savings no NaN",
          np.isfinite(foc_s) and np.isfinite(foc_b) and np.isfinite(euler),
          f"foc=({foc_s:.2e},{foc_b:.2e}) euler={euler:.2e}")

    summary()


# =====================================================================
# "foc_conv" — FOC-level K convergence (no solver, ~1 min)
# =====================================================================

def cmd_foc_conv():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    header("C1 — FOC-Level Quadrature Convergence")

    from solver import compute_foc_jac_retirement_quad, solve_terminal_age

    Phi_0_state = np.ascontiguousarray(model.Phi_0_state)
    Phi_11_arr = np.ascontiguousarray(model.Phi_11)
    N1 = len(pc.state_grids[1])
    N2 = len(pc.state_grids[2])

    print("  (JIT compiling terminal condition...)")
    c_T, _, _, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors, pc.r_bill_grid,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=model.constrained)

    z_idx = pc.n_z // 2
    c_next_slice = c_T[z_idx, :, :]

    test_points = []
    for i_s in [0, pc.N_state // 4, pc.N_state // 2, 3 * pc.N_state // 4, pc.N_state - 1]:
        s_i = pc.state_grid[i_s]
        test_points.append((
            i_s, s_i, np.exp(pc.r_bill_grid[i_s]),
            pc.survival_probs_2d[-2, z_idx],
            pc.pension_after_tax[-1, z_idx],
            pc.annuity_factors[i_s],
            c_next_slice, pc.wealth_grid[pc.n_w // 2] * 0.4, z_idx,
        ))

    M_mat = np.asarray(model.M, dtype=float)
    results = {}
    for K in [1, 2, 3, 4, 5]:
        v_nodes, v_weights = get_state_quadrature(model, n_nodes=K)
        M_v_nodes = v_nodes @ M_mat.T
        const_r = np.array(model.Phi_0_ret, dtype=float)
        A_r = np.array(model.Phi_21, dtype=float)

        focs_s, focs_b, eulers = [], [], []
        for (i_s, s_i, R_bill, psi, pension, annuity_f,
             c_slice, s_val, z_idx) in test_points:
            base_mu_r_i = const_r + A_r @ s_i
            foc_s, foc_b, _, _, _, euler = compute_foc_jac_retirement_quad(
                0.3, 0.3, s_val, z_idx, i_s,
                pc.wealth_grid, c_slice, pension, annuity_f,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11_arr, s_i,
                pc.state_grids[0], pc.state_grids[1], pc.state_grids[2], N1, N2,
                pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights, R_bill,
                model.gamma, psi, model.beta, model.b_bar,
            )
            focs_s.append(foc_s)
            focs_b.append(foc_b)
            eulers.append(euler)

        results[K] = {
            'foc_s': np.array(focs_s),
            'foc_b': np.array(focs_b),
            'euler': np.array(eulers),
        }
        print(f"  K={K} ({K**3:3d} nodes):")
        print(f"    euler = [{', '.join(f'{e:.6e}' for e in eulers)}]")
        print(f"    foc_s = [{', '.join(f'{f:.6e}' for f in focs_s)}]")
        print(f"    foc_b = [{', '.join(f'{f:.6e}' for f in focs_b)}]")

    print()
    for label, key in [("euler", "euler"), ("foc_s", "foc_s"), ("foc_b", "foc_b")]:
        diff_23 = np.max(np.abs(results[2][key] - results[3][key]))
        diff_34 = np.max(np.abs(results[3][key] - results[4][key]))
        diff_45 = np.max(np.abs(results[4][key] - results[5][key]))
        print(f"  {label} max|diff|: 2->3={diff_23:.2e}  3->4={diff_34:.2e}  4->5={diff_45:.2e}")
        check(f"C1 {label} converges (3->4 < 2->3)", diff_34 < diff_23,
              f"diff_23={diff_23:.2e}, diff_34={diff_34:.2e}")

    euler_scale = np.mean(np.abs(results[3]['euler']))
    if euler_scale > 1e-15:
        rel_35 = np.max(np.abs(results[3]['euler'] - results[5]['euler'])) / euler_scale
        print(f"  euler K=3 vs K=5 relative diff: {rel_35:.4e}")
        check("C1 euler K=3 vs K=5 < 1%", rel_35 < 0.01, f"rel diff = {rel_35:.4e}")

    summary()


# =====================================================================
# "lifecycle" — 1 full solve with verbose output, then diagnostics
# =====================================================================

def cmd_lifecycle():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    header("D/G — Full Lifecycle Solve + Diagnostics")

    from solver import run_lifecycle_solver

    print("  Solving (verbose=1, you will see per-age output)...\n")
    C, S, B, diag = run_lifecycle_solver(
        model, pc, solver_config=SolverConfig(),
        use_state_quadrature=True, verbose=1)

    # --- Diagnostics ---
    header("Diagnostics")

    total_calls = diag['total_calls']
    total_fail = diag['total_newton_failures']
    fail_rate = total_fail / max(total_calls, 1)
    worst_foc = diag['worst_foc_resid']
    total_mono = diag['total_mono_violations']

    print(f"  Newton calls: {total_calls}  failures: {total_fail} ({100*fail_rate:.2f}%)")
    print(f"  Worst FOC residual: {worst_foc:.2e}")
    print(f"  Monotonicity violations: {total_mono}")
    check("D3a Newton failure rate < 1%", fail_rate < 0.01, f"{100*fail_rate:.2f}%")
    check("D3b worst FOC residual < 1e-3", worst_foc < 1e-3, f"{worst_foc:.2e}")

    # Finite/positive
    check("T5.1a no NaN in C", not np.any(np.isnan(C)))
    check("T5.1b C non-negative", np.all(C >= 0))
    check("T5.3a alpha_s >= 0", np.all(S >= -1e-6), f"min S={S.min():.6f}")
    check("T5.3b alpha_b >= 0", np.all(B >= -1e-6), f"min B={B.min():.6f}")
    check("T5.3c alpha_s+alpha_b <= 1", np.all(S + B <= 1 + 1e-6), f"max S+B={(S+B).max():.6f}")

    # Regression watchlist
    iz = C.shape[1] // 2
    iw = C.shape[3] // 2
    t_30 = min(8, C.shape[0] - 1)

    mean_stock = np.mean(S[t_30, iz, :, iw])
    mean_bond = np.mean(B[t_30, iz, :, iw])
    print(f"\n  Mean stock share ~age 30: {mean_stock:.4f}")
    print(f"  Mean bond share ~age 30:  {mean_bond:.4f}")
    check("G1a stock share age 30 in [0.1, 0.9]", 0.1 < mean_stock < 0.9, f"{mean_stock:.4f}")
    check("G1b bond share age 30 in [0.0, 0.8]", 0.0 <= mean_bond < 0.8, f"{mean_bond:.4f}")

    mono_rate = total_mono / max(total_calls, 1)
    check("G1e mono violations < 5%", mono_rate < 0.05, f"{100*mono_rate:.2f}%")

    # State sensitivity
    N0, N1g, N2 = pc.state_grid_sizes
    print(f"\n  State sensitivity at age ~{model.start_age + t_30}, median z & wealth:")
    for d, name in enumerate(model.state_names):
        if d == 0:
            i_lo = 0 * N1g * N2 + N1g // 2 * N2 + N2 // 2
            i_hi = (N0-1) * N1g * N2 + N1g // 2 * N2 + N2 // 2
        elif d == 1:
            i_lo = N0 // 2 * N1g * N2 + 0 * N2 + N2 // 2
            i_hi = N0 // 2 * N1g * N2 + (N1g-1) * N2 + N2 // 2
        else:
            i_lo = N0 // 2 * N1g * N2 + N1g // 2 * N2 + 0
            i_hi = N0 // 2 * N1g * N2 + N1g // 2 * N2 + (N2-1)
        s_lo = S[t_30, iz, i_lo, iw]
        s_hi = S[t_30, iz, i_hi, iw]
        print(f"    {name}: stock(lo)={s_lo:.4f}  stock(hi)={s_hi:.4f}  diff={s_hi-s_lo:+.4f}")

    summary()


# =====================================================================
# "mc" — Monte Carlo cross-check of retirement FOC (~2 min)
# =====================================================================

def cmd_mc():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    header("F3 — Monte Carlo vs GH Quadrature")

    from solver import (compute_foc_jac_retirement_quad, solve_terminal_age,
                        bracket_state_3d, fast_interp_1d_with_slope)

    Phi_0_state = np.ascontiguousarray(model.Phi_0_state)
    Phi_11_arr = np.ascontiguousarray(model.Phi_11)
    M_mat = np.asarray(model.M, dtype=float)
    N1 = len(pc.state_grids[1])
    N2 = len(pc.state_grids[2])

    print("  (JIT compiling terminal + FOC...)")
    c_T, _, _, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors, pc.r_bill_grid,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=model.constrained)

    z_idx = pc.n_z // 2
    i_s = pc.N_state // 2
    s_i = pc.state_grid[i_s]
    base_mu_r_i = pc.const_r + pc.A_r @ s_i
    R_bill = np.exp(pc.r_bill_grid[i_s])
    psi = pc.survival_probs_2d[-2, z_idx]
    c_next_slice = c_T[z_idx, :, :]
    pension = pc.pension_after_tax[-1, z_idx]
    annuity_f = pc.annuity_factors[i_s]
    s_val = pc.wealth_grid[pc.n_w // 2] * 0.4
    alpha_s, alpha_b = 0.3, 0.3

    # GH quadrature FOC
    foc_s_gh, foc_b_gh, _, _, _, euler_gh = compute_foc_jac_retirement_quad(
        alpha_s, alpha_b, s_val, z_idx, i_s,
        pc.wealth_grid, c_next_slice, pension, annuity_f,
        pc.v_nodes, pc.v_weights, pc.M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11_arr, s_i,
        pc.state_grids[0], pc.state_grids[1], pc.state_grids[2], N1, N2,
        pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights, R_bill,
        model.gamma, psi, model.beta, model.b_bar,
    )
    print(f"  GH (K=3): euler={euler_gh:.6e}  foc_s={foc_s_gh:.6e}  foc_b={foc_b_gh:.6e}")

    # Monte Carlo
    rng = np.random.default_rng(42)
    L_ss = np.linalg.cholesky(np.asarray(model.Sigma_ss, dtype=float))
    L_rr_cond = np.linalg.cholesky(np.asarray(model.Sigma_r_cond, dtype=float))

    N_mc = 30000
    a_bill = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi
    gamma = model.gamma
    b_bar = model.b_bar

    foc_s_mc = 0.0
    foc_b_mc = 0.0
    euler_mc = 0.0

    print(f"  Running {N_mc} Monte Carlo draws...")
    t_mc = time.time()
    for n in range(N_mc):
        if n % 10000 == 0 and n > 0:
            print(f"    {n}/{N_mc} draws done...")

        v = L_ss @ rng.standard_normal(3)
        s_next = Phi_0_state + Phi_11_arr @ s_i + v
        mu_r_k = pc.const_r + pc.A_r @ s_i + M_mat @ v
        eps_r = L_rr_cond @ rng.standard_normal(2)
        log_ret = mu_r_k + eps_r

        R_s = R_bill * np.exp(log_ret[0])
        R_b = R_bill * np.exp(log_ret[1])
        R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

        w_inv = max(s_val * R_p, 1e-10)
        x_next = w_inv + pension

        lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
            s_next[0], s_next[1], s_next[2],
            pc.state_grids[0], pc.state_grids[1], pc.state_grids[2])

        c_interp = 0.0
        for d0 in range(2):
            for d1 in range(2):
                for d2 in range(2):
                    w = ((1-f0) if d0==0 else f0) * ((1-f1) if d1==0 else f1) * ((1-f2) if d2==0 else f2)
                    j = (lo0+d0)*N1*N2 + (lo1+d1)*N2 + (lo2+d2)
                    c_val, _ = fast_interp_1d_with_slope(x_next, pc.wealth_grid, c_next_slice[j, :])
                    c_interp += w * c_val
        c_interp = max(c_interp, 1e-10)

        mu_alive = c_interp ** (-gamma)
        w_A = w_inv / annuity_f
        mu_bequest = b_bar * w_A ** (-gamma) / annuity_f
        mu_comb = psi * mu_alive + prob_death * mu_bequest

        Rex_s = R_s - R_bill
        Rex_b = R_b - R_bill
        foc_s_mc += mu_comb * Rex_s / N_mc
        foc_b_mc += mu_comb * Rex_b / N_mc
        euler_mc += mu_comb * R_p / N_mc

    print(f"  MC done in {time.time() - t_mc:.1f}s")
    print(f"  MC (N={N_mc}): euler={euler_mc:.6e}  foc_s={foc_s_mc:.6e}  foc_b={foc_b_mc:.6e}")

    if abs(euler_gh) > 1e-15:
        rel_euler = abs(euler_gh - euler_mc) / abs(euler_gh)
        print(f"  Euler rel diff: {rel_euler:.4f}")
        check("F3 MC euler agrees with GH (<10%)", rel_euler < 0.10, f"rel diff = {rel_euler:.4f}")

    scale = max(abs(euler_gh), 1e-10)
    foc_s_diff = abs(foc_s_gh - foc_s_mc) / scale
    foc_b_diff = abs(foc_b_gh - foc_b_mc) / scale
    print(f"  FOC_s normalized diff: {foc_s_diff:.4f}")
    print(f"  FOC_b normalized diff: {foc_b_diff:.4f}")
    check("F3 MC foc_s agrees with GH (<15%)", foc_s_diff < 0.15, f"diff = {foc_s_diff:.4f}")
    check("F3 MC foc_b agrees with GH (<15%)", foc_b_diff < 0.15, f"diff = {foc_b_diff:.4f}")

    summary()


# =====================================================================
# "policy_conv" — Policy K convergence via partial solve (~15 min)
# =====================================================================

def cmd_policy_conv():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    header("C2 — Policy-Level K Convergence (5 periods per K)")

    from solver import (solve_terminal_age, solve_retirement_step_quad,
                        solve_working_age_step_quad)

    sc = SolverConfig()
    Phi_0_state = np.ascontiguousarray(model.Phi_0_state)
    Phi_11_arr = np.ascontiguousarray(model.Phi_11)
    M_mat = np.asarray(model.M, dtype=float)
    t_work = model.retire_age - model.start_age
    iz = pc.n_z // 2
    i_s_mid = pc.N_state // 2
    iw = pc.n_w // 2

    print("  (JIT compiling terminal condition...)")
    c_T, _, _, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors, pc.r_bill_grid,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=model.constrained)

    results = {}
    for K in [2, 3, 4]:
        v_nodes, v_weights = get_state_quadrature(model, n_nodes=K)
        M_v_nodes = v_nodes @ M_mat.T
        const_r = np.array(model.Phi_0_ret, dtype=float)
        A_r = np.array(model.Phi_21, dtype=float)

        print(f"\n  K={K} ({K**3} nodes):")

        # Retirement period 1
        psi = pc.survival_probs_2d[-2, :]
        pension = pc.pension_after_tax[-1, :]
        print(f"    Solving retirement period 1...")
        t1 = time.time()
        c_ret1, s_ret1, b_ret1, _, _ = solve_retirement_step_quad(
            pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
            c_T, pension, pc.annuity_factors, pc.r_bill_grid,
            pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
            v_nodes, v_weights, M_v_nodes, const_r, A_r,
            Phi_0_state, Phi_11_arr,
            pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
            model.gamma, psi, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
        print(f"      done ({time.time()-t1:.1f}s)  c_med={c_ret1[iz,i_s_mid,iw]:.6f}  s_med={s_ret1[iz,i_s_mid,iw]:.6f}")

        # Retirement period 2
        psi2 = pc.survival_probs_2d[-3, :]
        pension2 = pc.pension_after_tax[-2, :]
        print(f"    Solving retirement period 2...")
        t1 = time.time()
        c_ret2, s_ret2, b_ret2, _, _ = solve_retirement_step_quad(
            pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
            c_ret1, pension2, pc.annuity_factors, pc.r_bill_grid,
            pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
            v_nodes, v_weights, M_v_nodes, const_r, A_r,
            Phi_0_state, Phi_11_arr,
            pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
            model.gamma, psi2, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
        print(f"      done ({time.time()-t1:.1f}s)  c_med={c_ret2[iz,i_s_mid,iw]:.6f}  s_med={s_ret2[iz,i_s_mid,iw]:.6f}")

        # Working-age period 1
        psi_w1 = pc.survival_probs_2d[t_work - 1, :]
        log_det_w1 = pc.log_det_profile[min(t_work, len(pc.log_det_profile) - 1)]
        print(f"    Solving working-age period 1...")
        t1 = time.time()
        c_work1, s_work1, b_work1, _, _ = solve_working_age_step_quad(
            pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
            c_ret2, log_det_w1,
            pc.annuity_factors, model.rho, pc.eta_nodes, pc.eta_weights, pc.dz,
            pc.r_bill_grid,
            pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
            v_nodes, v_weights, M_v_nodes, const_r, A_r,
            Phi_0_state, Phi_11_arr,
            pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
            pc.eps_nodes, pc.eps_weights,
            model.gamma, psi_w1, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
        print(f"      done ({time.time()-t1:.1f}s)  c_med={c_work1[iz,i_s_mid,iw]:.6f}  s_med={s_work1[iz,i_s_mid,iw]:.6f}")

        # Working-age period 2
        psi_w2 = pc.survival_probs_2d[t_work - 2, :]
        log_det_w2 = pc.log_det_profile[min(t_work - 1, len(pc.log_det_profile) - 1)]
        print(f"    Solving working-age period 2...")
        t1 = time.time()
        c_work2, s_work2, b_work2, _, _ = solve_working_age_step_quad(
            pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
            c_work1, log_det_w2,
            pc.annuity_factors, model.rho, pc.eta_nodes, pc.eta_weights, pc.dz,
            pc.r_bill_grid,
            pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
            v_nodes, v_weights, M_v_nodes, const_r, A_r,
            Phi_0_state, Phi_11_arr,
            pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
            pc.eps_nodes, pc.eps_weights,
            model.gamma, psi_w2, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
        print(f"      done ({time.time()-t1:.1f}s)  c_med={c_work2[iz,i_s_mid,iw]:.6f}  s_med={s_work2[iz,i_s_mid,iw]:.6f}")

        c_med = [c_ret1[iz, i_s_mid, iw], c_ret2[iz, i_s_mid, iw],
                 c_work1[iz, i_s_mid, iw], c_work2[iz, i_s_mid, iw]]
        s_med = [s_ret1[iz, i_s_mid, iw], s_ret2[iz, i_s_mid, iw],
                 s_work1[iz, i_s_mid, iw], s_work2[iz, i_s_mid, iw]]
        results[K] = {'c': np.array(c_med), 's': np.array(s_med)}

    # Convergence
    print()
    cdiff_23 = np.max(np.abs(results[2]['c'] - results[3]['c']))
    cdiff_34 = np.max(np.abs(results[3]['c'] - results[4]['c']))
    print(f"  C policy diff: 2->3={cdiff_23:.6f}  3->4={cdiff_34:.6f}")
    check("C2a consumption converges", cdiff_34 < cdiff_23,
          f"cdiff_23={cdiff_23:.6f}, cdiff_34={cdiff_34:.6f}")

    sdiff_23 = np.max(np.abs(results[2]['s'] - results[3]['s']))
    sdiff_34 = np.max(np.abs(results[3]['s'] - results[4]['s']))
    print(f"  S policy diff: 2->3={sdiff_23:.6f}  3->4={sdiff_34:.6f}")
    check("C2b stock share converges", sdiff_34 < sdiff_23,
          f"sdiff_23={sdiff_23:.6f}, sdiff_34={sdiff_34:.6f}")

    mean_c3 = np.mean(results[3]['c'])
    if mean_c3 > 0:
        rel_c = cdiff_34 / mean_c3
        print(f"  C rel diff K=3->4: {rel_c:.4e}")
        check("C2c consumption K=3 vs K=4 < 2%", rel_c < 0.02, f"rel diff = {rel_c:.4e}")

    summary()


# =====================================================================
# "timing" — Quad vs Markov at 7^3
# =====================================================================

def cmd_timing():
    print("Building model and Precompute (7x7x7)...")
    t0 = time.time()
    bc, vc = load_var_config()
    model = build_model(bc, vc, verbose=False)
    dc = DiscretizationConfig(
        state_grid_sizes=(7, 7, 7), n_state_quad_nodes=3,
        n_wealth=50, n_savings=50, n_z=7,
        n_eps_nodes=3, n_eta_nodes=3, n_ret_nodes_1d=2,
    )
    pc = Precompute(model, dc, verbose=False)
    print(f"  Done in {time.time() - t0:.1f}s")

    header("A4 — Per-Period Timing: Quad vs Markov at 7^3")

    from solver import (solve_terminal_age, solve_retirement_step,
                        solve_retirement_step_quad)

    sc = SolverConfig()
    Phi_0_state = np.ascontiguousarray(model.Phi_0_state)
    Phi_11_arr = np.ascontiguousarray(model.Phi_11)

    c_T, _, _, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors, pc.r_bill_grid,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=model.constrained)

    psi = pc.survival_probs_2d[-2, :]
    pension = pc.pension_after_tax[-1, :]

    print("  JIT warmup (Markov)...")
    solve_retirement_step(
        pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
        c_T, pension, pc.annuity_factors,
        pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights, pc.r_bill_grid,
        model.gamma, psi, model.beta, model.b_bar,
        constrained=model.constrained, solver_config=sc)

    print("  JIT warmup (Quad)...")
    solve_retirement_step_quad(
        pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
        c_T, pension, pc.annuity_factors, pc.r_bill_grid,
        pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
        pc.v_nodes, pc.v_weights, pc.M_v_nodes, pc.const_r, pc.A_r,
        Phi_0_state, Phi_11_arr,
        pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
        model.gamma, psi, model.beta, model.b_bar,
        constrained=model.constrained, solver_config=sc)

    n_reps = 3
    print(f"  Timing ({n_reps} reps each)...")

    t0 = time.perf_counter()
    for _ in range(n_reps):
        solve_retirement_step(
            pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
            c_T, pension, pc.annuity_factors,
            pc.Pi_state, pc.mu_r, pc.ret_nodes, pc.ret_weights, pc.r_bill_grid,
            model.gamma, psi, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
    t_markov = (time.perf_counter() - t0) / n_reps

    t0 = time.perf_counter()
    for _ in range(n_reps):
        solve_retirement_step_quad(
            pc.wealth_grid, pc.s_grid, pc.z_grid, pc.N_state,
            c_T, pension, pc.annuity_factors, pc.r_bill_grid,
            pc.state_grid, pc.state_grids[0], pc.state_grids[1], pc.state_grids[2],
            pc.v_nodes, pc.v_weights, pc.M_v_nodes, pc.const_r, pc.A_r,
            Phi_0_state, Phi_11_arr,
            pc.exp_ret_stock, pc.exp_ret_bond, pc.ret_weights,
            model.gamma, psi, model.beta, model.b_bar,
            constrained=model.constrained, solver_config=sc)
    t_quad = (time.perf_counter() - t0) / n_reps

    ratio = t_quad / max(t_markov, 1e-6)
    print(f"  Markov: {t_markov:.2f}s  Quadrature: {t_quad:.2f}s  Ratio: {ratio:.2f}x")
    check("A4 quad not >3x slower than Markov at 7^3", ratio < 3.0, f"ratio = {ratio:.2f}")

    summary()


# =====================================================================
# "determinism" — 2 full solves, compare bit-exact
# =====================================================================

def cmd_determinism():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    header("D1 — Determinism (2 full solves)")

    from solver import run_lifecycle_solver

    print("  Solve 1 (verbose=1)...\n")
    C1, S1, B1, _ = run_lifecycle_solver(
        model, pc, solver_config=SolverConfig(),
        use_state_quadrature=True, verbose=1)

    print("\n  Solve 2 (verbose=1)...\n")
    C2, S2, B2, _ = run_lifecycle_solver(
        model, pc, solver_config=SolverConfig(),
        use_state_quadrature=True, verbose=1)

    check("D1a C deterministic", np.array_equal(C1, C2),
          f"max diff = {np.max(np.abs(C1-C2)):.2e}" if not np.array_equal(C1, C2) else "")
    check("D1b S deterministic", np.array_equal(S1, S2),
          f"max diff = {np.max(np.abs(S1-S2)):.2e}" if not np.array_equal(S1, S2) else "")
    check("D1c B deterministic", np.array_equal(B1, B2),
          f"max diff = {np.max(np.abs(B1-B2)):.2e}" if not np.array_equal(B1, B2) else "")

    summary()


# =====================================================================
# MAIN — dispatch by argument
# =====================================================================

# =====================================================================
# "immunize" — Duration matching / safe asset test
# =====================================================================

def cmd_immunize():
    print("Building model and Precompute (5x5x5)...")
    t0 = time.time()
    model, pc = build_once()
    print(f"  Done in {time.time() - t0:.1f}s")

    header("H — Duration Matching / Immunization Test")

    from model import annuity_factor as af_func

    # --- Analytical immunizing bond allocation ---

    # The bequest utility is: b_bar * (W / A(y_nom))^{1-gamma} / (1-gamma)
    # where A(y_nom) = sum_{k=1}^{b_bar} (1+y)^{-k}
    #
    # When y_nom changes by Δy:
    #   - Bond return effect on wealth: ΔW/W ≈ alpha_b * M[xb, y_nom] * Δy
    #   - Annuity factor change: ΔA/A ≈ -D_mod * Δy (D_mod = modified duration)
    #
    # The "immunized" portfolio makes W/A insensitive to y_nom shocks:
    #   d(W/A)/dy = 0  =>  alpha_b * M[xb, y_nom] = -D_mod
    #   alpha_b_immunize = D_mod / |M[xb, y_nom]|

    y_nom_idx = model.annuity_yield_index_in_state  # index in state vector
    y_nom_mean = float(model.z_bar_state[y_nom_idx])
    b_bar = model.b_bar

    # Modified duration of A(y_nom, b_bar)
    # A = sum_{k=1}^{b_bar} (1+y)^{-k}
    # dA/dy = -sum_{k=1}^{b_bar} k * (1+y)^{-(k+1)}
    # D_mod = -(dA/dy) / A
    y = y_nom_mean
    A_val = sum((1 + y)**(-k) for k in range(1, b_bar + 1))
    dA_dy = -sum(k * (1 + y)**(-(k+1)) for k in range(1, b_bar + 1))
    D_mod = -dA_dy / A_val

    # Verify against the closed-form annuity factor
    A_check = af_func(y, b_bar)
    print(f"  y_nom (unconditional mean): {y:.4f} ({y*100:.2f}%)")
    print(f"  b_bar: {b_bar}")
    print(f"  A(y_nom, b_bar) = {A_val:.4f}  (check: {float(A_check):.4f})")
    print(f"  Modified duration: {D_mod:.2f} years")

    # M[xb, y_nom]
    xb_idx = 1  # xb is second return variable
    M_xb_ynom = float(model.M[xb_idx, y_nom_idx])
    print(f"  M[xb, y_nom]: {M_xb_ynom:.2f}")

    alpha_b_immunize = D_mod / abs(M_xb_ynom)
    print(f"  Immunizing bond share: {alpha_b_immunize:.4f}")
    print()

    # --- Also compute the stock hedging ratio ---
    # M[xr, dp] captures stock return predictability from D/P
    dp_idx = 2  # dp is third state variable
    xr_idx = 0  # xr is first return variable
    M_xr_dp = float(model.M[xr_idx, dp_idx])
    print(f"  M[xr, dp]: {M_xr_dp:.4f}")
    print(f"  (Stock returns respond to dp innovations through this channel)")
    print()

    # --- Run solver and compare ---
    from solver import run_lifecycle_solver

    print("  Running lifecycle solver (verbose=1)...\n")
    C, S, B, diag = run_lifecycle_solver(
        model, pc, solver_config=SolverConfig(),
        use_state_quadrature=True, verbose=1)

    # --- Compare bond allocation to immunizing target ---
    header("Comparison: Solver vs Immunizing Allocation")

    iz_mid = pc.n_z // 2
    N0, N1, N2 = pc.state_grid_sizes
    # Median state = middle of each dimension
    is_mid = (N0 // 2) * N1 * N2 + (N1 // 2) * N2 + (N2 // 2)

    # High wealth to avoid binding constraints
    iw_high = min(int(pc.n_w * 0.8), pc.n_w - 1)
    iw_mid = pc.n_w // 2

    print(f"  Immunizing bond share (analytical): {alpha_b_immunize:.4f}")
    print(f"  Evaluating at median z, median state, wealth index {iw_high}")
    print()

    # Near-terminal retirement ages (where bequest dominates)
    n_age = C.shape[0]
    ages_to_check = []
    for offset in [1, 3, 5, 10, 20, 40]:
        t = n_age - 1 - offset
        if t >= 0:
            age = model.start_age + t
            ages_to_check.append((t, age, offset))

    print(f"  {'Age':>5s}  {'t_offset':>8s}  {'bond(hi_w)':>10s}  {'bond(mid_w)':>11s}  "
          f"{'stock(hi_w)':>11s}  {'bill(hi_w)':>10s}  {'diff_imm':>9s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*10}  {'-'*9}")

    for t, age, offset in ages_to_check:
        b_hi = B[t, iz_mid, is_mid, iw_high]
        b_mid = B[t, iz_mid, is_mid, iw_mid]
        s_hi = S[t, iz_mid, is_mid, iw_high]
        bill_hi = 1.0 - s_hi - b_hi
        diff = b_hi - alpha_b_immunize
        print(f"  {age:5d}  {offset:8d}  {b_hi:10.4f}  {b_mid:11.4f}  "
              f"{s_hi:11.4f}  {bill_hi:10.4f}  {diff:+9.4f}")

    # --- Sensitivity: bond allocation across y_nom grid ---
    print()
    print(f"  Bond share across y_nom grid (near-terminal, high wealth):")
    t_near_term = n_age - 2  # 1 period before terminal
    age_near = model.start_age + t_near_term

    print(f"  {'y_nom_idx':>9s}  {'y_nom':>8s}  {'A(y_nom)':>8s}  "
          f"{'bond':>6s}  {'stock':>6s}  {'bill':>6s}")
    print(f"  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")

    for i1 in range(N1):
        # State index: middle of dim0, vary dim1 (y_nom), middle of dim2
        i_s = (N0 // 2) * N1 * N2 + i1 * N2 + (N2 // 2)
        y_val = pc.state_grids[y_nom_idx][i1]
        A_at = float(af_func(y_val, b_bar))
        b_val = B[t_near_term, iz_mid, i_s, iw_high]
        s_val = S[t_near_term, iz_mid, i_s, iw_high]
        bill_val = 1.0 - s_val - b_val
        print(f"  {i1:9d}  {y_val:8.4f}  {A_at:8.3f}  "
              f"{b_val:6.3f}  {s_val:6.3f}  {bill_val:6.3f}")

    # --- The key question: does bond share vary with y_nom in the
    #     direction implied by duration matching? ---
    # Higher y_nom → shorter duration → need less bond hedging
    b_low_ynom = B[t_near_term, iz_mid,
                   (N0//2)*N1*N2 + 0*N2 + N2//2, iw_high]
    b_high_ynom = B[t_near_term, iz_mid,
                    (N0//2)*N1*N2 + (N1-1)*N2 + N2//2, iw_high]
    print()
    print(f"  Bond at low y_nom:  {b_low_ynom:.4f}")
    print(f"  Bond at high y_nom: {b_high_ynom:.4f}")

    # At low y_nom, duration is longer, so more bond hedging needed
    # => bond share should be higher at low y_nom (or at least not lower)
    # But this interacts with risk premia, so we just report
    print(f"  Difference: {b_high_ynom - b_low_ynom:+.4f}")
    print(f"  (Negative = more bonds at low y_nom = duration-matching behavior)")

    summary()


COMMANDS = {
    'fast': cmd_fast,
    'foc_conv': cmd_foc_conv,
    'lifecycle': cmd_lifecycle,
    'mc': cmd_mc,
    'policy_conv': cmd_policy_conv,
    'timing': cmd_timing,
    'determinism': cmd_determinism,
    'immunize': cmd_immunize,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python test_state_quadrature_extended.py <command>\n")
        print("Commands (run one at a time):")
        print("  fast          Numba, contiguity, boundary, stress    ~30s")
        print("  foc_conv      FOC-level K convergence (K=1..5)       ~1 min")
        print("  lifecycle     1 full solve + diagnostics + watchlist  ~20 min")
        print("  mc            Monte Carlo cross-check of FOC         ~2 min")
        print("  policy_conv   Policy K convergence (5 periods x3)    ~15 min")
        print("  timing        Quad vs Markov speed at 7^3            ~10 min")
        print("  determinism   2 full solves, bit-exact compare       ~40 min")
        print("  immunize      Duration matching / safe asset test     ~20 min")
        sys.exit(0)

    cmd = sys.argv[1]
    print(f"Running: {cmd}")
    t_total = time.time()
    COMMANDS[cmd]()
    print(f"\nTotal time: {time.time() - t_total:.1f}s")

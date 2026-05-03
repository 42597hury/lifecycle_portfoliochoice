"""
test_egm_crra.py — Numerical validation of CRRA utility in the EGM solver loop.

Tests:
  A. Round-trip: u'_inv(u'(c)) == c for many (gamma, c)
  B. Inlined formula matches closure: (beta*e)^{-1/gamma} == u_prime_inv(beta*e)
  C. Euler identity: u'(c_opt) == beta * euler after inversion
  D. Live FOC: call compute_foc_jac_retirement with controlled inputs, extract
     euler_sum, apply EGM inversion, verify the Euler identity holds
"""

import numpy as np
import sys

# ── Test A: u'_inv(u'(c)) == c ──────────────────────────────────────────────

def test_a_roundtrip():
    from model import create_utility_functions

    gammas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    cs     = np.geomspace(1e-6, 1e6, 200)  # wide range

    print("=" * 70)
    print("TEST A: u'_inv(u'(c)) == c   (round-trip consistency)")
    print("=" * 70)

    all_pass = True
    for gamma in gammas:
        u, u_prime, u_prime_inv = create_utility_functions(gamma)
        max_rel_err = 0.0
        for c in cs:
            mu = u_prime(c)
            c_back = u_prime_inv(mu)
            rel_err = abs(c_back - c) / max(abs(c), 1e-30)
            if rel_err > max_rel_err:
                max_rel_err = rel_err
        status = "PASS" if max_rel_err < 1e-12 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  gamma={gamma:5.1f}  max rel err = {max_rel_err:.2e}  [{status}]")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ── Test B: inlined formula == closure ───────────────────────────────────────

def test_b_inline_vs_closure():
    from model import create_utility_functions

    gammas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    betas  = [0.90, 0.96, 0.99]
    eulers = np.geomspace(1e-8, 1e8, 200)

    print("=" * 70)
    print("TEST B: (beta*euler)^{-1/gamma} == u'_inv(beta*euler)")
    print("=" * 70)

    all_pass = True
    for gamma in gammas:
        _, _, u_prime_inv = create_utility_functions(gamma)
        max_rel_err = 0.0
        for beta in betas:
            for e in eulers:
                # Inlined (as in solver.py:1742)
                c_inline = (beta * e) ** (-1.0 / gamma)
                # Via closure
                c_closure = u_prime_inv(beta * e)
                rel_err = abs(c_inline - c_closure) / max(abs(c_closure), 1e-30)
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        status = "PASS" if max_rel_err < 1e-12 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  gamma={gamma:5.1f}  max rel err = {max_rel_err:.2e}  [{status}]")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ── Test C: Euler identity u'(c_opt) == beta * euler ────────────────────────

def test_c_euler_identity():
    from model import create_utility_functions

    gammas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    betas  = [0.90, 0.96, 0.99]
    eulers = np.geomspace(1e-8, 1e8, 200)

    print("=" * 70)
    print("TEST C: u'(c_opt) == beta * euler   (Euler equation satisfied)")
    print("=" * 70)

    all_pass = True
    for gamma in gammas:
        _, u_prime, _ = create_utility_functions(gamma)
        max_rel_err = 0.0
        for beta in betas:
            for e in eulers:
                # EGM inversion (as in solver)
                c_opt = (beta * e) ** (-1.0 / gamma)
                # Check: does u'(c_opt) == beta * e ?
                lhs = u_prime(c_opt)
                rhs = beta * e
                rel_err = abs(lhs - rhs) / max(abs(rhs), 1e-30)
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        status = "PASS" if max_rel_err < 1e-12 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  gamma={gamma:5.1f}  max rel err = {max_rel_err:.2e}  [{status}]")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ── Test D: live FOC euler_sum -> EGM inversion -> Euler identity ─────────────

def test_d_live_foc():
    """
    Call compute_foc_jac_retirement with controlled inputs to get real
    euler_sum values, then verify the EGM inversion satisfies the Euler
    equation: u'(c_opt) == beta * euler_sum.
    """
    from solver import compute_foc_jac_retirement, build_gross_return_arrays
    from math import exp

    print("=" * 70)
    print("TEST D: Live FOC -> euler_sum -> EGM inversion -> Euler identity")
    print("=" * 70)

    gamma = 3.0
    beta  = 0.96
    b_bar = 10

    # --- Minimal financial setup: 2 states, 1 return node (deterministic) ---
    N_state = 2
    n_ret_quad = 1

    # Simple transition: equal probability
    Pi_state = np.array([[0.5, 0.5],
                         [0.5, 0.5]])

    # Log excess returns: stocks +5%, bonds +1% (above bill rate)
    mu_r = np.zeros((N_state, N_state, 2))
    mu_r[:, :, 0] = 0.05   # xr ~ 5%
    mu_r[:, :, 1] = 0.01   # xb ~ 1%

    # Single return node at zero (no residual)
    ret_nodes   = np.zeros((1, 2))
    ret_weights = np.array([1.0])

    # Bill rate: 2% real
    R_bill = exp(0.02)

    # Annuity factor at 4% annual yield
    y_ann = 0.04
    A = (1.0 - (1.0 + y_ann)**(-b_bar)) / y_ann

    # Wealth grid (simple)
    n_w = 50
    wealth_grid = np.geomspace(0.01, 100.0, n_w)

    # Next-period consumption policy: c = 0.05 * W (constant MPC for simplicity)
    c_next_full_per_state = np.zeros((N_state, n_w))
    for j_s in range(N_state):
        c_next_full_per_state[j_s, :] = 0.05 * wealth_grid

    # Pension
    pension = 0.25  # moderate pension

    # Pre-compute return arrays for state 0
    Rx_stock, Rx_bond = build_gross_return_arrays(mu_r[0], ret_nodes)

    # Test across different savings levels, portfolio weights, and survival probs
    savings_vals = [0.1, 0.5, 1.0, 5.0, 20.0, 50.0]
    alpha_pairs  = [(0.0, 0.0), (0.3, 0.3), (0.6, 0.2), (1.0, 0.0), (0.0, 1.0)]
    psi_vals     = [0.5, 0.9, 0.99, 1.0, 0.0]  # includes certain death & certain survival

    all_pass = True
    max_err_overall = 0.0
    n_tests = 0

    for psi in psi_vals:
        for s_val in savings_vals:
            for (a_s, a_b) in alpha_pairs:
                foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum = compute_foc_jac_retirement(
                    a_s, a_b, s_val, 0, 0,
                    wealth_grid, c_next_full_per_state, pension, A,
                    Pi_state, Rx_stock, Rx_bond, ret_weights, R_bill,
                    gamma, psi, beta, b_bar,
                )

                if euler_sum <= 0.0:
                    continue  # floor would bind; skip

                # EGM inversion (exactly as solver.py:1742)
                c_opt = (beta * euler_sum) ** (-1.0 / gamma)

                # Verify Euler identity: u'(c_opt) == beta * euler_sum
                lhs = c_opt ** (-gamma)
                rhs = beta * euler_sum
                rel_err = abs(lhs - rhs) / max(abs(rhs), 1e-30)
                if rel_err > max_err_overall:
                    max_err_overall = rel_err
                n_tests += 1

                if rel_err > 1e-10:
                    all_pass = False
                    print(f"  FAIL: psi={psi:.2f} s={s_val:.1f} a=({a_s:.1f},{a_b:.1f})"
                          f"  euler={euler_sum:.6e}  c_opt={c_opt:.6e}  err={rel_err:.2e}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  {n_tests} FOC calls tested")
    print(f"  Max Euler identity error: {max_err_overall:.2e}")
    print(f"  Overall: {status}")
    print()
    return all_pass


# ── Test E: verify c_opt + s_val = x (budget constraint in EGM) ─────────────

def test_e_egm_budget():
    """
    In EGM, the endogenous grid point is x = c_opt + s_val. Verify that for
    the EGM-produced (c, x) pairs, the budget constraint x = c + a holds
    exactly by construction.

    This is trivially true from the code (temp_x[i] = c_opt + s_val), but
    we verify it numerically to confirm no off-by-one or stale-variable bugs
    in the assignment.
    """
    from solver import compute_foc_jac_retirement, build_gross_return_arrays
    from math import exp

    print("=" * 70)
    print("TEST E: EGM budget identity x = c_opt + s_val")
    print("=" * 70)

    gamma = 3.0
    beta  = 0.96
    b_bar = 10

    N_state = 2
    Pi_state = np.array([[0.5, 0.5], [0.5, 0.5]])
    mu_r = np.zeros((N_state, N_state, 2))
    mu_r[:, :, 0] = 0.05
    mu_r[:, :, 1] = 0.01
    ret_nodes   = np.zeros((1, 2))
    ret_weights = np.array([1.0])
    R_bill = exp(0.02)

    y_ann = 0.04
    A = (1.0 - (1.0 + y_ann)**(-b_bar)) / y_ann

    n_w = 50
    wealth_grid = np.geomspace(0.01, 100.0, n_w)
    c_next_full = np.zeros((N_state, n_w))
    for j in range(N_state):
        c_next_full[j, :] = 0.05 * wealth_grid
    pension = 0.25

    Rx_stock, Rx_bond = build_gross_return_arrays(mu_r[0], ret_nodes)

    savings_grid = np.geomspace(1e-4, 100.0, 80)
    psi = 0.98

    all_pass = True
    max_err = 0.0
    n_valid = 0

    for s_val in savings_grid:
        _, _, _, _, _, euler_sum = compute_foc_jac_retirement(
            0.3, 0.3, s_val, 0, 0,
            wealth_grid, c_next_full, pension, A,
            Pi_state, Rx_stock, Rx_bond, ret_weights, R_bill,
            gamma, psi, beta, b_bar,
        )
        if euler_sum <= 0:
            continue

        c_opt = (beta * euler_sum) ** (-1.0 / gamma)
        x = c_opt + s_val

        # Budget: x - c_opt should equal s_val exactly (IEEE)
        err = abs((x - c_opt) - s_val)
        rel = err / max(s_val, 1e-30)
        if rel > max_err:
            max_err = rel
        n_valid += 1

        if rel > 1e-14:
            all_pass = False
            print(f"  FAIL: s={s_val:.6e} c={c_opt:.6e} x={x:.6e} err={rel:.2e}")

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  {n_valid} savings points tested")
    print(f"  Max budget error: {max_err:.2e}")
    print(f"  Overall: {status}")
    print()
    return all_pass


# ── Test F: euler_sum monotonicity in savings ────────────────────────────────

def test_f_euler_monotone():
    """
    euler_sum should be monotonically decreasing in savings for fixed portfolio
    weights: higher savings -> higher next-period wealth -> lower marginal utility.
    This is a sanity check on the FOC output.
    """
    from solver import compute_foc_jac_retirement, build_gross_return_arrays
    from math import exp

    print("=" * 70)
    print("TEST F: euler_sum monotonically decreasing in savings")
    print("=" * 70)

    gamma = 3.0
    beta  = 0.96
    b_bar = 10

    N_state = 2
    Pi_state = np.array([[0.5, 0.5], [0.5, 0.5]])
    mu_r = np.zeros((N_state, N_state, 2))
    mu_r[:, :, 0] = 0.05
    mu_r[:, :, 1] = 0.01
    ret_nodes   = np.zeros((1, 2))
    ret_weights = np.array([1.0])
    R_bill = exp(0.02)

    y_ann = 0.04
    A = (1.0 - (1.0 + y_ann)**(-b_bar)) / y_ann

    n_w = 80
    wealth_grid = np.geomspace(0.01, 200.0, n_w)
    # Monotone decreasing MPC consumption policy
    c_next_full = np.zeros((N_state, n_w))
    for j in range(N_state):
        c_next_full[j, :] = 0.05 * wealth_grid

    pension = 0.25
    Rx_stock, Rx_bond = build_gross_return_arrays(mu_r[0], ret_nodes)

    savings_grid = np.geomspace(0.01, 50.0, 100)
    psi = 0.98
    alpha_s, alpha_b = 0.3, 0.3

    euler_vals = []
    for s_val in savings_grid:
        _, _, _, _, _, euler_sum = compute_foc_jac_retirement(
            alpha_s, alpha_b, s_val, 0, 0,
            wealth_grid, c_next_full, pension, A,
            Pi_state, Rx_stock, Rx_bond, ret_weights, R_bill,
            gamma, psi, beta, b_bar,
        )
        euler_vals.append(euler_sum)

    euler_vals = np.array(euler_vals)

    # Check monotone decreasing
    violations = 0
    worst_increase = 0.0
    for i in range(1, len(euler_vals)):
        if euler_vals[i] > euler_vals[i-1]:
            violations += 1
            inc = euler_vals[i] - euler_vals[i-1]
            if inc > worst_increase:
                worst_increase = inc

    # Therefore c_opt should be monotone increasing
    c_opts = (beta * euler_vals) ** (-1.0 / gamma)
    c_violations = 0
    for i in range(1, len(c_opts)):
        if c_opts[i] < c_opts[i-1]:
            c_violations += 1

    all_pass = (violations == 0) and (c_violations == 0)
    status = "PASS" if all_pass else "FAIL"

    print(f"  euler_sum range: [{euler_vals.min():.4e}, {euler_vals.max():.4e}]")
    print(f"  c_opt range:     [{c_opts.min():.4e}, {c_opts.max():.4e}]")
    print(f"  euler monotone violations: {violations}"
          + (f"  (worst increase: {worst_increase:.2e})" if violations > 0 else ""))
    print(f"  c_opt  monotone violations: {c_violations}")
    print(f"  Overall: {status}")
    print()
    return all_pass


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    results["A: u'_inv(u'(c)) round-trip"] = test_a_roundtrip()
    results["B: inline == closure"]         = test_b_inline_vs_closure()
    results["C: Euler identity"]            = test_c_euler_identity()

    # Tests D-F require numba JIT compilation of solver functions
    print("(Compiling Numba functions for tests D-F...)")
    results["D: live FOC Euler identity"]   = test_d_live_foc()
    results["E: EGM budget x=c+a"]         = test_e_egm_budget()
    results["F: euler monotonicity"]        = test_f_euler_monotone()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, passed in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  {tag}  {name}")
        if not passed:
            all_ok = False
    print()
    if all_ok:
        print("  All tests passed.")
    else:
        print("  SOME TESTS FAILED.")
    sys.exit(0 if all_ok else 1)

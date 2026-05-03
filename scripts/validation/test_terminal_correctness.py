"""
Correctness tests for the terminal period optimization.

Tests:
  1. CRRA homogeneity: c/W constant across wealth grid
  2. z-independence: terminal policy identical across income states
  3. Portfolio FOC residual near zero at solution
  4. Brute-force grid search agrees with Newton solution
  5. Moment positivity and finiteness
  6. Consumption in (0, W) for all W > 0
  7. Gradient-free finite-difference check on FOC
"""
import numpy as np
from lifecycle.model import SolverConfig, DiscretizationConfig
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, Precompute
from lifecycle.solver import (
    _build_terminal_quad_returns,
    compute_terminal_portfolio_foc_jac,
    solve_portfolio_2d_terminal_constrained_njit,
    solve_portfolio_unconstrained_terminal_njit,
    solve_terminal_age,
    _terminal_prepare_scenarios,
)


def build_test_pc(state_grid_sizes=(5, 5, 5), n_z=7, gamma=3.0):
    var_config = build_nominal_system1_var_config_hardcoded()
    pz = 0.176; mu_eta1 = -0.524
    base_config = {
        'gamma': gamma, 'beta': 0.96, 'b_bar': 10,
        'start_age': 22, 'retire_age': 67, 'terminal_age': 99,
        'b0': -6.142, 'b1': 0.3040, 'b2': -0.051, 'b3': 0.002586,
        'rho': 0.991, 'pz': pz,
        'mu_eta1': mu_eta1, 'sigma_eta1': 0.113,
        'mu_eta2': -(pz / (1.0 - pz)) * mu_eta1, 'sigma_eta2': 0.046,
        'pe': 0.044, 'mu_eps1': 0.134, 'sigma_eps1': 0.762,
        'mu_eps2': 0.0, 'sigma_eps2': 0.055, 'constrained': True,
    }
    model = build_model(base_config, var_config, verbose=False)
    disc = DiscretizationConfig(
        state_grid_sizes=state_grid_sizes,
        n_wealth=10, n_savings=10, n_z=n_z,
    )
    pc = Precompute(model, disc)
    return model, pc


def eval_moment(alpha_s, alpha_b, v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights, gamma):
    """Compute E[R_port^{1-gamma}] in pure numpy for independent verification."""
    sw = np.asarray(v_weights)[:, None] * np.asarray(ret_weights)[None, :]
    R_bill = np.asarray(Rx_bill)
    R_stock = R_bill * np.asarray(Rx_stock_mult)
    R_bond = R_bill * np.asarray(Rx_bond_mult)
    a_bill = 1.0 - alpha_s - alpha_b
    R_port = alpha_s * R_stock + alpha_b * R_bond + a_bill * R_bill
    if np.any((sw > 0) & (R_port <= 0)):
        return np.inf
    return float(np.sum(sw * R_port ** (1.0 - gamma)))


def eval_foc_fd(alpha_s, alpha_b, v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
                ret_weights, gamma, h=1e-7):
    """Finite-difference FOC: d/d(alpha_k) E[R_port^{1-gamma}]."""
    def f(a_s, a_b):
        return eval_moment(a_s, a_b, v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights, gamma)
    foc_s_fd = (f(alpha_s + h, alpha_b) - f(alpha_s - h, alpha_b)) / (2 * h)
    foc_b_fd = (f(alpha_s, alpha_b + h) - f(alpha_s, alpha_b - h)) / (2 * h)
    return foc_s_fd, foc_b_fd


def test_foc_matches_finite_difference(model, pc, n_test=5):
    """Test 7: Analytic FOC matches finite-difference gradient."""
    print("\n=== Test 7: FOC vs finite-difference gradient ===")
    max_err = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)
    for i_s in test_states:
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)
        # Test at a few interior points
        for a_s, a_b in [(0.3, 0.3), (0.5, 0.2), (0.1, 0.6)]:
            foc_s, foc_b, _, _, _, _ = compute_terminal_portfolio_foc_jac(
                a_s, a_b, pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
                pc.ret_weights, model.gamma)
            foc_s_fd, foc_b_fd = eval_foc_fd(
                a_s, a_b, pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
                pc.ret_weights, model.gamma)
            # The analytic FOC drops the (1-gamma) factor, so scale fd accordingly
            scale = 1.0 - model.gamma  # negative for gamma > 1
            err_s = abs(foc_s - foc_s_fd / scale)
            err_b = abs(foc_b - foc_b_fd / scale)
            rel_s = err_s / max(abs(foc_s), 1e-15)
            rel_b = err_b / max(abs(foc_b), 1e-15)
            max_err = max(max_err, rel_s, rel_b)
    status = "PASS" if max_err < 1e-5 else "FAIL"
    print(f"  Max relative error between analytic and FD FOC: {max_err:.2e}  [{status}]")
    return max_err < 1e-5


def test_brute_force_grid_search(model, pc, n_grid=201, n_test=5):
    """Test 4: Brute-force grid search agrees with Newton solution.

    The objective is max E[R_port^{1-gamma} / (1-gamma)].
    For gamma > 1, (1-gamma) < 0, so this is equivalent to
    MINIMIZING E[R_port^{1-gamma}] (the "moment").
    """
    print("\n=== Test 4: Brute-force grid vs Newton ===")
    alphas = np.linspace(0.0, 1.0, n_grid)
    max_moment_err = 0.0
    max_alloc_err = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)

    for i_s in test_states:
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)

        # Newton solution
        opt_s, opt_b, moment_newton, ec, resid = solve_portfolio_2d_terminal_constrained_njit(
            pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma)

        # Grid search over simplex: MINIMIZE E[R_port^{1-gamma}] for gamma > 1
        best_moment = np.inf
        best_s, best_b = 0.0, 0.0
        for a_s in alphas:
            for a_b in alphas:
                if a_s + a_b > 1.0 + 1e-12:
                    continue
                m = eval_moment(a_s, a_b, pc.v_weights, Rx_bill, Rx_stock_mult,
                                Rx_bond_mult, pc.ret_weights, model.gamma)
                if np.isfinite(m) and m < best_moment:
                    best_moment = m
                    best_s, best_b = a_s, a_b

        moment_newton_check = eval_moment(opt_s, opt_b, pc.v_weights, Rx_bill,
                                          Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma)
        moment_err = abs(moment_newton_check - best_moment) / max(abs(best_moment), 1e-15)
        alloc_err = max(abs(opt_s - best_s), abs(opt_b - best_b))
        max_moment_err = max(max_moment_err, moment_err)
        max_alloc_err = max(max_alloc_err, alloc_err)
        print(f"  i_s={i_s:3d}  Newton=({opt_s:.4f},{opt_b:.4f}) moment={moment_newton_check:.8f}"
              f"  Grid=({best_s:.4f},{best_b:.4f}) moment={best_moment:.8f}"
              f"  moment_err={moment_err:.2e}  alloc_err={alloc_err:.4f}")

    # Grid is coarse (step=0.005), so allocation can differ by up to ~0.005
    moment_ok = max_moment_err < 1e-4
    alloc_ok = max_alloc_err < 0.01  # within one grid step
    status = "PASS" if (moment_ok and alloc_ok) else "FAIL"
    print(f"  Max moment relative error: {max_moment_err:.2e}  Max alloc error: {max_alloc_err:.4f}  [{status}]")
    return moment_ok and alloc_ok


def test_crra_homogeneity(model, pc):
    """Test 1: c/W should be constant across the wealth grid."""
    print("\n=== Test 1: CRRA homogeneity (c/W constant across wealth) ===")
    sc = SolverConfig()
    c_T, a_s_T, a_b_T, diag = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors,
        pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.v_weights,
        pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=True, solver_config=sc)

    max_cv = 0.0
    # Check c/W ratio constancy for each (z, i_s) pair
    for iz in range(pc.n_z):
        for i_s in range(pc.N_state):
            c_vec = c_T[iz, i_s, :]
            w_vec = pc.wealth_grid
            # Skip tiny wealth where min_consumption floor kicks in
            mask = w_vec > 0.01
            if mask.sum() < 3:
                continue
            ratios = c_vec[mask] / w_vec[mask]
            cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 0.0
            max_cv = max(max_cv, cv)

    status = "PASS" if max_cv < 1e-6 else "FAIL"
    print(f"  Max CV of c/W ratio across wealth grid: {max_cv:.2e}  [{status}]")
    return max_cv < 1e-6


def test_z_independence(model, pc):
    """Test 2: Terminal policy identical across income states z."""
    print("\n=== Test 2: z-independence (policy same across z for each i_s) ===")
    sc = SolverConfig()
    c_T, a_s_T, a_b_T, diag = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors,
        pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.v_weights,
        pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=True, solver_config=sc)

    max_c_spread = 0.0
    max_s_spread = 0.0
    max_b_spread = 0.0
    for i_s in range(pc.N_state):
        for iw in range(len(pc.wealth_grid)):
            c_vals = c_T[:, i_s, iw]
            s_vals = a_s_T[:, i_s, iw]
            b_vals = a_b_T[:, i_s, iw]
            max_c_spread = max(max_c_spread, np.ptp(c_vals))
            max_s_spread = max(max_s_spread, np.ptp(s_vals))
            max_b_spread = max(max_b_spread, np.ptp(b_vals))

    status = "PASS" if max(max_c_spread, max_s_spread, max_b_spread) < 1e-12 else "FAIL"
    print(f"  Max c spread across z: {max_c_spread:.2e}")
    print(f"  Max alpha_s spread:    {max_s_spread:.2e}")
    print(f"  Max alpha_b spread:    {max_b_spread:.2e}")
    print(f"  [{status}]")
    return max(max_c_spread, max_s_spread, max_b_spread) < 1e-12


def test_foc_residual_at_solution(model, pc, n_test=10):
    """Test 3: FOC residual near zero at the Newton solution."""
    print("\n=== Test 3: FOC residual at solution ===")
    max_resid = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)
    for i_s in test_states:
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)
        opt_s, opt_b, moment, ec, resid = solve_portfolio_2d_terminal_constrained_njit(
            pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma)
        foc_s, foc_b, _, _, _, e = compute_terminal_portfolio_foc_jac(
            opt_s, opt_b, pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
            pc.ret_weights, model.gamma)
        scale = max(abs(e), 1.0)
        # At corners/edges, FOC may not be zero but KKT conditions must hold
        if ec == 7:  # interior
            foc_norm = np.hypot(foc_s, foc_b) / scale
            max_resid = max(max_resid, foc_norm)
        elif ec == 1:  # all bills: foc_s <= 0, foc_b <= 0
            violation = max(foc_s / scale, foc_b / scale, 0.0)
            max_resid = max(max_resid, violation)
        elif ec == 2:  # all stocks: foc_s >= 0, foc_b <= foc_s
            violation = max(-foc_s / scale, (foc_b - foc_s) / scale, 0.0)
            max_resid = max(max_resid, violation)
        elif ec == 3:  # all bonds: foc_b >= 0, foc_s <= foc_b
            violation = max(-foc_b / scale, (foc_s - foc_b) / scale, 0.0)
            max_resid = max(max_resid, violation)
        elif ec == 4:  # stock+bill edge: foc_s ~ 0, foc_b <= 0
            violation = max(abs(foc_s) / scale - 1e-5, foc_b / scale, 0.0)
            max_resid = max(max_resid, max(0.0, violation))
        elif ec == 5:  # bond+bill edge: foc_b ~ 0, foc_s <= 0
            violation = max(abs(foc_b) / scale - 1e-5, foc_s / scale, 0.0)
            max_resid = max(max_resid, max(0.0, violation))
        elif ec == 6:  # stock+bond edge: foc_s ~ foc_b, both >= 0
            violation = max(abs(foc_s - foc_b) / scale - 1e-5, -foc_s / scale, 0.0)
            max_resid = max(max_resid, max(0.0, violation))

    status = "PASS" if max_resid < 1e-5 else "FAIL"
    print(f"  Max KKT violation across {len(test_states)} states: {max_resid:.2e}  [{status}]")
    return max_resid < 1e-5


def test_moment_positivity(model, pc):
    """Test 5: E[R_port^{1-gamma}] is finite and the right sign."""
    print("\n=== Test 5: Moment positivity and finiteness ===")
    n_finite = 0
    n_correct_sign = 0
    for i_s in range(pc.N_state):
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)
        opt_s, opt_b, moment, ec, resid = solve_portfolio_2d_terminal_constrained_njit(
            pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma)
        if np.isfinite(moment):
            n_finite += 1
        # For gamma > 1: R^{1-gamma} = R^{negative} > 0 always (since R > 0)
        # So moment = E[R^{1-gamma}] > 0
        if moment > 0:
            n_correct_sign += 1

    status = "PASS" if (n_finite == pc.N_state and n_correct_sign == pc.N_state) else "FAIL"
    print(f"  Finite moments: {n_finite}/{pc.N_state}")
    print(f"  Positive moments (gamma={model.gamma}>1 => R^{{1-gamma}}>0): {n_correct_sign}/{pc.N_state}")
    print(f"  [{status}]")
    return n_finite == pc.N_state and n_correct_sign == pc.N_state


def test_consumption_bounds(model, pc):
    """Test 6: 0 < c < W for all W > 0."""
    print("\n=== Test 6: Consumption in (0, W) ===")
    sc = SolverConfig()
    c_T, _, _, _ = solve_terminal_age(
        pc.wealth_grid, pc.annuity_factors,
        pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.v_weights,
        pc.ret_nodes, pc.ret_weights,
        model.gamma, model.beta, model.b_bar, pc.N_state, pc.n_z,
        constrained=True, solver_config=sc)

    w = pc.wealth_grid
    mask = w > 1e-4  # skip near-zero wealth
    all_positive = np.all(c_T[:, :, mask] > 0)
    all_below_w = np.all(c_T[:, :, mask] < w[None, None, mask])
    c_over_w = c_T[:, :, mask] / w[None, None, mask]
    min_ratio = float(np.min(c_over_w))
    max_ratio = float(np.max(c_over_w))

    status = "PASS" if (all_positive and all_below_w) else "FAIL"
    print(f"  All c > 0: {all_positive}")
    print(f"  All c < W: {all_below_w}")
    print(f"  c/W range: [{min_ratio:.6f}, {max_ratio:.6f}]")
    print(f"  [{status}]")
    return all_positive and all_below_w


def test_terminal_vs_retirement_returns(model, pc, n_test=5):
    """Test 8: Terminal return construction matches retirement FOC return construction.

    The retirement solver builds returns inline:
        base_mu_r_i = const_r + A_r @ s_i
        mu_r_k = base_mu_r_i + M_v_nodes[k_v]
        R_bill = exp(mu_r_k[0]) * exp(ret_nodes[k_r, 0])
        R_s    = R_bill * exp(mu_r_k[1]) * exp(ret_nodes[k_r, 1])
        R_b    = R_bill * exp(mu_r_k[2]) * exp(ret_nodes[k_r, 2])

    The terminal solver precomputes in _build_terminal_quad_returns:
        Rx_bill[k_v, k_r]       = exp(mu_r_k[0] + ret_nodes[k_r, 0])
        Rx_stock_mult[k_v, k_r] = exp(mu_r_k[1] + ret_nodes[k_r, 1])
        Rx_bond_mult[k_v, k_r]  = exp(mu_r_k[2] + ret_nodes[k_r, 2])
    then in the FOC:
        R_bill = Rx_bill[k_v, k_r]
        R_s    = R_bill * Rx_stock_mult[k_v, k_r]
        R_b    = R_bill * Rx_bond_mult[k_v, k_r]

    These must be bit-identical.
    """
    print("\n=== Test 8: Terminal vs retirement return construction ===")
    max_err = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)
    for i_s in test_states:
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)

        # Reconstruct using the retirement solver's inline approach
        base_mu_r_i = pc.const_r + pc.A_r @ pc.state_grid[i_s]
        for k_v in range(len(pc.v_weights)):
            mu_r_k = base_mu_r_i + pc.M_v_nodes[k_v]
            exp_mu_bill = np.exp(mu_r_k[0])
            exp_mu_s = np.exp(mu_r_k[1])
            exp_mu_b = np.exp(mu_r_k[2])
            for k_r in range(len(pc.ret_weights)):
                # Retirement approach
                R_bill_ret = exp_mu_bill * pc.exp_ret_bill[k_r]
                R_s_ret = R_bill_ret * exp_mu_s * pc.exp_ret_stock[k_r]
                R_b_ret = R_bill_ret * exp_mu_b * pc.exp_ret_bond[k_r]

                # Terminal approach
                R_bill_term = Rx_bill[k_v, k_r]
                R_s_term = R_bill_term * Rx_stock_mult[k_v, k_r]
                R_b_term = R_bill_term * Rx_bond_mult[k_v, k_r]

                err_bill = abs(R_bill_ret - R_bill_term) / max(abs(R_bill_ret), 1e-15)
                err_s = abs(R_s_ret - R_s_term) / max(abs(R_s_ret), 1e-15)
                err_b = abs(R_b_ret - R_b_term) / max(abs(R_b_ret), 1e-15)
                max_err = max(max_err, err_bill, err_s, err_b)

    status = "PASS" if max_err < 1e-12 else "FAIL"
    print(f"  Max relative error between terminal and retirement returns: {max_err:.2e}  [{status}]")
    return max_err < 1e-12


def test_quadrature_mean_and_covariance(model, pc, n_test=5):
    """Test 9: Quadrature nodes reproduce the correct conditional return moments.

    For each state i_s, the terminal quadrature should satisfy:
      E[log R_bill]  = Phi_0_ret[0] + Phi_21[0,:] @ s_i   (unconditional on v^s)
      E[log R_stock] = Phi_0_ret[0] + Phi_0_ret[1] + (Phi_21[0,:] + Phi_21[1,:]) @ s_i
      Var/Cov of log returns matches full Omega_rr (not Sigma_r_cond alone)

    The total variance of log returns integrates over BOTH v^s and eps^r:
      Var(r) = Var_v(E[r|v]) + E_v(Var(r|v))
             = M @ Sigma_ss @ M' + Sigma_r_cond
             = Sigma_rs @ inv(Sigma_ss) @ Sigma_ss @ inv(Sigma_ss)' @ Sigma_sr + Sigma_r_cond
             = Sigma_rs @ inv(Sigma_ss) @ Sigma_sr + Sigma_rr - Sigma_rs @ inv(Sigma_ss) @ Sigma_sr
             = Sigma_rr
    """
    print("\n=== Test 9: Quadrature mean and covariance of log returns ===")
    max_mean_err = 0.0
    max_cov_err = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)

    for i_s in test_states:
        s_i = pc.state_grid[i_s]
        target_mean = model.Phi_0_ret + model.Phi_21 @ s_i  # E[r_{t+1} | s_t = s_i]

        # Compute quadrature mean and covariance of log returns [rtb, xr, xb]
        base_mu_r = pc.const_r + pc.A_r @ s_i
        n_v = len(pc.v_weights)
        n_r = len(pc.ret_weights)

        # Collect all log-return scenarios and weights
        log_returns = np.empty((n_v * n_r, 3))
        weights = np.empty(n_v * n_r)
        idx = 0
        for k_v in range(n_v):
            mu_r_k = base_mu_r + pc.M_v_nodes[k_v]
            for k_r in range(n_r):
                log_returns[idx, 0] = mu_r_k[0] + pc.ret_nodes[k_r, 0]  # log R_bill
                log_returns[idx, 1] = mu_r_k[1] + pc.ret_nodes[k_r, 1]  # log(R_s/R_bill) = xr
                log_returns[idx, 2] = mu_r_k[2] + pc.ret_nodes[k_r, 2]  # log(R_b/R_bill) = xb
                weights[idx] = pc.v_weights[k_v] * pc.ret_weights[k_r]
                idx += 1

        # Weighted mean of [rtb, xr, xb]
        quad_mean = weights @ log_returns
        mean_err = np.max(np.abs(quad_mean - target_mean))
        max_mean_err = max(max_mean_err, mean_err)

        # Weighted covariance
        centered = log_returns - quad_mean[None, :]
        quad_cov = (centered * weights[:, None]).T @ centered

        # Target: full Omega_rr (not Sigma_r_cond) — because we integrate over v^s too
        Omega_rr = model.Sigma_rr  # = Omega[ret_idx, ret_idx]
        cov_err = np.max(np.abs(quad_cov - Omega_rr))
        max_cov_err = max(max_cov_err, cov_err)

    mean_status = "PASS" if max_mean_err < 1e-12 else "FAIL"
    cov_status = "PASS" if max_cov_err < 1e-4 else "FAIL"  # quadrature covariance is approximate
    print(f"  Max mean error (log returns): {max_mean_err:.2e}  [{mean_status}]")
    print(f"  Max cov error (vs Omega_rr):  {max_cov_err:.2e}  [{cov_status}]")
    both_pass = max_mean_err < 1e-12 and max_cov_err < 1e-4
    return both_pass


def test_return_correlation_structure(model, pc, n_test=5):
    """Test 10: Correlation between bill, stock, and bond returns is preserved.

    The terminal solver must capture that stock/bond returns are CORRELATED
    with bill returns (since R_stock = R_bill * exp(xr) and rtb, xr, xb are
    correlated via Omega). This test checks that the quadrature-implied
    correlation matrix of GROSS returns (R_bill, R_stock, R_bond) is reasonable.
    """
    print("\n=== Test 10: Return correlation structure ===")
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)

    for i_s in test_states:
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)

        # Build flat scenario arrays
        sw = pc.v_weights[:, None] * pc.ret_weights[None, :]  # (n_v, n_r)
        R_bill = Rx_bill.ravel()
        R_stock = (Rx_bill * Rx_stock_mult).ravel()
        R_bond = (Rx_bill * Rx_bond_mult).ravel()
        w = sw.ravel()

        # Weighted correlation of log returns
        log_bill = np.log(R_bill)
        log_stock = np.log(R_stock)
        log_bond = np.log(R_bond)
        logs = np.stack([log_bill, log_stock, log_bond], axis=1)
        wmean = w @ logs
        centered = logs - wmean[None, :]
        wcov = (centered * w[:, None]).T @ centered
        wstd = np.sqrt(np.diag(wcov))
        wcorr = wcov / np.outer(wstd, wstd)

        if i_s == test_states[0]:
            print(f"  Sample state i_s={i_s}, log-return correlation matrix:")
            print(f"               R_bill   R_stock   R_bond")
            for row_i, label in enumerate(["  R_bill ", "  R_stock", "  R_bond "]):
                print(f"  {label}  {wcorr[row_i, 0]:+.4f}   {wcorr[row_i, 1]:+.4f}   {wcorr[row_i, 2]:+.4f}")

        # Target correlation from full Omega
        Omega_full_ret = np.zeros((3, 3))
        # log R_bill = rtb, log R_stock = rtb + xr, log R_bond = rtb + xb
        # Cov matrix of [rtb, rtb+xr, rtb+xb]:
        Orr = model.Sigma_rr  # cov of [rtb, xr, xb]
        # Transform: [rtb, rtb+xr, rtb+xb] = T @ [rtb, xr, xb]
        T = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=float)
        Omega_full_ret = T @ Orr @ T.T
        target_std = np.sqrt(np.diag(Omega_full_ret))
        target_corr = Omega_full_ret / np.outer(target_std, target_std)

        if i_s == test_states[0]:
            print(f"  Target correlation (from Omega_rr via T=[rtb,rtb+xr,rtb+xb]):")
            print(f"               R_bill   R_stock   R_bond")
            for row_i, label in enumerate(["  R_bill ", "  R_stock", "  R_bond "]):
                print(f"  {label}  {target_corr[row_i, 0]:+.4f}   {target_corr[row_i, 1]:+.4f}   {target_corr[row_i, 2]:+.4f}")

    # Correlation should be close (GH quadrature is approximate for higher moments)
    corr_err = np.max(np.abs(wcorr - target_corr))
    status = "PASS" if corr_err < 0.05 else "FAIL"
    print(f"  Max correlation error (last state): {corr_err:.4f}  [{status}]")
    return corr_err < 0.05


def test_state_return_cross_covariance(model, pc, n_test=5):
    """Test 11: Quadrature reproduces the state-return cross-covariance Sigma_sr.

    The terminal solver integrates over v^s (state innovation, via v_nodes/v_weights)
    and eps^r (return residual, via ret_nodes/ret_weights). The return at scenario
    (k_v, k_r) has log-return innovation:

        e_r = M @ v_nodes[k_v] + ret_nodes[k_r]

    The cross-covariance between state innovation and return innovation is:
        Cov(v^s, e_r) = E[v^s * (M @ v^s + eps^r)']
                      = Sigma_ss @ M'  (since eps^r is independent of v^s)
                      = Sigma_sr       (by definition of M = Sigma_rs @ inv(Sigma_ss))

    This tests that the factored two-layer quadrature preserves the correlation
    between state innovations and return innovations.
    """
    print("\n=== Test 11: State-return cross-covariance (Sigma_sr) ===")

    n_v = len(pc.v_weights)
    n_r = len(pc.ret_weights)
    n_state = model.n_state
    n_ret = model.n_ret

    # Compute quadrature cross-covariance Cov(v^s, e_r)
    # v^s nodes: pc.v_nodes[k_v]  (n_v, n_state)
    # e_r at (k_v, k_r): M_v_nodes[k_v] + ret_nodes[k_r]  (n_ret,)
    # Means are zero by construction (GH nodes centered at 0)

    cross_cov = np.zeros((n_state, n_ret))
    for k_v in range(n_v):
        for k_r in range(n_r):
            w = pc.v_weights[k_v] * pc.ret_weights[k_r]
            v_s = pc.v_nodes[k_v]                             # (n_state,)
            e_r = pc.M_v_nodes[k_v] + pc.ret_nodes[k_r]      # (n_ret,)
            cross_cov += w * np.outer(v_s, e_r)

    # Target: Sigma_sr = Sigma_ss @ M'
    # Note: Sigma_sr = Omega[state_idx, ret_idx] and M = Sigma_rs @ inv(Sigma_ss)
    # So Sigma_ss @ M' = Sigma_ss @ inv(Sigma_ss) @ Sigma_sr = Sigma_sr
    target = np.array(model.Sigma_rs, dtype=float).T  # Sigma_sr = Sigma_rs' (n_state, n_ret)

    err = np.max(np.abs(cross_cov - target))
    rel_err = err / np.max(np.abs(target))

    print(f"  Quadrature Cov(v^s, e_r):")
    for i in range(n_state):
        print(f"    {model.state_names[i]:>5s}  vs  [rtb, xr, xb]:  "
              f"[{cross_cov[i,0]:+.6e}, {cross_cov[i,1]:+.6e}, {cross_cov[i,2]:+.6e}]")
    print(f"  Target Sigma_sr:")
    for i in range(n_state):
        print(f"    {model.state_names[i]:>5s}  vs  [rtb, xr, xb]:  "
              f"[{target[i,0]:+.6e}, {target[i,1]:+.6e}, {target[i,2]:+.6e}]")
    print(f"  Max absolute error: {err:.2e}")
    print(f"  Max relative error: {rel_err:.2e}")

    status = "PASS" if rel_err < 1e-6 else "FAIL"
    print(f"  [{status}]")
    return rel_err < 1e-6


def test_single_layer_vs_two_layer_quadrature(model, pc, n_test=5):
    """Test 12: Two-layer quadrature matches single-layer on marginal N(0, Omega_rr).

    The terminal solver doesn't need the next-period state (no continuation value).
    So E[R_port^{1-gamma}] can equivalently be computed with a SINGLE quadrature
    layer over the full return innovation e ~ N(0, Omega_rr) instead of the
    two-layer factorization (v^s outer, eps^r inner).

    If these disagree, the state-return correlation is handled incorrectly.
    """
    print("\n=== Test 12: Two-layer vs single-layer quadrature ===")
    from scipy.special import roots_hermite

    # Build single-layer quadrature nodes for N(0, Omega_rr)
    # Use same total order: n_nodes_1d = 3 per return dimension (for fair comparison)
    n_1d = 3
    nodes_1d, weights_1d = roots_hermite(n_1d)
    weights_1d = weights_1d / np.sqrt(np.pi)
    nodes_1d = nodes_1d * np.sqrt(2.0)

    grid_1d = np.meshgrid(*([nodes_1d] * model.n_ret), indexing="ij")
    weight_1d = np.meshgrid(*([weights_1d] * model.n_ret), indexing="ij")
    z_nodes = np.stack([g.ravel() for g in grid_1d], axis=1)
    single_weights = np.prod(np.stack(weight_1d, axis=0), axis=0).ravel()

    # Transform: full Omega_rr
    Omega_rr = np.array(model.Sigma_rr, dtype=float)
    Omega_rr = 0.5 * (Omega_rr + Omega_rr.T)
    eigvals, eigvecs = np.linalg.eigh(Omega_rr)
    eigvals = np.clip(eigvals, 0.0, None)
    L_full = eigvecs @ np.diag(np.sqrt(eigvals))
    single_nodes = z_nodes @ L_full.T  # (K_single, n_ret) — full return innovations

    max_moment_err = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)

    for i_s in test_states:
        s_i = pc.state_grid[i_s]
        mu_r = model.Phi_0_ret + model.Phi_21 @ s_i  # unconditional return mean

        # --- Two-layer (terminal solver's approach) ---
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)
        opt_s, opt_b, moment_two, ec, resid = solve_portfolio_2d_terminal_constrained_njit(
            pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma)

        # Recompute moment with two-layer quadrature at optimal portfolio
        moment_two_check = 0.0
        for k_v in range(len(pc.v_weights)):
            for k_r in range(len(pc.ret_weights)):
                w = pc.v_weights[k_v] * pc.ret_weights[k_r]
                R_bill = Rx_bill[k_v, k_r]
                R_s = R_bill * Rx_stock_mult[k_v, k_r]
                R_b = R_bill * Rx_bond_mult[k_v, k_r]
                a_bill = 1.0 - opt_s - opt_b
                R_p = opt_s * R_s + opt_b * R_b + a_bill * R_bill
                moment_two_check += w * R_p ** (1.0 - model.gamma)

        # --- Single-layer (direct N(0, Omega_rr) quadrature) ---
        moment_single = 0.0
        for k in range(len(single_weights)):
            log_rtb = mu_r[0] + single_nodes[k, 0]
            log_xr  = mu_r[1] + single_nodes[k, 1]
            log_xb  = mu_r[2] + single_nodes[k, 2]
            R_bill = np.exp(log_rtb)
            R_s = R_bill * np.exp(log_xr)
            R_b = R_bill * np.exp(log_xb)
            a_bill = 1.0 - opt_s - opt_b
            R_p = opt_s * R_s + opt_b * R_b + a_bill * R_bill
            moment_single += single_weights[k] * R_p ** (1.0 - model.gamma)

        rel_err = abs(moment_two_check - moment_single) / max(abs(moment_single), 1e-15)
        max_moment_err = max(max_moment_err, rel_err)
        print(f"  i_s={i_s:3d}  two-layer={moment_two_check:.10f}  single-layer={moment_single:.10f}"
              f"  rel_err={rel_err:.2e}")

    # Both are low-order GH, so they won't be identical — but should be close
    # The two-layer uses 27*8=216 nodes, single uses 3^3=27 nodes
    # Use generous tolerance since quadrature orders differ
    status = "PASS" if max_moment_err < 0.02 else "FAIL"
    print(f"  Max relative moment error: {max_moment_err:.2e}  [{status}]")
    return max_moment_err < 0.02


def test_monte_carlo_cross_validation(model, pc, n_mc=500_000, n_test=8, seed=42):
    """Test 13: Monte Carlo cross-validation of the quadrature moment.

    For each test state i_s, draw n_mc samples from the FULL return innovation
    distribution N(0, Omega_rr) (using the Cholesky of the 3x3 return covariance),
    compute E[R_port^{1-gamma}] by sample average, and compare to the quadrature
    result.

    This tests whether the low-order Gauss-Hermite quadrature (27 state nodes x
    8 return nodes = 216 scenarios) accurately integrates the highly nonlinear
    integrand R_port^{1-gamma} over the true joint return distribution. The
    quadrature exactly reproduces the first few polynomial moments of the
    distribution, but R^{1-gamma} for gamma=5 is R^{-4} — a steep, convex
    function that amplifies the tails. If the quadrature is too coarse for this
    integrand, the MC estimate will diverge.

    Uses a fixed seed for reproducibility.
    """
    print(f"\n=== Test 13: Monte Carlo cross-validation (n_mc={n_mc:,d}) ===")

    rng = np.random.default_rng(seed)

    # Cholesky of full Omega_rr for sampling
    Omega_rr = np.array(model.Sigma_rr, dtype=float)
    Omega_rr = 0.5 * (Omega_rr + Omega_rr.T)
    L_full = np.linalg.cholesky(Omega_rr)

    # Draw all MC samples at once: eps ~ N(0, Omega_rr)
    z_samples = rng.standard_normal((n_mc, model.n_ret))
    eps_samples = z_samples @ L_full.T  # (n_mc, 3) — [rtb, xr, xb] innovations

    max_rel_err = 0.0
    test_states = np.linspace(0, pc.N_state - 1, min(n_test, pc.N_state), dtype=int)

    for i_s in test_states:
        s_i = pc.state_grid[i_s]
        mu_r = model.Phi_0_ret + model.Phi_21 @ s_i  # (3,) conditional return mean

        # --- Quadrature moment at optimal portfolio ---
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes)
        opt_s, opt_b, moment_quad, ec, resid = solve_portfolio_2d_terminal_constrained_njit(
            pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma)

        # --- Monte Carlo moment at the same portfolio ---
        log_rtb = mu_r[0] + eps_samples[:, 0]  # (n_mc,)
        log_xr  = mu_r[1] + eps_samples[:, 1]
        log_xb  = mu_r[2] + eps_samples[:, 2]

        R_bill_mc = np.exp(log_rtb)
        R_stock_mc = R_bill_mc * np.exp(log_xr)
        R_bond_mc = R_bill_mc * np.exp(log_xb)

        a_bill = 1.0 - opt_s - opt_b
        R_port_mc = opt_s * R_stock_mc + opt_b * R_bond_mc + a_bill * R_bill_mc
        moment_mc = float(np.mean(R_port_mc ** (1.0 - model.gamma)))

        # MC standard error for reference
        mc_samples = R_port_mc ** (1.0 - model.gamma)
        mc_se = float(np.std(mc_samples) / np.sqrt(n_mc))

        rel_err = abs(moment_quad - moment_mc) / max(abs(moment_mc), 1e-15)
        # How many MC standard errors apart?
        n_se = abs(moment_quad - moment_mc) / max(mc_se, 1e-15)
        max_rel_err = max(max_rel_err, rel_err)

        print(f"  i_s={i_s:3d}  quad={moment_quad:.8f}  MC={moment_mc:.8f}"
              f"  rel_err={rel_err:.2e}  ({n_se:.1f} SE)")

    # Tolerance: quadrature is an approximation with finite nodes. For gamma=5,
    # R^{-4} amplifies tails, so some discrepancy is expected. But it should be
    # small — within a few percent.
    status = "PASS" if max_rel_err < 0.05 else "FAIL"
    print(f"  Max relative error (quad vs MC): {max_rel_err:.2e}  [{status}]")
    return max_rel_err < 0.05


if __name__ == "__main__":
    print("Building model and precomputed arrays...")
    model, pc = build_test_pc(state_grid_sizes=(7, 7, 7), n_z=7, gamma=5.0)
    print(f"N_state = {pc.N_state}, n_z = {pc.n_z}, n_wealth = {len(pc.wealth_grid)}")

    results = []
    results.append(("CRRA homogeneity",        test_crra_homogeneity(model, pc)))
    results.append(("z-independence",           test_z_independence(model, pc)))
    results.append(("FOC residual at solution", test_foc_residual_at_solution(model, pc, n_test=15)))
    results.append(("Brute-force grid search",  test_brute_force_grid_search(model, pc, n_grid=201, n_test=8)))
    results.append(("Moment positivity",        test_moment_positivity(model, pc)))
    results.append(("Consumption bounds",       test_consumption_bounds(model, pc)))
    results.append(("FOC vs finite-difference", test_foc_matches_finite_difference(model, pc, n_test=8)))
    results.append(("Terminal vs retirement returns", test_terminal_vs_retirement_returns(model, pc, n_test=8)))
    results.append(("Quadrature mean & covariance",  test_quadrature_mean_and_covariance(model, pc, n_test=8)))
    results.append(("Return correlation structure",   test_return_correlation_structure(model, pc, n_test=5)))
    results.append(("State-return cross-covariance", test_state_return_cross_covariance(model, pc)))
    results.append(("Single vs two-layer quadrature", test_single_layer_vs_two_layer_quadrature(model, pc, n_test=8)))
    results.append(("Monte Carlo cross-validation",  test_monte_carlo_cross_validation(model, pc, n_mc=500_000, n_test=8)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  {tag}  {name}")
        if not passed:
            all_pass = False
    print("=" * 60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")

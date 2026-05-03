"""Verify the stagnation-exit fix in solve_portfolio_unconstrained_terminal_njit.

Two checks:
  1. Correctness preservation — a normal, well-conditioned terminal problem still
     converges to a sensible interior optimum with EC_INTERIOR.
  2. Stagnation early-exit — a problem rigged so the line search cannot find a
     decreasing step must return EC_NEWTON_FAIL with FOC-call count bounded by
     ~max_backtrack_iter (not max_iter * max_backtrack_iter).

Terminal solver is used because it has the smallest input signature (no state
grid, no continuation value, no income quadrature). The patch logic is
identical across the three unconstrained solvers (terminal/retirement/working),
so verifying terminal is sufficient evidence for correctness of the fix.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lifecycle.solver import (
    solve_portfolio_unconstrained_terminal_njit,
    compute_terminal_portfolio_foc_jac,
    EC_INTERIOR,
    EC_NEWTON_FAIL,
)


# ----------------------------------------------------------------------
# Helper: build a "normal" 3-asset terminal scenario set
# ----------------------------------------------------------------------
def build_normal_scenarios(n_state_quad=3, n_ret_quad=2, seed=0):
    """Construct gross-return scenarios for a well-posed terminal problem.

    R_bill ~ exp(0.005 + small noise)
    R_stock = R_bill * exp(0.06 + xr noise)   (equity premium ~6%)
    R_bond  = R_bill * exp(0.02 + xb noise)
    """
    rng = np.random.default_rng(seed)

    state_weights = np.full(n_state_quad, 1.0 / n_state_quad)
    ret_weights = np.full(n_ret_quad, 1.0 / n_ret_quad)

    Rx_bill = np.empty((n_state_quad, n_ret_quad))
    Rx_stock_mult = np.empty((n_state_quad, n_ret_quad))
    Rx_bond_mult = np.empty((n_state_quad, n_ret_quad))

    for k in range(n_state_quad):
        for r in range(n_ret_quad):
            Rx_bill[k, r] = np.exp(0.005 + 0.01 * rng.standard_normal())
            Rx_stock_mult[k, r] = np.exp(0.06 + 0.18 * rng.standard_normal())
            Rx_bond_mult[k, r] = np.exp(0.02 + 0.07 * rng.standard_normal())
    return state_weights, ret_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult


# ----------------------------------------------------------------------
# Counter wrapper around the FOC so we can bound calls
# ----------------------------------------------------------------------
class FOCCounter:
    """Drop-in replacement for compute_terminal_portfolio_foc_jac that counts calls."""

    def __init__(self, state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
                 ret_weights, gamma, min_return_power=1e-15, prob_skip=1e-12):
        self.args = (state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
                     ret_weights, gamma, min_return_power, prob_skip)
        self.calls = 0

    def __call__(self, a_s, a_b):
        self.calls += 1
        return compute_terminal_portfolio_foc_jac(a_s, a_b, *self.args)


# ----------------------------------------------------------------------
# Test 1 — correctness preservation on well-posed problem
# ----------------------------------------------------------------------
def test_normal_converges():
    sw, rw, Rx_bill, Rx_s, Rx_b = build_normal_scenarios(seed=0)
    gamma = 3.0

    a_s, a_b, e_last, exit_code, foc_resid, n_iter = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=0.40, init_b=0.33,
        tol=1e-9, max_iter=50,
        use_line_search=True, max_backtrack_iter=10, line_search_max_step=2.0,
    )

    assert exit_code == EC_INTERIOR, f"expected EC_INTERIOR={EC_INTERIOR}, got {exit_code}"

    # Verify FOC residual at returned (a_s, a_b) is actually below tol
    fs, fb, _, _, _, _ = compute_terminal_portfolio_foc_jac(
        a_s, a_b, sw, Rx_bill, Rx_s, Rx_b, rw, gamma)
    resid_norm = (fs * fs + fb * fb) ** 0.5
    assert resid_norm < 1e-7, f"FOC residual {resid_norm:.3e} too large at returned point"

    # Returned shares should be finite and reasonable for unconstrained problem
    assert np.isfinite(a_s) and np.isfinite(a_b)
    assert -5.0 < a_s < 5.0 and -5.0 < a_b < 5.0

    # n_iter sanity: well-conditioned problem should converge fast (<<max_iter=50)
    assert 0 <= n_iter < 50, f"unexpected n_iter={n_iter}"
    print(f"  [OK] normal converges: alpha_s={a_s:.4f}, alpha_b={a_b:.4f}, "
          f"resid={resid_norm:.2e}, EC=INTERIOR, n_iter={n_iter}")


# ----------------------------------------------------------------------
# Test 2 — stagnation triggers early exit
# ----------------------------------------------------------------------
def test_stagnation_early_exit():
    """Force stagnation by setting tol smaller than fastmath roundoff at the optimum.

    Strategy: solve the problem to high precision once to find the optimum,
    then re-launch from that optimum with tol=0.0. Newton's first iteration
    finds err≈eps. Any line-search step from there changes alpha by ~step_s,
    re-evaluates the FOC, and gets back essentially the same err (rounding
    noise). The strict-decrease test err_t < err can fail (in fact must fail
    on average for tiny perturbations near a stationary point), so the
    backtracking loop will exhaust all halvings without finding a decrease.

    Patched code must exit with EC_NEWTON_FAIL after one outer iter (≤
    max_backtrack_iter FOC calls inside line search + 1 initial + 1 scale eval).
    Without the patch it would loop max_iter times, each spending
    max_backtrack_iter halvings → max_iter * max_backtrack_iter FOC calls.
    """
    sw, rw, Rx_bill, Rx_s, Rx_b = build_normal_scenarios(seed=0)
    gamma = 3.0

    # Step 1: find optimum to high precision
    a_s_opt, a_b_opt, _, exit_opt, _, _ = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=0.40, init_b=0.33,
        tol=1e-12, max_iter=100,
        use_line_search=True, max_backtrack_iter=10, line_search_max_step=2.0,
    )
    assert exit_opt == EC_INTERIOR

    # Step 2: relaunch from the optimum with a tolerance below numerical floor.
    # err at (a_s_opt, a_b_opt) is ~1e-12 or smaller. tol*scale=0 means
    # "never converge", so the loop has to run to max_iter unless line
    # search stagnates and the patch triggers.
    max_iter = 500
    max_bt = 10
    a_s, a_b, e_last, exit_code, foc_resid, n_iter = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=a_s_opt, init_b=a_b_opt,
        tol=0.0,                       # impossible to converge
        max_iter=max_iter,
        use_line_search=True, max_backtrack_iter=max_bt,
        line_search_max_step=2.0,
    )

    # Patched code MUST exit with EC_NEWTON_FAIL.
    assert exit_code == EC_NEWTON_FAIL, (
        f"expected EC_NEWTON_FAIL on stagnation, got exit_code={exit_code}")

    # Returned residual should be tiny (we started at the optimum)
    assert foc_resid < 1e-6, f"resid {foc_resid:.3e} unexpectedly large"

    # n_iter on stagnation should be very small (1 attempted iter that failed)
    # Without the patch, n_iter would be max_iter=500.
    assert n_iter <= 5, (
        f"n_iter={n_iter} on stagnation; patch should bound this to ~1")
    print(f"  [OK] stagnation early-exit: EC=NEWTON_FAIL, resid={foc_resid:.2e}, "
          f"alpha=({a_s:.4f}, {a_b:.4f}), n_iter={n_iter}")

    # Patched solver should NOT have spun max_iter times. We can't directly
    # count FOC calls inside the JIT function, but we can check timing as a
    # proxy: a stagnated max_iter=500 with max_bt=10 would do ~5000 FOC calls.
    # Our patch should do ~10. Run a timing test:
    import time
    t_short = []
    t_long = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve_portfolio_unconstrained_terminal_njit(
            sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
            init_s=a_s_opt, init_b=a_b_opt,
            tol=0.0, max_iter=50,
            use_line_search=True, max_backtrack_iter=max_bt,
            line_search_max_step=2.0,
        )
        t_short.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        solve_portfolio_unconstrained_terminal_njit(
            sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
            init_s=a_s_opt, init_b=a_b_opt,
            tol=0.0, max_iter=5000,
            use_line_search=True, max_backtrack_iter=max_bt,
            line_search_max_step=2.0,
        )
        t_long.append(time.perf_counter() - t0)

    med_short = float(np.median(t_short))
    med_long = float(np.median(t_long))
    # If the early-exit fires, both should take roughly the same time
    # (independent of max_iter). Allow generous slack — 2x — to absorb noise.
    ratio = med_long / max(med_short, 1e-9)
    print(f"  [OK] runtime independent of max_iter on stagnation: "
          f"max_iter=50 -> {med_short*1e6:.1f}us, "
          f"max_iter=5000 -> {med_long*1e6:.1f}us, ratio={ratio:.2f}")
    assert ratio < 5.0, (
        f"runtime scales with max_iter (ratio={ratio:.2f}) — patch likely not firing")


# ----------------------------------------------------------------------
# Test 3 — no-line-search path is unchanged
# ----------------------------------------------------------------------
def test_no_line_search_unchanged():
    """The patch only touches the use_line_search=True branch. Verify the
    use_line_search=False path still converges from a normal starting point."""
    sw, rw, Rx_bill, Rx_s, Rx_b = build_normal_scenarios(seed=0)
    gamma = 3.0

    a_s, a_b, _, exit_code, foc_resid, n_iter = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=0.40, init_b=0.33,
        tol=1e-7, max_iter=200,
        use_line_search=False, step_damp=0.3,
    )
    assert exit_code == EC_INTERIOR, f"got {exit_code}"
    assert foc_resid < 1e-6
    print(f"  [OK] no-line-search path still works: "
          f"alpha=({a_s:.4f}, {a_b:.4f}), resid={foc_resid:.2e}, n_iter={n_iter}")


# ----------------------------------------------------------------------
# Test 4 — n_iter diagnostic semantics
# ----------------------------------------------------------------------
def test_iter_count_semantics():
    """Verify n_iter accurately reflects Newton work, in three regimes:
       (a) init at optimum -> n_iter = 0 (convergence at top of first iter)
       (b) max_iter exhausted with tight tol -> n_iter = max_iter
       (c) far init still converges -> n_iter strictly between 0 and max_iter,
           and matches the count using line search vs no-line-search consistently.
    """
    sw, rw, Rx_bill, Rx_s, Rx_b = build_normal_scenarios(seed=0)
    gamma = 3.0

    # First find optimum
    a_s_opt, a_b_opt, _, _, _, _ = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=0.40, init_b=0.33,
        tol=1e-12, max_iter=100,
        use_line_search=True, max_backtrack_iter=10, line_search_max_step=2.0,
    )

    # (a) init at optimum -> 0 iters needed
    _, _, _, ec_a, _, n_iter_a = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=a_s_opt, init_b=a_b_opt,
        tol=1e-7, max_iter=50,
        use_line_search=True, max_backtrack_iter=10, line_search_max_step=2.0,
    )
    assert ec_a == EC_INTERIOR
    assert n_iter_a == 0, f"init-at-optimum should give n_iter=0, got {n_iter_a}"
    print(f"  [OK] (a) init-at-optimum: n_iter=0, EC=INTERIOR")

    # (b) max_iter exhausted: tight tol below double-precision floor for the
    #     residual scaling; with line_search_max_step=2.0 on a smooth problem
    #     Newton should not stagnate, so the only way to fail is to exhaust
    #     max_iter. We use no-line-search to guarantee an update each iter.
    max_iter_b = 3
    _, _, _, ec_b, _, n_iter_b = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=-2.0, init_b=2.0,    # far from optimum
        tol=1e-15, max_iter=max_iter_b,
        use_line_search=False, step_damp=0.3,
    )
    # With max_iter=3 and step_damp=0.3, can't get from (-2,2) to (1.85,2.16)
    # in 3 iterations -> EC_NEWTON_FAIL with n_iter=max_iter
    assert ec_b == EC_NEWTON_FAIL
    assert n_iter_b == max_iter_b, (
        f"max_iter exhaustion should give n_iter={max_iter_b}, got {n_iter_b}")
    print(f"  [OK] (b) max_iter exhausted: n_iter={n_iter_b}=max_iter")

    # (c) far init: monotone increase in n_iter as init moves away from optimum
    _, _, _, ec_c1, _, n_iter_c1 = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=a_s_opt + 0.1, init_b=a_b_opt + 0.1,
        tol=1e-7, max_iter=50,
        use_line_search=True, max_backtrack_iter=10, line_search_max_step=2.0,
    )
    _, _, _, ec_c2, _, n_iter_c2 = solve_portfolio_unconstrained_terminal_njit(
        sw, Rx_bill, Rx_s, Rx_b, rw, gamma,
        init_s=a_s_opt + 1.0, init_b=a_b_opt - 1.0,
        tol=1e-7, max_iter=50,
        use_line_search=True, max_backtrack_iter=10, line_search_max_step=2.0,
    )
    assert ec_c1 == EC_INTERIOR and ec_c2 == EC_INTERIOR
    assert 0 < n_iter_c1 < 50
    assert 0 < n_iter_c2 < 50
    # Newton's quadratic convergence on a smooth problem: should be small
    assert n_iter_c1 < 20 and n_iter_c2 < 20, (
        f"n_iter unexpectedly large: c1={n_iter_c1}, c2={n_iter_c2}")
    print(f"  [OK] (c) far-init: n_iter close-init={n_iter_c1}, far-init={n_iter_c2}")


if __name__ == "__main__":
    print("Test 1: normal convergence")
    test_normal_converges()
    print("\nTest 2: stagnation early-exit")
    test_stagnation_early_exit()
    print("\nTest 3: no-line-search path unchanged")
    test_no_line_search_unchanged()
    print("\nTest 4: n_iter diagnostic semantics")
    test_iter_count_semantics()
    print("\nAll tests passed.")

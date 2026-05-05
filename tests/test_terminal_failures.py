"""Fish for the cause of terminal-age Newton failures with smoke-test quadrature.

Setup mirrors main.ipynb's current disc_config (state_grid_sizes=(5,5,5),
n_state_quad_nodes=2, n_ret_nodes_1d=3, n_eta_nodes=2, n_eps_nodes=2).

For each i_s where the unconstrained terminal Newton returns EC_NEWTON_FAIL,
we do a brute-force grid scan of (alpha_s, alpha_b) ∈ [-3, 3]² to find the
true minimum of ||F||. Then we compare:
  - Newton's returned (alpha_s, alpha_b) vs the brute-force optimum
  - Newton's residual vs the brute-force minimum residual
  - State values at the failing states (in MODEL state-name order)
  - Whether R_p ever goes near min_return_power in the FOC scenarios

Hypothesis being tested: for unconstrained portfolios at certain financial
states, R_p can go ≤ min_return_power for some quadrature scenarios when
the optimum has significant short positions. The clamp `max(R_p,
min_return_power)` creates a non-smooth kink in the FOC surface that line
search rejects.
"""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.solver import (
    solve_portfolio_unconstrained_terminal_njit,
    compute_terminal_portfolio_foc_jac,
    _build_terminal_quad_returns,
    EC_INTERIOR,
    EC_NEWTON_FAIL,
)


# ----------------------------------------------------------------------
# Build minimal model matching main.ipynb smoke-test config
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

def build_smoke_precompute():
    var_config, _, _ = build_nominal_system1_var_config(
        csv_path=str(ROOT / "data" / "var_dataset.csv")
    )
    base_config = {
        "beta": 0.96, "gamma": 3.0, "b_bar": 10,
        "start_age": 22, "retire_age": 67, "terminal_age": 99,
        "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
        "rho": 0.991, "pz": 0.176,
        "mu_eta1": -0.524, "sigma_eta1": 0.113,
        "mu_eta2": -(0.176/(1-0.176))*(-0.524), "sigma_eta2": 0.046,
        "pe": 0.044,
        "mu_eps1": 0.134, "sigma_eps1": 0.762,
        "mu_eps2": 0.0, "sigma_eps2": 0.055,
        "constrained": False,
    }
    model = build_model(base_config, var_config, verbose=False)
    disc = DiscretizationConfig(
        n_wealth=150,
        n_savings=150,
        state_grid_sizes=(5, 5, 5),
        state_grid_mode="cholesky",
        state_n_stds=3.0,
        n_z=11,
        n_stds=3.0,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=3,
        n_state_quad_nodes=2,
    )
    pc = Precompute(model, disc, verbose=False)
    return model, pc


# ----------------------------------------------------------------------
# Brute-force minimum of ‖F(alpha_s, alpha_b)‖ over a coarse-then-fine grid
# ----------------------------------------------------------------------
def brute_force_min(state_weights, Rx_bill, Rx_s, Rx_b, ret_weights, gamma,
                    a_range=(-3.0, 3.0), b_range=(-3.0, 3.0), n=121):
    a_grid = np.linspace(a_range[0], a_range[1], n)
    b_grid = np.linspace(b_range[0], b_range[1], n)
    best = (None, None, np.inf, None, None)
    for a in a_grid:
        for b in b_grid:
            fs, fb, _, _, _, e = compute_terminal_portfolio_foc_jac(
                a, b, state_weights, Rx_bill, Rx_s, Rx_b, ret_weights, gamma)
            err = (fs * fs + fb * fb) ** 0.5
            if err < best[2]:
                best = (a, b, err, fs, fb)
    return best  # (a, b, err, fs, fb)


def refine(state_weights, Rx_bill, Rx_s, Rx_b, ret_weights, gamma, a0, b0,
           radius=0.1, n=121):
    return brute_force_min(state_weights, Rx_bill, Rx_s, Rx_b, ret_weights, gamma,
                           a_range=(a0 - radius, a0 + radius),
                           b_range=(b0 - radius, b0 + radius), n=n)


def check_rp_near_zero(alpha_s, alpha_b, Rx_bill, Rx_s, Rx_b):
    """Return min R_p across all quadrature scenarios at given alpha."""
    a_bill = 1.0 - alpha_s - alpha_b
    R_bill = Rx_bill                       # (n_state_quad, n_ret_quad)
    R_s = R_bill * Rx_s
    R_b = R_bill * Rx_b
    R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
    return float(R_p.min()), float(R_p.max())


# ----------------------------------------------------------------------
# Main fish
# ----------------------------------------------------------------------
def main():
    print("Building smoke-test Precompute...")
    model, pc = build_smoke_precompute()
    N_state = pc.N_state
    print(f"  N_state={N_state}, n_z={pc.n_z}, "
          f"v_nodes={len(pc.v_weights)}, ret_nodes={len(pc.ret_weights)}")

    # Solver config matching main.ipynb's unconstrained cell
    base_solver = SolverConfig()
    sc = base_solver._replace(
        max_iter_unconstrained=8000,
        init_alpha_s=0.85,
        init_alpha_b=0.44,
        use_line_search=True,
    )

    # Run the terminal Newton at every i_s
    failures = []
    interior_results = []
    for i_s in range(N_state):
        Rx_bill, Rx_s, Rx_b = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes
        )
        a, b, e, ec, resid, n_iter = solve_portfolio_unconstrained_terminal_njit(
            pc.v_weights, Rx_bill, Rx_s, Rx_b, pc.ret_weights, model.gamma,
            init_s=sc.init_alpha_s, init_b=sc.init_alpha_b,
            tol=sc.tol, max_iter=sc.max_iter_unconstrained,
            use_line_search=sc.use_line_search,
            max_backtrack_iter=sc.max_backtrack_iter,
            line_search_max_step=sc.line_search_max_step,
        )
        record = (i_s, a, b, e, ec, resid, n_iter, Rx_bill, Rx_s, Rx_b)
        if ec == EC_NEWTON_FAIL:
            failures.append(record)
        elif ec == EC_INTERIOR:
            interior_results.append(record)

    print(f"\nTerminal Newton results:")
    print(f"  EC_INTERIOR:    {len(interior_results)}/{N_state}")
    print(f"  EC_NEWTON_FAIL: {len(failures)}/{N_state}")
    if not failures:
        print("  -> no failures; nothing to fish for.")
        return

    # n_iter distribution
    iters_int = sorted(r[6] for r in interior_results)
    iters_fail = sorted(r[6] for r in failures)
    if iters_int:
        print(f"  Iters (interior): min={iters_int[0]}, median={iters_int[len(iters_int)//2]}, max={iters_int[-1]}")
    if iters_fail:
        print(f"  Iters (failed):   min={iters_fail[0]}, median={iters_fail[len(iters_fail)//2]}, max={iters_fail[-1]}")
        # Failures with low iter count = line-search stagnation
        # Failures with iter == max_iter = budget exhaustion
        n_stagnated = sum(1 for r in failures if r[6] < sc.max_iter_unconstrained)
        n_exhausted = sum(1 for r in failures if r[6] >= sc.max_iter_unconstrained)
        print(f"  Stagnated (iter<budget): {n_stagnated}")
        print(f"  Exhausted (iter=budget): {n_exhausted}")

    # ------------------------------------------------------------------
    # Brute-force scan for the first 5 failing states
    # ------------------------------------------------------------------
    print(f"\nBrute-force scan of first 5 failing states:")
    _state_hdr = "state " + ",".join(model.state_names)
    print(f"  {'i_s':>4}  {_state_hdr:<24}  "
          f"{'Newton (a,b)':<18} {'N_resid':>9}  "
          f"{'BruteOpt (a,b)':<18} {'BF_resid':>9}  "
          f"{'min R_p @ BF':>12}  {'iters':>5}")
    print(f"  {'-'*4:>4}  {'-'*24:<24}  {'-'*18:<18} {'-'*9:>9}  "
          f"{'-'*18:<18} {'-'*9:>9}  {'-'*12:>12}  {'-'*5:>5}")

    for rec in failures[:5]:
        i_s, a_n, b_n, _, _, n_resid, n_iter, Rx_bill, Rx_s, Rx_b = rec
        s_i = pc.state_grid[i_s]
        a_bf, b_bf, e_bf, _, _ = brute_force_min(
            pc.v_weights, Rx_bill, Rx_s, Rx_b, pc.ret_weights, model.gamma)
        a_bf, b_bf, e_bf, _, _ = refine(
            pc.v_weights, Rx_bill, Rx_s, Rx_b, pc.ret_weights, model.gamma,
            a_bf, b_bf, radius=0.1, n=51)
        rp_min, _ = check_rp_near_zero(a_bf, b_bf, Rx_bill, Rx_s, Rx_b)
        state_str = f"({s_i[0]:.3f},{s_i[1]:.3f},{s_i[2]:.2f})"
        print(f"  {i_s:>4}  {state_str:<24}  "
              f"({a_n:5.2f},{b_n:5.2f})  {n_resid:>9.2e}  "
              f"({a_bf:5.2f},{b_bf:5.2f})  {e_bf:>9.2e}  "
              f"{rp_min:>12.4e}  {n_iter:>5}")

    # ------------------------------------------------------------------
    # Summary diagnostic across ALL failures
    # ------------------------------------------------------------------
    print(f"\nAcross all {len(failures)} failures:")
    rp_mins_at_newton = []
    rp_mins_at_bf = []
    bf_residuals = []
    distance_newton_to_bf = []
    for rec in failures:
        i_s, a_n, b_n, _, _, n_resid, _, Rx_bill, Rx_s, Rx_b = rec
        # Quick BF (lower-res) to keep runtime reasonable
        a_bf, b_bf, e_bf, _, _ = brute_force_min(
            pc.v_weights, Rx_bill, Rx_s, Rx_b, pc.ret_weights, model.gamma, n=61)
        bf_residuals.append(e_bf)
        distance_newton_to_bf.append(((a_n - a_bf)**2 + (b_n - b_bf)**2)**0.5)
        rp_n, _ = check_rp_near_zero(a_n, b_n, Rx_bill, Rx_s, Rx_b)
        rp_b, _ = check_rp_near_zero(a_bf, b_bf, Rx_bill, Rx_s, Rx_b)
        rp_mins_at_newton.append(rp_n)
        rp_mins_at_bf.append(rp_b)

    bf_residuals = np.array(bf_residuals)
    distance = np.array(distance_newton_to_bf)
    rp_mins_at_newton = np.array(rp_mins_at_newton)
    rp_mins_at_bf = np.array(rp_mins_at_bf)

    print(f"  Brute-force min ||F||: median={np.median(bf_residuals):.2e}, max={bf_residuals.max():.2e}")
    print(f"  -> If BF residual is large, the FOC has no zero in [-3,3]² (extreme optimum).")
    print(f"  Newton-to-BF distance: median={np.median(distance):.3f}, max={distance.max():.3f}")
    print(f"  -> If small, Newton is near the optimum but failing tol; large = far off.")
    print(f"  min R_p at Newton iterate: median={np.median(rp_mins_at_newton):.3e}, "
          f"min={rp_mins_at_newton.min():.3e}")
    print(f"  min R_p at brute-force opt: median={np.median(rp_mins_at_bf):.3e}, "
          f"min={rp_mins_at_bf.min():.3e}")
    print(f"  -> If min R_p is near min_return_power (1e-15), the clamp is creating "
          f"a non-smooth FOC.")

    # Buckets
    n_bf_large = int(np.sum(bf_residuals > 1.0))
    n_close = int(np.sum(distance < 0.05))
    n_far = int(np.sum(distance > 0.5))
    n_rp_near_zero = int(np.sum(rp_mins_at_newton < 1e-3))
    print(f"\n  Buckets:")
    print(f"    Failures where BF residual > 1.0 (no zero in [-3,3]²): {n_bf_large}")
    print(f"    Failures where Newton is within 0.05 of BF optimum:    {n_close}")
    print(f"    Failures where Newton is >0.5 from BF optimum:          {n_far}")
    print(f"    Failures where R_p < 1e-3 at iterate (clamp territory): {n_rp_near_zero}")


if __name__ == "__main__":
    main()

"""
solver.py — Backward induction solver with EGM + 2D Newton-Raphson.

Three assets: Bills, Stocks, Nominal Bonds.
Catherine (2025) bequest motive: b(W, A) = b_bar * (W/A)^(1-gamma) / (1-gamma)

Contains:
  - Diagnostic constants (DI_*, DF_*, EC_*)
  - Interpolation utilities (fast_interp_1d, etc.)
  - FOC + Jacobian functions (retirement, working, terminal)
  - Newton portfolio solvers (constrained + unconstrained variants)
  - Period solvers: solve_retirement_step(), solve_working_age_step()
  - Terminal solver: solve_terminal_age()
  - Master solver: run_lifecycle_solver()
  - Post-solve diagnostic report

Policy function output shapes:
    C_mat[t, i_z, i_s, i_w]  -- optimal consumption        (n_age, n_z, N_state, n_w)
    S_mat[t, i_z, i_s, i_w]  -- optimal stock share         (n_age, n_z, N_state, n_w)
    B_mat[t, i_z, i_s, i_w]  -- optimal bond share          (n_age, n_z, N_state, n_w)

Dependencies: numpy, numba, model (for SolverConfig)
"""

import numpy as np
from numba import njit, prange
from math import exp
import time
from scipy.optimize import minimize, Bounds, LinearConstraint

from model import SolverConfig

# =============================================================================
# DIAGNOSTIC CONSTANTS
# =============================================================================
# Integer counter indices -- diag_int[i_s, idx], shape (N_state, N_DIAG_INT)
DI_CORNER_BILLS    = 0   # all-bills corner solution
DI_CORNER_STOCKS   = 1   # all-stocks corner
DI_CORNER_BONDS    = 2   # all-bonds corner
DI_EDGE_SB         = 3   # stock + bill edge
DI_EDGE_BB         = 4   # bond + bill edge
DI_EDGE_STOCKBOND  = 5   # stock + bond edge (no bills)
DI_INTERIOR        = 6   # interior Newton converged
DI_NEWTON_FAIL     = 7   # Newton hit max_iter without converging
DI_SINGULAR_JAC    = 8   # singular Jacobian event
DI_TINY_SAVINGS    = 9   # s_val < threshold, trivial all-bills
DI_TOTAL_CALLS     = 10  # total Newton calls
DI_NEG_CONSUMPTION = 11  # negative euler (before clamping)
DI_MONO_VIOLATIONS = 12  # EGM monotonicity violations
N_DIAG_INT = 13

# Float counter indices -- diag_float[i_s, idx], shape (N_state, N_DIAG_FLOAT)
DF_WORST_MONO_DROP  = 0  # largest endogenous grid inversion
DF_MAX_FOC_RESID    = 1  # worst FOC residual at Newton exit
DF_SUM_FOC_RESID_SQ = 2  # sum of squared FOC residuals (for RMS)
DF_MIN_ALPHA_S      = 3  # min stock share
DF_MAX_ALPHA_S      = 4  # max stock share
DF_MIN_ALPHA_B      = 5  # min bond share
DF_MAX_ALPHA_B      = 6  # max bond share
DF_SUM_ALPHA_S      = 7  # sum of stock shares (for mean)
DF_SUM_ALPHA_B      = 8  # sum of bond shares (for mean)
N_DIAG_FLOAT = 9

# Exit codes from Newton solvers
EC_TINY_SAVINGS = 0
EC_CORNER_BILLS = 1
EC_CORNER_STOCKS = 2
EC_CORNER_BONDS = 3
EC_EDGE_SB = 4       # stock + bill
EC_EDGE_BB = 5       # bond + bill
EC_EDGE_STOCKBOND = 6 # stock + bond
EC_INTERIOR = 7       # interior Newton converged
EC_NEWTON_FAIL = 8    # Newton did not converge

# Mapping from exit_code to diag_int index
_EC_TO_DI = np.array([
    DI_TINY_SAVINGS,    # EC=0
    DI_CORNER_BILLS,    # EC=1
    DI_CORNER_STOCKS,   # EC=2
    DI_CORNER_BONDS,    # EC=3
    DI_EDGE_SB,         # EC=4
    DI_EDGE_BB,         # EC=5
    DI_EDGE_STOCKBOND,  # EC=6
    DI_INTERIOR,        # EC=7
    DI_NEWTON_FAIL,     # EC=8
], dtype=np.int64)


# =============================================================================
# HELPERS: INTERPOLATION AND SIMPLEX PROJECTION
# =============================================================================

@njit(fastmath=True)
def fast_interp_slope_1d(x, x_grid, y_grid):
    """
    Slope of the piecewise-linear interpolant -- i.e. dc_next/dx_next (MPC).
    Same binary search as fast_interp_1d, but returns the interval slope.
    At the extrapolation boundaries, returns the nearest interior slope.
    """
    n = len(x_grid)
    if n < 2:
        return 0.0
    if x <= x_grid[0]:
        return (y_grid[1] - y_grid[0]) / (x_grid[1] - x_grid[0] + 1e-30)
    if x >= x_grid[n - 1]:
        return (y_grid[n-1] - y_grid[n-2]) / (x_grid[n-1] - x_grid[n-2] + 1e-30)
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    dx = x_grid[hi] - x_grid[lo]
    if dx < 1e-30:
        return 0.0
    return (y_grid[hi] - y_grid[lo]) / dx


@njit(fastmath=True)
def fast_interp_1d(x, x_grid, y_grid):
    """Linear interpolation on a sorted grid with binary search.
    Uses linear extrapolation beyond grid boundaries."""
    n = len(x_grid)
    if x <= x_grid[0]:
        dx = x_grid[1] - x_grid[0] + 1e-30
        return y_grid[0] + (y_grid[1] - y_grid[0]) * (x - x_grid[0]) / dx
    if x >= x_grid[n - 1]:
        dx = x_grid[n - 1] - x_grid[n - 2] + 1e-30
        return y_grid[n - 1] + (y_grid[n - 1] - y_grid[n - 2]) * (x - x_grid[n - 1]) / dx
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_grid[lo], x_grid[hi]
    y0, y1 = y_grid[lo], y_grid[hi]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@njit(fastmath=True)
def fast_interp_1d_with_slope(x, x_grid, y_grid):
    """Combined interpolation: returns (value, slope) from a single binary search."""
    n = len(x_grid)
    if n < 2:
        return y_grid[0], 0.0
    if x <= x_grid[0]:
        dx = x_grid[1] - x_grid[0] + 1e-30
        slope = (y_grid[1] - y_grid[0]) / dx
        val = y_grid[0] + slope * (x - x_grid[0])
        return val, slope
    if x >= x_grid[n - 1]:
        dx = x_grid[n - 1] - x_grid[n - 2] + 1e-30
        slope = (y_grid[n-1] - y_grid[n-2]) / dx
        val = y_grid[n - 1] + slope * (x - x_grid[n - 1])
        return val, slope
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_grid[lo], x_grid[hi]
    y0, y1 = y_grid[lo], y_grid[hi]
    dx = x1 - x0
    if dx < 1e-30:
        return y0, 0.0
    slope = (y1 - y0) / dx
    val = y0 + slope * (x - x0)
    return val, slope


@njit(fastmath=True)
def find_bracket(x, grid):
    """Binary search for interpolation bracket on a sorted grid.

    Returns (iw, frac_w, inv_dw) where:
      grid[iw] <= x < grid[iw+1]
      frac_w = (x - grid[iw]) / dw
      inv_dw = 1.0 / dw
    Clamps to valid range: iw in [0, len(grid)-2].
    """
    n = len(grid)
    if x <= grid[0]:
        dw = grid[1] - grid[0] + 1e-30
        return 0, 0.0, 1.0 / dw
    if x >= grid[n - 1]:
        dw = grid[n - 1] - grid[n - 2] + 1e-30
        return n - 2, 1.0, 1.0 / dw
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    dw = grid[hi] - grid[lo]
    if dw < 1e-30:
        return lo, 0.0, 0.0
    frac = (x - grid[lo]) / dw
    return lo, frac, 1.0 / dw


@njit(fastmath=True)
def project_to_triangle(alpha_s, alpha_b):
    """Project (alpha_s, alpha_b) onto feasible region: >=0, sum<=1."""
    alpha_s = max(0.0, alpha_s)
    alpha_b = max(0.0, alpha_b)
    if alpha_s + alpha_b > 1.0:
        excess = (alpha_s + alpha_b - 1.0) * 0.5
        alpha_s = max(0.0, min(1.0, alpha_s - excess))
        alpha_b = max(0.0, min(1.0, alpha_b - excess))
    return alpha_s, alpha_b


@njit
def build_gross_return_arrays(mu_r_i, ret_nodes):
    """Build exp(mu_r + residual_node) arrays for one current financial state."""
    n_state = mu_r_i.shape[0]
    n_ret_quad = ret_nodes.shape[0]

    rx_stock_next = np.empty((n_state, n_ret_quad))
    rx_bond_next = np.empty((n_state, n_ret_quad))

    for j_s in range(n_state):
        mu_stock = mu_r_i[j_s, 0]
        mu_bond = mu_r_i[j_s, 1]
        for k_r in range(n_ret_quad):
            rx_stock_next[j_s, k_r] = exp(mu_stock + ret_nodes[k_r, 0])
            rx_bond_next[j_s, k_r] = exp(mu_bond + ret_nodes[k_r, 1])

    return rx_stock_next, rx_bond_next


def _project_to_triangle_py(alpha_s, alpha_b):
    """Python helper matching the constrained simplex geometry used in the solver."""
    alpha_s = max(0.0, alpha_s)
    alpha_b = max(0.0, alpha_b)
    if alpha_s + alpha_b > 1.0:
        excess = 0.5 * (alpha_s + alpha_b - 1.0)
        alpha_s = max(0.0, min(1.0, alpha_s - excess))
        alpha_b = max(0.0, min(1.0, alpha_b - excess))
    return alpha_s, alpha_b


def _terminal_prepare_scenarios(Pi_state_row, Rx_stock_next, Rx_bond_next, ret_weights, R_bill):
    """Scenario weights and gross returns for the exact terminal objective."""
    scenario_weights = np.asarray(Pi_state_row, dtype=float)[:, None] * np.asarray(ret_weights, dtype=float)[None, :]
    R_stock = float(R_bill) * np.asarray(Rx_stock_next, dtype=float)
    R_bond = float(R_bill) * np.asarray(Rx_bond_next, dtype=float)
    Rex_s = R_stock - float(R_bill)
    Rex_b = R_bond - float(R_bill)
    return scenario_weights, R_stock, R_bond, Rex_s, Rex_b


def _terminal_portfolio_moment(alpha_s, alpha_b, R_bill, scenario_weights, R_stock, R_bond, gamma):
    """Exact E[R_port^(1-gamma)] term on the terminal simplex domain."""
    alpha_bill = 1.0 - alpha_s - alpha_b
    R_port = alpha_s * R_stock + alpha_b * R_bond + alpha_bill * float(R_bill)
    bad = (scenario_weights > 0.0) & (R_port <= 0.0)
    if np.any(bad):
        return np.inf
    return float(np.sum(scenario_weights * np.power(R_port, 1.0 - gamma)))


def _terminal_portfolio_grad(alpha_s, alpha_b, R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma):
    """Exact gradient of E[R_port^(1-gamma)] with respect to (alpha_s, alpha_b)."""
    alpha_bill = 1.0 - alpha_s - alpha_b
    R_port = alpha_s * R_stock + alpha_b * R_bond + alpha_bill * float(R_bill)
    bad = (scenario_weights > 0.0) & (R_port <= 0.0)
    if np.any(bad):
        return np.array([np.nan, np.nan], dtype=float)
    coef = (1.0 - gamma) * scenario_weights * np.power(R_port, -gamma)
    return np.array([
        float(np.sum(coef * Rex_s)),
        float(np.sum(coef * Rex_b)),
    ], dtype=float)


def _terminal_portfolio_hess(alpha_s, alpha_b, R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma):
    """Exact Hessian of E[R_port^(1-gamma)] with respect to (alpha_s, alpha_b)."""
    alpha_bill = 1.0 - alpha_s - alpha_b
    R_port = alpha_s * R_stock + alpha_b * R_bond + alpha_bill * float(R_bill)
    bad = (scenario_weights > 0.0) & (R_port <= 0.0)
    if np.any(bad):
        return np.full((2, 2), np.nan, dtype=float)
    coef = gamma * (gamma - 1.0) * scenario_weights * np.power(R_port, -gamma - 1.0)
    h_ss = float(np.sum(coef * Rex_s * Rex_s))
    h_bb = float(np.sum(coef * Rex_b * Rex_b))
    h_sb = float(np.sum(coef * Rex_s * Rex_b))
    return np.array([[h_ss, h_sb], [h_sb, h_bb]], dtype=float)


def _terminal_projected_grad_residual(alpha_s, alpha_b, grad):
    """Projected-gradient residual for constrained minimization on the simplex."""
    proj_s, proj_b = _project_to_triangle_py(alpha_s - grad[0], alpha_b - grad[1])
    return float(np.hypot(alpha_s - proj_s, alpha_b - proj_b))


def _classify_terminal_solution(alpha_s, alpha_b, tol_geom):
    """Map terminal simplex location to the existing corner/edge/interior exit codes."""
    a_bill = 1.0 - alpha_s - alpha_b

    near_s0 = abs(alpha_s) <= tol_geom
    near_b0 = abs(alpha_b) <= tol_geom
    near_bill0 = abs(a_bill) <= tol_geom
    near_s1 = abs(alpha_s - 1.0) <= tol_geom
    near_b1 = abs(alpha_b - 1.0) <= tol_geom

    if near_s0 and near_b0:
        return EC_CORNER_BILLS
    if near_s1 and near_b0:
        return EC_CORNER_STOCKS
    if near_s0 and near_b1:
        return EC_CORNER_BONDS
    if near_b0:
        return EC_EDGE_SB
    if near_s0:
        return EC_EDGE_BB
    if near_bill0:
        return EC_EDGE_STOCKBOND
    return EC_INTERIOR


def solve_portfolio_2d_terminal_exact(i_s, Pi_state, Rx_stock_next, Rx_bond_next,
                                      ret_weights, R_bill, gamma,
                                      init_s=0.1, init_b=0.4,
                                      tol=1e-9, max_iter=200):
    """
    Exact constrained terminal portfolio solve.

    Minimizes E[R_port^(1-gamma)] over the portfolio simplex:
        alpha_s >= 0, alpha_b >= 0, alpha_s + alpha_b <= 1.

    The closed-form terminal consumption rule then uses the minimized moment.
    """
    scenario_weights, R_stock, R_bond, Rex_s, Rex_b = _terminal_prepare_scenarios(
        Pi_state[i_s, :], Rx_stock_next, Rx_bond_next, ret_weights, R_bill
    )

    def obj(x):
        return _terminal_portfolio_moment(x[0], x[1], R_bill, scenario_weights, R_stock, R_bond, gamma)

    def jac(x):
        return _terminal_portfolio_grad(x[0], x[1], R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma)

    def hess(x):
        return _terminal_portfolio_hess(x[0], x[1], R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma)

    bounds = Bounds([0.0, 0.0], [1.0, 1.0])
    simplex = LinearConstraint(np.array([[1.0, 1.0]], dtype=float), [-np.inf], [1.0])

    starts = [
        np.array([min(max(init_s, 1e-8), 0.999), min(max(init_b, 1e-8), 0.999)], dtype=float),
        np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float),
        np.array([0.80, 0.10], dtype=float),
        np.array([0.10, 0.80], dtype=float),
        np.array([0.05, 0.05], dtype=float),
    ]
    starts = [np.array(_project_to_triangle_py(x[0], x[1]), dtype=float) for x in starts]

    best = None
    best_moment = np.inf

    def consider_candidate(alpha_s, alpha_b):
        nonlocal best, best_moment
        alpha_s, alpha_b = _project_to_triangle_py(float(alpha_s), float(alpha_b))
        moment = _terminal_portfolio_moment(alpha_s, alpha_b, R_bill, scenario_weights, R_stock, R_bond, gamma)
        if not np.isfinite(moment):
            return
        grad = jac(np.array([alpha_s, alpha_b], dtype=float))
        resid = np.inf if not np.all(np.isfinite(grad)) else _terminal_projected_grad_residual(alpha_s, alpha_b, grad)
        exit_code = _classify_terminal_solution(alpha_s, alpha_b, tol_geom=max(10.0 * tol, 1e-8))
        cand = (alpha_s, alpha_b, moment, exit_code, resid)
        if moment < best_moment:
            best_moment = moment
            best = cand

    # Always include exact simplex corners.
    consider_candidate(0.0, 0.0)
    consider_candidate(1.0, 0.0)
    consider_candidate(0.0, 1.0)

    options_tc = {
        "gtol": tol,
        "xtol": tol,
        "barrier_tol": tol,
        "maxiter": max(100, max_iter),
        "verbose": 0,
    }

    for x0 in starts:
        res = minimize(
            obj, x0=x0, method="trust-constr", jac=jac, hess=hess,
            bounds=bounds, constraints=[simplex], options=options_tc
        )
        if np.all(np.isfinite(res.x)):
            consider_candidate(res.x[0], res.x[1])

    if best is None or best[4] > max(10.0 * tol, 1e-7):
        ineq = {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1], "jac": lambda x: np.array([-1.0, -1.0])}
        options_slsqp = {"ftol": tol, "maxiter": max(100, max_iter), "disp": False}
        for x0 in starts:
            res = minimize(
                obj, x0=x0, method="SLSQP", jac=jac,
                bounds=[(0.0, 1.0), (0.0, 1.0)], constraints=[ineq], options=options_slsqp
            )
            if np.all(np.isfinite(res.x)):
                consider_candidate(res.x[0], res.x[1])

    if best is None:
        return 0.0, 0.0, np.inf, EC_NEWTON_FAIL, np.inf

    return best


# =============================================================================
# FOC AND JACOBIAN -- RETIREMENT
# =============================================================================


@njit(fastmath=True)
def compute_foc_jac_retirement(alpha_s, alpha_b, s_val, z_idx, i_s,
                                wealth_grid, c_next_full, pension_next_scalar,
                                annuity_factor_is,
                                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                gamma, psi, beta, b_bar,
                             min_wealth_inv=1e-10, min_consumption=1e-10,
                             prob_skip=1e-12):
    a_bill     = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi

    foc_s = 0.0; foc_b = 0.0
    J_ss  = 0.0; J_bb  = 0.0; J_sb  = 0.0
    euler_sum = 0.0

    N_state = Pi_state.shape[1]
    n_ret_quad = len(ret_weights)

    for j_s in range(N_state):
        pi_s = Pi_state[i_s, j_s]
        if pi_s < prob_skip:
            continue

        c_row  = c_next_full[j_s, :]
        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            if p_ret < prob_skip:
                continue

            weight = pi_s * p_ret
            if weight < prob_skip:
                continue

            R_s = R_bill * Rx_stock_next[j_s, k_r]
            R_b = R_bill * Rx_bond_next[j_s, k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            w_inv  = max(s_val * R_p, min_wealth_inv)
            x_next = w_inv + pension_next_scalar

            c_next, mpc = fast_interp_1d_with_slope(x_next, wealth_grid, c_row)
            c_next = max(c_next, min_consumption)
            mpc = max(0.0, min(1.0, mpc))

            mu_alive  = c_next ** (-gamma)
            w_A        = w_inv / annuity_factor_is
            mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
            mu_comb    = psi * mu_alive + prob_death * mu_bequest

            mup_alive   = -gamma * mu_alive / c_next * mpc
            mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
            mup_comb    = psi * mup_alive + prob_death * mup_bequest

            wmu  = weight * mu_comb
            wmup = weight * mup_comb

            euler_sum += wmu * R_p
            foc_s     += wmu * Rex_s
            foc_b     += wmu * Rex_b

            jac   = wmup * s_val
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum


# =============================================================================
# NEWTON PORTFOLIO SOLVER -- RETIREMENT
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_2d_retirement(s_val, z_idx, i_s,
                                   wealth_grid, c_next_full, pension_next_scalar,
                                   annuity_factor_is,
                                   Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                   gamma, psi, beta, b_bar,
                                   init_s=0.1, init_b=0.4,
                                   tol=1e-7, max_iter=20,
                                   tiny_savings=1e-6, corner_tol=1e-8,
                                   edge_max_iter=8, edge_accept_factor=10.0,
                                   singular_det=1e-15, grad_step_size=0.05,
                                   step_damp=0.2, grad_denom_eps=1e-10,
                                   min_wealth_inv=1e-10, min_consumption=1e-10,
                                   prob_skip=1e-12):
    """2D Newton-Raphson for optimal (alpha_stock, alpha_bond) in retirement.
    Returns: (alpha_s, alpha_b, euler_sum, exit_code, foc_resid)"""

    # Tiny savings: all bills
    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_retirement(
            0.0, 0.0, s_val, z_idx, i_s,
            wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0

    # Corner: all bills
    fs0, fb0, _, _, _, e0 = compute_foc_jac_retirement(
        0.0, 0.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)  # FOC scale for relative tolerance
    if fs0 <= corner_tol * scale and fb0 <= corner_tol * scale:
        return 0.0, 0.0, e0, EC_CORNER_BILLS, 0.0

    # Corner: all stocks
    fs1, fb1, _, _, _, e1 = compute_foc_jac_retirement(
        1.0, 0.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
    if fs1 >= -corner_tol * scale and fb1 <= fs1 + corner_tol * scale:
        return 1.0, 0.0, e1, EC_CORNER_STOCKS, 0.0

    # Corner: all bonds
    fs2, fb2, _, _, _, e2 = compute_foc_jac_retirement(
        0.0, 1.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
    if fb2 >= -corner_tol * scale and fs2 <= fb2 + corner_tol * scale:
        return 0.0, 1.0, e2, EC_CORNER_BONDS, 0.0

    # Edge: stocks + bills only (alpha_b = 0)
    if fs0 > 0.0 and fs1 < 0.0:
        a_s = fs0 / (fs0 - fs1)
        fs = fs0  # init for residual tracking
        for _ in range(edge_max_iter):
            fs, fb, Jss, _, _, e = compute_foc_jac_retirement(
                a_s, 0.0, s_val, z_idx, i_s,
                wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
            if abs(fs) < tol * scale:
                break
            if abs(Jss) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - fs / Jss))
        if abs(fs) < tol * scale * edge_accept_factor and fb <= tol * scale:
            return a_s, 0.0, e, EC_EDGE_SB, abs(fs) / scale

    # Edge: bonds + bills only (alpha_s = 0)
    if fb0 > 0.0 and fb2 < 0.0:
        a_b = fb0 / (fb0 - fb2)
        fb = fb0
        for _ in range(edge_max_iter):
            fs, fb, _, Jbb, _, e = compute_foc_jac_retirement(
                0.0, a_b, s_val, z_idx, i_s,
                wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
            if abs(fb) < tol * scale:
                break
            if abs(Jbb) < singular_det:
                break
            a_b = max(0.0, min(1.0, a_b - fb / Jbb))
        if abs(fb) < tol * scale * edge_accept_factor and fs <= tol * scale:
            return 0.0, a_b, e, EC_EDGE_BB, abs(fb) / scale

    # Edge: stocks + bonds only (no bills, alpha_s + alpha_b = 1)
    g1 = fs1 - fb1
    g2 = fs2 - fb2
    if g1 * g2 < 0.0:
        a_s = g2 / (g2 - g1)
        g = g2  # init
        for _ in range(edge_max_iter):
            a_b = 1.0 - a_s
            fs, fb, Jss, Jbb, Jsb, e = compute_foc_jac_retirement(
                a_s, a_b, s_val, z_idx, i_s,
                wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
            g = fs - fb
            if abs(g) < tol * scale:
                break
            dg = Jss - 2.0 * Jsb + Jbb
            if abs(dg) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - g / dg))
        if abs(fs - fb) < tol * scale * edge_accept_factor and fs >= -tol * scale:
            return a_s, 1.0 - a_s, e, EC_EDGE_STOCKBOND, abs(g) / scale

    # Interior Newton-Raphson
    a_s = init_s
    a_b = init_b
    e_last = 0.0
    err = 1.0

    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb, e_sum = compute_foc_jac_retirement(
            a_s, a_b, s_val, z_idx, i_s,
            wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        e_last = e_sum

        err = (fs * fs + fb * fb) ** 0.5
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d  = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        slen = (step_s * step_s + step_b * step_b) ** 0.5
        if slen > step_damp:
            sc = step_damp / slen
            step_s *= sc
            step_b *= sc

        a_s, a_b = project_to_triangle(a_s + step_s, a_b + step_b)

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale

# =============================================================================
# UNCONSTRAINED PORTFOLIO SOLVER -- RETIREMENT
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_unconstrained_retirement(s_val, z_idx, i_s,
                                              wealth_grid, c_next_full, pension_next_scalar,
                                              annuity_factor_is,
                                              Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                              gamma, psi, beta, b_bar,
                                              init_s=0.1, init_b=0.4,
                                              tol=1e-7, max_iter=30,
                                              tiny_savings=1e-6,
                                              singular_det=1e-15, grad_step_size=0.05,
                                              step_damp=0.3, grad_denom_eps=1e-10,
                                              min_wealth_inv=1e-10, min_consumption=1e-10,
                                              prob_skip=1e-12,
                                              use_line_search=True, max_backtrack_iter=10,
                                              line_search_max_step=2.0):
    """Unconstrained Newton for (alpha_stock, alpha_bond) in retirement.
    No short-sale or leverage constraints."""

    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_retirement(
            0.0, 0.0, s_val, z_idx, i_s,
            wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0

    # Scale from all-bills FOC
    _, _, _, _, _, e0 = compute_foc_jac_retirement(
        0.0, 0.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)

    a_s = init_s
    a_b = init_b

    # Initial evaluation outside loop so trial-point eval is reused each iteration
    fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_retirement(
        a_s, a_b, s_val, z_idx, i_s,
        wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    err = (fs * fs + fb * fb) ** 0.5

    for _ in range(max_iter):
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d  = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        if use_line_search:
            # Cap raw step then backtrack to find alpha that reduces residual
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > line_search_max_step:
                cap = line_search_max_step / slen
                step_s *= cap
                step_b *= cap

            alpha = 1.0
            found = False
            fs_new = fs; fb_new = fb; Jss_new = Jss; Jbb_new = Jbb; Jsb_new = Jsb
            e_new = e_last; err_new = err
            for _bt in range(max_backtrack_iter):
                a_s_t = a_s + alpha * step_s
                a_b_t = a_b + alpha * step_b
                fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = compute_foc_jac_retirement(
                    a_s_t, a_b_t, s_val, z_idx, i_s,
                    wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                    Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
                    min_wealth_inv, min_consumption, prob_skip)
                err_t = (fs_t * fs_t + fb_t * fb_t) ** 0.5
                if err_t < err:
                    fs_new = fs_t; fb_new = fb_t; Jss_new = Jss_t; Jbb_new = Jbb_t; Jsb_new = Jsb_t
                    e_new = e_t; err_new = err_t
                    a_s = a_s_t; a_b = a_b_t
                    found = True
                    break
                alpha *= 0.5

            if found:
                fs = fs_new; fb = fb_new; Jss = Jss_new; Jbb = Jbb_new; Jsb = Jsb_new
                e_last = e_new; err = err_new
            # else: no decrease found — keep current point, J, err unchanged
        else:
            # Original behaviour: clip step and apply blindly
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > step_damp:
                cap = step_damp / slen
                step_s *= cap
                step_b *= cap
            a_s += step_s
            a_b += step_b
            fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_retirement(
                a_s, a_b, s_val, z_idx, i_s,
                wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            err = (fs * fs + fb * fb) ** 0.5

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale


# =============================================================================
# FOC AND JACOBIAN -- WORKING AGE
# =============================================================================

@njit(fastmath=True)
def compute_foc_jac_working(alpha_s, alpha_b, s_val, z_idx, i_s,
                             wealth_grid, c_next_full, income_next_table,
                             annuity_factor_is,
                             z_grid, rho, eta_nodes, eta_weights, dz,
                             Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                             eps_nodes, eps_weights,
                             gamma, psi, beta, b_bar,
                             min_wealth_inv=1e-10, min_consumption=1e-10,
                             prob_skip=1e-12):
    a_bill     = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi

    foc_s = 0.0; foc_b = 0.0
    J_ss  = 0.0; J_bb  = 0.0; J_sb  = 0.0
    euler_sum = 0.0

    N_state = Pi_state.shape[1]
    n_z     = len(z_grid)
    n_eta   = len(eta_nodes)
    n_eps   = len(eps_nodes)
    n_ret_quad = len(ret_weights)

    for j_s in range(N_state):
        p_var = Pi_state[i_s, j_s]
        if p_var < prob_skip:
            continue

        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            if p_ret < prob_skip:
                continue

            p_state_ret = p_var * p_ret
            if p_state_ret < prob_skip:
                continue

            R_s = R_bill * Rx_stock_next[j_s, k_r]
            R_b = R_bill * Rx_bond_next[j_s, k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            w_inv = max(s_val * R_p, min_wealth_inv)

            w_A         = w_inv / annuity_factor_is
            mu_bequest  = b_bar * w_A ** (-gamma) / annuity_factor_is
            mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)

            # -- bequest contribution: once per (j_s, k_r), independent of income --
            death_mu  = p_state_ret * prob_death * mu_bequest
            death_mup = p_state_ret * prob_death * mup_bequest

            euler_sum += death_mu * R_p
            foc_s     += death_mu * Rex_s
            foc_b     += death_mu * Rex_b

            jac_b  = death_mup * s_val
            J_ss  += jac_b * Rex_s * Rex_s
            J_bb  += jac_b * Rex_b * Rex_b
            J_sb  += jac_b * Rex_s * Rex_b

            # -- alive contribution: quadrature over persistent and transitory innovations --
            for k_eta in range(n_eta):
                w_eta = eta_weights[k_eta]
                if w_eta < prob_skip:
                    continue

                # Next-period z (continuous, generally between grid points)
                z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]

                # Bracket z_next on the uniform z_grid
                iz_lo = int((z_next - z_grid[0]) / dz)
                iz_lo = max(0, min(iz_lo, n_z - 2))
                frac_z = (z_next - z_grid[iz_lo]) / dz
                frac_z = max(0.0, min(1.0, frac_z))

                p_out_base = p_state_ret * w_eta

                for i_e in range(n_eps):
                    weight = p_out_base * eps_weights[i_e]

                    # Interpolate income in z
                    income_next = ((1.0 - frac_z) * income_next_table[iz_lo, i_e]
                                   + frac_z * income_next_table[iz_lo + 1, i_e])
                    x_next = w_inv + income_next

                    # One bracket search, two direct reads
                    iw, frac_w, inv_dw = find_bracket(x_next, wealth_grid)

                    c_lo = (1.0 - frac_w) * c_next_full[iz_lo, j_s, iw] + frac_w * c_next_full[iz_lo, j_s, iw + 1]
                    c_hi = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]

                    mpc_lo = (c_next_full[iz_lo, j_s, iw + 1] - c_next_full[iz_lo, j_s, iw]) * inv_dw
                    mpc_hi = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw

                    c_next = (1.0 - frac_z) * c_lo + frac_z * c_hi
                    c_next = max(c_next, min_consumption)
                    mpc = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi
                    mpc = max(0.0, min(1.0, mpc))

                    mu_alive  = c_next ** (-gamma)
                    mup_alive = -gamma * mu_alive / c_next * mpc

                    wmu  = weight * psi * mu_alive
                    wmup = weight * psi * mup_alive

                    euler_sum += wmu * R_p
                    foc_s     += wmu * Rex_s
                    foc_b     += wmu * Rex_b

                    jac = wmup * s_val
                    J_ss += jac * Rex_s * Rex_s
                    J_bb += jac * Rex_b * Rex_b
                    J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum

# =============================================================================
# NEWTON PORTFOLIO SOLVER -- WORKING AGE
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_2d_working(s_val, z_idx, i_s,
                                wealth_grid, c_next_full, income_next_table,
                                annuity_factor_is,
                                z_grid, rho, eta_nodes, eta_weights, dz,
                                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                eps_nodes, eps_weights,
                                gamma, psi, beta, b_bar,
                                init_s=0.1, init_b=0.4,
                                tol=1e-7, max_iter=20,
                                   tiny_savings=1e-6, corner_tol=1e-8,
                                   edge_max_iter=8, edge_accept_factor=10.0,
                                   singular_det=1e-15, grad_step_size=0.05,
                                   step_damp=0.2, grad_denom_eps=1e-10,
                                   min_wealth_inv=1e-10, min_consumption=1e-10,
                                   prob_skip=1e-12):
    """2D Newton-Raphson for optimal (alpha_stock, alpha_bond) during working years.
    Returns: (alpha_s, alpha_b, euler_sum, exit_code, foc_resid)"""

    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_working(
            0.0, 0.0, s_val, z_idx, i_s,
            wealth_grid, c_next_full, income_next_table, annuity_factor_is,
            z_grid, rho, eta_nodes, eta_weights, dz,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
            eps_nodes, eps_weights, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0

    # Corner: all bills
    fs0, fb0, _, _, _, e0 = compute_foc_jac_working(
        0.0, 0.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, income_next_table, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
        eps_nodes, eps_weights, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)  # FOC scale for relative tolerance
    if fs0 <= corner_tol * scale and fb0 <= corner_tol * scale:
        return 0.0, 0.0, e0, EC_CORNER_BILLS, 0.0

    # Corner: all stocks
    fs1, fb1, _, _, _, e1 = compute_foc_jac_working(
        1.0, 0.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, income_next_table, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
        eps_nodes, eps_weights, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
    if fs1 >= -corner_tol * scale and fb1 <= fs1 + corner_tol * scale:
        return 1.0, 0.0, e1, EC_CORNER_STOCKS, 0.0

    # Corner: all bonds
    fs2, fb2, _, _, _, e2 = compute_foc_jac_working(
        0.0, 1.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, income_next_table, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
        eps_nodes, eps_weights, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
    if fb2 >= -corner_tol * scale and fs2 <= fb2 + corner_tol * scale:
        return 0.0, 1.0, e2, EC_CORNER_BONDS, 0.0

    # Edge: stocks + bills only (alpha_b = 0)
    if fs0 > 0.0 and fs1 < 0.0:
        a_s = fs0 / (fs0 - fs1)
        fs = fs0
        for _ in range(edge_max_iter):
            fs, fb, Jss, _, _, e = compute_foc_jac_working(
                a_s, 0.0, s_val, z_idx, i_s,
                wealth_grid, c_next_full, income_next_table, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            if abs(fs) < tol * scale:
                break
            if abs(Jss) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - fs / Jss))
        if abs(fs) < tol * scale * edge_accept_factor and fb <= tol * scale:
            return a_s, 0.0, e, EC_EDGE_SB, abs(fs) / scale

    # Edge: bonds + bills only (alpha_s = 0)
    if fb0 > 0.0 and fb2 < 0.0:
        a_b = fb0 / (fb0 - fb2)
        fb = fb0
        for _ in range(edge_max_iter):
            fs, fb, _, Jbb, _, e = compute_foc_jac_working(
                0.0, a_b, s_val, z_idx, i_s,
                wealth_grid, c_next_full, income_next_table, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            if abs(fb) < tol * scale:
                break
            if abs(Jbb) < singular_det:
                break
            a_b = max(0.0, min(1.0, a_b - fb / Jbb))
        if abs(fb) < tol * scale * edge_accept_factor and fs <= tol * scale:
            return 0.0, a_b, e, EC_EDGE_BB, abs(fb) / scale

    # Edge: stocks + bonds only (no bills, alpha_s + alpha_b = 1)
    g1 = fs1 - fb1
    g2 = fs2 - fb2
    if g1 * g2 < 0.0:
        a_s = g2 / (g2 - g1)
        g = g2
        for _ in range(edge_max_iter):
            a_b = 1.0 - a_s
            fs, fb, Jss, Jbb, Jsb, e = compute_foc_jac_working(
                a_s, a_b, s_val, z_idx, i_s,
                wealth_grid, c_next_full, income_next_table, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            g = fs - fb
            if abs(g) < tol * scale:
                break
            dg = Jss - 2.0 * Jsb + Jbb
            if abs(dg) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - g / dg))
        if abs(fs - fb) < tol * scale * edge_accept_factor and fs >= -tol * scale:
            return a_s, 1.0 - a_s, e, EC_EDGE_STOCKBOND, abs(g) / scale

    # Interior Newton-Raphson
    a_s = init_s
    a_b = init_b
    e_last = 0.0
    err = 1.0

    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb, e_sum = compute_foc_jac_working(
            a_s, a_b, s_val, z_idx, i_s,
            wealth_grid, c_next_full, income_next_table, annuity_factor_is,
            z_grid, rho, eta_nodes, eta_weights, dz,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
            eps_nodes, eps_weights, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        e_last = e_sum

        err = (fs * fs + fb * fb) ** 0.5
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d  = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        slen = (step_s * step_s + step_b * step_b) ** 0.5
        if slen > step_damp:
            sc = step_damp / slen
            step_s *= sc
            step_b *= sc

        a_s, a_b = project_to_triangle(a_s + step_s, a_b + step_b)

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale


# =============================================================================
# UNCONSTRAINED PORTFOLIO SOLVER -- TERMINAL
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_unconstrained_terminal(i_s, Pi_state, Rx_stock_next, Rx_bond_next,
                                            ret_weights, R_bill, gamma,
                                            init_s=0.1, init_b=0.4,
                                            tol=1e-7, max_iter=30,
                                            singular_det=1e-15, grad_step_size=0.05,
                                            step_damp=0.3, grad_denom_eps=1e-10,
                                            min_return_power=1e-15,
                                   prob_skip=1e-12):
    """Unconstrained Newton for terminal portfolio (Numba, legacy).
    No short-sale or leverage constraints.
    Returns: (alpha_s, alpha_b, exit_code, foc_resid)"""

    scale = R_bill ** (-gamma)

    a_s = init_s
    a_b = init_b
    err = 1.0

    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb = compute_terminal_portfolio_foc_jac(
            a_s, a_b, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)

        err = (fs * fs + fb * fb) ** 0.5
        if err < tol * scale:
            return a_s, a_b, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d  = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        slen = (step_s * step_s + step_b * step_b) ** 0.5
        if slen > step_damp:
            sc = step_damp / slen
            step_s *= sc
            step_b *= sc

        a_s += step_s
        a_b += step_b

    return a_s, a_b, EC_NEWTON_FAIL, err / scale


def solve_portfolio_unconstrained_terminal_exact(i_s, Pi_state, Rx_stock_next, Rx_bond_next,
                                                  ret_weights, R_bill, gamma,
                                                  init_s=0.1, init_b=0.4,
                                                  tol=1e-9, max_iter=200):
    """
    Exact unconstrained terminal portfolio solve.

    Minimizes E[R_port^(1-gamma)] over R^2 (no short-sale or leverage constraints).
    Uses scipy trust-region optimizer with exact Hessian and multi-start.

    The Hessian is guaranteed PSD in the feasible interior (where R_port > 0
    for all positive-weight scenarios), so trust-ncg converges reliably even
    in the ill-conditioned high-leverage region where the Numba Newton stalls.

    Returns: (alpha_s, alpha_b, moment, exit_code, foc_resid)
    """
    scenario_weights, R_stock, R_bond, Rex_s, Rex_b = _terminal_prepare_scenarios(
        Pi_state[i_s, :], Rx_stock_next, Rx_bond_next, ret_weights, R_bill
    )

    def obj(x):
        return _terminal_portfolio_moment(x[0], x[1], R_bill, scenario_weights, R_stock, R_bond, gamma)

    def jac(x):
        return _terminal_portfolio_grad(x[0], x[1], R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma)

    def hess(x):
        return _terminal_portfolio_hess(x[0], x[1], R_bill, scenario_weights, R_stock, R_bond, Rex_s, Rex_b, gamma)

    starts = [
        np.array([init_s, init_b], dtype=float),
        np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float),
        np.array([0.6, 0.2], dtype=float),
        np.array([1.0, -0.5], dtype=float),
        np.array([1.5, -1.0], dtype=float),
        np.array([2.0, -2.0], dtype=float),
    ]

    best = None
    best_moment = np.inf

    def consider_candidate(alpha_s, alpha_b):
        nonlocal best, best_moment
        alpha_s, alpha_b = float(alpha_s), float(alpha_b)
        moment = _terminal_portfolio_moment(alpha_s, alpha_b, R_bill, scenario_weights, R_stock, R_bond, gamma)
        if not np.isfinite(moment):
            return
        grad = jac(np.array([alpha_s, alpha_b], dtype=float))
        resid = np.inf if not np.all(np.isfinite(grad)) else float(np.hypot(grad[0], grad[1]))
        cand = (alpha_s, alpha_b, moment, EC_INTERIOR, resid)
        if moment < best_moment:
            best_moment = moment
            best = cand

    options_ncg = {
        "gtol": tol,
        "maxiter": max(100, max_iter),
    }

    for x0 in starts:
        if not np.isfinite(obj(x0)):
            continue
        try:
            res = minimize(obj, x0=x0, method="trust-ncg", jac=jac, hess=hess,
                           options=options_ncg)
            if np.all(np.isfinite(res.x)):
                consider_candidate(res.x[0], res.x[1])
        except Exception:
            continue

    # BFGS fallback if trust-ncg failed for all starts.
    if best is None or best[4] > max(10.0 * tol, 1e-7):
        options_bfgs = {"gtol": tol, "maxiter": max(200, 2 * max_iter)}
        for x0 in starts:
            if not np.isfinite(obj(x0)):
                continue
            try:
                res = minimize(obj, x0=x0, method="BFGS", jac=jac, options=options_bfgs)
                if np.all(np.isfinite(res.x)):
                    consider_candidate(res.x[0], res.x[1])
            except Exception:
                continue

    if best is None:
        return 0.0, 0.0, np.inf, EC_NEWTON_FAIL, np.inf

    return best

# =============================================================================
# TERMINAL CONDITION
# =============================================================================

# =============================================================================
# UNCONSTRAINED PORTFOLIO SOLVER -- WORKING AGE
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_unconstrained_working(s_val, z_idx, i_s,
                                           wealth_grid, c_next_full, income_next_table,
                                           annuity_factor_is,
                                           z_grid, rho, eta_nodes, eta_weights, dz,
                                           Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                           eps_nodes, eps_weights,
                                           gamma, psi, beta, b_bar,
                                           init_s=0.1, init_b=0.4,
                                           tol=1e-7, max_iter=30,
                                           tiny_savings=1e-6,
                                           singular_det=1e-15, grad_step_size=0.05,
                                           step_damp=0.3, grad_denom_eps=1e-10,
                                           min_wealth_inv=1e-10, min_consumption=1e-10,
                                           prob_skip=1e-12,
                                           use_line_search=True, max_backtrack_iter=10,
                                           line_search_max_step=2.0):
    """Unconstrained Newton for (alpha_stock, alpha_bond) at working age.
    No short-sale or leverage constraints."""

    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_working(
            0.0, 0.0, s_val, z_idx, i_s,
            wealth_grid, c_next_full, income_next_table, annuity_factor_is,
            z_grid, rho, eta_nodes, eta_weights, dz,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
            eps_nodes, eps_weights, gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0

    # Scale from all-bills FOC
    _, _, _, _, _, e0 = compute_foc_jac_working(
        0.0, 0.0, s_val, z_idx, i_s,
        wealth_grid, c_next_full, income_next_table, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
        eps_nodes, eps_weights, gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)

    a_s = init_s
    a_b = init_b

    # Initial evaluation outside loop so trial-point eval is reused each iteration
    fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_working(
        a_s, a_b, s_val, z_idx, i_s,
        wealth_grid, c_next_full, income_next_table, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
        eps_nodes, eps_weights, gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    err = (fs * fs + fb * fb) ** 0.5

    for _ in range(max_iter):
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d  = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        if use_line_search:
            # Cap raw step then backtrack to find alpha that reduces residual
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > line_search_max_step:
                cap = line_search_max_step / slen
                step_s *= cap
                step_b *= cap

            alpha = 1.0
            found = False
            fs_new = fs; fb_new = fb; Jss_new = Jss; Jbb_new = Jbb; Jsb_new = Jsb
            e_new = e_last; err_new = err
            for _bt in range(max_backtrack_iter):
                a_s_t = a_s + alpha * step_s
                a_b_t = a_b + alpha * step_b
                fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = compute_foc_jac_working(
                    a_s_t, a_b_t, s_val, z_idx, i_s,
                    wealth_grid, c_next_full, income_next_table, annuity_factor_is,
                    z_grid, rho, eta_nodes, eta_weights, dz,
                    Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                    eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                    min_wealth_inv, min_consumption, prob_skip)
                err_t = (fs_t * fs_t + fb_t * fb_t) ** 0.5
                if err_t < err:
                    fs_new = fs_t; fb_new = fb_t; Jss_new = Jss_t; Jbb_new = Jbb_t; Jsb_new = Jsb_t
                    e_new = e_t; err_new = err_t
                    a_s = a_s_t; a_b = a_b_t
                    found = True
                    break
                alpha *= 0.5

            if found:
                fs = fs_new; fb = fb_new; Jss = Jss_new; Jbb = Jbb_new; Jsb = Jsb_new
                e_last = e_new; err = err_new
            # else: no decrease found — keep current point, J, err unchanged
        else:
            # Original behaviour: clip step and apply blindly
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > step_damp:
                cap = step_damp / slen
                step_s *= cap
                step_b *= cap
            a_s += step_s
            a_b += step_b
            fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_working(
                a_s, a_b, s_val, z_idx, i_s,
                wealth_grid, c_next_full, income_next_table, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            err = (fs * fs + fb * fb) ** 0.5

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale


# =============================================================================
# TERMINAL AGE PORTFOLIO SOLVER
# =============================================================================

@njit(fastmath=True)
def compute_terminal_portfolio_foc_jac(alpha_s, alpha_b, i_s,
                                        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights,
                                        R_bill, gamma,
                                          min_return_power=1e-15,
                                          prob_skip=1e-12):
    """
    FOC and Jacobian for the terminal-period portfolio problem.

    Because the bequest b(a*R_port, A) = b_bar*(a*R_port/A)^{1-gamma}/(1-gamma)
    is CRRA in terminal wealth, the portfolio FOC is proportional to a^{1-gamma}
    and therefore independent of savings a (hence independent of W and c).

    FOC_k = sum_j pi(j|i) * R_port(j)^{-gamma} * (R_k(j) - R_bill) = 0,  k in {s,b}
    J_kl  = sum_j pi(j|i) * (-gamma) * R_port(j)^{-gamma-1} * Rex_k * Rex_l
    """
    foc_s = 0.0;  foc_b = 0.0
    J_ss  = 0.0;  J_bb  = 0.0;  J_sb  = 0.0
    a_bill = 1.0 - alpha_s - alpha_b

    N_state = Pi_state.shape[1]
    n_ret_quad = len(ret_weights)
    for j_s in range(N_state):
        pi_s = Pi_state[i_s, j_s]
        if pi_s < prob_skip:
            continue
        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            if p_ret < prob_skip:
                continue
            weight = pi_s * p_ret
            if weight < prob_skip:
                continue

            R_s   = R_bill * Rx_stock_next[j_s, k_r]
            R_b   = R_bill * Rx_bond_next[j_s, k_r]
            R_p   = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            Rp_mg  = max(R_p, min_return_power) ** (-gamma)
            Rp_mg1 = max(R_p, min_return_power) ** (-gamma - 1.0)

            foc_s += weight * Rp_mg * Rex_s
            foc_b += weight * Rp_mg * Rex_b

            jac   = weight * (-gamma) * Rp_mg1
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb


@njit(fastmath=True)
def solve_portfolio_2d_terminal(i_s, Pi_state, Rx_stock_next, Rx_bond_next,
                                 ret_weights, R_bill, gamma,
                                 init_s=0.1, init_b=0.4, tol=1e-7, max_iter=20,
                                    tiny_savings=1e-6, corner_tol=1e-8,
                                    edge_max_iter=8, edge_accept_factor=10.0,
                                    singular_det=1e-15, grad_step_size=0.05,
                                    step_damp=0.2, grad_denom_eps=1e-10,
                                    min_return_power=1e-15,
                                   prob_skip=1e-12):
    """2D Newton-Raphson for optimal (alpha_s, alpha_b) at the terminal age.
    Returns: (alpha_s, alpha_b, exit_code, foc_resid)"""

    scale = R_bill ** (-gamma)  # FOC scale for relative tolerance

    # Corner: all bills
    fs0, fb0, _, _, _ = compute_terminal_portfolio_foc_jac(
        0.0, 0.0, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
    if fs0 <= corner_tol * scale and fb0 <= corner_tol * scale:
        return 0.0, 0.0, EC_CORNER_BILLS, 0.0

    # Corner: all stocks
    fs1, fb1, _, _, _ = compute_terminal_portfolio_foc_jac(
        1.0, 0.0, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
    if fs1 >= -corner_tol * scale and fb1 <= fs1 + corner_tol * scale:
        return 1.0, 0.0, EC_CORNER_STOCKS, 0.0

    # Corner: all bonds
    fs2, fb2, _, _, _ = compute_terminal_portfolio_foc_jac(
        0.0, 1.0, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
    if fb2 >= -corner_tol * scale and fs2 <= fb2 + corner_tol * scale:
        return 0.0, 1.0, EC_CORNER_BONDS, 0.0

    # Edge: stocks + bills only (alpha_b = 0)
    if fs0 > 0.0 and fs1 < 0.0:
        a_s = fs0 / (fs0 - fs1)
        fs = fs0
        for _ in range(edge_max_iter):
            fs, fb, Jss, _, _ = compute_terminal_portfolio_foc_jac(
                a_s, 0.0, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
            if abs(fs) < tol * scale:
                break
            if abs(Jss) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - fs / Jss))
        if abs(fs) < tol * scale * edge_accept_factor and fb <= tol * scale:
            return a_s, 0.0, EC_EDGE_SB, abs(fs) / scale

    # Edge: bonds + bills only (alpha_s = 0)
    if fb0 > 0.0 and fb2 < 0.0:
        a_b = fb0 / (fb0 - fb2)
        fb = fb0
        for _ in range(edge_max_iter):
            fs, fb, _, Jbb, _ = compute_terminal_portfolio_foc_jac(
                0.0, a_b, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
            if abs(fb) < tol * scale:
                break
            if abs(Jbb) < singular_det:
                break
            a_b = max(0.0, min(1.0, a_b - fb / Jbb))
        if abs(fb) < tol * scale * edge_accept_factor and fs <= tol * scale:
            return 0.0, a_b, EC_EDGE_BB, abs(fb) / scale

    # Edge: stocks + bonds only (no bills, alpha_s + alpha_b = 1)
    g1 = fs1 - fb1
    g2 = fs2 - fb2
    if g1 * g2 < 0.0:
        a_s = g2 / (g2 - g1)
        g = g2
        for _ in range(edge_max_iter):
            a_b = 1.0 - a_s
            fs, fb, Jss, Jbb, Jsb = compute_terminal_portfolio_foc_jac(
                a_s, a_b, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
            g = fs - fb
            if abs(g) < tol * scale:
                break
            dg = Jss - 2.0 * Jsb + Jbb
            if abs(dg) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - g / dg))
        if abs(fs - fb) < tol * scale * edge_accept_factor and fs >= -tol * scale:
            return a_s, 1.0 - a_s, EC_EDGE_STOCKBOND, abs(g) / scale

    # Interior Newton-Raphson
    a_s = init_s
    a_b = init_b
    err = 1.0
    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb = compute_terminal_portfolio_foc_jac(
            a_s, a_b, i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, min_return_power, prob_skip)
        err = (fs * fs + fb * fb) ** 0.5
        if err < tol * scale:
            return a_s, a_b, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d  = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        slen = (step_s * step_s + step_b * step_b) ** 0.5
        if slen > step_damp:
            sc    = step_damp / slen
            step_s *= sc
            step_b *= sc

        a_s, a_b = project_to_triangle(a_s + step_s, a_b + step_b)

    return a_s, a_b, EC_NEWTON_FAIL, err / scale


def solve_terminal_age(wealth_grid, annuity_factors, r_bill_grid, Pi_state, mu_r,
                       ret_nodes, ret_weights,
                       gamma, beta, b_bar, N_state, n_z, constrained=True, solver_config=None,
                       min_return_power=1e-15, min_consumption=1e-10):
    """Solve the terminal age exactly on the constrained portfolio simplex."""
    if solver_config is None:
        solver_config = SolverConfig()

    n_w = len(wealth_grid)
    out_c = np.empty((n_z, N_state, n_w))
    out_alpha_s = np.empty((n_z, N_state, n_w))
    out_alpha_b = np.empty((n_z, N_state, n_w))
    terminal_diag_int = np.zeros(N_state, dtype=np.int64)

    for i_s in range(N_state):
        R_bill = exp(r_bill_grid[i_s])
        A_is = annuity_factors[i_s]
        Rx_stock_next, Rx_bond_next = build_gross_return_arrays(mu_r[i_s, :, :], ret_nodes)

        if constrained:
            opt_s, opt_b, moment, exit_code, foc_resid = solve_portfolio_2d_terminal_exact(
                i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=max(100, 20 * solver_config.max_iter),
            )
        else:
            opt_s, opt_b, moment, exit_code, foc_resid = solve_portfolio_unconstrained_terminal_exact(
                i_s, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=max(100, 20 * solver_config.max_iter),
            )

        terminal_diag_int[i_s] = exit_code

        if np.isfinite(moment) and moment > 0.0:
            omega = b_bar * A_is ** (gamma - 1.0) * moment
            ratio = (beta * omega) ** (-1.0 / gamma)
            c_vec = np.maximum(wealth_grid * ratio / (ratio + 1.0), min_consumption)
        else:
            c_vec = np.maximum(wealth_grid, min_consumption)
        out_c[:, i_s, :] = c_vec[None, :]
        out_alpha_s[:, i_s, :] = opt_s
        out_alpha_b[:, i_s, :] = opt_b

    return out_c, out_alpha_s, out_alpha_b, terminal_diag_int


# =============================================================================
# PERIOD SOLVER -- RETIREMENT
# =============================================================================

@njit(parallel=True)
def _solve_retirement_step_jit(wealth_grid, savings_grid, z_grid, N_state,
                               c_next_full, pension_1d,
                               annuity_factors, Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                               gamma, psi_vec, beta, b_bar,
                               constrained, solver_config):
    """
    Solve one retirement period using EGM + 2D Newton.
    Parallelised over financial state i_s (prange).

    Parameters
    ----------
    c_next_full  : (n_z, N_state, n_w)  consumption policy at t+1
    pension_1d   : (n_z,)               after-tax pension at t+1
    annuity_factors : (N_state,)         A(y_nom * 4, b_bar) per state
    mu_r         : (N_state, N_state, 2) conditional return means
    r_bill_grid  : (N_state,)            log real bill rate per state

    Returns: policy_c, policy_alpha_s, policy_alpha_b -- each (n_z, N_state, n_w)
             diag_int (N_state, N_DIAG_INT), diag_float (N_state, N_DIAG_FLOAT)
    """

    sc = solver_config

    n_z      = len(z_grid)
    n_savings = len(savings_grid)
    n_wealth  = len(wealth_grid)

    policy_c       = np.empty((n_z, N_state, n_wealth))
    policy_alpha_s = np.empty((n_z, N_state, n_wealth))
    policy_alpha_b = np.empty((n_z, N_state, n_wealth))
    diag_int   = np.zeros((N_state, 13), dtype=np.int64)
    diag_float = np.zeros((N_state, 9))

    for i_s in prange(N_state):
        R_bill = exp(r_bill_grid[i_s])
        annuity_factor_is = annuity_factors[i_s]

        # Pre-compute gross excess returns for all next states (avoids re-exp in Newton)
        Rx_stock_next, Rx_bond_next = build_gross_return_arrays(mu_r[i_s, :, :], ret_nodes)

        last_a_s = sc.init_alpha_s
        last_a_b = sc.init_alpha_b

        # Init min/max trackers for this i_s
        diag_float[i_s, 3] = 2.0   # DF_MIN_ALPHA_S (init high)
        diag_float[i_s, 5] = 2.0   # DF_MIN_ALPHA_B (init high)

        # Scratch buffers reused across z_i (size invariant; fully overwritten each z_i)
        temp_x = np.empty(n_savings + 1)
        temp_c = np.empty(n_savings + 1)
        temp_s = np.empty(n_savings + 1)
        temp_b = np.empty(n_savings + 1)

        for z_i in range(n_z):
            psi = psi_vec[z_i]
            c_next_slice    = c_next_full[z_i, :, :]  # (N_state, n_w)
            pension_next    = pension_1d[z_i]

            # Anchor at zero savings
            temp_x[0] = sc.egm_anchor;  temp_c[0] = sc.egm_anchor
            temp_s[0] = 0.0;    temp_b[0] = 0.0

            for s_i in range(n_savings):
                s_val = savings_grid[s_i]

                if constrained:
                    opt_s, opt_b, euler, exit_code, foc_resid = solve_portfolio_2d_retirement(
                    s_val, z_i, i_s,
                    wealth_grid, c_next_slice, pension_next,
                    annuity_factor_is, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                    gamma, psi, beta, b_bar,
                    init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter,
                        tiny_savings=sc.tiny_savings, corner_tol=sc.corner_tol,
                        edge_max_iter=sc.edge_max_iter, edge_accept_factor=sc.edge_accept_factor,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_constrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold)
                else:
                    opt_s, opt_b, euler, exit_code, foc_resid = solve_portfolio_unconstrained_retirement(
                    s_val, z_i, i_s,
                    wealth_grid, c_next_slice, pension_next,
                    annuity_factor_is, Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                    gamma, psi, beta, b_bar,
                    init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter_unconstrained,
                        tiny_savings=sc.tiny_savings,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_unconstrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold,
                        use_line_search=sc.use_line_search,
                        max_backtrack_iter=sc.max_backtrack_iter,
                        line_search_max_step=sc.line_search_max_step)

                # -- Diagnostic tracking --
                diag_int[i_s, 10] += 1  # DI_TOTAL_CALLS
                if exit_code == 0:
                    diag_int[i_s, 9] += 1   # DI_TINY_SAVINGS
                elif exit_code == 1:
                    diag_int[i_s, 0] += 1   # DI_CORNER_BILLS
                elif exit_code == 2:
                    diag_int[i_s, 1] += 1   # DI_CORNER_STOCKS
                elif exit_code == 3:
                    diag_int[i_s, 2] += 1   # DI_CORNER_BONDS
                elif exit_code == 4:
                    diag_int[i_s, 3] += 1   # DI_EDGE_SB
                elif exit_code == 5:
                    diag_int[i_s, 4] += 1   # DI_EDGE_BB
                elif exit_code == 6:
                    diag_int[i_s, 5] += 1   # DI_EDGE_STOCKBOND
                elif exit_code == 7:
                    diag_int[i_s, 6] += 1   # DI_INTERIOR
                elif exit_code == 8:
                    diag_int[i_s, 7] += 1   # DI_NEWTON_FAIL

                # FOC residual tracking
                if foc_resid > diag_float[i_s, 1]:  # DF_MAX_FOC_RESID
                    diag_float[i_s, 1] = foc_resid
                diag_float[i_s, 2] += foc_resid * foc_resid  # DF_SUM_FOC_RESID_SQ

                # Portfolio stats
                diag_float[i_s, 7] += opt_s  # DF_SUM_ALPHA_S
                diag_float[i_s, 8] += opt_b  # DF_SUM_ALPHA_B
                if opt_s < diag_float[i_s, 3]:
                    diag_float[i_s, 3] = opt_s  # DF_MIN_ALPHA_S
                if opt_s > diag_float[i_s, 4]:
                    diag_float[i_s, 4] = opt_s  # DF_MAX_ALPHA_S
                if opt_b < diag_float[i_s, 5]:
                    diag_float[i_s, 5] = opt_b  # DF_MIN_ALPHA_B
                if opt_b > diag_float[i_s, 6]:
                    diag_float[i_s, 6] = opt_b  # DF_MAX_ALPHA_B

                # EGM: invert Euler equation for optimal consumption
                if beta * euler <= 0.0:
                    diag_int[i_s, 11] += 1  # DI_NEG_CONSUMPTION
                c_opt = max(beta * euler, sc.euler_inv_floor) ** (-1.0 / gamma)

                temp_x[s_i + 1] = c_opt + s_val
                temp_c[s_i + 1] = c_opt
                temp_s[s_i + 1] = opt_s
                temp_b[s_i + 1] = opt_b

                last_a_s = opt_s
                last_a_b = opt_b

            # EGM monotonicity check
            for s_i in range(n_savings):
                if temp_x[s_i + 1] <= temp_x[s_i]:
                    diag_int[i_s, 12] += 1  # DI_MONO_VIOLATIONS
                    drop = temp_x[s_i] - temp_x[s_i + 1]
                    if drop > diag_float[i_s, 0]:  # DF_WORST_MONO_DROP
                        diag_float[i_s, 0] = drop

            # Interpolate endogenous grid -> exogenous wealth grid
            for w_i in range(n_wealth):
                w = wealth_grid[w_i]
                policy_c      [z_i, i_s, w_i] = fast_interp_1d(w, temp_x, temp_c)
                policy_alpha_s[z_i, i_s, w_i] = fast_interp_1d(w, temp_x, temp_s)
                policy_alpha_b[z_i, i_s, w_i] = fast_interp_1d(w, temp_x, temp_b)

    return policy_c, policy_alpha_s, policy_alpha_b, diag_int, diag_float


def solve_retirement_step(wealth_grid, savings_grid, z_grid, N_state,
                          c_next_full, pension_1d,
                          annuity_factors, Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                          gamma, psi_vec, beta, b_bar,
                          constrained=True, solver_config=None):
    if solver_config is None:
        solver_config = SolverConfig()
    return _solve_retirement_step_jit(
        wealth_grid, savings_grid, z_grid, N_state,
        c_next_full, pension_1d,
        annuity_factors, Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
        gamma, psi_vec, beta, b_bar,
        constrained, solver_config)


# =============================================================================
# PERIOD SOLVER -- WORKING AGE
# =============================================================================

@njit(parallel=True)
def _solve_working_age_step_jit(wealth_grid, savings_grid, z_grid, N_state,
                                c_next_full, income_next_table,
                                annuity_factors, rho, eta_nodes, eta_weights, dz,
                                Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                                eps_nodes, eps_weights,
                                gamma, psi_vec, beta, b_bar,
                                constrained, solver_config):
    """
    Solve one working-age period using EGM + 2D Newton.
    Parallelised over financial state i_s (prange).

    Parameters
    ----------
    c_next_full        : (n_z, N_state, n_w)  consumption policy at t+1
    income_next_table  : (n_z, n_eps)          after-tax labor income at t+1
    annuity_factors    : (N_state,)             A(y_nom * 4, b_bar) per state

    Returns: policy_c, policy_alpha_s, policy_alpha_b -- each (n_z, N_state, n_w)
             diag_int (N_state, N_DIAG_INT), diag_float (N_state, N_DIAG_FLOAT)
    """

    sc = solver_config

    n_z      = len(z_grid)
    n_savings = len(savings_grid)
    n_wealth  = len(wealth_grid)

    policy_c       = np.empty((n_z, N_state, n_wealth))
    policy_alpha_s = np.empty((n_z, N_state, n_wealth))
    policy_alpha_b = np.empty((n_z, N_state, n_wealth))
    diag_int   = np.zeros((N_state, 13), dtype=np.int64)
    diag_float = np.zeros((N_state, 9))

    for i_s in prange(N_state):
        R_bill = exp(r_bill_grid[i_s])
        annuity_factor_is = annuity_factors[i_s]

        Rx_stock_next, Rx_bond_next = build_gross_return_arrays(mu_r[i_s, :, :], ret_nodes)

        last_a_s = sc.init_alpha_s
        last_a_b = sc.init_alpha_b

        # Init min/max trackers for this i_s
        diag_float[i_s, 3] = 2.0   # DF_MIN_ALPHA_S (init high)
        diag_float[i_s, 5] = 2.0   # DF_MIN_ALPHA_B (init high)

        # Scratch buffers reused across z_i (size invariant; fully overwritten each z_i)
        temp_x = np.empty(n_savings + 1)
        temp_c = np.empty(n_savings + 1)
        temp_s = np.empty(n_savings + 1)
        temp_b = np.empty(n_savings + 1)

        for z_i in range(n_z):
            psi = psi_vec[z_i]

            temp_x[0] = sc.egm_anchor;  temp_c[0] = sc.egm_anchor
            temp_s[0] = 0.0;    temp_b[0] = 0.0

            for s_i in range(n_savings):
                s_val = savings_grid[s_i]

                if constrained:
                    opt_s, opt_b, euler, exit_code, foc_resid = solve_portfolio_2d_working(
                    s_val, z_i, i_s,
                    wealth_grid, c_next_full, income_next_table,
                    annuity_factor_is, z_grid, rho, eta_nodes, eta_weights, dz,
                    Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                    eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                    init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter,
                        tiny_savings=sc.tiny_savings, corner_tol=sc.corner_tol,
                        edge_max_iter=sc.edge_max_iter, edge_accept_factor=sc.edge_accept_factor,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_constrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold)
                else:
                    opt_s, opt_b, euler, exit_code, foc_resid = solve_portfolio_unconstrained_working(
                    s_val, z_i, i_s,
                    wealth_grid, c_next_full, income_next_table,
                    annuity_factor_is, z_grid, rho, eta_nodes, eta_weights, dz,
                    Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                    eps_nodes, eps_weights, gamma, psi, beta, b_bar,
                    init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter_unconstrained,
                        tiny_savings=sc.tiny_savings,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_unconstrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold,
                        use_line_search=sc.use_line_search,
                        max_backtrack_iter=sc.max_backtrack_iter,
                        line_search_max_step=sc.line_search_max_step)

                # -- Diagnostic tracking --
                diag_int[i_s, 10] += 1  # DI_TOTAL_CALLS
                if exit_code == 0:
                    diag_int[i_s, 9] += 1
                elif exit_code == 1:
                    diag_int[i_s, 0] += 1
                elif exit_code == 2:
                    diag_int[i_s, 1] += 1
                elif exit_code == 3:
                    diag_int[i_s, 2] += 1
                elif exit_code == 4:
                    diag_int[i_s, 3] += 1
                elif exit_code == 5:
                    diag_int[i_s, 4] += 1
                elif exit_code == 6:
                    diag_int[i_s, 5] += 1
                elif exit_code == 7:
                    diag_int[i_s, 6] += 1
                elif exit_code == 8:
                    diag_int[i_s, 7] += 1

                if foc_resid > diag_float[i_s, 1]:
                    diag_float[i_s, 1] = foc_resid
                diag_float[i_s, 2] += foc_resid * foc_resid

                diag_float[i_s, 7] += opt_s
                diag_float[i_s, 8] += opt_b
                if opt_s < diag_float[i_s, 3]:
                    diag_float[i_s, 3] = opt_s
                if opt_s > diag_float[i_s, 4]:
                    diag_float[i_s, 4] = opt_s
                if opt_b < diag_float[i_s, 5]:
                    diag_float[i_s, 5] = opt_b
                if opt_b > diag_float[i_s, 6]:
                    diag_float[i_s, 6] = opt_b

                if beta * euler <= 0.0:
                    diag_int[i_s, 11] += 1

                c_opt = max(beta * euler, sc.euler_inv_floor) ** (-1.0 / gamma)

                temp_x[s_i + 1] = c_opt + s_val
                temp_c[s_i + 1] = c_opt
                temp_s[s_i + 1] = opt_s
                temp_b[s_i + 1] = opt_b

                last_a_s = opt_s
                last_a_b = opt_b

            # EGM monotonicity check
            for s_i in range(n_savings):
                if temp_x[s_i + 1] <= temp_x[s_i]:
                    diag_int[i_s, 12] += 1
                    drop = temp_x[s_i] - temp_x[s_i + 1]
                    if drop > diag_float[i_s, 0]:
                        diag_float[i_s, 0] = drop

            for w_i in range(n_wealth):
                w = wealth_grid[w_i]
                policy_c      [z_i, i_s, w_i] = fast_interp_1d(w, temp_x, temp_c)
                policy_alpha_s[z_i, i_s, w_i] = fast_interp_1d(w, temp_x, temp_s)
                policy_alpha_b[z_i, i_s, w_i] = fast_interp_1d(w, temp_x, temp_b)

    return policy_c, policy_alpha_s, policy_alpha_b, diag_int, diag_float


def solve_working_age_step(wealth_grid, savings_grid, z_grid, N_state,
                           c_next_full, income_next_table,
                           annuity_factors, rho, eta_nodes, eta_weights, dz,
                           Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                           eps_nodes, eps_weights,
                           gamma, psi_vec, beta, b_bar,
                           constrained=True, solver_config=None):
    if solver_config is None:
        solver_config = SolverConfig()
    return _solve_working_age_step_jit(
        wealth_grid, savings_grid, z_grid, N_state,
        c_next_full, income_next_table,
        annuity_factors, rho, eta_nodes, eta_weights, dz,
        Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
        eps_nodes, eps_weights,
        gamma, psi_vec, beta, b_bar,
        constrained, solver_config)


# =============================================================================
# MASTER SOLVER
# =============================================================================

def _reduce_diag(diag_int, diag_float):
    """Reduce per-i_s diagnostic arrays to totals."""
    ti = diag_int.sum(axis=0)
    tf_sum = diag_float.sum(axis=0)
    tf_max = diag_float.max(axis=0)
    tf_min = diag_float.min(axis=0)
    return ti, tf_sum, tf_max, tf_min


def _format_pct(count, total):
    if total == 0:
        return "  0%"
    return f"{100.0 * count / total:3.0f}%"


def run_lifecycle_solver(model, pc, n_s_points=None, solver_config=None, verbose=1):
    """
    Lifecycle backward induction solver.

    Parameters
    ----------
    model : LifecyclePortfolioModel
    pc    : Precompute
    n_s_points : int, optional  -- override savings grid size
    verbose : int  -- 0=silent, 1=per-age table + post-solve report (default)

    Returns
    -------
    C_mat, S_mat, B_mat : np.ndarray, shape (n_age, n_z, N_state, n_w)
        Optimal consumption, stock share, and bond share.
    diagnostics : dict
        Diagnostic summary from the solve.
    """
    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"LIFECYCLE PORTFOLIO SOLVER  (EGM + 2D Newton)")
        mode_str = "CONSTRAINED" if model.constrained else "UNCONSTRAINED"
        print(f"  Mode: {mode_str}")
        print(f"  Solver: {solver_config}")
        print(f"  Discretization: {pc.disc_config}")
        print(f"{'='*70}")

    # ---- Grids ----
    w_grid = pc.wealth_grid
    s_grid = pc.s_grid if n_s_points is None else pc.regenerate_savings_grid(n_points=n_s_points)
    z_grid = pc.z_grid
    ages   = pc.ages

    n_w     = len(w_grid)
    n_z     = pc.n_z
    N_state = pc.N_state
    n_age   = pc.n_age

    # ---- Transitions and returns ----
    Pi_state        = pc.Pi_state
    rho             = model.rho
    eta_nodes       = pc.eta_nodes
    eta_weights     = pc.eta_weights
    dz              = pc.dz
    mu_r            = pc.mu_r
    ret_nodes       = pc.ret_nodes
    ret_weights     = pc.ret_weights
    r_bill_grid     = pc.r_bill_grid
    annuity_factors = pc.annuity_factors

    # ---- Income tables ----
    pension_table        = pc.pension_after_tax      # (n_age, n_z)
    working_income_table = pc.working_income          # (n_age, n_z, n_eps)
    eps_nodes   = pc.eps_nodes
    eps_weights = pc.eps_weights

    # ---- Model parameters ----
    gamma          = model.gamma
    beta           = model.beta
    b_bar          = model.b_bar
    survival_probs = pc.survival_probs_2d   # (n_age, n_z)
    retire_age     = model.retire_age
    start_age      = model.start_age
    terminal_age   = model.terminal_age
    constrained    = model.constrained

    if solver_config is None:
        solver_config = SolverConfig()

    if verbose >= 1:
        print(f"  Ages {start_age}\u2013{terminal_age}  ({n_age} periods)")
        print(f"  Grids: n_w={n_w}, n_s={len(s_grid)}, n_z={n_z}, N_state={N_state}")
        print(f"  gamma={gamma}, beta={beta}, b_bar={b_bar}")
        print(f"  States per period: {n_z} \u00d7 {N_state} = {n_z * N_state:,}")

    # ---- Median indices for per-age summary ----
    i_z_med = n_z // 2
    i_s_med = N_state // 2
    i_w_med = n_w // 2

    # ---- Policy arrays ----
    shape = (n_age, n_z, N_state, n_w)
    C_mat = np.zeros(shape)
    S_mat = np.zeros(shape)
    B_mat = np.zeros(shape)

    # ---- Per-age diagnostic accumulators ----
    age_diag_int     = np.zeros((n_age, N_DIAG_INT), dtype=np.int64)
    age_diag_fsum    = np.zeros((n_age, N_DIAG_FLOAT))
    age_diag_fmax    = np.zeros((n_age, N_DIAG_FLOAT))
    age_diag_fmin    = np.full((n_age, N_DIAG_FLOAT), np.inf)

    # ---- Terminal condition ----
    if verbose >= 1:
        print(f"\n  Terminal condition (age {terminal_age}) ... ", end="", flush=True)
    c_T, a_s_T, a_b_T, term_diag = solve_terminal_age(
        w_grid, annuity_factors, r_bill_grid, Pi_state, mu_r, ret_nodes, ret_weights,
        gamma, beta, b_bar, N_state, n_z, constrained=constrained, solver_config=solver_config)
    C_mat[-1] = c_T
    S_mat[-1] = a_s_T
    B_mat[-1] = a_b_T
    if verbose >= 1:
        n_term_interior = int(np.sum(term_diag == EC_INTERIOR))
        n_term_fail = int(np.sum(term_diag == EC_NEWTON_FAIL))
        print(f"done  [c range: {c_T.min():.3f}\u2013{c_T.max():.3f}]  "
              f"[portfolio: {n_term_interior} interior, {N_state - n_term_interior - n_term_fail} corner/edge"
              f"{f', {n_term_fail} FAIL' if n_term_fail > 0 else ''}]")

    # ---- Backward induction ----
    if verbose >= 1:
        print(f"\n{'='*120}")
        hdr = (f" {'Age':>3}  {'Phase':<6} {'Time':>5}  {'Newt%':>5} {'Fail':>6}"
               f"  {'alpha_s':>7}  {'alpha_b':>7}  {'a_bill':>7}  {'c/W':>5}"
               f"  {'%int':>4}  {'%edge':>5}  {'%corn':>5}  {'mono':>4}")
        print(hdr)
        print(f"{'='*120}")

    t_start = time.time()

    for t in reversed(range(n_age - 1)):
        age    = ages[t]
        psi    = survival_probs[t, :]      # (n_z,) -- z-dependent survival
        c_next = C_mat[t + 1]

        if age >= retire_age:
            c, a_s, a_b, _di, _df = solve_retirement_step(
                w_grid, s_grid, z_grid, N_state,
                c_next, pension_table[t + 1, :],
                annuity_factors, Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                gamma, psi, beta, b_bar, constrained=constrained, solver_config=solver_config)
            label = "RETIRE"
        else:
            c, a_s, a_b, _di, _df = solve_working_age_step(
                w_grid, s_grid, z_grid, N_state,
                c_next, working_income_table[t + 1, :, :],
                annuity_factors, rho, eta_nodes, eta_weights, dz,
                Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                eps_nodes, eps_weights,
                gamma, psi, beta, b_bar, constrained=constrained, solver_config=solver_config)
            label = "WORK  "

        C_mat[t] = c
        S_mat[t] = a_s
        B_mat[t] = a_b

        # Reduce diagnostics for this age
        ti, tf_sum, tf_max, tf_min = _reduce_diag(_di, _df)
        age_diag_int[t]  = ti
        age_diag_fsum[t] = tf_sum
        age_diag_fmax[t] = tf_max
        age_diag_fmin[t] = tf_min

        # Per-age one-line summary
        if verbose >= 1:
            elapsed = time.time() - t_start
            total_calls = int(ti[DI_TOTAL_CALLS])
            n_fail = int(ti[DI_NEWTON_FAIL])
            newton_pct = 100.0 * (total_calls - n_fail) / max(total_calls, 1)

            n_interior = int(ti[DI_INTERIOR])
            n_edge = int(ti[DI_EDGE_SB] + ti[DI_EDGE_BB] + ti[DI_EDGE_STOCKBOND])
            n_corner = int(ti[DI_CORNER_BILLS] + ti[DI_CORNER_STOCKS] + ti[DI_CORNER_BONDS]
                           + ti[DI_TINY_SAVINGS])
            mono_v = int(ti[DI_MONO_VIOLATIONS])

            # Median-state policy values
            med_as = float(a_s[i_z_med, i_s_med, i_w_med])
            med_ab = float(a_b[i_z_med, i_s_med, i_w_med])
            med_bill = 1.0 - med_as - med_ab
            med_c = float(c[i_z_med, i_s_med, i_w_med])
            med_w = float(w_grid[i_w_med])
            c_over_w = med_c / med_w if med_w > 0 else 0.0

            mono_str = f"{mono_v:4d}" if mono_v == 0 else f"\033[91m{mono_v:4d}\033[0m"

            print(f" {age:3d}  {label:<6} {elapsed:5.1f}s  {newton_pct:5.1f}% {n_fail:>6}"
                  f"  {med_as:7.3f}  {med_ab:7.3f}  {med_bill:7.3f}  {c_over_w:5.3f}"
                  f"  {_format_pct(n_interior, total_calls)}"
                  f"  {_format_pct(n_edge, total_calls)}"
                  f"  {_format_pct(n_corner, total_calls)}"
                  f"  {mono_str}", flush=True)

    total = time.time() - t_start

    # ========================================================================
    # POST-SOLVE DIAGNOSTICS
    # ========================================================================

    # Aggregate across all ages (exclude terminal t=-1 which has no diag arrays)
    all_int = age_diag_int[:-1].sum(axis=0)  # sum across ages
    all_fsum = age_diag_fsum[:-1].sum(axis=0)
    all_fmax = age_diag_fmax[:-1].max(axis=0)
    all_fmin = age_diag_fmin[:-1].min(axis=0)

    total_calls = int(all_int[DI_TOTAL_CALLS])
    total_fail = int(all_int[DI_NEWTON_FAIL])
    total_mono = int(all_int[DI_MONO_VIOLATIONS])
    worst_mono = float(all_fmax[DF_WORST_MONO_DROP])
    worst_foc = float(all_fmax[DF_MAX_FOC_RESID])
    rms_foc = (all_fsum[DF_SUM_FOC_RESID_SQ] / max(total_calls, 1)) ** 0.5

    # Build diagnostics dict
    diagnostics = {
        'age_diag_int': age_diag_int,
        'age_diag_fsum': age_diag_fsum,
        'age_diag_fmax': age_diag_fmax,
        'age_diag_fmin': age_diag_fmin,
        'total_mono_violations': total_mono,
        'worst_mono_drop': worst_mono,
        'total_newton_failures': total_fail,
        'worst_foc_resid': worst_foc,
        'total_calls': total_calls,
        'constrained': constrained,
        'solver_config': solver_config,
        'disc_config': pc.disc_config,
    }

    if verbose >= 1:
        print(f"\n{'='*120}")
        print(f"  DONE in {total / 60:.2f} min  (avg {total / max(n_age - 1, 1):.2f}s per age)")
        print(f"{'='*120}")

        # --- Section 1: Newton Convergence ---
        print(f"\n{'='*70}")
        print(f"  POST-SOLVE DIAGNOSTICS")
        print(f"{'='*70}")

        print(f"\n  1. NEWTON CONVERGENCE")
        print(f"     Total calls:  {total_calls:>12,}")
        print(f"     Converged:    {total_calls - total_fail:>12,}  ({100.0 * (total_calls - total_fail) / max(total_calls, 1):.3f}%)")
        print(f"     Failed:       {total_fail:>12,}  ({100.0 * total_fail / max(total_calls, 1):.3f}%)")
        print(f"     Worst FOC:    {worst_foc:>12.2e}")
        print(f"     RMS FOC:      {rms_foc:>12.2e}")
        if total_fail > 0:
            print(f"\n     Ages with failures:")
            for t in range(n_age - 1):
                nf = int(age_diag_int[t, DI_NEWTON_FAIL])
                if nf > 0:
                    age = ages[t]
                    lbl = "RETIRE" if age >= retire_age else "WORK"
                    mfoc = float(age_diag_fmax[t, DF_MAX_FOC_RESID])
                    print(f"       Age {age:3d} {lbl:>6}: {nf:4d} failures  (max resid {mfoc:.2e})")

        # --- Section 2: Portfolio Regime Breakdown ---
        print(f"\n  2. PORTFOLIO REGIME BREAKDOWN")
        retire_mask = np.array([ages[t] >= retire_age for t in range(n_age - 1)])
        work_mask   = ~retire_mask

        def _regime_row(mask, label):
            sel = age_diag_int[:-1][mask].sum(axis=0) if mask.any() else np.zeros(N_DIAG_INT, dtype=np.int64)
            tot = int(sel[DI_TOTAL_CALLS])
            if tot == 0:
                return
            bills  = int(sel[DI_CORNER_BILLS] + sel[DI_TINY_SAVINGS])
            stocks = int(sel[DI_CORNER_STOCKS])
            bonds  = int(sel[DI_CORNER_BONDS])
            sb     = int(sel[DI_EDGE_SB])
            bb     = int(sel[DI_EDGE_BB])
            sB     = int(sel[DI_EDGE_STOCKBOND])
            intr   = int(sel[DI_INTERIOR])
            fail   = int(sel[DI_NEWTON_FAIL])
            print(f"     {label:<12}"
                  f"  {100*bills/tot:5.1f}%"
                  f"  {100*stocks/tot:5.1f}%"
                  f"  {100*bonds/tot:5.1f}%"
                  f"  {100*sb/tot:5.1f}%"
                  f"  {100*bb/tot:5.1f}%"
                  f"  {100*sB/tot:5.1f}%"
                  f"  {100*intr/tot:5.1f}%"
                  f"  {100*fail/tot:5.1f}%")

        print(f"     {'':12}  {'Bills':>6}  {'Stocks':>6}  {'Bonds':>6}"
              f"  {'S+Bill':>6}  {'B+Bill':>6}  {'S+Bond':>6}  {'Inter.':>6}  {'Fail':>6}")
        _regime_row(retire_mask, "Retirement:")
        _regime_row(work_mask,   "Working:")
        _regime_row(np.ones(n_age - 1, dtype=bool), "Overall:")

        # --- Section 3: Portfolio Share Ranges ---
        print(f"\n  3. PORTFOLIO SHARE RANGES")
        mean_as = all_fsum[DF_SUM_ALPHA_S] / max(total_calls, 1)
        mean_ab = all_fsum[DF_SUM_ALPHA_B] / max(total_calls, 1)
        print(f"     Stock:  [{all_fmin[DF_MIN_ALPHA_S]:.3f}, {all_fmax[DF_MAX_ALPHA_S]:.3f}]  mean={mean_as:.3f}")
        print(f"     Bond:   [{all_fmin[DF_MIN_ALPHA_B]:.3f}, {all_fmax[DF_MAX_ALPHA_B]:.3f}]  mean={mean_ab:.3f}")
        print(f"     Bill:   mean={1.0 - mean_as - mean_ab:.3f}  (inferred: 1-s-b)")

        # --- Section 4: EGM Monotonicity ---
        print(f"\n  4. EGM MONOTONICITY")
        if total_mono > 0:
            n_affected_ages = int(np.sum(age_diag_int[:-1, DI_MONO_VIOLATIONS] > 0))
            print(f"     WARNING: {total_mono} total violations across {n_affected_ages}/{n_age-1} ages")
            print(f"     Worst drop: {worst_mono:.2e}")
            for t in range(n_age - 1):
                mv = int(age_diag_int[t, DI_MONO_VIOLATIONS])
                if mv > 0:
                    age = ages[t]
                    lbl = "RETIRE" if age >= retire_age else "WORK"
                    wd = float(age_diag_fmax[t, DF_WORST_MONO_DROP])
                    print(f"       Age {age:3d} {lbl:>6}: {mv:4d} violations, worst drop {wd:.2e}")
        else:
            print(f"     PASS")

        # --- Section 5: Policy Function Sanity ---
        print(f"\n  5. POLICY FUNCTION SANITY")
        nan_c = int(np.isnan(C_mat).sum())
        nan_s = int(np.isnan(S_mat).sum())
        nan_b = int(np.isnan(B_mat).sum())
        inf_c = int(np.isinf(C_mat).sum())
        inf_s = int(np.isinf(S_mat).sum())
        inf_b = int(np.isinf(B_mat).sum())
        neg_c = int((C_mat < 0).sum())
        neg_euler = int(all_int[DI_NEG_CONSUMPTION])
        alpha_s_neg = int((S_mat < -1e-6).sum())
        alpha_b_neg = int((B_mat < -1e-6).sum())
        alpha_sum_viol = int(((S_mat + B_mat) > 1.0 + 1e-6).sum())
        total_el = C_mat.size + S_mat.size + B_mat.size

        if constrained:
            all_ok = (nan_c + nan_s + nan_b + inf_c + inf_s + inf_b
                      + neg_c + alpha_s_neg + alpha_b_neg + alpha_sum_viol + neg_euler == 0)
        else:
            # Unconstrained: negative alphas and sum > 1 are expected
            all_ok = (nan_c + nan_s + nan_b + inf_c + inf_s + inf_b
                      + neg_c + neg_euler == 0)
        if all_ok:
            print(f"     PASS  ({total_el:,} elements checked)")
        else:
            print(f"     NaN count:       C={nan_c}  S={nan_s}  B={nan_b}")
            print(f"     Inf count:       C={inf_c}  S={inf_s}  B={inf_b}")
            print(f"     C < 0:           {neg_c}")
            print(f"     Neg euler:       {neg_euler}")
            print(f"     alpha_s < -1e-6: {alpha_s_neg}")
            print(f"     alpha_b < -1e-6: {alpha_b_neg}")
            print(f"     alpha_s+b > 1:   {alpha_sum_viol}")

        print(f"{'='*70}\n")

    return C_mat, S_mat, B_mat, diagnostics

"""
solver.py — Backward induction solver with EGM + 2D Newton-Raphson.

Three assets: Bills, Stocks, Nominal Bonds.
Catherine (2025) bequest motive: b(W, A) = b_bar * (W/A)^(1-gamma) / (1-gamma)

Contains:
  - Diagnostic constants (DI_*, DF_*, EC_*)
  - Interpolation utilities (fast_interp_1d, etc.)
  - FOC + Jacobian functions (quadrature-based: retirement, working, terminal)
  - Newton portfolio solvers (constrained + unconstrained, quadrature)
  - Period solvers: solve_retirement_step_quad(), solve_working_age_step_quad()
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
import csv
from functools import lru_cache
from pathlib import Path

from lifecycle.model import SolveControl, SolverConfig, scalar_disposable_income
from lifecycle.numerics import _pchip_slope_uniform, _pchip_eval_with_basis

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
DI_SUM_ITER        = 13  # total Newton outer-iterations summed over unconstrained calls
DI_WARM_RESET      = 14  # times warm-start was reset to cold init after EC_NEWTON_FAIL
N_DIAG_INT = 15

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
DF_MAX_NEWTON_ITER  = 9  # max outer iters in any single unconstrained Newton call
N_DIAG_FLOAT = 10

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


_AWI_2019_USD = 54_099.99
_AWI_2019_KUSD = _AWI_2019_USD / 1_000.0
_SCF_WEALTH_CSV = Path(__file__).resolve().parent.parent / "data" / "scf_net_worth_by_age_2022.csv"
_PROGRESS_WEALTH_SOURCES = frozenset({"grid_midpoint", "scf_median", "scf_mean"})


def _pad_state_solver_inputs_to_3d(
    state_grid,
    state_bracket_grids,
    state_bracket_shift,
    state_bracket_L_inv,
    v_nodes,
    Phi_0_state,
    Phi_11,
    A_r,
):
    """Pad 1D/2D state objects to the solver's 3-coordinate layout.

    The hot JIT kernels are still written in a 3-coordinate style. For System I
    and II/III we embed the lower-dimensional state into that layout with
    singleton dummy axes. The omitted coordinates stay identically zero, so the
    economic model is unchanged; this just preserves the existing kernel shape.
    """
    n_state = int(state_grid.shape[1])
    if n_state == 3:
        grids_0 = np.ascontiguousarray(state_bracket_grids[0], dtype=float)
        grids_1 = np.ascontiguousarray(state_bracket_grids[1], dtype=float)
        grids_2 = np.ascontiguousarray(state_bracket_grids[2], dtype=float)
        return (
            np.ascontiguousarray(state_grid, dtype=float),
            grids_0,
            grids_1,
            grids_2,
            np.ascontiguousarray(state_bracket_shift, dtype=float),
            np.ascontiguousarray(state_bracket_L_inv, dtype=float),
            np.ascontiguousarray(v_nodes, dtype=float),
            np.ascontiguousarray(Phi_0_state, dtype=float),
            np.ascontiguousarray(Phi_11, dtype=float),
            np.ascontiguousarray(A_r, dtype=float),
        )

    state_grid_pad = np.zeros((state_grid.shape[0], 3), dtype=float)
    state_grid_pad[:, :n_state] = np.asarray(state_grid, dtype=float)

    grids = []
    for d in range(3):
        if d < n_state:
            grids.append(np.ascontiguousarray(np.asarray(state_bracket_grids[d], dtype=float)))
        else:
            grids.append(np.zeros(1, dtype=float))

    shift_pad = np.zeros(3, dtype=float)
    shift_pad[:n_state] = np.asarray(state_bracket_shift, dtype=float)

    L_inv_pad = np.zeros((3, 3), dtype=float)
    L_inv_pad[:n_state, :n_state] = np.asarray(state_bracket_L_inv, dtype=float)

    v_nodes_pad = np.zeros((v_nodes.shape[0], 3), dtype=float)
    v_nodes_pad[:, :n_state] = np.asarray(v_nodes, dtype=float)

    Phi_0_state_pad = np.zeros(3, dtype=float)
    Phi_0_state_pad[:n_state] = np.asarray(Phi_0_state, dtype=float)

    Phi_11_pad = np.zeros((3, 3), dtype=float)
    Phi_11_pad[:n_state, :n_state] = np.asarray(Phi_11, dtype=float)

    A_r_pad = np.zeros((A_r.shape[0], 3), dtype=float)
    A_r_pad[:, :n_state] = np.asarray(A_r, dtype=float)

    return (
        np.ascontiguousarray(state_grid_pad),
        grids[0],
        grids[1],
        grids[2],
        np.ascontiguousarray(shift_pad),
        np.ascontiguousarray(L_inv_pad),
        np.ascontiguousarray(v_nodes_pad),
        np.ascontiguousarray(Phi_0_state_pad),
        np.ascontiguousarray(Phi_11_pad),
        np.ascontiguousarray(A_r_pad),
    )


@lru_cache(maxsize=1)
def _load_scf_wealth_age_table():
    """Load SCF wealth-by-age targets once per process.

    The source CSV stores wealth in thousands of 2022 USD. We keep those units
    here and normalize to model units only when building the per-age probe
    schedule.
    """
    age_mid = []
    med_kusd = []
    mean_kusd = []

    with _SCF_WEALTH_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
        for row in reader:
            if row["age_group"].strip().lower() == "all":
                continue
            age_mid.append(float(row["age_midpoint"]))
            med_kusd.append(float(row["median_2022_k2022usd"]))
            mean_kusd.append(float(row["mean_2022_k2022usd"]))

    if len(age_mid) == 0:
        raise ValueError(f"SCF wealth file {_SCF_WEALTH_CSV} is empty")

    age_mid = np.asarray(age_mid, dtype=float)
    med_kusd = np.asarray(med_kusd, dtype=float)
    mean_kusd = np.asarray(mean_kusd, dtype=float)
    valid = np.isfinite(age_mid) & np.isfinite(med_kusd) & np.isfinite(mean_kusd)
    if not np.all(valid):
        raise ValueError(f"SCF wealth file {_SCF_WEALTH_CSV} has no valid age rows")

    return age_mid, med_kusd, mean_kusd


def _build_progress_wealth_schedule(ages, w_grid, source):
    """Return the per-age wealth probe used in verbose progress output."""
    source = str(source).strip().lower()
    w_grid = np.asarray(w_grid, dtype=float)
    ages = np.asarray(ages, dtype=float)

    if w_grid.ndim != 1 or w_grid.size < 2:
        raise ValueError("w_grid must be a 1D array with at least two points")

    if source == "grid_midpoint":
        probe_w = float(w_grid[len(w_grid) // 2])
        return np.full(ages.shape, probe_w, dtype=float), (
            f"grid midpoint (legacy constant probe, W={probe_w:.3f})"
        )

    age_mid, med_kusd, mean_kusd = _load_scf_wealth_age_table()
    if source == "scf_median":
        wealth_kusd = med_kusd
        label = "SCF median wealth by age (2022 $, linear age interp, AWI-normalized)"
    elif source == "scf_mean":
        wealth_kusd = mean_kusd
        label = "SCF mean wealth by age (2022 $, linear age interp, AWI-normalized)"
    else:
        raise ValueError(
            "progress_wealth_source must be one of "
            f"{sorted(_PROGRESS_WEALTH_SOURCES)}, got {source!r}"
        )

    wealth_model_units = wealth_kusd / _AWI_2019_KUSD
    schedule = np.interp(ages, age_mid, wealth_model_units)
    schedule = np.clip(schedule, float(w_grid[0]), float(w_grid[-1]))
    return schedule, label


def _interp_progress_policy_at_wealth(policy_by_wealth, w_grid, wealth):
    """Interpolate one 1D policy slice at the requested reporting wealth."""
    return float(
        np.interp(
            float(wealth),
            np.asarray(w_grid, dtype=float),
            np.asarray(policy_by_wealth, dtype=float),
        )
    )


# =============================================================================
# HELPERS: INTERPOLATION AND SIMPLEX PROJECTION
# =============================================================================


@njit(fastmath=True)
def fast_interp_1d(x, x_grid, y_grid):
    """Linear interpolation on a sorted grid with binary search,
    using LINEAR extrapolation beyond grid boundaries.

    Solver-internal: the policy continuation values must remain smooth
    at the wealth-grid edge so Newton-Raphson converges cleanly across
    boundary cells. simulation.py has its own `fast_interp_1d` that uses
    FLAT extrapolation; the two functions are deliberately divergent.
    Do not "fix" either one to match the other.
    """
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
      iw is clamped to a valid segment index in [0, len(grid)-2]
      frac_w = (x - grid[iw]) / dw on that segment
      inv_dw = 1.0 / dw

    For off-grid x, frac_w is allowed to lie outside [0, 1]. This keeps the
    returned affine coordinate consistent with the last-segment slope used by
    the working-age Jacobian.
    """
    n = len(grid)
    if x <= grid[0]:
        dw = grid[1] - grid[0] + 1e-30
        return 0, (x - grid[0]) / dw, 1.0 / dw
    if x >= grid[n - 1]:
        dw = grid[n - 1] - grid[n - 2] + 1e-30
        return n - 2, (x - grid[n - 2]) / dw, 1.0 / dw
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
    """Build exp(mu_r + residual_node) arrays for one current financial state.

    Returns 3 arrays (bill, stock, bond) for n_ret=3 return dimensions.
    """
    n_state = mu_r_i.shape[0]
    n_ret_quad = ret_nodes.shape[0]

    rx_bill_next = np.empty((n_state, n_ret_quad))
    rx_stock_next = np.empty((n_state, n_ret_quad))
    rx_bond_next = np.empty((n_state, n_ret_quad))

    for j_s in range(n_state):
        mu_bill = mu_r_i[j_s, 0]
        mu_stock = mu_r_i[j_s, 1]
        mu_bond = mu_r_i[j_s, 2]
        for k_r in range(n_ret_quad):
            rx_bill_next[j_s, k_r] = exp(mu_bill + ret_nodes[k_r, 0])
            rx_stock_next[j_s, k_r] = exp(mu_stock + ret_nodes[k_r, 1])
            rx_bond_next[j_s, k_r] = exp(mu_bond + ret_nodes[k_r, 2])

    return rx_bill_next, rx_stock_next, rx_bond_next


@njit(fastmath=True)
def bracket_state_3d(s0, s1, s2, grids_0, grids_1, grids_2):
    """Bracket (s0,s1,s2) in the 3D state grid.

    Returns (lo0, lo1, lo2, f0, f1, f2) where:
      grids_d[lo_d] <= s_d < grids_d[lo_d + 1]
      f_d = (s_d - grids_d[lo_d]) / (grids_d[lo_d+1] - grids_d[lo_d])
    Clamped to valid range.
    """
    # Dimension 0
    n0 = len(grids_0)
    if n0 == 1:
        lo0 = 0; f0 = 0.0
    elif s0 <= grids_0[0]:
        lo0 = 0; f0 = 0.0
    elif s0 >= grids_0[n0 - 1]:
        lo0 = n0 - 2; f0 = 1.0
    else:
        lo0 = 0
        for ii in range(n0 - 1):
            if grids_0[ii + 1] > s0:
                lo0 = ii
                break
        dg = grids_0[lo0 + 1] - grids_0[lo0]
        f0 = (s0 - grids_0[lo0]) / dg if dg > 1e-30 else 0.0
        f0 = max(0.0, min(1.0, f0))

    # Dimension 1
    n1 = len(grids_1)
    if n1 == 1:
        lo1 = 0; f1 = 0.0
    elif s1 <= grids_1[0]:
        lo1 = 0; f1 = 0.0
    elif s1 >= grids_1[n1 - 1]:
        lo1 = n1 - 2; f1 = 1.0
    else:
        lo1 = 0
        for ii in range(n1 - 1):
            if grids_1[ii + 1] > s1:
                lo1 = ii
                break
        dg = grids_1[lo1 + 1] - grids_1[lo1]
        f1 = (s1 - grids_1[lo1]) / dg if dg > 1e-30 else 0.0
        f1 = max(0.0, min(1.0, f1))

    # Dimension 2
    n2 = len(grids_2)
    if n2 == 1:
        lo2 = 0; f2 = 0.0
    elif s2 <= grids_2[0]:
        lo2 = 0; f2 = 0.0
    elif s2 >= grids_2[n2 - 1]:
        lo2 = n2 - 2; f2 = 1.0
    else:
        lo2 = 0
        for ii in range(n2 - 1):
            if grids_2[ii + 1] > s2:
                lo2 = ii
                break
        dg = grids_2[lo2 + 1] - grids_2[lo2]
        f2 = (s2 - grids_2[lo2]) / dg if dg > 1e-30 else 0.0
        f2 = max(0.0, min(1.0, f2))

    return lo0, lo1, lo2, f0, f1, f2


@njit(fastmath=True)
def transform_state_for_bracketing_3d(s0, s1, s2, bracket_shift, bracket_L_inv):
    """Map an economic state into the coordinates used by state-grid bracketing."""
    ds0 = s0 - bracket_shift[0]
    ds1 = s1 - bracket_shift[1]
    ds2 = s2 - bracket_shift[2]

    u0 = bracket_L_inv[0, 0] * ds0 + bracket_L_inv[0, 1] * ds1 + bracket_L_inv[0, 2] * ds2
    u1 = bracket_L_inv[1, 0] * ds0 + bracket_L_inv[1, 1] * ds1 + bracket_L_inv[1, 2] * ds2
    u2 = bracket_L_inv[2, 0] * ds0 + bracket_L_inv[2, 1] * ds1 + bracket_L_inv[2, 2] * ds2
    return u0, u1, u2


@njit(fastmath=True, inline='always')
def _pchip_slope_nonuniform(d_left, d_right, h_left, h_right):
    # Fritsch-Carlson slope at an interior node on a non-uniform grid.
    # Returns 0 at sign changes (preserves monotonicity at extrema).
    if d_left == 0.0 or d_right == 0.0:
        return 0.0
    if d_left * d_right <= 0.0:
        return 0.0
    w_left = 2.0 * h_right + h_left
    w_right = h_right + 2.0 * h_left
    return (w_left + w_right) / (w_left / d_left + w_right / d_right)


@njit(fastmath=True, inline='always')
def _pchip_endpoint_slope(d_near, d_far, h_near, h_far):
    # Three-point one-sided slope at a grid boundary, with monotonicity clamp.
    # Hyman / Fritsch-Butland formula.
    m = ((2.0 * h_near + h_far) * d_near - h_near * d_far) / (h_near + h_far)
    if m * d_near <= 0.0:
        return 0.0
    if abs(m) > 3.0 * abs(d_near):
        return 3.0 * d_near
    return m


@njit(fastmath=True)
def pchip_interp_1d(x, x_grid, y_grid):
    """Monotonicity-preserving cubic Hermite interpolation on a non-uniform grid.

    Linear extrapolation outside [x_grid[0], x_grid[-1]] to match
    fast_interp_1d boundary semantics. Boundary-segment slopes use
    three-point one-sided estimates with monotonicity clamping.
    """
    n = len(x_grid)
    if n < 2:
        return y_grid[0]

    if x <= x_grid[0]:
        h = x_grid[1] - x_grid[0] + 1e-30
        return y_grid[0] + (y_grid[1] - y_grid[0]) * (x - x_grid[0]) / h
    if x >= x_grid[n - 1]:
        h = x_grid[n - 1] - x_grid[n - 2] + 1e-30
        return y_grid[n - 1] + (y_grid[n - 1] - y_grid[n - 2]) * (x - x_grid[n - 1]) / h

    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    i = lo

    h_i = x_grid[i + 1] - x_grid[i]
    if h_i < 1e-30:
        return y_grid[i]
    d_i = (y_grid[i + 1] - y_grid[i]) / h_i

    if i == 0:
        h_next = x_grid[2] - x_grid[1] if n >= 3 else h_i
        d_next = (y_grid[2] - y_grid[1]) / h_next if n >= 3 else d_i
        m_left = _pchip_endpoint_slope(d_i, d_next, h_i, h_next)
    else:
        h_prev = x_grid[i] - x_grid[i - 1]
        d_prev = (y_grid[i] - y_grid[i - 1]) / h_prev
        m_left = _pchip_slope_nonuniform(d_prev, d_i, h_prev, h_i)

    if i == n - 2:
        h_prev = x_grid[i] - x_grid[i - 1] if n >= 3 else h_i
        d_prev = (y_grid[i] - y_grid[i - 1]) / h_prev if n >= 3 else d_i
        m_right = _pchip_endpoint_slope(d_i, d_prev, h_i, h_prev)
    else:
        h_next = x_grid[i + 2] - x_grid[i + 1]
        d_next = (y_grid[i + 2] - y_grid[i + 1]) / h_next
        m_right = _pchip_slope_nonuniform(d_i, d_next, h_i, h_next)

    t = (x - x_grid[i]) / h_i
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return (h00 * y_grid[i] + h10 * h_i * m_left
            + h01 * y_grid[i + 1] + h11 * h_i * m_right)


@njit(fastmath=True)
def _interp_z_wealth(c_next_full, j_s, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_c):
    """Interpolate c_next and mpc at a single state-grid corner j_s.

    Uses PCHIP (Fritsch-Carlson, monotonicity-preserving) cubic Hermite in z
    on interior intervals and linear fallback on the first and last z
    intervals. Wealth interpolation is linear on [iw, iw+1].

    Interpolation order (cubic branch): wealth-blend the 4-point z stencil
    FIRST, then apply PCHIP to the blended stencil. This preserves the
    monotonicity-in-z property at off-grid wealth points -- the
    "PCHIP-then-blend" alternative violates shape preservation because
    Fritsch-Carlson is non-linear in stencil values, so it does not
    commute with linear wealth blending.

    `mpc` is the secant slope of c_val between iw and iw+1, computed as
    (PCHIP(stencil_iw1) - PCHIP(stencil_iw)) / dw. At frac_w in {0, 1}
    this matches the corresponding endpoint evaluation exactly. On the
    interior, c_val under blend-then-PCHIP is non-linear in frac_w, so
    mpc is the segment secant rather than the pointwise derivative.

    `frac_w` is the affine wealth coordinate on the selected segment and
    may be outside [0, 1] for linear extrapolation.
    """
    if use_cubic:
        f2 = frac_z * frac_z
        f3 = f2 * frac_z
        h00 = 2.0 * f3 - 3.0 * f2 + 1.0
        h10 = f3 - 2.0 * f2 + frac_z
        h01 = -2.0 * f3 + 3.0 * f2
        h11 = f3 - f2

        # Read 4-point z stencil at both wealth corners.
        p0 = c_next_full[iz_lo - 1, j_s, iw]
        p1 = c_next_full[iz_lo,     j_s, iw]
        p2 = c_next_full[iz_lo + 1, j_s, iw]
        p3 = c_next_full[iz_lo + 2, j_s, iw]

        q0 = c_next_full[iz_lo - 1, j_s, iw + 1]
        q1 = c_next_full[iz_lo,     j_s, iw + 1]
        q2 = c_next_full[iz_lo + 1, j_s, iw + 1]
        q3 = c_next_full[iz_lo + 2, j_s, iw + 1]

        # Blend-then-PCHIP for c_val: wealth-blend the 4 z stencil values,
        # then run PCHIP once on the blended stencil. Shape-preserving in z
        # at the actually-blended wealth slice.
        one_minus_fw = 1.0 - frac_w
        b0 = one_minus_fw * p0 + frac_w * q0
        b1 = one_minus_fw * p1 + frac_w * q1
        b2 = one_minus_fw * p2 + frac_w * q2
        b3 = one_minus_fw * p3 + frac_w * q3
        c_val = _pchip_eval_with_basis(b0, b1, b2, b3, h00, h10, h01, h11)
        c_val = max(c_val, min_c)

        # mpc = secant slope of c_val between fw=0 and fw=1.  At fw=0 the
        # blended stencil equals the iw stencil, so PCHIP(b at fw=0) =
        # PCHIP(p..) = c_iw; same logic at fw=1.  The secant slope is the
        # diagonal Jacobian element of c_val w.r.t. x_next.
        c_iw = _pchip_eval_with_basis(p0, p1, p2, p3, h00, h10, h01, h11)
        c_iw1 = _pchip_eval_with_basis(q0, q1, q2, q3, h00, h10, h01, h11)
        mpc_val = (c_iw1 - c_iw) * inv_dw
        mpc_val = max(0.0, min(1.0, mpc_val))
    else:
        c_lo = (1.0 - frac_w) * c_next_full[iz_lo,     j_s, iw] + frac_w * c_next_full[iz_lo,     j_s, iw + 1]
        c_hi = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]
        c_val = (1.0 - frac_z) * c_lo + frac_z * c_hi
        c_val = max(c_val, min_c)

        mpc_lo = (c_next_full[iz_lo,     j_s, iw + 1] - c_next_full[iz_lo,     j_s, iw]) * inv_dw
        mpc_hi = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw
        mpc_val = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi
        mpc_val = max(0.0, min(1.0, mpc_val))

    return c_val, mpc_val


# =============================================================================
# FOC AND JACOBIAN -- RETIREMENT (QUADRATURE)
# =============================================================================


@njit(fastmath=True)
def compute_foc_jac_retirement_quad(
    alpha_s, alpha_b, s_val, z_idx, i_s,
    wealth_grid, c_next_full, pension_next_scalar,
    annuity_factor_is,
    # --- State quadrature arrays ---
    v_nodes, v_weights, M_v_nodes,
    base_mu_r_i,          # const_r + A_r @ s_i, precomputed per i_s
    Phi_0_state, Phi_11, state_grid_i,  # for computing s_next
    state_bracket_shift, state_bracket_L_inv,
    grids_0, grids_1, grids_2,          # marginal grids for bracketing
    N1, N2,                              # grid sizes dim 1, dim 2
    # --- Return quadrature ---
    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
    # --- Model parameters ---
    gamma, psi, beta, b_bar,
    min_wealth_inv=1e-10, min_consumption=1e-10,
    prob_skip=1e-12,
):
    """FOC and Jacobian for retirement portfolio with state innovation quadrature.

    Replaces the discrete Pi_state loop with GH quadrature over v^s ~ N(0, Sigma_ss).
    Policy lookups use trilinear interpolation across 8 state-grid corners.
    """
    a_bill = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi
    foc_s = 0.0; foc_b = 0.0
    J_ss = 0.0; J_bb = 0.0; J_sb = 0.0
    euler_sum = 0.0

    n_state_quad = len(v_weights)
    n_ret_quad = len(ret_weights)

    for k_v in range(n_state_quad):
        w_v = v_weights[k_v]
        if w_v < prob_skip:
            continue

        # --- Continuous next state ---
        s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * state_grid_i[0] + Phi_11[0, 1] * state_grid_i[1] + Phi_11[0, 2] * state_grid_i[2] + v_nodes[k_v, 0]
        s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * state_grid_i[0] + Phi_11[1, 1] * state_grid_i[1] + Phi_11[1, 2] * state_grid_i[2] + v_nodes[k_v, 1]
        s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * state_grid_i[0] + Phi_11[2, 1] * state_grid_i[1] + Phi_11[2, 2] * state_grid_i[2] + v_nodes[k_v, 2]

        b_next_0, b_next_1, b_next_2 = transform_state_for_bracketing_3d(
            s_next_0, s_next_1, s_next_2, state_bracket_shift, state_bracket_L_inv
        )

        # --- Bracket transformed next state in 3D grid ---
        lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
            b_next_0, b_next_1, b_next_2, grids_0, grids_1, grids_2
        )
        hi0 = lo0 if len(grids_0) == 1 else lo0 + 1
        hi1 = lo1 if len(grids_1) == 1 else lo1 + 1
        hi2 = lo2 if len(grids_2) == 1 else lo2 + 1

        # 8 trilinear weights
        w000 = (1.0 - f0) * (1.0 - f1) * (1.0 - f2)
        w001 = (1.0 - f0) * (1.0 - f1) * f2
        w010 = (1.0 - f0) * f1 * (1.0 - f2)
        w011 = (1.0 - f0) * f1 * f2
        w100 = f0 * (1.0 - f1) * (1.0 - f2)
        w101 = f0 * (1.0 - f1) * f2
        w110 = f0 * f1 * (1.0 - f2)
        w111 = f0 * f1 * f2

        # 8 flat indices into c_next_full's j_s dimension
        j000 = lo0 * N1 * N2 + lo1 * N2 + lo2
        j001 = lo0 * N1 * N2 + lo1 * N2 + hi2
        j010 = lo0 * N1 * N2 + hi1 * N2 + lo2
        j011 = lo0 * N1 * N2 + hi1 * N2 + hi2
        j100 = hi0 * N1 * N2 + lo1 * N2 + lo2
        j101 = hi0 * N1 * N2 + lo1 * N2 + hi2
        j110 = hi0 * N1 * N2 + hi1 * N2 + lo2
        j111 = hi0 * N1 * N2 + hi1 * N2 + hi2

        # --- Conditional return mean (3 returns: rtb, xr, xb) ---
        mu_r_bill  = base_mu_r_i[0] + M_v_nodes[k_v, 0]
        mu_r_stock = base_mu_r_i[1] + M_v_nodes[k_v, 1]
        mu_r_bond  = base_mu_r_i[2] + M_v_nodes[k_v, 2]
        exp_mu_bill = exp(mu_r_bill)
        exp_mu_s = exp(mu_r_stock)
        exp_mu_b = exp(mu_r_bond)

        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            weight = w_v * p_ret
            if weight < prob_skip:
                continue

            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            sR_p = s_val * R_p
            if sR_p > 0.0:
                x_next = sR_p + pension_next_scalar
            else:
                x_next = pension_next_scalar

            # --- Trilinear interpolation of c_next and mpc ---
            c000, mpc000 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j000, :])
            c001, mpc001 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j001, :])
            c010, mpc010 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j010, :])
            c011, mpc011 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j011, :])
            c100, mpc100 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j100, :])
            c101, mpc101 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j101, :])
            c110, mpc110 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j110, :])
            c111, mpc111 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[j111, :])

            c_next = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011
                      + w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)
            c_next = max(c_next, min_consumption)

            mpc = (w000 * mpc000 + w001 * mpc001 + w010 * mpc010 + w011 * mpc011
                   + w100 * mpc100 + w101 * mpc101 + w110 * mpc110 + w111 * mpc111)
            mpc = max(0.0, min(1.0, mpc))

            # --- Marginal utilities ---
            mu_alive = c_next ** (-gamma)
            mup_alive = -gamma * mu_alive / c_next * mpc
            if sR_p > 0.0:
                w_A = sR_p / annuity_factor_is
                mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
                mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
            else:
                mu_bequest = 0.0
                mup_bequest = 0.0
            mu_comb = psi * mu_alive + prob_death * mu_bequest
            mup_comb = psi * mup_alive + prob_death * mup_bequest

            wmu = weight * mu_comb
            wmup = weight * mup_comb

            euler_sum += wmu * R_p
            foc_s += wmu * Rex_s
            foc_b += wmu * Rex_b

            jac = wmup * s_val
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum


# =============================================================================
# FOC AND JACOBIAN -- WORKING AGE (QUADRATURE)
# =============================================================================


@njit(fastmath=True)
def compute_foc_jac_working_quad(
    alpha_s, alpha_b, s_val, z_idx, i_s,
    wealth_grid, c_next_full, log_det_next,
    annuity_factor_is,
    z_grid, rho, eta_nodes, eta_weights, dz,
    # --- State quadrature arrays ---
    v_nodes, v_weights, M_v_nodes,
    base_mu_r_i,
    Phi_0_state, Phi_11, state_grid_i,
    state_bracket_shift, state_bracket_L_inv,
    grids_0, grids_1, grids_2,
    N1, N2,
    # --- Return quadrature ---
    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
    eps_nodes, eps_weights,
    # --- Model parameters ---
    gamma, psi, beta, b_bar,
    # --- Work-to-retirement transition (active only when t+1 == retire_age) ---
    use_pension_next, pension_next_by_z,
    min_wealth_inv=1e-10, min_consumption=1e-10,
    prob_skip=1e-12,
):
    """FOC and Jacobian for working-age portfolio with state innovation quadrature.

    Same structure as retirement version but adds income quadrature loops
    (eta for persistent, eps for transitory). Z-interpolation uses Catmull-Rom
    cubic wrapped inside trilinear state interpolation.

    At the work-to-retirement transition year (`t = retire_age - 1`), set
    `use_pension_next=True` and pass `pension_next_by_z = pension_table[t+1, :]`.
    The eps loop is preserved (eps_weights sum to 1) but next-period income is
    replaced by linear z-interpolation of the next-period pension table.
    """
    a_bill = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi

    foc_s = 0.0; foc_b = 0.0
    J_ss = 0.0; J_bb = 0.0; J_sb = 0.0
    euler_sum = 0.0

    n_state_quad = len(v_weights)
    n_z = len(z_grid)
    n_eta = len(eta_nodes)
    n_eps = len(eps_nodes)
    n_ret_quad = len(ret_weights)

    # Precompute exp(eps) and exp(eta)
    exp_eps = np.empty(n_eps)
    for ie in range(n_eps):
        exp_eps[ie] = exp(eps_nodes[ie])

    exp_eta = np.empty(n_eta)
    for ke in range(n_eta):
        exp_eta[ke] = exp(eta_nodes[ke])

    base_det_z = exp(log_det_next + rho * z_grid[z_idx])

    for k_v in range(n_state_quad):
        w_v = v_weights[k_v]
        if w_v < prob_skip:
            continue

        # --- Continuous next state ---
        s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * state_grid_i[0] + Phi_11[0, 1] * state_grid_i[1] + Phi_11[0, 2] * state_grid_i[2] + v_nodes[k_v, 0]
        s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * state_grid_i[0] + Phi_11[1, 1] * state_grid_i[1] + Phi_11[1, 2] * state_grid_i[2] + v_nodes[k_v, 1]
        s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * state_grid_i[0] + Phi_11[2, 1] * state_grid_i[1] + Phi_11[2, 2] * state_grid_i[2] + v_nodes[k_v, 2]

        b_next_0, b_next_1, b_next_2 = transform_state_for_bracketing_3d(
            s_next_0, s_next_1, s_next_2, state_bracket_shift, state_bracket_L_inv
        )

        # --- Bracket transformed next state in 3D grid ---
        lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
            b_next_0, b_next_1, b_next_2, grids_0, grids_1, grids_2
        )
        hi0 = lo0 if len(grids_0) == 1 else lo0 + 1
        hi1 = lo1 if len(grids_1) == 1 else lo1 + 1
        hi2 = lo2 if len(grids_2) == 1 else lo2 + 1

        # 8 trilinear weights
        w000 = (1.0 - f0) * (1.0 - f1) * (1.0 - f2)
        w001 = (1.0 - f0) * (1.0 - f1) * f2
        w010 = (1.0 - f0) * f1 * (1.0 - f2)
        w011 = (1.0 - f0) * f1 * f2
        w100 = f0 * (1.0 - f1) * (1.0 - f2)
        w101 = f0 * (1.0 - f1) * f2
        w110 = f0 * f1 * (1.0 - f2)
        w111 = f0 * f1 * f2

        # 8 flat indices
        j000 = lo0 * N1 * N2 + lo1 * N2 + lo2
        j001 = lo0 * N1 * N2 + lo1 * N2 + hi2
        j010 = lo0 * N1 * N2 + hi1 * N2 + lo2
        j011 = lo0 * N1 * N2 + hi1 * N2 + hi2
        j100 = hi0 * N1 * N2 + lo1 * N2 + lo2
        j101 = hi0 * N1 * N2 + lo1 * N2 + hi2
        j110 = hi0 * N1 * N2 + hi1 * N2 + lo2
        j111 = hi0 * N1 * N2 + hi1 * N2 + hi2

        # --- Conditional return mean (3 returns: rtb, xr, xb) ---
        mu_r_bill  = base_mu_r_i[0] + M_v_nodes[k_v, 0]
        mu_r_stock = base_mu_r_i[1] + M_v_nodes[k_v, 1]
        mu_r_bond  = base_mu_r_i[2] + M_v_nodes[k_v, 2]
        exp_mu_bill = exp(mu_r_bill)
        exp_mu_s = exp(mu_r_stock)
        exp_mu_b = exp(mu_r_bond)

        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            p_state_ret = w_v * p_ret
            if p_state_ret < prob_skip:
                continue

            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            sR_p = s_val * R_p
            if sR_p > 0.0:
                w_inv = sR_p
                w_A = w_inv / annuity_factor_is
                mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
                mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)

                # Bequest contribution (once per (k_v, k_r), independent of income)
                death_mu = p_state_ret * prob_death * mu_bequest
                death_mup = p_state_ret * prob_death * mup_bequest

                euler_sum += death_mu * R_p
                foc_s += death_mu * Rex_s
                foc_b += death_mu * Rex_b

                jac_b = death_mup * s_val
                J_ss += jac_b * Rex_s * Rex_s
                J_bb += jac_b * Rex_b * Rex_b
                J_sb += jac_b * Rex_s * Rex_b
            else:
                # Bankruptcy: heirs inherit nothing; alive branch uses w_inv = 0.
                w_inv = 0.0

            # Alive contribution: quadrature over persistent and transitory innovations
            for k_eta in range(n_eta):
                w_eta = eta_weights[k_eta]
                if w_eta < prob_skip:
                    continue

                z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]

                iz_lo = int((z_next - z_grid[0]) / dz)
                iz_lo = max(0, min(iz_lo, n_z - 2))
                frac_z = (z_next - z_grid[iz_lo]) / dz
                frac_z = max(0.0, min(1.0, frac_z))

                use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)

                p_out_base = p_state_ret * w_eta
                det_z_eta = base_det_z * exp_eta[k_eta]

                # At work->retirement boundary, next-period income = pension(z_next).
                # Linear interpolation in z reuses iz_lo / frac_z above; result is
                # constant across i_e (retirement has no transitory shock).
                if use_pension_next:
                    income_next_const = (1.0 - frac_z) * pension_next_by_z[iz_lo] + frac_z * pension_next_by_z[iz_lo + 1]
                else:
                    income_next_const = 0.0

                for i_e in range(n_eps):
                    weight = p_out_base * eps_weights[i_e]

                    if use_pension_next:
                        income_next = income_next_const
                    else:
                        y_gross_next = det_z_eta * exp_eps[i_e]
                        income_next = scalar_disposable_income(y_gross_next)
                    x_next = w_inv + income_next

                    iw, frac_w, inv_dw = find_bracket(x_next, wealth_grid)

                    # Trilinear blend of z-wealth interpolated values at 8 corners
                    c000, mpc000 = _interp_z_wealth(c_next_full, j000, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c001, mpc001 = _interp_z_wealth(c_next_full, j001, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c010, mpc010 = _interp_z_wealth(c_next_full, j010, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c011, mpc011 = _interp_z_wealth(c_next_full, j011, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c100, mpc100 = _interp_z_wealth(c_next_full, j100, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c101, mpc101 = _interp_z_wealth(c_next_full, j101, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c110, mpc110 = _interp_z_wealth(c_next_full, j110, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)
                    c111, mpc111 = _interp_z_wealth(c_next_full, j111, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_consumption)

                    c_next = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011
                              + w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)
                    c_next = max(c_next, min_consumption)

                    mpc = (w000 * mpc000 + w001 * mpc001 + w010 * mpc010 + w011 * mpc011
                           + w100 * mpc100 + w101 * mpc101 + w110 * mpc110 + w111 * mpc111)
                    mpc = max(0.0, min(1.0, mpc))

                    mu_alive = c_next ** (-gamma)
                    mup_alive = -gamma * mu_alive / c_next * mpc

                    wmu = weight * psi * mu_alive
                    wmup = weight * psi * mup_alive

                    euler_sum += wmu * R_p
                    foc_s += wmu * Rex_s
                    foc_b += wmu * Rex_b

                    jac = wmup * s_val
                    J_ss += jac * Rex_s * Rex_s
                    J_bb += jac * Rex_b * Rex_b
                    J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum


def _terminal_prepare_scenarios(state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights):
    """Scenario weights and gross returns for the exact terminal objective.

    Parameters
    ----------
    state_weights : (n_scenarios,)  -- quadrature weights (v_weights) or Pi_state row
    Rx_bill : (n_scenarios, n_ret_quad)  -- gross real bill returns
    Rx_stock_mult : (n_scenarios, n_ret_quad)  -- exp(mu_xr + eps_xr), multiply by R_bill
    Rx_bond_mult : (n_scenarios, n_ret_quad)  -- exp(mu_xb + eps_xb), multiply by R_bill
    ret_weights : (n_ret_quad,)
    """
    scenario_weights = np.asarray(state_weights, dtype=float)[:, None] * np.asarray(ret_weights, dtype=float)[None, :]
    R_bill = np.asarray(Rx_bill, dtype=float)
    R_stock = R_bill * np.asarray(Rx_stock_mult, dtype=float)
    R_bond = R_bill * np.asarray(Rx_bond_mult, dtype=float)
    Rex_s = R_stock - R_bill
    Rex_b = R_bond - R_bill
    return scenario_weights, R_bill, R_stock, R_bond, Rex_s, Rex_b


def _build_terminal_quad_returns(i_s, state_grid, const_r, A_r, M_v_nodes, ret_nodes):
    """Build gross return scenario arrays using state quadrature.

    For current financial state i_s, computes conditional return means at each
    state innovation quadrature node, then expands with return residual nodes.

    Returns
    -------
    Rx_bill : (n_state_quad, n_ret_quad)  -- exp(mu_rtb + eps_rtb)
    Rx_stock_mult : (n_state_quad, n_ret_quad)  -- exp(mu_xr + eps_xr)
    Rx_bond_mult : (n_state_quad, n_ret_quad)  -- exp(mu_xb + eps_xb)
    """
    n_state_quad = M_v_nodes.shape[0]
    n_ret_quad = ret_nodes.shape[0]

    # base conditional return mean: const_r + A_r @ s_i  (shape (3,))
    base_mu_r = const_r + A_r @ state_grid[i_s]

    Rx_bill = np.empty((n_state_quad, n_ret_quad))
    Rx_stock_mult = np.empty((n_state_quad, n_ret_quad))
    Rx_bond_mult = np.empty((n_state_quad, n_ret_quad))

    for k_v in range(n_state_quad):
        mu_r_k = base_mu_r + M_v_nodes[k_v]  # (3,): [mu_rtb, mu_xr, mu_xb]
        for k_r in range(n_ret_quad):
            Rx_bill[k_v, k_r] = np.exp(mu_r_k[0] + ret_nodes[k_r, 0])
            Rx_stock_mult[k_v, k_r] = np.exp(mu_r_k[1] + ret_nodes[k_r, 1])
            Rx_bond_mult[k_v, k_r] = np.exp(mu_r_k[2] + ret_nodes[k_r, 2])

    return Rx_bill, Rx_stock_mult, Rx_bond_mult



# =============================================================================
# TERMINAL AGE PORTFOLIO SOLVER
# =============================================================================

@njit(fastmath=True)
def compute_terminal_portfolio_foc_jac(alpha_s, alpha_b,
                                        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
                                        gamma,
                                          min_return_power=1e-15,
                                          prob_skip=1e-12):
    """
    FOC and Jacobian for the terminal-period portfolio problem.

    Because the bequest b(a*R_port, A) = b_bar*(a*R_port/A)^{1-gamma}/(1-gamma)
    is CRRA in terminal wealth, the portfolio FOC is proportional to a^{1-gamma}
    and therefore independent of savings a (hence independent of W and c).

    Uses state quadrature (v_weights) for scenario weighting, consistent with
    the rest of the solver.

    FOC_k = sum_{k_v,k_r} w_v * w_r * R_port^{-gamma} * (R_k - R_bill) = 0
    J_kl  = sum_{k_v,k_r} w_v * w_r * (-gamma) * R_port^{-gamma-1} * Rex_k * Rex_l

    Parameters
    ----------
    state_weights : (n_state_quad,) -- v_weights from state quadrature
    Rx_bill : (n_state_quad, n_ret_quad) -- exp(mu_rtb + eps_rtb)
    Rx_stock_mult : (n_state_quad, n_ret_quad) -- exp(mu_xr + eps_xr)
    Rx_bond_mult : (n_state_quad, n_ret_quad) -- exp(mu_xb + eps_xb)
    """
    foc_s = 0.0;  foc_b = 0.0
    J_ss  = 0.0;  J_bb  = 0.0;  J_sb  = 0.0
    euler_sum = 0.0
    a_bill = 1.0 - alpha_s - alpha_b

    n_state_quad = len(state_weights)
    n_ret_quad = len(ret_weights)
    for k_v in range(n_state_quad):
        w_v = state_weights[k_v]
        if w_v < prob_skip:
            continue
        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            if p_ret < prob_skip:
                continue
            weight = w_v * p_ret
            if weight < prob_skip:
                continue

            R_bill_kr = Rx_bill[k_v, k_r]
            R_s   = R_bill_kr * Rx_stock_mult[k_v, k_r]
            R_b   = R_bill_kr * Rx_bond_mult[k_v, k_r]
            R_p   = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill_kr
            Rex_s = R_s - R_bill_kr
            Rex_b = R_b - R_bill_kr

            Rp_mg  = max(R_p, min_return_power) ** (-gamma)
            Rp_mg1 = max(R_p, min_return_power) ** (-gamma - 1.0)

            foc_s += weight * Rp_mg * Rex_s
            foc_b += weight * Rp_mg * Rex_b
            euler_sum += weight * Rp_mg * R_p   # = weight * R_p^{1-gamma}

            jac   = weight * (-gamma) * Rp_mg1
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum


# =============================================================================
# CONSTRAINED TERMINAL PORTFOLIO SOLVER (NJIT)
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_2d_terminal_constrained_njit(
        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
        gamma,
        init_s=0.1, init_b=0.4,
        tol=1e-7, max_iter=20,
        corner_tol=1e-8,
        edge_max_iter=8, edge_accept_factor=10.0,
        singular_det=1e-15, grad_step_size=0.05,
        step_damp=0.2, grad_denom_eps=1e-10,
        min_return_power=1e-15, prob_skip=1e-12):
    """
    Constrained 2D Newton for the terminal portfolio problem.

    Mirrors solve_portfolio_2d_retirement_quad but without continuation value,
    income, or state interpolation — only the bequest FOC matters.

    Corner → edge → interior Newton with projection onto the simplex.

    Returns: (alpha_s, alpha_b, euler_sum, exit_code, normalized_error)
    """
    # --- Helper: evaluate terminal FOC at (a_s, a_b) ---
    def _foc(a_s, a_b):
        return compute_terminal_portfolio_foc_jac(
            a_s, a_b, state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
            ret_weights, gamma, min_return_power, prob_skip)

    # Corner: all bills
    fs0, fb0, _, _, _, e0 = _foc(0.0, 0.0)
    scale = max(abs(e0), 1.0)
    if fs0 <= corner_tol * scale and fb0 <= corner_tol * scale:
        return 0.0, 0.0, e0, EC_CORNER_BILLS, 0.0

    # Corner: all stocks
    fs1, fb1, _, _, _, e1 = _foc(1.0, 0.0)
    if fs1 >= -corner_tol * scale and fb1 <= fs1 + corner_tol * scale:
        return 1.0, 0.0, e1, EC_CORNER_STOCKS, 0.0

    # Corner: all bonds
    fs2, fb2, _, _, _, e2 = _foc(0.0, 1.0)
    if fb2 >= -corner_tol * scale and fs2 <= fb2 + corner_tol * scale:
        return 0.0, 1.0, e2, EC_CORNER_BONDS, 0.0

    # Edge: stocks + bills (alpha_b = 0)
    if fs0 > 0.0 and fs1 < 0.0:
        a_s = fs0 / (fs0 - fs1)
        fs = fs0
        for _ in range(edge_max_iter):
            fs, fb, Jss, _, _, e = _foc(a_s, 0.0)
            if abs(fs) < tol * scale:
                break
            if abs(Jss) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - fs / Jss))
        if abs(fs) < tol * scale * edge_accept_factor and fb <= tol * scale:
            return a_s, 0.0, e, EC_EDGE_SB, abs(fs) / scale

    # Edge: bonds + bills (alpha_s = 0)
    if fb0 > 0.0 and fb2 < 0.0:
        a_b = fb0 / (fb0 - fb2)
        fb = fb0
        for _ in range(edge_max_iter):
            fs, fb, _, Jbb, _, e = _foc(0.0, a_b)
            if abs(fb) < tol * scale:
                break
            if abs(Jbb) < singular_det:
                break
            a_b = max(0.0, min(1.0, a_b - fb / Jbb))
        if abs(fb) < tol * scale * edge_accept_factor and fs <= tol * scale:
            return 0.0, a_b, e, EC_EDGE_BB, abs(fb) / scale

    # Edge: stocks + bonds (alpha_bill = 0, alpha_s + alpha_b = 1)
    g1 = fs1 - fb1
    g2 = fs2 - fb2
    if g1 * g2 < 0.0:
        a_s = g2 / (g2 - g1)
        g = g2
        for _ in range(edge_max_iter):
            a_b = 1.0 - a_s
            fs, fb, Jss, Jbb, Jsb, e = _foc(a_s, a_b)
            g = fs - fb
            if abs(g) < tol * scale:
                break
            dg = Jss - 2.0 * Jsb + Jbb
            if abs(dg) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - g / dg))
        if abs(fs - fb) < tol * scale * edge_accept_factor and fs >= -tol * scale:
            return a_s, 1.0 - a_s, e, EC_EDGE_STOCKBOND, abs(g) / scale

    # Interior Newton with projection
    a_s = init_s
    a_b = init_b
    e_last = 0.0
    err = 1.0

    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb, e_sum = _foc(a_s, a_b)
        e_last = e_sum

        err = (fs * fs + fb * fb) ** 0.5
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d = 1.0 / det
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
# UNCONSTRAINED TERMINAL PORTFOLIO SOLVER (NJIT)
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_unconstrained_terminal_njit(
        state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights,
        gamma,
        init_s=0.1, init_b=0.4,
        tol=1e-7, max_iter=30,
        singular_det=1e-15, grad_step_size=0.05,
        step_damp=0.3, grad_denom_eps=1e-10,
        min_return_power=1e-15, prob_skip=1e-12,
        use_line_search=True, max_backtrack_iter=10,
        line_search_max_step=2.0,
        alpha_min=-1e30, alpha_max=+1e30):
    """
    Unconstrained 2D Newton for the terminal portfolio problem.

    Mirrors solve_portfolio_unconstrained_retirement_quad but without
    continuation value, income, or state interpolation.

    Interior Newton with optional backtracking line search, no simplex
    projection (allows leverage and short-selling).

    Returns: (alpha_s, alpha_b, euler_sum, exit_code, normalized_error)
    """
    def _foc(a_s, a_b):
        return compute_terminal_portfolio_foc_jac(
            a_s, a_b, state_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult,
            ret_weights, gamma, min_return_power, prob_skip)

    # Scale from all-bills evaluation
    _, _, _, _, _, e0 = _foc(0.0, 0.0)
    scale = max(abs(e0), 1.0)

    a_s = init_s
    a_b = init_b

    fs, fb, Jss, Jbb, Jsb, e_last = _foc(a_s, a_b)
    err = (fs * fs + fb * fb) ** 0.5

    # n_iter = number of outer Newton iterations actually entered
    # (= 0 if init was already inside tol, = max_iter on full exhaustion).
    for k in range(max_iter):
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale, k

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        if use_line_search:
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > line_search_max_step:
                cap = line_search_max_step / slen
                step_s *= cap
                step_b *= cap

            alpha = 1.0
            found = False
            for _bt in range(max_backtrack_iter):
                a_s_t = a_s + alpha * step_s
                a_b_t = a_b + alpha * step_b
                if a_s_t < alpha_min: a_s_t = alpha_min
                elif a_s_t > alpha_max: a_s_t = alpha_max
                if a_b_t < alpha_min: a_b_t = alpha_min
                elif a_b_t > alpha_max: a_b_t = alpha_max
                fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = _foc(a_s_t, a_b_t)
                err_t = (fs_t * fs_t + fb_t * fb_t) ** 0.5
                if err_t < err:
                    fs = fs_t; fb = fb_t; Jss = Jss_t; Jbb = Jbb_t; Jsb = Jsb_t
                    e_last = e_t; err = err_t
                    a_s = a_s_t; a_b = a_b_t
                    found = True
                    break
                alpha *= 0.5
            if not found:
                return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale, k + 1
        else:
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > step_damp:
                cap = step_damp / slen
                step_s *= cap
                step_b *= cap
            a_s += step_s
            a_b += step_b
            if a_s < alpha_min: a_s = alpha_min
            elif a_s > alpha_max: a_s = alpha_max
            if a_b < alpha_min: a_b = alpha_min
            elif a_b > alpha_max: a_b = alpha_max
            fs, fb, Jss, Jbb, Jsb, e_last = _foc(a_s, a_b)
            err = (fs * fs + fb * fb) ** 0.5

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale, max_iter


def solve_terminal_age(wealth_grid, annuity_factors,
                       state_grid, const_r, A_r, M_v_nodes, v_weights,
                       ret_nodes, ret_weights,
                       gamma, beta, b_bar, N_state, n_z, constrained=True, solver_config=None,
                       min_return_power=1e-15, min_consumption=1e-10):
    """Solve the terminal age using state quadrature (consistent with rest of solver).

    Integrates over state innovation quadrature nodes (v_nodes via M_v_nodes)
    and return residual nodes (ret_nodes) to compute E[R_port^(1-gamma)].
    """
    if solver_config is None:
        solver_config = SolverConfig()

    n_w = len(wealth_grid)
    out_c = np.empty((n_z, N_state, n_w))
    out_alpha_s = np.empty((n_z, N_state, n_w))
    out_alpha_b = np.empty((n_z, N_state, n_w))
    terminal_diag_int = np.zeros(N_state, dtype=np.int64)

    for i_s in range(N_state):
        A_is = annuity_factors[i_s]

        # Build return scenario arrays from state quadrature
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, state_grid, const_r, A_r, M_v_nodes, ret_nodes
        )

        if constrained:
            opt_s, opt_b, moment, exit_code, foc_resid = solve_portfolio_2d_terminal_constrained_njit(
                v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights, gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=solver_config.max_iter,
            )
        else:
            # Unconstrained returns 6-tuple including n_newton_iter; discarded here
            # because the terminal step is N_state calls — diagnostic value is in
            # the period (working/retirement) solvers which run ~566k calls each.
            opt_s, opt_b, moment, exit_code, foc_resid, _ = solve_portfolio_unconstrained_terminal_njit(
                v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, ret_weights, gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=solver_config.max_iter_unconstrained,
                singular_det=solver_config.singular_det,
                grad_step_size=solver_config.grad_step_size,
                step_damp=solver_config.step_damp_unconstrained,
                grad_denom_eps=solver_config.grad_denom_eps,
                min_return_power=solver_config.min_return_power,
                prob_skip=solver_config.prob_skip_threshold,
                use_line_search=solver_config.use_line_search,
                max_backtrack_iter=solver_config.max_backtrack_iter,
                line_search_max_step=solver_config.line_search_max_step,
                alpha_min=solver_config.alpha_min,
                alpha_max=solver_config.alpha_max,
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
# NEWTON PORTFOLIO SOLVER -- RETIREMENT (QUADRATURE)
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_2d_retirement_quad(s_val, z_idx, i_s,
                                       wealth_grid, c_next_full, pension_next_scalar,
                                       annuity_factor_is,
                                       v_nodes, v_weights, M_v_nodes,
                                       base_mu_r_i,
                                       Phi_0_state, Phi_11, state_grid_i,
                                       state_bracket_shift, state_bracket_L_inv,
                                       grids_0, grids_1, grids_2, N1, N2,
                                       exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                       gamma, psi, beta, b_bar,
                                       init_s=0.1, init_b=0.4,
                                       tol=1e-7, max_iter=20,
                                       tiny_savings=1e-6, corner_tol=1e-8,
                                       edge_max_iter=8, edge_accept_factor=10.0,
                                       singular_det=1e-15, grad_step_size=0.05,
                                       step_damp=0.2, grad_denom_eps=1e-10,
                                       min_wealth_inv=1e-10, min_consumption=1e-10,
                                       prob_skip=1e-12):
    """2D Newton for retirement portfolio using state quadrature."""



    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_retirement_quad(
            0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
            v_nodes, v_weights, M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11, state_grid_i,
            state_bracket_shift, state_bracket_L_inv,
            grids_0, grids_1, grids_2, N1, N2,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0

    # Corner: all bills
    fs0, fb0, _, _, _, e0 = compute_foc_jac_retirement_quad(
        0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)
    if fs0 <= corner_tol * scale and fb0 <= corner_tol * scale:
        return 0.0, 0.0, e0, EC_CORNER_BILLS, 0.0

    # Corner: all stocks
    fs1, fb1, _, _, _, e1 = compute_foc_jac_retirement_quad(
        1.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    if fs1 >= -corner_tol * scale and fb1 <= fs1 + corner_tol * scale:
        return 1.0, 0.0, e1, EC_CORNER_STOCKS, 0.0

    # Corner: all bonds
    fs2, fb2, _, _, _, e2 = compute_foc_jac_retirement_quad(
        0.0, 1.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    if fb2 >= -corner_tol * scale and fs2 <= fb2 + corner_tol * scale:
        return 0.0, 1.0, e2, EC_CORNER_BONDS, 0.0

    # Edge: stocks + bills
    if fs0 > 0.0 and fs1 < 0.0:
        a_s = fs0 / (fs0 - fs1)
        fs = fs0
        for _ in range(edge_max_iter):
            fs, fb, Jss, _, _, e = compute_foc_jac_retirement_quad(
                a_s, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            if abs(fs) < tol * scale:
                break
            if abs(Jss) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - fs / Jss))
        if abs(fs) < tol * scale * edge_accept_factor and fb <= tol * scale:
            return a_s, 0.0, e, EC_EDGE_SB, abs(fs) / scale

    # Edge: bonds + bills
    if fb0 > 0.0 and fb2 < 0.0:
        a_b = fb0 / (fb0 - fb2)
        fb = fb0
        for _ in range(edge_max_iter):
            fs, fb, _, Jbb, _, e = compute_foc_jac_retirement_quad(
                0.0, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            if abs(fb) < tol * scale:
                break
            if abs(Jbb) < singular_det:
                break
            a_b = max(0.0, min(1.0, a_b - fb / Jbb))
        if abs(fb) < tol * scale * edge_accept_factor and fs <= tol * scale:
            return 0.0, a_b, e, EC_EDGE_BB, abs(fb) / scale

    # Edge: stocks + bonds
    g1 = fs1 - fb1
    g2 = fs2 - fb2
    if g1 * g2 < 0.0:
        a_s = g2 / (g2 - g1)
        g = g2
        for _ in range(edge_max_iter):
            a_b = 1.0 - a_s
            fs, fb, Jss, Jbb, Jsb, e = compute_foc_jac_retirement_quad(
                a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                gamma, psi, beta, b_bar,
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

    # Interior Newton
    a_s = init_s
    a_b = init_b
    e_last = 0.0
    err = 1.0

    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb, e_sum = compute_foc_jac_retirement_quad(
            a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
            v_nodes, v_weights, M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11, state_grid_i,
            state_bracket_shift, state_bracket_L_inv,
            grids_0, grids_1, grids_2, N1, N2,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            gamma, psi, beta, b_bar,
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
            inv_d = 1.0 / det
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
# UNCONSTRAINED PORTFOLIO SOLVER -- RETIREMENT (QUADRATURE)
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_unconstrained_retirement_quad(s_val, z_idx, i_s,
                                                   wealth_grid, c_next_full, pension_next_scalar,
                                                   annuity_factor_is,
                                                   v_nodes, v_weights, M_v_nodes,
                                                   base_mu_r_i,
                                                   Phi_0_state, Phi_11, state_grid_i,
                                                   state_bracket_shift, state_bracket_L_inv,
                                                   grids_0, grids_1, grids_2, N1, N2,
                                                   exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                                   gamma, psi, beta, b_bar,
                                                   init_s=0.1, init_b=0.4,
                                                   tol=1e-7, max_iter=30,
                                                   tiny_savings=1e-6,
                                                   singular_det=1e-15, grad_step_size=0.05,
                                                   step_damp=0.3, grad_denom_eps=1e-10,
                                                   min_wealth_inv=1e-10, min_consumption=1e-10,
                                                   prob_skip=1e-12,
                                                   use_line_search=True, max_backtrack_iter=10,
                                                   line_search_max_step=2.0,
                                                   alpha_min=-1e30, alpha_max=+1e30):
    """Unconstrained Newton for retirement portfolio using state quadrature."""



    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_retirement_quad(
            0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
            v_nodes, v_weights, M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11, state_grid_i,
            state_bracket_shift, state_bracket_L_inv,
            grids_0, grids_1, grids_2, N1, N2,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            gamma, psi, beta, b_bar,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0, 0

    _, _, _, _, _, e0 = compute_foc_jac_retirement_quad(
        0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)

    a_s = init_s
    a_b = init_b

    fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_retirement_quad(
        a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi, beta, b_bar,
        min_wealth_inv, min_consumption, prob_skip)
    err = (fs * fs + fb * fb) ** 0.5

    # n_iter = number of outer Newton iterations actually entered
    # (= 0 if init was already inside tol, = max_iter on full exhaustion).
    for k in range(max_iter):
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale, k

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        if use_line_search:
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > line_search_max_step:
                cap = line_search_max_step / slen
                step_s *= cap
                step_b *= cap

            alpha = 1.0
            found = False
            for _bt in range(max_backtrack_iter):
                a_s_t = a_s + alpha * step_s
                a_b_t = a_b + alpha * step_b
                if a_s_t < alpha_min: a_s_t = alpha_min
                elif a_s_t > alpha_max: a_s_t = alpha_max
                if a_b_t < alpha_min: a_b_t = alpha_min
                elif a_b_t > alpha_max: a_b_t = alpha_max
                fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = compute_foc_jac_retirement_quad(
                    a_s_t, a_b_t, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                    v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                    Phi_0_state, Phi_11, state_grid_i,
                    state_bracket_shift, state_bracket_L_inv,
                    grids_0, grids_1, grids_2, N1, N2,
                    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                    gamma, psi, beta, b_bar,
                    min_wealth_inv, min_consumption, prob_skip)
                err_t = (fs_t * fs_t + fb_t * fb_t) ** 0.5
                if err_t < err:
                    fs = fs_t; fb = fb_t; Jss = Jss_t; Jbb = Jbb_t; Jsb = Jsb_t
                    e_last = e_t; err = err_t
                    a_s = a_s_t; a_b = a_b_t
                    found = True
                    break
                alpha *= 0.5
            if not found:
                return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale, k + 1
        else:
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > step_damp:
                cap = step_damp / slen
                step_s *= cap
                step_b *= cap
            a_s += step_s
            a_b += step_b
            if a_s < alpha_min: a_s = alpha_min
            elif a_s > alpha_max: a_s = alpha_max
            if a_b < alpha_min: a_b = alpha_min
            elif a_b > alpha_max: a_b = alpha_max
            fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_retirement_quad(
                a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, pension_next_scalar, annuity_factor_is,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                gamma, psi, beta, b_bar,
                min_wealth_inv, min_consumption, prob_skip)
            err = (fs * fs + fb * fb) ** 0.5

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale, max_iter


# =============================================================================
# NEWTON PORTFOLIO SOLVER -- WORKING AGE (QUADRATURE)
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_2d_working_quad(s_val, z_idx, i_s,
                                     wealth_grid, c_next_full, log_det_next,
                                     annuity_factor_is,
                                     z_grid, rho, eta_nodes, eta_weights, dz,
                                     v_nodes, v_weights, M_v_nodes,
                                     base_mu_r_i,
                                     Phi_0_state, Phi_11, state_grid_i,
                                     state_bracket_shift, state_bracket_L_inv,
                                     grids_0, grids_1, grids_2, N1, N2,
                                     exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                     eps_nodes, eps_weights,
                                     gamma, psi, beta, b_bar,
                                     use_pension_next, pension_next_by_z,
                                     init_s=0.1, init_b=0.4,
                                     tol=1e-7, max_iter=20,
                                     tiny_savings=1e-6, corner_tol=1e-8,
                                     edge_max_iter=8, edge_accept_factor=10.0,
                                     singular_det=1e-15, grad_step_size=0.05,
                                     step_damp=0.2, grad_denom_eps=1e-10,
                                     min_wealth_inv=1e-10, min_consumption=1e-10,
                                     prob_skip=1e-12):
    """2D Newton for working-age portfolio using state quadrature."""



    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_working_quad(
            0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
            z_grid, rho, eta_nodes, eta_weights, dz,
            v_nodes, v_weights, M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11, state_grid_i,
            state_bracket_shift, state_bracket_L_inv,
            grids_0, grids_1, grids_2, N1, N2,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            eps_nodes, eps_weights,
            gamma, psi, beta, b_bar,
            use_pension_next, pension_next_by_z,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0

    fs0, fb0, _, _, _, e0 = compute_foc_jac_working_quad(
        0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        eps_nodes, eps_weights,
        gamma, psi, beta, b_bar,
        use_pension_next, pension_next_by_z,
        min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)
    if fs0 <= corner_tol * scale and fb0 <= corner_tol * scale:
        return 0.0, 0.0, e0, EC_CORNER_BILLS, 0.0

    fs1, fb1, _, _, _, e1 = compute_foc_jac_working_quad(
        1.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        eps_nodes, eps_weights,
        gamma, psi, beta, b_bar,
        use_pension_next, pension_next_by_z,
        min_wealth_inv, min_consumption, prob_skip)
    if fs1 >= -corner_tol * scale and fb1 <= fs1 + corner_tol * scale:
        return 1.0, 0.0, e1, EC_CORNER_STOCKS, 0.0

    fs2, fb2, _, _, _, e2 = compute_foc_jac_working_quad(
        0.0, 1.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        eps_nodes, eps_weights,
        gamma, psi, beta, b_bar,
        use_pension_next, pension_next_by_z,
        min_wealth_inv, min_consumption, prob_skip)
    if fb2 >= -corner_tol * scale and fs2 <= fb2 + corner_tol * scale:
        return 0.0, 1.0, e2, EC_CORNER_BONDS, 0.0

    # Edge: stocks + bills
    if fs0 > 0.0 and fs1 < 0.0:
        a_s = fs0 / (fs0 - fs1)
        fs = fs0
        for _ in range(edge_max_iter):
            fs, fb, Jss, _, _, e = compute_foc_jac_working_quad(
                a_s, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                eps_nodes, eps_weights,
                gamma, psi, beta, b_bar,
                use_pension_next, pension_next_by_z,
                min_wealth_inv, min_consumption, prob_skip)
            if abs(fs) < tol * scale:
                break
            if abs(Jss) < singular_det:
                break
            a_s = max(0.0, min(1.0, a_s - fs / Jss))
        if abs(fs) < tol * scale * edge_accept_factor and fb <= tol * scale:
            return a_s, 0.0, e, EC_EDGE_SB, abs(fs) / scale

    # Edge: bonds + bills
    if fb0 > 0.0 and fb2 < 0.0:
        a_b = fb0 / (fb0 - fb2)
        fb = fb0
        for _ in range(edge_max_iter):
            fs, fb, _, Jbb, _, e = compute_foc_jac_working_quad(
                0.0, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                eps_nodes, eps_weights,
                gamma, psi, beta, b_bar,
                use_pension_next, pension_next_by_z,
                min_wealth_inv, min_consumption, prob_skip)
            if abs(fb) < tol * scale:
                break
            if abs(Jbb) < singular_det:
                break
            a_b = max(0.0, min(1.0, a_b - fb / Jbb))
        if abs(fb) < tol * scale * edge_accept_factor and fs <= tol * scale:
            return 0.0, a_b, e, EC_EDGE_BB, abs(fb) / scale

    # Edge: stocks + bonds
    g1 = fs1 - fb1
    g2 = fs2 - fb2
    if g1 * g2 < 0.0:
        a_s = g2 / (g2 - g1)
        g = g2
        for _ in range(edge_max_iter):
            a_b = 1.0 - a_s
            fs, fb, Jss, Jbb, Jsb, e = compute_foc_jac_working_quad(
                a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                eps_nodes, eps_weights,
                gamma, psi, beta, b_bar,
                use_pension_next, pension_next_by_z,
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

    # Interior Newton
    a_s = init_s
    a_b = init_b
    e_last = 0.0
    err = 1.0

    for _ in range(max_iter):
        fs, fb, Jss, Jbb, Jsb, e_sum = compute_foc_jac_working_quad(
            a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
            z_grid, rho, eta_nodes, eta_weights, dz,
            v_nodes, v_weights, M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11, state_grid_i,
            state_bracket_shift, state_bracket_L_inv,
            grids_0, grids_1, grids_2, N1, N2,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            eps_nodes, eps_weights,
            gamma, psi, beta, b_bar,
            use_pension_next, pension_next_by_z,
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
            inv_d = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        slen = (step_s * step_s + step_b * step_b) ** 0.5
        if slen > step_damp:
            sc_f = step_damp / slen
            step_s *= sc_f
            step_b *= sc_f

        a_s, a_b = project_to_triangle(a_s + step_s, a_b + step_b)

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale


# =============================================================================
# UNCONSTRAINED PORTFOLIO SOLVER -- WORKING AGE (QUADRATURE)
# =============================================================================

@njit(fastmath=True)
def solve_portfolio_unconstrained_working_quad(s_val, z_idx, i_s,
                                                wealth_grid, c_next_full, log_det_next,
                                                annuity_factor_is,
                                                z_grid, rho, eta_nodes, eta_weights, dz,
                                                v_nodes, v_weights, M_v_nodes,
                                                base_mu_r_i,
                                                Phi_0_state, Phi_11, state_grid_i,
                                                state_bracket_shift, state_bracket_L_inv,
                                                grids_0, grids_1, grids_2, N1, N2,
                                                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                                eps_nodes, eps_weights,
                                                gamma, psi, beta, b_bar,
                                                use_pension_next, pension_next_by_z,
                                                init_s=0.1, init_b=0.4,
                                                tol=1e-7, max_iter=30,
                                                tiny_savings=1e-6,
                                                singular_det=1e-15, grad_step_size=0.05,
                                                step_damp=0.3, grad_denom_eps=1e-10,
                                                min_wealth_inv=1e-10, min_consumption=1e-10,
                                                prob_skip=1e-12,
                                                use_line_search=True, max_backtrack_iter=10,
                                                line_search_max_step=2.0,
                                                alpha_min=-1e30, alpha_max=+1e30):
    """Unconstrained Newton for working-age portfolio using state quadrature."""

    # Line search is the global default — it prevents Newton overshoots and now
    # also applies cleanly at the work->retirement boundary because off-grid
    # wealth lookups use a value/slope-consistent affine extrapolation.
    eff_line_search = use_line_search

    if s_val < tiny_savings:
        _, _, _, _, _, e = compute_foc_jac_working_quad(
            0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
            z_grid, rho, eta_nodes, eta_weights, dz,
            v_nodes, v_weights, M_v_nodes, base_mu_r_i,
            Phi_0_state, Phi_11, state_grid_i,
            state_bracket_shift, state_bracket_L_inv,
            grids_0, grids_1, grids_2, N1, N2,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            eps_nodes, eps_weights,
            gamma, psi, beta, b_bar,
            use_pension_next, pension_next_by_z,
            min_wealth_inv, min_consumption, prob_skip)
        return 0.0, 0.0, e, EC_TINY_SAVINGS, 0.0, 0

    _, _, _, _, _, e0 = compute_foc_jac_working_quad(
        0.0, 0.0, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        eps_nodes, eps_weights,
        gamma, psi, beta, b_bar,
        use_pension_next, pension_next_by_z,
        min_wealth_inv, min_consumption, prob_skip)
    scale = max(abs(e0), 1.0)

    a_s = init_s
    a_b = init_b

    fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_working_quad(
        a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
        z_grid, rho, eta_nodes, eta_weights, dz,
        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
        Phi_0_state, Phi_11, state_grid_i,
        state_bracket_shift, state_bracket_L_inv,
        grids_0, grids_1, grids_2, N1, N2,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        eps_nodes, eps_weights,
        gamma, psi, beta, b_bar,
        use_pension_next, pension_next_by_z,
        min_wealth_inv, min_consumption, prob_skip)
    err = (fs * fs + fb * fb) ** 0.5

    # n_iter = number of outer Newton iterations actually entered
    # (= 0 if init was already inside tol, = max_iter on full exhaustion).
    for k in range(max_iter):
        if err < tol * scale:
            return a_s, a_b, e_last, EC_INTERIOR, err / scale, k

        det = Jss * Jbb - Jsb * Jsb
        if abs(det) < singular_det:
            step_s = grad_step_size * fs / (err + grad_denom_eps)
            step_b = grad_step_size * fb / (err + grad_denom_eps)
        else:
            inv_d = 1.0 / det
            step_s = -(Jbb * fs - Jsb * fb) * inv_d
            step_b = -(-Jsb * fs + Jss * fb) * inv_d

        if eff_line_search:
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > line_search_max_step:
                cap = line_search_max_step / slen
                step_s *= cap
                step_b *= cap

            alpha = 1.0
            found = False
            for _bt in range(max_backtrack_iter):
                a_s_t = a_s + alpha * step_s
                a_b_t = a_b + alpha * step_b
                if a_s_t < alpha_min: a_s_t = alpha_min
                elif a_s_t > alpha_max: a_s_t = alpha_max
                if a_b_t < alpha_min: a_b_t = alpha_min
                elif a_b_t > alpha_max: a_b_t = alpha_max
                fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = compute_foc_jac_working_quad(
                    a_s_t, a_b_t, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
                    z_grid, rho, eta_nodes, eta_weights, dz,
                    v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                    Phi_0_state, Phi_11, state_grid_i,
                    state_bracket_shift, state_bracket_L_inv,
                    grids_0, grids_1, grids_2, N1, N2,
                    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                    eps_nodes, eps_weights,
                    gamma, psi, beta, b_bar,
                    use_pension_next, pension_next_by_z,
                    min_wealth_inv, min_consumption, prob_skip)
                err_t = (fs_t * fs_t + fb_t * fb_t) ** 0.5
                if err_t < err:
                    fs = fs_t; fb = fb_t; Jss = Jss_t; Jbb = Jbb_t; Jsb = Jsb_t
                    e_last = e_t; err = err_t
                    a_s = a_s_t; a_b = a_b_t
                    found = True
                    break
                alpha *= 0.5
            if not found:
                return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale, k + 1
        else:
            slen = (step_s * step_s + step_b * step_b) ** 0.5
            if slen > step_damp:
                cap = step_damp / slen
                step_s *= cap
                step_b *= cap
            a_s += step_s
            a_b += step_b
            if a_s < alpha_min: a_s = alpha_min
            elif a_s > alpha_max: a_s = alpha_max
            if a_b < alpha_min: a_b = alpha_min
            elif a_b > alpha_max: a_b = alpha_max
            fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_working_quad(
                a_s, a_b, s_val, z_idx, i_s, wealth_grid, c_next_full, log_det_next, annuity_factor_is,
                z_grid, rho, eta_nodes, eta_weights, dz,
                v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                Phi_0_state, Phi_11, state_grid_i,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N1, N2,
                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                eps_nodes, eps_weights,
                gamma, psi, beta, b_bar,
                use_pension_next, pension_next_by_z,
                min_wealth_inv, min_consumption, prob_skip)
            err = (fs * fs + fb * fb) ** 0.5

    return a_s, a_b, e_last, EC_NEWTON_FAIL, err / scale, max_iter


# =============================================================================
# PERIOD SOLVER -- RETIREMENT (QUADRATURE)
# =============================================================================

@njit(parallel=True)
def _solve_retirement_step_quad_jit(wealth_grid, savings_grid, z_grid, N_state,
                                     c_next_full, pension_1d,
                                     annuity_factors,
                                     state_grid, grids_0, grids_1, grids_2,
                                     state_bracket_shift, state_bracket_L_inv,
                                     v_nodes, v_weights, M_v_nodes, const_r, A_r,
                                     Phi_0_state, Phi_11,
                                     exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                     gamma, psi_vec, beta, b_bar,
                                     constrained, solver_config,
                                     policy_c, policy_alpha_s, policy_alpha_b):
    """Solve one retirement period using quadrature over state innovations."""

    sc = solver_config
    n_z = len(z_grid)
    n_savings = len(savings_grid)
    n_wealth = len(wealth_grid)
    N1 = len(grids_1)
    N2 = len(grids_2)

    # Slot conventions: see DI_*/DF_* constants at top of solver.py.
    # Sizes are 14 (DI) and 10 (DF); slot 13 (DI_SUM_ITER) and slot 9 (DF_MAX_NEWTON_ITER)
    # are populated only on the unconstrained branch.
    diag_int = np.zeros((N_state, 15), dtype=np.int64)
    diag_float = np.zeros((N_state, 10))

    for i_s in prange(N_state):
        annuity_factor_is = annuity_factors[i_s]

        # Precompute base_mu_r_i for this state (3 returns: rtb, xr, xb)
        s_i = state_grid[i_s]
        base_mu_r_i = np.empty(3)
        base_mu_r_i[0] = const_r[0] + A_r[0, 0] * s_i[0] + A_r[0, 1] * s_i[1] + A_r[0, 2] * s_i[2]
        base_mu_r_i[1] = const_r[1] + A_r[1, 0] * s_i[0] + A_r[1, 1] * s_i[1] + A_r[1, 2] * s_i[2]
        base_mu_r_i[2] = const_r[2] + A_r[2, 0] * s_i[0] + A_r[2, 1] * s_i[1] + A_r[2, 2] * s_i[2]

        last_a_s = sc.init_alpha_s
        last_a_b = sc.init_alpha_b

        diag_float[i_s, 3] = 2.0
        diag_float[i_s, 5] = 2.0

        temp_x = np.empty(n_savings + 1)
        temp_c = np.empty(n_savings + 1)
        temp_s = np.empty(n_savings + 1)
        temp_b = np.empty(n_savings + 1)

        for z_i in range(n_z):
            psi = psi_vec[z_i]
            c_next_slice = c_next_full[z_i, :, :]
            pension_next = pension_1d[z_i]

            temp_x[0] = sc.egm_anchor; temp_c[0] = sc.egm_anchor
            temp_s[0] = 0.0; temp_b[0] = 0.0

            for s_i_idx in range(n_savings):
                s_val = savings_grid[s_i_idx]

                if constrained:
                    opt_s, opt_b, euler, exit_code, foc_resid = solve_portfolio_2d_retirement_quad(
                        s_val, z_i, i_s,
                        wealth_grid, c_next_slice, pension_next,
                        annuity_factor_is,
                        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                        Phi_0_state, Phi_11, s_i,
                        state_bracket_shift, state_bracket_L_inv,
                        grids_0, grids_1, grids_2, N1, N2,
                        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                        gamma, psi, beta, b_bar,
                        init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter,
                        tiny_savings=sc.tiny_savings, corner_tol=sc.corner_tol,
                        edge_max_iter=sc.edge_max_iter, edge_accept_factor=sc.edge_accept_factor,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_constrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold)
                    n_newton_iter = 0  # not tracked for constrained
                else:
                    opt_s, opt_b, euler, exit_code, foc_resid, n_newton_iter = solve_portfolio_unconstrained_retirement_quad(
                        s_val, z_i, i_s,
                        wealth_grid, c_next_slice, pension_next,
                        annuity_factor_is,
                        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                        Phi_0_state, Phi_11, s_i,
                        state_bracket_shift, state_bracket_L_inv,
                        grids_0, grids_1, grids_2, N1, N2,
                        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
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
                        line_search_max_step=sc.line_search_max_step,
                        alpha_min=sc.alpha_min, alpha_max=sc.alpha_max)

                # Diagnostic tracking (same as original)
                diag_int[i_s, 10] += 1
                if exit_code == 0:    diag_int[i_s, 9] += 1
                elif exit_code == 1:  diag_int[i_s, 0] += 1
                elif exit_code == 2:  diag_int[i_s, 1] += 1
                elif exit_code == 3:  diag_int[i_s, 2] += 1
                elif exit_code == 4:  diag_int[i_s, 3] += 1
                elif exit_code == 5:  diag_int[i_s, 4] += 1
                elif exit_code == 6:  diag_int[i_s, 5] += 1
                elif exit_code == 7:  diag_int[i_s, 6] += 1
                elif exit_code == 8:  diag_int[i_s, 7] += 1

                # Newton iter accounting (unconstrained only; constrained → 0)
                diag_int[i_s, 13] += n_newton_iter
                if n_newton_iter > diag_float[i_s, 9]:
                    diag_float[i_s, 9] = float(n_newton_iter)

                if foc_resid > diag_float[i_s, 1]:
                    diag_float[i_s, 1] = foc_resid
                diag_float[i_s, 2] += foc_resid * foc_resid

                diag_float[i_s, 7] += opt_s
                diag_float[i_s, 8] += opt_b
                if opt_s < diag_float[i_s, 3]: diag_float[i_s, 3] = opt_s
                if opt_s > diag_float[i_s, 4]: diag_float[i_s, 4] = opt_s
                if opt_b < diag_float[i_s, 5]: diag_float[i_s, 5] = opt_b
                if opt_b > diag_float[i_s, 6]: diag_float[i_s, 6] = opt_b

                if beta * euler <= 0.0:
                    diag_int[i_s, 11] += 1
                c_opt = max(beta * euler, sc.euler_inv_floor) ** (-1.0 / gamma)

                temp_x[s_i_idx + 1] = c_opt + s_val
                temp_c[s_i_idx + 1] = c_opt
                temp_s[s_i_idx + 1] = opt_s
                temp_b[s_i_idx + 1] = opt_b

                # Warm-start update: on unconstrained Newton failure, the returned
                # (opt_s, opt_b) is the last interior iterate before max_iter or
                # stagnation — can be wildly off (e.g. (1.5, -2.0)) and poisons
                # subsequent solves in the s_val/z_i chain. Reset to cold init.
                # Constrained failures keep the simplex-projected iterate which
                # is bounded to [0,1]^2 and remains usable.
                if (not constrained) and exit_code == EC_NEWTON_FAIL:
                    last_a_s = sc.init_alpha_s
                    last_a_b = sc.init_alpha_b
                    diag_int[i_s, 14] += 1   # DI_WARM_RESET
                else:
                    last_a_s = opt_s
                    last_a_b = opt_b

            for s_i_idx in range(n_savings):
                if temp_x[s_i_idx + 1] <= temp_x[s_i_idx]:
                    diag_int[i_s, 12] += 1
                    drop = temp_x[s_i_idx] - temp_x[s_i_idx + 1]
                    if drop > diag_float[i_s, 0]:
                        diag_float[i_s, 0] = drop

            for w_i in range(n_wealth):
                w = wealth_grid[w_i]
                policy_c[z_i, i_s, w_i] = pchip_interp_1d(w, temp_x, temp_c)
                policy_alpha_s[z_i, i_s, w_i] = pchip_interp_1d(w, temp_x, temp_s)
                policy_alpha_b[z_i, i_s, w_i] = pchip_interp_1d(w, temp_x, temp_b)

    return diag_int, diag_float


def solve_retirement_step_quad(wealth_grid, savings_grid, z_grid, N_state,
                               c_next_full, pension_1d,
                               annuity_factors,
                               state_grid, grids_0, grids_1, grids_2,
                               state_bracket_shift, state_bracket_L_inv,
                               v_nodes, v_weights, M_v_nodes, const_r, A_r,
                               Phi_0_state, Phi_11,
                               exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                               gamma, psi_vec, beta, b_bar,
                               constrained=True, solver_config=None,
                               out_c=None, out_s=None, out_b=None):
    if solver_config is None:
        solver_config = SolverConfig()
    n_z = len(z_grid)
    n_w = len(wealth_grid)
    if out_c is None:
        out_c = np.empty((n_z, N_state, n_w))
    if out_s is None:
        out_s = np.empty((n_z, N_state, n_w))
    if out_b is None:
        out_b = np.empty((n_z, N_state, n_w))
    _di, _df = _solve_retirement_step_quad_jit(
        wealth_grid, savings_grid, z_grid, N_state,
        c_next_full, pension_1d,
        annuity_factors,
        state_grid, grids_0, grids_1, grids_2,
        state_bracket_shift, state_bracket_L_inv,
        v_nodes, v_weights, M_v_nodes, const_r, A_r,
        Phi_0_state, Phi_11,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi_vec, beta, b_bar,
        constrained, solver_config,
        out_c, out_s, out_b)
    return out_c, out_s, out_b, _di, _df


# =============================================================================
# PERIOD SOLVER -- WORKING AGE (QUADRATURE)
# =============================================================================

@njit(parallel=True)
def _solve_working_age_step_quad_jit(wealth_grid, savings_grid, z_grid, N_state,
                                      c_next_full, log_det_next,
                                      annuity_factors, rho, eta_nodes, eta_weights, dz,
                                      state_grid, grids_0, grids_1, grids_2,
                                      state_bracket_shift, state_bracket_L_inv,
                                      v_nodes, v_weights, M_v_nodes, const_r, A_r,
                                      Phi_0_state, Phi_11,
                                      exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                      eps_nodes, eps_weights,
                                      gamma, psi_vec, beta, b_bar,
                                      use_pension_next, pension_next_by_z,
                                      constrained, solver_config,
                                      policy_c, policy_alpha_s, policy_alpha_b):
    """Solve one working-age period using quadrature over state innovations."""

    sc = solver_config
    n_z = len(z_grid)
    n_savings = len(savings_grid)
    n_wealth = len(wealth_grid)
    N1 = len(grids_1)
    N2 = len(grids_2)

    # Slot conventions: see DI_*/DF_* constants at top of solver.py.
    # Sizes are 14 (DI) and 10 (DF); slot 13 (DI_SUM_ITER) and slot 9 (DF_MAX_NEWTON_ITER)
    # are populated only on the unconstrained branch.
    diag_int = np.zeros((N_state, 15), dtype=np.int64)
    diag_float = np.zeros((N_state, 10))

    for i_s in prange(N_state):
        annuity_factor_is = annuity_factors[i_s]

        # Precompute base_mu_r_i for this state (3 returns: rtb, xr, xb)
        s_i = state_grid[i_s]
        base_mu_r_i = np.empty(3)
        base_mu_r_i[0] = const_r[0] + A_r[0, 0] * s_i[0] + A_r[0, 1] * s_i[1] + A_r[0, 2] * s_i[2]
        base_mu_r_i[1] = const_r[1] + A_r[1, 0] * s_i[0] + A_r[1, 1] * s_i[1] + A_r[1, 2] * s_i[2]
        base_mu_r_i[2] = const_r[2] + A_r[2, 0] * s_i[0] + A_r[2, 1] * s_i[1] + A_r[2, 2] * s_i[2]

        last_a_s = sc.init_alpha_s
        last_a_b = sc.init_alpha_b

        diag_float[i_s, 3] = 2.0
        diag_float[i_s, 5] = 2.0

        temp_x = np.empty(n_savings + 1)
        temp_c = np.empty(n_savings + 1)
        temp_s = np.empty(n_savings + 1)
        temp_b = np.empty(n_savings + 1)

        for z_i in range(n_z):
            psi = psi_vec[z_i]

            temp_x[0] = sc.egm_anchor; temp_c[0] = sc.egm_anchor
            temp_s[0] = 0.0; temp_b[0] = 0.0

            for s_i_idx in range(n_savings):
                s_val = savings_grid[s_i_idx]

                if constrained:
                    opt_s, opt_b, euler, exit_code, foc_resid = solve_portfolio_2d_working_quad(
                        s_val, z_i, i_s,
                        wealth_grid, c_next_full, log_det_next,
                        annuity_factor_is,
                        z_grid, rho, eta_nodes, eta_weights, dz,
                        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                        Phi_0_state, Phi_11, s_i,
                        state_bracket_shift, state_bracket_L_inv,
                        grids_0, grids_1, grids_2, N1, N2,
                        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                        eps_nodes, eps_weights,
                        gamma, psi, beta, b_bar,
                        use_pension_next, pension_next_by_z,
                        init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter,
                        tiny_savings=sc.tiny_savings, corner_tol=sc.corner_tol,
                        edge_max_iter=sc.edge_max_iter, edge_accept_factor=sc.edge_accept_factor,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_constrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold)
                    n_newton_iter = 0  # not tracked for constrained
                else:
                    opt_s, opt_b, euler, exit_code, foc_resid, n_newton_iter = solve_portfolio_unconstrained_working_quad(
                        s_val, z_i, i_s,
                        wealth_grid, c_next_full, log_det_next,
                        annuity_factor_is,
                        z_grid, rho, eta_nodes, eta_weights, dz,
                        v_nodes, v_weights, M_v_nodes, base_mu_r_i,
                        Phi_0_state, Phi_11, s_i,
                        state_bracket_shift, state_bracket_L_inv,
                        grids_0, grids_1, grids_2, N1, N2,
                        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                        eps_nodes, eps_weights,
                        gamma, psi, beta, b_bar,
                        use_pension_next, pension_next_by_z,
                        init_s=last_a_s, init_b=last_a_b,
                        tol=sc.tol, max_iter=sc.max_iter_unconstrained,
                        tiny_savings=sc.tiny_savings,
                        singular_det=sc.singular_det, grad_step_size=sc.grad_step_size,
                        step_damp=sc.step_damp_unconstrained, grad_denom_eps=sc.grad_denom_eps,
                        min_wealth_inv=sc.min_wealth_inv, min_consumption=sc.min_consumption,
                        prob_skip=sc.prob_skip_threshold,
                        use_line_search=sc.use_line_search,
                        max_backtrack_iter=sc.max_backtrack_iter,
                        line_search_max_step=sc.line_search_max_step,
                        alpha_min=sc.alpha_min, alpha_max=sc.alpha_max)

                diag_int[i_s, 10] += 1
                if exit_code == 0:    diag_int[i_s, 9] += 1
                elif exit_code == 1:  diag_int[i_s, 0] += 1
                elif exit_code == 2:  diag_int[i_s, 1] += 1
                elif exit_code == 3:  diag_int[i_s, 2] += 1
                elif exit_code == 4:  diag_int[i_s, 3] += 1
                elif exit_code == 5:  diag_int[i_s, 4] += 1
                elif exit_code == 6:  diag_int[i_s, 5] += 1
                elif exit_code == 7:  diag_int[i_s, 6] += 1
                elif exit_code == 8:  diag_int[i_s, 7] += 1

                # Newton iter accounting (unconstrained only; constrained → 0)
                diag_int[i_s, 13] += n_newton_iter
                if n_newton_iter > diag_float[i_s, 9]:
                    diag_float[i_s, 9] = float(n_newton_iter)

                if foc_resid > diag_float[i_s, 1]:
                    diag_float[i_s, 1] = foc_resid
                diag_float[i_s, 2] += foc_resid * foc_resid

                diag_float[i_s, 7] += opt_s
                diag_float[i_s, 8] += opt_b
                if opt_s < diag_float[i_s, 3]: diag_float[i_s, 3] = opt_s
                if opt_s > diag_float[i_s, 4]: diag_float[i_s, 4] = opt_s
                if opt_b < diag_float[i_s, 5]: diag_float[i_s, 5] = opt_b
                if opt_b > diag_float[i_s, 6]: diag_float[i_s, 6] = opt_b

                if beta * euler <= 0.0:
                    diag_int[i_s, 11] += 1

                c_opt = max(beta * euler, sc.euler_inv_floor) ** (-1.0 / gamma)

                temp_x[s_i_idx + 1] = c_opt + s_val
                temp_c[s_i_idx + 1] = c_opt
                temp_s[s_i_idx + 1] = opt_s
                temp_b[s_i_idx + 1] = opt_b

                # Warm-start update: on unconstrained Newton failure, the returned
                # (opt_s, opt_b) is the last interior iterate before max_iter or
                # stagnation — can be wildly off (e.g. (1.5, -2.0)) and poisons
                # subsequent solves in the s_val/z_i chain. Reset to cold init.
                # Constrained failures keep the simplex-projected iterate which
                # is bounded to [0,1]^2 and remains usable.
                if (not constrained) and exit_code == EC_NEWTON_FAIL:
                    last_a_s = sc.init_alpha_s
                    last_a_b = sc.init_alpha_b
                    diag_int[i_s, 14] += 1   # DI_WARM_RESET
                else:
                    last_a_s = opt_s
                    last_a_b = opt_b

            for s_i_idx in range(n_savings):
                if temp_x[s_i_idx + 1] <= temp_x[s_i_idx]:
                    diag_int[i_s, 12] += 1
                    drop = temp_x[s_i_idx] - temp_x[s_i_idx + 1]
                    if drop > diag_float[i_s, 0]:
                        diag_float[i_s, 0] = drop

            for w_i in range(n_wealth):
                w = wealth_grid[w_i]
                policy_c[z_i, i_s, w_i] = pchip_interp_1d(w, temp_x, temp_c)
                policy_alpha_s[z_i, i_s, w_i] = pchip_interp_1d(w, temp_x, temp_s)
                policy_alpha_b[z_i, i_s, w_i] = pchip_interp_1d(w, temp_x, temp_b)

    return diag_int, diag_float


def solve_working_age_step_quad(wealth_grid, savings_grid, z_grid, N_state,
                                c_next_full, log_det_next,
                                annuity_factors, rho, eta_nodes, eta_weights, dz,
                                state_grid, grids_0, grids_1, grids_2,
                                state_bracket_shift, state_bracket_L_inv,
                                v_nodes, v_weights, M_v_nodes, const_r, A_r,
                                Phi_0_state, Phi_11,
                                exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                                eps_nodes, eps_weights,
                                gamma, psi_vec, beta, b_bar,
                                use_pension_next, pension_next_by_z,
                                constrained=True, solver_config=None,
                                out_c=None, out_s=None, out_b=None):
    if solver_config is None:
        solver_config = SolverConfig()
    n_z = len(z_grid)
    n_w = len(wealth_grid)
    if out_c is None:
        out_c = np.empty((n_z, N_state, n_w))
    if out_s is None:
        out_s = np.empty((n_z, N_state, n_w))
    if out_b is None:
        out_b = np.empty((n_z, N_state, n_w))
    _di, _df = _solve_working_age_step_quad_jit(
        wealth_grid, savings_grid, z_grid, N_state,
        c_next_full, log_det_next,
        annuity_factors, rho, eta_nodes, eta_weights, dz,
        state_grid, grids_0, grids_1, grids_2,
        state_bracket_shift, state_bracket_L_inv,
        v_nodes, v_weights, M_v_nodes, const_r, A_r,
        Phi_0_state, Phi_11,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        eps_nodes, eps_weights,
        gamma, psi_vec, beta, b_bar,
        use_pension_next, pension_next_by_z,
        constrained, solver_config,
        out_c, out_s, out_b)
    return out_c, out_s, out_b, _di, _df


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


def _default_checkpoint_path(model, disc_config, youngest_age_to_solve):
    """Build a natural default bundle path for partial-solve checkpoints."""
    label = "constrained" if model.constrained else "unconstrained"
    grid_sizes = "x".join(str(v) for v in disc_config.state_grid_sizes)
    age_suffix = (
        f"_to_age{int(youngest_age_to_solve)}"
        if youngest_age_to_solve is not None
        else "_partial"
    )
    name = (
        f"{label}_{disc_config.state_grid_mode}"
        f"_grid{grid_sizes}_nz{disc_config.n_z}{age_suffix}"
    )
    return str(Path("saved_runs") / "checkpoints" / name)


def _normalize_solve_control(model, pc, solve_control):
    """Validate and normalize SolveControl inputs."""
    if solve_control is None:
        return SolveControl(), False
    if not isinstance(solve_control, SolveControl):
        # In notebook workflows with autoreload, NamedTuple classes can be
        # redefined, so a SolveControl instance created from the reloaded
        # `model` module may fail `isinstance(..., SolveControl)` here even
        # though it has the correct fields. Normalize any solve-control-shaped
        # object into the local SolveControl class before validating values.
        try:
            defaults = SolveControl()._asdict()
            solve_control = SolveControl(
                **{
                    field: getattr(solve_control, field, default)
                    for field, default in defaults.items()
                }
            )
        except Exception as exc:
            raise TypeError(
                "solve_control must be a SolveControl instance or None"
            ) from exc

    youngest = solve_control.youngest_age_to_solve
    if youngest is not None:
        youngest = int(youngest)
        if youngest < model.start_age or youngest > model.terminal_age:
            raise ValueError(
                "youngest_age_to_solve must lie within "
                f"[{model.start_age}, {model.terminal_age}], got {youngest}"
            )

    every = solve_control.checkpoint_every_n_ages
    if every is not None:
        every = int(every)
        if every <= 0:
            raise ValueError("checkpoint_every_n_ages must be positive")

    checkpoint_path = solve_control.checkpoint_path
    if checkpoint_path is None and (
        youngest is not None or every is not None or solve_control.save_on_interrupt
    ):
        checkpoint_path = _default_checkpoint_path(model, pc.disc_config, youngest)

    if checkpoint_path is not None:
        checkpoint_path = str(Path(checkpoint_path))

    progress_wealth_source = solve_control.progress_wealth_source
    if progress_wealth_source is None:
        progress_wealth_source = SolveControl().progress_wealth_source
    progress_wealth_source = str(progress_wealth_source).strip().lower()
    if progress_wealth_source not in _PROGRESS_WEALTH_SOURCES:
        raise ValueError(
            "progress_wealth_source must be one of "
            f"{sorted(_PROGRESS_WEALTH_SOURCES)}, got {progress_wealth_source!r}"
        )

    return solve_control._replace(
        youngest_age_to_solve=youngest,
        checkpoint_path=checkpoint_path,
        checkpoint_every_n_ages=every,
        progress_wealth_source=progress_wealth_source,
    ), True


def _build_solver_diagnostics(
    *,
    age_diag_int,
    age_diag_fsum,
    age_diag_fmax,
    age_diag_fmin,
    solved_age_mask,
    ages,
    retire_age,
    solver_config,
    disc_config,
    constrained,
    solve_control,
    solve_status,
    wall_time_sec,
    checkpoint_save_count,
    checkpoint_path,
):
    """Aggregate per-age diagnostics, respecting partially solved outputs."""
    solved_idx = np.flatnonzero(solved_age_mask)
    solved_nonterminal_mask = solved_age_mask.copy()
    if len(solved_nonterminal_mask) > 0:
        solved_nonterminal_mask[-1] = False
    solved_nonterminal_idx = np.flatnonzero(solved_nonterminal_mask)

    if solved_nonterminal_idx.size > 0:
        all_int = age_diag_int[solved_nonterminal_idx].sum(axis=0)
        all_fsum = age_diag_fsum[solved_nonterminal_idx].sum(axis=0)
        all_fmax = age_diag_fmax[solved_nonterminal_idx].max(axis=0)
        all_fmin = age_diag_fmin[solved_nonterminal_idx].min(axis=0)
    else:
        all_int = np.zeros(N_DIAG_INT, dtype=np.int64)
        all_fsum = np.zeros(N_DIAG_FLOAT)
        all_fmax = np.zeros(N_DIAG_FLOAT)
        all_fmin = np.zeros(N_DIAG_FLOAT)

    total_calls = int(all_int[DI_TOTAL_CALLS])
    total_fail = int(all_int[DI_NEWTON_FAIL])
    total_mono = int(all_int[DI_MONO_VIOLATIONS])
    worst_mono = float(all_fmax[DF_WORST_MONO_DROP])
    worst_foc = float(all_fmax[DF_MAX_FOC_RESID])
    rms_foc = (all_fsum[DF_SUM_FOC_RESID_SQ] / max(total_calls, 1)) ** 0.5

    total_iter = int(all_int[DI_SUM_ITER])
    max_iter_used = int(all_fmax[DF_MAX_NEWTON_ITER])
    avg_iter = total_iter / max(total_calls, 1)
    total_warm_reset = int(all_int[DI_WARM_RESET])

    solved_ages = ages[solved_idx] if solved_idx.size > 0 else np.array([], dtype=np.int64)
    youngest_solved_age = int(solved_ages.min()) if solved_ages.size > 0 else None
    oldest_solved_age = int(solved_ages.max()) if solved_ages.size > 0 else None
    is_partial = solve_status != "complete" or solved_idx.size != len(ages)

    return {
        "age_diag_int": age_diag_int,
        "age_diag_fsum": age_diag_fsum,
        "age_diag_fmax": age_diag_fmax,
        "age_diag_fmin": age_diag_fmin,
        "aggregate_int": all_int,
        "aggregate_fsum": all_fsum,
        "aggregate_fmax": all_fmax,
        "aggregate_fmin": all_fmin,
        "total_mono_violations": total_mono,
        "worst_mono_drop": worst_mono,
        "total_newton_failures": total_fail,
        "worst_foc_resid": worst_foc,
        "total_calls": total_calls,
        "total_newton_iter": total_iter,
        "avg_newton_iter": avg_iter,
        "max_newton_iter": max_iter_used,
        "total_warm_reset": total_warm_reset,
        "constrained": constrained,
        "solver_config": solver_config,
        "disc_config": disc_config,
        "solve_control": solve_control,
        "solve_status": solve_status,
        "is_partial": is_partial,
        "solved_age_mask": solved_age_mask.copy(),
        "solved_age_indices": solved_idx.copy(),
        "youngest_solved_age": youngest_solved_age,
        "oldest_solved_age": oldest_solved_age,
        "n_ages_solved": int(solved_idx.size),
        "n_nonterminal_ages_solved": int(solved_nonterminal_idx.size),
        "wall_time_sec": float(wall_time_sec),
        "checkpoint_save_count": int(checkpoint_save_count),
        "checkpoint_path": checkpoint_path,
    }


def _prepare_policy_snapshot(C_mat, S_mat, B_mat, solved_age_mask):
    """Return arrays suitable for saving, masking unsolved ages with NaN."""
    if np.all(solved_age_mask):
        return C_mat, S_mat, B_mat

    unsolved_mask = ~solved_age_mask
    C_save = C_mat.copy()
    S_save = S_mat.copy()
    B_save = B_mat.copy()
    C_save[unsolved_mask] = np.nan
    S_save[unsolved_mask] = np.nan
    B_save[unsolved_mask] = np.nan
    return C_save, S_save, B_save


def _mask_unsolved_ages_in_place(C_mat, S_mat, B_mat, solved_age_mask):
    """Mask unsolved age slices before returning partial results."""
    if np.all(solved_age_mask):
        return
    unsolved_mask = ~solved_age_mask
    C_mat[unsolved_mask] = np.nan
    S_mat[unsolved_mask] = np.nan
    B_mat[unsolved_mask] = np.nan


def _save_policy_checkpoint(checkpoint_path, C_mat, S_mat, B_mat, diagnostics):
    """Persist a solver checkpoint bundle."""
    from lifecycle.policy_io import save_policy_bundle

    C_save, S_save, B_save = _prepare_policy_snapshot(
        C_mat, S_mat, B_mat, diagnostics["solved_age_mask"]
    )
    return save_policy_bundle(
        checkpoint_path,
        C_save,
        S_save,
        B_save,
        diagnostics=diagnostics,
        overwrite=True,
    )


def run_lifecycle_solver(model, pc, solver_config=None, n_s_points=None, verbose=1, solve_control=None):
    """
    Lifecycle backward induction solver.

    Parameters
    ----------
    model : LifecyclePortfolioModel
    pc    : Precompute
    n_s_points : int, optional  -- override savings grid size
    verbose : int  -- 0=silent, 1=per-age table + post-solve report (default)
    solve_control : SolveControl | None
        Optional non-numerical controls for partial solves and checkpoints.

    Returns
    -------
    C_mat, S_mat, B_mat : np.ndarray, shape (n_age, n_z, N_state, n_w)
        Optimal consumption, stock share, and bond share.
    diagnostics : dict
        Diagnostic summary from the solve.
    """
    if solver_config is None:
        solver_config = SolverConfig()
    solve_control, control_active = _normalize_solve_control(model, pc, solve_control)

    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"LIFECYCLE PORTFOLIO SOLVER  (EGM + 2D Newton)")
        mode_str = "CONSTRAINED" if model.constrained else "UNCONSTRAINED"
        print(f"  Mode: {mode_str} | STATE_QUADRATURE")
        print(f"  Solver: {solver_config}")
        print(f"  Discretization: {pc.disc_config}")
        if control_active:
            print(f"  Solve control: {solve_control}")
        print(f"{'='*70}")

    # ---- Grids ----
    w_grid = pc.wealth_grid
    s_grid = pc.s_grid if n_s_points is None else pc.regenerate_savings_grid(n_s_points)
    z_grid = pc.z_grid
    ages   = pc.ages

    n_w     = len(w_grid)
    n_z     = pc.n_z
    N_state = pc.N_state
    n_age   = pc.n_age

    # ---- Transitions and returns ----
    rho             = model.rho
    eta_nodes       = pc.eta_nodes
    eta_weights     = pc.eta_weights
    dz              = pc.dz
    ret_nodes       = pc.ret_nodes
    ret_weights     = pc.ret_weights
    annuity_factors = pc.annuity_factors

    # ---- State quadrature arrays ----
    v_nodes         = pc.v_nodes
    v_weights       = pc.v_weights
    M_v_nodes       = pc.M_v_nodes
    const_r         = pc.const_r
    A_r             = pc.A_r
    (state_grid,
     grids_0,
     grids_1,
     grids_2,
     state_bracket_shift,
     state_bracket_L_inv,
     v_nodes_solver,
     Phi_0_state_solver,
     Phi_11_solver,
     A_r_solver) = _pad_state_solver_inputs_to_3d(
        pc.state_grid,
        pc.state_bracket_grids,
        pc.state_bracket_shift,
        pc.state_bracket_L_inv,
        pc.v_nodes,
        model.Phi_0_state,
        model.Phi_11,
        pc.A_r,
    )
    exp_ret_bill    = pc.exp_ret_bill
    exp_ret_stock   = pc.exp_ret_stock
    exp_ret_bond    = pc.exp_ret_bond
    Phi_0_state     = Phi_0_state_solver
    Phi_11          = Phi_11_solver
    v_nodes         = v_nodes_solver
    A_r             = A_r_solver

    # ---- Income tables ----
    pension_table        = pc.pension_after_tax      # (n_age, n_z)
    # working_income_table: retained in pc for simulation/diagnostics, not used here
    log_det_profile      = pc.log_det_profile        # (n_age,)
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

    if verbose >= 1:
        n_v_q = len(v_weights)
        n_r_q = len(ret_weights)
        n_eta_q = len(eta_nodes)
        n_eps_q = len(eps_nodes)
        print(f"  Ages {start_age}\u2013{terminal_age}  ({n_age} periods)")
        print(f"  Grids: n_w={n_w}, n_s={len(s_grid)}, n_z={n_z}, N_state={N_state}")
        print(f"  gamma={gamma}, beta={beta}, b_bar={b_bar}")
        print(f"  States per period: {n_z} \u00d7 {N_state} = {n_z * N_state:,}")
        print(f"  Quadrature nodes: state-innov={n_v_q}, return-resid={n_r_q}, "
              f"labor eta={n_eta_q}, labor eps={n_eps_q}  "
              f"(FOC evals/call: retire={n_v_q*n_r_q:,}, work={n_v_q*n_r_q*(1+n_eta_q*n_eps_q):,})")

    # ---- Representative state for per-age summary ----
    i_z_med = n_z // 2
    i_s_med = N_state // 2
    progress_wealth_source = solve_control.progress_wealth_source
    progress_wealth_by_age = None
    progress_wealth_label = None

    if verbose >= 1:
        try:
            progress_wealth_by_age, progress_wealth_label = _build_progress_wealth_schedule(
                ages=ages,
                w_grid=w_grid,
                source=progress_wealth_source,
            )
        except Exception as exc:
            progress_wealth_source = "grid_midpoint"
            progress_wealth_by_age, progress_wealth_label = _build_progress_wealth_schedule(
                ages=ages,
                w_grid=w_grid,
                source=progress_wealth_source,
            )
            print(
                "  WARNING: could not build SCF wealth probe "
                f"({exc}); falling back to {progress_wealth_label}."
            )

    # ---- Policy arrays ----
    shape = (n_age, n_z, N_state, n_w)
    C_mat = np.zeros(shape)
    S_mat = np.zeros(shape)
    B_mat = np.zeros(shape)
    solved_age_mask = np.zeros(n_age, dtype=bool)

    checkpoint_path = solve_control.checkpoint_path
    youngest_age_to_solve = solve_control.youngest_age_to_solve
    checkpoint_every_n_ages = solve_control.checkpoint_every_n_ages
    save_on_interrupt = solve_control.save_on_interrupt
    return_partial_on_interrupt = solve_control.return_partial_on_interrupt
    checkpoint_save_count = 0
    last_saved_nonterminal_count = -1

    # ---- Per-age diagnostic accumulators ----
    age_diag_int     = np.zeros((n_age, N_DIAG_INT), dtype=np.int64)
    age_diag_fsum    = np.zeros((n_age, N_DIAG_FLOAT))
    age_diag_fmax    = np.zeros((n_age, N_DIAG_FLOAT))
    age_diag_fmin    = np.full((n_age, N_DIAG_FLOAT), np.inf)

    # ---- Optional resume from checkpoint ----
    # If `checkpoint_path` exists on disk and contains a valid bundle whose
    # arrays match the current shape, pre-fill C/S/B and solved_age_mask from
    # it so the loop below skips already-solved ages. Mismatched checkpoints
    # are refused loudly to avoid silently mixing incompatible policies.
    if checkpoint_path is not None:
        ckpt_dir = Path(checkpoint_path)
        if (ckpt_dir / "policy_arrays.npz").exists():
            from lifecycle.policy_io import load_policy_bundle
            try:
                Cc, Sc, Bc, ckpt_diag, _ = load_policy_bundle(ckpt_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Found checkpoint at {ckpt_dir} but failed to load it: {exc}. "
                    "Delete the checkpoint or fix the bundle before retrying."
                ) from exc
            if Cc.shape != C_mat.shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch at {ckpt_dir}: "
                    f"got {Cc.shape}, expected {C_mat.shape}. "
                    "Different grid/quadrature/system — refuse to resume."
                )
            ckpt_mask = None
            if ckpt_diag is not None:
                ckpt_mask = ckpt_diag.get("solved_age_mask")
                ckpt_age_int = ckpt_diag.get("age_diag_int")
                ckpt_age_fsum = ckpt_diag.get("age_diag_fsum")
                ckpt_age_fmax = ckpt_diag.get("age_diag_fmax")
                ckpt_age_fmin = ckpt_diag.get("age_diag_fmin")
            if ckpt_mask is None or len(ckpt_mask) != n_age:
                raise RuntimeError(
                    f"Checkpoint at {ckpt_dir} missing or malformed "
                    "solved_age_mask in diagnostics — refuse to resume."
                )
            ckpt_mask = np.asarray(ckpt_mask, dtype=bool)
            for t in range(n_age):
                if ckpt_mask[t]:
                    C_mat[t] = Cc[t]
                    S_mat[t] = Sc[t]
                    B_mat[t] = Bc[t]
                    solved_age_mask[t] = True
            if ckpt_age_int is not None and np.shape(ckpt_age_int) == age_diag_int.shape:
                age_diag_int[:] = ckpt_age_int
            if ckpt_age_fsum is not None and np.shape(ckpt_age_fsum) == age_diag_fsum.shape:
                age_diag_fsum[:] = ckpt_age_fsum
            if ckpt_age_fmax is not None and np.shape(ckpt_age_fmax) == age_diag_fmax.shape:
                age_diag_fmax[:] = ckpt_age_fmax
            if ckpt_age_fmin is not None and np.shape(ckpt_age_fmin) == age_diag_fmin.shape:
                age_diag_fmin[:] = ckpt_age_fmin
            n_resumed = int(np.sum(solved_age_mask))
            if verbose >= 1 and n_resumed > 0:
                resumed_ages = ages[np.flatnonzero(solved_age_mask)]
                print(
                    f"\n  Resumed from checkpoint {ckpt_dir}: "
                    f"{n_resumed}/{n_age} ages already solved "
                    f"(ages {int(resumed_ages.min())}-{int(resumed_ages.max())})"
                )

    # ---- Terminal condition ----
    if not solved_age_mask[-1]:
        if verbose >= 1:
            print(f"\n  Terminal condition (age {terminal_age}) ... ", end="", flush=True)
        c_T, a_s_T, a_b_T, term_diag = solve_terminal_age(
            w_grid, annuity_factors,
            state_grid, const_r, A_r, M_v_nodes, v_weights,
            ret_nodes, ret_weights,
            gamma, beta, b_bar, N_state, n_z, constrained=constrained, solver_config=solver_config)
        C_mat[-1] = c_T
        S_mat[-1] = a_s_T
        B_mat[-1] = a_b_T
        solved_age_mask[-1] = True
    else:
        # Skip terminal solve; it was loaded from the checkpoint. Use the
        # loaded slice for downstream summary so subsequent prints are sane.
        c_T = C_mat[-1]
        a_s_T = S_mat[-1]
        a_b_T = B_mat[-1]
        term_diag = np.full((n_z, N_state), EC_INTERIOR, dtype=np.int8)
        if verbose >= 1:
            print(f"\n  Terminal condition (age {terminal_age}) ... loaded from checkpoint", end="", flush=True)
    if verbose >= 1:
        n_term_interior = int(np.sum(term_diag == EC_INTERIOR))
        n_term_fail = int(np.sum(term_diag == EC_NEWTON_FAIL))
        print(f"done  [c range: {c_T.min():.3f}\u2013{c_T.max():.3f}]  "
              f"[portfolio: {n_term_interior} interior, {N_state - n_term_interior - n_term_fail} corner/edge"
              f"{f', {n_term_fail} FAIL' if n_term_fail > 0 else ''}]")

    # ---- Backward induction ----
    if verbose >= 1:
        print(f"\n{'='*120}")
        print(
            "  Live policy probe: "
            f"z=z_grid[{i_z_med}], state midpoint, wealth={progress_wealth_label}"
        )
        print("  W column reports the probe wealth in model units.")
        # avg_it / max_it columns are only meaningful for unconstrained
        # (constrained Newton iter counts are not tracked → always 0).
        iter_hdr = f"  {'avg_it':>6} {'max_it':>6}" if not constrained else ""
        hdr = (f" {'Age':>3}  {'Phase':<6} {'Time':>5}  {'Newt%':>5} {'Fail':>6}"
               f"  {'alpha_s':>7}  {'alpha_b':>7}  {'a_bill':>7}  {'W':>6}  {'c/W':>5}"
               f"  {'%int':>4}  {'%edge':>5}  {'%corn':>5}  {'mono':>4}"
               f"{iter_hdr}")
        print(hdr)
        print(f"{'='*120}")

    t_start = time.time()
    solve_status = "complete"
    last_saved_bundle_path = None

    # Dummy pension array for working ages where t+1 is also working (the
    # working FOC ignores it; numba still needs a real float64[n_z] array).
    pension_dummy_z = np.zeros(n_z, dtype=np.float64)

    try:
        for t in reversed(range(n_age - 1)):
            age = ages[t]
            if youngest_age_to_solve is not None and age < youngest_age_to_solve:
                solve_status = "stopped_early"
                break
            if solved_age_mask[t]:
                # Already loaded from checkpoint — skip the solve for this age.
                continue
            psi = survival_probs[t, :]      # (n_z,) -- z-dependent survival
            c_next = C_mat[t + 1]

            # Output buffers - JIT writes directly into C_mat/S_mat/B_mat slices
            out_c = C_mat[t]
            out_s = S_mat[t]
            out_b = B_mat[t]

            if age >= retire_age:
                _, _, _, _di, _df = solve_retirement_step_quad(
                    w_grid, s_grid, z_grid, N_state,
                    c_next, pension_table[t + 1, :],
                    annuity_factors,
                    state_grid, grids_0, grids_1, grids_2,
                    state_bracket_shift, state_bracket_L_inv,
                    v_nodes, v_weights, M_v_nodes, const_r, A_r,
                    Phi_0_state, Phi_11,
                    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                    gamma, psi, beta, b_bar, constrained=constrained, solver_config=solver_config,
                    out_c=out_c, out_s=out_s, out_b=out_b)
                label = "RETIRE"
            else:
                # Work-to-retirement boundary: at age = retire_age - 1, next-period
                # income is pension(z_next), not the labor-income polynomial.
                use_pension_next = (age == retire_age - 1)
                pension_next_by_z = pension_table[t + 1, :] if use_pension_next else pension_dummy_z
                _, _, _, _di, _df = solve_working_age_step_quad(
                    w_grid, s_grid, z_grid, N_state,
                    c_next, log_det_profile[t + 1],
                    annuity_factors, rho, eta_nodes, eta_weights, dz,
                    state_grid, grids_0, grids_1, grids_2,
                    state_bracket_shift, state_bracket_L_inv,
                    v_nodes, v_weights, M_v_nodes, const_r, A_r,
                    Phi_0_state, Phi_11,
                    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
                    eps_nodes, eps_weights,
                    gamma, psi, beta, b_bar,
                    use_pension_next, pension_next_by_z,
                    constrained=constrained, solver_config=solver_config,
                    out_c=out_c, out_s=out_s, out_b=out_b)
                label = "WORK  "

            # No copy needed - JIT wrote directly into C_mat[t], S_mat[t], B_mat[t]

            # Reduce diagnostics for this age
            ti, tf_sum, tf_max, tf_min = _reduce_diag(_di, _df)
            age_diag_int[t] = ti
            age_diag_fsum[t] = tf_sum
            age_diag_fmax[t] = tf_max
            age_diag_fmin[t] = tf_min
            solved_age_mask[t] = True

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

                # Representative-state policy values at the chosen age-specific
                # wealth probe. This stays outside the hot JIT kernels.
                probe_w = float(progress_wealth_by_age[t])
                probe_as = _interp_progress_policy_at_wealth(
                    out_s[i_z_med, i_s_med, :], w_grid, probe_w
                )
                probe_ab = _interp_progress_policy_at_wealth(
                    out_b[i_z_med, i_s_med, :], w_grid, probe_w
                )
                probe_bill = 1.0 - probe_as - probe_ab
                probe_c = _interp_progress_policy_at_wealth(
                    out_c[i_z_med, i_s_med, :], w_grid, probe_w
                )
                c_over_w = probe_c / probe_w if probe_w > 0 else 0.0

                mono_str = f"{mono_v:4d}" if mono_v == 0 else f"\033[91m{mono_v:4d}\033[0m"

                # Iter columns: only printed when unconstrained
                if not constrained:
                    sum_it = int(ti[DI_SUM_ITER])
                    max_it = int(tf_max[DF_MAX_NEWTON_ITER])
                    avg_it = sum_it / max(total_calls, 1)
                    iter_str = f"  {avg_it:6.1f} {max_it:6d}"
                else:
                    iter_str = ""

                print(f" {age:3d}  {label:<6} {elapsed:5.1f}s  {newton_pct:5.1f}% {n_fail:>6}"
                      f"  {probe_as:7.3f}  {probe_ab:7.3f}  {probe_bill:7.3f}"
                      f"  {probe_w:6.2f}  {c_over_w:5.3f}"
                      f"  {_format_pct(n_interior, total_calls)}"
                      f"  {_format_pct(n_edge, total_calls)}"
                      f"  {_format_pct(n_corner, total_calls)}"
                      f"  {mono_str}{iter_str}", flush=True)

            if checkpoint_every_n_ages is not None and checkpoint_path is not None:
                solved_nonterminal_count = int(np.sum(solved_age_mask[:-1]))
                if solved_nonterminal_count - last_saved_nonterminal_count >= checkpoint_every_n_ages:
                    checkpoint_save_count += 1
                    checkpoint_diag = _build_solver_diagnostics(
                        age_diag_int=age_diag_int,
                        age_diag_fsum=age_diag_fsum,
                        age_diag_fmax=age_diag_fmax,
                        age_diag_fmin=age_diag_fmin,
                        solved_age_mask=solved_age_mask,
                        ages=ages,
                        retire_age=retire_age,
                        solver_config=solver_config,
                        disc_config=pc.disc_config,
                        constrained=constrained,
                        solve_control=solve_control,
                        solve_status="checkpoint",
                        wall_time_sec=time.time() - t_start,
                        checkpoint_save_count=checkpoint_save_count,
                        checkpoint_path=checkpoint_path,
                    )
                    last_saved_bundle_path = str(_save_policy_checkpoint(
                        checkpoint_path, C_mat, S_mat, B_mat, checkpoint_diag
                    ))
                    last_saved_nonterminal_count = solved_nonterminal_count
                    if verbose >= 1:
                        print(f"    checkpoint saved -> {last_saved_bundle_path}", flush=True)
    except KeyboardInterrupt:
        solve_status = "interrupted"
        if verbose >= 1:
            print("\n  Solve interrupted. Finalizing partial output...", flush=True)

    total = time.time() - t_start

    solved_nonterminal_count = int(np.sum(solved_age_mask[:-1]))
    final_save_needed = (
        checkpoint_path is not None
        and (
            solve_status == "stopped_early"
            or (solve_status == "interrupted" and save_on_interrupt)
            or (
                solve_status == "complete"
                and
                checkpoint_every_n_ages is not None
                and solved_nonterminal_count != last_saved_nonterminal_count
            )
        )
    )

    if final_save_needed:
        next_save_count = checkpoint_save_count + 1
        diagnostics = _build_solver_diagnostics(
            age_diag_int=age_diag_int,
            age_diag_fsum=age_diag_fsum,
            age_diag_fmax=age_diag_fmax,
            age_diag_fmin=age_diag_fmin,
            solved_age_mask=solved_age_mask,
            ages=ages,
            retire_age=retire_age,
            solver_config=solver_config,
            disc_config=pc.disc_config,
            constrained=constrained,
            solve_control=solve_control,
            solve_status=solve_status,
            wall_time_sec=total,
            checkpoint_save_count=next_save_count,
            checkpoint_path=checkpoint_path,
        )
        last_saved_bundle_path = str(_save_policy_checkpoint(
            checkpoint_path, C_mat, S_mat, B_mat, diagnostics
        ))
        checkpoint_save_count = next_save_count
    else:
        diagnostics = _build_solver_diagnostics(
            age_diag_int=age_diag_int,
            age_diag_fsum=age_diag_fsum,
            age_diag_fmax=age_diag_fmax,
            age_diag_fmin=age_diag_fmin,
            solved_age_mask=solved_age_mask,
            ages=ages,
            retire_age=retire_age,
            solver_config=solver_config,
            disc_config=pc.disc_config,
            constrained=constrained,
            solve_control=solve_control,
            solve_status=solve_status,
            wall_time_sec=total,
            checkpoint_save_count=checkpoint_save_count,
            checkpoint_path=checkpoint_path,
        )

    diagnostics["checkpoint_save_count"] = checkpoint_save_count
    diagnostics["last_saved_bundle_path"] = last_saved_bundle_path

    all_int = diagnostics["aggregate_int"]
    all_fsum = diagnostics["aggregate_fsum"]
    all_fmax = diagnostics["aggregate_fmax"]
    all_fmin = diagnostics["aggregate_fmin"]
    total_calls = diagnostics["total_calls"]
    total_fail = diagnostics["total_newton_failures"]
    total_mono = diagnostics["total_mono_violations"]
    worst_mono = diagnostics["worst_mono_drop"]
    worst_foc = diagnostics["worst_foc_resid"]
    rms_foc = (all_fsum[DF_SUM_FOC_RESID_SQ] / max(total_calls, 1)) ** 0.5
    total_iter = diagnostics["total_newton_iter"]
    max_iter_used = diagnostics["max_newton_iter"]
    avg_iter = diagnostics["avg_newton_iter"]
    total_warm_reset = diagnostics["total_warm_reset"]
    n_nonterminal_ages_solved = diagnostics["n_nonterminal_ages_solved"]
    is_partial = diagnostics["is_partial"]

    if verbose >= 1:
        print(f"\n{'='*120}")
        print(f"  DONE in {total / 60:.2f} min  (avg {total / max(n_nonterminal_ages_solved, 1):.2f}s per age)")
        print(f"{'='*120}")

        # --- Section 1: Newton Convergence ---
        print(f"\n{'='*70}")
        print(f"  POST-SOLVE DIAGNOSTICS")
        print(f"{'='*70}")
        if is_partial:
            print(
                f"\n  Solve status: {diagnostics['solve_status']}  "
                f"(solved ages {diagnostics['youngest_solved_age']} to "
                f"{diagnostics['oldest_solved_age']})"
            )
            if last_saved_bundle_path is not None:
                print(f"  Saved partial bundle: {last_saved_bundle_path}")

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

        # --- Section 1b: Newton iteration counts (unconstrained only) ---
        if not constrained and total_calls > 0:
            print(f"\n  1b. NEWTON ITERATIONS  (unconstrained Newton only)")
            print(f"     Total iters:  {total_iter:>12,}  (across {total_calls:,} calls)")
            print(f"     Avg iters:    {avg_iter:>12.2f}  per call")
            print(f"     Max iters:    {max_iter_used:>12,}  in any single call"
                  f"  (max_iter_unconstrained={solver_config.max_iter_unconstrained})")
            print(f"     Warm resets:  {total_warm_reset:>12,}  "
                  f"(times init_alpha_* fell back after EC_NEWTON_FAIL)")
            n_at_cap = int(np.sum(age_diag_fmax[:-1, DF_MAX_NEWTON_ITER]
                                  >= solver_config.max_iter_unconstrained))
            if max_iter_used >= solver_config.max_iter_unconstrained:
                print(f"     WARNING: at least one call hit max_iter_unconstrained "
                      f"({n_at_cap}/{n_age-1} ages saw the cap reached)")
            # Per-age top offenders by avg iter
            age_avg = []
            for t in range(n_age - 1):
                tc = int(age_diag_int[t, DI_TOTAL_CALLS])
                if tc == 0:
                    continue
                ai = age_diag_int[t, DI_SUM_ITER] / tc
                mi = int(age_diag_fmax[t, DF_MAX_NEWTON_ITER])
                age_avg.append((ages[t], ai, mi, tc))
            age_avg.sort(key=lambda r: -r[1])
            top_k = age_avg[:5]
            if top_k:
                print(f"\n     Top 5 ages by avg iters:")
                print(f"       {'Age':>3}  {'Phase':<6}  {'avg_iter':>8}  {'max_iter':>8}  {'calls':>8}")
                for age_t, ai, mi, tc in top_k:
                    lbl = "RETIRE" if age_t >= retire_age else "WORK"
                    print(f"       {age_t:3d}  {lbl:<6}  {ai:>8.2f}  {mi:>8}  {tc:>8,}")

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
            print(f"     WARNING: {total_mono} total violations across {n_affected_ages}/{max(n_nonterminal_ages_solved, 1)} ages")
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
        C_eval = C_mat[solved_age_mask]
        S_eval = S_mat[solved_age_mask]
        B_eval = B_mat[solved_age_mask]
        nan_c = int(np.isnan(C_eval).sum())
        nan_s = int(np.isnan(S_eval).sum())
        nan_b = int(np.isnan(B_eval).sum())
        inf_c = int(np.isinf(C_eval).sum())
        inf_s = int(np.isinf(S_eval).sum())
        inf_b = int(np.isinf(B_eval).sum())
        neg_c = int((C_eval < 0).sum())
        neg_euler = int(all_int[DI_NEG_CONSUMPTION])
        alpha_s_neg = int((S_eval < -1e-6).sum())
        alpha_b_neg = int((B_eval < -1e-6).sum())
        alpha_sum_viol = int(((S_eval + B_eval) > 1.0 + 1e-6).sum())
        total_el = C_eval.size + S_eval.size + B_eval.size

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

    if diagnostics["is_partial"]:
        _mask_unsolved_ages_in_place(C_mat, S_mat, B_mat, solved_age_mask)

    if solve_status == "interrupted" and not return_partial_on_interrupt:
        raise KeyboardInterrupt

    return C_mat, S_mat, B_mat, diagnostics

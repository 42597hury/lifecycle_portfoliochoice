"""
model.py — Model definitions and stateless helper functions.

Contains:
  - LifecyclePortfolioModel(NamedTuple) — full model specification
  - DiscretizationConfig(NamedTuple) — grid/quadrature choices
  - SolverConfig(NamedTuple) — Newton solver tuning knobs
  - SolveControl(NamedTuple) — optional non-numerical solver controls
  - CRRA utility constructors
  - Bequest utility functions (Catherine 2025)
  - Tax and income helper functions

Dependencies: numpy, scipy (no project imports)
"""

import numpy as np
from numba import njit
from scipy.stats import norm
from typing import NamedTuple, Callable, Any


# =============================================================================
# MODEL CLASS
# =============================================================================

class LifecyclePortfolioModel(NamedTuple):
    """Model specification with generic state-return partition."""

    # Preferences
    u: Callable
    u_prime: Callable
    u_prime_inv: Callable
    beta: float
    gamma: float

    # Bequest
    b_bar: int       # bequest horizon in years (Catherine 2025: 10)

    # Lifecycle
    start_age: int
    retire_age: int
    terminal_age: int
    # survival_probs: moved to Precompute as survival_probs_2d (n_age, n_z)

    # Labor income (Catherine 2025 / Guvenen et al. 2022)
    b0: float            # age-earnings intercept
    b1: float            # age-earnings linear
    b2: float            # age-earnings quadratic (/10)
    b3: float            # age-earnings cubic (/100)
    rho: float
    pz: float
    mu_eta1: float
    sigma_eta1: float
    mu_eta2: float   # DERIVED (not free): pinned by E[eta]=0, so mu_eta2 = -(pz/(1-pz))*mu_eta1. Quadrature recomputes from this formula; stored value here is informational and may be ignored.
    sigma_eta2: float
    pe: float
    mu_eps1: float
    sigma_eps1: float
    mu_eps2: float   # DERIVED (not free): pinned by E[eps]=0, so mu_eps2 = -(pe/(1-pe))*mu_eps1. Quadrature recomputes from this formula; stored value here is informational and may be ignored.
    sigma_eps2: float

    # Partitioned VAR structure
    n_state: int
    n_ret: int
    state_names: tuple
    ret_names: tuple

    z_bar_state: np.ndarray
    z_bar_ret: np.ndarray

    Phi_0_state: np.ndarray
    Phi_11: np.ndarray
    Phi_0_ret: np.ndarray
    Phi_21: np.ndarray

    Sigma_ss: np.ndarray
    Sigma_rr: np.ndarray
    Sigma_rs: np.ndarray
    M: np.ndarray
    Sigma_r_cond: np.ndarray

    y_1_index_in_state: int | None      # Index of y_1 in state vector if present on the grid; None if y_1 is supplied as a scalar fallback
    spr_index_in_state: int | None      # Index of spr in state vector if present on the grid; None if spread is supplied as a scalar fallback
    y_1_scalar_fallback: float | None   # Required iff y_1_index_in_state is None; sample mean of y_1 used by the bequest annuity factor
    spr_scalar_fallback: float | None   # Required iff spr_index_in_state is None; sample mean of spread used by the bequest annuity factor

    # Portfolio constraints
    constrained: bool            # True = no short-selling/leverage, False = unconstrained


# =============================================================================
# DISCRETIZATION CONFIG
# =============================================================================

class DiscretizationConfig(NamedTuple):
    """Discretization choices for grids and quadrature. Passed to Precompute."""

    # Wealth grid
    n_wealth: int = 150
    wealth_min: float = 0.05
    wealth_max: float = 200.0

    # Savings grid (EGM)
    n_savings: int = 150
    savings_min: float = 1e-8
    savings_max: float | None = None   # None => use wealth_max (backward compatible)

    # Financial state VAR discretization
    state_grid_sizes: tuple = (5, 5, 5)
    state_grid_mode: str = "naive"      # "naive" | "lyapunov-axis" | "cholesky" (legacy alias: "principal")
    state_n_stds: Any = 3.0             # half-width in standardized state-grid units; scalar (broadcast) or length-3 sequence (per-axis bounds in pre-transform u-coords)

    # Income process
    n_z: int = 7                        # persistent income grid points
    n_stds: float = 3.0                 # z-grid covers ±n_stds unconditional std devs
    n_eps_nodes: int = 3                # total Judd-mixture nodes for transitory shock (poly. exactness 2n-1)
    n_eta_nodes: int = 3                # total Judd-mixture nodes for persistent innovation (poly. exactness 2n-1)
    n_ret_nodes_1d: Any = 2             # Gauss-Hermite order per return dim; int (uniform) OR tuple of length n_ret
    n_state_quad_nodes: Any = 3         # GH order per state dim; int (uniform) OR tuple of length n_state for per-axis (e.g. (2,2,5) refines axis 2 only)

    # Prescribed-tails (Lobatto-style) Gauss-Hermite, per axis. None means
    # standard Gauss-Hermite on that axis (default; backward compatible).
    # A float means tail nodes prescribed at +/- Z on that axis with closed-
    # form K in {3,5,7}. A length-n tuple selects per-axis: e.g.
    # (None, None, 4.0) puts Lobatto on axis 2 only with Z=4.
    ret_lobatto_Z: Any = None           # None | float | tuple[None|float, ...]
    state_lobatto_Z: Any = None         # None | float | tuple[None|float, ...]


# =============================================================================
# SOLVER CONFIG
# =============================================================================

class SolverConfig(NamedTuple):
    """Newton solver numerical choices. Passed to run_lifecycle_solver."""

    # --- Newton iteration ---
    tol: float = 1e-7                         # FOC convergence tolerance
    max_iter: int = 20                         # max Newton iterations (constrained)
    max_iter_unconstrained: int = 5000         # max Newton iterations (unconstrained)
    edge_max_iter: int = 8                     # max iterations for 1D edge Newton

    # --- Initial guess ---
    # NOTE: these scalars are COLD-START fallbacks only. They are consulted in
    # exactly three places:
    #   (1) The first per-savings cell of each (state, z) sweep inside the
    #       lifecycle/IH inner kernels — terminal sweep in solver.py
    #       (`_solve_terminal_age_step_quad_jit`-style block: `warm_s = ... .init_alpha_s`),
    #       `_solve_retirement_step_quad_jit`, and `_solve_working_age_step_quad_jit`
    #       (`last_a_s = sc.init_alpha_s`). Subsequent cells in the same sweep
    #       are warm-started from the previous cell's optimum
    #       (`warm_s = opt_s` / `last_a_s = opt_s`).
    #   (2) The IH outer-loop initial-policy buffers when the corresponding
    #       `warm_start_s` / `warm_start_b` argument is None (see
    #       `_prepare_initial_policies` in lifecycle/inf_horizon_solver.py:
    #       the `S_old is None` / `B_old is None` branches), AND in the
    #       inner-kernel smoke test (`compile_inner_kernel_smoke_test`) which
    #       cold-starts the next-period policy buffer with these scalars.
    #   (3) Per-cell warm-start RESEED after an unconstrained Newton failure
    #       (`EC_NEWTON_FAIL` branch in `_solve_retirement_step_quad_jit` and
    #       `_solve_working_age_step_quad_jit`, search for `DI_WARM_RESET`):
    #       the failed cell's warm-start is reset to these scalars before the
    #       next savings index is solved.
    # In iterative paths (lifecycle backward induction, IH outer loop), once
    # iteration 0 completes, the per-(z, state, wealth) warm-start ARRAY
    # (`c_next_full`, `s_next_full`, `b_next_full` in inf_horizon_solver.py;
    # `policy_alpha_{s,b}` arrays in lifecycle solver) drives the inner kernel,
    # and these scalars no longer influence the result. Tweaking them after
    # iter 0 has converged the policy will NOT change anything except the
    # value used by the per-cell reseed in case (3).
    init_alpha_s: float = 0.1                  # cold-start stock weight (see note above)
    init_alpha_b: float = 0.4                  # cold-start bond weight (see note above)

    # --- Step control ---
    step_damp_constrained: float = 0.2         # max Newton step length (constrained)
    step_damp_unconstrained: float = 0.3       # max Newton step length (unconstrained, line search off)
    grad_step_size: float = 0.05               # gradient descent step when Jacobian singular

    # --- Backtracking line search (unconstrained solver only) ---
    use_line_search: bool = True               # enable backtracking line search
    max_backtrack_iter: int = 10               # max halvings: alpha_min = 1/2^10 ≈ 0.001
    line_search_max_step: float = 2.0          # raw step cap before backtracking (replaces step_damp when on)

    # --- Thresholds ---
    tiny_savings: float = 1e-6                 # below this, skip solver (all-bills)
    corner_tol: float = 1e-8                   # KKT tolerance multiplier for corner acceptance
    edge_accept_factor: float = 10.0           # edge acceptance = tol * scale * this factor
    singular_det: float = 1e-15                # Jacobian determinant singularity threshold
    grad_denom_eps: float = 1e-10              # epsilon in gradient fallback denominator

    # --- Safety clamps ---
    min_wealth_inv: float = 1e-10              # floor for max(s_val * R_p, ...)
    min_consumption: float = 1e-10             # floor for max(c_next, ...)
    min_return_power: float = 1e-15            # floor for max(R_p, ...) before power

    # --- Probability / EGM / Euler ---
    prob_skip_threshold: float = 1e-12         # skip states with prob below this
    euler_inv_floor: float = 1e-20             # floor for beta*euler before inversion
    egm_anchor: float = 1e-10                  # anchor value for EGM grid at zero savings

    # --- Numerical leverage cap (unconstrained branch only) ---
    alpha_min: float = -10.0                   # lower bound on alpha_s and alpha_b in the unconstrained Newton
    alpha_max: float = +10.0                   # upper bound on alpha_s and alpha_b in the unconstrained Newton

    # --- Bequest regularization (luxury-bequest shifter; De Nardi 2004) ---
    # Sentinel: any value < 0 means "use the module-level DELTA_BEQUEST". This
    # preserves the legacy workflow where the canonical default lives at line
    # ~273 of model.py and smoke tests can override by editing it. Override
    # cleanly via SolverConfig._replace(delta_bequest=X) for sweep cells —
    # values >= 0 take precedence over DELTA_BEQUEST. See scripts/_smoke_delta_zero.py
    # for the legacy DELTA_BEQUEST-edit workflow.
    delta_bequest: float = -1.0                # < 0 => use module DELTA_BEQUEST

    # --- Wealth dynamics specification ---
    # "ccv_log":      x_{t+1} = s * exp(r_p^CCV) + pi    (no clamp; r_p^CCV = log return,
    #                 Campbell-Viceira 2002 / CCV w8566 eq.10 with Jensen + Ito corrections)
    # "simple_clamp": x_{t+1} = max(s * (alpha_s R_s + alpha_b R_b + a_bill R_bill), 0) + pi
    #                 (legacy spec; kept as a regression backstop)
    # The portfolio FOC under "ccv_log" is the gradient of V (NOT the asset-pricing
    # moment condition E[mu_comb*(R_j-R_bill)]); see docs/CCV_RETURNS.md.
    # Solver and simulator MUST use the same value (otherwise sR_p disagrees and
    # all Euler-residual diagnostics become meaningless).
    wealth_dynamics_spec: str = "ccv_log"


# =============================================================================
# SOLVE CONTROL
# =============================================================================

class SolveControl(NamedTuple):
    """Optional non-numerical controls for partial solves and checkpoints."""

    youngest_age_to_solve: int | None = None
    checkpoint_path: str | None = None
    checkpoint_every_n_ages: int | None = None
    save_on_interrupt: bool = False
    return_partial_on_interrupt: bool = False
    progress_wealth_source: str = "scf_median"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_utility_functions(gamma):
    """Create CRRA utility functions for given risk aversion."""
    if gamma == 1.0:
        def u(c):
            return np.log(c)

        def u_prime(c):
            return 1.0 / c

        def u_prime_inv(mu):
            return 1.0 / mu
    else:
        def u(c):
            return c ** (1.0 - gamma) / (1.0 - gamma)

        def u_prime(c):
            return c ** (-gamma)

        def u_prime_inv(mu):
            return mu ** (-1.0 / gamma)

    return u, u_prime, u_prime_inv


# =============================================================================
# BEQUEST UTILITY FUNCTIONS  (Catherine 2025 + De Nardi 2004 luxury shift)
# =============================================================================

def annuity_factor(y_1, spr, b_bar):
    """
    Annuity factor with linearly interpolated term structure.

    Recovers y_20 = y_1 + spr, then interpolates discount rates between
    y_1 (1-year yield) and y_20 (20-year yield).

    A = sum_{k=1}^{b_bar} (1 + y(k))^{-k}
    where y(k) = y_1 + spr * min(k - 1, 19) / 19

    For k=1:   y(1)  = y_1.
    For k=20:  y(20) = y_1 + spr = y_20.
    For k>=20: y(k)  = y_20 (capped — do NOT extrapolate).

    Uses DISCRETE compounding (1+y)^{-k} to match the existing codebase
    convention.  Do NOT use exp(-y*k) — that's continuous compounding and
    gives a ~12 bp/yr gap at y=5%, accumulating over b_bar periods.

    Capping (rather than extrapolating) avoids unbounded discount rates if
    b_bar > 20. With Catherine's b_bar = 10 the cap never binds, but the
    defensive code documents intended behaviour.

    Parameters
    ----------
    y_1 : float or array   1-year nominal yield (annual decimal).
    spr : float or array   Yield spread: y_20 - y_1.
    b_bar : int             Bequest horizon in years (= 10).
    """
    y_1 = np.asarray(y_1, dtype=float)
    spr = np.asarray(spr, dtype=float)
    A = np.zeros_like(y_1)
    for k in range(1, b_bar + 1):
        frac = min(k - 1, 19) / 19.0
        y_k = y_1 + spr * frac
        A += (1.0 + y_k) ** (-k)
    return A


# Annuity-normalised luxury-bequest shifter (De Nardi 2004).
#
# Bound on marginal bequest utility:  mu_max = b_bar * DELTA_BEQUEST**(-gamma) / A.
# For (b_bar, gamma, A) = (10, 5, 4):
#     DELTA = 0.005  ->  mu_max ~ 8e11   (vs. raw spike ~1e30 in the unshifted spec)
#     DELTA = 0.01   ->  mu_max ~ 2.5e10
#     DELTA = 0.02   ->  mu_max ~ 7.8e8
# Sensitivity sweep gate: ship at 0.005 if optimal alpha is stable to ~5%
# across {0.001, 0.005, 0.01, 0.02}.
DELTA_BEQUEST = 0.005


def bequest_utility(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """
    Shifted (luxury) bequest utility (De Nardi 2004):
        b(W, A) = b_bar * (max(W, 0)/A + delta)^(1-gamma) / (1-gamma)

    The shift `delta` bounds marginal bequest utility at the bankruptcy
    boundary, removing the unshifted CRRA-with-clamp `infty` discontinuity.
    Convergence to pure CRRA: as delta -> 0 the shifted spec recovers the
    original `b_bar * (W/A)**(1-gamma) / (1-gamma)` pointwise on W > 0.

    Parameters
    ----------
    W     : float or array  End-of-period wealth (bequest).
    A     : float or array  Annuity factor A(y_1, spr, b_bar) at current state.
    gamma : float           CRRA risk aversion.
    b_bar : int             Bequest weight / horizon (Catherine 2025: 10).
    delta : float           Luxury shift (default DELTA_BEQUEST).
    """
    C_bar = np.maximum(W, 0.0) / A + delta
    return b_bar * C_bar ** (1.0 - gamma) / (1.0 - gamma)


def bequest_marginal(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """
    Marginal shifted bequest utility:
        db/dW = b_bar * (W/A + delta)^(-gamma) / A     for W > 0
              = 0                                       for W <= 0

    The bankruptcy clamp returns 0 on the dead branch (heirs inherit nothing
    when realised estate is non-positive).
    """
    pos = W > 0.0
    C_bar = np.where(pos, W / A + delta, 1.0)  # placeholder for W<=0 branch
    mu = b_bar * C_bar ** (-gamma) / A
    return np.where(pos, mu, 0.0)


def bequest_marginal_inv(mu, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """
    Inverse marginal of the shifted bequest. Domain: mu in (0, mu_max] with
    mu_max = b_bar * delta**(-gamma) / A. Above mu_max the wealth constraint
    W = 0 binds and the inverse clamps to zero.

        W = A * max((mu * A / b_bar)^(-1/gamma) - delta, 0)
    """
    mu_max = b_bar * delta ** (-gamma) / A
    mu_clamped = np.minimum(mu, mu_max)
    inner = (mu_clamped * A / b_bar) ** (-1.0 / gamma) - delta
    return A * np.maximum(inner, 0.0)


# =============================================================================
# TAX AND INCOME FUNCTIONS
# =============================================================================

def disposable_income_working(y_gross):
    """After-tax labor income using the same progressive schedule as prior model."""
    y = np.asarray(y_gross, dtype=float)

    payroll_tax = 0.106 * np.minimum(y, 2.5)
    taxable_income = np.maximum(0.0, y - payroll_tax)

    tax = np.zeros_like(taxable_income)
    m = taxable_income <= 0.18
    tax[m] = taxable_income[m] * 0.10
    m = (taxable_income > 0.18) & (taxable_income <= 0.72)
    tax[m] = 0.018 + (taxable_income[m] - 0.18) * 0.12
    m = (taxable_income > 0.72) & (taxable_income <= 1.54)
    tax[m] = 0.0828 + (taxable_income[m] - 0.72) * 0.22
    m = (taxable_income > 1.54) & (taxable_income <= 2.94)
    tax[m] = 0.2632 + (taxable_income[m] - 1.54) * 0.24
    m = (taxable_income > 2.94) & (taxable_income <= 3.73)
    tax[m] = 0.5992 + (taxable_income[m] - 2.94) * 0.32
    m = (taxable_income > 3.73) & (taxable_income <= 9.32)
    tax[m] = 0.8520 + (taxable_income[m] - 3.73) * 0.35
    m = taxable_income > 9.32
    tax[m] = 2.8085 + (taxable_income[m] - 9.32) * 0.37

    return taxable_income - tax


@njit(fastmath=True)
def scalar_disposable_income(y_gross):
    """After-tax labor income for a single scalar gross income value.

    Identical tax schedule to disposable_income_working() but operates
    on a single float for use inside Numba-compiled solver loops.

    Parameters
    ----------
    y_gross : float
        Gross labor income in model units.

    Returns
    -------
    float
        Disposable (after-tax, after-payroll) income.
    """
    payroll_tax = 0.106 * min(y_gross, 2.5)
    taxable = max(0.0, y_gross - payroll_tax)

    if taxable <= 0.18:
        tax = taxable * 0.10
    elif taxable <= 0.72:
        tax = 0.018 + (taxable - 0.18) * 0.12
    elif taxable <= 1.54:
        tax = 0.0828 + (taxable - 0.72) * 0.22
    elif taxable <= 2.94:
        tax = 0.2632 + (taxable - 1.54) * 0.24
    elif taxable <= 3.73:
        tax = 0.5992 + (taxable - 2.94) * 0.32
    elif taxable <= 9.32:
        tax = 0.8520 + (taxable - 3.73) * 0.35
    else:
        tax = 2.8085 + (taxable - 9.32) * 0.37

    return taxable - tax


def compute_pension_after_tax(z_grid, avg_det):
    """
    Social Security benefits following Catherine (2025, eq. 19).

    Parameters
    ----------
    z_grid : array, shape (n_z,)
        Persistent income grid (log, mean-zero).
    avg_det : float
        Mean of exp(f(age)) over working ages. Converts the persistent
        component exp(z) to an AIME proxy: AIME(z) = exp(z) * avg_det.

    Returns
    -------
    pension_net : array, shape (n_z,)
        After-tax annual pension benefit in model units.
    """
    z = np.asarray(z_grid, dtype=float)

    # AIME: career-average earnings, capped at taxable maximum (2.5 * L_bar)
    # Catherine eq. (20): AIYE = L_bar_t * sum min{L_tilde_is, 2.5}
    aime = np.minimum(np.exp(z) * avg_det, 2.5)

    # PIA formula -- Catherine eq. (19)
    b1, b2 = 0.21, 1.25
    r1, r2, r3 = 0.90, 0.32, 0.15

    pension = np.zeros_like(aime)

    lo = aime <= b1
    pension[lo] = aime[lo] * r1

    mid = (aime > b1) & (aime <= b2)
    pension[mid] = r1 * b1 + r2 * (aime[mid] - b1)

    hi = aime > b2
    pension[hi] = r1 * b1 + r2 * (b2 - b1) + r3 * (aime[hi] - b2)

    tax = np.zeros_like(pension)
    m = pension <= 0.18
    tax[m] = pension[m] * 0.10
    m = (pension > 0.18) & (pension <= 0.72)
    tax[m] = 0.018 + (pension[m] - 0.18) * 0.12
    m = (pension > 0.72) & (pension <= 1.54)
    tax[m] = 0.0828 + (pension[m] - 0.72) * 0.22
    m = (pension > 1.54) & (pension <= 2.94)
    tax[m] = 0.2632 + (pension[m] - 1.54) * 0.24
    m = (pension > 2.94) & (pension <= 3.73)
    tax[m] = 0.5992 + (pension[m] - 2.94) * 0.32
    m = (pension > 3.73) & (pension <= 9.32)
    tax[m] = 0.8520 + (pension[m] - 3.73) * 0.35
    m = pension > 9.32
    tax[m] = 2.8085 + (pension[m] - 9.32) * 0.37

    return pension - tax

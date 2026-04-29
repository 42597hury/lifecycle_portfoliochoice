"""
model.py — Model definitions and stateless helper functions.

Contains:
  - LifecyclePortfolioModel(NamedTuple) — full model specification
  - DiscretizationConfig(NamedTuple) — grid/quadrature choices
  - SolverConfig(NamedTuple) — Newton solver tuning knobs
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

    y_1_index_in_state: int       # Index of y_1 (1-year nominal yield) in state vector (= 0)
    spr_index_in_state: int       # Index of spr (yield spread) in state vector (= 1)

    # Portfolio constraints
    constrained: bool            # True = no short-selling/leverage, False = unconstrained


# =============================================================================
# DISCRETIZATION CONFIG
# =============================================================================

class DiscretizationConfig(NamedTuple):
    """Discretization choices for grids and quadrature. Passed to Precompute."""

    # Wealth grid
    n_wealth: int = 150
    wealth_min: float = 1e-4
    wealth_max: float = 200.0

    # Savings grid (EGM)
    n_savings: int = 150
    savings_min: float = 1e-8

    # Financial state VAR discretization
    state_grid_sizes: tuple = (5, 5, 5)
    state_grid_mode: str = "naive"      # "naive" | "lyapunov-axis" | "principal"
    state_n_stds: float = 3.0           # half-width in standardized state-grid units

    # Income process
    n_z: int = 7                        # persistent income grid points
    n_stds: float = 3.0                 # z-grid covers ±n_stds unconditional std devs
    n_eps_nodes: int = 3                # total Judd-mixture nodes for transitory shock (poly. exactness 2n-1)
    n_eta_nodes: int = 3                # total Judd-mixture nodes for persistent innovation (poly. exactness 2n-1)
    n_ret_nodes_1d: Any = 2             # Gauss-Hermite order per return dim; int (uniform) OR tuple of length n_ret
    n_state_quad_nodes: int = 3         # GH order per state dimension for state innovation quadrature


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
    init_alpha_s: float = 0.1                  # initial stock weight guess
    init_alpha_b: float = 0.4                  # initial bond weight guess

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
# BEQUEST UTILITY FUNCTIONS  (Catherine 2025, equations 21-22)
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


def bequest_utility(W, A, gamma, b_bar):
    """
    Bequest utility:  b(W, r_f) = b_bar * (W / A)^(1 - gamma) / (1 - gamma)

    where  A = annuity_factor(r_f, ...)  is precomputed for the relevant state
    and    C_bar = W / A  is the flow-equivalent consumption implied by wealth W
    spread over b_bar annuity periods.

    Parameters
    ----------
    W     : float or array  End-of-period wealth (bequest).
    A     : float or array  Annuity factor A(r_f, b_bar) at current financial state.
    gamma : float           CRRA risk aversion.
    b_bar : int             Bequest weight / horizon (Catherine 2025: 10).
    """
    C_bar = W / A
    return b_bar * C_bar**(1.0 - gamma) / (1.0 - gamma)


def bequest_marginal(W, A, gamma, b_bar):
    """
    Marginal bequest utility:  db/dW = b_bar * (W / A)^(-gamma) / A

    Parameters
    ----------
    W     : float or array  End-of-period wealth.
    A     : float or array  Annuity factor at current financial state.
    gamma : float           CRRA risk aversion.
    b_bar : int             Bequest weight / horizon.
    """
    C_bar = W / A
    return b_bar * C_bar**(-gamma) / A


def bequest_marginal_inv(mu, A, gamma, b_bar):
    """
    Inverse of bequest_marginal: given  mu = db/dW,  solve for W.

    W = A * (mu * A / b_bar)^(-1/gamma)

    Used in the EGM terminal condition: the period-T+1 "value" is bequest
    utility, so the marginal value of wealth is bequest_marginal(W, A, ...).
    Inverting gives the optimal terminal wealth as a function of the shadow
    price mu.
    """
    return A * (mu * A / b_bar)**(-1.0 / gamma)


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

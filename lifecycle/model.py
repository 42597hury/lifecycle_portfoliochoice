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
from typing import NamedTuple, Any


# =============================================================================
# MODEL CLASS
# =============================================================================

class LifecyclePortfolioModel(NamedTuple):
    """Model specification with generic state-return partition."""

    # Preferences
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

    y_1_index_in_state: int | None      # Index of y_1 in state vector. In the real-yields pivot y_1 is the real bill yield and IS the bill anchor (R_bill_{t+1} = y_1_t); must not be None for a solvable model. None is permitted only for legacy var_configs that supply y_1 as a scalar fallback.
    spr_index_in_state: int | None      # Index of spr in state vector if present on the grid; None if spread is supplied as a scalar fallback
    rtb_index_in_state: int | None      # Index of rtb in state vector under the legacy nominal rtb-as-state model. None in the real-yields pivot, where the bill anchor is y_1 (real bill yield) and there is no separate rtb axis.
    y_1_scalar_fallback: float | None   # Required iff y_1_index_in_state is None; sample mean of y_1 used by the bequest annuity factor
    spr_scalar_fallback: float | None   # Required iff spr_index_in_state is None; sample mean of spread used by the bequest annuity factor


# =============================================================================
# DISCRETIZATION CONFIG
# =============================================================================

class DiscretizationConfig(NamedTuple):
    """Discretization choices for grids and quadrature. Passed to Precompute."""

    # Wealth grid
    n_wealth: int = 150
    wealth_min: float = 0.05
    wealth_max: float = 200.0
    wealth_grid_path: str | None = None

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
    """Newton solver numerical choices. Passed to run_lifecycle_solver.

    Canonical (unconstrained, JAX, CCV log-wealth) only. The constrained
    Newton, alpha leverage caps, edge/corner solvers, and the simple_clamp
    wealth-dynamics branch were removed in the JAX rewrite (handoff 2).
    Surviving fields configure the JAX 2D Newton + line search + EGM scan.
    """

    # --- Newton iteration ---
    tol: float = 1e-7                          # FOC convergence tolerance
    max_iter: int = 5000                       # max Newton outer iterations
    # Legacy alias: pre-rewrite this was ``max_iter_unconstrained``. The JAX
    # solver's run_lifecycle_solver still reads ``max_iter_unconstrained`` for
    # backwards compatibility with old configs; new code should use ``max_iter``.
    max_iter_unconstrained: int = 5000
    # Newton outer/inner loop implementation.
    # When True, use lax.fori_loop + masked updates (GPU-friendly: deterministic
    # dispatch graph, no warp divergence). When False, use lax.while_loop with
    # data-dependent termination (CPU-friendly: real early termination).
    # WARNING: under fori_loop the Newton body runs ``max_iter`` iters
    # unconditionally — early termination is mask-based, not loop-exit. Tune
    # ``max_iter`` accordingly; the canonical 5000 default is unsuitable for
    # fori_loop mode (override to <= 200 in production runs).
    use_fori_newton: bool = True

    # --- Initial guess ---
    init_alpha_s: float = 0.1                  # initial stock weight guess
    init_alpha_b: float = 0.4                  # initial bond weight guess

    # --- Step control ---
    step_damp_unconstrained: float = 0.3       # legacy; line search supersedes this
    grad_step_size: float = 0.05               # gradient descent step when Jacobian singular

    # --- Backtracking line search ---
    use_line_search: bool = True               # always True post-rewrite
    max_backtrack_iter: int = 10               # max halvings
    line_search_max_step: float = 2.0          # raw Newton step cap

    # --- Thresholds ---
    tiny_savings: float = 1e-6                 # below this, hold cold-init alphas, c -> floor
    singular_det: float = 1e-15                # Jacobian determinant singularity threshold
    grad_denom_eps: float = 1e-10              # epsilon in gradient fallback denominator

    # --- Safety clamps ---
    min_consumption: float = 1e-10             # floor for c_next interpolation

    # --- EGM / Euler ---
    euler_inv_floor: float = 1e-20             # floor for beta*V_dot before inversion
    egm_anchor: float = 1e-10                  # anchor value for EGM grid at zero savings

    # --- Bequest regularization (luxury-bequest shifter; De Nardi 2004) ---
    # Sentinel: any value < 0 means "use the module-level DELTA_BEQUEST".
    delta_bequest: float = -1.0

    # --- Wealth dynamics specification ---
    # The JAX solver implements ``ccv_log`` only. The constant is kept for
    # config-print compatibility; setting it to anything else raises in
    # run_lifecycle_solver.
    wealth_dynamics_spec: str = "ccv_log"

    # --- Newton init source ---
    # When True, every non-terminal age seeds Newton at each (z, state) cell
    # from the previous (older) age's converged policy at that same cell
    # (gathered at mid-wealth). When False, every cell uses the canonical
    # scalar (init_alpha_s, init_alpha_b). Terminal age is always cold.
    use_backward_age_warm_start: bool = True

    # --- Cell-axis chunking (single-/multi-device memory bounding) ---
    # Splits the per-age vmap over (n_z * N_state) cells into this many
    # sequential chunks. Each chunk has fixed shape (chunk_size, ...) so XLA
    # traces the inner vmap once and reuses for all chunks. Per-chunk peak
    # HBM = (worst-case full materialisation / K), giving a deterministic
    # memory bound independent of XLA's compilation choices.
    #
    # When 1: no chunking (default; matches today's behaviour, fastest dispatch).
    # When > 1: K sequential chunk calls per age. Adds ~K kernel-launch
    # dispatches per age but bounds memory. On pmap, each chunk is rounded up
    # to a multiple of the device count before being sharded across devices.
    #
    # Heuristic for picking K on a single GPU:
    #   per_cell_memory_MB = n_state_quad * n_z * 2**n_state * n_w * 8 / 1e6
    #   total_worst_case   = per_cell_memory_MB * (n_z * N_state)
    #   K_min              = ceil(total_worst_case / target_HBM_budget_MB)
    # On GH200 (97 GB HBM, ~30 GB headroom for other state):
    #   target_HBM_budget = 60 GB → K = ceil(total_worst_case / 60_000)
    #
    # Both vmap-only and pmap dispatch paths honour this knob.
    cell_vmap_chunks: int = 1

    # --- Mixed precision toggle ---
    # When "f64" (default): all solver arithmetic in fp64. Bit-identical to the
    # pre-toggle baseline.
    # When "f32": the multilinear/bilinear interpolation boundary runs in fp32;
    # results cast back to fp64 BEFORE any CRRA / FOC / Newton arithmetic.
    # Captures memory-bandwidth savings without precision loss in
    # convergence-critical paths.
    #
    # Cast scope (closed set; see _cast_for_gather, solver.py):
    #   IN  (f64 -> f32): c_corners (working & retirement), w_corners,
    #                     wealth_grid, frac_z, x_next_scalar
    #   OUT (f32 -> f64): c_g, mpc_g (immediately before the min_consumption
    #                     floor and [0,1] MPC clip)
    # Boundary closes inside _interp_c_and_mpc_at_cell (solver.py:954-955)
    # and the inline per_kv_kr in retirement_foc_jac_ccv (solver.py:1025-1026).
    # Nothing fp32 ever exits the multilinear closure.
    #
    # WARNING: f32 path produces alphas that differ from f64 by ~1e-5 relative
    # (real arithmetic noise, not just bit-shuffle). Bit-identity test does
    # not apply; agreement test at 1e-4 relative does.
    gather_precision: str = "f64"


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

def crra_u(c, gamma):
    if gamma == 1.0:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def crra_uprime(c, gamma):
    return c ** (-gamma) if gamma != 1.0 else 1.0 / c


def crra_uprime_inv(mu, gamma):
    return mu ** (-1.0 / gamma) if gamma != 1.0 else 1.0 / mu


def create_utility_functions(gamma):
    """Legacy CRRA factory; superseded by module-level crra_u/crra_uprime/crra_uprime_inv.

    Kept only so any external caller importing this name does not break.
    No production code in this package calls it.
    """
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

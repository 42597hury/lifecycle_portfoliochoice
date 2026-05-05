"""
precompute.py — Precompute grids, transitions, and lookup tables.

Contains:
  - Precompute (NamedTuple) — all arrays consumed by the solver
  - build_precompute() — factory assembling Precompute from a model + DiscretizationConfig
  - build_model() — factory assembling LifecyclePortfolioModel from configs

Dependencies: model, var, discretization, mortality
"""

from typing import Any, NamedTuple

import numpy as np

from lifecycle.model import (
    LifecyclePortfolioModel, DiscretizationConfig,
    annuity_factor,
    disposable_income_working, compute_pension_after_tax,
)
from lifecycle.var import partition_var
from lifecycle.discretization import (
    build_state_grid, discretize_income_ar1_mixture,
    get_eps_quadrature_corrected, get_eta_quadrature_mixture, get_return_quadrature,
    get_state_quadrature, _normalize_ret_nodes,
)
from lifecycle.mortality import calibrate_earnings_dependent_mortality


# =============================================================================
# Banner helper
# =============================================================================

def _format_lobatto_axes(lobatto_Z):
    """Render '  (Lobatto axes 0:Z=3.0, 2:Z=4.0)' suffix for the print banner.

    Returns '' when Lobatto is unused so the existing banner is unchanged
    for production-default configs.
    """
    if lobatto_Z is None:
        return ""
    if isinstance(lobatto_Z, (int, float)):
        return f"  (Lobatto all axes Z={float(lobatto_Z)})"
    try:
        seq = list(lobatto_Z)
    except TypeError:
        return ""
    parts = [f"{d}:Z={float(v)}" for d, v in enumerate(seq) if v is not None]
    if not parts:
        return ""
    return "  (Lobatto " + ", ".join(parts) + ")"


# =============================================================================
# PRECOMPUTE (NamedTuple — built via build_precompute)
# =============================================================================

class Precompute(NamedTuple):
    """
    Precompute grids, transitions, and lookup tables.

    Generic design:
    - Grid only over model state variables.
    - Return variables integrated via conditional means mu_r[i, j, k].

    Construction is via :func:`build_precompute`. Direct positional construction
    via ``Precompute(...)`` is supported by the NamedTuple machinery but
    callers should always go through ``build_precompute(model, disc_config)``.

    Solver input reference (all arrays consumed by the solver):
    -------------------------------------------------------
    Grids:
      wealth_grid  (n_w,)               cash-on-hand interpolation points [geom]
      s_grid       (n_s,)               savings grid for EGM endogenous gridpoints
      ages         (n_age,)             integer ages; index t -> ages[t]

    Financial state (VAR):
      state_grid   (N_state, n_state)   joint state grid; row i = slow-state vector
      Pi_state     (N_state, N_state)   Pi_state[i,j] = P(s_{t+1}=j | s_t=i)
      mu_r         (N_state, N_state, n_ret)
                                        mu_r[i,j,0] = E[rtb | s_t=i, s_{t+1}=j]  log real bill return
                                        mu_r[i,j,1] = E[xr  | s_t=i, s_{t+1}=j]  log excess stock return
                                        mu_r[i,j,2] = E[xb  | s_t=i, s_{t+1}=j]  log excess bond return
      ret_nodes    (n_ret_quad, n_ret)  residual log-return shocks drawn from N(0, Sigma_r_cond)
      ret_weights  (n_ret_quad,)        quadrature weights; sum=1
      exp_ret_bill (n_ret_quad,)        exp(ret_nodes[:, 0]) for rtb residuals

    Income:
      z_grid       (n_z,)               persistent income states (log, mean-zero)
      Pi_z         (n_z, n_z)           Pi_z[i,j] = P(z_{t+1}=j | z_t=i)
      eps_nodes    (n_eps,)             Gauss-Hermite nodes for transitory shock eps
      eps_weights  (n_eps,)             quadrature weights; sum=1, E[eps]=0 enforced

    Bequest:
      annuity_factors  (N_state,)       A(y_1, spr, b_bar) annuity factor at each state
                                        (used by the shifted bequest_utility /
                                        bequest_marginal; shift parameter
                                        DELTA_BEQUEST lives in lifecycle.model)

    Lookup tables:
      working_income      (n_age, n_z, n_eps)
                                        working_income[t, iz, ie] = after-tax labor income
                                        at age ages[t], persistent state iz, transitory node ie
      working_income_next (n_age, n_z, n_eta, n_eps)
                                        Next-period working-age income evaluated at
                                        z_{t+1} = rho*z_t + eta_{t+1}, transitory eps_{t+1}.
                                        Zero for rows where t+1 is not a working-age period.
                                        Consumed by the JAX FOC kernel (handoff 2).
      pension_after_tax   (n_age, n_z)
                                        pension_after_tax[t, iz] = after-tax Social Security
                                        benefit; constant across ages, indexed by career z

    Dimension counters: n_w, n_s, n_z, n_eps, n_eta, n_ret_quad, n_age, N_state
    """

    # Config back-references
    disc_config: DiscretizationConfig
    model: LifecyclePortfolioModel

    # Quadrature size config (normalized)
    n_ret_nodes_1d: tuple

    # Grids
    wealth_grid: np.ndarray
    savings_max: float
    s_grid: np.ndarray
    ages: np.ndarray

    # Financial state VAR
    state_grid_sizes: list
    state_grid_mode: str
    state_grid_mu_s: np.ndarray
    state_grid_sigma_z: np.ndarray
    state_bracket_shift: np.ndarray
    state_bracket_L_inv: np.ndarray
    state_bracket_grids: Any
    state_indices: np.ndarray
    state_grid: np.ndarray
    Pi_state: np.ndarray
    state_stationary_probs: np.ndarray
    N_state: int

    # Returns
    mu_r: np.ndarray
    ret_nodes: np.ndarray
    ret_weights: np.ndarray

    # State quadrature
    v_nodes: np.ndarray
    v_weights: np.ndarray
    n_state_quad: int

    # Conditional return formula constants
    const_r: np.ndarray
    A_r: np.ndarray
    M_v_nodes: np.ndarray

    # Exp of return-quadrature residual nodes
    exp_ret_bill: np.ndarray
    exp_ret_stock: np.ndarray
    exp_ret_bond: np.ndarray

    # CCV conditional-covariance scalars
    sigma2_xr: float
    sigma2_xb: float
    sigma_xrxb: float

    # Bequest
    annuity_factors: np.ndarray

    # Income discretization
    z_grid: np.ndarray
    Pi_z: np.ndarray
    eps_nodes: np.ndarray
    eps_weights: np.ndarray
    eta_nodes: np.ndarray
    eta_weights: np.ndarray
    dz: float
    log_det_profile: np.ndarray
    avg_det: float

    # Lookup tables
    working_income: np.ndarray
    pension_after_tax: np.ndarray
    working_income_next: np.ndarray

    # Mortality
    survival_probs_2d: np.ndarray
    chi_vec: np.ndarray
    mortality_diag: dict

    # Dimension counters
    n_w: int
    n_s: int
    n_z: int
    n_eps: int
    n_eta: int
    n_ret_quad: int
    n_age: int

    def regenerate_savings_grid(self, n_s_points):
        """Utility for sensitivity runs in Part 2."""
        return np.expm1(np.linspace(
            np.log1p(self.disc_config.savings_min),
            np.log1p(self.savings_max),
            int(n_s_points),
        ))


# =============================================================================
# PRECOMPUTE FACTORY
# =============================================================================

def build_precompute(model, disc_config=None, verbose=True):
    """Build a :class:`Precompute` from a model and discretization config.

    Carries out exactly the work the legacy ``Precompute.__init__`` did, then
    constructs the NamedTuple. No solver-visible behaviour changes.
    """
    if disc_config is None:
        disc_config = DiscretizationConfig()

    K_ret_per_dim = _normalize_ret_nodes(disc_config.n_ret_nodes_1d, model.n_ret)
    if any(k < 1 for k in K_ret_per_dim):
        raise ValueError("All entries of disc_config.n_ret_nodes_1d must be >= 1")
    n_ret_nodes_1d = K_ret_per_dim

    # --- Grids ---
    effective_savings_max = disc_config.wealth_max if disc_config.savings_max is None else float(disc_config.savings_max)
    if effective_savings_max <= disc_config.savings_min:
        raise ValueError("savings_max must be strictly greater than savings_min")
    if effective_savings_max > disc_config.wealth_max:
        raise ValueError("savings_max cannot exceed wealth_max; widen wealth_max instead")

    wealth_grid = np.expm1(np.linspace(
        np.log1p(disc_config.wealth_min),
        np.log1p(disc_config.wealth_max),
        disc_config.n_wealth,
    ))
    savings_max = effective_savings_max
    s_grid = np.expm1(np.linspace(
        np.log1p(disc_config.savings_min),
        np.log1p(savings_max),
        disc_config.n_savings,
    ))
    ages = np.arange(model.start_age, model.terminal_age + 1)

    # --- Financial state VAR discretization ---
    state_grid_sizes_in = list(disc_config.state_grid_sizes)
    if len(state_grid_sizes_in) != model.n_state:
        raise ValueError("state_grid_sizes length must equal model.n_state")
    state_grid_sizes = list(state_grid_sizes_in)

    grid_info = build_state_grid(
        N_vec=state_grid_sizes_in,
        mu_intercept=model.Phi_0_state,
        Phi=model.Phi_11,
        Sigma_innov=model.Sigma_ss,
        n_stds=disc_config.state_n_stds,
        mode=disc_config.state_grid_mode,
    )
    state_grid_mode = grid_info["mode"]
    state_grid_mu_s = grid_info["mu_s"]
    state_grid_sigma_z = grid_info["sigma_z"]
    state_bracket_shift = grid_info["bracket_shift"]
    state_bracket_L_inv = grid_info["bracket_L_inv"]
    state_bracket_grids = grid_info["state_bracket_grids"]
    state_indices = grid_info["state_indices"]
    state_grid = grid_info["state_grid"]
    Pi_state = grid_info["Pi_state"]
    state_stationary_probs = grid_info["stationary_probs"]
    N_state = state_grid.shape[0]

    # --- Conditional return means and bill rate ---
    mu_r = _precompute_conditional_returns(model, state_grid)

    ret_nodes, ret_weights = get_return_quadrature(
        model,
        n_nodes=disc_config.n_ret_nodes_1d,
        lobatto_Z=getattr(disc_config, "ret_lobatto_Z", None),
    )

    # --- State innovation quadrature ---
    v_nodes, v_weights = get_state_quadrature(
        model,
        n_nodes=disc_config.n_state_quad_nodes,
        lobatto_Z=getattr(disc_config, "state_lobatto_Z", None),
    )
    n_state_quad = len(v_weights)

    # --- Precomputed return formula constants ---
    const_r = np.array(model.Phi_0_ret, dtype=float)
    A_r = np.array(model.Phi_21, dtype=float)
    M_v_nodes = v_nodes @ model.M.T

    exp_ret_bill = np.exp(ret_nodes[:, 0])
    exp_ret_stock = np.exp(ret_nodes[:, 1])
    exp_ret_bond = np.exp(ret_nodes[:, 2])

    # --- CCV log-return covariance scalars ---
    # CCV w8566 eq. (10) takes expectations over the FULL VAR innovation
    # v_{t+1}, so the constants in r_p are sourced from the unconditional
    # return-block covariance Sigma_rr — NOT the residual Sigma_r_cond used
    # for the inner-quadrature Cholesky in discretization.get_return_quadrature.
    # Sigma_r_cond differs from Sigma_rr by the M·Sigma_ss·M' projection term,
    # which can be ~10–30x in this calibration; using the wrong matrix shrinks
    # the Itô vol-drag and inflates converged alpha_s past sensible levels.
    # Patched 2026-05-06 (Sigma_rr matches CCV Table 2 vols + Markowitz alpha).
    sigma2_xr = float(model.Sigma_rr[1, 1])
    sigma2_xb = float(model.Sigma_rr[2, 2])
    sigma_xrxb = float(model.Sigma_rr[1, 2])

    # --- Bequest annuity factors ---
    y_1_idx = model.y_1_index_in_state
    spr_idx = model.spr_index_in_state

    if y_1_idx is not None and spr_idx is not None:
        _y_1 = state_grid[:, model.y_1_index_in_state]
        _spr = state_grid[:, model.spr_index_in_state]
    else:
        if y_1_idx is not None:
            _y_1 = state_grid[:, y_1_idx]
        else:
            _y_1 = np.full(N_state, model.y_1_scalar_fallback, dtype=float)
        if spr_idx is not None:
            _spr = state_grid[:, spr_idx]
        else:
            _spr = np.full(N_state, model.spr_scalar_fallback, dtype=float)

    annuity_factors = annuity_factor(_y_1, _spr, model.b_bar)

    # --- Income discretization ---
    z_grid, Pi_z = discretize_income_ar1_mixture(
        rho=model.rho,
        p=model.pz,
        mu1=model.mu_eta1,
        sigma1=model.sigma_eta1,
        mu2=model.mu_eta2,
        sigma2=model.sigma_eta2,
        N=disc_config.n_z,
        n_stds=disc_config.n_stds,
    )

    eps_nodes, eps_weights = get_eps_quadrature_corrected(model, n_nodes=disc_config.n_eps_nodes)
    eta_nodes, eta_weights = get_eta_quadrature_mixture(model, n_nodes=disc_config.n_eta_nodes)

    dz = z_grid[1] - z_grid[0]

    # --- Deterministic age-earnings profile ---
    log_det_profile = (model.b0
                       + model.b1 * ages
                       + model.b2 * ages**2 / 10.0
                       + model.b3 * ages**3 / 100.0)

    _working_ages = np.arange(model.start_age, model.retire_age)
    _log_det_working = (model.b0
                        + model.b1 * _working_ages
                        + model.b2 * _working_ages**2 / 10.0
                        + model.b3 * _working_ages**3 / 100.0)
    avg_det = float(np.mean(np.exp(_log_det_working)))

    # --- Income lookup tables ---
    working_income = _precompute_working_income(log_det_profile, z_grid, eps_nodes)
    pension_after_tax = _precompute_pension(z_grid, ages, avg_det)

    # NEW: next-period working income table for the JAX FOC kernel (handoff 2)
    retire_age_idx = model.retire_age - model.start_age
    working_income_next = _precompute_working_income_next(
        log_det_profile=log_det_profile,
        z_grid=z_grid,
        eta_nodes=eta_nodes,
        eps_nodes=eps_nodes,
        rho=model.rho,
        retire_age_idx=retire_age_idx,
    )

    # --- Dimension counters ---
    n_w = len(wealth_grid)
    n_s = len(s_grid)
    n_z = len(z_grid)
    n_eps = len(eps_nodes)
    n_eta = len(eta_nodes)
    n_ret_quad = len(ret_weights)
    n_age = len(ages)

    # --- Earnings-dependent mortality ---
    survival_probs_2d, chi_vec, mortality_diag = calibrate_earnings_dependent_mortality(
        start_age=model.start_age,
        terminal_age=model.terminal_age,
        z_grid=z_grid,
        rho=model.rho,
        pz=model.pz,
        mu_eta1=model.mu_eta1,
        sigma_eta1=model.sigma_eta1,
        mu_eta2=model.mu_eta2,
        sigma_eta2=model.sigma_eta2,
        verbose=verbose,
    )

    pc = Precompute(
        disc_config=disc_config,
        model=model,
        n_ret_nodes_1d=n_ret_nodes_1d,
        wealth_grid=wealth_grid,
        savings_max=savings_max,
        s_grid=s_grid,
        ages=ages,
        state_grid_sizes=state_grid_sizes,
        state_grid_mode=state_grid_mode,
        state_grid_mu_s=state_grid_mu_s,
        state_grid_sigma_z=state_grid_sigma_z,
        state_bracket_shift=state_bracket_shift,
        state_bracket_L_inv=state_bracket_L_inv,
        state_bracket_grids=state_bracket_grids,
        state_indices=state_indices,
        state_grid=state_grid,
        Pi_state=Pi_state,
        state_stationary_probs=state_stationary_probs,
        N_state=N_state,
        mu_r=mu_r,
        ret_nodes=ret_nodes,
        ret_weights=ret_weights,
        v_nodes=v_nodes,
        v_weights=v_weights,
        n_state_quad=n_state_quad,
        const_r=const_r,
        A_r=A_r,
        M_v_nodes=M_v_nodes,
        exp_ret_bill=exp_ret_bill,
        exp_ret_stock=exp_ret_stock,
        exp_ret_bond=exp_ret_bond,
        sigma2_xr=sigma2_xr,
        sigma2_xb=sigma2_xb,
        sigma_xrxb=sigma_xrxb,
        annuity_factors=annuity_factors,
        z_grid=z_grid,
        Pi_z=Pi_z,
        eps_nodes=eps_nodes,
        eps_weights=eps_weights,
        eta_nodes=eta_nodes,
        eta_weights=eta_weights,
        dz=dz,
        log_det_profile=log_det_profile,
        avg_det=avg_det,
        working_income=working_income,
        pension_after_tax=pension_after_tax,
        working_income_next=working_income_next,
        survival_probs_2d=survival_probs_2d,
        chi_vec=chi_vec,
        mortality_diag=mortality_diag,
        n_w=n_w,
        n_s=n_s,
        n_z=n_z,
        n_eps=n_eps,
        n_eta=n_eta,
        n_ret_quad=n_ret_quad,
        n_age=n_age,
    )

    _validate_state_quadrature(pc, verbose=verbose)
    if verbose:
        _print_precompute_summary(pc)

    return pc


# =============================================================================
# FREE-FUNCTION HELPERS
# =============================================================================

def _precompute_conditional_returns(model, state_grid):
    """
    mu_r[i, j, :] = E[r_{t+1} | state_t=i, state_{t+1}=j]

    Vectorized form:
      mu_r[i,j] = const + A @ s_i + M @ s_j
      where const = Phi_0_ret - M @ Phi_0_state    (n_ret,)
            A     = Phi_21    - M @ Phi_11          (n_ret, n_state)
    """
    const = model.Phi_0_ret - model.M @ model.Phi_0_state
    A = model.Phi_21 - model.M @ model.Phi_11

    term_i = state_grid @ A.T
    term_j = state_grid @ model.M.T

    return const[None, None, :] + term_i[:, None, :] + term_j[None, :, :]


def _validate_state_quadrature(pc, verbose=True):
    """Verify state quadrature reproduces conditional return moments.

    For each source state i, check:
      sum_k w_k * mu_r_k == Phi_0_ret + Phi_21 @ s_i  (unconditional return mean)
    """
    model = pc.model
    max_err_mean = 0.0
    for i in range(pc.N_state):
        s_i = pc.state_grid[i]
        base_mu_r = pc.const_r + pc.A_r @ s_i
        avg_mu_r = base_mu_r + pc.v_weights @ pc.M_v_nodes
        target = model.Phi_0_ret + model.Phi_21 @ s_i
        err = np.max(np.abs(avg_mu_r - target))
        max_err_mean = max(max_err_mean, err)

    if verbose:
        print(f"  State quadrature return-mean consistency: max err = {max_err_mean:.2e}")

    assert max_err_mean < 1e-10, (
        f"State quadrature return-mean error {max_err_mean:.2e} too large"
    )


def _precompute_working_income(log_det_profile, z_grid, eps_nodes):
    """
    After-tax labor income table: shape (n_age, n_z, n_eps).

    Gross income at each (age, z, eps) grid point:
        Y_gross = exp(f(age) + z + eps)
    where f(age) = b0 + b1*age + b2*age^2/10 + b3*age^3/100 is the
    deterministic age-earnings profile (Guvenen et al. 2022).

    After-tax income applies payroll tax (10.6%, capped at 2.5) plus
    progressive income tax (7 TCJA brackets) via disposable_income_working.
    """
    z = z_grid[None, :, None]
    eps = eps_nodes[None, None, :]
    det = log_det_profile[:, None, None]

    y_gross = np.exp(det + z + eps)
    return disposable_income_working(y_gross)


def _precompute_pension(z_grid, ages, avg_det):
    """
    After-tax pension table: shape (n_age, n_z).

    Catherine (2025) eq. (20):
        AIYE_it = L_bar_t * sum min{L_tilde_is, 2.5}
    Approximation:
        AIME(z) ~ min(exp(z) * avg_det, 2.5)
    """
    base_pension = compute_pension_after_tax(z_grid, avg_det)
    n_age = len(ages)
    n_z = len(z_grid)
    return np.broadcast_to(base_pension, (n_age, n_z)).copy()


def _precompute_working_income_next(
    log_det_profile,
    z_grid,
    eta_nodes,
    eps_nodes,
    rho,
    retire_age_idx,
):
    """
    Next-period working-age disposable income at (t, iz, k_eta, i_e):

        log y_{t+1} = log_det_profile[t+1]
                      + rho * z_grid[iz]
                      + eta_nodes[k_eta]
                      + eps_nodes[i_e]
        table[t, iz, k_eta, i_e] = disposable_income_working(exp(log y_{t+1}))

    Defined only for transitions into a working-age period
    (t + 1 < retire_age_idx). Other rows are zero — the FOC kernel uses
    pension_after_tax with linear z-interpolation across the retirement
    boundary instead of this table.

    Shape: (n_age, n_z, n_eta, n_eps), dtype float64.

    Not consumed by the existing Numba solver; the JAX FOC kernel
    (handoff 2) replaces its in-loop scalar_disposable_income call
    with a lookup into this table.
    """
    n_age = len(log_det_profile)
    n_z = len(z_grid)
    n_eta = len(eta_nodes)
    n_eps = len(eps_nodes)
    table = np.zeros((n_age, n_z, n_eta, n_eps), dtype=np.float64)

    t_idx = np.arange(n_age)
    next_is_working = (t_idx + 1) < retire_age_idx
    if not np.any(next_is_working):
        return table

    log_det_next = log_det_profile[1:][:, None, None, None]   # (n_age-1, 1, 1, 1)
    z_term = (rho * z_grid)[None, :, None, None]              # (1, n_z, 1, 1)
    eta_term = eta_nodes[None, None, :, None]                 # (1, 1, n_eta, 1)
    eps_term = eps_nodes[None, None, None, :]                 # (1, 1, 1, n_eps)

    y_gross = np.exp(log_det_next + z_term + eta_term + eps_term)
    table[:-1] = disposable_income_working(y_gross)

    for t in range(n_age):
        if not next_is_working[t]:
            table[t] = 0.0
    return table


def _print_precompute_summary(pc):
    print("=" * 64)
    print("PRECOMPUTE SUMMARY")
    print("=" * 64)
    sizes_str = " x ".join(str(n) for n in pc.state_grid_sizes)
    print(f"Ages         : {pc.model.start_age} to {pc.model.terminal_age}"
          f"  ({pc.n_age} periods,"
          f" retire at {pc.model.retire_age})")
    print(f"State grid   : {sizes_str} = {pc.N_state} joint states")
    ns = pc.disc_config.state_n_stds
    ns_arr = np.atleast_1d(np.asarray(ns, dtype=float))
    if ns_arr.size == 1:
        ns_str = f"{float(ns_arr[0]):.2f}"
    else:
        ns_str = "(" + ", ".join(f"{x:.2f}" for x in ns_arr) + ")"
    print(f"  mode       : {pc.state_grid_mode}  |  half-width = {ns_str}")
    print(f"  state vars : {list(pc.model.state_names)}")
    print(f"  return vars: {list(pc.model.ret_names)}")
    print(f"Income grid  : {pc.n_z} persistent states"
          f"  x  {pc.n_eps} transitory nodes")
    K_str = "x".join(str(k) for k in pc.n_ret_nodes_1d)
    ret_Z_axes = _format_lobatto_axes(getattr(pc.disc_config, "ret_lobatto_Z", None))
    print(f"Return quad  : ({K_str}) nodes/dim"
          f"  ->  {pc.n_ret_quad} joint nodes" + ret_Z_axes)
    K_state_disp = pc.disc_config.n_state_quad_nodes
    if hasattr(K_state_disp, "__len__"):
        K_state_str = "(" + "x".join(str(k) for k in K_state_disp) + ")"
    else:
        K_state_str = str(K_state_disp)
    state_Z_axes = _format_lobatto_axes(getattr(pc.disc_config, "state_lobatto_Z", None))
    print(f"State quad   : {K_state_str} nodes/dim"
          f"  ->  {pc.n_state_quad} joint nodes" + state_Z_axes)
    print(f"Wealth grid  : {pc.n_w} points  [{pc.wealth_grid[0]:.3e}, {pc.wealth_grid[-1]:.3e}]")
    print(f"Savings grid : {pc.n_s} points  [{pc.s_grid[0]:.3e}, {pc.s_grid[-1]:.3e}]")
    print(f"mu_r         : {pc.mu_r.shape}"
          f"  ({pc.N_state * pc.N_state * pc.model.n_ret:,} values)")
    print(f"ret_nodes    : {pc.ret_nodes.shape}")
    if pc.model.y_1_index_in_state is not None:
        print(f"y_1 idx in state : {pc.model.y_1_index_in_state}"
              f"  ({pc.model.state_names[pc.model.y_1_index_in_state]})")
    else:
        print(f"y_1 (scalar)     : {pc.model.y_1_scalar_fallback:.4%}")
    if pc.model.spr_index_in_state is not None:
        print(f"spr idx in state : {pc.model.spr_index_in_state}"
              f"  ({pc.model.state_names[pc.model.spr_index_in_state]})")
    else:
        print(f"spr (scalar)     : {pc.model.spr_scalar_fallback:.4%}")
    print(f"annuity_factors      : {pc.annuity_factors.shape}  range=[{pc.annuity_factors.min():.2f}, {pc.annuity_factors.max():.2f}]")
    print(f"working_income       : {pc.working_income.shape}  (n_age x n_z x n_eps)")
    print(f"working_income_next  : {pc.working_income_next.shape}  (n_age x n_z x n_eta x n_eps)")
    print(f"pension_after_tax    : {pc.pension_after_tax.shape}  (n_age x n_z)")
    print("=" * 64)


# =============================================================================
# MODEL FACTORY (CONFIG DRIVEN)
# =============================================================================

def build_model(base_config, var_config, verbose=True):
    """Build LifecyclePortfolioModel from primitive configs."""
    parts = partition_var(
        Phi_full=np.asarray(var_config["Phi"], dtype=float),
        Omega_full=np.asarray(var_config["Omega"], dtype=float),
        z_bar=np.asarray(var_config["z_bar"], dtype=float),
        state_idx=var_config["state_indices"],
        ret_idx=var_config["return_indices"],
        variable_names=var_config["variable_names"],
        verbose=verbose,
    )

    y_1_idx_raw = var_config.get("y_1_index_in_state", None)
    y_1_scalar = var_config.get("y_1_scalar_fallback", None)
    if y_1_idx_raw is None:
        if y_1_scalar is None:
            raise ValueError(
                "var_config must provide either y_1_index_in_state "
                "or y_1_scalar_fallback (both are None)"
            )
        y_1_index_in_state = None
        y_1_scalar_fallback = float(y_1_scalar)
    else:
        y_1_index_in_state = int(y_1_idx_raw)
        if y_1_index_in_state < 0 or y_1_index_in_state >= parts["n_state"]:
            raise ValueError(
                f"y_1_index_in_state ({y_1_index_in_state}) out of bounds "
                f"for state vector of size {parts['n_state']}"
            )
        y_1_scalar_fallback = None

    spr_idx_raw = var_config.get("spr_index_in_state", None)
    spr_scalar = var_config.get("spr_scalar_fallback", None)
    if spr_idx_raw is None:
        if spr_scalar is None:
            raise ValueError(
                "var_config must provide either spr_index_in_state "
                "or spr_scalar_fallback (both are None)"
            )
        spr_index_in_state = None
        spr_scalar_fallback = float(spr_scalar)
    else:
        spr_index_in_state = int(spr_idx_raw)
        if spr_index_in_state < 0 or spr_index_in_state >= parts["n_state"]:
            raise ValueError(
                f"spr_index_in_state ({spr_index_in_state}) out of bounds "
                f"for state vector of size {parts['n_state']}"
            )
        spr_scalar_fallback = None

    if (y_1_index_in_state is not None
            and spr_index_in_state is not None
            and y_1_index_in_state == spr_index_in_state):
        raise ValueError(
            "y_1_index_in_state and spr_index_in_state must be distinct "
            "when both are grid indices"
        )

    return LifecyclePortfolioModel(
        beta=float(base_config["beta"]),
        gamma=float(base_config["gamma"]),
        b_bar=int(base_config["b_bar"]),
        start_age=int(base_config["start_age"]),
        retire_age=int(base_config["retire_age"]),
        terminal_age=int(base_config["terminal_age"]),
        b0=float(base_config["b0"]),
        b1=float(base_config["b1"]),
        b2=float(base_config["b2"]),
        b3=float(base_config["b3"]),
        rho=float(base_config["rho"]),
        pz=float(base_config["pz"]),
        mu_eta1=float(base_config["mu_eta1"]),
        sigma_eta1=float(base_config["sigma_eta1"]),
        mu_eta2=float(base_config["mu_eta2"]),
        sigma_eta2=float(base_config["sigma_eta2"]),
        pe=float(base_config["pe"]),
        mu_eps1=float(base_config["mu_eps1"]),
        sigma_eps1=float(base_config["sigma_eps1"]),
        mu_eps2=float(base_config["mu_eps2"]),
        sigma_eps2=float(base_config["sigma_eps2"]),
        n_state=parts["n_state"],
        n_ret=parts["n_ret"],
        state_names=parts["state_names"],
        ret_names=parts["ret_names"],
        z_bar_state=parts["z_bar_state"],
        z_bar_ret=parts["z_bar_ret"],
        Phi_0_state=parts["Phi_0_state"],
        Phi_11=parts["Phi_11"],
        Phi_0_ret=parts["Phi_0_ret"],
        Phi_21=parts["Phi_21"],
        Sigma_ss=parts["Sigma_ss"],
        Sigma_rr=parts["Sigma_rr"],
        Sigma_rs=parts["Sigma_rs"],
        M=parts["M"],
        Sigma_r_cond=parts["Sigma_r_cond"],
        y_1_index_in_state=y_1_index_in_state,
        spr_index_in_state=spr_index_in_state,
        y_1_scalar_fallback=y_1_scalar_fallback,
        spr_scalar_fallback=spr_scalar_fallback,
    )

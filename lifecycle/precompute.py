"""
precompute.py — Precompute grids, transitions, and lookup tables.

Contains:
  - Precompute class — all arrays consumed by the solver
  - build_model() — factory assembling LifecyclePortfolioModel from configs

Dependencies: model, var, discretization, mortality
"""

import numpy as np

from lifecycle.model import (
    LifecyclePortfolioModel, DiscretizationConfig,
    create_utility_functions, annuity_factor,
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
# PRECOMPUTE CLASS (GENERIC STATE/RETURN VERSION)
# =============================================================================

class Precompute:
    """
    Precompute grids, transitions, and lookup tables.

    Generic design:
    - Grid only over model state variables.
    - Return variables integrated via conditional means mu_r[i, j, k].

    Grid-size choice belongs here, not in LifecyclePortfolioModel.
    LifecyclePortfolioModel holds economic parameters; Precompute holds
    numerical approximation parameters. Use state_grid_sizes to control
    the trade-off between accuracy and computation time:
      - [5, 5, 5]   = 125 states, fast, coarser approximation
      - [7, 7, 7]   = 343 states, good default for production runs
      - [9, 9, 9]   = 729 states, finer
    Different sizes per dimension are allowed, e.g. [9, 7, 7] if one
    state variable has higher persistence and needs more resolution.
    The consistency check below reports the approximation error so you
    can verify the chosen grid is adequate.

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
      annuity_factors  (N_state,)           A(r_f, b_bar) annuity factor at each state
                                            (used in bequest_utility / bequest_marginal)

    Lookup tables:
      working_income    (n_age, n_z, n_eps)
                                        working_income[t, iz, ie] = after-tax labor income
                                        at age ages[t], persistent state iz, transitory node ie
      pension_after_tax (n_age, n_z)
                                        pension_after_tax[t, iz] = after-tax Social Security
                                        benefit; constant across ages, indexed by career z

    Dimension counters: n_w, n_s, n_z, n_eps, n_ret_quad, n_age, N_state
    -------------------------------------------------------
    """

    def __init__(
        self,
        model,
        disc_config=None,
        verbose=True,
    ):
        # --- Config ---
        if disc_config is None:
            disc_config = DiscretizationConfig()
        self.disc_config = disc_config
        self.model = model
        self.verbose = verbose

        K_ret_per_dim = _normalize_ret_nodes(disc_config.n_ret_nodes_1d, model.n_ret)
        if any(k < 1 for k in K_ret_per_dim):
            raise ValueError("All entries of disc_config.n_ret_nodes_1d must be >= 1")
        self.n_ret_nodes_1d = K_ret_per_dim   # tuple of ints, length n_ret (always normalized)

        # --- Grids ---
        effective_savings_max = disc_config.wealth_max if disc_config.savings_max is None else float(disc_config.savings_max)
        if effective_savings_max <= disc_config.savings_min:
            raise ValueError("savings_max must be strictly greater than savings_min")
        if effective_savings_max > disc_config.wealth_max:
            raise ValueError("savings_max cannot exceed wealth_max; widen wealth_max instead")

        self.wealth_grid = np.expm1(np.linspace(
            np.log1p(disc_config.wealth_min),
            np.log1p(disc_config.wealth_max),
            disc_config.n_wealth,
        ))
        self.savings_max = effective_savings_max
        self.s_grid      = np.expm1(np.linspace(
            np.log1p(disc_config.savings_min),
            np.log1p(self.savings_max),
            disc_config.n_savings,
        ))
        self.ages        = np.arange(model.start_age, model.terminal_age + 1)

        # --- Financial state VAR discretization ---
        state_grid_sizes = list(disc_config.state_grid_sizes)
        if len(state_grid_sizes) != model.n_state:
            raise ValueError("state_grid_sizes length must equal model.n_state")
        self.state_grid_sizes = list(state_grid_sizes)

        grid_info = build_state_grid(
            N_vec=state_grid_sizes,
            mu_intercept=model.Phi_0_state,
            Phi=model.Phi_11,
            Sigma_innov=model.Sigma_ss,
            n_stds=disc_config.state_n_stds,
            mode=disc_config.state_grid_mode,
        )
        self.state_grid_mode = grid_info["mode"]
        self.state_grid_mu_s = grid_info["mu_s"]
        self.state_grid_sigma_z = grid_info["sigma_z"]
        self.state_bracket_shift = grid_info["bracket_shift"]
        self.state_bracket_L_inv = grid_info["bracket_L_inv"]
        self.state_bracket_grids = grid_info["state_bracket_grids"]
        self.state_indices = grid_info["state_indices"]
        self.state_grid = grid_info["state_grid"]
        self.Pi_state = grid_info["Pi_state"]
        self.state_stationary_probs = grid_info["stationary_probs"]
        # state_bracket_grids: interpolation axes; in principal mode these live in transformed coordinates.
        # state_grid: economic slow-state vectors at the flat lattice points.
        self.N_state = self.state_grid.shape[0]  # int — total joint states = prod(state_grid_sizes)


        # --- Conditional return means and bill rate ---
        self.mu_r = self._precompute_conditional_returns()
        # (N_state, N_state, n_ret) float64
        # mu_r[i, j, 0] = E[rtb | s_t=i, s_{t+1}=j]  - log real bill return
        # mu_r[i, j, 1] = E[xr  | s_t=i, s_{t+1}=j]  - log excess stock return
        # mu_r[i, j, 2] = E[xb  | s_t=i, s_{t+1}=j]  - log excess bond return

        self.ret_nodes, self.ret_weights = get_return_quadrature(
            model,
            n_nodes=disc_config.n_ret_nodes_1d,
            lobatto_Z=getattr(disc_config, "ret_lobatto_Z", None),
        )
        # ret_nodes:   (n_ret_quad, n_ret) float64 - residual log-return shocks around mu_r
        # ret_weights: (n_ret_quad,) float64       - tensor-product weights, sum(ret_weights)=1
        # Total joint return nodes = prod(n_ret_nodes_1d).  All-K=1 yields one zero residual node.

        # --- State innovation quadrature ---
        self.v_nodes, self.v_weights = get_state_quadrature(
            model,
            n_nodes=disc_config.n_state_quad_nodes,
            lobatto_Z=getattr(disc_config, "state_lobatto_Z", None),
        )
        # v_nodes:   (n_state_quad, n_state) float64 — innovation nodes in original coords
        # v_weights: (n_state_quad,) float64 — tensor-product weights, sum to 1
        self.n_state_quad = len(self.v_weights)

        # --- Precomputed return formula constants (for on-the-fly mu_r computation) ---
        # Conditional return mean: mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ v_k
        # We store const_r = Phi_0_ret and A_r = Phi_21 so:
        #   mu_r_k = const_r + A_r @ s_i + M @ v_nodes[k_v]
        self.const_r = np.array(model.Phi_0_ret, dtype=float)             # (n_ret,)
        self.A_r = np.array(model.Phi_21, dtype=float)                     # (n_ret, n_state)

        # Precompute M @ v_nodes for each quadrature node (avoids matmul in hot loop)
        self.M_v_nodes = self.v_nodes @ model.M.T     # (n_state_quad, n_ret)
        # Usage in solver: mu_r_k = base_mu_r_i + M_v_nodes[k_v, :]
        # where base_mu_r_i = const_r + A_r @ s_i  (computed once per i_s)

        # Precompute exp of the return-quadrature residual nodes (avoids exp in hot loop)
        # Return columns are now [rtb_resid, xr_resid, xb_resid]
        self.exp_ret_bill  = np.exp(self.ret_nodes[:, 0])  # (n_ret_quad,) — rtb residuals
        self.exp_ret_stock = np.exp(self.ret_nodes[:, 1])  # (n_ret_quad,) — xr residuals
        self.exp_ret_bond  = np.exp(self.ret_nodes[:, 2])  # (n_ret_quad,) — xb residuals

        # No r_bill_grid — bill rate is now uncertain (part of return quadrature).

        # --- Bequest annuity factors (one per financial state) ---
        # A(y_1, spr, b_bar): PV of b_bar annual payments discounted at a
        # linearly interpolated term structure from y_1 to y_20 = y_1 + spr.
        # Read y_1 and spread from the state grid when they are state variables
        # (the original System IV path), or fall back to scalar values when one
        # or both are omitted from the state vector.
        y_1_idx = model.y_1_index_in_state
        spr_idx = model.spr_index_in_state

        if y_1_idx is not None and spr_idx is not None:
            _y_1 = self.state_grid[:, model.y_1_index_in_state]
            _spr = self.state_grid[:, model.spr_index_in_state]
        else:
            if y_1_idx is not None:
                _y_1 = self.state_grid[:, y_1_idx]
            else:
                _y_1 = np.full(self.N_state, model.y_1_scalar_fallback, dtype=float)
            if spr_idx is not None:
                _spr = self.state_grid[:, spr_idx]
            else:
                _spr = np.full(self.N_state, model.spr_scalar_fallback, dtype=float)

        self.annuity_factors = annuity_factor(_y_1, _spr, model.b_bar)
        # (N_state,) float64 - A(y_1, spr, b_bar) for each financial state
        # Used by bequest_utility / bequest_marginal / bequest_marginal_inv in solver.

        # --- Income discretization ---
        self.z_grid, self.Pi_z = discretize_income_ar1_mixture(
            rho=model.rho,
            p=model.pz,
            mu1=model.mu_eta1,
            sigma1=model.sigma_eta1,
            mu2=model.mu_eta2,
            sigma2=model.sigma_eta2,
            N=disc_config.n_z,
            n_stds=disc_config.n_stds,
        )
        # z_grid: (n_z,) float64 - persistent income states (log deviation from mean, mean-zero by construction)
        # Pi_z:   (n_z, n_z) float64 - Pi_z[iz, jz] = P(z_{t+1}=jz | z_t=iz)

        self.eps_nodes, self.eps_weights = get_eps_quadrature_corrected(model, n_nodes=disc_config.n_eps_nodes)
        # eps_nodes:   (n_eps,) float64 - Judd-mixture quadrature nodes for transitory income shock eps
        # eps_weights: (n_eps,) float64 - quadrature weights; sum(eps_weights) = 1,  E[eps] = 0 enforced

        self.eta_nodes, self.eta_weights = get_eta_quadrature_mixture(model, n_nodes=disc_config.n_eta_nodes)
        # eta_nodes:   (n_eta,) float64 - Judd-mixture quadrature nodes for persistent innovation eta
        # eta_weights: (n_eta,) float64 - quadrature weights; sum(eta_weights) = 1,  E[eta] = 0 enforced

        self.dz = self.z_grid[1] - self.z_grid[0]  # uniform grid spacing, used for z-interpolation in solver

        # --- Deterministic age-earnings profile (used by simulation for direct income computation) ---
        self.log_det_profile = (model.b0
                                + model.b1 * self.ages
                                + model.b2 * self.ages**2 / 10.0
                                + model.b3 * self.ages**3 / 100.0)
        # (n_age,) float64 — f(age) = b0 + b1*age + b2*age^2/10 + b3*age^3/100
        # log_det_profile[t] corresponds to ages[t]; used as exp(log_det_profile[t] + z + eps)

        _working_ages = np.arange(model.start_age, model.retire_age)
        _log_det_working = (model.b0
                            + model.b1 * _working_ages
                            + model.b2 * _working_ages**2 / 10.0
                            + model.b3 * _working_ages**3 / 100.0)
        self.avg_det = float(np.mean(np.exp(_log_det_working)))
        # scalar — mean of exp(f(age)) over working ages; used for AIME in pension calculation

        # --- Income lookup tables ---
        self.working_income = self._precompute_working_income()
        # (n_age, n_z, n_eps) float64
        # working_income[t, iz, ie] = after-tax net labor income
        #   at age ages[t], persistent state z_grid[iz], transitory shock eps_nodes[ie]
        # Gross income: Y = exp(f(age) + z_grid[iz] + eps_nodes[ie]);  net = disposable_income_working(Y)

        self.pension_after_tax = self._precompute_pension()
        # (n_age, n_z) float64
        # pension_after_tax[t, iz] = after-tax Social Security pension benefit
        #   given career rank exp(z_grid[iz]); constant across retirement ages (same row repeated)

        # --- Dimension counters ---
        self.n_w   = len(self.wealth_grid)  # int — wealth grid points
        self.n_s   = len(self.s_grid)       # int — savings grid points
        self.n_z   = len(self.z_grid)       # int — persistent income states
        self.n_eps = len(self.eps_nodes)    # int — transitory shock quadrature nodes (= n_eps_nodes total)
        self.n_ret_quad = len(self.ret_weights)  # int — joint return quadrature nodes (= prod(n_ret_nodes_1d))
        self.n_age = len(self.ages)         # int — number of age periods

        # --- Earnings-dependent mortality (Catherine 2025, eq. 35) ---
        self.survival_probs_2d, self._chi_vec, self._mortality_diag = calibrate_earnings_dependent_mortality(
            start_age=model.start_age,
            terminal_age=model.terminal_age,
            z_grid=self.z_grid,
            rho=model.rho,
            pz=model.pz,
            mu_eta1=model.mu_eta1,
            sigma_eta1=model.sigma_eta1,
            mu_eta2=model.mu_eta2,
            sigma_eta2=model.sigma_eta2,
            verbose=self.verbose,
        )
        # survival_probs_2d: (n_age, n_z) float64
        # survival_probs_2d[t, iz] = 1 - min(chi[iz] * m_baseline(age_t), 1)

        # Diagnostics (quadrature moment checks only; old Markov consistency check removed)
        self._validate_state_quadrature()
        if self.verbose:
            self._print_summary()

    def _precompute_conditional_returns(self):
        """
        mu_r[i, j, :] = E[r_{t+1} | state_t=i, state_{t+1}=j]

        Derived from the conditional formula:
          mu_r[i,j] = Phi_0_ret + Phi_21 @ s_i + M @ (s_j - Phi_0_state - Phi_11 @ s_i)

        Rearranged into a sum of three independent terms:
          mu_r[i,j] = const + A @ s_i + M @ s_j
          where  const = Phi_0_ret - M @ Phi_0_state       (n_ret,)
                 A     = Phi_21    - M @ Phi_11             (n_ret, n_state)

        This vectorized form avoids O(N_state^2) Python loops.
        """
        const  = self.model.Phi_0_ret - self.model.M @ self.model.Phi_0_state  # (n_ret,)
        A      = self.model.Phi_21    - self.model.M @ self.model.Phi_11        # (n_ret, n_state)

        term_i = self.state_grid @ A.T             # (N_state, n_ret)
        term_j = self.state_grid @ self.model.M.T  # (N_state, n_ret)

        return const[None, None, :] + term_i[:, None, :] + term_j[None, :, :]

    def _validate_state_quadrature(self):
        """Verify state quadrature reproduces conditional return moments.

        For each source state i, check:
          sum_k w_k * mu_r_k == Phi_0_ret + Phi_21 @ s_i  (unconditional return mean)
          sum_k w_k * v_k @ v_k.T == Sigma_ss             (innovation covariance)
        """
        model = self.model
        max_err_mean = 0.0
        for i in range(self.N_state):
            s_i = self.state_grid[i]
            base_mu_r = self.const_r + self.A_r @ s_i
            # Weighted average of mu_r_k = base_mu_r + M @ v_k
            avg_mu_r = base_mu_r + self.v_weights @ self.M_v_nodes
            target = model.Phi_0_ret + model.Phi_21 @ s_i
            err = np.max(np.abs(avg_mu_r - target))
            max_err_mean = max(max_err_mean, err)

        if self.verbose:
            print(f"  State quadrature return-mean consistency: max err = {max_err_mean:.2e}")

        assert max_err_mean < 1e-10, (
            f"State quadrature return-mean error {max_err_mean:.2e} too large"
        )

    def _precompute_working_income(self):
        """
        After-tax labor income table: shape (n_age, n_z, n_eps).

        Gross income at each (age, z, eps) grid point:
            Y_gross = exp(f(age) + z + eps)
        where f(age) = b0 + b1*age + b2*age^2/10 + b3*age^3/100
        is the deterministic age-earnings profile (Guvenen et al. 2022).

        After-tax income applies payroll tax (10.6%, capped at 2.5) plus
        progressive income tax (7 TCJA brackets) via disposable_income_working.

        Vectorized: broadcasts over all (age, z, eps) simultaneously.
        """
        z    = self.z_grid[None, :, None]        # (1, n_z, 1)
        eps  = self.eps_nodes[None, None, :]     # (1, 1, n_eps)
        det  = self.log_det_profile[:, None, None]  # (n_age, 1, 1)

        y_gross = np.exp(det + z + eps)
        return disposable_income_working(y_gross)

    def _precompute_pension(self):
        """
        After-tax pension table: shape (n_age, n_z).

        Uses self.avg_det = mean(exp(f(age))) over working ages, then
        passes it to compute_pension_after_tax so AIME is correctly
        scaled by the deterministic lifecycle profile.

        Catherine (2025) eq. (20):
            AIYE_it = L_bar_t * sum min{L_tilde_is, 2.5}
        Approximation:
            AIME(z) ~ min(exp(z) * avg_det, 2.5)
        """
        base_pension = compute_pension_after_tax(self.z_grid, self.avg_det)
        n_age = len(self.ages)
        n_z = len(self.z_grid)
        return np.broadcast_to(base_pension, (n_age, n_z)).copy()

    def regenerate_savings_grid(self, n_s_points):
        """Utility for sensitivity runs in Part 2."""
        return np.expm1(np.linspace(
            np.log1p(self.disc_config.savings_min),
            np.log1p(self.savings_max),
            int(n_s_points),
        ))

    def _print_summary(self):
        print("=" * 64)
        print("PRECOMPUTE SUMMARY")
        print("=" * 64)
        sizes_str = " x ".join(str(n) for n in self.state_grid_sizes)
        print(f"Ages         : {self.model.start_age} to {self.model.terminal_age}"
              f"  ({self.n_age} periods,"
              f" retire at {self.model.retire_age})")
        print(f"State grid   : {sizes_str} = {self.N_state} joint states")
        ns = self.disc_config.state_n_stds
        ns_arr = np.atleast_1d(np.asarray(ns, dtype=float))
        if ns_arr.size == 1:
            ns_str = f"{float(ns_arr[0]):.2f}"
        else:
            ns_str = "(" + ", ".join(f"{x:.2f}" for x in ns_arr) + ")"
        print(f"  mode       : {self.state_grid_mode}  |  half-width = {ns_str}")
        print(f"  state vars : {list(self.model.state_names)}")
        print(f"  return vars: {list(self.model.ret_names)}")
        print(f"Income grid  : {self.n_z} persistent states"
              f"  x  {self.n_eps} transitory nodes")
        K_str = "x".join(str(k) for k in self.n_ret_nodes_1d)
        ret_Z_axes = _format_lobatto_axes(getattr(self.disc_config, "ret_lobatto_Z", None))
        print(f"Return quad  : ({K_str}) nodes/dim"
              f"  ->  {self.n_ret_quad} joint nodes" + ret_Z_axes)
        K_state_disp = self.disc_config.n_state_quad_nodes
        if hasattr(K_state_disp, "__len__"):
            K_state_str = "(" + "x".join(str(k) for k in K_state_disp) + ")"
        else:
            K_state_str = str(K_state_disp)
        state_Z_axes = _format_lobatto_axes(getattr(self.disc_config, "state_lobatto_Z", None))
        print(f"State quad   : {K_state_str} nodes/dim"
              f"  ->  {self.n_state_quad} joint nodes" + state_Z_axes)
        print(f"Wealth grid  : {self.n_w} points  [{self.wealth_grid[0]:.3e}, {self.wealth_grid[-1]:.3e}]")
        print(f"Savings grid : {self.n_s} points  [{self.s_grid[0]:.3e}, {self.s_grid[-1]:.3e}]")
        print(f"mu_r         : {self.mu_r.shape}"
              f"  ({self.N_state * self.N_state * self.model.n_ret:,} values)")
        print(f"ret_nodes    : {self.ret_nodes.shape}")
        if self.model.y_1_index_in_state is not None:
            print(f"y_1 idx in state : {self.model.y_1_index_in_state}"
                  f"  ({self.model.state_names[self.model.y_1_index_in_state]})")
        else:
            print(f"y_1 (scalar)     : {self.model.y_1_scalar_fallback:.4%}")
        if self.model.spr_index_in_state is not None:
            print(f"spr idx in state : {self.model.spr_index_in_state}"
                  f"  ({self.model.state_names[self.model.spr_index_in_state]})")
        else:
            print(f"spr (scalar)     : {self.model.spr_scalar_fallback:.4%}")
        print(f"annuity_factors   : {self.annuity_factors.shape}  range=[{self.annuity_factors.min():.2f}, {self.annuity_factors.max():.2f}]")
        print(f"working_income    : {self.working_income.shape}  (n_age x n_z x n_eps)")
        print(f"pension_after_tax : {self.pension_after_tax.shape}  (n_age x n_z)")
        print("=" * 64)


# =============================================================================
# MODEL FACTORY (CONFIG DRIVEN)
# =============================================================================

def build_model(base_config, var_config, verbose=True):
    """Build LifecyclePortfolioModel from primitive configs."""
    u, u_prime, u_prime_inv = create_utility_functions(base_config["gamma"])

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
        u=u,
        u_prime=u_prime,
        u_prime_inv=u_prime_inv,
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
        constrained=bool(base_config.get("constrained", True)),
    )

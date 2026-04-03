"""
precompute.py — Precompute grids, transitions, and lookup tables.

Contains:
  - Precompute class — all arrays consumed by the solver
  - build_model() — factory assembling LifecyclePortfolioModel from configs

Dependencies: model, var, discretization, earnings_dependent_mortality
"""

import numpy as np

from model import (
    LifecyclePortfolioModel, DiscretizationConfig,
    create_utility_functions, annuity_factor,
    disposable_income_working, compute_pension_after_tax,
)
from var import partition_var
from discretization import (
    rouwenhorst_multivariate, discretize_income_ar1_mixture,
    get_eps_quadrature_corrected, get_return_quadrature,
)
from earnings_dependent_mortality import calibrate_earnings_dependent_mortality


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
      - [9, 9, 9]   = 729 states, finer, check memory (Pi_state: N_s^2 floats)
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
                                        mu_r[i,j,0] = E[xr    | s_t=i, s_{t+1}=j]  log excess stock return
                                        mu_r[i,j,1] = E[xtips | s_t=i, s_{t+1}=j]  log excess TIPS return
      ret_nodes    (n_ret_quad, n_ret)  residual log-return shocks drawn from N(0, Sigma_r_cond)
      ret_weights  (n_ret_quad,)        quadrature weights; sum=1
      r_bill_grid  (N_state,)           log real bill rate at each slow state;
                                        R_bill = exp(r_bill_grid[i_s])  (known at decision time)

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
    Backward-compat aliases: slow_grid, Pi_slow, slow_grids, slow_state_indices, N_s
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

        if disc_config.n_ret_nodes_1d < 1:
            raise ValueError("disc_config.n_ret_nodes_1d must be >= 1")

        # --- Grids ---
        self.wealth_grid = np.geomspace(disc_config.wealth_min, disc_config.wealth_max, disc_config.n_wealth)
        self.s_grid      = np.geomspace(disc_config.savings_min, disc_config.wealth_max, disc_config.n_savings)
        self.ages        = np.arange(model.start_age, model.terminal_age + 1)

        # --- Financial state VAR discretization ---
        state_grid_sizes = list(disc_config.state_grid_sizes)
        if len(state_grid_sizes) != model.n_state:
            raise ValueError("state_grid_sizes length must equal model.n_state")
        self.state_grid_sizes = list(state_grid_sizes)

        Sigma_state_chol = np.linalg.cholesky(model.Sigma_ss)
        self.state_grids, self.Pi_state, self.state_indices = rouwenhorst_multivariate(
            N_vec=state_grid_sizes,
            mu=model.Phi_0_state,
            Phi=model.Phi_11,
            Sigma=Sigma_state_chol,
            method="independent",
        )
        # state_grids:   list[n_state] of 1-D marginal grids, state_grids[d] has shape (state_grid_sizes[d],)
        # Pi_state:      (N_state, N_state) float64 - joint transition matrix; Pi_state[i,j] = P(s_{t+1}=j|s_t=i)
        # state_indices: (N_state, n_state) int64  - multi-index into marginal grids; row i maps to state_grids

        self.state_grid = self._build_state_grid(self.state_grids, self.state_indices)
        # (N_state, n_state) float64 - flat Cartesian grid of slow-state vectors
        # row i = [rtb, y_nom, dp] values at joint state i

        self.N_state = self.state_grid.shape[0]  # int — total joint states = prod(state_grid_sizes)

        # Backward-compatibility aliases (used by Part 2 solver code)
        self.slow_grids         = self.state_grids    # alias for state_grids
        self.Pi_slow            = self.Pi_state        # (N_state, N_state) alias for Pi_state
        self.slow_state_indices = self.state_indices   # (N_state, n_state) alias for state_indices
        self.slow_grid          = self.state_grid      # (N_state, n_state) alias for state_grid
        self.N_s                = self.N_state         # int alias for N_state

        # --- Conditional return means and bill rate ---
        self.mu_r = self._precompute_conditional_returns()
        # (N_state, N_state, n_ret) float64
        # mu_r[i, j, 0] = E[xr    | s_t=i, s_{t+1}=j]  - log excess stock return, conditional on transition i-j
        # mu_r[i, j, 1] = E[xtips | s_t=i, s_{t+1}=j]  - log excess TIPS return,  conditional on transition i-j
        # Use exp(mu_r[i,j,k]) to get gross excess return multiplier.

        self.ret_nodes, self.ret_weights = get_return_quadrature(
            model, n_nodes=disc_config.n_ret_nodes_1d
        )
        # ret_nodes:   (n_ret_quad, n_ret) float64 - residual log-return shocks around mu_r
        # ret_weights: (n_ret_quad,) float64       - tensor-product weights, sum(ret_weights)=1
        # Total joint return nodes = n_ret_nodes_1d ** model.n_ret.  K=1 yields one zero residual node.

        self.r_bill_grid = self.state_grid[:, model.bill_rate_index_in_state]
        # (N_state,) float64 - log real bill rate at each slow state
        # R_bill = exp(r_bill_grid[i_s]); bill rate is KNOWN at decision time (no uncertainty)

        # --- Bequest annuity factors (one per financial state) ---
        # A(y_nom, b_bar): PV of b_bar annual payments of 1 discounted at the
        # 10-year nominal bond yield.  Using y_nom is coherent because the
        # bequest horizon b_bar equals the bond maturity: the nominal bond is
        # the natural pricing instrument for the heir's consumption stream.
        # y_nom is stored in quarterly decimal (SVENY10/400); multiply by 4
        # to recover the annual yield used for discounting.
        _y_ann = self.state_grid[:, model.annuity_yield_index_in_state] * 4.0   # quarterly -> annual yield
        self.annuity_factors = annuity_factor(_y_ann, model.b_bar)
        # (N_state,) float64 - A(y_nom, b_bar) for each financial state
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
        )
        # z_grid: (n_z,) float64 - persistent income states (log deviation from mean, mean-zero by construction)
        # Pi_z:   (n_z, n_z) float64 - Pi_z[iz, jz] = P(z_{t+1}=jz | z_t=iz)

        self.eps_nodes, self.eps_weights = get_eps_quadrature_corrected(model, n_nodes=disc_config.n_eps_nodes)
        # eps_nodes:   (n_eps,) float64 - Gauss-Hermite quadrature nodes for transitory income shock eps
        # eps_weights: (n_eps,) float64 - quadrature weights; sum(eps_weights) = 1,  E[eps] = 0 enforced

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
        self.n_eps = len(self.eps_nodes)    # int — transitory shock quadrature nodes (= 2 * n_eps_nodes)
        self.n_ret_quad = len(self.ret_weights)  # int — joint return quadrature nodes (= n_ret_nodes_1d ** n_ret)
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

        # Diagnostics
        self._validate_conditional_returns(
            tol_warn=disc_config.consistency_tol_warn,
            tol_error=disc_config.consistency_tol_error,
        )
        if self.verbose:
            self._print_summary()

    @staticmethod
    def _build_state_grid(state_grids, state_indices):
        n_total, n_dim = state_indices.shape
        out = np.empty((n_total, n_dim), dtype=float)
        for i in range(n_total):
            for d in range(n_dim):
                out[i, d] = state_grids[d][state_indices[i, d]]
        return out

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

    def _validate_conditional_returns(self, tol_warn=2e-2, tol_error=1e-1):
        """
        Verify sum_j Pi[i,j] * mu_r[i,j] == Phi_0_ret + Phi_21 @ s_i for all i.

        Theoretical error = M @ Phi_11_off @ s_i, where
        Phi_11_off = Phi_11 - diag(diag(Phi_11)).

        Source: independence Rouwenhorst uses only diagonal(Phi_11) per marginal,
        so it cannot match E[s_{t+1}|s_i] when Phi_11 has off-diagonal elements.
        The error grows linearly with the off-diagonal cross-persistence and with
        how far each state is from its mean (worst at grid extremes).
        Finer grids do NOT reduce this error; it is a structural approximation.
        """
        N = self.N_state
        n_ret = self.model.n_ret

        errors = np.empty((N, n_ret))
        for i in range(N):
            target = self.model.Phi_0_ret + self.model.Phi_21 @ self.state_grid[i]
            avg    = self.Pi_state[i, :] @ self.mu_r[i, :, :]
            errors[i, :] = np.abs(avg - target)

        max_err_per_ret  = errors.max(axis=0)
        mean_err_per_ret = errors.mean(axis=0)
        overall_max      = errors.max()
        worst_i          = errors.max(axis=1).argmax()

        Phi_11_off = self.model.Phi_11 - np.diag(np.diag(self.model.Phi_11))

        if self.verbose:
            print("=" * 64)
            print("CONDITIONAL RETURN CONSISTENCY CHECK")
            print("=" * 64)
            print("Error source: independence Rouwenhorst uses only diag(Phi_11).")
            print("Theoretical error at state i = M @ Phi_11_off @ s_i, where")
            print("Phi_11_off = Phi_11 - diag(diag(Phi_11)).")
            print("Error is worst at grid extremes; finer grids do not fix it.")
            print()
            print(f"  ||Phi_11_off||_F = {np.linalg.norm(Phi_11_off):.4f}"
                  "  (cross-persistence magnitude)")
            print(f"  ||M||_F          = {np.linalg.norm(self.model.M):.4f}"
                  "  (return-state conditioning strength)")
            print(f"  Product ||M @ Phi_11_off||_F = "
                  f"{np.linalg.norm(self.model.M @ Phi_11_off):.4f}"
                  "  (amplification factor)")
            print()
            print(f"  {'Variable':<12}  {'max error':>10}  {'mean error':>10}")
            print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
            for k, name in enumerate(self.model.ret_names):
                flag = "  << WARN" if max_err_per_ret[k] > tol_warn else ""
                print(f"  {name:<12}  {max_err_per_ret[k]:>10.3e}"
                      f"  {mean_err_per_ret[k]:>10.3e}{flag}")
            print()
            worst_vals = "  ".join(
                f"{name}={self.state_grid[worst_i, d]:.4f}"
                for d, name in enumerate(self.model.state_names)
            )
            print(f"  Worst state: index={worst_i}  ({worst_vals})")
            print(f"  Overall max error: {overall_max:.3e}")
            print()
            if overall_max > tol_error:
                print(f"  STATUS: FAIL  (max {overall_max:.3e} > hard limit {tol_error:.3e})")
                print(f"  Current grid: {self.state_grid_sizes}  ->  {self.N_state} states")
                print("  Off-diagonal Phi_11 is too large for independence Rouwenhorst.")
                print("  See ||M @ Phi_11_off|| above for the amplified error magnitude.")
            elif overall_max > tol_warn:
                print(f"  STATUS: WARN  (max {overall_max:.3e} > soft limit {tol_warn:.3e})")
                print(f"  Current grid: {self.state_grid_sizes}  ->  {self.N_state} states")
            else:
                print(f"  STATUS: PASS  (max {overall_max:.3e} < warn limit {tol_warn:.3e})")
            print("=" * 64)

        if overall_max > tol_error:
            raise RuntimeError(
                f"Conditional return consistency error {overall_max:.3e} exceeds "
                f"hard limit {tol_error:.3e}. See printed diagnostics above."
            )

    def _precompute_working_income(self):
        """After-tax labor income table: [age, z_state, eps_node]."""
        n_age = len(self.ages)
        n_z = len(self.z_grid)
        n_eps = len(self.eps_nodes)

        out = np.empty((n_age, n_z, n_eps), dtype=float)
        for t_idx, age in enumerate(self.ages):
            det = (self.model.b0 + self.model.b1 * age
                   + self.model.b2 * (age ** 2) / 10.0
                   + self.model.b3 * (age ** 3) / 100.0)

            for iz in range(n_z):
                p_val = self.z_grid[iz]
                for ie in range(n_eps):
                    y_gross = np.exp(det + p_val + self.eps_nodes[ie])
                    out[t_idx, iz, ie] = disposable_income_working(y_gross)

        return out

    def _precompute_pension(self):
        """After-tax pension table: [age, z_state]."""
        base_pension = compute_pension_after_tax(self.z_grid)
        n_age = len(self.ages)
        n_z = len(self.z_grid)
        out = np.empty((n_age, n_z), dtype=float)
        for t_idx in range(n_age):
            out[t_idx, :] = base_pension
        return out

    def regenerate_savings_grid(self, n_s_points):
        """Utility for sensitivity runs in Part 2."""
        return np.geomspace(self.disc_config.savings_min, self.wealth_grid[-1], int(n_s_points))

    def _print_summary(self):
        print("=" * 64)
        print("PRECOMPUTE SUMMARY")
        print("=" * 64)
        sizes_str = " x ".join(str(n) for n in self.state_grid_sizes)
        print(f"Ages         : {self.model.start_age} to {self.model.terminal_age}"
              f"  ({self.n_age} periods,"
              f" retire at {self.model.retire_age})")
        print(f"State grid   : {sizes_str} = {self.N_state} joint states")
        print(f"  state vars : {list(self.model.state_names)}")
        print(f"  return vars: {list(self.model.ret_names)}")
        print(f"Income grid  : {self.n_z} persistent states"
              f"  x  {self.n_eps} transitory nodes")
        print(f"Return quad  : {self.disc_config.n_ret_nodes_1d} nodes/dim"
              f"  ->  {self.n_ret_quad} joint nodes")
        print(f"mu_r         : {self.mu_r.shape}"
              f"  ({self.N_state * self.N_state * self.model.n_ret:,} values)")
        print(f"ret_nodes    : {self.ret_nodes.shape}")
        print(f"Bill-rate idx: {self.model.bill_rate_index_in_state}"
              f"  ({self.model.state_names[self.model.bill_rate_index_in_state]})")
        print(f"r_bill range : [{self.r_bill_grid.min():.4f},"
              f" {self.r_bill_grid.max():.4f}]")
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

    bill_rate_index_in_state = int(var_config["bill_rate_index_in_state"])
    if bill_rate_index_in_state < 0 or bill_rate_index_in_state >= parts["n_state"]:
        raise ValueError("bill_rate_index_in_state is out of bounds for state vector")

    annuity_yield_index_in_state = int(var_config["annuity_yield_index_in_state"])
    if annuity_yield_index_in_state < 0 or annuity_yield_index_in_state >= parts["n_state"]:
        raise ValueError("annuity_yield_index_in_state is out of bounds for state vector")

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
        bill_rate_index_in_state=bill_rate_index_in_state,
        annuity_yield_index_in_state=annuity_yield_index_in_state,
        constrained=bool(base_config.get("constrained", True)),
    )



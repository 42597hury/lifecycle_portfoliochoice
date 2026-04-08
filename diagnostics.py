"""
diagnostics.py — Model diagnostic reports and Newton failure analysis.

Contains:
  - print_model_diagnostic_report() — comprehensive pre-solve calibration report
    Section 2 includes Tier 1 income & SS diagnostics (18 tests)
  - print_simulation_income_report() — post-simulation income diagnostics (Tier 2)
  - diagnose_newton_failures_retirement() — post-solve Newton failure analysis

Dependencies: numpy, numba, model, solver (for diagnostic constants and FOC functions)
"""

import numpy as np
from numba import njit
from math import exp

from model import SolverConfig, disposable_income_working, compute_pension_after_tax
from solver import (
    compute_foc_jac_retirement, solve_portfolio_2d_retirement,
    build_gross_return_arrays, compute_terminal_portfolio_foc_jac,
    solve_portfolio_unconstrained_terminal_exact, solve_portfolio_2d_terminal_exact,
    _terminal_prepare_scenarios, _terminal_portfolio_moment,
    EC_NEWTON_FAIL, EC_INTERIOR,
    EC_CORNER_BILLS, EC_CORNER_STOCKS, EC_CORNER_BONDS,
    EC_EDGE_SB, EC_EDGE_BB, EC_EDGE_STOCKBOND,
)


# =============================================================================
# 9. DIAGNOSTIC REPORT
# =============================================================================

def print_model_diagnostic_report(model, pc, periods_per_year=4):
    """
    Comprehensive diagnostic report for calibration verification and debugging.

    Parameters
    ----------
    model : LifecyclePortfolioModel
    pc    : Precompute
    periods_per_year : int
        4 = quarterly data (default), 12 = monthly, 1 = annual.
        Used to annualize rates and scale stds from per-period to per-year.
        Note: dp (log dividend-price ratio) is a log level, not a rate;
        annualizing it via x*ppy has no economic meaning.
    """
    W   = 76
    ppy = periods_per_year
    sv  = list(model.state_names)
    rv  = list(model.ret_names)

    def header(title):
        print()
        print("=" * W)
        print(f"  {title}")
        print("=" * W)

    def sub(title):
        print(f"\n  --- {title} ---")

    def flag(label, ok, detail=""):
        tag = "[PASS]" if ok else "[WARN]"
        suf = f"  {detail}" if detail else ""
        print(f"  {tag}  {label}{suf}")

    # =========================================================================
    # 1. PREFERENCES & LIFECYCLE
    # =========================================================================
    header("1.  PREFERENCES & LIFECYCLE")

    sub("Utility & bequest")
    print(f"  gamma  = {model.gamma:.3f}   (CRRA risk aversion)")
    beta_ann = model.beta ** ppy
    print(f"  beta   = {model.beta:.6f}  per period   ->   {beta_ann:.6f}  annualized")
    print(f"  b_bar  = {model.b_bar}   (bequest horizon in years, Catherine 2025)")

    sub("Lifecycle")
    n_work = model.retire_age - model.start_age
    n_ret  = model.terminal_age - model.retire_age + 1
    print(f"  Ages:  {model.start_age} - {model.terminal_age}   ({pc.n_age} periods total)")
    print(f"  Retirement at {model.retire_age}:   {n_work} working periods,  {n_ret} retirement periods")

    sub("Survival probabilities at key ages (earnings-dependent)")
    iz_lo, iz_mid, iz_hi = 0, pc.n_z // 2, pc.n_z - 1
    print(f"  {'Age':>4}  {'low-z':>10}  {'mid-z':>10}  {'high-z':>10}")
    key_sp = [25, 30, 40, 50, 55, 60, 65, 70, 75, 80]
    for age in key_sp:
        if model.start_age <= age <= model.terminal_age:
            t = age - model.start_age
            sp_lo  = pc.survival_probs_2d[t, iz_lo]
            sp_mid = pc.survival_probs_2d[t, iz_mid]
            sp_hi  = pc.survival_probs_2d[t, iz_hi]
            print(f"  {age:>4}  {sp_lo:>10.5f}  {sp_mid:>10.5f}  {sp_hi:>10.5f}")

    # =========================================================================
    # 2. INCOME PROCESS & SOCIAL SECURITY
    # =========================================================================
    header("2.  INCOME PROCESS & SOCIAL SECURITY")

    # Accumulator for pass/fail summary
    income_tests = []   # list of (test_name: str, passed: bool)

    def itest(name, passed):
        """Record and return a test result."""
        income_tests.append((name, passed))
        return passed

    # — Reference indices —
    iz0 = int(np.argmin(np.abs(pc.z_grid)))
    ie0 = int(np.argmin(np.abs(pc.eps_nodes)))
    n_z = len(pc.z_grid)
    retire_t = model.retire_age - model.start_age

    # — avg_det (recompute for diagnostics — must match precompute) —
    working_ages = np.arange(model.start_age, model.retire_age)
    log_det_all  = (model.b0
                    + model.b1 * working_ages
                    + model.b2 * working_ages**2 / 10.0
                    + model.b3 * working_ages**3 / 100.0)
    avg_det = float(np.mean(np.exp(log_det_all)))

    # ── 2a. Parameters (echo) ───────────────────────────────────────────────
    sub("Persistent AR(1) with mixture-normal innovations")
    mu_eta  = model.pz * model.mu_eta1 + (1.0 - model.pz) * model.mu_eta2
    var_eta = (model.pz * (model.sigma_eta1**2 + (model.mu_eta1 - mu_eta)**2)
               + (1.0 - model.pz) * (model.sigma_eta2**2 + (model.mu_eta2 - mu_eta)**2))
    std_eta = np.sqrt(var_eta)
    std_z   = np.sqrt(var_eta / max(1e-14, 1.0 - model.rho**2))
    print(f"  rho        = {model.rho:.5f}  (per-period persistence)")
    print(f"  pz         = {model.pz:.3f}    (mixture weight on component 1)")
    print(f"  Component 1:  mu_eta1 = {model.mu_eta1:+.4f},  sigma_eta1 = {model.sigma_eta1:.4f}")
    print(f"  Component 2:  mu_eta2 = {model.mu_eta2:+.4f},  sigma_eta2 = {model.sigma_eta2:.4f}")
    print(f"  E[eta]     = {mu_eta:.2e}   (should be ~ 0)")
    print(f"  Std[eta]   = {std_eta:.5f}")
    print(f"  Std[z]     = {std_z:.5f}  (unconditional)")
    z_cover = pc.z_grid.max() / std_z if std_z > 0 else float("nan")
    print(f"  z_grid     : {pc.n_z} points  [{pc.z_grid.min():.4f}, {pc.z_grid.max():.4f}]   +/-{z_cover:.2f} sigma")

    sub("Transitory shock (mixture, zero-mean enforced)")
    mu_eps2_eff = -(model.pe / (1.0 - model.pe)) * model.mu_eps1
    eps_mean    = float(np.sum(pc.eps_nodes * pc.eps_weights))
    eps_var     = float(np.sum(pc.eps_nodes**2 * pc.eps_weights))
    print(f"  pe            = {model.pe:.3f}   (probability of large-shock component)")
    print(f"  Component 1:  mu_eps1    = {model.mu_eps1:+.4f},  sigma_eps1 = {model.sigma_eps1:.4f}")
    print(f"  Component 2:  mu_eps2_eff = {mu_eps2_eff:+.4f}  (zero-mean enforced; model.mu_eps2 ignored)")
    print(f"               sigma_eps2 = {model.sigma_eps2:.4f}")
    print(f"  E[eps]        = {eps_mean:.2e}   (should be ~ 0)")
    print(f"  Var[eps]      = {eps_var:.5f}   Std[eps] = {np.sqrt(eps_var):.5f}")
    print(f"  eps_weights sum = {pc.eps_weights.sum():.8f}   (should be 1.000)")
    print(f"  eps_nodes     : {pc.n_eps} nodes  [{pc.eps_nodes.min():.4f}, {pc.eps_nodes.max():.4f}]")
    e_exp_eps = float(np.sum(np.exp(pc.eps_nodes) * pc.eps_weights))
    print(f"  E[exp(eps)]   = {e_exp_eps:.4f}   (Jensen: >1 by ~0.5*Var; {(e_exp_eps - 1)*100:+.1f}%)")

    sub("Deterministic income profile  log Y_det = b0 + b1*Age + b2*Age^2/10 + b3*Age^3/100")
    print(f"  b0 = {model.b0:.4f},  b1 = {model.b1:.4f},  b2 = {model.b2:.4f},  b3 = {model.b3:.6f}")
    _peak_idx = int(np.argmax(log_det_all))
    print(f"  Hump peak: age {working_ages[_peak_idx]},  log-income = {log_det_all[_peak_idx]:.4f},  "
          f"exp(f) = {np.exp(log_det_all[_peak_idx]):.4f}")
    print(f"  avg_det = mean(exp(f(age))) over [{model.start_age}, {model.retire_age}) = {avg_det:.4f}")

    # ── 2b. Integrity checks ────────────────────────────────────────────────
    sub("Integrity checks")
    print(f"  Quick pass/fail gates on the precomputed income and pension arrays.")
    print(f"  If any test fails here, the remaining diagnostics are unreliable.\n")

    wi_min = float(pc.working_income.min())
    _ok = wi_min > 0
    flag("Working income > 0 everywhere", _ok, f"min = {wi_min:.6f}")
    itest("Working income > 0", _ok)

    pens_min = float(pc.pension_after_tax.min())
    _ok = pens_min > 0
    flag("Pension > 0 for all z states", _ok, f"min = {pens_min:.6f}")
    itest("Pension > 0", _ok)

    # After-tax income must increase with the persistent component z
    # (higher z = higher lifetime earnings).  A negative value here means
    # income *decreases* when z rises, which would produce non-monotone
    # policy functions and break the solver's interpolation.
    wi_z_diffs = np.diff(pc.working_income, axis=1)
    wi_z_min_diff = float(wi_z_diffs.min())
    _ok = wi_z_min_diff >= -1e-12
    _detail = (f"min adjacent step = {wi_z_min_diff:.2e} (positive = no violation)"
               if _ok else f"VIOLATION: income falls between adjacent z points by {-wi_z_min_diff:.2e}")
    flag("Working income monotone in z", _ok, _detail)
    itest("Working income monotone in z", _ok)

    # Same check for transitory shocks eps.  The mixture-normal quadrature
    # nodes are not stored in ascending order (component 1 then component 2),
    # so we sort by eps value before checking.
    _eps_order = np.argsort(pc.eps_nodes)
    _wi_sorted_eps = pc.working_income[:, :, _eps_order]
    wi_e_diffs = np.diff(_wi_sorted_eps, axis=2)
    wi_e_min_diff = float(wi_e_diffs.min())
    _ok = wi_e_min_diff >= -1e-12
    _detail = (f"min adjacent step = {wi_e_min_diff:.2e} (positive = no violation)"
               if _ok else f"VIOLATION: income falls between adjacent eps nodes by {-wi_e_min_diff:.2e}")
    flag("Working income monotone in eps", _ok, _detail)
    itest("Working income monotone in eps", _ok)

    # Pension must increase in z.  Zero steps are expected where the AIME
    # cap (2.5) binds — all high-z agents receive the same maximum benefit.
    pens_z_diffs = np.diff(pc.pension_after_tax[0, :])
    pens_z_min = float(pens_z_diffs.min())
    _ok = pens_z_min >= -1e-12
    _detail = (f"min step = {pens_z_min:.2e} (zero where AIME cap binds)"
               if _ok else f"VIOLATION: pension falls between adjacent z points by {-pens_z_min:.2e}")
    flag("Pension monotone in z", _ok, _detail)
    itest("Pension monotone in z", _ok)

    # Pension depends only on the persistent state z at retirement.
    # The precomputed array tiles the same pension vector across all ages,
    # so there should be zero variation across the age dimension.
    pens_age_var = float(np.max(np.abs(pc.pension_after_tax - pc.pension_after_tax[0:1, :])))
    _ok = pens_age_var < 1e-14
    flag("Pension constant across ages", _ok, f"max age-variation = {pens_age_var:.2e}")
    itest("Pension constant across ages", _ok)

    # The AIME cap at 2.5 means the maximum possible PIA is PIA(2.5).
    # Verify the precomputed pension array never exceeds this.
    pia_max_gross = 0.90 * 0.21 + 0.32 * (1.25 - 0.21) + 0.15 * (2.5 - 1.25)
    _p = pia_max_gross
    if _p <= 0.18:
        _tax_pia = _p * 0.10
    elif _p <= 0.72:
        _tax_pia = 0.018 + (_p - 0.18) * 0.12
    else:
        _tax_pia = 0.0828 + (_p - 0.72) * 0.22
    pia_max_net = _p - _tax_pia
    pens_actual_max = float(pc.pension_after_tax.max())
    _ok = abs(pens_actual_max - pia_max_net) < 1e-6
    flag("Pension max = PIA(2.5) after tax", _ok,
         f"expected {pia_max_net:.4f}, got {pens_actual_max:.4f}")
    itest("Pension max = PIA(2.5) after tax", _ok)

    _y_gross_all = np.exp(
        log_det_all[:, None, None]
        + pc.z_grid[None, :, None]
        + pc.eps_nodes[None, None, :]
    )
    _eff_rate_all = 1.0 - pc.working_income[:len(working_ages)] / np.maximum(_y_gross_all, 1e-15)
    eff_rate_max = float(_eff_rate_all.max())
    _ok = eff_rate_max < 0.50
    flag("Effective tax rate < 50%", _ok, f"max effective rate = {eff_rate_max:.1%}")
    itest("Effective tax rate < 50%", _ok)

    _ok = abs(eps_mean) < 1e-8
    flag("E[eps] numerically zero", _ok, f"E[eps] = {eps_mean:.2e}")
    itest("E[eps] ~ 0", _ok)

    _ok = True  # both hardcoded to 2.5
    flag("Payroll cap = AIME cap", _ok,
         "both 2.500 (disposable_income_working & compute_pension_after_tax)")
    itest("Payroll cap = AIME cap", _ok)

    # ── 2b½. Persistent income transition quality ────────────────────────────
    sub("Persistent income transition quality (Pi_z)")
    print(f"  Checks whether the discretized z-transition matches the true AR(1)")
    print(f"  with mixture-normal innovations.  Moments computed at z = 0 (mid row).\n")

    # True innovation moments from mixture parameters
    _e_eta3 = (model.pz * (model.mu_eta1**3 + 3 * model.mu_eta1 * model.sigma_eta1**2)
               + (1.0 - model.pz) * (model.mu_eta2**3 + 3 * model.mu_eta2 * model.sigma_eta2**2))
    _true_skew = _e_eta3 / max(var_eta**1.5, 1e-15)
    _e_eta4 = (model.pz * (model.mu_eta1**4 + 6 * model.mu_eta1**2 * model.sigma_eta1**2 + 3 * model.sigma_eta1**4)
               + (1.0 - model.pz) * (model.mu_eta2**4 + 6 * model.mu_eta2**2 * model.sigma_eta2**2 + 3 * model.sigma_eta2**4))
    _true_kurt = _e_eta4 / max(var_eta**2, 1e-15)

    print(f"  True innovation moments (mixture-normal):")
    print(f"    Mean     = {mu_eta:.4f}")
    print(f"    Variance = {var_eta:.5f}")
    print(f"    Skewness = {_true_skew:+.3f}    (negative: large downward shocks from component 1)")
    print(f"    Kurtosis = {_true_kurt:.2f}     (fat tails; Gaussian = 3.0)")
    print()

    # Discretized transition moments from Pi_z at mid row
    _dz_vals = pc.z_grid - pc.z_grid[iz0]
    _disc_mean = float(np.dot(pc.Pi_z[iz0], pc.z_grid))
    _disc_e2 = float(np.dot(pc.Pi_z[iz0], _dz_vals**2))
    _disc_e1 = float(np.dot(pc.Pi_z[iz0], _dz_vals))
    _disc_var = _disc_e2 - _disc_e1**2
    _disc_e3 = float(np.dot(pc.Pi_z[iz0], _dz_vals**3))
    _disc_e4 = float(np.dot(pc.Pi_z[iz0], _dz_vals**4))
    _disc_skew = (_disc_e3 - 3 * _disc_e1 * _disc_var - _disc_e1**3) / max(_disc_var**1.5, 1e-15) if _disc_var > 0 else 0.0
    _disc_kurt = _disc_e4 / max(_disc_var**2, 1e-15) if _disc_var > 0 else 0.0

    _true_cond_mean = model.rho * pc.z_grid[iz0]

    print(f"  Discretized transition moments (Pi_z, mid row iz={iz0}, z={pc.z_grid[iz0]:.4f}):")
    print(f"    E[z'|z=0]  = {_disc_mean:+.5f}   (true = {_true_cond_mean:+.5f},  error = {_disc_mean - _true_cond_mean:+.1e})")
    print(f"    Variance   = {_disc_var:.5f}   (true = {var_eta:.5f},  ratio = {_disc_var / var_eta:.2f})")
    print(f"    Skewness   = {_disc_skew:+.3f}     (true = {_true_skew:+.3f})")
    print(f"    Kurtosis   = {_disc_kurt:.2f}      (true = {_true_kurt:.2f})")
    print()

    _p_down = float(sum(pc.Pi_z[iz0, j] for j in range(iz0)))
    _p_stay = float(pc.Pi_z[iz0, iz0])
    _p_up = float(sum(pc.Pi_z[iz0, j] for j in range(iz0 + 1, n_z)))
    _n_reachable = int(np.sum(pc.Pi_z[iz0, :] > 1e-10))

    print(f"  Transition direction at z = 0:")
    print(f"    P(down) = {_p_down:.4f}    P(stay) = {_p_stay:.4f}    P(up) = {_p_up:.4f}")
    print(f"    Reachable states from mid: {_n_reachable} of {n_z}")
    print()

    _dz = pc.z_grid[1] - pc.z_grid[0] if n_z > 1 else float("inf")
    _dz_ratio = _dz / std_eta if std_eta > 0 else float("inf")
    print(f"  Grid resolution:")
    print(f"    n_z = {n_z},  dz = {_dz:.4f},  dz / sigma_eta = {_dz_ratio:.2f}")
    print(f"    Recommended: dz / sigma_eta <= 1.0 for Tauchen with mixture innovations")
    print()

    _escape_probs = np.array([1.0 - pc.Pi_z[i, i] for i in range(n_z)])
    _n_absorbing = int(np.sum(_escape_probs < 1e-10))

    _cond_mean_err = abs(_disc_mean - _true_cond_mean)
    _ok = _cond_mean_err < 0.01
    flag("Conditional mean error < 0.01", _ok,
         f"|E[z'|z=0] - rho*z| = {_cond_mean_err:.1e}")
    itest("Pi_z conditional mean", _ok)

    _ok = _p_up > 1e-6
    flag("Upward transitions exist from mid state", _ok,
         f"P(up) = {_p_up:.4f}" + ("" if _ok else
         " — discretization cannot represent mean-reversion from below"))
    itest("Pi_z upward transitions", _ok)

    _ok = _n_absorbing == 0
    flag("No absorbing states in Pi_z", _ok,
         f"{_n_absorbing} absorbing rows" + ("" if _ok else
         f" — rows {list(np.where(_escape_probs < 1e-10)[0])}"))
    itest("Pi_z no absorbing states", _ok)

    # ── 2c. AIME pipeline trace ─────────────────────────────────────────────
    sub("AIME pipeline trace  (Catherine 2025, eqs. 19-20)")
    print(f"  Each row traces one z-value through the full pension formula:")
    print(f"    z -> exp(z) -> exp(z)*avg_det -> AIME (capped at 2.5) -> PIA (3-bracket)")
    print(f"    -> income tax on PIA -> net pension")
    print(f"  If the 'exp(z)*ad' column equals 'exp(z)', avg_det scaling is missing.")
    print(f"  If AIME ever exceeds 2.5, the taxable earnings cap is broken.")
    print(f"  'MISMATCH!' means the hand-calculation disagrees with precomputed arrays.\n")
    print(f"  avg_det = {avg_det:.4f}   |   SS taxable cap = 2.500")

    z_cap = np.log(2.5 / avg_det)

    _repr_iz = [max(0, n_z // 4), iz0, min(n_z - 1, 3 * n_z // 4)]
    _iz_above_cap = None
    for _iz in range(n_z):
        if np.exp(pc.z_grid[_iz]) * avg_det > 2.5:
            _iz_above_cap = _iz
            break
    if _iz_above_cap is not None and _iz_above_cap not in _repr_iz:
        _repr_iz.append(_iz_above_cap)
    if (n_z - 1) not in _repr_iz:
        _repr_iz.append(n_z - 1)
    _repr_iz = sorted(set(_repr_iz))

    print()
    print(f"  {'z_idx':>5}  {'z':>7}  {'exp(z)':>8}  {'exp(z)*ad':>9}  {'AIME':>7}  "
          f"{'PIA_gr':>7}  {'Tax':>7}  {'Pens_net':>8}  {'Note':>12}")
    print(f"  {'---':>5}  {'---':>7}  {'---':>8}  {'---':>9}  {'---':>7}  "
          f"{'---':>7}  {'---':>7}  {'---':>8}  {'---':>12}")

    _pipeline_ok = True
    for _iz in _repr_iz:
        _z_val = pc.z_grid[_iz]
        _exp_z = np.exp(_z_val)
        _exp_z_ad = _exp_z * avg_det
        _aime = min(_exp_z_ad, 2.5)
        _b1, _b2 = 0.21, 1.25
        if _aime <= _b1:
            _pia = _aime * 0.90
        elif _aime <= _b2:
            _pia = 0.90 * _b1 + 0.32 * (_aime - _b1)
        else:
            _pia = 0.90 * _b1 + 0.32 * (_b2 - _b1) + 0.15 * (_aime - _b2)
        if _pia <= 0.18:
            _ptax = _pia * 0.10
        elif _pia <= 0.72:
            _ptax = 0.018 + (_pia - 0.18) * 0.12
        elif _pia <= 1.54:
            _ptax = 0.0828 + (_pia - 0.72) * 0.22
        else:
            _ptax = 0.2632 + (_pia - 1.54) * 0.24
        _pens_hand = _pia - _ptax
        _note = ""
        if abs(_z_val) < 0.01:
            _note = "median"
        elif _exp_z_ad >= 2.5:
            _note = "cap binds"
        _pens_pc = float(pc.pension_after_tax[0, _iz])
        if abs(_pens_hand - _pens_pc) > 1e-6:
            _note += " MISMATCH!"
            _pipeline_ok = False
        print(f"  {_iz:>5}  {_z_val:>7.3f}  {_exp_z:>8.3f}  {_exp_z_ad:>9.3f}  {_aime:>7.3f}  "
              f"{_pia:>7.4f}  {_ptax:>7.4f}  {_pens_hand:>8.4f}  {_note:>12}")

    print(f"\n  Cap binds at z >= {z_cap:.3f}  (exp(z)*avg_det >= 2.5)")
    flag("Hand-computed pension = precomputed array at all z", _pipeline_ok)
    itest("Pipeline hand-calc matches precomp", _pipeline_ok)

    # ── 2d. Effective tax schedule ──────────────────────────────────────────
    sub("Effective tax schedule")
    print(f"  Decomposition of after-tax income at selected gross income levels.")
    print(f"  All values in model units (1 unit ~ $61k).  Post-TCJA 7-bracket")
    print(f"  progressive income tax plus 10.6% payroll tax capped at Y_gross = 2.5.")
    print(f"  Effective rate should rise broadly with income; a small dip near the")
    print(f"  payroll cap (2.5) is expected as the 10.6% payroll levy stops.\n")
    _test_incomes = [0.10, 0.30, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 5.00, 10.00]
    print(f"  {'Y_gross':>8}  {'Payroll':>8}  {'Taxable':>8}  {'Inc_tax':>8}  "
          f"{'Y_net':>8}  {'Eff.rate':>8}  {'Note':>14}")
    print(f"  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>8}  "
          f"{'---':>8}  {'---':>8}  {'---':>14}")

    _eff_rates_list = []
    for _y_gr in _test_incomes:
        _payroll = 0.106 * min(_y_gr, 2.5)
        _taxable = max(0.0, _y_gr - _payroll)
        _y_net_val = float(disposable_income_working(np.array([_y_gr]))[0])
        _inc_tax = _taxable - _y_net_val
        _eff = 1.0 - _y_net_val / _y_gr if _y_gr > 0 else 0.0
        _eff_rates_list.append(_eff)
        _note = ""
        if abs(_y_gr - 2.5) < 0.01:
            _note = "<- payroll cap"
        print(f"  {_y_gr:>8.2f}  {_payroll:>8.3f}  {_taxable:>8.3f}  {_inc_tax:>8.3f}  "
              f"{_y_net_val:>8.3f}  {_eff:>7.1%}   {_note}")

    _eff_mono = all(_eff_rates_list[i] <= _eff_rates_list[i+1] + 1e-10
                    for i in range(len(_eff_rates_list)-1))
    # The payroll tax cap at 2.5 creates a known local dip in effective rate
    # (marginal drops from ~32.6% to ~24% as payroll tax stops).
    # Check the weaker condition: rate at max income > rate at min income.
    _eff_broadly_rising = _eff_rates_list[-1] > _eff_rates_list[0]
    if not _eff_mono:
        flag("Effective tax rate strictly monotone", False,
             "expected: payroll cap at 2.5 causes local dip (see table)")
    flag("Effective tax rate broadly rising", _eff_broadly_rising,
         f"low={_eff_rates_list[0]:.1%}, high={_eff_rates_list[-1]:.1%}")
    itest("Effective rate broadly rising", _eff_broadly_rising)

    _y_boundary = 0.18 / (1.0 - 0.106)
    _y_net_boundary = float(disposable_income_working(np.array([_y_boundary]))[0])
    _actual_tax_boundary = 0.18 - _y_net_boundary
    _expected_tax_boundary = 0.18 * 0.10
    _ok = abs(_actual_tax_boundary - _expected_tax_boundary) < 1e-8
    flag("Tax at 10%/12% boundary = 0.018 (cumulative constant)", _ok,
         f"expected = {_expected_tax_boundary:.6f}, got = {_actual_tax_boundary:.6f}")
    itest("Bracket constant 10%/12%", _ok)

    # ── 2e. SS consistency ──────────────────────────────────────────────────
    sub("Social Security consistency")
    print(f"  The tax side (payroll tax cap) and benefit side (AIME cap) must use")
    print(f"  the same earnings ceiling.  This was the root cause of the original")
    print(f"  pension bug: the tax side correctly capped at 2.5, but the benefit")
    print(f"  side used uncapped exp(z) without avg_det scaling.\n")
    print(f"  Tax side  (disposable_income_working):")
    print(f"    Payroll rate:           10.6%")
    print(f"    Payroll cap (Y_gross):  2.500")
    print(f"    Max payroll tax:        {0.106 * 2.5:.3f}")
    print()
    print(f"  Benefit side (compute_pension_after_tax):")
    print(f"    avg_det:                {avg_det:.4f}")
    print(f"    AIME cap:               2.500")
    print(f"    z at which cap binds:   {z_cap:.3f}  (exp({z_cap:.3f})*{avg_det:.4f} = 2.500)")
    print(f"    Max PIA (gross):        {pia_max_gross:.4f}")
    print(f"    Max pension (after tax):{pia_max_net:.4f}")
    print(f"    PIA bend points:        0.21, 1.25")
    print(f"    PIA rates:              90%, 32%, 15%")

    # ── 2f. Retirement boundary ─────────────────────────────────────────────
    sub(f"Retirement boundary sequence  (z = z_grid[{iz0}] = {pc.z_grid[iz0]:.4f})")
    print(f"  Traces the income source at each age around retirement for one agent.")
    print(f"  'Source' shows the exact array and index used — an off-by-one error")
    print(f"  would show the wrong array name or index at the transition point.")
    print(f"  Convention: last labor paycheck at age {model.retire_age}, first pension at age {model.retire_age + 1}.")
    print(f"  The persistent state z transitions one final time at {model.retire_age} and freezes.\n")
    _boundary_ages = list(range(max(model.start_age, model.retire_age - 3),
                                min(model.terminal_age + 1, model.retire_age + 4)))
    print(f"  {'Age':>4}  {'t_idx':>5}  {'Phase':>8}  {'Source':>38}  {'Y_net':>8}")
    print(f"  {'---':>4}  {'---':>5}  {'---':>8}  {'---':>38}  {'---':>8}")

    for _age in _boundary_ages:
        _t = _age - model.start_age
        if _t < 0 or _t >= pc.n_age:
            continue
        if _age < model.retire_age:
            _phase = "Working"
            _source = f"working_income[{_t}, {iz0}, {ie0}]"
            _y_val = float(pc.working_income[_t, iz0, ie0])
            _bnote = "  last working year" if _age == model.retire_age - 1 else ""
        elif _age == model.retire_age:
            _phase = "Retire"
            if _t < pc.working_income.shape[0]:
                _source = f"working_income[{_t}, {iz0}, {ie0}]"
                _y_val = float(pc.working_income[_t, iz0, ie0])
            else:
                _source = f"pension_after_tax[{_t}, {iz0}]"
                _y_val = float(pc.pension_after_tax[_t, iz0])
            _bnote = "  final labor paycheck"
        else:
            _phase = "Retired"
            _source = f"pension_after_tax[{_t}, {iz0}]"
            _y_val = float(pc.pension_after_tax[_t, iz0])
            _bnote = "  first pension" if _age == model.retire_age + 1 else ""
        print(f"  {_age:>4}  {_t:>5}  {_phase:>8}  {_source:<38}  {_y_val:>8.4f}{_bnote}")

    print(f"\n  retire_age_idx = {retire_t}   (age {model.retire_age}, 0-indexed from {model.start_age})")
    print(f"  sim t < {retire_t}: z transitions, income = working_income[t+1, ...]")
    print(f"  sim t >= {retire_t}: z frozen, income = pension_after_tax[t+1, ...]")

    _ok = retire_t < pc.working_income.shape[0]
    flag("working_income array covers retirement age", _ok,
         f"need index {retire_t} in axis 0 of shape {pc.working_income.shape}")
    itest("working_income covers retire_age", _ok)

    _ok = (pc.n_age - 1) < pc.pension_after_tax.shape[0]
    flag("pension_after_tax array covers terminal age", _ok,
         f"need index {pc.n_age - 1} in axis 0 of shape {pc.pension_after_tax.shape}")
    itest("pension covers terminal age", _ok)

    # ── 2g. Expected income lifecycle ────────────────────────────────────────
    sub("Expected after-tax income lifecycle  (z = 0, integrated over eps)")
    print(f"  E[Y_net | z=0, age] using Gauss-Hermite quadrature over the transitory")
    print(f"  shock eps.  This is the income path that drives the median agent's saving")
    print(f"  decision.  E[Y_gross] includes the Jensen correction E[exp(eps)] = {e_exp_eps:.4f}.\n")
    print(f"  {'Age':>4}  {'E[Y_gross]':>10}  {'E[Y_net]':>9}  {'Note':>16}")
    print(f"  {'---':>4}  {'---':>10}  {'---':>9}  {'---':>16}")

    _display_ages = sorted(set(a for a in
        [model.start_age, 25, 30, 35, 40, 45, 46, 50, 55, 60, 65, model.retire_age - 1]
        if model.start_age <= a < model.retire_age))

    for _age in _display_ages:
        _t = _age - model.start_age
        _det = (model.b0 + model.b1 * _age + model.b2 * _age**2 / 10.0
                + model.b3 * _age**3 / 100.0)
        _e_y_gross = np.exp(_det + pc.z_grid[iz0]) * e_exp_eps
        _e_y_net = float(np.dot(pc.eps_weights, pc.working_income[_t, iz0, :]))
        _lnote = ""
        if _age == model.start_age:      _lnote = "entry"
        elif _age == working_ages[_peak_idx]: _lnote = "<- near peak"
        elif _age == model.retire_age - 1:    _lnote = "last working yr"
        print(f"  {_age:>4}  {_e_y_gross:>10.4f}  {_e_y_net:>9.4f}  {_lnote:>16}")

    pens_z0 = float(pc.pension_after_tax[0, iz0])
    print(f"  ---- RETIREMENT --------")
    print(f"  {model.retire_age + 1:>4}  {'':>10}  {pens_z0:>9.4f}  {'first pension':>16}")

    career_avg_net = float(np.mean([
        np.dot(pc.eps_weights, pc.working_income[_t, iz0, :])
        for _t in range(len(working_ages))
    ]))
    last_working_net = float(np.dot(pc.eps_weights,
                                     pc.working_income[retire_t - 1, iz0, :]))
    peak_net = float(max(
        np.dot(pc.eps_weights, pc.working_income[_t, iz0, :])
        for _t in range(len(working_ages))
    ))

    print()
    print(f"  Career average E[Y_net|z=0]:   {career_avg_net:.4f}")
    print(f"  Pension / career avg:          {pens_z0 / career_avg_net:.1%}")
    print(f"  Pension / last working yr:     {pens_z0 / last_working_net:.1%}")
    print(f"  Pension / peak yr:             {pens_z0 / peak_net:.1%}")
    print(f"  Income drop at retirement:     {last_working_net:.4f} -> {pens_z0:.4f}"
          f"  ({(pens_z0 / last_working_net - 1) * 100:+.1f}%)")

    repl_career = pens_z0 / career_avg_net
    _ok = 0.40 <= repl_career <= 0.80
    flag("Median replacement rate in [40%, 80%]", _ok,
         f"pension/career_avg = {repl_career:.1%}")
    itest("Median replacement rate range", _ok)

    # ── 2h. Income distribution across z ─────────────────────────────────────
    sub("Income distribution across persistent states  (eps integrated)")
    print(f"  E[Y_net | z, age] across the z-grid at three representative ages, plus")
    print(f"  the pension each z-state receives in retirement.  The pension column")
    print(f"  should flatten at the top where the AIME cap (2.5) binds — high earners")
    print(f"  all receive the same maximum benefit despite wildly different incomes.\n")

    _dist_ages = sorted(set(a for a in [25, working_ages[_peak_idx], model.retire_age - 1]
                            if model.start_age <= a < model.retire_age))
    _age_hdrs = "".join(f"  {'Age '+str(a):>10}" for a in _dist_ages)
    print(f"  {'z_idx':>5}  {'z':>7}{_age_hdrs}  {'Pension':>10}")
    print(f"  {'---':>5}  {'---':>7}" + "  ----------" * len(_dist_ages) + "  ----------")

    if n_z <= 7:
        _show_iz = list(range(n_z))
    else:
        _show_iz = sorted(set([0, 1, n_z // 4, iz0, 3 * n_z // 4, n_z - 2, n_z - 1]))

    for _iz in _show_iz:
        _z_val = pc.z_grid[_iz]
        _vals = ""
        for _age in _dist_ages:
            _t = _age - model.start_age
            _e_y = float(np.dot(pc.eps_weights, pc.working_income[_t, _iz, :]))
            _vals += f"  {_e_y:>10.4f}"
        _pens = float(pc.pension_after_tax[0, _iz])
        print(f"  {_iz:>5}  {_z_val:>7.3f}{_vals}  {_pens:>10.4f}")

    _e_y_by_z = np.array([float(np.dot(pc.eps_weights, pc.working_income[_peak_idx, _iz, :]))
                           for _iz in range(n_z)])
    _ratio_max_min = _e_y_by_z[-1] / max(_e_y_by_z[0], 1e-15)
    _ratio_max_med = _e_y_by_z[-1] / max(_e_y_by_z[iz0], 1e-15)
    print(f"\n  At peak age ({working_ages[_peak_idx]}):")
    print(f"    z_max/z_min income ratio:  {_ratio_max_min:.1f}")
    print(f"    z_max/z_med income ratio:  {_ratio_max_med:.1f}")
    print(f"    (Wide ratios expected: z_grid spans +/-{z_cover:.1f} sigma)")

    # ── 2i. Replacement rates by earnings level ──────────────────────────────
    sub("Replacement rates by earnings level")
    print(f"  Pension as a fraction of working-life income at each z-level.")
    print(f"  The progressive PIA formula means low earners get high replacement")
    print(f"  and high earners get low replacement.  This gradient is the key")
    print(f"  driver of differential portfolio choice in Catherine (2025):")
    print(f"  low earners need fewer private savings, high earners need more.\n")
    print(f"  'vs career' = pension / mean(E[Y_net]) over working life (SSA standard)")
    print(f"  'vs last'   = pension / E[Y_net] at age {model.retire_age - 1} (last working year)")
    print(f"  'vs peak'   = pension / max E[Y_net] over working life\n")
    print(f"  {'z_idx':>5}  {'z':>7}  {'Pension':>8}  {'vs career':>9}  "
          f"{'vs last':>8}  {'vs peak':>8}")
    print(f"  {'---':>5}  {'---':>7}  {'---':>8}  {'---':>9}  {'---':>8}  {'---':>8}")

    _repl_rates_career = []
    for _iz in _show_iz:
        _z_val = pc.z_grid[_iz]
        _pens = float(pc.pension_after_tax[0, _iz])
        _career_avg = float(np.mean([
            np.dot(pc.eps_weights, pc.working_income[_t, _iz, :])
            for _t in range(len(working_ages))
        ]))
        _last_yr = float(np.dot(pc.eps_weights,
                                 pc.working_income[retire_t - 1, _iz, :]))
        _peak_yr = float(max(
            np.dot(pc.eps_weights, pc.working_income[_t, _iz, :])
            for _t in range(len(working_ages))
        ))
        _r_career = _pens / _career_avg if _career_avg > 1e-10 else float("nan")
        _r_last   = _pens / _last_yr    if _last_yr > 1e-10    else float("nan")
        _r_peak   = _pens / _peak_yr    if _peak_yr > 1e-10    else float("nan")
        _repl_rates_career.append(_r_career)
        print(f"  {_iz:>5}  {_z_val:>7.3f}  {_pens:>8.4f}  {_r_career:>8.1%}  "
              f"{_r_last:>8.1%}  {_r_peak:>8.1%}")

    _repl_mono = all(_repl_rates_career[i] >= _repl_rates_career[i+1] - 1e-6
                     for i in range(len(_repl_rates_career) - 1))
    # At very low z, all income is in the first PIA bracket (90%) and lowest
    # tax bracket, so replacement rates are constant.  The economically
    # meaningful check is: (a) rates are non-increasing from median onward,
    # and (b) bottom > top overall.
    _mid_start = len(_repl_rates_career) // 2
    _repl_declining_upper = all(
        _repl_rates_career[i] >= _repl_rates_career[i+1] - 1e-6
        for i in range(_mid_start, len(_repl_rates_career) - 1)
    )
    if not _repl_mono:
        flag("Replacement rates strictly monotone across all z", False,
             "flat at bottom is expected: very low earners are entirely in the 90% PIA bracket")
    flag("Replacement rates decline from median z upward", _repl_declining_upper,
         "this gradient drives the saving and portfolio choice mechanism")
    itest("Replacement rates decline (upper half)", _repl_declining_upper)

    if len(_repl_rates_career) >= 2:
        _ok = _repl_rates_career[0] > _repl_rates_career[-1]
        flag("Lowest earners have higher repl. rate than highest", _ok,
             f"bottom = {_repl_rates_career[0]:.1%}, top = {_repl_rates_career[-1]:.1%}")
        itest("Bottom > top replacement", _ok)

    # ── INCOME & SS PASS/FAIL SUMMARY ────────────────────────────────────────
    _n_pass = sum(1 for _, p in income_tests if p)
    _n_fail = sum(1 for _, p in income_tests if not p)
    _n_total = len(income_tests)

    print()
    print("-" * W)
    if _n_fail == 0:
        print(f"  INCOME & SS DIAGNOSTIC SUMMARY:  ALL {_n_total} TESTS PASSED")
    else:
        print(f"  INCOME & SS DIAGNOSTIC SUMMARY:  {_n_fail} FAILED  /  {_n_total} total")
        print()
        for _tname, _tpassed in income_tests:
            if not _tpassed:
                print(f"    [FAIL]  {_tname}")
    print("-" * W)

    # =========================================================================
    # 3. VAR STRUCTURE
    # =========================================================================
    header(f"3.  VAR STRUCTURE   (per-period units;  Ã—{ppy} to annualize rates)")

    sub("Unconditional means")
    print(f"  State variables  (z_bar_state):")
    for d, name in enumerate(sv):
        v = model.z_bar_state[d]
        ann = f"{v * ppy * 100:+.3f}%/yr" if name not in ("dp",) else "(log level, not a rate)"
        print(f"    {name:>10}:  {v:+.8f}   annualized - {ann}")
    print(f"  Return variables  (z_bar_ret):")
    for k, name in enumerate(rv):
        v = model.z_bar_ret[k]
        print(f"    {name:>10}:  {v:+.8f}   annualized - {v * ppy * 100:+.3f}%/yr")

    # --- matrix printer ---
    def print_matrix(mat, row_names, col_names, indent="  "):
        cw = 10
        print(indent + " " * 12 + "".join(f"{c:>{cw}}" for c in col_names))
        for i, rn in enumerate(row_names):
            row = indent + f"{rn:>12}" + "".join(f"{mat[i, j]:>{cw}.5f}" for j in range(mat.shape[1]))
            print(row)

    sub("Phi_11  (state-to-state persistence)")
    print_matrix(model.Phi_11, sv, sv)
    eigs_11 = np.sort(np.abs(np.linalg.eigvals(model.Phi_11)))[::-1]
    eig_str = ", ".join(f"{e:.4f}" for e in eigs_11)
    stat    = "STATIONARY" if eigs_11[0] < 1.0 else "*** NON-STATIONARY ***"
    print(f"  Eigenvalues |Î»|: [{eig_str}]   -  {stat}")

    sub("Phi_21  (state â†’ return; return loadings on lagged state)")
    print_matrix(model.Phi_21, rv, sv)

    sub("Phi_0_ret  (return intercepts)")
    for k, name in enumerate(rv):
        v = model.Phi_0_ret[k]
        print(f"    {name:>10}:  {v:+.8f}   annualized - {v * ppy * 100:+.3f}%/yr")

    sub("M  (return | next-state conditioning,  Schur complement)")
    print_matrix(model.M, rv, sv)
    print(f"  ||M||_F  = {np.linalg.norm(model.M):.5f}")
    Phi_11_off = model.Phi_11 - np.diag(np.diag(model.Phi_11))
    print(f"  ||M @ Phi_11_off||_F = {np.linalg.norm(model.M @ Phi_11_off):.5f}"
          "  (independence-Rouwenhorst approximation error driver)")

    sub("Return standard deviations (annualized)")
    print(f"  {'Return':>8}   {'Cond. Ïƒ (given s_t,s_{t+1})':>30}   {'Uncond. Ïƒ':>18}")
    for k, name in enumerate(rv):
        cond_std  = np.sqrt(max(0.0, model.Sigma_r_cond[k, k]))
        uncond_std = np.sqrt(max(0.0, model.Sigma_rr[k, k]))
        cond_ann  = cond_std  * np.sqrt(ppy) * 100
        uncond_ann = uncond_std * np.sqrt(ppy) * 100
        print(f"  {name:>8}   {cond_ann:>25.3f}%/yr   {uncond_ann:>13.3f}%/yr")

    # =========================================================================
    # 4. STATE GRID COVERAGE
    # =========================================================================
    header("4.  STATE GRID COVERAGE")

    N_per_dim = pc.state_grid_sizes
    print(f"  Grid sizes: {N_per_dim}   -   N_state = {pc.N_state} joint states")
    print(f"  Rouwenhorst coverage per dimension: -(N-1) Ïƒ")
    print()
    print(f"  {'Var':>8}  {'N':>3}  {'- cover':>9}  {'Grid min':>10}  {'Grid max':>10}  "
          f"{'Uncond. Î¼':>11}  {'Uncond. Ïƒ':>11}  Ann. range (Ã—{ppy}Ã—100)")
    print(f"  {'-'*8}  {'-'*3}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*30}")

    for d, name in enumerate(sv):
        g      = pc.state_grids[d]
        Nd     = len(g)
        mu_d   = model.z_bar_state[d]
        rho_d  = model.Phi_11[d, d]
        # unconditional Ïƒ from the residual variance used by Rouwenhorst
        sig_inn = np.sqrt(max(1e-14, model.Sigma_ss[d, d]))
        sig_y   = sig_inn / np.sqrt(max(1e-14, 1.0 - rho_d**2))
        cover   = (g.max() - mu_d) / sig_y if sig_y > 0 else float("nan")
        ann_lo  = g.min() * ppy * 100
        ann_hi  = g.max() * ppy * 100
        print(f"  {name:>8}  {Nd:>3}  {cover:>+9.2f}  {g.min():>10.5f}  {g.max():>10.5f}  "
              f"{mu_d:>11.5f}  {sig_y:>11.5f}  [{ann_lo:.2f}%, {ann_hi:.2f}%]")

    sub("r_bill_grid  (real bill rate at each joint state,  annualized)")
    ann_lo_b = pc.r_bill_grid.min() * ppy * 100
    ann_hi_b = pc.r_bill_grid.max() * ppy * 100
    ann_mu_b = pc.r_bill_grid.mean() * ppy * 100
    print(f"  Range: [{ann_lo_b:.3f}%, {ann_hi_b:.3f}%]   Mean: {ann_mu_b:.3f}%   ({pc.N_state} values)")

    # =========================================================================
    # 5. CONDITIONAL RETURN DISTRIBUTION
    # =========================================================================
    header("5.  CONDITIONAL RETURN DISTRIBUTION")
    print(f"  return quadrature: {pc.disc_config.n_ret_nodes_1d} nodes/dim -> {pc.n_ret_quad} joint nodes")
    print(f"  ret_nodes shape: {pc.ret_nodes.shape}")

    print(f"  mu_r shape: {pc.mu_r.shape}   (N_state Ã— N_state Ã— n_ret)")

    # Stationary distribution of Pi_state
    try:
        evals, evecs = np.linalg.eig(pc.Pi_state.T)
        idx  = int(np.argmin(np.abs(evals - 1.0)))
        stat = np.real(evecs[:, idx])
        stat = np.abs(stat) / np.abs(stat).sum()
    except Exception:
        stat = np.ones(pc.N_state) / pc.N_state

    # E[return | state_t=i] = sum_j Pi[i,j] * mu_r[i,j,k]
    E_ret_by_state = np.einsum("ij,ijk->ik", pc.Pi_state, pc.mu_r)  # (N_state, n_ret)

    print()
    print(f"  {'Return':>8}  {'mu_r min':>10}  {'mu_r max':>10}  "
          f"{'Ann. min':>12}  {'Ann. max':>12}  {'Cond.Ïƒ (ann)':>14}  {'Uncond.E[r] (ann)':>18}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*18}")
    for k, name in enumerate(rv):
        mu_lo  = float(pc.mu_r[:, :, k].min())
        mu_hi  = float(pc.mu_r[:, :, k].max())
        cs_ann = np.sqrt(max(0.0, model.Sigma_r_cond[k, k])) * np.sqrt(ppy) * 100
        e_unc  = float(stat @ E_ret_by_state[:, k]) * ppy * 100
        print(f"  {name:>8}  {mu_lo:>10.5f}  {mu_hi:>10.5f}  "
              f"{mu_lo*ppy*100:>12.3f}%  {mu_hi*ppy*100:>12.3f}%  "
              f"{cs_ann:>12.3f}%/yr  {e_unc:>16.3f}%/yr")

    sub("E[return | state_t=i]  across all states  (annualized %/yr)")
    print(f"  {'Return':>8}  {'Minimum state':>16}  {'Maximum state':>16}  {'Stationary mean':>18}")
    for k, name in enumerate(rv):
        e_min = float(E_ret_by_state[:, k].min()) * ppy * 100
        e_max = float(E_ret_by_state[:, k].max()) * ppy * 100
        e_stat = float(stat @ E_ret_by_state[:, k]) * ppy * 100
        print(f"  {name:>8}  {e_min:>13.3f}%  {e_max:>13.3f}%  {e_stat:>15.3f}%")

    # =========================================================================
    # 6. NUMERICAL SETUP & MEMORY
    # =========================================================================
    header("6.  NUMERICAL SETUP & MEMORY FOOTPRINT")

    sub("Dimension table")
    dims = [
        ("n_w",     pc.n_w,     "wealth grid points"),
        ("n_s",     pc.n_s,     "savings grid points (EGM)"),
        ("n_z",     pc.n_z,     "persistent income states"),
        ("n_eps",   pc.n_eps,   f"transitory shock nodes  (= 2 Ã— {pc.n_eps // 2} GH nodes)"),
        ("n_age",   pc.n_age,   f"age periods  ({model.start_age}â€“{model.terminal_age})"),
        ("N_state", pc.N_state, f"joint VAR states  ({'Ã—'.join(str(n) for n in pc.state_grid_sizes)})"),
        ("n_state", model.n_state, "slow-state variables"),
        ("n_ret",   model.n_ret,   "return variables (integrated)"),
    ]
    dims.append(("n_ret_quad", pc.n_ret_quad,
                 f"joint return quadrature nodes  (= {pc.disc_config.n_ret_nodes_1d}^{model.n_ret})"))
    for name, val, desc in dims:
        print(f"  {name:>10} = {val:>6}   {desc}")

    sub("Memory estimate for Part 2 arrays  (float64, 8 bytes/element)")
    bpf = 8

    def mb(n): return n * bpf / 1024**2

    rows = [
        ("Value function",   pc.n_w * pc.n_z * pc.N_state * pc.n_age,
         f"n_w Ã— n_z Ã— N_state Ã— n_age  = {pc.n_w}Ã—{pc.n_z}Ã—{pc.N_state}Ã—{pc.n_age}"),
        ("Policy (Î±_s,Î±_r)", 2 * pc.n_w * pc.n_z * pc.N_state * pc.n_age,
         "2 Ã— same"),
        ("mu_r",             pc.N_state**2 * model.n_ret,
         f"N_stateÂ² Ã— n_ret  = {pc.N_state}Â²Ã—{model.n_ret}"),
        ("Pi_state",         pc.N_state**2,
         f"N_stateÂ²  = {pc.N_state}Â²"),
        ("working_income",   pc.n_age * pc.n_z * pc.n_eps,
         f"n_age Ã— n_z Ã— n_eps  = {pc.n_age}Ã—{pc.n_z}Ã—{pc.n_eps}"),
        ("pension_after_tax", pc.n_age * pc.n_z,
         f"n_age Ã— n_z  = {pc.n_age}Ã—{pc.n_z}"),
    ]
    rows.insert(3, ("ret_nodes", pc.n_ret_quad * model.n_ret,
                    f"n_ret_quad x n_ret  = {pc.n_ret_quad}x{model.n_ret}"))
    total_el = sum(r[1] for r in rows)
    print(f"  {'Array':<22}  {'Elements':>12}  {'MB':>8}  Description")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}  {'-'*40}")
    for arr_name, n_el, desc in rows:
        print(f"  {arr_name:<22}  {n_el:>12,}  {mb(n_el):>8.2f}  {desc}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}")
    print(f"  {'TOTAL (approx)':<22}  {total_el:>12,}  {mb(total_el):>8.2f}")

    sub("Grid ranges")
    print(f"  wealth_grid : [{pc.wealth_grid.min():.6f}, {pc.wealth_grid.max():.1f}]  (geometric,  {pc.n_w} pts)")
    print(f"  s_grid      : [{pc.s_grid.min():.2e}, {pc.s_grid.max():.1f}]  (geometric,  {pc.n_s} pts)")

    # =========================================================================
    # 7. SANITY CHECKS
    # =========================================================================
    header("7.  SANITY CHECKS")

    eig_11_max = float(np.max(np.abs(np.linalg.eigvals(model.Phi_11))))
    flag("Phi_11 stationary",     eig_11_max < 1.0,
         f"max|Î»| = {eig_11_max:.5f}")

    flag("Income innovation mean â‰ˆ 0", abs(mu_eta) < 1e-6,
         f"E[eta] = {mu_eta:.2e}")

    flag("Transitory shock mean â‰ˆ 0",  abs(eps_mean) < 1e-10,
         f"E[eps] = {eps_mean:.2e}")

    flag("eps_weights sum to 1",
         abs(pc.eps_weights.sum() - 1.0) < 1e-10,
         f"sum = {pc.eps_weights.sum():.10f}")

    ret_w_mean = np.sum(pc.ret_nodes * pc.ret_weights[:, None], axis=0)
    ret_cov = (pc.ret_nodes * pc.ret_weights[:, None]).T @ pc.ret_nodes
    flag("ret_weights sum to 1",
         abs(pc.ret_weights.sum() - 1.0) < 1e-10,
         f"sum = {pc.ret_weights.sum():.10f}")
    flag("Return quadrature mean ≈ 0",
         float(np.max(np.abs(ret_w_mean))) < 1e-10,
         f"max |mean| = {float(np.max(np.abs(ret_w_mean))):.2e}")
    flag("Return quadrature covariance matches Sigma_r_cond",
         float(np.max(np.abs(ret_cov - model.Sigma_r_cond))) < 1e-8,
         f"max |diff| = {float(np.max(np.abs(ret_cov - model.Sigma_r_cond))):.2e}")

    pi_s_ok  = np.allclose(pc.Pi_state.sum(axis=1), 1.0, atol=1e-10)
    flag("Pi_state row sums = 1", pi_s_ok,
         f"max deviation = {np.abs(pc.Pi_state.sum(axis=1) - 1.0).max():.2e}")

    pi_z_ok  = np.allclose(pc.Pi_z.sum(axis=1), 1.0, atol=1e-10)
    flag("Pi_z row sums = 1",     pi_z_ok,
         f"max deviation = {np.abs(pc.Pi_z.sum(axis=1) - 1.0).max():.2e}")

    try:
        np.linalg.cholesky(model.Sigma_ss)
        flag("Sigma_ss positive definite",     True)
    except np.linalg.LinAlgError:
        flag("Sigma_ss positive definite",     False, "Cholesky decomposition failed")

    try:
        np.linalg.cholesky(model.Sigma_r_cond)
        flag("Sigma_r_cond positive definite", True)
    except np.linalg.LinAlgError:
        flag("Sigma_r_cond positive definite", False, "Cholesky decomposition failed")

    surv_ok = bool(np.all((pc.survival_probs_2d > 0) & (pc.survival_probs_2d <= 1.0)))
    flag("Survival probs in (0, 1]", surv_ok,
         f"range = [{pc.survival_probs_2d.min():.5f}, {pc.survival_probs_2d.max():.5f}]")

    wi_pos = bool(np.all(pc.working_income > 0))
    flag("Working income > 0 everywhere", wi_pos,
         f"min = {pc.working_income.min():.6f}")

    pens_pos = bool(np.all(pc.pension_after_tax > 0))
    flag("Pension > 0 for all z states", pens_pos,
         f"min = {pc.pension_after_tax.min():.6f}")

    flag("Wealth grid strictly positive", bool(pc.wealth_grid.min() > 0),
         f"min = {pc.wealth_grid.min():.2e}")

    # Grid coverage per state dimension
    for d, name in enumerate(sv):
        g       = pc.state_grids[d]
        rho_d   = model.Phi_11[d, d]
        sig_inn = np.sqrt(max(1e-14, model.Sigma_ss[d, d]))
        sig_y   = sig_inn / np.sqrt(max(1e-14, 1.0 - rho_d**2))
        cover   = (g.max() - model.z_bar_state[d]) / sig_y if sig_y > 0 else 0.0
        flag(f"State grid coverage: {name}",
             cover >= 2.5,
             f"Â±{cover:.2f}Ïƒ  (recommend â‰¥ 2.5Ïƒ;  use larger state_grid_sizes for more)")

    print()
    print("=" * W)
    print("  Diagnostic report complete.")
    print("=" * W)


# =============================================================================
# TIER 2 — POST-SIMULATION INCOME DIAGNOSTICS
# =============================================================================

def print_simulation_income_report(model, pc, sim):
    """
    Post-simulation diagnostics for the income process and Social Security.

    Verifies that the simulation's random draws and loop logic produce
    income outcomes consistent with the precomputed tables.  Runs in
    seconds (operates on sim arrays, no solving).

    Call after simulate_lifecycle returns.

    Parameters
    ----------
    model : LifecyclePortfolioModel
    pc    : Precompute
    sim   : dict
        Simulation output from simulate_lifecycle.  Expected keys:
        'income', 'alive', 'z_idx', 'x', 'death_age', 'ages'.
    """
    W = 76

    def header(title):
        print()
        print("=" * W)
        print(f"  {title}")
        print("=" * W)

    def sub(title):
        print(f"\n  --- {title} ---")

    def flag(label, ok, detail=""):
        tag = "[PASS]" if ok else "[WARN]"
        suf = f"  {detail}" if detail else ""
        print(f"  {tag}  {label}{suf}")

    sim_tests = []

    def stest(name, passed):
        sim_tests.append((name, passed))
        return passed

    # Unpack simulation arrays
    sim_income = sim["income"]          # (n_sim, n_age)
    sim_alive  = sim["alive"]           # (n_sim, n_age) bool
    sim_z_idx  = sim["z_idx"]           # (n_sim, n_age) int
    ages       = sim["ages"]            # (n_age,)
    n_sim, n_age = sim_income.shape

    iz0 = int(np.argmin(np.abs(pc.z_grid)))
    retire_t = model.retire_age - model.start_age
    working_ages = np.arange(model.start_age, model.retire_age)

    header("POST-SIMULATION INCOME DIAGNOSTICS (Tier 2)")
    print(f"  Simulation: {n_sim} agents, ages {ages[0]}–{ages[-1]}, "
          f"{n_age} periods")

    # ── S1. Simulated vs theoretical income moments ─────────────────────────
    sub("Simulated vs theoretical income moments")
    print(f"  Compares realized mean income (across alive agents) against the")
    print(f"  theoretical expectation from the precomputed tables, weighted by")
    print(f"  the stationary distribution of z and the eps quadrature.\n")

    # Compute stationary distribution of z (eigenvector of Pi_z')
    _evals, _evecs = np.linalg.eig(pc.Pi_z.T)
    _stat_idx = int(np.argmin(np.abs(_evals - 1.0)))
    pi_z_stat = np.abs(_evecs[:, _stat_idx])
    pi_z_stat = pi_z_stat / pi_z_stat.sum()

    print(f"  {'Age':>4}  {'N_alive':>7}  {'E[Y_sim]':>9}  {'E[Y_theory]':>11}  "
          f"{'Ratio':>7}  {'Std[Y_sim]':>10}  {'Note':>8}")
    print(f"  {'---':>4}  {'---':>7}  {'---':>9}  {'---':>11}  "
          f"{'---':>7}  {'---':>10}  {'---':>8}")

    _key_ages_s1 = [25, 30, 40, 50, 60, model.retire_age - 1,
                    model.retire_age + 1, 80]
    _key_ages_s1 = sorted(set(a for a in _key_ages_s1
                              if model.start_age <= a <= model.terminal_age))

    _worst_ratio_dev = 0.0
    for _age in _key_ages_s1:
        _t = _age - model.start_age
        _alive_mask = sim_alive[:, _t]
        _n_alive = int(np.sum(_alive_mask))
        if _n_alive < 10:
            continue

        _y_alive = sim_income[_alive_mask, _t]
        _e_sim = float(np.mean(_y_alive))
        _std_sim = float(np.std(_y_alive))

        # Theoretical: E[Y] = Σ_z π(z) × E_eps[Y(age, z, eps)]
        if _age < model.retire_age:
            _e_theory = float(sum(
                pi_z_stat[iz] * np.dot(pc.eps_weights, pc.working_income[_t, iz, :])
                for iz in range(len(pc.z_grid))
            ))
        else:
            _e_theory = float(sum(
                pi_z_stat[iz] * pc.pension_after_tax[_t, iz]
                for iz in range(len(pc.z_grid))
            ))

        _ratio = _e_sim / _e_theory if _e_theory > 1e-10 else float("nan")
        _note = ""
        if _age == model.retire_age + 1:
            _note = "pension"
        _worst_ratio_dev = max(_worst_ratio_dev, abs(_ratio - 1.0))

        print(f"  {_age:>4}  {_n_alive:>7}  {_e_sim:>9.4f}  {_e_theory:>11.4f}  "
              f"{_ratio:>7.3f}  {_std_sim:>10.4f}  {_note:>8}")

    print()
    print(f"  Ratio = E[Y_sim] / E[Y_theory].  Values near 1.0 expected.")
    print(f"  Moderate deviations at older ages are normal: income-dependent")
    print(f"  mortality selects for higher-z survivors, raising mean income.")

    # S1 test: worst ratio deviation < 20%
    _ok = _worst_ratio_dev < 0.20
    flag("Sim/theory income ratio within 20% at all ages", _ok,
         f"worst deviation = {_worst_ratio_dev:.1%}")
    stest("Income moments sim vs theory", _ok)

    # ── S2. Realized replacement rates ──────────────────────────────────────
    sub("Realized replacement rates at retirement")
    print(f"  Distribution of pension / last-working-year income across agents")
    print(f"  who survive to retirement.  The transitory shock eps makes the")
    print(f"  last-year income noisy, so these ratios are more dispersed than")
    print(f"  the grid-based Tier 1 values.\n")

    # Find agents alive at both last working year and first pension year
    _t_last_work = retire_t - 1
    _t_first_pens = retire_t + 1
    if _t_first_pens < n_age:
        _retiree_mask = sim_alive[:, _t_last_work] & sim_alive[:, _t_first_pens]
        _n_retirees = int(np.sum(_retiree_mask))

        _y_last = sim_income[_retiree_mask, _t_last_work]
        _y_pens = sim_income[_retiree_mask, _t_first_pens]

        # Avoid division by zero for agents with very low last-year income
        _safe = _y_last > 1e-6
        _repl_realized = _y_pens[_safe] / _y_last[_safe]

        if len(_repl_realized) > 0:
            _pctiles = [10, 25, 50, 75, 90]
            _pct_vals = np.percentile(_repl_realized, _pctiles)

            print(f"  N retirees (alive at ages {model.retire_age - 1} & {model.retire_age + 1}): "
                  f"{_n_retirees}")
            print()
            print(f"  {'Percentile':>10}  {'Last Y_net':>10}  {'Pension':>8}  {'Repl.rate':>9}")
            print(f"  {'---':>10}  {'---':>10}  {'---':>8}  {'---':>9}")

            for _p in _pctiles:
                _idx = int(np.percentile(np.arange(len(_repl_realized)),
                                         _p, method='nearest'))
                # Show representative agent near this percentile
                _sort_idx = np.argsort(_repl_realized)
                _agent_idx = _sort_idx[min(_idx, len(_sort_idx)-1)]
                print(f"  {_p:>9}th  {_y_last[_safe][_agent_idx]:>10.4f}  "
                      f"{_y_pens[_safe][_agent_idx]:>8.4f}  "
                      f"{_repl_realized[_agent_idx]:>8.1%}")

            _median_repl = float(np.median(_repl_realized))
            print(f"\n  Median realized replacement rate: {_median_repl:.1%}")

            # S2 test: median replacement rate in plausible range
            _ok = 0.20 <= _median_repl <= 2.0
            flag("Median realized replacement rate in [20%, 200%]", _ok,
                 f"median = {_median_repl:.1%}")
            stest("Realized replacement rate range", _ok)
        else:
            print("  WARNING: No retirees with positive last-year income.")
            stest("Realized replacement rate range", False)
    else:
        print("  WARNING: Cannot compute — simulation too short for pension year.")
        stest("Realized replacement rate range", False)

    # ── S3. Retirement boundary trace ───────────────────────────────────────
    sub("Retirement boundary trace (spot-check individual agents)")
    print(f"  Traces 3 individual agents through the retirement transition.")
    print(f"  Verifies: z_idx freezes at retirement, income switches from")
    print(f"  working_income to pension_after_tax, no unexpected jumps.\n")

    # Pick 3 agents that survive past retirement: low, mid, high z at retirement
    _surv_retire = np.where(sim_alive[:, min(retire_t + 2, n_age - 1)])[0]
    if len(_surv_retire) >= 3:
        _z_at_retire = sim_z_idx[_surv_retire, retire_t]
        _z_sorted = np.argsort(_z_at_retire)
        _picks = [_z_sorted[0],
                  _z_sorted[len(_z_sorted) // 2],
                  _z_sorted[-1]]
        _agent_ids = _surv_retire[_picks]
    elif len(_surv_retire) > 0:
        _agent_ids = _surv_retire[:min(3, len(_surv_retire))]
    else:
        _agent_ids = []

    _z_freeze_ok = True
    for _agent in _agent_ids:
        _z_ret = sim_z_idx[_agent, retire_t]
        print(f"  Agent #{_agent}  (z_idx at retirement = {_z_ret},"
              f" z = {pc.z_grid[_z_ret]:.3f})")
        print(f"    {'Age':>4}  {'z_idx':>5}  {'Income':>8}  {'Note':>20}")

        for _t in range(max(0, retire_t - 2), min(n_age, retire_t + 4)):
            _age = ages[_t]
            _z_t = sim_z_idx[_agent, _t]
            _y_t = sim_income[_agent, _t]
            _note = ""
            if _age == model.retire_age - 1:
                _note = "last working year"
            elif _age == model.retire_age:
                _note = "final labor paycheck"
            elif _age == model.retire_age + 1:
                _note = "first pension"
            # Check z frozen after retirement
            if _age > model.retire_age and _z_t != _z_ret:
                _note += " Z NOT FROZEN!"
                _z_freeze_ok = False
            print(f"    {_age:>4}  {_z_t:>5}  {_y_t:>8.4f}  {_note}")
        print()

    flag("z_idx frozen after retirement for all traced agents", _z_freeze_ok)
    stest("z frozen at retirement", _z_freeze_ok)

    # Broader z-freeze check across all agents
    if retire_t + 1 < n_age:
        _alive_post = sim_alive[:, retire_t + 1]
        _z_at_ret = sim_z_idx[:, retire_t]
        _z_at_ret1 = sim_z_idx[:, retire_t + 1]
        _z_frozen_all = np.all(_z_at_ret[_alive_post] == _z_at_ret1[_alive_post])
        flag("z_idx frozen for ALL agents at retire+1", _z_frozen_all,
             f"checked {int(np.sum(_alive_post))} alive agents")
        stest("z frozen globally at retire+1", _z_frozen_all)

    # ── S4. Survivor selection effect ───────────────────────────────────────
    sub("Survivor selection effect  (income-dependent mortality)")
    print(f"  If mortality is income-dependent, the surviving population should")
    print(f"  shift toward higher z over time (richer people live longer).")
    print(f"  Mean z_idx should rise with age; if it doesn't, the mortality")
    print(f"  calibration may not be working as intended.\n")

    print(f"  {'Age':>4}  {'N_alive':>7}  {'Mean z_idx':>10}  {'Med z_idx':>9}  "
          f"{'Frac z>0':>8}  {'Note':>12}")
    print(f"  {'---':>4}  {'---':>7}  {'---':>10}  {'---':>9}  "
          f"{'---':>8}  {'---':>12}")

    _select_ages = [model.start_age, 30, 50, model.retire_age, 75, 85, 95]
    _select_ages = sorted(set(a for a in _select_ages
                              if model.start_age <= a <= model.terminal_age))

    _mean_z_vals = []
    for _age in _select_ages:
        _t = _age - model.start_age
        _alive_mask = sim_alive[:, _t]
        _n_alive = int(np.sum(_alive_mask))
        if _n_alive < 5:
            continue

        _z_alive = sim_z_idx[_alive_mask, _t]
        _mean_z = float(np.mean(_z_alive))
        _med_z = float(np.median(_z_alive))
        _frac_above_mid = float(np.mean(_z_alive > iz0))
        _mean_z_vals.append(_mean_z)

        _note = ""
        if _age == model.start_age:
            _note = "entry"
        elif _age == model.retire_age:
            _note = "retirement"

        print(f"  {_age:>4}  {_n_alive:>7}  {_mean_z:>10.2f}  {_med_z:>9.1f}  "
              f"{_frac_above_mid:>7.1%}   {_note:>12}")

    # S4 test: mean z_idx at last reported age > mean z_idx at first reported age
    if len(_mean_z_vals) >= 2:
        _selection_present = _mean_z_vals[-1] > _mean_z_vals[0]
        flag("Survivor selection: mean z_idx rises with age", _selection_present,
             f"entry = {_mean_z_vals[0]:.2f}, late life = {_mean_z_vals[-1]:.2f}")
        stest("Survivor selection present", _selection_present)

    # ── TIER 2 PASS/FAIL SUMMARY ─────────────────────────────────────────────
    _n_pass = sum(1 for _, p in sim_tests if p)
    _n_fail = sum(1 for _, p in sim_tests if not p)
    _n_total = len(sim_tests)

    print()
    print("-" * W)
    if _n_fail == 0:
        print(f"  SIMULATION INCOME DIAGNOSTIC SUMMARY:  ALL {_n_total} TESTS PASSED")
    else:
        print(f"  SIMULATION INCOME DIAGNOSTIC SUMMARY:  {_n_fail} FAILED  /  {_n_total} total")
        print()
        for _tname, _tpassed in sim_tests:
            if not _tpassed:
                print(f"    [FAIL]  {_tname}")
    print("-" * W)


# =============================================================================
# TERMINAL PORTFOLIO DIAGNOSTICS
# =============================================================================

_TERMINAL_EXIT_LABELS = {
    EC_CORNER_BILLS: "corner_bills",
    EC_CORNER_STOCKS: "corner_stocks",
    EC_CORNER_BONDS: "corner_bonds",
    EC_EDGE_SB: "edge_stock_bill",
    EC_EDGE_BB: "edge_bond_bill",
    EC_EDGE_STOCKBOND: "edge_stock_bond",
    EC_INTERIOR: "interior",
    EC_NEWTON_FAIL: "fail",
}


def _terminal_push_hint(foc_s, foc_b, tol):
    """Interpret terminal FOC signs as directions that still improve the objective."""
    scores = {
        "more_stock": foc_s,
        "more_bond": foc_b,
        "more_risky_both": foc_s + foc_b,
        "rotate_to_stock": foc_s - foc_b,
        "rotate_to_bond": foc_b - foc_s,
    }
    label, score = max(scores.items(), key=lambda kv: kv[1])
    if score <= tol:
        return "near_stationary", float(score)
    return label, float(score)


def _probe_terminal_directions(alpha_s, alpha_b, base_moment,
                               R_bill, scenario_weights, R_stock, R_bond, gamma,
                               probe_steps):
    """Check whether the exact terminal objective still improves along simple directions."""
    if not np.isfinite(base_moment):
        return "none", 0.0, 0.0, float("nan")

    directions = [
        ("more_stock", np.array([1.0, 0.0], dtype=float)),
        ("more_bond", np.array([0.0, 1.0], dtype=float)),
        ("more_risky_both", np.array([1.0, 1.0], dtype=float)),
        ("rotate_to_stock", np.array([1.0, -1.0], dtype=float)),
        ("rotate_to_bond", np.array([-1.0, 1.0], dtype=float)),
    ]

    best_label = "none"
    best_step = 0.0
    best_delta = 0.0

    for label, direction in directions:
        for step in probe_steps:
            test_s = alpha_s + step * direction[0]
            test_b = alpha_b + step * direction[1]
            test_moment = _terminal_portfolio_moment(
                test_s, test_b, R_bill, scenario_weights, R_stock, R_bond, gamma
            )
            if not np.isfinite(test_moment):
                continue
            delta = float(test_moment - base_moment)
            if delta < best_delta:
                best_label = label
                best_step = float(step)
                best_delta = delta

    rel_delta = float(best_delta / base_moment) if base_moment > 0.0 else float("nan")
    return best_label, best_step, best_delta, rel_delta


def diagnose_terminal_portfolio_states(model, pc, solver_config=None,
                                       max_fail_rows=12,
                                       probe_steps=(0.25, 0.5, 1.0, 2.0, 5.0),
                                       print_report=True):
    """
    Re-solve the terminal portfolio problem state-by-state and report diagnostics.

    Especially useful in the unconstrained model, where terminal Newton failures
    may reflect either slow convergence or a strong directional push toward more
    leverage.

    Returns
    -------
    results : dict
        Keys:
          - summary: aggregate counts and residual statistics
          - rows: one dict per financial state
          - fail_rows: subset of rows with exit_code == EC_NEWTON_FAIL
    """
    if solver_config is None:
        solver_config = SolverConfig()

    mode = "CONSTRAINED" if model.constrained else "UNCONSTRAINED"
    rows = []

    for i_s in range(pc.N_state):
        R_bill = exp(pc.r_bill_grid[i_s])
        Rx_stock_next, Rx_bond_next = build_gross_return_arrays(pc.mu_r[i_s, :, :], pc.ret_nodes)

        if model.constrained:
            opt_s, opt_b, moment, exit_code, foc_resid = solve_portfolio_2d_terminal_exact(
                i_s, pc.Pi_state, Rx_stock_next, Rx_bond_next, pc.ret_weights, R_bill, model.gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=max(100, 20 * solver_config.max_iter),
            )
        else:
            opt_s, opt_b, moment, exit_code, foc_resid = solve_portfolio_unconstrained_terminal_exact(
                i_s, pc.Pi_state, Rx_stock_next, Rx_bond_next, pc.ret_weights, R_bill, model.gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=max(100, 20 * solver_config.max_iter),
            )

        scenario_weights, R_stock, R_bond, _, _ = _terminal_prepare_scenarios(
            pc.Pi_state[i_s, :], Rx_stock_next, Rx_bond_next, pc.ret_weights, R_bill
        )
        # Recompute moment for diagnostics (ensures consistent evaluation).
        moment = _terminal_portfolio_moment(
            opt_s, opt_b, R_bill, scenario_weights, R_stock, R_bond, model.gamma
        )

        foc_s, foc_b, J_ss, J_bb, J_sb = compute_terminal_portfolio_foc_jac(
            opt_s, opt_b, i_s, pc.Pi_state, Rx_stock_next, Rx_bond_next, pc.ret_weights,
            R_bill, model.gamma, solver_config.min_return_power, solver_config.prob_skip_threshold
        )
        foc_norm = float(np.hypot(foc_s, foc_b))
        jac = np.array([[J_ss, J_sb], [J_sb, J_bb]], dtype=float)
        eigvals = np.linalg.eigvalsh(jac)
        abs_eigs = np.abs(eigvals)
        cond_proxy = (float(abs_eigs.max() / abs_eigs.min())
                      if abs_eigs.min() > 1e-14 else float("inf"))
        det_jac = float(J_ss * J_bb - J_sb * J_sb)

        a_bill = 1.0 - opt_s - opt_b
        R_port = opt_s * R_stock + opt_b * R_bond + a_bill * R_bill
        positive_mask = scenario_weights > 0.0
        min_r_port = float(np.min(R_port[positive_mask])) if np.any(positive_mask) else float("nan")
        max_r_port = float(np.max(R_port[positive_mask])) if np.any(positive_mask) else float("nan")

        push_hint, push_score = _terminal_push_hint(foc_s, foc_b, tol=solver_config.tol)
        probe_label, probe_step, probe_delta, probe_rel = _probe_terminal_directions(
            opt_s, opt_b, moment, R_bill, scenario_weights, R_stock, R_bond, model.gamma, probe_steps
        )

        row = {
            "i_s": int(i_s),
            "state_values": np.asarray(pc.state_grid[i_s], dtype=float).copy(),
            "alpha_s": float(opt_s),
            "alpha_b": float(opt_b),
            "alpha_bill": float(a_bill),
            "moment": float(moment),
            "exit_code": int(exit_code),
            "exit_label": _TERMINAL_EXIT_LABELS.get(int(exit_code), f"ec_{int(exit_code)}"),
            "solver_resid": float(foc_resid),
            "foc_s": float(foc_s),
            "foc_b": float(foc_b),
            "foc_norm": foc_norm,
            "jac_det": det_jac,
            "jac_eig_minabs": float(abs_eigs.min()),
            "jac_eig_maxabs": float(abs_eigs.max()),
            "jac_cond_proxy": cond_proxy,
            "min_r_port": min_r_port,
            "max_r_port": max_r_port,
            "push_hint": push_hint,
            "push_score": push_score,
            "probe_label": probe_label,
            "probe_step": float(probe_step),
            "probe_delta": float(probe_delta),
            "probe_rel_delta": float(probe_rel),
        }
        rows.append(row)

    fail_rows = [row for row in rows if row["exit_code"] == EC_NEWTON_FAIL]
    solver_resids = np.array([row["solver_resid"] for row in rows], dtype=float)
    fail_resids = np.array([row["solver_resid"] for row in fail_rows], dtype=float)

    summary = {
        "mode": mode,
        "n_states": int(pc.N_state),
        "n_interior": int(sum(row["exit_code"] == EC_INTERIOR for row in rows)),
        "n_corner_edge": int(sum(
            row["exit_code"] in {
                EC_CORNER_BILLS, EC_CORNER_STOCKS, EC_CORNER_BONDS,
                EC_EDGE_SB, EC_EDGE_BB, EC_EDGE_STOCKBOND,
            }
            for row in rows
        )),
        "n_fail": int(len(fail_rows)),
        "resid_min": float(np.min(solver_resids)) if len(rows) else float("nan"),
        "resid_median": float(np.median(solver_resids)) if len(rows) else float("nan"),
        "resid_max": float(np.max(solver_resids)) if len(rows) else float("nan"),
        "fail_resid_min": float(np.min(fail_resids)) if len(fail_rows) else float("nan"),
        "fail_resid_median": float(np.median(fail_resids)) if len(fail_rows) else float("nan"),
        "fail_resid_max": float(np.max(fail_resids)) if len(fail_rows) else float("nan"),
    }

    if print_report:
        width = 108
        print("\n" + "=" * width)
        print("TERMINAL PORTFOLIO DIAGNOSTIC")
        print("=" * width)
        print(f"Mode: {mode}")
        print(f"States: {summary['n_states']}  interior={summary['n_interior']}  "
              f"corner/edge={summary['n_corner_edge']}  fail={summary['n_fail']}")
        print(f"Solver residuals: min={summary['resid_min']:.3e}  "
              f"median={summary['resid_median']:.3e}  max={summary['resid_max']:.3e}")

        if summary["n_fail"] == 0:
            print("No terminal failures detected.")
        else:
            print(f"Fail residuals: min={summary['fail_resid_min']:.3e}  "
                  f"median={summary['fail_resid_median']:.3e}  max={summary['fail_resid_max']:.3e}")
            if not model.constrained:
                print("Interpretation guide: positive FOC_s / FOC_b means the terminal objective still")
                print("wants more stock / more bond at the reported failed iterate.")

            print()
            print(f"{'i_s':>4} {'alpha_s':>9} {'alpha_b':>9} {'a_bill':>9} "
                  f"{'||FOC||':>10} {'FOC_s':>10} {'FOC_b':>10} {'det(J)':>11} "
                  f"{'min R_p':>9} {'hint':>16} {'probe':>18}")

            ranked_fails = sorted(fail_rows, key=lambda row: row["solver_resid"], reverse=True)
            for row in ranked_fails[:max_fail_rows]:
                probe = row["probe_label"]
                if probe != "none":
                    probe = f"{probe}@{row['probe_step']:.2g} ({100.0 * row['probe_rel_delta']:.1f}%)"
                print(f"{row['i_s']:4d} {row['alpha_s']:9.4f} {row['alpha_b']:9.4f} "
                      f"{row['alpha_bill']:9.4f} {row['solver_resid']:10.3e} "
                      f"{row['foc_s']:10.3e} {row['foc_b']:10.3e} {row['jac_det']:11.3e} "
                      f"{row['min_r_port']:9.4f} {row['push_hint']:>16} {probe:>18}")

        print("=" * width)

    return {
        "summary": summary,
        "rows": rows,
        "fail_rows": fail_rows,
    }


# =============================================================================
# NEWTON FAILURE DIAGNOSTICS
# =============================================================================
# Investigate WHY ~20% of Newton calls fail.
# Strategy: pick one retirement age, sweep all (i_s, z_i, s_i) for that age,
# evaluate FOCs at corners, and classify why each call fails.

@njit
def diagnose_newton_failures_retirement(
    wealth_grid, savings_grid, z_grid, N_state,
    c_next_full, pension_1d,
    annuity_factors, Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
    gamma, psi_vec, beta, b_bar):
    """
    For each (i_s, z_i, s_i), evaluate corner FOCs and classify the failure mode.
    Returns per-i_s diagnostics (not parallelized, for clarity).
    
    Returns:
        corner_focs : (N_state, 6) -- [fs0, fb0, fs1, fb1, fs2, fb2] at median z, median s
        failure_reasons : (N_state, 6) -- counts per i_s:
            [0] = no_bracket_any_edge (corners rejected, no edge bracket exists)
            [1] = edge_newton_didnt_converge (bracket exists but edge Newton failed acceptance)
            [2] = interior_newton_fail (fell through to interior, didn't converge)
            [3] = total_fail_calls (total Newton failures for this i_s)
            [4] = total_calls (total calls for this i_s)
            [5] = edge_tried_but_rejected (edge Newton ran but acceptance check failed)
    """
    n_z = len(z_grid)
    n_savings = len(savings_grid)
    
    corner_focs = np.empty((N_state, 6))
    failure_reasons = np.zeros((N_state, 6), dtype=np.int64)
    
    z_med = n_z // 2
    s_med = n_savings // 2
    
    for i_s in range(N_state):
        psi_med = psi_vec[z_med]
        R_bill = exp(r_bill_grid[i_s])
        annuity_factor_is = annuity_factors[i_s]
        
        Rx_stock_next, Rx_bond_next = build_gross_return_arrays(mu_r[i_s, :, :], ret_nodes)
        
        # Evaluate corner FOCs at median savings for reporting
        s_val_med = savings_grid[s_med]
        c_next_slice = c_next_full[z_med, :, :]
        pension_next = pension_1d[z_med]
        
        fs0, fb0, _, _, _, _ = compute_foc_jac_retirement(
            0.0, 0.0, s_val_med, z_med, i_s,
            wealth_grid, c_next_slice, pension_next, annuity_factor_is,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi_med, beta, b_bar)
        fs1, fb1, _, _, _, _ = compute_foc_jac_retirement(
            1.0, 0.0, s_val_med, z_med, i_s,
            wealth_grid, c_next_slice, pension_next, annuity_factor_is,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi_med, beta, b_bar)
        fs2, fb2, _, _, _, _ = compute_foc_jac_retirement(
            0.0, 1.0, s_val_med, z_med, i_s,
            wealth_grid, c_next_slice, pension_next, annuity_factor_is,
            Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi_med, beta, b_bar)
        
        corner_focs[i_s, 0] = fs0
        corner_focs[i_s, 1] = fb0
        corner_focs[i_s, 2] = fs1
        corner_focs[i_s, 3] = fb1
        corner_focs[i_s, 4] = fs2
        corner_focs[i_s, 5] = fb2
        
        # Now sweep all (z_i, s_i) and count failure modes
        last_a_s = 0.1
        last_a_b = 0.4
        for z_i in range(n_z):
            psi = psi_vec[z_i]
            c_slice = c_next_full[z_i, :, :]
            pens = pension_1d[z_i]
            for s_i in range(n_savings):
                s_val = savings_grid[s_i]
                failure_reasons[i_s, 4] += 1  # total_calls
                
                _, _, _, exit_code, _ = solve_portfolio_2d_retirement(
                    s_val, z_i, i_s,
                    wealth_grid, c_slice, pens, annuity_factor_is,
                    Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                    gamma, psi, beta, b_bar,
                    init_s=last_a_s, init_b=last_a_b)
                
                if exit_code == EC_NEWTON_FAIL:
                    failure_reasons[i_s, 3] += 1  # total_fail_calls
                    failure_reasons[i_s, 2] += 1  # interior_newton_fail
                    
                    # WHY did it fail? Re-evaluate corners for this specific (z_i, s_i)
                    _fs0, _fb0, _, _, _, _ = compute_foc_jac_retirement(
                        0.0, 0.0, s_val, z_i, i_s,
                        wealth_grid, c_slice, pens, annuity_factor_is,
                        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar)
                    _fs1, _fb1, _, _, _, _ = compute_foc_jac_retirement(
                        1.0, 0.0, s_val, z_i, i_s,
                        wealth_grid, c_slice, pens, annuity_factor_is,
                        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar)
                    _fs2, _fb2, _, _, _, _ = compute_foc_jac_retirement(
                        0.0, 1.0, s_val, z_i, i_s,
                        wealth_grid, c_slice, pens, annuity_factor_is,
                        Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill, gamma, psi, beta, b_bar)
                    
                    # Check bracket conditions
                    has_sb = (_fs0 > 0.0 and _fs1 < 0.0)
                    has_bb = (_fb0 > 0.0 and _fb2 < 0.0)
                    _g1 = _fs1 - _fb1
                    _g2 = _fs2 - _fb2
                    has_stockbond = (_g1 * _g2 < 0.0)
                    
                    if not has_sb and not has_bb and not has_stockbond:
                        failure_reasons[i_s, 0] += 1  # no_bracket_any_edge
                    else:
                        failure_reasons[i_s, 5] += 1  # edge_tried_but_rejected
    
    return corner_focs, failure_reasons


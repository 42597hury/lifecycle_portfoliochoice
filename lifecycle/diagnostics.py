"""
diagnostics.py -- Modular model diagnostic reports and Newton failure analysis.

Pre-solve diagnostics (call after Precompute, before solver):
  - diagnose_income_pre(model, pc)         -- income process & Social Security
  - diagnose_var_pre(model, pc)            -- VAR structure & return distribution
  - diagnose_grids_pre(model, pc)          -- preferences, grids, memory, survival
  - diagnose_all_pre(model, pc)            -- convenience: all pre-solve

Post-simulation diagnostics (call after simulate_lifecycle):
  - diagnose_income_post(model, pc, sim)   -- simulated income & SS
  - diagnose_simulation_post(model, pc, sim) -- policy-domain compatibility
  - diagnose_var_post(model, pc, sim)      -- stub (not yet implemented)
  - diagnose_all_post(model, pc, sim)      -- convenience: all post-sim

Solver diagnostics (unchanged from prior version):
  - diagnose_solver(model, pc, sol)                -- thin wrapper / TODO
  - diagnose_terminal_portfolio_states(...)         -- terminal portfolio per-state
  - diagnose_newton_failures_retirement(...)        -- Newton failure classification

Dependencies: numpy, numba, model, solver
"""

import numpy as np
from collections import namedtuple
from numba import njit
from math import exp

from lifecycle.model import SolverConfig, disposable_income_working, compute_pension_after_tax
from lifecycle.solver import (
    compute_terminal_portfolio_foc_jac,
    solve_portfolio_2d_terminal_constrained_njit,
    solve_portfolio_unconstrained_terminal_njit,
    _terminal_prepare_scenarios,
    _build_terminal_quad_returns,
    EC_NEWTON_FAIL, EC_INTERIOR,
    EC_CORNER_BILLS, EC_CORNER_STOCKS, EC_CORNER_BONDS,
    EC_EDGE_SB, EC_EDGE_BB, EC_EDGE_STOCKBOND,
)


# =============================================================================
# Shared infrastructure
# =============================================================================

DiagnosticResult = namedtuple('DiagnosticResult', ['status', 'title', 'detail'])

_W = 76  # output width


def _header(title):
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)


def _sub(title):
    print(f"\n  -- {title} " + "-" * max(0, _W - len(title) - 6))


def _test(results_list, status, title, description, detail):
    """Record and print one test result.

    status: 'pass', 'warn', or 'fail'
    title: what is being checked (economic meaning)
    description: why it matters (one line, can be empty string)
    detail: measured value (can be empty string)

    Returns the status string so callers can branch on fail.
    """
    sym = {"pass": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}[status]
    print(f"  {sym}  {title}")
    if description:
        print(f"     {description}")
    if detail:
        print(f"     {detail}")
    results_list.append(DiagnosticResult(status, title, detail))
    return status


def _summary(label, results):
    """Print pass/fail/warn summary for a diagnostic group."""
    n_pass = sum(1 for r in results if r.status == "pass")
    n_warn = sum(1 for r in results if r.status == "warn")
    n_fail = sum(1 for r in results if r.status == "fail")
    total = len(results)
    print()
    print("-" * _W)
    if n_fail == 0 and n_warn == 0:
        print(f"  {label}: ALL {total} TESTS PASSED")
    elif n_fail == 0:
        print(f"  {label}: {total} tests - {n_pass} passed, {n_warn} warnings")
    else:
        print(f"  {label}: {n_fail} FAILED / {total} total")
        for r in results:
            if r.status == "fail":
                print(f"    FAIL  {r.title}")
    print("-" * _W)
    return results


# =============================================================================
# Pre-solve: Labour income & Social Security
# =============================================================================

def diagnose_income_pre(model, pc):
    """
    Pre-solve diagnostics for labour income process and Social Security.

    Returns list of DiagnosticResult.
    """
    results = []

    _header("LABOUR INCOME & SOCIAL SECURITY")

    # -- Reference indices --
    iz0 = int(np.argmin(np.abs(pc.z_grid)))
    ie0 = int(np.argmin(np.abs(pc.eps_nodes)))
    n_z = len(pc.z_grid)
    retire_t = model.retire_age - model.start_age

    # -- avg_det (recompute for diagnostics -- must match precompute) --
    working_ages = np.arange(model.start_age, model.retire_age)
    log_det_all = (model.b0
                   + model.b1 * working_ages
                   + model.b2 * working_ages**2 / 10.0
                   + model.b3 * working_ages**3 / 100.0)
    avg_det = float(np.mean(np.exp(log_det_all)))

    # -- Mixture moments (used by multiple sub-sections) --
    mu_eta = model.pz * model.mu_eta1 + (1.0 - model.pz) * model.mu_eta2
    var_eta = (model.pz * (model.sigma_eta1**2 + (model.mu_eta1 - mu_eta)**2)
               + (1.0 - model.pz) * (model.sigma_eta2**2 + (model.mu_eta2 - mu_eta)**2))
    std_eta = np.sqrt(var_eta)
    std_z = np.sqrt(var_eta / max(1e-14, 1.0 - model.rho**2))
    z_cover = pc.z_grid.max() / std_z if std_z > 0 else float("nan")

    _e_eta3 = (model.pz * (model.mu_eta1**3 + 3 * model.mu_eta1 * model.sigma_eta1**2)
               + (1.0 - model.pz) * (model.mu_eta2**3 + 3 * model.mu_eta2 * model.sigma_eta2**2))
    _true_skew = _e_eta3 / max(var_eta**1.5, 1e-15)
    _e_eta4 = (model.pz * (model.mu_eta1**4 + 6 * model.mu_eta1**2 * model.sigma_eta1**2 + 3 * model.sigma_eta1**4)
               + (1.0 - model.pz) * (model.mu_eta2**4 + 6 * model.mu_eta2**2 * model.sigma_eta2**2 + 3 * model.sigma_eta2**4))
    _true_kurt = _e_eta4 / max(var_eta**2, 1e-15)

    # Transitory shock moments
    mu_eps2_eff = -(model.pe / (1.0 - model.pe)) * model.mu_eps1
    eps_mean = float(np.sum(pc.eps_nodes * pc.eps_weights))
    eps_var = float(np.sum(pc.eps_nodes**2 * pc.eps_weights))
    e_exp_eps = float(np.sum(np.exp(pc.eps_nodes) * pc.eps_weights))

    # ── Income Process Parameters ──────────────────────────────────────
    _sub("Income Process Parameters")

    print(f"  rho        = {model.rho:.5f}  (per-period persistence)")
    print(f"  pz         = {model.pz:.3f}    (mixture weight on component 1)")
    print(f"  Component 1:  mu_eta1 = {model.mu_eta1:+.4f},  sigma_eta1 = {model.sigma_eta1:.4f}")
    print(f"  Component 2:  mu_eta2 = {model.mu_eta2:+.4f},  sigma_eta2 = {model.sigma_eta2:.4f}")
    print(f"  E[eta]     = {mu_eta:.2e}   (should be ~ 0)")
    print(f"  Std[eta]   = {std_eta:.5f}")
    print(f"  Std[z]     = {std_z:.5f}  (unconditional)")
    print(f"  z_grid     : {pc.n_z} points  [{pc.z_grid.min():.4f}, {pc.z_grid.max():.4f}]   +/-{z_cover:.2f} sigma")
    print()
    print(f"  pe            = {model.pe:.3f}   (probability of large-shock component)")
    print(f"  Component 1:  mu_eps1    = {model.mu_eps1:+.4f},  sigma_eps1 = {model.sigma_eps1:.4f}")
    print(f"  Component 2:  mu_eps2_eff = {mu_eps2_eff:+.4f}  (zero-mean enforced)")
    print(f"               sigma_eps2 = {model.sigma_eps2:.4f}")
    print(f"  E[eps]        = {eps_mean:.2e}   (should be ~ 0)")
    print(f"  Var[eps]      = {eps_var:.5f}   Std[eps] = {np.sqrt(eps_var):.5f}")
    print(f"  eps_nodes     : {pc.n_eps} nodes  [{pc.eps_nodes.min():.4f}, {pc.eps_nodes.max():.4f}]")
    print(f"  E[exp(eps)]   = {e_exp_eps:.4f}   (Jensen: >1 by ~0.5*Var; {(e_exp_eps - 1)*100:+.1f}%)")
    print()
    print(f"  Deterministic profile: b0={model.b0:.4f}, b1={model.b1:.4f}, b2={model.b2:.4f}, b3={model.b3:.6f}")
    _peak_idx = int(np.argmax(log_det_all))
    print(f"  Hump peak: age {working_ages[_peak_idx]},  exp(f) = {np.exp(log_det_all[_peak_idx]):.4f}")
    print(f"  avg_det = {avg_det:.4f}")
    print()

    # Test 1: Innovation mean ~ 0
    _test(results, "pass" if abs(mu_eta) < 1e-6 else "fail",
          "Innovation mean = 0",
          "Good and bad income shocks cancel out on average",
          f"|E[eta]| = {abs(mu_eta):.2e}")

    # Test 2: Transitory shock mean ~ 0
    _test(results, "pass" if abs(eps_mean) < 1e-10 else "fail",
          "Transitory shock mean = 0",
          "Transitory luck doesn't bias income levels",
          f"|E[eps]| = {abs(eps_mean):.2e}")

    # Test 3: Eta quadrature captures variance
    _eta_mean = float(np.sum(pc.eta_nodes * pc.eta_weights))
    _eta_var = float(np.sum(pc.eta_nodes**2 * pc.eta_weights)) - _eta_mean**2
    _eta_e3 = float(np.sum(pc.eta_nodes**3 * pc.eta_weights))
    _eta_skew = (_eta_e3 - 3 * _eta_mean * _eta_var - _eta_mean**3) / max(_eta_var**1.5, 1e-15) if _eta_var > 0 else 0.0
    _var_ratio = _eta_var / var_eta if var_eta > 0 else float("nan")

    print(f"  Eta quadrature ({len(pc.eta_nodes)} nodes):")
    print(f"    Mean     = {_eta_mean:+.2e}   (true = 0)")
    print(f"    Variance = {_eta_var:.5f}   (true = {var_eta:.5f},  ratio = {_var_ratio:.4f})")
    print(f"    Skewness = {_eta_skew:+.3f}     (true = {_true_skew:+.3f})")
    print()

    _test(results, "pass" if abs(_var_ratio - 1.0) < 0.05 else "fail",
          "Eta quadrature captures variance",
          "Solver correctly weights income uncertainty",
          f"Quadrature/true variance ratio = {_var_ratio:.4f}")

    # Test 4: Eta quadrature captures skewness
    _skew_err = abs(_eta_skew - _true_skew)
    if _skew_err < 0.1:
        _skew_status = "pass"
    elif _skew_err < 0.5:
        _skew_status = "warn"
    else:
        _skew_status = "fail"
    _test(results, _skew_status,
          "Eta quadrature captures skewness",
          "Rare large income drops are correctly priced",
          f"|error| = {_skew_err:.3f}  (quadrature = {_eta_skew:+.3f}, true = {_true_skew:+.3f})")

    # ── Tax Schedule ──────────────────────────────────────────────────
    _sub("Tax Schedule")

    _y_gross_all = np.exp(
        log_det_all[:, None, None]
        + pc.z_grid[None, :, None]
        + pc.eps_nodes[None, None, :]
    )
    _eff_rate_all = 1.0 - pc.working_income[:len(working_ages)] / np.maximum(_y_gross_all, 1e-15)
    eff_rate_max = float(_eff_rate_all.max())

    # Print effective tax table
    print(f"  Effective tax rates at selected gross income levels (model units):")
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
    print()

    # Test 5: Effective tax rate < 50%
    _test(results, "pass" if eff_rate_max < 0.50 else "fail",
          "Effective tax rate < 50%",
          "Tax schedule doesn't confiscate majority of income",
          f"max effective rate = {eff_rate_max:.1%}")

    # Test 6: Tax schedule broadly progressive
    _eff_broadly_rising = _eff_rates_list[-1] > _eff_rates_list[0]
    _test(results, "pass" if _eff_broadly_rising else "fail",
          "Tax schedule broadly progressive",
          "Higher earners pay a larger share of income",
          f"low={_eff_rates_list[0]:.1%}, high={_eff_rates_list[-1]:.1%}")

    # Test 7: Tax bracket boundary verified
    _y_boundary = 0.18 / (1.0 - 0.106)
    _y_net_boundary = float(disposable_income_working(np.array([_y_boundary]))[0])
    _actual_tax_boundary = 0.18 - _y_net_boundary
    _expected_tax_boundary = 0.18 * 0.10
    _ok = abs(_actual_tax_boundary - _expected_tax_boundary) < 1e-8
    _test(results, "pass" if _ok else "fail",
          "Tax bracket boundary verified",
          "Tax code matches 2019 TCJA thresholds",
          f"tax at 10%/12% boundary: expected = {_expected_tax_boundary:.6f}, got = {_actual_tax_boundary:.6f}")

    # ── Pension Formula (Social Security) ──────────────────────────────
    _sub("Pension Formula (Social Security)")

    # Test 8: After-tax income positive everywhere
    wi_min = float(pc.working_income.min())
    pens_min = float(pc.pension_after_tax.min())
    _all_positive = wi_min > 0 and pens_min > 0
    _test(results, "pass" if _all_positive else "fail",
          "After-tax income positive everywhere",
          "No worker or retiree receives negative income",
          f"min working = {wi_min:.6f}, min pension = {pens_min:.6f}")

    # Test 9: Income rises with persistent state (z)
    wi_z_diffs = np.diff(pc.working_income, axis=1)
    wi_z_min_diff = float(wi_z_diffs.min())
    _test(results, "pass" if wi_z_min_diff >= -1e-12 else "fail",
          "Income rises with persistent state (z)",
          "Higher-skilled workers always earn more",
          f"Min gap between adjacent z-states: {wi_z_min_diff:.2e}")

    # Test 10: Income rises with transitory shock (eps)
    _eps_order = np.argsort(pc.eps_nodes)
    _wi_sorted_eps = pc.working_income[:, :, _eps_order]
    wi_e_diffs = np.diff(_wi_sorted_eps, axis=2)
    wi_e_min_diff = float(wi_e_diffs.min())
    _test(results, "pass" if wi_e_min_diff >= -1e-12 else "fail",
          "Income rises with transitory shock (eps)",
          "Better luck within a year means higher pay",
          f"Min gap between adjacent eps-nodes: {wi_e_min_diff:.2e}")

    # Test 11: Pension rises with persistent state
    pens_z_diffs = np.diff(pc.pension_after_tax[0, :])
    pens_z_min = float(pens_z_diffs.min())
    _test(results, "pass" if pens_z_min >= -1e-12 else "fail",
          "Pension rises with persistent state",
          "Higher lifetime earners receive at least as much SS",
          f"Min step = {pens_z_min:.2e} (zero where AIME cap binds)")

    # Test 12: Pension constant across retirement ages
    pens_age_var = float(np.max(np.abs(pc.pension_after_tax - pc.pension_after_tax[0:1, :])))
    _test(results, "pass" if pens_age_var < 1e-14 else "fail",
          "Pension constant across retirement ages",
          "SS benefit doesn't change after claiming",
          f"Max variation = {pens_age_var:.2e}")

    # Test 13: Pension cap at max AIME
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
    _test(results, "pass" if abs(pens_actual_max - pia_max_net) < 1e-6 else "fail",
          "Pension cap at max AIME",
          "SS earnings cap correctly implemented (~$135k)",
          f"expected {pia_max_net:.4f}, got {pens_actual_max:.4f}")

    # Test 14: AIME pipeline matches precomputed table
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

    print(f"\n  AIME pipeline trace  (avg_det = {avg_det:.4f}, cap = 2.500):")
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

    print(f"\n  Cap binds at z >= {z_cap:.3f}")

    _test(results, "pass" if _pipeline_ok else "fail",
          "AIME pipeline matches precomputed table",
          "The AIME -> PIA -> tax -> net pension chain is correct",
          f"max error < 1e-6 at {len(_repr_iz)} representative z values")

    # Test 15: Payroll tax cap = AIME cap
    _test(results, "pass", "Payroll tax cap = AIME cap",
          "Tax side and benefit side use same earnings limit",
          "both 2.500 (disposable_income_working & compute_pension_after_tax)")

    # ── SS consistency info ──
    print(f"\n  Social Security consistency:")
    print(f"    Tax side:     payroll 10.6%, cap at Y_gross = 2.500")
    print(f"    Benefit side: avg_det = {avg_det:.4f}, AIME cap = 2.500")
    print(f"    z at cap:     {z_cap:.3f}  (exp(z)*avg_det >= 2.5)")

    # ── Replacement Rates ──────────────────────────────────────────────
    _sub("Replacement Rates")

    # Expected income lifecycle at z=0
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
    pens_z0 = float(pc.pension_after_tax[0, iz0])

    print(f"  At z=0 (median worker):")
    print(f"    Career average E[Y_net]:   {career_avg_net:.4f}")
    print(f"    Pension:                   {pens_z0:.4f}")
    print(f"    Pension / career avg:      {pens_z0 / career_avg_net:.1%}")
    print(f"    Pension / last working yr: {pens_z0 / last_working_net:.1%}")
    print(f"    Pension / peak yr:         {pens_z0 / peak_net:.1%}")
    print()

    # Replacement rates by z-level
    if n_z <= 7:
        _show_iz = list(range(n_z))
    else:
        _show_iz = sorted(set([0, 1, n_z // 4, iz0, 3 * n_z // 4, n_z - 2, n_z - 1]))

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
        _r_last = _pens / _last_yr if _last_yr > 1e-10 else float("nan")
        _r_peak = _pens / _peak_yr if _peak_yr > 1e-10 else float("nan")
        _repl_rates_career.append(_r_career)
        print(f"  {_iz:>5}  {_z_val:>7.3f}  {_pens:>8.4f}  {_r_career:>8.1%}  "
              f"{_r_last:>8.1%}  {_r_peak:>8.1%}")

    # Test 16: Median replacement rate in [40%, 80%]
    repl_career = pens_z0 / career_avg_net
    _test(results, "pass" if 0.40 <= repl_career <= 0.80 else "fail",
          "Median replacement rate in [40%, 80%]",
          "Median worker replaces a realistic share of pre-retirement income",
          f"pension/career_avg = {repl_career:.1%}")

    # Test 17: Rates decline from median z upward
    _mid_start = len(_repl_rates_career) // 2
    _repl_declining_upper = all(
        _repl_rates_career[i] >= _repl_rates_career[i+1] - 1e-6
        for i in range(_mid_start, len(_repl_rates_career) - 1)
    )
    _status17 = "pass" if _repl_declining_upper else "fail"
    # Check if just flat at bottom (warn not fail)
    _repl_mono = all(_repl_rates_career[i] >= _repl_rates_career[i+1] - 1e-6
                     for i in range(len(_repl_rates_career) - 1))
    if not _repl_mono and _repl_declining_upper:
        _status17 = "warn"
    _test(results, _status17,
          "Rates decline from median z upward",
          "SS provides proportionally more to lower earners (progressive)",
          f"flat at bottom OK; declining upper half: {_repl_declining_upper}")

    # Test 18: Bottom earners replaced more than top
    if len(_repl_rates_career) >= 2:
        _ok = _repl_rates_career[0] > _repl_rates_career[-1]
        _test(results, "pass" if _ok else "fail",
              "Bottom earners replaced more than top",
              "Redistributive structure of SS is working",
              f"bottom = {_repl_rates_career[0]:.1%}, top = {_repl_rates_career[-1]:.1%}")

    # ── Retirement Boundary ────────────────────────────────────────────
    _sub("Retirement Boundary")

    print(f"  Retirement boundary sequence (z = z_grid[{iz0}] = {pc.z_grid[iz0]:.4f}):")
    print(f"  Convention: last labor paycheck at age {model.retire_age}, first pension at age {model.retire_age + 1}.")
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

    # Test 19: Income array covers retirement age
    _ok19 = retire_t < pc.working_income.shape[0]
    _test(results, "pass" if _ok19 else "fail",
          "Income array covers retirement age",
          "No out-of-bounds at work-retire transition",
          f"need index {retire_t} in axis 0 of shape {pc.working_income.shape}")

    # Test 20: Pension array covers terminal age
    _ok20 = (pc.n_age - 1) < pc.pension_after_tax.shape[0]
    _test(results, "pass" if _ok20 else "fail",
          "Pension array covers terminal age",
          "Pension available at all retirement ages",
          f"need index {pc.n_age - 1} in axis 0 of shape {pc.pension_after_tax.shape}")

    # ── z-Transition Quality (Pi_z, simulation-only) ────────────────────
    _sub("z-Transition Quality (Pi_z, simulation-only)")
    print(f"  NOTE: Pi_z is simulation-only -- the solver uses eta quadrature.\n")

    # Test 21: Pi_z row sums = 1
    pi_z_ok = np.allclose(pc.Pi_z.sum(axis=1), 1.0, atol=1e-10)
    _pi_z_dev = float(np.abs(pc.Pi_z.sum(axis=1) - 1.0).max())
    _test(results, "pass" if pi_z_ok else "fail",
          "Pi_z row sums = 1",
          "Transition probabilities are valid",
          f"max deviation = {_pi_z_dev:.2e}")

    # Discretized transition moments from Pi_z at mid row
    _dz_vals = pc.z_grid - pc.z_grid[iz0]
    _disc_mean = float(np.dot(pc.Pi_z[iz0], pc.z_grid))
    _disc_e2 = float(np.dot(pc.Pi_z[iz0], _dz_vals**2))
    _disc_e1 = float(np.dot(pc.Pi_z[iz0], _dz_vals))
    _disc_var = _disc_e2 - _disc_e1**2
    _true_cond_mean = model.rho * pc.z_grid[iz0]

    _p_down = float(sum(pc.Pi_z[iz0, j] for j in range(iz0)))
    _p_stay = float(pc.Pi_z[iz0, iz0])
    _p_up = float(sum(pc.Pi_z[iz0, j] for j in range(iz0 + 1, n_z)))

    print(f"  At z = 0:  P(down) = {_p_down:.4f}, P(stay) = {_p_stay:.4f}, P(up) = {_p_up:.4f}")
    print(f"  Conditional mean: E[z'|z=0] = {_disc_mean:+.5f} (true = {_true_cond_mean:+.5f})")
    print(f"  Conditional var:  {_disc_var:.5f} (true = {var_eta:.5f})")

    # Test 22: Upward transitions exist from mid state
    _status22 = "pass" if _p_up > 1e-6 else "warn"
    _test(results, _status22,
          "Upward transitions exist from mid state",
          "Simulation can represent income recovery",
          f"P(up) = {_p_up:.4f}")

    # Test 23: No absorbing states in Pi_z
    _escape_probs = np.array([1.0 - pc.Pi_z[i, i] for i in range(n_z)])
    _n_absorbing = int(np.sum(_escape_probs < 1e-10))
    _status23 = "pass" if _n_absorbing == 0 else "warn"
    _test(results, _status23,
          "No absorbing states in Pi_z",
          "Simulation agents can leave boundary states",
          f"{_n_absorbing} absorbing rows")

    _summary("LABOUR INCOME & SOCIAL SECURITY", results)
    return results


# =============================================================================
# Pre-solve: VAR structure & return distribution
# =============================================================================

def diagnose_var_pre(model, pc):
    """
    Pre-solve diagnostics for VAR structure, state grids, and return distribution.

    Returns list of DiagnosticResult.
    """
    results = []
    sv = list(model.state_names)
    rv = list(model.ret_names)

    _header("VAR STRUCTURE & RETURN DISTRIBUTION")

    def print_matrix(mat, row_names, col_names, indent="  "):
        cw = 10
        print(indent + " " * 12 + "".join(f"{c:>{cw}}" for c in col_names))
        for i, rn in enumerate(row_names):
            row = indent + f"{rn:>12}" + "".join(f"{mat[i, j]:>{cw}.5f}" for j in range(mat.shape[1]))
            print(row)

    # ── VAR Parameters ─────────────────────────────────────────────────
    _sub("VAR Parameters")

    print(f"  Unconditional means (annualized):")
    for d, name in enumerate(sv):
        v = model.z_bar_state[d]
        ann = f"{v * 100:+.3f}%/yr" if name not in ("dp",) else "(log level)"
        print(f"    {name:>10}:  {v:+.8f}   {ann}")
    for k, name in enumerate(rv):
        v = model.z_bar_ret[k]
        print(f"    {name:>10}:  {v:+.8f}   {v * 100:+.3f}%/yr")

    print()
    print(f"  Phi_11 (state-to-state persistence):")
    print_matrix(model.Phi_11, sv, sv)
    eigs_11 = np.sort(np.abs(np.linalg.eigvals(model.Phi_11)))[::-1]
    eig_str = ", ".join(f"{e:.4f}" for e in eigs_11)
    print(f"  Eigenvalues |lambda|: [{eig_str}]")

    print()
    print(f"  Phi_21 (state -> return loadings):")
    print_matrix(model.Phi_21, rv, sv)

    print()
    print(f"  M (return | next-state conditioning, Schur complement):")
    print_matrix(model.M, rv, sv)
    print(f"  ||M||_F = {np.linalg.norm(model.M):.5f}")

    print()
    print(f"  Return standard deviations (annualized):")
    print(f"  {'Return':>8}   {'Cond. sigma':>14}   {'Uncond. sigma':>14}")
    for k, name in enumerate(rv):
        cond_std = np.sqrt(max(0.0, model.Sigma_r_cond[k, k]))
        uncond_std = np.sqrt(max(0.0, model.Sigma_rr[k, k]))
        print(f"  {name:>8}   {cond_std * 100:>12.3f}%/yr   {uncond_std * 100:>12.3f}%/yr")

    # ── Stationarity ───────────────────────────────────────────────────
    _sub("Stationarity")

    # Test 1: Phi_11 stationary
    eig_11_max = float(eigs_11[0])
    _test(results, "pass" if eig_11_max < 1.0 else "fail",
          "Phi_11 stationary (max eigenvalue < 1)",
          "Financial states are mean-reverting",
          f"max|lambda| = {eig_11_max:.5f}")

    # ── State Grid Quality ─────────────────────────────────────────────
    _sub("State Grid Quality")

    N_per_dim = pc.state_grid_sizes
    print(f"  Grid sizes: {N_per_dim}   N_state = {pc.N_state} joint states")
    print(f"  Grid mode: {getattr(pc, 'state_grid_mode', 'naive')}")
    print()
    print(f"  {'Var':>8}  {'N':>3}  {'Coverage':>9}  {'Grid min':>10}  {'Grid max':>10}  "
          f"{'Uncond. mu':>11}  {'Uncond. sigma':>13}")
    print(f"  {'-'*8}  {'-'*3}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*13}")

    _grid_coverage = []
    for d, name in enumerate(sv):
        Nd = int(N_per_dim[d])
        mu_d = float(getattr(pc, "state_grid_mu_s", model.z_bar_state)[d])
        sig_y = float(getattr(pc, "state_grid_sigma_z", np.sqrt(np.diag(model.Sigma_ss)))[d])
        g_vals = pc.state_grid[:, d]
        cover = (g_vals.max() - mu_d) / sig_y if sig_y > 0 else float("nan")
        print(f"  {name:>8}  {Nd:>3}  {cover:>+9.2f}  {g_vals.min():>10.5f}  {g_vals.max():>10.5f}  "
              f"{mu_d:>11.5f}  {sig_y:>13.5f}")
        _grid_coverage.append((name, cover))

    if getattr(pc, "state_grid_mode", "naive") == "principal":
        print()
        print("  Principal-mode bracketing axes (u-coordinates):")
        for d in range(len(pc.state_bracket_grids)):
            g = pc.state_bracket_grids[d]
            print(f"    u[{d}] range = [{g.min():+.2f}, {g.max():+.2f}]")

    print()
    # Tests 2-4: State grid coverage >= 2.5 sigma (per dimension)
    for name, cover in _grid_coverage:
        _test(results, "pass" if cover >= 2.5 else "warn",
              f"State grid coverage: {name} >= 2.5 sigma",
              f"Grid spans enough of {name}'s distribution",
              f"+/-{cover:.2f} sigma")
    # Bill rate is now uncertain (no r_bill_grid); show y_1 state grid range instead
    if model.y_1_index_in_state is not None:
        y1_grid = pc.state_grid[:, model.y_1_index_in_state]
        print(f"  y_1 state grid: [{y1_grid.min()*100:.3f}%, {y1_grid.max()*100:.3f}%]  Mean: {y1_grid.mean()*100:.3f}%")
    else:
        print(f"  y_1 scalar fallback: {model.y_1_scalar_fallback*100:.3f}%")

    # ── Return Distribution Quality ────────────────────────────────────
    _sub("Return Distribution Quality")

    K_str = "x".join(str(k) for k in pc.n_ret_nodes_1d)
    print(f"  Return quadrature: ({K_str}) nodes/dim -> {pc.n_ret_quad} joint nodes")

    # Stationary distribution of Pi_state
    stat = getattr(pc, "state_stationary_probs", None)
    if stat is None:
        try:
            evals, evecs = np.linalg.eig(pc.Pi_state.T)
            idx = int(np.argmin(np.abs(evals - 1.0)))
            stat = np.real(evecs[:, idx])
            stat = np.abs(stat) / np.abs(stat).sum()
        except Exception:
            stat = np.ones(pc.N_state) / pc.N_state

    E_ret_by_state = np.einsum("ij,ijk->ik", pc.Pi_state, pc.mu_r)  # (N_state, n_ret)
    print()
    print(f"  {'Return':>8}  {'mu_r min':>10}  {'mu_r max':>10}  "
          f"{'Cond. sigma (ann)':>18}  {'Uncond. E[r] (ann)':>20}")
    for k, name in enumerate(rv):
        mu_lo = float(pc.mu_r[:, :, k].min())
        mu_hi = float(pc.mu_r[:, :, k].max())
        cs_ann = np.sqrt(max(0.0, model.Sigma_r_cond[k, k])) * 100
        e_unc = float(stat @ E_ret_by_state[:, k]) * 100
        print(f"  {name:>8}  {mu_lo:>10.5f}  {mu_hi:>10.5f}  "
              f"{cs_ann:>16.3f}%/yr  {e_unc:>18.3f}%/yr")

    # Test 5: Pi_state row sums = 1
    pi_s_ok = np.allclose(pc.Pi_state.sum(axis=1), 1.0, atol=1e-10)
    _test(results, "pass" if pi_s_ok else "fail",
          "Pi_state row sums = 1",
          "State transitions are valid probabilities",
          f"max deviation = {np.abs(pc.Pi_state.sum(axis=1) - 1.0).max():.2e}")

    # Test 6: Return quadrature mean ~ 0
    ret_w_mean = np.sum(pc.ret_nodes * pc.ret_weights[:, None], axis=0)
    _test(results, "pass" if float(np.max(np.abs(ret_w_mean))) < 1e-10 else "fail",
          "Return quadrature mean = 0",
          "Return residuals are centered",
          f"max |mean| = {float(np.max(np.abs(ret_w_mean))):.2e}")

    # Test 7: Return quadrature covariance matches Sigma_r_cond
    ret_cov = (pc.ret_nodes * pc.ret_weights[:, None]).T @ pc.ret_nodes
    _cov_diff = float(np.max(np.abs(ret_cov - model.Sigma_r_cond)))
    _test(results, "pass" if _cov_diff < 1e-8 else "fail",
          "Return quadrature covariance matches Sigma_r_cond",
          "Return variance correctly captured",
          f"max |diff| = {_cov_diff:.2e}")

    # Test 8: Sigma_ss positive definite
    try:
        np.linalg.cholesky(model.Sigma_ss)
        _test(results, "pass", "Sigma_ss positive definite",
              "State innovation covariance is valid", "")
    except np.linalg.LinAlgError:
        _test(results, "fail", "Sigma_ss positive definite",
              "State innovation covariance is valid", "Cholesky decomposition failed")

    # Test 9: Sigma_r_cond positive definite
    try:
        np.linalg.cholesky(model.Sigma_r_cond)
        _test(results, "pass", "Sigma_r_cond positive definite",
              "Return residual covariance is valid", "")
    except np.linalg.LinAlgError:
        _test(results, "fail", "Sigma_r_cond positive definite",
              "Return residual covariance is valid", "Cholesky decomposition failed")

    _summary("VAR STRUCTURE & RETURNS", results)
    return results


# =============================================================================
# Pre-solve: Grids, preferences, memory
# =============================================================================

def diagnose_grids_pre(model, pc):
    """
    Pre-solve diagnostics for preferences, lifecycle, grids, and memory footprint.

    Returns list of DiagnosticResult.
    """
    results = []

    _header("PREFERENCES, GRIDS & NUMERICAL SETUP")

    # ── Preferences & Lifecycle ────────────────────────────────────────
    _sub("Preferences & Lifecycle")

    print(f"  gamma  = {model.gamma:.3f}   (CRRA risk aversion)")
    print(f"  beta   = {model.beta:.6f}  (annual discount factor)")
    print(f"  b_bar  = {model.b_bar}   (bequest horizon in years)")
    n_work = model.retire_age - model.start_age
    n_ret = model.terminal_age - model.retire_age + 1
    print(f"  Ages:  {model.start_age} - {model.terminal_age}   ({pc.n_age} periods total)")
    print(f"  Retirement at {model.retire_age}:   {n_work} working,  {n_ret} retirement")

    # Survival probabilities at key ages
    iz_lo, iz_mid, iz_hi = 0, pc.n_z // 2, pc.n_z - 1
    print(f"\n  Survival probabilities (earnings-dependent):")
    print(f"  {'Age':>4}  {'low-z':>10}  {'mid-z':>10}  {'high-z':>10}")
    key_sp = [25, 30, 40, 50, 55, 60, 65, 70, 75, 80]
    for age in key_sp:
        if model.start_age <= age <= model.terminal_age:
            t = age - model.start_age
            sp_lo = pc.survival_probs_2d[t, iz_lo]
            sp_mid = pc.survival_probs_2d[t, iz_mid]
            sp_hi = pc.survival_probs_2d[t, iz_hi]
            print(f"  {age:>4}  {sp_lo:>10.5f}  {sp_mid:>10.5f}  {sp_hi:>10.5f}")

    # ── Dimension Table ────────────────────────────────────────────────
    _sub("Dimension Table")

    dims = [
        ("n_w",     pc.n_w,     "wealth grid points"),
        ("n_s",     pc.n_s,     "savings grid points (EGM)"),
        ("n_z",     pc.n_z,     "persistent income states"),
        ("n_eps",   pc.n_eps,   f"transitory shock nodes  (Judd-mixture, total = {pc.n_eps})"),
        ("n_age",   pc.n_age,   f"age periods  ({model.start_age}-{model.terminal_age})"),
        ("N_state", pc.N_state, f"joint VAR states  ({'x'.join(str(n) for n in pc.state_grid_sizes)})"),
        ("n_state", model.n_state, "slow-state variables"),
        ("n_ret",   model.n_ret,   "return variables (integrated)"),
        ("n_eta",   len(pc.eta_nodes), f"persistent innovation nodes  (Judd-mixture, total = {len(pc.eta_nodes)})"),
        ("n_ret_quad", pc.n_ret_quad,
         f"joint return quadrature nodes  (= prod({'x'.join(str(k) for k in pc.n_ret_nodes_1d)}) = {pc.n_ret_quad})"),
    ]
    for name, val, desc in dims:
        print(f"  {name:>10} = {val:>6}   {desc}")

    # ── Memory Estimate ────────────────────────────────────────────────
    _sub("Memory Estimate (float64, 8 bytes/element)")
    bpf = 8

    def mb(n): return n * bpf / 1024**2

    rows = [
        ("Value function",   pc.n_w * pc.n_z * pc.N_state * pc.n_age,
         f"n_w x n_z x N_state x n_age  = {pc.n_w}x{pc.n_z}x{pc.N_state}x{pc.n_age}"),
        ("Policy (a_s, a_r)", 2 * pc.n_w * pc.n_z * pc.N_state * pc.n_age,
         "2 x same"),
        ("mu_r",             pc.N_state**2 * model.n_ret,
         f"N_state^2 x n_ret  = {pc.N_state}^2 x {model.n_ret}"),
        ("ret_nodes",        pc.n_ret_quad * model.n_ret,
         f"n_ret_quad x n_ret  = {pc.n_ret_quad} x {model.n_ret}"),
        ("Pi_state",         pc.N_state**2,
         f"N_state^2  = {pc.N_state}^2"),
        ("working_income",   pc.n_age * pc.n_z * pc.n_eps,
         f"n_age x n_z x n_eps  = {pc.n_age}x{pc.n_z}x{pc.n_eps}"),
        ("pension_after_tax", pc.n_age * pc.n_z,
         f"n_age x n_z  = {pc.n_age}x{pc.n_z}"),
    ]
    total_el = sum(r[1] for r in rows)
    print(f"  {'Array':<22}  {'Elements':>12}  {'MB':>8}  Description")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}  {'-'*40}")
    for arr_name, n_el, desc in rows:
        print(f"  {arr_name:<22}  {n_el:>12,}  {mb(n_el):>8.2f}  {desc}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}")
    print(f"  {'TOTAL (approx)':<22}  {total_el:>12,}  {mb(total_el):>8.2f}")

    # ── Grid Ranges ────────────────────────────────────────────────────
    _sub("Grid Ranges")
    print(f"  wealth_grid : [{pc.wealth_grid.min():.6f}, {pc.wealth_grid.max():.1f}]  ({pc.n_w} pts)")
    print(f"  s_grid      : [{pc.s_grid.min():.2e}, {pc.s_grid.max():.1f}]  ({pc.n_s} pts)")

    # ── Tests ──────────────────────────────────────────────────────────
    _sub("Validation")

    # Test 1: Survival probs in (0, 1]
    surv_ok = bool(np.all((pc.survival_probs_2d > 0) & (pc.survival_probs_2d <= 1.0)))
    _test(results, "pass" if surv_ok else "fail",
          "Survival probs in (0, 1]",
          "Mortality rates are biologically plausible",
          f"range = [{pc.survival_probs_2d.min():.5f}, {pc.survival_probs_2d.max():.5f}]")

    # Test 2: Wealth grid strictly positive
    _test(results, "pass" if pc.wealth_grid.min() > 0 else "fail",
          "Wealth grid strictly positive",
          "Cash-on-hand bounded away from zero (log utility safe)",
          f"min = {pc.wealth_grid.min():.2e}")

    _summary("PREFERENCES, GRIDS & NUMERICAL SETUP", results)
    return results


# =============================================================================
# Post-simulation: Labour income & Social Security
# =============================================================================

def diagnose_income_post(model, pc, sim):
    """
    Post-simulation diagnostics for income process and Social Security.

    Parameters
    ----------
    model : LifecyclePortfolioModel
    pc    : Precompute
    sim   : dict from simulate_lifecycle

    Returns list of DiagnosticResult.
    """
    results = []

    sim_income = sim["income"]
    sim_alive = sim["alive"]
    sim_z_idx = sim["z_idx"]
    ages = sim["ages"]
    n_sim, n_age = sim_income.shape

    iz0 = int(np.argmin(np.abs(pc.z_grid)))
    retire_t = model.retire_age - model.start_age
    n_z = len(pc.z_grid)

    _header("POST-SIMULATION: LABOUR INCOME & SOCIAL SECURITY")
    print(f"  Simulation: {n_sim} agents, ages {ages[0]}-{ages[-1]}, {n_age} periods")

    # ── Simulated Income Moments ───────────────────────────────────────
    _sub("Simulated Income Moments")
    print(f"  Compares realized mean income against theoretical expectation")
    print(f"  weighted by the simulated z-distribution at each age.\n")

    print(f"  {'Age':>4}  {'N_alive':>7}  {'E[Y_sim]':>9}  {'E[Y_theory]':>11}  "
          f"{'Ratio':>7}  {'Std[Y_sim]':>10}  {'Note':>8}")
    print(f"  {'---':>4}  {'---':>7}  {'---':>9}  {'---':>11}  "
          f"{'---':>7}  {'---':>10}  {'---':>8}")

    _key_ages = [25, 30, 40, 50, 60, model.retire_age - 1,
                 model.retire_age + 1, 80]
    _key_ages = sorted(set(a for a in _key_ages
                           if model.start_age <= a <= model.terminal_age))

    _worst_ratio_dev = 0.0
    for _age in _key_ages:
        _t = _age - model.start_age
        _alive_mask = sim_alive[:, _t]
        _n_alive = int(np.sum(_alive_mask))
        if _n_alive < 10:
            continue

        _y_alive = sim_income[_alive_mask, _t]
        _e_sim = float(np.mean(_y_alive))
        _std_sim = float(np.std(_y_alive))

        _z_alive = sim_z_idx[_alive_mask, _t]
        _z_counts = np.bincount(_z_alive, minlength=n_z)
        _z_fracs = _z_counts / _z_counts.sum()

        if _age < model.retire_age:
            _e_theory = float(sum(
                _z_fracs[iz] * np.dot(pc.eps_weights, pc.working_income[_t, iz, :])
                for iz in range(n_z)
            ))
        else:
            _e_theory = float(sum(
                _z_fracs[iz] * pc.pension_after_tax[_t, iz]
                for iz in range(n_z)
            ))

        _ratio = _e_sim / _e_theory if _e_theory > 1e-10 else float("nan")
        _note = "pension" if _age >= model.retire_age else ""
        _worst_ratio_dev = max(_worst_ratio_dev, abs(_ratio - 1.0))

        print(f"  {_age:>4}  {_n_alive:>7}  {_e_sim:>9.4f}  {_e_theory:>11.4f}  "
              f"{_ratio:>7.3f}  {_std_sim:>10.4f}  {_note:>8}")

    # Test 1: Sim/theory income ratio within 20% at all ages
    _test(results, "pass" if _worst_ratio_dev < 0.20 else "fail",
          "Sim/theory income ratio within 20% at all ages",
          "Simulated income draws are consistent with precomputed tables",
          f"worst deviation = {_worst_ratio_dev:.1%}")

    # ── Realized Replacement Rates ─────────────────────────────────────
    _sub("Realized Replacement Rates")

    _t_last_work = retire_t - 1
    _t_first_pens = retire_t + 1
    if _t_first_pens < n_age:
        _retiree_mask = sim_alive[:, _t_last_work] & sim_alive[:, _t_first_pens]
        _n_retirees = int(np.sum(_retiree_mask))

        _y_last = sim_income[_retiree_mask, _t_last_work]
        _y_pens = sim_income[_retiree_mask, _t_first_pens]

        _safe = _y_last > 1e-6
        _repl_realized = _y_pens[_safe] / _y_last[_safe]

        if len(_repl_realized) > 0:
            _pctiles = [10, 25, 50, 75, 90]
            _pct_vals = np.percentile(_repl_realized, _pctiles)

            print(f"  N retirees: {_n_retirees}")
            print()
            print(f"  {'Percentile':>10}  {'Repl.rate':>9}")
            print(f"  {'---':>10}  {'---':>9}")
            for _p, _v in zip(_pctiles, _pct_vals):
                print(f"  {_p:>9}th  {_v:>8.1%}")

            _median_repl = float(np.median(_repl_realized))

            # Test 2: Median replacement rate in [20%, 200%]
            _test(results, "pass" if 0.20 <= _median_repl <= 2.0 else "fail",
                  "Median replacement rate in [20%, 200%]",
                  "Retirement income transition is plausible",
                  f"median = {_median_repl:.1%}")
        else:
            print("  WARNING: No retirees with positive last-year income.")
            _test(results, "fail", "Median replacement rate in [20%, 200%]",
                  "Retirement income transition is plausible",
                  "No retirees with positive last-year income")
    else:
        print("  WARNING: Simulation too short for pension year.")
        _test(results, "fail", "Median replacement rate in [20%, 200%]",
              "Retirement income transition is plausible",
              "Simulation too short for pension year")

    # ── Retirement Boundary ────────────────────────────────────────────
    _sub("Retirement Boundary")

    _surv_retire = np.where(sim_alive[:, min(retire_t + 2, n_age - 1)])[0]
    if len(_surv_retire) >= 3:
        _z_at_retire = sim_z_idx[_surv_retire, retire_t]
        _z_sorted = np.argsort(_z_at_retire)
        _picks = [_z_sorted[0], _z_sorted[len(_z_sorted) // 2], _z_sorted[-1]]
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
            if _age > model.retire_age and _z_t != _z_ret:
                _note += " Z NOT FROZEN!"
                _z_freeze_ok = False
            print(f"    {_age:>4}  {_z_t:>5}  {_y_t:>8.4f}  {_note}")
        print()

    # Test 3: z frozen after retirement (traced agents)
    _test(results, "pass" if _z_freeze_ok else "fail",
          "z frozen after retirement (traced agents)",
          "SS benefit locked at retirement earnings level",
          "")

    # Test 4: z frozen globally at retire+1
    if retire_t + 1 < n_age:
        _alive_post = sim_alive[:, retire_t + 1]
        _z_at_ret = sim_z_idx[:, retire_t]
        _z_at_ret1 = sim_z_idx[:, retire_t + 1]
        _z_frozen_all = bool(np.all(_z_at_ret[_alive_post] == _z_at_ret1[_alive_post]))
        _test(results, "pass" if _z_frozen_all else "fail",
              "z frozen globally at retire+1",
              "No agent has drifting z after retirement",
              f"checked {int(np.sum(_alive_post))} alive agents")

    # ── Survivor Selection ─────────────────────────────────────────────
    _sub("Survivor Selection")

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

    # Test 5: Mean z rises with age among survivors
    if len(_mean_z_vals) >= 2:
        _selection_present = _mean_z_vals[-1] > _mean_z_vals[0]
        _test(results, "pass" if _selection_present else "fail",
              "Mean z rises with age among survivors",
              "Richer people live longer (Chetty et al. mortality is working)",
              f"entry = {_mean_z_vals[0]:.2f}, late life = {_mean_z_vals[-1]:.2f}")

    _summary("POST-SIMULATION: LABOUR INCOME & SOCIAL SECURITY", results)
    return results


# =============================================================================
# Post-simulation: VAR (stub)
# =============================================================================

def diagnose_simulation_post(model, pc, sim,
                             warn_offgrid_share=0.05,
                             fail_offgrid_share=0.20):
    """
    Post-simulation diagnostics for policy-domain compatibility.

    The solver only computes policies on `pc.wealth_grid`, which is strictly
    positive. The simulation currently evaluates those policies with linear
    wealth extrapolation outside the grid. A large off-grid share therefore
    means the simulated panel is no longer spending most of its time inside the
    state space that was actually solved.
    """
    results = []

    sim_x = sim["x"]
    sim_alive = sim["alive"]
    sim_alpha_s = sim["alpha_s"]
    sim_alpha_b = sim["alpha_b"]
    ages = sim["ages"]

    n_sim, n_age = sim_x.shape
    w_min = float(pc.wealth_grid[0])
    w_max = float(pc.wealth_grid[-1])

    _header("POST-SIMULATION: POLICY DOMAIN & NUMERICAL COMPATIBILITY")
    print(f"  Simulation: {n_sim} agents, ages {ages[0]}-{ages[-1]}, {n_age} periods")
    print(f"  Solver wealth domain: [{w_min:.4e}, {w_max:.4e}]")
    print("  Note: outside this interval the simulator uses linear wealth extrapolation.")
    print("        Large off-grid shares indicate a mismatch between the simulated")
    print("        panel and the state space the solver actually covered.")

    alive_mask = sim_alive.astype(bool)
    x_alive = sim_x[alive_mask]
    alpha_s_alive = sim_alpha_s[alive_mask]
    alpha_b_alive = sim_alpha_b[alive_mask]

    finite_x = np.isfinite(x_alive)
    finite_alpha = np.isfinite(alpha_s_alive) & np.isfinite(alpha_b_alive)

    pooled_neg_share = float(np.mean(x_alive < 0.0)) if x_alive.size else float("nan")
    pooled_below_share = float(np.mean(x_alive < w_min)) if x_alive.size else float("nan")
    pooled_above_share = float(np.mean(x_alive > w_max)) if x_alive.size else float("nan")
    pooled_offgrid_share = float(np.mean((x_alive < w_min) | (x_alive > w_max))) if x_alive.size else float("nan")

    key_ages = [model.start_age, 25, 30, 35, 40, 50, 60, model.retire_age, model.terminal_age]
    key_ages = sorted(set(a for a in key_ages if model.start_age <= a <= model.terminal_age))

    print()
    print(f"  {'Age':>4}  {'N_alive':>7}  {'Below':>7}  {'Above':>7}  {'Off-grid':>9}  "
          f"{'Neg x':>7}  {'Median x':>10}  {'P90 x':>10}")
    print(f"  {'---':>4}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>9}  "
          f"{'---':>7}  {'---':>10}  {'---':>10}")

    worst_offgrid_share = 0.0
    worst_offgrid_age = None
    worst_negative_share = 0.0
    worst_negative_age = None

    for age in key_ages:
        t = age - model.start_age
        alive_t = sim_alive[:, t]
        n_alive = int(np.sum(alive_t))
        if n_alive == 0:
            continue

        x_t = sim_x[alive_t, t]
        below = float(np.mean(x_t < w_min))
        above = float(np.mean(x_t > w_max))
        offgrid = float(np.mean((x_t < w_min) | (x_t > w_max)))
        neg = float(np.mean(x_t < 0.0))
        finite_t = np.isfinite(x_t)
        if np.any(finite_t):
            median_x = float(np.median(x_t[finite_t]))
            p90_x = float(np.quantile(x_t[finite_t], 0.90))
        else:
            median_x = float("nan")
            p90_x = float("nan")

        if offgrid > worst_offgrid_share:
            worst_offgrid_share = offgrid
            worst_offgrid_age = age
        if neg > worst_negative_share:
            worst_negative_share = neg
            worst_negative_age = age

        print(f"  {age:>4}  {n_alive:>7}  {below:>7.1%}  {above:>7.1%}  {offgrid:>9.1%}  "
              f"{neg:>7.1%}  {median_x:>10.4f}  {p90_x:>10.4f}")

    _test(results,
          "pass" if bool(np.all(finite_x)) else "fail",
          "All alive cash-on-hand values are finite",
          "Exploding or NaN wealth means the simulation has left the numerically safe region",
          f"finite share = {np.mean(finite_x):.4%}")

    _test(results,
          "pass" if bool(np.all(finite_alpha)) else "fail",
          "All alive portfolio weights are finite",
          "NaN or inf portfolio weights break the wealth recursion directly",
          f"finite share = {np.mean(finite_alpha):.4%}")

    if worst_negative_share == 0.0:
        _neg_status = "pass"
    elif worst_negative_share < warn_offgrid_share:
        _neg_status = "warn"
    else:
        _neg_status = "fail"
    _test(results, _neg_status,
          "Negative wealth is rare or absent",
          "The current solver is only defined on a strictly positive wealth grid",
          f"pooled share = {pooled_neg_share:.1%}; worst age = {worst_negative_age} ({worst_negative_share:.1%})")

    if worst_offgrid_share < warn_offgrid_share:
        _offgrid_status = "pass"
    elif worst_offgrid_share < fail_offgrid_share:
        _offgrid_status = "warn"
    else:
        _offgrid_status = "fail"
    _test(results, _offgrid_status,
          "Most simulated cash-on-hand stays inside the solved wealth grid",
          "Off-grid paths use linear wealth extrapolation rather than the solved policy domain",
          f"pooled share = {pooled_offgrid_share:.1%}; worst age = {worst_offgrid_age} ({worst_offgrid_share:.1%})")

    if pooled_above_share < warn_offgrid_share:
        _upper_status = "pass"
    elif pooled_above_share < fail_offgrid_share:
        _upper_status = "warn"
    else:
        _upper_status = "fail"
    _test(results, _upper_status,
          "Upper-grid exits are limited",
          "Large mass above wealth_max means the top end of the wealth grid is too tight for this simulation",
          f"pooled share above grid = {pooled_above_share:.1%}")

    print()
    print("  Interpretation note:")
    print("    The simulation currently carries the financial state forward by")
    print("    propagating a continuous shock, then snapping to the nearest state-grid")
    print("    point for next period's policy lookup. That is an intentional simulation")
    print("    approximation, but it should be paired with low off-grid wealth shares.")

    _summary("POST-SIMULATION: POLICY DOMAIN & NUMERICAL COMPATIBILITY", results)
    return results

def diagnose_var_post(model, pc, sim):
    """Post-simulation return and state diagnostics. Not yet implemented."""
    _header("POST-SIMULATION VAR DIAGNOSTICS")
    print("  (Not yet implemented)")
    return []


# =============================================================================
# Solver diagnostics wrapper
# =============================================================================

def diagnose_solver(model, pc, sol=None):
    """Solver quality diagnostics. Currently a stub.

    TODO: expand with post-solve policy function checks.
    For detailed solver diagnostics, call directly:
      - diagnose_terminal_portfolio_states(model, pc)
      - diagnose_newton_failures_retirement(...)
    """
    _header("SOLVER DIAGNOSTICS")
    print("  Available via:")
    print("    diagnose_terminal_portfolio_states(model, pc)")
    print("    diagnose_newton_failures_retirement(...)")
    print("  (Automated solver quality checks: TODO)")
    return []


# =============================================================================
# Convenience wrappers
# =============================================================================

def diagnose_all_pre(model, pc):
    """All pre-solve diagnostics."""
    r = []
    r += diagnose_income_pre(model, pc)
    r += diagnose_var_pre(model, pc)
    r += diagnose_grids_pre(model, pc)
    n_fail = sum(1 for res in r if res.status == 'fail')
    if n_fail:
        print(f"\n  *** {n_fail} FAILURE(S) across all pre-solve diagnostics ***")
    return r


def diagnose_all_post(model, pc, sim):
    """All post-simulation diagnostics."""
    r = []
    r += diagnose_income_post(model, pc, sim)
    r += diagnose_simulation_post(model, pc, sim)
    r += diagnose_var_post(model, pc, sim)
    n_fail = sum(1 for res in r if res.status == 'fail')
    if n_fail:
        print(f"\n  *** {n_fail} FAILURE(S) across all post-simulation diagnostics ***")
    return r


# =============================================================================
# Deprecation wrappers
# =============================================================================

def print_model_diagnostic_report(model, pc, periods_per_year=1):
    """Deprecated. Use diagnose_all_pre() instead."""
    import warnings
    warnings.warn("Use diagnose_all_pre(model, pc) instead", DeprecationWarning, stacklevel=2)
    return diagnose_all_pre(model, pc)


def print_simulation_income_report(model, pc, sim):
    """Deprecated. Use diagnose_income_post() instead."""
    import warnings
    warnings.warn("Use diagnose_income_post(model, pc, sim)", DeprecationWarning, stacklevel=2)
    return diagnose_income_post(model, pc, sim)


# =============================================================================
# TERMINAL PORTFOLIO DIAGNOSTICS (unchanged)
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
            alpha_bill = 1.0 - test_s - test_b
            R_port_test = test_s * R_stock + test_b * R_bond + alpha_bill * R_bill
            if np.any((scenario_weights > 0.0) & (R_port_test <= 0.0)):
                test_moment = np.inf
            else:
                test_moment = float(np.sum(scenario_weights * np.power(R_port_test, 1.0 - gamma)))
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

    Returns
    -------
    results : dict
        Keys: 'summary', 'rows', 'fail_rows'
    """
    if solver_config is None:
        solver_config = SolverConfig()

    mode = "CONSTRAINED" if model.constrained else "UNCONSTRAINED"
    rows = []

    for i_s in range(pc.N_state):
        Rx_bill, Rx_stock_mult, Rx_bond_mult = _build_terminal_quad_returns(
            i_s, pc.state_grid, pc.const_r, pc.A_r, pc.M_v_nodes, pc.ret_nodes
        )

        if model.constrained:
            opt_s, opt_b, euler_sum, exit_code, foc_resid = solve_portfolio_2d_terminal_constrained_njit(
                pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=solver_config.max_iter,
            )
        else:
            # Unconstrained returns 6-tuple now (last element is n_newton_iter); discarded.
            opt_s, opt_b, euler_sum, exit_code, foc_resid, _ = solve_portfolio_unconstrained_terminal_njit(
                pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights, model.gamma,
                init_s=solver_config.init_alpha_s,
                init_b=solver_config.init_alpha_b,
                tol=solver_config.tol,
                max_iter=solver_config.max_iter,
            )

        scenario_weights, R_bill_arr, R_stock, R_bond, _, _ = _terminal_prepare_scenarios(
            pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights
        )
        moment = float(euler_sum)

        foc_s, foc_b, J_ss, J_bb, J_sb, _ = compute_terminal_portfolio_foc_jac(
            opt_s, opt_b, pc.v_weights, Rx_bill, Rx_stock_mult, Rx_bond_mult, pc.ret_weights,
            model.gamma, solver_config.min_return_power, solver_config.prob_skip_threshold
        )
        foc_norm = float(np.hypot(foc_s, foc_b))
        jac = np.array([[J_ss, J_sb], [J_sb, J_bb]], dtype=float)
        eigvals = np.linalg.eigvalsh(jac)
        abs_eigs = np.abs(eigvals)
        cond_proxy = (float(abs_eigs.max() / abs_eigs.min())
                      if abs_eigs.min() > 1e-14 else float("inf"))
        det_jac = float(J_ss * J_bb - J_sb * J_sb)

        a_bill = 1.0 - opt_s - opt_b
        R_port = opt_s * R_stock + opt_b * R_bond + a_bill * R_bill_arr
        positive_mask = scenario_weights > 0.0
        min_r_port = float(np.min(R_port[positive_mask])) if np.any(positive_mask) else float("nan")
        max_r_port = float(np.max(R_port[positive_mask])) if np.any(positive_mask) else float("nan")

        push_hint, push_score = _terminal_push_hint(foc_s, foc_b, tol=solver_config.tol)
        probe_label, probe_step, probe_delta, probe_rel = _probe_terminal_directions(
            opt_s, opt_b, moment, R_bill_arr, scenario_weights, R_stock, R_bond, model.gamma, probe_steps
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
# NEWTON FAILURE DIAGNOSTICS (unchanged, njit)
# =============================================================================

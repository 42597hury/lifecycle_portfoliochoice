"""
diagnostics.py — Model diagnostic reports and Newton failure analysis.

Contains:
  - print_model_diagnostic_report() — comprehensive pre-solve calibration report
  - diagnose_newton_failures_retirement() — post-solve Newton failure analysis

Dependencies: numpy, numba, model, solver (for diagnostic constants and FOC functions)
"""
# Hello

import numpy as np
from numba import njit
from math import exp

from model import SolverConfig
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
    # 2. INCOME PROCESS
    # =========================================================================
    header("2.  INCOME PROCESS")

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
    print(f"  E[eta]     = {mu_eta:.2e}   (should be - 0)")
    print(f"  Std[eta]   = {std_eta:.5f}")
    print(f"  Std[z]     = {std_z:.5f}  (unconditional)")
    z_cover = pc.z_grid.max() / std_z if std_z > 0 else float("nan")
    print(f"  z_grid     : {pc.n_z} points  [{pc.z_grid.min():.4f}, {pc.z_grid.max():.4f}]   Â±{z_cover:.2f} Ïƒ")

    sub("Transitory shock (mixture, zero-mean enforced)")
    mu_eps2_eff = -(model.pe / (1.0 - model.pe)) * model.mu_eps1
    eps_mean    = float(np.sum(pc.eps_nodes * pc.eps_weights))
    eps_var     = float(np.sum(pc.eps_nodes**2 * pc.eps_weights))
    print(f"  pe            = {model.pe:.3f}   (probability of large-shock component)")
    print(f"  Component 1:  mu_eps1    = {model.mu_eps1:+.4f},  sigma_eps1 = {model.sigma_eps1:.4f}")
    print(f"  Component 2:  mu_eps2_eff = {mu_eps2_eff:+.4f}  (zero-mean enforced; model.mu_eps2 ignored)")
    print(f"               sigma_eps2 = {model.sigma_eps2:.4f}")
    print(f"  E[eps]        = {eps_mean:.2e}   (should be - 0)")
    print(f"  Var[eps]      = {eps_var:.5f}   Std[eps] = {np.sqrt(eps_var):.5f}")
    print(f"  eps_weights sum = {pc.eps_weights.sum():.8f}   (should be 1.000)")
    print(f"  eps_nodes     : {pc.n_eps} nodes  [{pc.eps_nodes.min():.4f}, {pc.eps_nodes.max():.4f}]")

    sub("Deterministic income profile  log Y_det = b0 + b1*Age + b2*Age^2/10 + b3*Age^3/100")
    print(f"  b0 = {model.b0:.4f},  b1 = {model.b1:.4f},  b2 = {model.b2:.4f},  b3 = {model.b3:.6f}")
    # Find peak age numerically
    _ages_det = np.arange(model.start_age, model.retire_age)
    _det_vals = model.b0 + model.b1 * _ages_det + model.b2 * _ages_det**2 / 10.0 + model.b3 * _ages_det**3 / 100.0
    _peak_idx = int(np.argmax(_det_vals))
    print(f"  Hump peak: age {_ages_det[_peak_idx]},  log-income = {_det_vals[_peak_idx]:.4f}")

    # Find grid points closest to z=0 and eps=0
    iz0 = int(np.argmin(np.abs(pc.z_grid)))
    ie0 = int(np.argmin(np.abs(pc.eps_nodes)))
    print(f"\n  Income at z - 0 (grid point {iz0}: z={pc.z_grid[iz0]:.4f}),")
    print(f"             eps - 0 (node {ie0}: eps={pc.eps_nodes[ie0]:.4f}):")
    print()
    print(f"  {'Age':>4}  {'t':>4}  {'log-det':>9}  {'Y_gross':>10}  {'Y_net (after-tax)':>18}")
    key_ages_work = [a for a in [25, 30, 35, 40, 45, 50, 55, 60, 64]
                     if model.start_age <= a < model.retire_age]
    for age in key_ages_work:
        t    = age - model.start_age
        det  = model.b0 + model.b1 * age + model.b2 * (age ** 2) / 10.0 + model.b3 * (age ** 3) / 100.0
        y_gr = np.exp(det + pc.z_grid[iz0] + pc.eps_nodes[ie0])
        y_nt = float(pc.working_income[t, iz0, ie0])
        print(f"  {age:>4}  {t:>4}  {det:>9.4f}  {y_gr:>10.4f}  {y_nt:>18.4f}")

    last_t     = model.retire_age - 1 - model.start_age
    y_last     = float(pc.working_income[last_t, iz0, ie0])
    pens_mean  = float(pc.pension_after_tax[last_t, iz0])
    repl_rate  = pens_mean / y_last if y_last > 0 else float("nan")
    print()
    print(f"  Last working year (age {model.retire_age - 1}): Y_net = {y_last:.4f}")
    print(f"  Pension at z - 0:                   {pens_mean:.4f}")
    print(f"  Replacement rate:                   {repl_rate:.2%}")
    print(f"  Pension range (min z, max z):        [{pc.pension_after_tax[0, 0]:.4f}, {pc.pension_after_tax[0, -1]:.4f}]")

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

from numba import njit
from math import exp

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
                
                _, _, _, exit_code, foc_resid = solve_portfolio_2d_retirement(
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


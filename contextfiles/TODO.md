# TODO — Living Task List

**Project:** Lifecycle Portfolio Choice Model with Stocks, Nominal Bonds, and Bills  
**Deadline:** 18 May 2026  
**Last updated:** 8 April 2026  

Status markers: `[ ]` open, `[~]` in progress, `[x]` done

---

## VALIDATION & DEBUGGING

- [ ] **1. Policy function visual inspection**
  - Plot Figures 1–3 from RESULTS.md (portfolio shares vs age, vs wealth, vs financial state)
  - Check: stock share declining with age, bond share hump-shaped or rising, bills filling the residual
  - Check: no discontinuities or wild jumps across adjacent grid points
  - Check: portfolio shares respond sensibly to state variables (e.g., higher dp → more stocks)
  - This is the single most important debugging step — do it first

- [ ] **2. Simulation sanity**
  - Run baseline simulation (1,000–10,000 paths), plot mean lifecycle profiles
  - Check: hump-shaped consumption, wealth accumulation then decumulation
  - Check: survival attrition — fraction alive should match SSA/Chetty targets
  - Check: no negative consumption, no NaN/Inf in any sim array
  - Check: portfolio shares from simulation are consistent with policy function plots
  - Compare qualitative patterns to Cocco/Gomes/Maenhout (2005) or Catherine (2025)

- [ ] **3. Newton failure review**
  - Read the post-solve diagnostics dict carefully
  - Check: convergence rate > 99%, worst FOC residual < 1e-4
  - If failures cluster at specific ages or financial states, investigate root cause
  - Use `diagnose_newton_failures_retirement()` for targeted analysis if needed

- [ ] **4. EGM monotonicity check**
  - Check: total monotonicity violations from diagnostics are negligible (< 0.1% of grid points)
  - If violations cluster at specific ages, check whether the worst drop magnitude is economically meaningful
  - Large violations suggest the endogenous grid is crossing itself — may need finer savings grid

---

## MODEL CALIBRATION CHECKS

- [x] **5. Pin down the money unit** — RESOLVED 8 Apr 2026
  - **1 model unit = SS Wage Index ≈ $54,100 (2019 dollars)**
  - Source: Catherine (2025) Section 5.1: "Households enter working life with 0.1× the national wage index in net worth, the equivalent of $5,400 in 2019."
  - Previous estimate of ~$61k was wrong (used 2023 brackets instead of 2019)
  - **Action:** Update any documentation that references "$61k" to "$54,100 (2019)"

- [x] **6. Initial wealth calibration** — RESOLVED 8 Apr 2026
  - `initial_wealth = 0.1` = 0.1 × $54,100 = $5,410
  - Matches Catherine (2025) exactly: "0.1 × national wage index = $5,400"
  - No change needed

- [x] **7. Confirm income/return unit consistency** — RESOLVED 8 Apr 2026
  - All tax brackets match 2019 TCJA single-filer schedule at $54,100/unit to within 1–2%:
    - 10%/12% boundary: 0.18 × $54,100 = $9,738 vs real $9,700
    - 12%/22%: 0.72 × $54,100 = $38,952 vs real $39,475
    - 22%/24%: 1.54 × $54,100 = $83,314 vs real $84,200
    - 24%/32%: 2.94 × $54,100 = $159,054 vs real $160,725
    - 32%/35%: 3.73 × $54,100 = $201,793 vs real $204,100
    - 35%/37%: 9.32 × $54,100 = $504,212 vs real $510,300
  - Payroll cap: 2.5 × $54,100 = $135,250 vs real 2019 SS taxable max $132,900 — consistent
  - Pension replacement rates verified correct at multiple z values:
    - z=0: 62% of career-avg after-tax, 74% of last-year after-tax, 49% of peak-year after-tax
    - The ~49% vs peak is closest to the SSA headline "~40%" figure
    - Rates decline progressively with earnings (98.8% at z_min → 0.7% at z_max)
  - SSA PIA bend points (0.21 and 1.25) match 2019 SSA thresholds to within 2%
  - Returns are real by VAR construction — no additional adjustment needed
  - wealth_max = 200 = ~$10.8M — adequate ceiling

- [x] **8. Pension formula fix** — RESOLVED 8 Apr 2026
  - `compute_pension_after_tax` now takes `avg_det` parameter
  - Input: `aime = min(exp(z) * avg_det, 2.5)` — scales by career-average deterministic income, caps at SSA taxable max
  - `_precompute_pension` computes `avg_det` from model's age-earnings coefficients
  - `diagnostics.py` includes full AIME pipeline trace with hand-calculation cross-check
  - DESIGN.md updated with complete pension formula documentation

- [x] **9. Memory fix — hoist temp array allocation** — RESOLVED 8 Apr 2026
  - `temp_x/c/s/b` moved from inside `for z_i` to just after `for i_s in prange`
  - Applies to both `_solve_retirement_step_jit` and `_solve_working_age_step_jit`

- [ ] **10. Document inflation treatment in thesis**
  - Returns are real by VAR construction: rtb = nominal bill minus CPI inflation; xr and xb are excess returns (inflation cancels)
  - Portfolio return R_port is therefore a gross real return — consumption and wealth are in real terms
  - Bequest annuity factor uses nominal yield y_nom by design (Catherine 2025): the heir purchases a nominal annuity, so inflation erodes bequests — this is an intentional economic channel
  - Income process is in real terms (Catherine 2025 calibration)
  - Write a clear paragraph in the calibration chapter explaining all of this

---

## RETURNS & VAR QUALITY

- [ ] **11. VAR moment validation**
  - Simulate the standalone annual VAR forward (10,000 draws, 200 periods, burn 100)
  - Compare simulated means to `z_bar`: rtb ≈ -0.08%, xr ≈ 5.36%, xb ≈ 2.36%, y_nom ≈ 3.65%, dp ≈ -4.15
  - Compare simulated stds and autocorrelations to sample estimates
  - Check: no explosive paths, stationary distribution looks reasonable
  - Plot marginal histograms of each variable from simulation vs sample

- [ ] **12. Annualization verification**
  - Confirm `annualize_var_config` output: annual xr mean ≈ 5.4%, xb ≈ 2.4%
  - Check Phi_11 diagonal after annualization: rtb persistence should drop, y_nom and dp stay high
  - Verify `mean_return_ratio ≈ 4.0` for both xr and xb (exact for sums of quarterly returns)

- [ ] **13. Conditional return economic sense check**
  - Inspect `mu_r[i,j,:]` for a few state transitions
  - Check: when dp is high (high expected returns), conditional E[xr] should be higher
  - Check: when y_nom rises, conditional E[xb] should be negative (bond prices fall)
  - Check: the `M` matrix signs make economic sense (M captures how state innovations predict return residuals)

- [ ] **14. Residual variance check**
  - The partition reports variance explained by conditioning: xb should be ~99.5% explained (bond returns are nearly deterministic given state)
  - xr residual should be larger — stocks are noisier
  - Check: `Sigma_r_cond` diagonal values are reasonable (residual annual stock vol ~13%, bond vol ~1%)

- [ ] **15. Nominal vs real yield investigation**
  - y_nom conflates real rate and expected inflation channels
  - These have opposite implications for hedging demand (Nijman et al. 2005, Figure 2)
  - TIPS system data available (`build_tips_system2_var_config`) but short sample (~87 obs vs 183)
  - Decision needed: run TIPS system as robustness (Option B) or discuss as limitation (Option A)
  - At minimum, document this conflation in the thesis

- [ ] **16. Predictability sensitivity**
  - Consider: alternative sample windows (post-2000, post-GFC)
  - Consider: shrinking Phi_21 toward zero to show what happens when predictability is turned off
  - Consider: comparing restricted vs unrestricted VAR estimation
  - This is Phase 2/3 priority — only if time allows after Phase 1 figures are done

---

## RESULTS & FIGURES

- [ ] **17. Code Phase 1 figures** (18 items from RESULTS.md, one solve + one sim)
  - Tables 1, 2, 4
  - Figures 1–7, 10–17
  - All use baseline solve + one simulation — no re-solving needed
  - Write as a single `results.py` module taking `(model, pc, C_mat, S_mat, B_mat, sim)`

- [ ] **18. Code Phase 2 figures** (requires additional solves)
  - Figure 8: Social Security decomposition (solve with pension=0)
  - Figure 9: Risk aversion sensitivity (solves at γ=2, 5, 8)
  - Table 3: Welfare costs of approximation rules
  - Figure 18: Interest rate duration profile
  - Figure 19: Bond premium sensitivity (5–7 solves varying E[xb])
  - Only if time allows after Phase 1

- [ ] **19. Decide on Phase 3 scope**
  - Figure 7 full version (IID solve for clean hedging decomposition)
  - Table 3 rules 6–7 (restricted asset menus)
  - No-labor-income solve (pure financial model, CCV comparison)
  - Grid convergence check (included in production ladder, item 21)

---

## PRODUCTION RUN

- [ ] **20. AWS instance setup**
  - Instance type: c5.4xlarge (16 vCPUs, 32 GB RAM) or similar
  - Peak memory estimate: ~3.5 GB — comfortable on 32 GB
  - Set up environment: Python 3.10+, numba, numpy, scipy, matplotlib

- [ ] **21. Grid convergence ladder**
  - Solve at 5×5×5 → 7×7×7 → 9×9×9 → 10×10×10
  - At each step check: all diagnostics clean, Newton convergence > 99%, no NaN/Inf
  - Compare median portfolio shares at key ages (30, 45, 67, 80) across grid sizes
  - Policy functions should converge — if 9×9×9 and 10×10×10 agree closely, the solution is stable
  - Only the 10×10×10 step is expensive; the rest are quick sanity gates

- [ ] **22. Solver output checklist**
  - Confirm `run_lifecycle_solver` prints per-age convergence table
  - Confirm `policy_io.save_policy_bundle` saves arrays + diagnostics + metadata
  - Verify saved bundle can be reloaded with `load_policy_bundle`
  - Save the full `disc_config` and `solver_config` in metadata for reproducibility

- [ ] **23. Production solve (10×10×10)**
  - `disc_config = DiscretizationConfig(state_grid_sizes=(10,10,10), n_ret_nodes_1d=3, ...)`
  - Expected runtime: several hours (dominated by working-age periods)
  - Save full policy bundle to disk immediately after solve

- [ ] **24. Production simulation + regenerate all figures**
  - Run 10k-path simulation from production arrays
  - Regenerate all Phase 1 figures from production output
  - These are the figures that go in the thesis

---

## CODE QUALITY CHECKS

- [ ] **25. Wealth grid ceiling check**
  - After simulation, check `max(sim["x"])` across all alive agents
  - If any agent's cash-on-hand exceeds `wealth_max=200`, `fast_interp_1d` uses linear extrapolation
  - For consumption this is likely fine (MPC is well-behaved at the boundary)
  - For portfolio shares, extrapolation could push values outside [0,1] — check `sim["alpha_s"]` and `sim["alpha_b"]` ranges
  - If ceiling is hit, raise `wealth_max` or add a runtime warning

- [ ] **26. Terminal solver speed**
  - `solve_portfolio_2d_terminal_exact` calls `scipy.optimize.minimize` (Python, not Numba)
  - At 10×10×10: 1,000 states × up to 5 starting points × potential SLSQP fallback
  - Estimate: 15–30 minutes for terminal age alone
  - Profile on a small run to get actual timing
  - If bottleneck: consider caching terminal portfolios from smaller grid as warm starts

# TODO — Living Task List

**Project:** Lifecycle Portfolio Choice Model with Stocks, Nominal Bonds, and Bills  
**Deadline:** 18 May 2026  
**Last updated:** 7 April 2026  

Status markers: `[ ]` open, `[~]` in progress, `[x]` done

---

## RECENT FIXES (verify before removing)

- [~] **0. Pension AIME calibration fix** (2026-04-07)
  - `compute_pension_after_tax` in `model.py` now takes `(z_grid, avg_det)` and applies the SSA PIA formula to `AIME(z) = min(exp(z) * avg_det, 2.5)` per Catherine (2025) eqs. (19)–(20), instead of feeding raw `exp(z)` with no cap.
  - `_precompute_pension` in `precompute.py` computes `avg_det = mean(exp(f(age)))` over `[start_age, retire_age)` (≈ 0.5069) and passes it through.
  - DESIGN.md §1 retirement-income section rewritten to document the new formula.
  - Sanity numbers: pension at z=0 ≈ 0.254 (was 0.392), cap binds for z ≥ ~1.6 at ≈ 0.628 (was 26.6 at z_max). Replacement rate at z=0 ≈ 63% of career-average after-tax income.
  - **Qualifier — verify before removing this item:**
    - Re-solve the model end-to-end and confirm the solver still converges cleanly (no new Newton failures, no NaN/Inf).
    - Run diagnostics and confirm `pension_after_tax` numbers match the expected table in `PENSION_FIX_HANDOFF.md` §5.1.
    - Re-run a baseline simulation and confirm the qualitative effects are present: more working-life saving, lower early-retirement consumption, higher explicit bond demand.
    - Only after these three checks pass should this item be marked `[x]` and deleted.

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

- [ ] **5. Pin down the money unit**
  - Compute `det = b0 + b1*40 + b2*40²/10 + b3*40³/100` with b0=-6.142, b1=0.304, b2=-0.051, b3=0.002586
  - Compute `exp(det)` at age 40 with z=0, eps=0 — this is median gross income in model units
  - Compare to US median household income (~$70k in 2023 real dollars)
  - Determine the scaling factor: if `exp(det) ≈ 0.55`, then 1.0 model unit ≈ $127k
  - Verify this scaling is consistent with Catherine (2025) — check their Table 1 or calibration section
  - Document the mapping explicitly: "1 model unit = $X in 2023 real dollars"

- [ ] **6. Initial wealth calibration**
  - The `0.1` default in `simulation.py` `_initialize_initial_wealth()` is flagged with a TODO
  - Using the unit from item 5, check what `0.1` means in real dollars (e.g., 0.1 × $127k = $12.7k)
  - Compare to SCF data on median net worth of 22-year-olds
  - Catherine (2025) uses "0.1 × national wage index" — verify this is what you're implementing
  - Consider whether the initial wealth distribution matters for results (sensitivity check)

- [ ] **7. Confirm income/return unit consistency**
  - Income is in real terms (age-earnings polynomial from Catherine 2025)
  - Returns are in real terms (rtb = nominal bill rate minus CPI inflation; xr, xb are excess over rtb)
  - Check: `exp(det(age=40)) ≈ 0.5` in model units, and `wealth_max = 200` means ~$25M — is that enough headroom?
  - Check: `pension_after_tax` at median z gives a replacement rate of ~40-60% of last working income
  - Check: the tax brackets in `disposable_income_working()` align with the model unit (thresholds at 0.18, 0.72, 1.54, etc. should correspond to real bracket boundaries in model dollars)
  - Check: bequest `annuity_factor` at median y_nom gives A ≈ 8–9, meaning the heir spreads wealth over ~8-9 years of consumption — sensible for a 10-year horizon

- [ ] **8. Document inflation treatment in thesis**
  - Returns are real by VAR construction: rtb = nominal bill minus CPI inflation; xr and xb are excess returns (inflation cancels)
  - Portfolio return R_port is therefore a gross real return — consumption and wealth are in real terms
  - Bequest annuity factor uses nominal yield y_nom by design (Catherine 2025): the heir purchases a nominal annuity, so inflation erodes bequests — this is an intentional economic channel
  - Income process is in real terms (Catherine 2025 calibration)
  - Write a clear paragraph in the calibration chapter explaining all of this

---

## RETURNS & VAR QUALITY

- [ ] **9. VAR moment validation**
  - Simulate the standalone annual VAR forward (10,000 draws, 200 periods, burn 100)
  - Compare simulated means to `z_bar`: rtb ≈ -0.08%, xr ≈ 5.36%, xb ≈ 2.36%, y_nom ≈ 3.65%, dp ≈ -4.15
  - Compare simulated stds and autocorrelations to sample estimates
  - Check: no explosive paths, stationary distribution looks reasonable
  - Plot marginal histograms of each variable from simulation vs sample

- [ ] **10. Annualization verification**
  - Confirm `annualize_var_config` output: annual xr mean ≈ 5.4%, xb ≈ 2.4%
  - Check Phi_11 diagonal after annualization: rtb persistence should drop, y_nom and dp stay high
  - Verify `mean_return_ratio ≈ 4.0` for both xr and xb (exact for sums of quarterly returns)

- [ ] **11. Conditional return economic sense check**
  - Inspect `mu_r[i,j,:]` for a few state transitions
  - Check: when dp is high (high expected returns), conditional E[xr] should be higher
  - Check: when y_nom rises, conditional E[xb] should be negative (bond prices fall)
  - Check: the `M` matrix signs make economic sense (M captures how state innovations predict return residuals)

- [ ] **12. Residual variance check**
  - The partition reports variance explained by conditioning: xb should be ~99.5% explained (bond returns are nearly deterministic given state)
  - xr residual should be larger — stocks are noisier
  - Check: `Sigma_r_cond` diagonal values are reasonable (residual annual stock vol ~13%, bond vol ~1%)

- [ ] **13. Nominal vs real yield investigation**
  - y_nom conflates real rate and expected inflation channels
  - These have opposite implications for hedging demand (Nijman et al. 2005, Figure 2)
  - TIPS system data available (`build_tips_system2_var_config`) but short sample (~87 obs vs 183)
  - Decision needed: run TIPS system as robustness (Option B) or discuss as limitation (Option A)
  - At minimum, document this conflation in the thesis

- [ ] **14. Predictability sensitivity**
  - Consider: alternative sample windows (post-2000, post-GFC)
  - Consider: shrinking Phi_21 toward zero to show what happens when predictability is turned off
  - Consider: comparing restricted vs unrestricted VAR estimation
  - This is Phase 2/3 priority — only if time allows after Phase 1 figures are done

---

## RESULTS & FIGURES

- [ ] **15. Code Phase 1 figures** (18 items from RESULTS.md, one solve + one sim)
  - Tables 1, 2, 4
  - Figures 1–7, 10–17
  - All use baseline solve + one simulation — no re-solving needed
  - Write as a single `results.py` module taking `(model, pc, C_mat, S_mat, B_mat, sim)`

- [ ] **16. Code Phase 2 figures** (requires additional solves)
  - Figure 8: Social Security decomposition (solve with pension=0)
  - Figure 9: Risk aversion sensitivity (solves at γ=2, 5, 8)
  - Table 3: Welfare costs of approximation rules
  - Figure 18: Interest rate duration profile
  - Figure 19: Bond premium sensitivity (5–7 solves varying E[xb])
  - Only if time allows after Phase 1

- [ ] **17. Decide on Phase 3 scope**
  - Figure 7 full version (IID solve for clean hedging decomposition)
  - Table 3 rules 6–7 (restricted asset menus)
  - No-labor-income solve (pure financial model, CCV comparison)
  - Grid convergence check (included in production ladder, item 19)

---

## PRODUCTION RUN

- [ ] **18. AWS instance setup**
  - Instance type: c5.4xlarge (16 vCPUs, 32 GB RAM) or similar
  - Peak memory estimate: ~3.5 GB — comfortable on 32 GB
  - Set up environment: Python 3.10+, numba, numpy, scipy, matplotlib

- [ ] **19. Grid convergence ladder**
  - Solve at 5×5×5 → 7×7×7 → 9×9×9 → 10×10×10
  - At each step check: all diagnostics clean, Newton convergence > 99%, no NaN/Inf
  - Compare median portfolio shares at key ages (30, 45, 67, 80) across grid sizes
  - Policy functions should converge — if 9×9×9 and 10×10×10 agree closely, the solution is stable
  - Only the 10×10×10 step is expensive; the rest are quick sanity gates

- [ ] **20. Solver output checklist**
  - Confirm `run_lifecycle_solver` prints per-age convergence table
  - Confirm `policy_io.save_policy_bundle` saves arrays + diagnostics + metadata
  - Verify saved bundle can be reloaded with `load_policy_bundle`
  - Save the full `disc_config` and `solver_config` in metadata for reproducibility

- [x] **21. Memory fix — hoist temp array allocation**
  - In `_solve_retirement_step_jit`: move `temp_x/c/s/b = np.empty(...)` from inside `for z_i` to just after `for i_s in prange`
  - In `_solve_working_age_step_jit`: same fix
  - Currently creates 3.4M needless allocations per full solve (4 arrays × 11 z × 1000 states × 77 ages)
  - Not a crash risk but creates allocation overhead and GC pressure inside Numba's runtime
  - Quick fix: literally move 4 lines up by one indentation level, contents are overwritten each z_i anyway

- [ ] **22. Production solve (10×10×10)**
  - `disc_config = DiscretizationConfig(state_grid_sizes=(10,10,10), n_ret_nodes_1d=3, ...)`
  - Expected runtime: several hours (dominated by working-age periods)
  - Save full policy bundle to disk immediately after solve

- [ ] **23. Production simulation + regenerate all figures**
  - Run 10k-path simulation from production arrays
  - Regenerate all Phase 1 figures from production output
  - These are the figures that go in the thesis

---

## CODE QUALITY CHECKS

- [ ] **24. Wealth grid ceiling check**
  - After simulation, check `max(sim["x"])` across all alive agents
  - If any agent's cash-on-hand exceeds `wealth_max=200`, `fast_interp_1d` uses linear extrapolation
  - For consumption this is likely fine (MPC is well-behaved at the boundary)
  - For portfolio shares, extrapolation could push values outside [0,1] — check `sim["alpha_s"]` and `sim["alpha_b"]` ranges
  - If ceiling is hit, raise `wealth_max` or add a runtime warning

- [ ] **25. Terminal solver speed**
  - `solve_portfolio_2d_terminal_exact` calls `scipy.optimize.minimize` (Python, not Numba)
  - At 10×10×10: 1,000 states × up to 5 starting points × potential SLSQP fallback
  - Estimate: 15–30 minutes for terminal age alone
  - Profile on a small run to get actual timing
  - If bottleneck: consider caching terminal portfolios from smaller grid as warm starts

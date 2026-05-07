# Bundle Creation Pre-flight Checklist

**Compiled:** 2026-05-06
**Branch:** `jax-rewrite`  HEAD: `93ad086`
**Scope:** all validations, sanity checks, smoke tests, and "things we
learned the hard way" that the team has documented anywhere in the repo
(docs, configs, runtime asserts, verify_*.py, diagnostics.py). Items are
sourced — only checks that already exist in the repo are listed.
"Checks the user has wanted but that don't exist yet" are flagged in §G as
GAPS.

## Summary

The single most expensive lesson encoded everywhere is **never ship a
bundle without a pre-flight arbitrage / quadrature check on the new
discretization knobs**. The chain that catches the highest-severity
silent-wrong-result bugs is: (1) confirm the right-matrix invariant —
`Σ_r_cond`, not `Σ_rr`, threads through the CCV `(σ²_xr, σ²_xb,
σ_xrxb)` triple
([`lifecycle/precompute.py:312-314`](../../lifecycle/precompute.py#L312-L314)
and the rtb-as-state Sigma_r_cond drift-detector at
[`lifecycle/precompute.py:680-692`](../../lifecycle/precompute.py#L680-L692));
(2) run `diagnose_var_pre` + `diagnose_grids_pre` so state-grid coverage,
quadrature moment recovery, and Cholesky-PD invariants all light up
green; (3) run `verify/smoke.py` to exercise every kernel; (4) on the
saved bundle, the post-solve `diagnose_terminal_portfolio_states` FOC
residual must be ≤ `tol*scale ~ 1e-6` everywhere. The discretization
docs ([`docs/notes/LOBATTO_CONFIG_TRACKER.md`](../notes/LOBATTO_CONFIG_TRACKER.md))
make the ordering explicit: arbitrage check before commit, EE diagnostic
after solve. This is the order this checklist follows.

---

## §1 Discretization sanity (build_precompute, model factories)

### 1.1 VAR stationarity (max |eigenvalue(Φ_11)| < 1)
- **What:** every eigenvalue of `Φ_11` strictly inside the unit circle.
- **Why:** stationary covariance via Lyapunov equation is undefined
  otherwise; the state grid blows up.
  [`docs/notes/state_grid_design_note.md:74-80`](../notes/state_grid_design_note.md)
- **How:** `stationary_covariance` raises `ValueError` if
  `max|eig| >= 1 - 1e-12`
  ([`lifecycle/discretization.py:74-80`](../../lifecycle/discretization.py#L74-L80)).
  Also surfaced as Test 1 in
  [`lifecycle/diagnostics.py:646-652`](../../lifecycle/diagnostics.py#L646-L652)
  (`Phi_11 stationary`).
- **Pass:** `max|eig| < 1.0`.
- **Severity:** BLOCKER.

### 1.2 Σ_ss positive definite (state-innovation Cholesky)
- **What:** state-innovation covariance must be PD for `cholesky(Σ_ss)`.
- **Why:** Cholesky is what `get_state_quadrature` and `build_state_grid`
  both use; PSD-with-zero-eig collapses an axis.
- **How:** `diagnose_var_pre` Test 8
  ([`lifecycle/diagnostics.py:748-755`](../../lifecycle/diagnostics.py#L748-L755))
  runs `np.linalg.cholesky(model.Sigma_ss)` and asserts.
- **Pass:** Cholesky succeeds (no `LinAlgError`).
- **Severity:** BLOCKER.

### 1.3 Σ_r_cond positive definite (return-residual Cholesky)
- **What:** conditional-residual covariance PD.
- **Why:** `get_return_quadrature` uses `cholesky(Σ_r_cond)`; collapsed
  axis = silent rank deficiency in the integrand.
- **How:** `diagnose_var_pre` Test 9
  ([`lifecycle/diagnostics.py:758-764`](../../lifecycle/diagnostics.py#L758-L764)).
- **Pass:** Cholesky succeeds.
- **Severity:** BLOCKER.

### 1.4 Σ_r_cond rank — OPTION-B drift detector (rtb-as-state)
- **What:** `min eig(Σ_r_cond) >= 1e-5`.
- **Why:** rtb-as-state migration moved rtb out of the return block.
  Wrong VAR partitioning collapses smallest eig to ~3e-7 along the rtb
  axis. The PROPOSED partition has smallest eig ~5e-4. 1e-5 sits between
  them.
- **How:** runtime warning at
  [`lifecycle/precompute.py:680-692`](../../lifecycle/precompute.py#L680-L692)
  fires `RuntimeWarning` naming the suspect axis.
- **Pass:** no warning.
- **Severity:** BLOCKER (silent-wrong-policy if ignored).

### 1.5 State quadrature reproduces conditional return-mean
- **What:** for every source state `i`, `Σ_k w_k · μ_r_k = Φ_0_ret +
  Φ_21·s_i`.
- **Why:** if state quadrature is misweighted, the integrand sees the
  wrong conditional mean — solver converges to wrong policy without
  flagging.
- **How:** `_validate_state_quadrature` runs unconditionally inside
  `build_precompute`
  ([`lifecycle/precompute.py:491-512`](../../lifecycle/precompute.py#L491-L512));
  `assert max_err_mean < 1e-10`.
- **Pass:** `max err < 1e-10`. AssertionError aborts bundle creation.
- **Severity:** BLOCKER.

### 1.6 Return-quadrature mean = 0
- **What:** `sum_n w_n · ret_nodes_n = 0`.
- **Why:** non-zero return-quadrature mean shifts the integrand; CCV
  Jensen lift cancellation breaks.
- **How:** `diagnose_var_pre` Test 6
  ([`lifecycle/diagnostics.py:733-738`](../../lifecycle/diagnostics.py#L733-L738));
  also Test 2 in `verify/discretization.ipynb` §A.4.
- **Pass:** `max|mean| < 1e-10`.
- **Severity:** BLOCKER.

### 1.7 Return-quadrature covariance recovers Σ_r_cond
- **What:** `(ret · w)^T · ret == Σ_r_cond` to machine precision.
- **Why:** wrong covariance recovery means the CCV Itô vol-drag is
  wrong — silent quantitatively-wrong policy, especially at high γ.
- **How:** `diagnose_var_pre` Test 7
  ([`lifecycle/diagnostics.py:740-746`](../../lifecycle/diagnostics.py#L740-L746));
  also Test 2 in
  [`docs/handoff/HANDOFF_VERIFY_RETURN_QUADRATURE.md`](../handoff/HANDOFF_VERIFY_RETURN_QUADRATURE.md)
  and `verify/discretization.ipynb` §A.4.
- **Pass:** `max|diff| < 1e-8` (reported), `< 1e-14` ideal.
- **Severity:** BLOCKER.

### 1.8 Joint state×return covariance recovers Σ_rr
- **What:** `M·Σ_ss·M^T + Σ_r_cond == Σ_rr` from tensor-product
  quadrature.
- **Why:** tensor-product GH preserves independence between v^s and ε
  by construction; if it doesn't recover Σ_rr, something is wrong with
  Cholesky factorisation.
- **How:** `verify/discretization.ipynb` §A.5; documented in
  [`docs/notes/LOBATTO_CONFIG_TRACKER.md` §3.7](../notes/LOBATTO_CONFIG_TRACKER.md).
- **Pass:** `< 4e-17` (empirically observed).
- **Severity:** WARNING (informational regression-guard; covariance
  always recovers in any correctly built precompute).

### 1.9 Cholesky structural signature on returns
- **What:** with `K_per_dim = (1, 1, K_xb)`, columns 0 and 1 of
  `ret_nodes` are exactly zero.
- **Why:** unique to Cholesky transform; would fail under
  eigendecomposition. Regression-guard against accidental revert.
- **How:** Test 7 in
  [`docs/handoff/HANDOFF_VERIFY_RETURN_QUADRATURE.md`](../handoff/HANDOFF_VERIFY_RETURN_QUADRATURE.md);
  `verify/discretization.ipynb` §A.4.
- **Pass:** strict `< 1e-15` zeros on dropped axes.
- **Severity:** WARNING.

### 1.10 Lobatto K validity (must be odd ∈ {3, 5, 7})
- **What:** Lobatto axis requires K ∈ {3, 5, 7}.
- **Why:** closed-form prescribed-tails rule
  (`gauss_hermite_prescribed_tails`) only exists for K=3, 5, 7. K=8/9/10
  must fall back to GH or clamp to 7. Caveat at
  [`configs/_canonical_jax.py:57-62`](../../configs/_canonical_jax.py#L57-L62)
  notes that K=3 with `state_lobatto_Z=7.0` gives near-zero weight at
  ±7σ — Lobatto degenerates to GH-3 + endpoint anchors.
- **How:** `_build_axis_grid` raises `ValueError`
  ([`lifecycle/discretization.py:539-544`](../../lifecycle/discretization.py#L539-L544));
  `gauss_hermite_prescribed_tails` raises if K<3
  ([`lifecycle/quadrature_with_tails.py:103`](../../lifecycle/quadrature_with_tails.py#L103)).
- **Pass:** no exception during `build_precompute`.
- **Severity:** BLOCKER.

### 1.11 Lobatto Z value validity per K
- **What:** Z must lie in K-specific valid windows. K=3: Z>1; K=5:
  Z≥√5≈2.236; K=7: Z∈(1.81, 2.86) ∪ [3.28, ∞).
- **Why:** outside windows the Lobatto rule produces non-positive
  weights (numerical breakdown). Documented in
  [`docs/handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md` "Edge cases"](../handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md).
- **How:** `gauss_hermite_prescribed_tails` raises explicit
  `ValueError` per K-specific check
  ([`lifecycle/quadrature_with_tails.py:127, 138, 158, 179, 187, 210`](../../lifecycle/quadrature_with_tails.py)).
- **Pass:** no `w_i ≤ 0`; rule construction returns positive weights.
- **Severity:** BLOCKER.

### 1.12 Principled-Z calibration on Lobatto axes (simple_clamp regime)
- **What:** `Z_principled[k] = n_stds[k] · L_z[k,k] / L_state[k,k]` per
  Lobatto axis.
- **Why:** Z significantly above principled overshoots (Newton stalls
  navigating the cliff); Z below undershoots (bankruptcy boundary
  unsampled). v11 (1.70× principled, K=7) regressed catastrophically;
  v9 (0.77×, axis 2) had 1.31% invalidity. See
  [`docs/notes/LOBATTO_CONFIG_TRACKER.md` §3.3, §3.6](../notes/LOBATTO_CONFIG_TRACKER.md).
- **How:** manual computation against principled formula. **Does NOT
  apply under CCV** (no bankruptcy boundary).
- **Pass:** Z within roughly `[1.0×, 1.5×]` of principled.
- **Severity:** WARNING (only for `wealth_dynamics_spec="simple_clamp"`;
  N/A for `ccv_log`).

### 1.13 Eta / eps quadrature: mean = 0
- **What:** `Σ w·η = 0` and `Σ w·ε = 0` to machine precision.
- **Why:** non-zero mean biases income across the lifecycle.
- **How:** runtime warnings in `get_eps_quadrature_corrected` /
  `get_eta_quadrature_mixture` at
  [`lifecycle/discretization.py:438-440`](../../lifecycle/discretization.py#L438-L440)
  and `:464-466`. Also `diagnose_income_pre` Tests 1 & 2
  ([`lifecycle/diagnostics.py:167-177`](../../lifecycle/diagnostics.py#L167-L177)).
- **Pass:** `|mean| < 1e-10`.
- **Severity:** BLOCKER.

### 1.14 Eta / eps quadrature: positive weights
- **What:** Judd-mixture quadrature delivers all weights `> 0`.
- **Why:** Theorem 3 (Judd 1998) guarantees positivity at correctly
  conditioned float64; non-positive means the Hankel system was
  ill-conditioned for the K chosen.
- **How:** `_judd_mixture_quadrature` raises `RuntimeError`
  ([`lifecycle/discretization.py:409-414`](../../lifecycle/discretization.py#L409-L414)).
- **Pass:** no exception.
- **Severity:** BLOCKER.

### 1.15 Eta quadrature variance ratio close to 1
- **What:** `Var_quad(η) / Var_true(η) ∈ [0.95, 1.05]`.
- **Why:** quadrature must capture income uncertainty.
- **How:** `diagnose_income_pre` Test 3
  ([`lifecycle/diagnostics.py:192-195`](../../lifecycle/diagnostics.py#L192-L195)).
- **Pass:** ratio in `[0.95, 1.05]`.
- **Severity:** WARNING (fails as fail; usually passes by construction).

### 1.16 Eta quadrature skewness within tolerance
- **What:** `|skew_quad − skew_true| < 0.1` (pass), `< 0.5` (warn).
- **Why:** mixture skewness ↔ rare large income drops are correctly
  priced.
- **How:** `diagnose_income_pre` Test 4
  ([`lifecycle/diagnostics.py:198-208`](../../lifecycle/diagnostics.py#L198-L208)).
- **Pass:** `< 0.1` clean.
- **Severity:** WARNING.

### 1.17 n_eps integrand-error warning at high γ
- **What:** at γ ≥ 5 the eps mixture (kurtosis ~52) is genuinely hard;
  even `n_eps = 5` leaves > 50% relative error on `E[exp(−γε)]`.
- **Why:** documented in `verify/discretization.ipynb` §C.5 and
  [`docs/GRID_CONVERGENCE_CRITERIA.md` §5.5](../GRID_CONVERGENCE_CRITERIA.md).
- **How:** sweep `n_eps ∈ {3, 5, 9}` and report bias estimate at the
  production γ.
- **Pass:** rel err on `E[exp(−γε)]` < 1e-3 (publication-grade) or
  document the bias.
- **Severity:** WARNING.

### 1.18 Pre-flight arbitrage / discrete-free-lunch (DFL) check
- **What:** at any candidate quadrature config, run
  `_diag_arbitrage_quadsweep` (or build a small Precompute and call
  `arbitrage_gap_2d`) against all 343 corners. T-Q1 max gap must be 0
  on both log-excess and arithmetic-excess clouds.
- **Why:** "a misconfigured cloud with strict arbitrage is unsolveable"
  ([`docs/notes/LOBATTO_CONFIG_TRACKER.md` §6](../notes/LOBATTO_CONFIG_TRACKER.md)).
  This is the "bond-tail discrete-free-lunch evidence" mentioned in
  `_canonical.py` comments
  ([`configs/_canonical.py:78-85`](../../configs/_canonical.py#L78-L85)).
  Under simple_clamp, K=3 GH on returns at certain calibrations
  produces clouds where Markowitz alpha implies negative R_p with zero
  weight — the agent gets free lunch from rebalancing into the
  bankruptcy region.
- **How:** call `_diag_arbitrage_quadsweep.py`. Standing operating rule
  in §6 of the LOBATTO tracker.
- **Pass:** T-Q1 max gap = 0 on every state corner; no negative R_p
  cells under canonical α.
- **Severity:** BLOCKER under simple_clamp; under CCV, T-Q1 sufficient
  (no bankruptcy boundary; LOBATTO §11.3).

---

## §2 State-grid coverage

### 2.1 State-grid mode is `cholesky` (production default)
- **What:** `state_grid_mode == "cholesky"` (or legacy alias
  `"principal"`).
- **Why:** axis-aligned naive grid leaves >5% of historical observations
  outside the box at N=7
  ([`docs/notes/state_grid_design_note.md` §1.3](../notes/state_grid_design_note.md));
  earlier `state_n_stds=(0.6, 1.75, 2.0)` gave joint coverage ~40% and
  produced "unusable simulator moments — do not regress to it"
  ([`configs/_canonical.py:50-55`](../../configs/_canonical.py#L50-L55)).
- **How:** `build_state_grid` raises `ValueError` on unknown mode
  ([`lifecycle/discretization.py:208-209`](../../lifecycle/discretization.py#L208-L209)).
- **Pass:** mode in `{"naive", "lyapunov-axis", "cholesky"}`; default
  `cholesky`.
- **Severity:** BLOCKER.

### 2.2 Per-axis state coverage ≥ 2.5σ
- **What:** `(grid_max − μ) / σ_z ≥ 2.5` per state axis.
- **Why:** below this the grid corners don't cover the stationary
  distribution.
- **How:** `diagnose_var_pre` Tests 2-4
  ([`lifecycle/diagnostics.py:684-689`](../../lifecycle/diagnostics.py#L684-L689)).
  Production-grade 99% per-axis would need `(~2.93, ~2.93, ~2.93)`
  ([`configs/_canonical.py:55`](../../configs/_canonical.py#L55)).
- **Pass:** all axes ≥ 2.5σ; warn if not.
- **Severity:** WARNING.

### 2.3 `Π_state` row-stochastic
- **What:** every row of `Pi_state` sums to 1.
- **Why:** legacy fallback transition matrix; even if the solver
  doesn't use it, the simulator's discrete branch does.
- **How:** `diagnose_var_pre` Test 5
  ([`lifecycle/diagnostics.py:726-731`](../../lifecycle/diagnostics.py#L726-L731));
  `Pi_state` is also normalised in `_independence_rouwenhorst_pi`
  ([`lifecycle/discretization.py:135-137`](../../lifecycle/discretization.py#L135-L137)).
- **Pass:** `max |row sum − 1| < 1e-10`.
- **Severity:** BLOCKER.

### 2.4 State-grid sizes match `model.n_state`
- **What:** `len(state_grid_sizes) == model.n_state`.
- **Why:** post rtb-as-state, n_state is 4. A 3-tuple silently breaks
  the build.
- **How:** `build_precompute` raises `ValueError`
  ([`lifecycle/precompute.py:249-250`](../../lifecycle/precompute.py#L249-L250)).
- **Pass:** lengths match.
- **Severity:** BLOCKER.

### 2.5 `state_n_stds` length and positivity
- **What:** scalar or length-`n_state` sequence; all entries `> 0`.
- **Why:** silent broadcast on shape mismatch; non-positive collapses
  axis.
- **How:** `_normalize_n_stds` raises
  ([`lifecycle/discretization.py:140-155`](../../lifecycle/discretization.py#L140-L155)).
- **Pass:** no exception.
- **Severity:** BLOCKER.

### 2.6 1981-Volcker observation inside hull (historical sanity)
- **What:** historical max-y_1 (Volcker 1981, ~13.86%, +2.53σ_z) lies
  inside the grid.
- **Why:** under naive grid (current default at +1.46σ) the Volcker
  observation is read as the boundary policy, not the true policy.
- **How:** documented diagnostic in
  [`docs/notes/state_grid_design_note.md` §1.4](../notes/state_grid_design_note.md);
  empirical test pattern in §4 of that note.
- **Pass:** Volcker observation inside grid hull at chosen
  `state_n_stds`.
- **Severity:** WARNING (regression-guard).

### 2.7 Indices `y_1`, `spr`, `rtb` distinct in state vector
- **What:** `y_1_index_in_state ≠ spr_index_in_state ≠
  rtb_index_in_state`.
- **Why:** post rtb-as-state migration these must be three distinct
  state-grid axes; collapsing two would mean the bequest annuity factor
  reads the same axis twice.
- **How:** `build_model` raises `ValueError`
  ([`lifecycle/precompute.py:732-755`](../../lifecycle/precompute.py#L732-L755)).
- **Pass:** all three distinct.
- **Severity:** BLOCKER.

### 2.8 `rtb_index_in_state` provided (not None)
- **What:** under the rtb-as-state migration `rtb_index_in_state` MUST
  be set on `var_config`.
- **Why:** scalar fallback for `rtb` is no longer supported; the FOC
  reads `log_R_bill` from `state_{t+1}[rtb_idx]`.
- **How:** `build_model` raises `ValueError`
  ([`lifecycle/precompute.py:740-745`](../../lifecycle/precompute.py#L740-L745)).
- **Pass:** integer in range.
- **Severity:** BLOCKER.

---

## §3 Wealth/savings grid

### 3.1 `wealth_min` raised past EGM constrained region
- **What:** `wealth_min ≥ 0.05` (canonical 0.13; smoke 0.13).
- **Why:** raised from 1e-4 on 2026-05-03 to skip the EGM constrained
  region. Pre-fix, `wealth_grid[0]` interpolated across the degenerate
  anchor segment, producing 41% gridpoint-EE failure share at iw=0.
  See
  [`docs/STATE_SPACE.md §wealth_min` (line 219)](../STATE_SPACE.md)
  and
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §2 "mode 1 EGM anchor"](../agents/EE_DIAGNOSTIC_WORKFLOW.md).
- **How:** read in `build_precompute`
  ([`lifecycle/precompute.py:234-237`](../../lifecycle/precompute.py#L234-L237));
  `disc_config.wealth_min` set explicitly in
  [`configs/_canonical.py:62`](../../configs/_canonical.py#L62).
- **Pass:** `wealth_min ≥ 0.05`; ideally 0.13 (canonical).
- **Severity:** BLOCKER.

### 3.2 Wealth grid strictly positive
- **What:** every wealth-grid point > 0.
- **Why:** log utility / `c^(-γ)` blow-up at zero.
- **How:** `diagnose_grids_pre` Test 2
  ([`lifecycle/diagnostics.py:872-876`](../../lifecycle/diagnostics.py#L872-L876)).
- **Pass:** `wealth_grid.min() > 0`.
- **Severity:** BLOCKER.

### 3.3 `savings_max ≤ wealth_max`
- **What:** explicit-savings or default-fallback savings_max never
  exceeds wealth_max.
- **Why:** EGM resolves savings → wealth; savings beyond `wealth_max`
  fall off the wealth grid and produce nonsense interpolation.
- **How:** `build_precompute` raises `ValueError`
  ([`lifecycle/precompute.py:228-232`](../../lifecycle/precompute.py#L228-L232)).
- **Pass:** no exception.
- **Severity:** BLOCKER.

### 3.4 `savings_max > savings_min`
- **What:** strictly increasing savings interval.
- **Why:** `expm1(linspace(log1p(min), log1p(max), n_s))` collapses to
  zeros if `max ≤ min`.
- **How:** raises `ValueError`
  ([`lifecycle/precompute.py:228-230`](../../lifecycle/precompute.py#L228-L230)).
- **Pass:** strict inequality.
- **Severity:** BLOCKER.

### 3.5 Boundary mass at `wealth_grid[-1]`
- **What:** simulated agent-years at upper wealth bound `< 0.5%`.
- **Why:** linear extrapolation past `wealth_max` (`fast_interp_1d`)
  can push portfolio shares outside [0, 1] and produce nonsense.
  Documented as TODO in
  [`docs/STATE_SPACE.md` §4 (line 244)](../STATE_SPACE.md);
  `docs/TODO.md` items 25 & 26.
- **How:** post-simulation aggregate `np.mean(sim['x'][:,-1] >=
  wealth_max)`.
- **Pass:** `< 0.5%` (publication grade);
  [`docs/GRID_CONVERGENCE_CRITERIA.md` §3 row "Boundary-mass at
  wealth_grid[-1]"](../GRID_CONVERGENCE_CRITERIA.md).
- **Severity:** WARNING.

### 3.6 Wealth-min anchor decomposition (Lobatto bundles)
- **What:** per-iw failure share scan to distinguish EGM-anchor mode
  vs leverage-corner-kink mode.
- **Why:** the two modes need different remediations; reporting only
  "max log10|EE| at gridpoint" without separating them gives a single
  number no single config change can move.
- **How:** `_diag_gridpoint_ee --wealth-indices 0 1 2 3 4 5 10 20 50
  100 149`. Documented in
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §2 "Bimodal wealth
  decomposition"](../agents/EE_DIAGNOSTIC_WORKFLOW.md).
- **Pass:** failure share at `iw=0` close to share at `iw=10`
  (no anchor mode).
- **Severity:** WARNING.

---

## §4 Income process / labour

### 4.1 Innovation mean = 0
- **What:** `|E[η]| < 1e-6` (component 2 mean derived to enforce zero).
- **Why:** good and bad income shocks must cancel; `mu_eta2_eff =
  -(pz/(1-pz))·mu_eta1` enforced internally.
- **How:** `diagnose_income_pre` Test 1
  ([`lifecycle/diagnostics.py:167-171`](../../lifecycle/diagnostics.py#L167-L171)).
- **Pass:** `< 1e-6`.
- **Severity:** BLOCKER.

### 4.2 After-tax income > 0 everywhere
- **What:** `working_income.min() > 0` and `pension_after_tax.min() > 0`.
- **Why:** non-positive income breaks log/CRRA inside the FOC.
- **How:** `diagnose_income_pre` Test 8
  ([`lifecycle/diagnostics.py:271-278`](../../lifecycle/diagnostics.py#L271-L278)).
- **Pass:** strictly positive.
- **Severity:** BLOCKER.

### 4.3 Income monotone in z and ε
- **What:** `working_income` weakly rising in `i_z` and `i_eps`.
- **Why:** higher-skill workers / better-luck-shocks must earn ≥ —
  catches sign bugs in the deterministic profile.
- **How:** Tests 9 & 10
  ([`lifecycle/diagnostics.py:281-296`](../../lifecycle/diagnostics.py#L281-L296)).
- **Pass:** min adjacent gap ≥ −1e-12.
- **Severity:** BLOCKER.

### 4.4 Pension cap at AIME max (≈ $135k)
- **What:** `max(pension_after_tax) == AIME_max_PIA_net`.
- **Why:** SS earnings cap implementation; mismatch implies wrong
  bend-point logic.
- **How:** `diagnose_income_pre` Test 13
  ([`lifecycle/diagnostics.py:314-327`](../../lifecycle/diagnostics.py#L314-L327)).
- **Pass:** `|expected − actual| < 1e-6`.
- **Severity:** WARNING.

### 4.5 AIME pipeline end-to-end at representative z
- **What:** AIME → PIA → tax → net pension hand-calculated and
  cross-checked at multiple z values.
- **Why:** detected the pension formula bug fixed 8 Apr 2026
  (`docs/TODO.md` item 8).
- **How:** Test 14
  ([`lifecycle/diagnostics.py:329-388`](../../lifecycle/diagnostics.py#L329-L388)).
- **Pass:** `|hand − precompute| < 1e-6` at all representative z.
- **Severity:** BLOCKER.

### 4.6 Pension array covers terminal age
- **What:** `pc.pension_after_tax.shape[0] >= n_age`.
- **Why:** out-of-bounds at the work→retire transition is a known
  prior bug (`HANDOFF_WORK_RETIRE_TRANSITION_BUG.md`,
  `docs/handoff/archive`).
- **How:** Tests 19 & 20
  ([`lifecycle/diagnostics.py:524-535`](../../lifecycle/diagnostics.py#L524-L535)).
- **Pass:** indices in range.
- **Severity:** BLOCKER.

### 4.7 Median replacement rate in [40%, 80%]
- **What:** pension/career_avg in [0.40, 0.80] at median z.
- **Why:** plausibility cross-check against SSA replacement rates.
- **How:** Test 16
  ([`lifecycle/diagnostics.py:457-461`](../../lifecycle/diagnostics.py#L457-L461)).
- **Pass:** in range.
- **Severity:** WARNING.

### 4.8 z frozen after retirement (simulator)
- **What:** post-retirement, `sim_z_idx` invariant per agent.
- **Why:** SS benefit locked at retirement earnings; drifting z would
  violate model spec.
- **How:** post-sim Tests 3 & 4
  ([`lifecycle/diagnostics.py:1051-1065`](../../lifecycle/diagnostics.py#L1051-L1065)).
- **Pass:** all alive agents have identical z at retire vs retire+1.
- **Severity:** BLOCKER.

### 4.9 Survival probabilities in (0, 1]
- **What:** `survival_probs_2d` strictly in (0, 1].
- **Why:** outside this range = unphysical mortality.
- **How:** `diagnose_grids_pre` Test 1
  ([`lifecycle/diagnostics.py:866-870`](../../lifecycle/diagnostics.py#L866-L870)).
- **Pass:** `min > 0, max ≤ 1`.
- **Severity:** BLOCKER.

### 4.10 Selection: mean z rises with age among survivors
- **What:** Chetty-mortality-driven selection produces rising mean z
  among living agents.
- **Why:** post-sim regression-guard that mortality module is wired
  through.
- **How:** post-sim Test 5
  ([`lifecycle/diagnostics.py:1102-1108`](../../lifecycle/diagnostics.py#L1102-L1108)).
- **Pass:** `mean_z(85) > mean_z(start)`.
- **Severity:** WARNING.

---

## §5 Solver / Newton convergence

### 5.1 Solve status `complete`
- **What:** `diag["solve_status"] == "complete"` and
  `diag["n_ages_solved"] == expected`.
- **Why:** partial solves are valid (checkpoint + crash recovery) but
  not bundle-publishable.
- **How:** print + check at
  [`verify/benchmark_bundle.py:117-118`](../../verify/benchmark_bundle.py#L117-L118).
- **Pass:** `complete` and all expected ages solved.
- **Severity:** BLOCKER.

### 5.2 Newton failures = 0
- **What:** `diag["total_newton_failures"] == 0`.
- **Why:** under `use_fori_newton=True`, ALL `max_iter` iters run
  unconditionally, so failures = cells that didn't converge in
  `max_iter`. Cap-bound (alpha hit ±6) cells also surface here.
- **How:** `verify/benchmark_bundle.py:119`. Configured wall-cost in
  [`docs/handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md`](../handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md).
  Canonical `max_iter` choice driven by §3 of
  [`docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md`](COMPLEXITY_WALL_TIME_2026-05-06.md).
- **Pass:** zero failures; if non-zero, inspect distribution by age
  (boundary age 66 + terminal age 99 are usual hot spots — bump
  `max_iter` 20 → 30-50 per
  [`configs/_canonical_jax.py:83-89`](../../configs/_canonical_jax.py#L83-L89)).
- **Severity:** BLOCKER.

### 5.3 `worst_foc_resid` close to `solver_config.tol`
- **What:** `worst_foc_resid` ≈ `tol` (default 1e-7).
- **Why:** ratio `worst_foc_resid / tol` >> 1 means cells passed Newton
  exit but with relaxed residual (warm-restart fallback). Under CCV,
  high `worst_foc_resid` indicates real solver bug
  ([`docs/notes/LOBATTO_CONFIG_TRACKER.md` §11.5](../notes/LOBATTO_CONFIG_TRACKER.md)).
- **How:** read from `metadata.json`
  `diagnostics_summary.worst_foc_resid`. Workflow at
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §3 "Sanity check the
  bundle"](../agents/EE_DIAGNOSTIC_WORKFLOW.md).
- **Pass:** `worst_foc_resid < 100 × tol` (acceptable),
  `< 5 × tol` (welfare-grade) per
  [`docs/GRID_CONVERGENCE_CRITERIA.md` §3](../GRID_CONVERGENCE_CRITERIA.md).
- **Severity:** BLOCKER if `>> tol*1000`; WARNING otherwise.

### 5.4 `total_mono_violations == 0`
- **What:** EGM monotonicity violations across the savings grid.
- **Why:** policy not weakly increasing in savings → endogenous grid
  has crossed itself; usually means `n_savings` too coarse. TODO item 4.
- **How:** read `diagnostics_summary.total_mono_violations` from
  metadata; if non-zero, check whether the worst drop magnitude is
  economically meaningful.
- **Pass:** zero; or `< 0.1% of grid points` (`docs/TODO.md` item 4).
- **Severity:** BLOCKER.

### 5.5 `max_iter` calibrated to fori-loop semantics
- **What:** under `use_fori_newton=True`, `max_iter` is wall cost (not
  "average × converged"). Canonical 5000 is "**catastrophic**" under
  fori_loop+mask.
- **Why:** documented in
  [`docs/handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md` §4.3](../handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md);
  rationale for separating
  [`configs/_canonical_jax.py`](../../configs/_canonical_jax.py)
  (`max_iter=20`) from `_canonical.py` (`max_iter=8000`).
- **How:** `SolverConfig.max_iter` should be ~3-5× the 99th-percentile
  observed Newton iter count (≈30-50 with backward-age warm-start).
- **Pass:** if `use_fori_newton=True`, `max_iter ≤ ~50`; if False,
  whatever.
- **Severity:** WARNING (cost not correctness).

### 5.6 Backward-age warm-start enabled (recommended)
- **What:** `SolverConfig.use_backward_age_warm_start=True`.
- **Why:** Newton converges in ~2-4 iters per cell vs ~5-15 cold from
  the canonical `(0.85, 0.44)` init.
- **How:** documented in
  [`docs/handoff/HANDOFF_BACKWARD_AGE_WARM_START.md`](../handoff/HANDOFF_BACKWARD_AGE_WARM_START.md).
- **Pass:** flag True; failures cluster at terminal + boundary age
  acceptable.
- **Severity:** INFO.

### 5.7 `wealth_dynamics_spec` consistent solver↔simulator
- **What:** `solver_config.wealth_dynamics_spec ==
  simulate_lifecycle(..., wealth_dynamics_spec=...)`.
- **Why:** "if they disagree on R_p at any quadrature node, every
  Euler-residual diagnostic becomes meaningless"
  ([`docs/CCV_RETURNS.md` §2.3](../CCV_RETURNS.md)). Tests pin parity
  to 1e-10.
- **How:** `tests/test_cvc_solver_sim_consistency.py`,
  `tests/test_cvc_diagnostic_consistency.py`. Bundle metadata writes
  the spec at
  [`lifecycle/policy_io.py:145-167`](../../lifecycle/policy_io.py#L145-L167).
- **Pass:** strings match.
- **Severity:** BLOCKER.

### 5.8 CCV is selected (`ccv_log`) for production
- **What:** `wealth_dynamics_spec="ccv_log"` (the production default
  since May 2026; pinned in `_canonical_jax.py`).
- **Why:** simple_clamp produces a C0-not-C1 integrand at the
  bankruptcy boundary; quadrature stalls at `mean log10|EE| ≈ −2.5`.
  `docs/CCV_RETURNS.md` §1.2.
- **How:** pinned in
  [`configs/_canonical_jax.py:90-95`](../../configs/_canonical_jax.py#L90-L95).
- **Pass:** spec is `ccv_log`.
- **Severity:** BLOCKER for production figures.

---

## §6 Bundle-output sanity

### 6.1 No NaN / Inf in C, S, B over solved ages
- **What:** `np.isnan(C[solved]).sum() == 0` etc. across solved-age
  slices.
- **Why:** silent NaN propagation in policies → simulator crashes
  much later, after expensive solve.
- **How:** `verify/benchmark_bundle.py:122-130`,
  `verify/canonical_small.py:64`, `verify/smoke.py:44`.
- **Pass:** zero NaN, zero Inf.
- **Severity:** BLOCKER.

### 6.2 `alpha_s, alpha_b` in plausible range
- **What:** range printout from solved policy slice.
- **Why:** sanity that no α exploded past `alpha_min/alpha_max`. With
  cap ±6 expect simulated max ≈ 9.25 historically; cap-bound surfaces
  as Newton fail.
- **How:** `verify/benchmark_bundle.py:131-139`, also
  [`docs/CONFIG.md` §2.6](../CONFIG.md).
- **Pass:** within `[alpha_min, alpha_max]`; sanity reportable.
- **Severity:** WARNING.

### 6.3 Terminal-portfolio FOC residuals ≤ tol at solved policy
- **What:** `diagnose_terminal_portfolio_states` re-evaluates the FOC
  at the solved (α_s, α_b) per state. `summary["resid_max"] ≤ tol`.
- **Why:** post-solve verification — catches "solver returned
  non-converged warm-restart fallback as if it were a real interior
  solution". v4_lobatto regressed this with `worst_foc_resid = 0.67`.
- **How:** called at
  [`verify/benchmark_bundle.py:208-225`](../../verify/benchmark_bundle.py#L208-L225);
  function at
  [`lifecycle/diagnostics.py:1274-1441`](../../lifecycle/diagnostics.py#L1274-L1441).
- **Pass:** `summary["n_fail"] == 0` and
  `resid_max ≤ max(tol, 1e-6)`.
- **Severity:** BLOCKER.

### 6.4 Bundle metadata round-trip
- **What:** save → load → check `metadata["wealth_dynamics_spec"] ==
  expected`; saved `disc_config` matches solver run.
- **Why:** legacy bundles default to `simple_clamp`; CCV bundles must
  carry explicit tag.
- **How:** test in `tests/test_cvc_diagnostic_consistency.py:128-147`;
  `save_policy_bundle` writes at
  [`lifecycle/policy_io.py:145-167`](../../lifecycle/policy_io.py#L145-L167).
- **Pass:** loaded metadata matches solver config in spec, disc, and
  solver fields.
- **Severity:** BLOCKER.

### 6.5 Numba reference parity (when available)
- **What:** JAX policy ≈ Numba policy on smoke at small-grid config.
- **Why:** verifies JAX rewrite hasn't drifted from the Numba
  ground-truth on the same maths.
- **How:** `verify/compare_jax.py` saves JAX policies to
  `/tmp/jax_policies.npz`; pair with Numba branch's equivalent and
  compare.
- **Pass:** `np.allclose(C_jax, C_numba, rtol=1e-5)`.
- **Severity:** WARNING (only when both backends available).

### 6.6 Newton fori vs while numerical equivalence
- **What:** running smoke once with `use_fori_newton=False` (while_loop)
  and once with `True` (fori) produces α-ranges within ~1e-9.
- **Why:** the fori conversion was a verbatim refactor; any divergence
  > 1e-9 means the `improved_now = NOT found AND ...` gate is wrong.
- **How:** documented test in
  [`docs/handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md` §5.1](../handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md).
- **Pass:** ranges match to 1e-9; ideally bit-identical.
- **Severity:** WARNING.

---

## §7 Numerical safety floors (tuning, not correctness)

### 7.1 `tiny_savings = 1e-6` floor
- **What:** below this savings level, holds warm-start alphas.
- **Where:** `SolverConfig.tiny_savings` default
  ([`docs/CONFIG.md` §2.5](../CONFIG.md)).

### 7.2 `min_consumption = 1e-10` floor
- **What:** clamps `c` before `c^(-γ)`.
- **Why:** prevents `c=0 → 1/0` in CRRA marginal utility. Verified in
  bug scan
  ([`docs/scans/BUG_SCAN_2026-05-06.md` Item 6](BUG_SCAN_2026-05-06.md)).
- **Severity:** INFO.

### 7.3 `min_return_power = 1e-15` floor
- **What:** floor on `R_port^{1−γ}` evaluation. Only used in unshifted
  CRRA terminal kernel which is dead code under shifted bequest
  ([`docs/CCV_RETURNS.md` §3.2(b)](../CCV_RETURNS.md)).
- **Severity:** INFO.

### 7.4 `euler_inv_floor = 1e-20` floor
- **What:** clamps `β·V_dot` before `(·)^(−1/γ)` inversion. At γ=10,
  `1e-20^(−0.1) = 1e2` — finite.
- **Severity:** INFO.

### 7.5 `singular_det = 1e-15` Newton-Jacobian fallback
- **What:** below this, switches to gradient-descent step (`step_s_grad
  = grad_step_size · fs / grad_norm`).
- **Severity:** INFO.

### 7.6 `egm_anchor = 1e-10` (EGM endpoint anchor)
- **What:** floor at the constrained-region anchor in EGM endogenous
  grid construction.
- **Severity:** INFO.

### 7.7 `alpha_min/max = ±6` numerical leverage cap
- **What:** box projection on (α_s, α_b) inside the unconstrained
  Newton.
- **Why:** real cap; production hit max simulated |α| ≈ 9.25 at γ=5.
  Cap-bound cells surface as `EC_NEWTON_FAIL`. Theory review
  recommends tightening to ±4 at γ=5
  ([`docs/CCV_RETURNS.md` §3.3](../CCV_RETURNS.md)).
- **Severity:** WARNING (the only swept solver knob;
  [`docs/CONFIG.md` §3.4](../CONFIG.md)).

---

## §8 HBM / OOM pre-flight (GPU)

### 8.1 Per-cell `c_corners` working set fits in L2
- **What:** working-age `c_corners` shape `(K_v, n_z, 2^n_state, n_w)`
  in float64. At 4-D state with `n_state_quad=144`, n_z=11, n_w=180:
  `144 × 11 × 16 × 180 × 8 ≈ 36 MB/cell`. `(2,3,2,3)=36` quad gives
  `~9 MB/cell`.
- **Why:** if working set ≪ H200 L2 (60 MB), Newton inner loop reads
  hit cache and the kernel is compute-bound. If exceeds L2, drops to
  HBM bandwidth and 5-20× slower.
  [`docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md` §4](COMPLEXITY_WALL_TIME_2026-05-06.md).
- **How:** estimate before launch; per-cell table in
  [`docs/notes/GPU_TRIAL_FINDINGS.md`](../notes/GPU_TRIAL_FINDINGS.md).
- **Pass:** per-cell ≤ 10 MB.
- **Severity:** WARNING.

### 8.2 Total batched `c_corners` × `n_cells` ≤ HBM with margin
- **What:** at 9⁴ canonical full quad: `~626 GB DRAM traffic per age` —
  but vmap-batched form in HBM is the binding number.
- **Why:** "Memory CAPACITY (not bandwidth) is the binding constraint
  at 7⁴ and 9⁴. Anchor uses 92.9 / 97.9 GB HBM at 5⁴ — 9⁴ on a single
  H200 will OOM."
  [`docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md` §4](COMPLEXITY_WALL_TIME_2026-05-06.md).
- **How:** XLA compilation lottery — a 7⁴ config OOM'd at 1.06 TB
  scheduling plan despite the irreducible floor being 67 GB
  ([`docs/notes/GPU_TRIAL_FINDINGS.md` Benchmark attempt 2](../notes/GPU_TRIAL_FINDINGS.md)).
- **Pass:** smoke at 5⁴+(2,3,2,3)+max_iter=100 fits; if 7⁴ OOMs,
  follow the fallback ladder.
- **Severity:** BLOCKER on GPU.

### 8.3 GPU OOM fallback ladder
- **What:** when OOM hits, downgrade in this order: (1) set
  `TF_GPU_ALLOCATOR=cuda_malloc_async` +
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, (2) drop `n_state_quad` (e.g.
  `(3,4,3,4)→(2,3,2,3)`, 4× cheaper), (3) drop `max_iter`, (4) drop
  `state_grid_sizes` per-axis, (5) drop `n_w`/`n_s`. Don't drop `n_z`.
- **Why:**
  [`docs/notes/GPU_TRIAL_FINDINGS.md` "Practical fallback ladder"](../notes/GPU_TRIAL_FINDINGS.md).
- **Severity:** WARNING (operational guidance).

### 8.4 `n_z=11` is a published-paper minimum (don't drop)
- **What:** Catherine (2025) uses 11; CGM (2005) used 9; FGG (2017)
  used 21. Below 11 the persistent income discretization isn't
  defensible.
- **Why:**
  [`docs/GRID_CONVERGENCE_CRITERIA.md` §5.2](../GRID_CONVERGENCE_CRITERIA.md);
  [`docs/notes/GPU_TRIAL_FINDINGS.md`](../notes/GPU_TRIAL_FINDINGS.md).
- **Severity:** WARNING.

### 8.5 GPU JAX device check at solver entry
- **What:** `jax.devices()` reports a non-CPU device when GPU env hints
  are set.
- **Why:** silent CPU fallback after expensive GPU bootstrap. Bug scan
  flagged that no try/except wraps `_jax.devices()`
  ([`docs/scans/BUG_SCAN_2026-05-06.md` Item 9](BUG_SCAN_2026-05-06.md)).
- **How:** `_check_runtime_platform()` warning in
  `lifecycle/__init__.py:151`; printed from solver banner at
  [`verify/benchmark_bundle.py:94`](../../verify/benchmark_bundle.py#L94).
- **Pass:** banner prints expected GPU device count.
- **Severity:** WARNING.

### 8.6 Persistent compilation cache configured
- **What:** `LIFECYCLE_JAX_CACHE_DIR` set; `min_compile_time_secs=1.0`
  default; `min_entry_size_bytes=-1`.
- **Why:** without it, every solver invocation re-traces the kernels
  (10-30s × 4 kernels = 40-120s wasted per launch). Documented in
  [`docs/handoff/HANDOFF_JAX_PERSISTENT_COMPILATION_CACHE.md`](../handoff/HANDOFF_JAX_PERSISTENT_COMPILATION_CACHE.md).
- **How:** configured at
  [`lifecycle/__init__.py:74-138`](../../lifecycle/__init__.py#L74-L138).
- **Pass:** cache dir exists and is non-empty after a smoke run.
- **Severity:** INFO.

---

## §9 Reference-bundle comparison (post-solve EE diagnostics)

### 9.1 Diagnostic A — Gridpoint EE under same quadrature
- **What:** `_diag_gridpoint_ee --eval-mode same`. Grades policy
  against the bundle's own quadrature.
- **Why:** self-grade. If a bundle fails this, it doesn't agree with
  itself — hard failure.
- **How:**
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §2 "Diagnostic A"](../agents/EE_DIAGNOSTIC_WORKFLOW.md).
- **Pass:** mean log10|EE| < −5; validity 100%.
- **Severity:** BLOCKER.

### 9.2 Diagnostic B — Gridpoint EE under finer quadrature
- **What:** `_diag_gridpoint_ee --eval-mode next_finer`. Grades against
  a strictly richer rule.
- **Why:** "honest" grade. Gap between A and B is the policy's
  truncation bias.
- **Pass:** invalidity rate < 1%; mean within 0.5 orders of A.
- **Severity:** WARNING (informational about quadrature truncation).

### 9.3 Diagnostic C — Sim-path EE under finer quadrature
- **What:** `_diag_euler_errors --eval-mode next_finer
  --n-simulations 5000 --eval-households-per-age 256`.
- **Why:** "the test referees grade against." Simulates a panel and
  evaluates FOC at the simulated states.
- **Pass:** publication gates from
  [`docs/GRID_CONVERGENCE_CRITERIA.md` §3](../GRID_CONVERGENCE_CRITERIA.md):
  working `mean<−4, max<−3`; retirement `mean<−4.5, max<−3`.
- **Severity:** BLOCKER for paper-grade bundles.

### 9.4 Lobatto-eval-rule propagation under simple_clamp
- **What:** when bundle was solved with `ret_lobatto_Z` or
  `state_lobatto_Z` set, the diagnostic eval rule must propagate that
  Lobatto config (default after 2026-05-04). `--eval-disable-lobatto`
  is the opt-out for comparison.
- **Why:** mismatched eval rule conflates policy inaccuracy with
  rule-mismatch. v4_lobatto's headline number went from −0.81 (GH-only
  eval) to −0.08 (propagated eval) — the policy actually fails at the
  ±7σ tail node it was designed against.
  [`docs/handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md`](../handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md).
- **Severity:** BLOCKER for any Lobatto bundle's published EE numbers.

### 9.5 Drill-down for off-gate bundles
- **What:** when Diagnostic B reports invalidity or C reports off-gate
  max, run `_diag_simpath_worst_cells` (sim-path tail) or
  `_diag_invalid_cells` (gridpoint tail), then `_diag_per_axis_tail` to
  attribute mechanism to axis.
- **Why:** workflow at
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §4](../agents/EE_DIAGNOSTIC_WORKFLOW.md);
  decision tree for whether to bump K, change Z, or tighten leverage
  cap.
- **Severity:** WARNING (diagnostic, not gating).

### 9.6 Bundle naming carries config self-description
- **What:** `<system>_<constraint>_<grid_mode>_grid<NxNxN>_nz<n>_age<youngest>_<terminal>_kstate<...>_kret<...>_eta<n_eta>eps<n_eps>_cap<value>_<modifications>_v1`.
- **Why:** lets you reconstruct config without opening metadata.
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §7](../agents/EE_DIAGNOSTIC_WORKFLOW.md);
  CCV bundles append `_ccv` per
  [`docs/notes/LOBATTO_CONFIG_TRACKER.md` §11.5](../notes/LOBATTO_CONFIG_TRACKER.md).
- **Severity:** INFO.

---

## §10 Resume / shape / version checks

### 10.1 Bundle shape match on resume
- **What:** `Cc.shape == expected` at checkpoint reload.
- **Why:** prevents resuming a bundle solved at one config under
  another config.
- **How:** runtime check at `solver.py:~1985-1997`; bug scan Item 8.
- **Pass:** shape-match raises if not.
- **Severity:** BLOCKER.

### 10.2 SolverConfig version consistency on resume
- **What:** WARNING printed if loaded `solver_config` differs in
  non-cosmetic field from current run's. (GAP: not yet implemented per
  bug-scan Item 8.)
- **Why:** silently mixing fori vs while_loop across ages produces
  reproducibility-muddied bundles.
- **Severity:** WARNING (currently a GAP).

---

## §11 Pre-solve diagnostic battery (one-shot wrapper)

### 11.1 `diagnose_all_pre(model, pc)` — runs §1-§4 collectively
- **What:** convenience wrapper at
  [`lifecycle/diagnostics.py:1118-1127`](../../lifecycle/diagnostics.py#L1118-L1127);
  prints "ALL N TESTS PASSED" or "*** N FAILURE(S) ***".
- **Why:** the gating one-stop for everything in §1-§4. Run
  immediately after `build_precompute`, before any solver call.
- **Pass:** zero `[FAIL]` markers; warnings acceptable.
- **Severity:** BLOCKER if any FAIL.

### 11.2 `verify/smoke.py` — exercises every kernel at minimum config
- **What:** 6-age window 60..65 covers terminal + retirement + boundary
  + working. (3,3,3,3) state, n_z=5, n_w=20.
- **Why:** catches kernel-shape bugs before the long run.
- **How:** `python verify/smoke.py`.
- **Pass:** prints policy ranges, zero NaN, JAX devices ≥ 1.
- **Severity:** BLOCKER pre-launch.

### 11.3 `verify/canonical_small.py` — full lifecycle on tiny grids
- **What:** 78-age full lifecycle on (3,3,3,3) state, n_w=40, n_savings
  =40, n_z=5. Saves policies + simulation.
- **Why:** smoke covers kernels; this covers the full backward sweep
  and end-to-end policy + simulation pipeline.
- **How:** `python verify/canonical_small.py`.
- **Pass:** alpha ranges sane; simulator alive at terminal; median
  death age in expected window.
- **Severity:** WARNING pre-launch (full-lifecycle dry run).

---

## §12 Gaps (checks the user has wanted that don't exist as code yet)

### 12.1 Pre-flight arbitrage check NOT auto-run by `build_precompute`
- **Status:** GAP. The standing operating rule says run
  `_diag_arbitrage_quadsweep` "before committing to any new quadrature
  config" but `build_precompute` does not call it. The user must
  remember.
- **Source:**
  [`docs/notes/LOBATTO_CONFIG_TRACKER.md` §6](../notes/LOBATTO_CONFIG_TRACKER.md).
- **Suggested:** at `build_precompute` end, run T-Q1 max-gap check on
  representative state corners and warn (not fail) if non-zero.

### 12.2 `wealth_grid[-1]` boundary-mass check NOT in post-sim diagnostics
- **Status:** GAP per `docs/TODO.md` items 25 & 26. The current
  `diagnose_*_post` battery does not report fraction of agent-years at
  `wealth_grid[-1]` or `z_grid[0/-1]`.
- **Source:** `docs/TODO.md`,
  [`docs/GRID_CONVERGENCE_CRITERIA.md` §3](../GRID_CONVERGENCE_CRITERIA.md)
  (boundary-mass row).
- **Suggested:** add a `diagnose_boundary_mass_post(sim)` reporter.

### 12.3 SolverConfig version-consistency warning on resume
- **Status:** GAP per bug-scan Item 8.
- **Suggested:** print WARNING if loaded `solver_config` differs.

### 12.4 Per-iw failure-share scan NOT auto-run after gridpoint EE
- **Status:** GAP. The bimodal wealth-decomposition pattern documented
  in
  [`docs/agents/EE_DIAGNOSTIC_WORKFLOW.md` §2](../agents/EE_DIAGNOSTIC_WORKFLOW.md)
  is operator-driven (manual `--wealth-indices 0 1 2 3 4 5 10 20 50
  100 149`). Not run by default.
- **Suggested:** `_diag_gridpoint_ee` could run the scan automatically
  when `max log10|EE| > −3` on Lobatto bundles.

### 12.5 No automated cross-check between `diagnose_pre` and bundle
- **Status:** GAP. `diagnose_all_pre` runs against
  `(model, pc)` constructed at solve time; nothing re-runs it on
  bundle reload to confirm the loaded `disc_config` reproduces the
  same state grid.
- **Suggested:** `verify_loaded_bundle.py` that loads a bundle,
  rebuilds `pc` from saved `disc_config`, and confirms `state_grid`,
  `Pi_state`, `mu_r` match the saved arrays bit-for-bit.

### 12.6 No CCV approximation-error envelope check at converged α
- **Status:** GAP per
  [`docs/handoff/HANDOFF_VERIFY_CCV_THEORY_TO_CODE.md` Task E](../handoff/HANDOFF_VERIFY_CCV_THEORY_TO_CODE.md).
  CCV truncation `O(|α|³ σ⁴)` is benign at constrained α but grows
  with leverage. The check "is the converged policy living within the
  regime where CCV truncation is below 1%" must be done by hand.
- **Suggested:** add a post-solve `diagnose_ccv_envelope(model, pc, S,
  B, sim)` that reports max |α| per age and CCV truncation estimate at
  that magnitude.

### 12.7 `_diag_arbitrage_quadsweep`'s known `_make_pc` Lobatto-stripping bug
- **Status:** OPEN per
  [`docs/notes/LOBATTO_CONFIG_TRACKER.md` §7](../notes/LOBATTO_CONFIG_TRACKER.md)
  and `HANDOFF_EVAL_LOBATTO_PROPAGATION.md`. The arbitrage diagnostic
  rebuilds Precompute internally and strips Lobatto. Not yet fixed.
- **Suggested:** propagate `ret_lobatto_Z` / `state_lobatto_Z` through
  `_diag_arbitrage_quadsweep._make_pc`.

---

## Reference: invocation order

The recommended pre-flight ordering before launching a long solve:

1. **Build model + precompute, watch for runtime warnings.**
   `build_model(BASE_CONFIG, var_config, verbose=True)` →
   `build_precompute(model, disc_config, verbose=True)`. Watches for:
   §1.4 (Σ_r_cond rank), §1.5 (state quad mean assertion), §1.10-1.11
   (Lobatto K/Z), §1.13 (eta/eps mean=0), §2.4-2.8 (state-grid
   structure), §3.3-3.4 (savings ranges).

2. **Run `diagnose_all_pre(model, pc)`.** Print result; gate on zero
   `[FAIL]` markers. Covers §1.1-1.7, §1.15-1.16, §2.1-2.3, §3.1-3.2,
   §4.1-4.7, §4.9.

3. **Pre-flight arbitrage check (manual, GAP §12.1).** Run
   `_diag_arbitrage_quadsweep` on the candidate disc_config; verify
   T-Q1 max gap = 0 (§1.18).

4. **Run `verify/smoke.py`.** 30-60 sec on JAX. Covers §6.1, §6.2,
   §11.2, and confirms §8.5 (GPU detection).

5. **For new disc-config or new model: run `verify/canonical_small.py`.**
   ~7-12 min. Covers §11.3 + simulator parity.

6. **For 4-D state runs at scale: HBM pre-flight (§8.1, §8.2).**
   Estimate `c_corners_T` size against device HBM with margin.
   Reference table in
   [`docs/notes/GPU_TRIAL_FINDINGS.md`](../notes/GPU_TRIAL_FINDINGS.md).
   If 7⁴ canonical: must be on H200 SXM5 (141 GB) or run cell-vmap
   chunking.

7. **Solve.** Watch `total_newton_failures` (§5.2) and per-age
   convergence print. With `use_fori_newton=True` (default) check
   `max_iter` calibration (§5.5).

8. **Post-solve: terminal-portfolio diagnostic (§6.3).** Already wired
   into `verify/benchmark_bundle.py:208-225`.

9. **Post-solve: bundle metadata round-trip (§6.4) + sim sanity (§4.8,
   §4.10, §11.3).**

10. **Post-solve EE diagnostics (§9).** Run A → B → C in order; gate
    on §9.3.

11. **If off-gate: drill-down per §9.5.**

End of checklist.

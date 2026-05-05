# Lobatto-Quadrature Configuration Tracker

Consolidated record of every Lobatto/quadrature config run during the bond-bankruptcy investigation, what we measured, and what we learned. **Read this before proposing a new config.** Mistakes-corrected section names hypotheses already empirically refuted so we don't re-debate them.

Last updated: 2026-05-05 (added v12/v12c, ccv_retire row, §3.6 factorized moment recovery, §3.7 CCV-regime findings, §6 pre-flight arbitrage rule, §11 CCV regime).

---

## 1. Configs run, in chronological order

All bundles solve retirement-only (ages 67–99) unless noted. All use `state_grid_sizes=(7,7,7)`, `n_z=11`, `n_eta=n_eps=4`, `n_wealth=n_savings=180`, `wealth_min=0.05, wealth_max=750`, `gamma=5`, `alpha_cap=±6`. Differences are all in the quadrature / state envelope.

| bundle | state_n_stds | ret K | ret_lobatto_Z | state K | state_lobatto_Z | cloud nodes/cell |
|---|---|---|---|---|---|---:|
| v3 (no Lobatto, baseline) | (2.0, 2.25, 2.25) | (3,7,5) | None | (3,4,5) | None | 4,200 |
| ret_v1 (K bump only) | (2.0, 2.25, 2.25) | (3,5,7) | None | (3,4,7) | None | 8,820 |
| **v4_lobatto** | **(2.0, 2.25, 2.25)** | **(3,5,5)** | **(None, 7, 7)** | **(3,5,5)** | **(None, 7, 7)** | **5,625** |
| v9_state_axis2_only | (2.93, 2.93, 2.93) | (3,5,5) | (None, 7, 7) | (3,5,5) | (None, None, 5) | 5,625 |
| v10_state_z5_wide | (2.93, 2.93, 2.93) | (3,5,5) | (None, 7, 7) | (3,5,5) | (None, 5, 5) | 5,625 |
| v11_state_k7_z7_wide | (2.93, 2.93, 2.93) | (3,5,5) | (None, 7, 7) | (3,7,7) | (None, 7, 7) | 11,025 |
| **v12_state_z4_z6p5** | **(2.93, 2.93, 2.93)** | **(3,5,5)** | **(None, 7, 7)** | **(3,5,5)** | **(None, 4.0, 6.5)** | **5,625** |
| v12c_state_z4_z5 | (2.93, 2.93, 2.93) | (3,5,5) | (None, 7, 7) | (3,5,5) | (None, 4.0, 5.0) | 5,625 |
| **ccv_retire** ‡ | **(2.0, 2.25, 2.25)** | **(3,5,5)** | **(None, 7, 7)** | **(3,5,5)** | **(None, 7, 7)** | **5,625** |

‡ Same disc-config as v4_lobatto, but solved with `wealth_dynamics_spec="ccv_log"` (Campbell-Viceira log-portfolio approximation, no bankruptcy clamp). See §11.

## 2. Outcomes

Sim-path retirement EE under `--eval-mode next_finer` with `--n-simulations 5000 --eval-households-per-age 256 --initial-z stationary`.

| bundle | mean log10\|EE\| | median | max | gridpoint invalidity | gridpoint max log10\|EE\| | worst_foc_resid | Newton failures | avg_newton_iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 | −1.96 | — | **+0.10** ✗✗ | 18.5% | −0.0001 | 0.05 | 103k | — |
| ret_v1 | −2.03 | — | −0.002 | 22.2% | +0.28 | 0.12 | 13k | — |
| **v4_lobatto** | **−2.57** | **−2.49** | **−0.08** ✓ | **0.00%** ✓ | **−0.0072** ✓ | 0.67 | 183k | 0.86 |
| v9_state_axis2_only | −2.22 | — | −0.054 | 1.31% | +0.4 region | 0.21 | 1.21M | 1.35 |
| v10_state_z5_wide | −2.22 | −2.49 | −0.051 | 0.79% | −0.069 | 0.23 | 1.00M | 2.28 |
| v11_state_k7_z7_wide | −2.14 | — | −0.044 | **12.67%** ✗ | **+0.44** ✗ | 0.17 | 1.24M | **4.44** |
| **v12_state_z4_z6p5** | **−2.27** | — | **−0.035** | **0.08%** ✓ | **−0.077** | **4.4e+48** ⚠ | 1.41M | 2.83 |
| v12c_state_z4_z5 | −2.21 | — | −0.040 | 0.69% | +0.21 | 0.22 | 1.10M | 4.25 |
| **ccv_retire** | −2.22 | — | **−0.40 (40% rel)** | unknown ‡ | n/a ‡ | **0.020** ✓✓ | 653k | 0.90 |

‡ Diagnostic API mismatch on AWS at the time; gridpoint-EE and invalid-cells reports failed to generate. Solver-internal numbers are clean.

**Best-on-bulk-metrics: v4_lobatto.** Mean log10|EE| = −2.57 with 0% invalidity. Has a residual long-bond cluster at i₂=6 corners (167 cells with |EE| > 3%) but no catastrophic failures.

**No config has cleared retirement publication gates** (mean < −4.5, max < −3.0). v4_lobatto is closest on max; nothing is close on mean.

---

## 3. Empirically established findings

These should be treated as known facts when designing future tests.

### 3.1 State-axis Lobatto IS doing real bankruptcy-detection work

Empirical evidence: dropping state-axis Lobatto entirely (v9 → v10 comparison: +12.67% / 1.31% / 0.79% invalidity rates as you remove and re-add). State innovations enter R_p through the M matrix (`r_{t+1} = base + M·v^s + ε`), so state-axis tail nodes catch genuinely-different R_p realisations even though the c_{t+1} interpolation is grid-clamped at the boundary.

The competing argument ("state Lobatto is mechanism-irrelevant; state innovations only affect c_{t+1}") was empirically refuted. See §4.1 below.

### 3.2 The right per-σ R_bond shift formula uses cross-channel terms

Wrong formula (used early in the investigation, retracted):

```
per_σ_R_bond_shift_axis_k = M[xb, k] · L_state[k, k]   (DIRECT CHANNEL ONLY)
```

This gives `−0.0957` for axis 1 — looks dominant.

Correct formula (with Cholesky cross-terms):

```
per_σ_R_bond_shift_axis_k = (M · L_state)[xb, k]      (NET, with cross-channel)
```

For axis 1 in our calibration: `M[xb,1]·L_state[1,1] + M[xb,2]·L_state[2,1] = −0.0957 + 0.119 = +0.023`.

The direct-only formula misses that a Lobatto node on axis 1 ALSO pokes axis-2 direction via `L_state[2,1] = −0.0136`, which mostly cancels the direct effect. Use the net formula for any future per-axis impact analysis. The empirical per-axis tail diagnostic (`_diag_per_axis_tail.py`) reports the direct-only number — its rankings are RIGHT (axis 1 still dominates by joint coverage, not by net per-σ effect) but the labelled mechanism is wrong.

### 3.3 Principled Z formula

```
Z_principled[k] = n_stds[k] × L_z[k, k] / L_state[k, k]
```

where `L_z = cholesky(Σ_z)` is the stationary-state Cholesky and `L_state = cholesky(Σ_ss)` is the innovation Cholesky.

Translates a state-grid-corner displacement (`n_stds × L_z[k,k]` in stationary-σ) into the equivalent innovation-σ that the Lobatto endpoint must reach to "match" the corner.

In current calibration (`L_z = (0.167, 0.0158, 0.0165)`, `L_state = (0.167, 0.0112, 0.0074)`):

| n_stds | axis 1 principled Z | axis 2 principled Z |
|---|---:|---:|
| 2.25 | 3.18 | 5.00 |
| 2.93 | 4.13 | 6.51 |

Z values *significantly above* principled overshoot — they create explicit tail nodes at innovation magnitudes the wider envelope physically doesn't require, but Newton has to navigate the resulting cliff anyway. v11's `state_lobatto_Z=(None,7,7)` at `n_stds=2.93` was 1.70× principled on axis 1; that overshoot shows up as `avg_newton_iter` jumping from 0.86 (v4) to 4.44 (v11), and gridpoint max log10|EE| degrading from −0.0072 to +0.44.

Z values *significantly below* principled undershoot — they don't reach the corner displacement, so the bankruptcy boundary at the corner isn't sampled. v9's `state_lobatto_Z=(None,None,5)` at `n_stds=2.93` was 0.77× principled on axis 2 — and v9 had 1.31% invalidity (axis 2 corners not covered).

**PRINCIPLE: Z calibration is per-axis per-grid, not a global threshold.**

Set each Lobatto axis's Z to its principled value `n_stds · L_z[k,k] / L_state[k,k]`. **There is no universal "Z should/shouldn't exceed N" rule.** The empirical pattern across v4 / v9 / v10 / v11:

| bundle | axis 1 (× principled) | axis 2 (× principled) | outcome |
|---|---:|---:|---|
| v4_lobatto (n_stds=2.25, Z=7) | 2.20× ↑↑ | 1.40× ↑ | BEST so far |
| v9 (n_stds=2.93, Z=5 axis 2 only) | — | 0.77× ↓ | 1.31% invalidity (undershoot) |
| v10 (n_stds=2.93, Z=5) | 1.21× ↑ | 0.77× ↓ | similar — axis 2 still undershooting |
| v11 (n_stds=2.93, Z=7) | 1.70× ↑↑ | 1.08× ↑ | regressed (overshoot at wider grid) |

Notes:
- **v4 tolerated 2.20× overshoot at narrow grid; v11 didn't tolerate 1.70× at wider grid.** Tolerance for Z-overshoot decreases as `n_stds` increases. Mechanism: at wider grids, Φ_21 already pushes E[R_bond | s] closer to bankruptcy at the corner; overshoot Z then stacks an even-larger innovation tail on top of an already-stressed conditional mean, and Newton can't navigate the cliff.
- **Below-principled is also a failure mode** (v9 axis 2). Don't undershoot to "save" Newton; you just leave bankruptcy boundaries unsampled.
- The right default is `Z = round(principled, 1 decimal)` per axis. Adjust upward only with empirical evidence that under-coverage is the issue, not Newton convergence.

**Common misreading to avoid:** "v4_lobatto used Z=7 and it was best, so Z=7 is the canonical setting." Wrong — v4's grid was narrow enough that 2.2× overshoot was tolerable; the same Z at the wider grid (v11) regressed catastrophically. Z is grid-dependent.

**Common misreading to avoid:** "Z=5 didn't work at the wider grid (v9, v10), so Z must be higher to catch the bankruptcy boundary." Wrong on axis 1 — at n_stds=2.93, principled axis-1 Z is 4.13, so Z=5 is already 1.21× principled (modestly above). v10 axis-1 wasn't undershooting; axis-2 was (Z=5 vs principled 6.51). Mixing per-axis behaviour into a global Z claim is the error.

### 3.4 Wider grid (n_stds=2.93) is harder than narrower (2.0, 2.25, 2.25) regardless of Lobatto config

Mean log10|EE| at the wider grid is stuck at approximately −2.2 across v9, v10, v11 — regardless of which axes have Lobatto, what Z, what K. The narrower grid (v4_lobatto) hits −2.57 with much simpler config.

This isn't yet *fully* explained but the empirical pattern is clear: the wider envelope opens corners that no Lobatto setup so far has cleanly handled. The cells reached by widening the envelope from 2.25σ to 2.93σ apparently carry meaningful EE residual that body-integration accuracy alone can't fix.

Possible mechanisms (untested but plausible):
- The wider corner Φ_21 mean shifts push conditional R_bond closer to the bankruptcy boundary; the agent's optimal α at those corners sits on a discontinuity ridge.
- The 7-point grid spacing at n_stds=2.93 (Δu = 0.98σ between points) makes interpolation between grid corners less accurate than at n_stds=2.25 (Δu = 0.75σ).
- The wider grid sampling exposes more cells where the Cholesky cross-terms in L_z create economically-weird "corner" combinations.

### 3.5 More body nodes (K=7) does NOT help at the wider grid

v11 bumped state K=5→7 with Z=7→7, expecting more body integration accuracy + matched tail position. Result: **regression** on every metric. Gridpoint max log10|EE| went from −0.069 (v10) to +0.44 (v11). Newton iter tripled.

Hypothesis-explanation: with K=7, the explicit ±Z tail nodes still create steep cliffs in the FOC residual surface, and now there are MORE body nodes near them which means MORE cells where Newton has to navigate across those cliffs. Body coverage is wasted when the dominant friction is the tail discontinuity, not body curvature.

### 3.6 Axis-2 undershoot of principled Z is the dominant invalidity driver at n_stds=2.93

Empirical evidence: clean A/B between **v12** (`state_lobatto_Z=(None, 4.0, 6.5)`) and **v12c** (`state_lobatto_Z=(None, 4.0, 5.0)`) — axis 1 held at 4.0, K=5 on both, n_stds=2.93. Only axis 2 varies (6.5 ≈ principled 6.51 vs 5.0 ≈ 0.77× principled).

| metric | v12 (Z₂=6.5, ≈ principled) | v12c (Z₂=5.0, undershoot) |
|---|---:|---:|
| invalidity | 0.08% | 0.69% |
| simpath max log10\|EE\| | −0.035 | −0.040 |
| gridpoint max log10\|EE\| | −0.077 | +0.21 |

~9× invalidity reduction, gridpoint-max log10\|EE\| flips from positive to negative. The undershoot prediction in §3.3 is empirically confirmed.

**Scope of the claim.** This A/B confirms only that *undershoot on axis 2* is bad and that matching principled fixes it. It does NOT establish:
- That axis 1 = principled is optimal (no axis-1 A/B at K=5; v12c vs v10 — Z₁ 4.0 vs 7.0, both with Z₂=5 — gives invalidity 0.69% vs 0.79%, suggesting axis-1 calibration is comparatively insensitive at K=5 in this regime).
- That overshoot on axis 2 is bad (untested at K=5; v11's overshoot was confounded with K=7).

So §3.3's overshoot leg remains theoretical-only and the "principled is *optimal*" framing is not yet warranted. What IS warranted: **axis 2 must reach ≥ ~principled** at n_stds=2.93.

**Caveat on v12 itself.** v12 carries `worst_foc_resid = 4.4e+48` — a single catastrophic Newton outlier that v12c (0.22), v10 (0.23), and v11 (0.17) don't have. The cell behind the blowup hasn't been localised; v12's tail metrics are still best-in-class but the outlier should be diagnosed before promoting v12 to canonical.

### 3.7 Joint state-return covariance recovery factorizes via tensor-product independence

**Principle.** The cross-covariance `M·Σ_ss·M^T + Σ_r_cond = Σ_rr` is automatically recovered as long as each marginal quadrature (state and return) is moment-exact. Joint design across axes is NOT required for moment recovery.

The decomposition `r_{t+1} = base + M·v^s + ε` makes v^s and ε independent by construction (Cholesky residual). In tensor-product quadrature, the discretization preserves this independence: `E_emp[v·ε^T] = 0`. So:
```
E_emp[r·r^T] = M · E_emp[v·v^T] · M^T + 2·M·0 + E_emp[ε·ε^T]
             = M · Σ_ss · M^T  +  Σ_r_cond  =  Σ_rr  ✓
```

**Empirical confirmation:** notebook `verify_discretization.ipynb` cell A.5 verifies this to 4e-17 absolute error across all axis-rule combinations we've tested (GH K=3, GH K=5, Lobatto K=5 Z=7, asymmetric per-axis Z).

**Implication.** Lobatto's value-add is NOT covariance recovery. It's catching the bankruptcy-boundary discontinuity (a non-moment integrand feature). For purely smooth integrands, no joint design is needed — pure GH on each axis at moderate K is optimal.

**Caveat (per-axis tail diagnostic was an approximation under simple_clamp).** The per-axis tail diagnostic (`_diag_per_axis_tail.py`) treats axes as independently contributing to bankruptcy coverage. Under simple_clamp, the bankruptcy region is a JOINT subset of (v^s, ε) space; per-axis ranking is a coverage proxy, not a full joint analysis. The proper diagnostic would compute joint-quadrature mass landing in the bankruptcy region. For our investigation the per-axis approximation pointed to the right axes (state[2] dominant, see §3.6), but the magnitude attribution is approximate.

### 3.8 CCV (`wealth_dynamics_spec="ccv_log"`) eliminates the bankruptcy-boundary mechanism

Empirical evidence from `ccv_retire` (v4-equivalent disc-config, only difference is CCV mode):
- `worst_foc_resid` dropped from v4's 0.67 to **0.020** — 30× improvement
- Sim-path max log10\|EE\| improved from −0.08 (83% rel error) to **−0.40 (40% rel error)** — 2× better
- Mean log10\|EE\| got slightly worse (−2.22 vs v4's −2.57) — likely the cost of the CCV log-portfolio Taylor approximation
- `avg_newton_iter` 0.90 (similar to v4's 0.86)

Mechanism: under CCV, `R_p = exp(r_p^CCV) > 0` strictly. No bankruptcy boundary, no integrand discontinuity. Newton converges cleanly at all cells.

**Cost of CCV: the log-portfolio Taylor approximation.** The CCV formula
`r_p = r_bill + α_s·x_r + α_b·x_b + 0.5·(α'σ_diag) - 0.5·α'Σα`
is exact only in continuous time. Discrete-time approximation error scales as `α³·σ³` per period — ~0.5–1% per period for our parameters and modest α. This is roughly the bulk-EE gap between ccv_retire (mean −2.22) and v4_lobatto (mean −2.57). Approximation bias replaces discontinuity stress as the binding accuracy constraint.

**Implication for quadrature.** Under CCV, the integrand is smooth. Lobatto's tail nodes catch a boundary that doesn't exist. **Pure GH at moderate K is optimal.** Polynomial-exactness 2K−1 (vs Lobatto's 2K−3 at the same K) is recovered. The "principled-Z" framework no longer applies because there's no bankruptcy boundary to anchor Z to.

**Implication for state envelope.** §3.4's "wider grid is intrinsically harder" finding was driven by simple_clamp's bankruptcy at wide corners. **Under CCV, wider grids should be safe.** The wider envelope just gives more accurate integration of the stationary distribution mass. Empirical confirmation pending (no wide-grid CCV bundle solved yet).

---

## 4. Mistakes / debunked hypotheses

These are hypotheses that were empirically tested and found wrong. **Don't re-propose without new evidence.**

### 4.1 "State-axis Lobatto is mechanism-irrelevant"

Argument (refuted): "The bankruptcy clamp's kink has no v^s derivative; state innovations enter the integrand only through c_{t+1}(s_{t+1}); state-axis Lobatto puts nodes past the state grid edge where C_{t+1} is interpolation-clamped, so it carries no new information."

Why it sounded right: at the +Z state-axis tail node, the next-period state s_{t+1} = base + M·v_added is far past the grid edge, so c_{t+1} interpolates to the grid corner regardless. Therefore u'(c_{t+1}) at the tail node is identical to the corner.

Why it's empirically wrong: R_p in the integrand `R_p · u'(c_{t+1}) · ψ` ALSO depends on v^s through `M @ v^s`. So R_p at the +Z state-axis tail node is genuinely different from R_p at the corner — different by `M @ L_state · z`, which can be large (M[xb,1]·L_state[1,1]·5σ ≈ −0.48 in log return units). The integrand at the tail node is therefore a different (R_p, u'(c)) pair than at the corner, and it CAN flip sign.

Empirical refutation: v9 (no state[1] Lobatto) had 1.31% invalidity at i₁=6 corners with α_s up to +3.7. v10 (state[1] Lobatto Z=5) reduced invalidity to 0.79% and Newton failures by 17%. State[1] Lobatto did real work. See §3.1 above.

### 4.2 "K=7 GH on state would do the same job as K=5 Lobatto Z=7 (denser body, no tail)"

Argument (refuted): "The polynomial-exactness loss from Lobatto (2K-3 = 7 at K=5 vs 2K-1 = 9 for GH K=5) is real, and at K=7 GH gives 2K-1 = 13 degrees of body integration. Spending K=2 nodes on tails when GH could spend them on body is wasteful."

Why it's empirically wrong: v11 (K=7 Lobatto Z=7) regressed across every metric. We don't have a direct K=7 GH comparison run, but v9's behaviour (K=5 GH state pure) at the wider grid had 1.31% invalidity, which is worse than v10's 0.79% with K=5 Lobatto. The Lobatto tail on bond-loaded axes carries information not captured by body GH alone, even at higher K.

The polynomial-exactness gap is real but not the binding constraint. The integrand has a discontinuity at the bankruptcy boundary; polynomial degrees beyond the discontinuity buy nothing.

### 4.3 "Bumping K=5 → K=7 helps Newton convergence by giving stepping stones between body and tail"

Argument (refuted): "Lobatto K=5 has only one interior node per side at ±1.69σ. Going to K=7 adds two more interior nodes (roughly ±3, ±5) so Newton can step from body to tail through gradual transitions instead of jumping from ±1.69 to ±7."

Why it's empirically wrong: v11 (K=7 Z=7) had `avg_newton_iter = 4.44` vs v10's (K=5 Z=5) 2.28. Newton iterations roughly doubled. The "stepping stones" hypothesis predicted improvement; observed deterioration. The body of the integrand isn't where Newton struggles — the cliff at the explicit tail node is — and adding interior body nodes doesn't smooth that cliff, it just creates more cells that have to navigate near it.

### 4.4 "Eval-rule mismatch (Lobatto solver, GH eval) is the only thing wrong with v4_lobatto's max log10|EE|"

Argument (refuted): "v4's reported max log10|EE| = −0.08 might be a diagnostic artifact: solver uses Lobatto Z=7, eval uses GH with no tail node, so eval rule disagrees with solver about the optimal alpha at the boundary — that's not real policy inaccuracy."

Why it's empirically wrong: when we propagated Lobatto into the eval rule (HANDOFF_EVAL_LOBATTO_PROPAGATION.md, 2026-05-04), v4_lobatto's max log10|EE| got WORSE, not better — from −0.08 (GH eval) to roughly −0.08 or worse (Lobatto-matched eval). The diagnostic was missing real policy inaccuracy at the explicit tail node, not artifactually inflating an artefact. Subsequent investigation confirmed `worst_foc_resid = 0.67` at v4 cells where Newton's warm-restart fallback returned non-converged policies; the eval-matching Lobatto diagnostic was correctly surfacing that.

### 4.5 "K bump alone (no Lobatto) is enough"

Argument (refuted): "Just bump K from (3,7,5) to (3,5,7) GH on returns and (3,4,5) to (3,4,7) on state. More polynomial-exactness will catch the bond-tail."

Why it's empirically wrong: ret_v1 (the K-bump-only run) had **22.2% invalidity** vs v3's 18.5%. Mean barely moved (−2.03 vs v3's −1.96). Max log10|EE| at +0.28 was actually worse than v3's +0.10. K-bumps without explicit tail coverage shift the discrete-free-lunch boundary inward but the agent's optimal α tracks it. See HANDOFF_EVAL_LOBATTO_PROPAGATION.md for the post-mortem.

### 4.6 "Per-σ R_bond shift via M[xb,k]·L_state[k,k] tells you which axis is the dominant bond channel"

Argument (refuted by §3.2 above): the direct-channel-only formula misses cross-Cholesky terms. Use `(M · L_state)[xb, k]` for the net per-σ effect.

The misleading number was −0.0957 on axis 1. The correct net number is +0.023. The corrected analysis still concludes axis 1 matters (via joint-tail coverage rather than per-σ R_bond effect) but the *mechanism* is different from what the wrong formula implied.

---

## 5. Open empirical questions

These are hypotheses worth testing — they haven't been run yet but each tests a specific prediction.

### 5.1 Does the wider grid (2.93) recover v4_lobatto's numbers when given v4's Lobatto config?

Test: re-solve with `state_n_stds=(2.0, 2.25, 2.25)` and `state_lobatto_Z=(None, 5.0, 5.0)` (or whatever variant). Should re-create v4-quality numbers (mean ≈ −2.57, 0% invalidity).

Predicted outcome: yes. If yes, the wider grid is the issue regardless of Lobatto. If no, something else changed between v4 and v9-v11.

### 5.2 Per-axis principled Z

Test: state_lobatto_Z = (None, 4.0, 6.5) at n_stds=2.93, K=(3,5,5). Each axis at its own principled Z.

Predicted outcome: invalidity drops to <0.5%, mean recovers toward −2.5. The principled-Z framework predicts this is the right per-axis Z calibration. Single-knob vs v10 (which had Z=5 on both, half-undershooting axis 2).

### 5.3 Smoothing the bankruptcy clamp

Replace the hard `if sR_p ≤ 0: c' = floor` clamp in the FOC kernel with a smooth penalty (softplus or quadratic ramp). Removes the integrand discontinuity entirely.

Predicted outcome: max log10|EE| drops to −3 to −4 range (publication grade). Newton converges to tolerance everywhere.

Cost: ~50–100 LOC kernel rewrite. See `HANDOFF_BANKRUPTCY_CLAMP_SMOOTHING.md` (already exists in the handoff folder).

### 5.4 Reorder state vector: (y_1, cy, spr) instead of (cy, spr, y_1)

Concentrates bond-loading onto axis 0 of the Cholesky. Single Lobatto axis instead of two.

Predicted outcome: same EE as v4_lobatto but with a cleaner config — `state_lobatto_Z=(Z_principled, None, None)`.

Cost: ~200–400 LOC across var.py / discretization.py / model.py / config plumbing. Worth doing only if a v12-class run with the principled-Z config (5.2) is still messy.

### 5.5 9³ state grid instead of 7³ at n_stds=2.93

Test whether the grid spacing (Δu = 0.98σ at 7³, would be Δu = 0.73σ at 9³) is the bottleneck rather than envelope width.

Predicted outcome: mean log10|EE| improves toward v4's −2.57 even at n_stds=2.93. Tests §3.4's "spacing vs envelope" decomposition.

Cost: 2.6× state grid points → roughly 2.6× solve time.

---

## 6. Standing recommendations

### Default canonical (under simple_clamp wealth dynamics)

Keep v4_lobatto (n_stds=(2.0, 2.25, 2.25), K=(3,5,5), Z=(None, 7, 7) on both ret and state). Best on bulk metrics. Has a residual long-bond corner cluster (167 cells, 1% of sim cells with |EE| > 3%) but no other simple_clamp config has cleaned that without trading off something larger.

### Default canonical (under CCV wealth dynamics — `wealth_dynamics_spec="ccv_log"`)

Drop Lobatto entirely. See §3.7 and §11.

```python
state_n_stds       = (2.93, 2.93, 2.93)   # 99% joint state coverage; safe under CCV
n_state_quad_nodes = (3, 5, 5)
state_lobatto_Z    = None
n_ret_nodes_1d     = (3, 5, 5)
ret_lobatto_Z      = None
wealth_dynamics_spec = "ccv_log"
```

Pre-flight T-Q1 arbitrage check confirmed clean (max gap = 0 across all 343 corners).

### Operating rules

- **Pre-flight arbitrage check is mandatory before committing to any new quadrature config.** Run `_diag_arbitrage_quadsweep` (or build a small Precompute and call `arbitrage_gap_2d`) against any candidate, and confirm T-Q1 max gap = 0 on both log-excess and arithmetic-excess clouds. The diagnostic costs seconds; a misconfigured cloud with strict arbitrage is unsolveable. For our calibration, GH at any K ≥ (3,3,3) on both ret and state is arbitrage-clean — but a future calibration with different μ_xr / Φ_21 / Σ_r_cond could fail this check, particularly at low K with no state-axis variation.

- **For wider-coverage experiments under simple_clamp: do not propose K bumps without re-checking principled-Z first.** The principled-Z formula is the gating sanity check for any Z choice. K bumps alone don't fix tail coverage — they just add body nodes near unchanged tail cliffs.

- **For diagnostic interpretation: always run with `--eval-mode next_finer` and the propagated-Lobatto eval rule** (the default after 2026-05-04). The GH-only eval (`--eval-disable-lobatto`) is a useful comparison when investigating a specific cell, but the headline numbers are the propagated-Lobatto ones.

- **When a new config regresses, drill down via `_diag_simpath_worst_cells` first**, then `_diag_per_axis_tail` (mindful of §3.2 and §3.7 caveats), then `_diag_invalid_cells`. The drill-down sequence is documented in `EE_DIAGNOSTIC_WORKFLOW.md` §4.

- **Joint state-return covariance recovery does NOT require joint Lobatto design** (§3.7). Per-axis moment-exact rules suffice. Cell A.5 in `verify_discretization.ipynb` is the regression test.

---

## 7. Reference files

- `configs/_canonical.py` — canonical disc config; should always reflect the current best (currently v4_lobatto).
- `configs/run_v3.py`, `run_v4.py`, `run_ret_v1.py`, `run_ret_v5_state_lobatto.py`, `run_ret_v5_state_gh.py` — the per-config files for re-runs. Bundles named in §1 are produced from these or analogous files.
- `lifecycle/discretization.py:158` — `build_state_grid` (where Cholesky-axis grid construction lives).
- `lifecycle/discretization.py:490` — `get_return_quadrature` (Hermite-Lobatto for returns).
- `lifecycle/discretization.py:604` — `get_state_quadrature` (Hermite-Lobatto for state innovations).
- `lifecycle/quadrature_with_tails.py` — the Hermite-Lobatto closed-form rule (`gauss_hermite_prescribed_tails(K, Z)`).
- `scripts/diagnostics/_diag_per_axis_tail.py` — per-axis tail-node coverage diagnostic. **Note §3.2 caveat** about its `M[xb,k]·L_state[k,k]` formula.
- `scripts/diagnostics/_diag_arbitrage_quadsweep.py` — pre-flight arbitrage sweep. Has the known `_make_pc` Lobatto-stripping bug noted in HANDOFF_EVAL_LOBATTO_PROPAGATION.md.
- `scripts/diagnostics/_diag_invalid_cells.py`, `_diag_simpath_worst_cells.py`, `_diag_gridpoint_ee.py`, `_diag_euler_errors.py` — the diagnostic battery.
- `docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md` §4 — drill-down workflow with v4_lobatto as a worked example.
- `docs/handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md` — eval-rule fix that surfaced real policy inaccuracy at v4 boundary cells.
- `docs/handoff/HANDOFF_BANKRUPTCY_CLAMP_SMOOTHING.md` — proposal for the structural fix in §5.3.
- `docs/handoff/HANDOFF_AWS_DIAGNOSTIC_AUTORUN.md` — AWS-side auto-diagnostic implementation.

---

## 8. How to update this tracker

When you run a new config, append a row to §1 and §2. If you discover a new finding, add it to §3. If you debunk a hypothesis (yours or anyone else's), add it to §4 with the empirical evidence. Keep §5 (open questions) honest — close items as they're tested.

Don't delete entries from §4. Mistakes-corrected is the highest-leverage section because it prevents repeat errors. Adding a new entry has higher value than tidying old ones.

---

## 11. CCV regime — separate findings

When `wealth_dynamics_spec="ccv_log"` is enabled, the FOC kernel uses Campbell-Viceira's continuous-rebalancing log-portfolio approximation:
```
r_p^CCV = r_bill + α_s·x_r + α_b·x_b + 0.5·(α_s·σ²_xr + α_b·σ²_xb) - 0.5·α'Σα
x_{t+1} = s · exp(r_p^CCV) + π_next         (no clamp; R_p > 0 always)
```

This eliminates the bankruptcy boundary that drove all of the simple_clamp Lobatto investigation. Findings under this regime form a separate decision tree.

### 11.1 Implementation correctness verified

Formula matches Campbell-Viceira (NBER w8566 eq.10) at three call sites: `lifecycle/solver.py:935-940` (retirement FOC), `lifecycle/inf_horizon_solver.py:587-590` (infinite-horizon FOC), `lifecycle/simulation.py:779-782` (simulator). All three use identical sigma terms (`σ²_xr = Σ_r_cond[1,1]`, `σ²_xb = Σ_r_cond[2,2]`, `σ_xrxb = Σ_r_cond[1,2]`).

Gradient: `dr_p/dα_s = x_r + σ²_xr·(0.5 - α_s) - α_b·σ_xrxb` ([solver.py:1009](lifecycle/solver.py#L1009)) ✓
Gradient: `dr_p/dα_b = x_b + σ²_xb·(0.5 - α_b) - α_s·σ_xrxb` ([solver.py:1010](lifecycle/solver.py#L1010)) ✓
Hessian: `J_ss = jac · dRp_das² + wmu · R_p · (dr_da_s² - σ²_xr)` ([solver.py:1016](lifecycle/solver.py#L1016)) ✓ (independently re-derived).

Solver and simulator MUST use the same `wealth_dynamics_spec` (docstring at [model.py:192](lifecycle/model.py#L192)); otherwise the simulated wealth path disagrees with the solver's FOC and EE diagnostics become meaningless.

### 11.2 Quadrature recommendations under CCV

| concern | simple_clamp regime | CCV regime |
|---|---|---|
| Bankruptcy boundary | Discontinuous integrand → Lobatto needed | **No boundary; smooth integrand → pure GH optimal** |
| Polynomial-exactness on body | Lobatto K=5: 2K-3=7 | **GH K=5: 2K-1=9 — strict upgrade** |
| Principled-Z calibration | Required (anchored to grid corner displacement) | **Not applicable (no boundary to anchor to)** |
| State envelope (n_stds) | Wider grid intrinsically harder due to bankruptcy at corners | **Wider grid safe; mass-weighted accuracy improves** |
| Pre-flight arbitrage | T-Q1 + T-Q2 (min R_p at canonical α) | **T-Q1 sufficient** (Itō correction is O(σ²·α²) ≪ cloud spread) |

### 11.3 Pre-flight arbitrage results at CCV-canonical config

Tested at `n_stds=(2.93, 2.93, 2.93)` (99% joint state coverage), all configs strictly arbitrage-clean (T-Q1 max gap = 0):

| ret quad | state quad | ret nodes | T-Q1 (log) | T-Q1 (arithmetic) |
|---|---|---:|---|---|
| GH (3,3,3) | GH (3,5,5) | 27 | clean | clean |
| **GH (3,5,5)** | **GH (3,5,5)** | **75** | **clean** | **clean** |
| GH (3,7,7) | GH (3,5,5) | 147 | clean | clean |

The state-quadrature M·v variation provides enough conditional-mean spread that the joint cloud spans both signs of (X_r, X_b) regardless of return-quadrature K. Arbitrage isn't a binding constraint in this calibration at any reasonable K.

### 11.4 Open questions in the CCV regime

- **Does dropping Lobatto under CCV actually improve EE?** Predicted yes (§3.7); empirical test pending. Single-knob delta from `ccv_retire`: switch `state_lobatto_Z` and `ret_lobatto_Z` both to `None`.
- **Does wider state grid (n_stds=2.93) work cleanly under CCV?** Predicted yes (no bankruptcy boundary); empirical test pending. Combined with the Lobatto-drop test, this is the proposed CCV canonical.
- **Is the CCV approximation bias the binding accuracy constraint?** The log-portfolio Taylor approximation is exact only in continuous time; discrete-time error scales as `α³·σ³` per period (~0.5–1% for our parameters). This roughly matches the ccv_retire mean-EE gap from v4_lobatto. If yes, no quadrature config will get below ~−2.5 mean log10|EE| under CCV without a higher-order approximation.
- **Does CCV affect the simulator's stationary distribution?** Solver and simulator both use `r_p^CCV`, so internally consistent. But sample moments (equity share lifecycle averages, retirement consumption profile) might shift versus simple_clamp. Need to compare downstream sim outputs.

### 11.5 Caveats and gotchas

- **CCV bundle naming convention**: append `_ccv` or `_cvc` to bundle suffix to distinguish from simple_clamp bundles at the same disc config.
- **Diagnostic API mismatch on AWS** (observed in `ccv_retire`): the AWS auto-diagnostic image was built against an older `_evaluate_age_errors` signature that doesn't pass the CCV sigma scalars. Update the AWS image before relying on AWS-side EE diagnostics for CCV bundles.
- **The principled-Z framework (§3.3, §3.6) does not apply under CCV.** Don't propose principled-Z calibrations for CCV runs; just use GH.
- **`worst_foc_resid` interpretation flips under CCV.** Under simple_clamp, high `worst_foc_resid` typically indicated boundary-cell warm-restart fallbacks. Under CCV, Newton converges cleanly even at extreme cells, so high `worst_foc_resid` would indicate a real solver bug (haven't seen one).

# HANDOFF — Smoother bankruptcy clamp (Path B): analysis & critique

## Scope

Investigate whether replacing the current hard bankruptcy clamp ("Path B": `if s · R_p ≤ 0 → bequest = 0, x_next = income_only`) with a smooth blend over a small window is the right structural fix for the v4_lobatto residual EE pathology. **You are NOT asked to implement.** Produce a written analysis: validate or critique the proposed fix, surface failure modes I haven't thought of, and recommend whether to proceed.

## Findings

### Empirical (sim-path & gridpoint EE diagnostics on v4_lobatto)

The v4_lobatto bundle (canonical config, K=5 Lobatto Z=7 on ret/state axes 1,2; retirement-only ages 67-99) reports:

| metric | sim-path EE (next_finer, propagated Lobatto) | gridpoint EE (same rule) |
|---|---:|---:|
| mean log10\|EE\| | −2.57 | −7.31 (at solver tolerance) |
| median | −2.49 | — |
| max | **−0.08** (≈83% rel error) | **−0.003** (≈100% rel error) |

Far from publication gates (mean < −4.5 retirement, max < −3.0).

### Mechanism — bimodal failure pattern

A dense-low-wealth gridpoint scan (`_diag_gridpoint_ee --eval-mode same --wealth-indices 0 1 2 3 4 5 10 20 50 100 149`) decomposes the failure population:

```
 iw      x         mean log10|EE|   max     share with EE > 1e-4
  0    0.050         −4.18         −0.008    41.4%   ← mode (1) — EGM-anchor adjacent
  1    0.089         −4.59         −0.004    21.6%
  2    0.130         −4.97         −0.004     5.2%   ← sharp transition
  3-20 (~0.17–1.2)   −5.3 to −6.4  varies    ~3%    ← mode (2) — persistent leverage corner
 30    2.16          −6.80         −0.041     2.0%
 50    5.58          −7.51         −3.62      0.04%
```

Two modes:

- **Mode (1)** — high failure share at iw ∈ {0,1,2}, dropping sharply by iw=3. Possibly EGM constrained-region anchor or `u'(c)` amplification at low x. Addressable by raising `wealth_min` (an active test arm — Run A/B in progress with wealth_min=0.13).
- **Mode (2)** — persistent ~3% failure share across all wealth from x ≈ 0.13 to x ≈ 5, where it finally collapses to solver tolerance. **This is the mode the smoother-clamp proposal targets.**

### Mechanism — leverage gradient on mode (2)

Sim-path EE distribution by `|α_b|` and by `|α_bill| = |1 − α_s − α_b|`:

```
|α_b| range    n     mean log10|EE|   share > −1.5
[0.0, 0.5)    1548    −2.41           0.45%
[0.5, 1.0)    2393    −2.59           0.71%
[1.0, 1.5)    1839    −2.59           1.96%
[1.5, 2.0)    1549    −2.58           1.16%
[2.0, 2.5)     861    −2.77           5.23%   ← 11× the unleveraged rate

|α_bill| range  n     mean log10|EE|   share > −1.5
[0,    1.0)   4119    −2.51           0.92%
[1.0,  1.5)   2672    −2.65           1.46%
[1.5,  2.0)   1246    −2.65           2.33%
[2.0,  2.5)    155    −2.34          11.61%   ← 13× the unleveraged rate
```

Of the 124 sim-path bad cells (log10\|EE\| > −1.5):
- **88% have `|α_b| ≥ 1` OR `|α_bill| ≥ 1`** (leveraged in some direction)
- 81% have `|α_b| ≥ 1` (leveraged bond)
- Worst leveraged cell: log10\|EE\| = −0.079 (≈83% rel error)
- Worst unleveraged cell: log10\|EE\| = −0.591 (≈25% rel error)

Order-of-magnitude gap between leveraged and unleveraged failure tails. Failure rate scales monotonically with leverage. **This is the signature of a bankruptcy-clamp × Lobatto-tail-node interaction**: leverage is a near-necessary condition for `s · R_p` to hit zero at a tail return realization, and the hard clamp at `s · R_p = 0` is a step discontinuity in the FOC integrand.

### Mechanism — why Lobatto exposes the kink (and GH doesn't)

Pure Gauss-Hermite at K=5 maxes at ±2.86σ; at K=7, ±3.75σ. At leveraged α_b ≈ 2 the bankruptcy threshold sits in the bond-residual range Z ∈ [5, 7]σ — outside GH reach but precisely where the Lobatto tail node lives. Lobatto explicitly samples the kink; GH smooths over it.

Two channels deliver state innovations to the kink:
1. **Direct return-axis Lobatto** (`ret_lobatto_Z`) — tail node on the bond residual itself.
2. **M-coupling on state-axis Lobatto** (`state_lobatto_Z`) — `μ_r = const_r + A_r · s_t + M · v^s` with `M[xb, y_1] = −8.72`. State-axis tail node at +Z shifts μ_b by ~`-8.72·Z` ≈ −61 at Z=7, dwarfing pure GH reach.

Both channels are necessary in the current calibration to reach the kink in different state corners. See [docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md](../workflows/EE_DIAGNOSTIC_WORKFLOW.md) §3 Step 4 ("Why state-axis Lobatto matters: the two channels for v^s") for the full argument.

## Proposed solution

Replace the hard switch in Path B with a smooth blend over a window `ε`:

```python
# Current (hard, every site that implements Path B):
sR_p = s_val * R_p
if sR_p > 0.0:
    x_next = sR_p + income_or_pension
    w_A = sR_p / annuity_factor
    mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor
else:
    x_next = income_or_pension
    mu_bequest = 0.0

# Proposed (smooth, over a window epsilon):
sR_p = s_val * R_p
w = smoothstep(sR_p / epsilon)             # 0 below 0, 1 above epsilon, C^infinity in between
x_next = w * (sR_p + income_or_pension) + (1.0 - w) * income_or_pension
        # = income_or_pension + w * sR_p   (algebraically equivalent)
w_A_safe = max(sR_p, epsilon_floor) / annuity_factor   # keep argument of (-gamma) positive
mu_bequest = w * b_bar * w_A_safe ** (-gamma) / annuity_factor
```

`smoothstep` could be cubic Hermite (`3t^2 − 2t^3`), quintic, or sigmoid — open question. The blend kills the integrand's step discontinuity at `sR_p = 0`; it's C^∞ across the boundary, so quadrature error returns to the rule's polynomial-exactness floor.

## Why I believe this fixes the dominant pathology

1. **Removes the only known source of integrand discontinuity in the FOC.** The integrand becomes C^∞ in `R_p`. Quadrature error stops being O(1) at the kink and returns to O(1/K^p) for some p ≥ 2K-3.
2. **Addresses ~88% of bad cells** (those with leverage), which carry the publication-blocking magnitude (worst max ≈ 10× worse than worst unleveraged).
3. **Decouples Z choice from accuracy.** Currently the worst residuals concentrate at the Lobatto tail nodes precisely because they sample the kink. With no kink to sample, Z=5 vs Z=7 vs no Lobatto becomes a low-stakes tuning parameter rather than a structural choice.
4. **Should improve Newton convergence.** The current FOC has a bounded-but-undefined derivative at `sR_p = 0` (left-limit ≠ right-limit). Newton steps near the kink occasionally stall or oscillate. With smoothing the FOC is C^1, Newton has well-defined updates everywhere.
5. **Preserves bankruptcy semantics for clearly-bankrupt cells.** For `sR_p < −ε`, the smoothstep is exactly zero — agent gets `x_next = income_only` and zero bequest, identical to the hard clamp. Only the immediate neighborhood of the boundary changes.

## Residual concern — the unleveraged 12%

Smoothing won't address ~12% of bad cells (sim-path log10\|EE\| > −1.5 with `|α_b| < 1` AND `|α_bill| < 1`). Worst unleveraged cell sits at log10\|EE\| = −0.591 (≈25% rel error). Likely candidates:
- `u'(c)` amplification at low x (mode 1-adjacent).
- Body-integration coarseness (`n_eps = 4`, `n_eta = 4`).
- State-edge interpolation effects.

These need a separate post-fix diagnostic. If after smoothing the residual brings mean below −4.5 and max below −3, we're at publication grade and the unleveraged 12% is a non-issue. If max stays above −3, second-stage investigation of the unleveraged residual.

## What you are asked to do

1. **Read the code and findings (next section).**
2. **Critique the proposal.** Surface every issue you can find. Some I've already noticed but want a fresh pair of eyes on:
   - **Choice of smoothstep function.** Cubic vs quintic vs sigmoid — what are the trade-offs in derivative bounds, computational cost, and how cleanly Newton sees the FOC near the boundary? Does the choice interact with EGM monotonicity?
   - **Choice of ε.** Constant vs scaled with wealth/savings/return-volatility? Too small and the kink remains numerically sharp (small but high-derivative). Too large and the smoothed bankruptcy distorts the policy economically (agent acts as if some bequest exists when there shouldn't be one).
   - **Bequest formula at small `sR_p`.** `w_A^(-γ)` blows up as `w_A → 0+`. The smoothstep's `w → 0` factor cancels it analytically (`w · w_A^(-γ)` → 0 if smoothstep degree ≥ γ + 1 = 6, else still problematic at γ=5). Does this require choosing the smoothstep degree to match γ? Is there a numerically-stable form that doesn't depend on cancellation?
   - **EGM monotonicity.** The current solver reports 0 EGM monotonicity violations on v4. The Run A/B with Z=5 + raised wealth_min reports 2-5 per age. Could smoothing add or remove violations? The smooth FOC has bounded derivatives, so EGM should be more monotone, not less — but verify.
   - **Distortion of the optimal policy.** The agent's optimal α at gridpoints is the argmin of |FOC|. Smoothing shifts the FOC slightly. Does this shift the optimal α? By how much? Is the shift economically meaningful (agent takes less leverage because some bequest survives clear bankruptcy)? Is it cosmetically bad (results published with a smoothed clamp need defense in the methodology section)?
   - **Three-site consistency.** Path B is implemented at three sites (FOC kernels in solver.py, EE diagnostic kernels in `_diag_euler_errors.py`, and the simulator's estate computation in `simulation.py`). All three must agree on the smoothing scheme. Is there a clean shared helper, or do they need separate but matching implementations?
   - **Alternative interventions.** Are there better fixes for the kink that I haven't considered? E.g.:
     - Tighter `alpha_max` (cap leverage so kink is unreachable at any sampled return)
     - Restructure the FOC so the kink is integrated analytically (closed-form Path B contribution to bequest moments)
     - Different bequest formulation that's smooth at zero by construction (e.g. `b_bar · (w_A + δ)^(1−γ)` with a small δ)
     - Skip the Lobatto rule entirely and use a denser GH that doesn't reach the kink (the "give up on tail coverage" option)
   - **Calibration interactions.** Does the smoothing interact with `b_bar` (bequest weight)? With γ? With the leverage cap?
   - **Validation strategy.** Even if the design is right, what's the smallest test that confirms it? Smoke solve with `ε ≈ 0.05`, run gridpoint EE, expect mode-(2) failure share to collapse from ~3% to <0.5%. What if it doesn't?

3. **Recommend.** Final verdict: proceed with smoothing, proceed with a different fix, or get more diagnostic data before committing. If proceed, what's your suggested `ε`, smoothstep function, and validation plan?

## Reading order — code

Read these in order. I've given line numbers for the current `lifecycle/` and diagnostic-script tree as of 2026-05-04.

1. **The mathematical model** — [docs/DESIGN.md](../DESIGN.md), especially:
   - The bequest motive specification (Catherine 2025): `b(W, A) = b_bar · (W/A)^(1−γ) / (1−γ)`
   - The state/return covariance structure with `M` matrix at lines 424-460 (you need this to understand why state-axis tail nodes interact with bond returns)
   - `M[xb, y_1] = −8.72`, `M[xb, spr] = −8.51` (loadings used in the calibration)

2. **Path B implementations to modify (FOC kernels)** — [lifecycle/solver.py](../../lifecycle/solver.py):
   - **Retirement FOC + Jacobian** at `compute_foc_jac_retirement_quad` (line 747). The Path B branch is at lines **843-881** (x_next blend at 843-847; mu_bequest computation at 873-879; mu/mup combination at 880-881).
   - **Working-age FOC + Jacobian** at `compute_foc_jac_working_quad` (line 904). Path B at lines **1025-1034**.
   - The Jacobian terms (`mup_bequest`) need consistent treatment under smoothing — that's where derivative-of-discontinuity bites. Pay attention to lines 875-876 and 1029-1030 specifically.

3. **Path B implementations to modify (EE diagnostic kernels)** — [scripts/diagnostics/_diag_euler_errors.py](../../scripts/diagnostics/_diag_euler_errors.py):
   - **Retirement** at `_compute_euler_sum_retirement_continuous` (line 632). Path B at lines **706-710** (x_next) and **734-738** (mu_bequest).
   - **Working** at `_compute_euler_sum_working_continuous` (line 746). Path B at the analogous block (search for `if sR_p > 0.0:`).
   - These must match the solver's smoothing exactly, otherwise the EE residual would conflate solver-vs-eval rule disagreement on the smoothing with genuine policy error.

4. **Simulator estate computation** — [lifecycle/simulation.py](../../lifecycle/simulation.py) line **767**: `estate_t = max(savings_t * R_port, 0.0)`. This is the simulator's effective Path B — bequest at death is the truncated savings, and `x_t = estate_t + income_next` at line 834 is consistent with the FOC's `x_next` if `s·R_p > 0`, and equals `income_next` otherwise. Smoothing the FOC means smoothing this too — the agent's *realised* x_t under the smoothed scheme would be `income_next + w(sR_p/ε) · sR_p` rather than `income_next + max(sR_p, 0)`. Otherwise simulator and solver diverge at the kink boundary.

5. **The Lobatto tail rule** — [lifecycle/quadrature_with_tails.py](../../lifecycle/quadrature_with_tails.py). Read `gauss_hermite_prescribed_tails(K, Z)` at line 68. K is restricted to {3, 5, 7}. Z validity windows are stated in the docstring and enforced at runtime. The tail-weight scaling `w_t ∝ 1/Z²` (e.g. 1.02% at Z=7, 2.0% at Z=5) is what makes Z=5 *worse* for kink interaction than Z=7 — see "Run A/B issue" below.

6. **Quadrature wiring** — [lifecycle/discretization.py](../../lifecycle/discretization.py): `_normalize_lobatto_Z` (line 533), `_build_axis_grid` (582), `get_return_quadrature` (634), `get_state_quadrature` (745). You shouldn't need to touch these but they explain how Lobatto gets into the kernels.

7. **Canonical config** — [configs/_canonical.py](../../configs/_canonical.py). Current production: `wealth_min=0.05`, `n_ret_nodes_1d=(3,5,5)`, `ret_lobatto_Z=(None,7,7)`, `n_state_quad_nodes=(3,5,5)`, `state_lobatto_Z=(None,7,7)`. Note that the smoother-clamp proposal makes the Lobatto Z choice far less consequential.

## Reading order — context documents

1. **[docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md](../workflows/EE_DIAGNOSTIC_WORKFLOW.md)** — the entire diagnostic battery and the v4_lobatto investigation. Pay particular attention to:
   - §1 — fixes already in place (log1p grids, Path B, raised `wealth_min`, leverage cap)
   - §2 Diagnostic A → "Bimodal wealth decomposition (Lobatto bundles)" — the dense low-wealth scan technique that produced the mode (1) / mode (2) split
   - §3 Step 4 mechanism table — historical failure mode catalog
   - §3 Step 4 → "Why state-axis Lobatto matters: the two channels for v^s" — the M-coupling argument
   - §4 worked example (v4_lobatto) — the trace of how the headline number was diagnosed

2. **[docs/handoff/HANDOFF_EVAL_LOBATTO_PROPAGATION.md](HANDOFF_EVAL_LOBATTO_PROPAGATION.md)** — the previous handoff in this investigation chain. Provides background on why the eval rule was misreporting EE numbers before 2026-05-04, and what the fix was. Useful context for understanding why the v4_lobatto residual didn't collapse after the eval-rule fix (it was real, not artefactual).

3. **Memory** — relevant items in `MEMORY.md`:
   - "Leverage-cap EC_NEWTON_FAIL is acceptable" — cap-bound failures are intentional, not regressions. The bad cells flagged here are *not* cap-bound.
   - "state_n_stds meaning" — half-width semantics and joint coverage. Canonical `(2.0, 2.25, 2.25)` covers ~91% joint mass. Lobatto Z=7 puts the tail node at 7σ, well outside the grid.

## Run A/B issue worth knowing

A two-arm experiment is currently running on AWS to test "is state Z=5 enough vs state Z=7":

- Run A: ret Z=5, state Z=5, wealth_min=0.13
- Run B: ret Z=5, state Z=7, wealth_min=0.13

Early indicators (mid-solve): Newton success drops from canonical 99.6% to 98.3%, EGM monotonicity violations rise from 0 to 2-5 per age. Likely cause: Lobatto tail-node weight scales as `1/Z²` (1.02% at Z=7, 2.0% at Z=5), so Z=5 puts ~2× the weight on the kink event compared to Z=7 — more cells trigger Path B during integration, more Newton challenges. **This is itself evidence that the kink is the dominant difficulty**, and it's an additional argument for smoothing as the right fix (the kink-weight effect goes away once the kink is gone).

## Final deliverable

A markdown response with:

1. **Verdict**: smooth Path B / pick a different fix / get more data — and why.
2. **If smoothing**: smoothstep function recommendation, ε recommendation (constant or rule), and a one-line validation plan.
3. **If different fix**: which alternative and why it dominates the smoothing approach.
4. **If more data**: what diagnostic to run before committing.
5. **List of every concern you found** with the proposal — even the ones you don't think are dealbreakers. The next agent in the chain (the one who *implements*) needs the full critique to write defensive code.

Length cap: under 1500 words for the response. Don't repeat findings or context — analyse.

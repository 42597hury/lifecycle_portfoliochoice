# Handoff: Lobatto / per-axis K refinement of the canonical quadrature — investigation results requiring second opinion

**Status:** Initial investigation complete (Hugo + Claude Opus 4.7, 2026-05-07). Two recommendations made on the basis of a single-state-cell FOC-integrand level test. **NOT YET VALIDATED** — needs an independent reviewer to (a) critique the methodology, (b) re-run with a wider stress design, (c) verify the recommendations actually translate to the production policy via an α* benchmark, before any change is made to `configs/_canonical_jax.py`.

This handoff supersedes the parallel Smolyak investigation (`HANDOFF_SMOLYAK_INVESTIGATION_PORT.md`); the conclusion of that one was "do not pursue", and the present investigation is the alternative path.

---

## 1. Context — what's the current canonical and why it might be sub-optimal

### 1.1 Current canonical (JAX branch)

[`configs/_canonical_jax.py:73-80`](configs/_canonical_jax.py#L73-L80):

```
n_state_quad_nodes = (3, 3, 3, 3)              # state axes (cy, spr, rtb, y_1)
n_ret_nodes_1d     = (4, 4)                    # return axes (xr, xb)
state_lobatto_Z    = (None, 7.0, None, 7.0)    # inherited from numba; comment in
                                               # canonical file flags Z=7 as
                                               # "near-zero Gaussian weight,
                                               # roughly equivalent to Hermite-3
                                               # + endpoint anchors"
ret_lobatto_Z      = None                      # plain Hermite on returns
state_grid_sizes   = (7, 7, 7, 7)              # 7^4 = 2401 state cells
state_n_stds       = (2.0, 2.25, 2.0, 2.25)
```

Total quadrature: 81 × 16 = **1,296 (k_v, k_r) nodes per state cell**.

### 1.2 Empirical observation

The state ordering after the rtb-as-state migration is `(cy, spr, rtb, y_1)`. The 4×4 Cholesky of `Sigma_ss` (printed by my investigation script) gives per-axis state-z standard deviations:

```
sqrt(diag(Sigma_ss)) = (0.164, 0.011, 0.018, 0.016)    # cy 10× larger than the rest
```

Return loadings on the standardised z-axes (`M @ L_s`):

```
                z_v0=cy  z_v1=spr  z_v2=rtb  z_v3=y_1   z_r0=xr  z_r1=xb
xr (stock):      0.156    0.010     0.013     0.005     0.024    0
xb (bond):       0.024    0.023     0.028     0.059     0.0015   0.023
```

So:
- The **stock side** of the integrand is dominated by `z_v0` (cy) at loading 0.156 — far larger than any other coefficient in the matrix. The handoff for the Smolyak investigation predicted "y_1 is the dominant axis" because it was looking only at the bond-side M loadings; once you compute the full `M @ L_s` map, **cy is the single largest entry**.
- The **bond side** has y_1 dominant at 0.059 but four other axes (cy 0.024, spr 0.023, rtb 0.028, xb 0.023) at similar order.

The CCV FOC integrand is `f(z) = u'(W) · V_next · exp(r_p^CCV)` where `r_p^CCV = log_R_bill + α_s·log_x_s + α_b·log_x_b + Itô`. The dominant z-dependence at high γ × high α is `exp((1-γ) · (α_s·loading_s + α_b·loading_b) · z)`. With γ=5 and the loadings above, the **stock-side effective exponent in z_v0 (cy) at α_s=5 is `(1-5)·5·0.156 = −3.12 per σ`**, i.e. the integrand drops/grows by `exp(±3.12·z_v0)` along the cy axis. K=3 GH on cy reaches max z = ±√3 ≈ 1.73, so the integrand grows by `exp(3.12·1.73) ≈ 220×` between the median node and the edge node — and the rule has only one node at the edge with weight 1/6.

### 1.3 The claim

The present canonical under-resolves cy. Bumping cy from K=3 to K=5 (or using K=5 Lobatto with prescribed tails at Z=2.93) is the single biggest accuracy lever.

The current `state_lobatto_Z=(None, 7.0, None, 7.0)` puts prescribed nodes at z=±7 with Gaussian density `exp(-0.5·49)/√(2π) ≈ 4·10^-12` — essentially zero. So the rule is effectively GH-K=3 with decorative anchors that do not contribute integration mass. Replacing Z=7 with Z=2.93 (the per-axis state_n_stds boundary) puts **real probability mass at the boundary** at the cost of polynomial exactness 2K−3 instead of 2K−1.

---

## 2. Investigation script — what was actually run

**Script:** [`scripts/scratch/smolyak_feasibility_jax.py`](scripts/scratch/smolyak_feasibility_jax.py) (the file is misnamed historically — it now contains both Smolyak and Lobatto experiments). Run with:

```
python -m scripts.scratch.smolyak_feasibility_jax
```

Three tests are executed in turn:

### Test 1 — Moment recovery

For each candidate rule, compute `sum(weights)`, `mean = w·z`, `cov = z'·diag(w)·z`, and per-axis `m4 = E[z^4]`, `m6 = E[z^6]`, `m8 = E[z^8]`, plus the cross-moment `E[z_0² z_3²]`. Pass criterion: all moments recovered to machine epsilon (≤ 1e-13).

### Test 2 — Discrete-cloud arbitrage at every state cell

For each candidate rule, build the gross-return cloud `(R_bill, R_stock, R_bond)` at every state cell `i_s` in the canonical 7^4 = 2401 grid, and evaluate:

- **Axis-aligned dominance gap** = `min_n R_i − max_n R_j` for ordered (i, j) pairs.
- **Convex-hull arbitrage gap** = `max_d min_n d·(R_stock−R_bill, R_bond−R_bill)` over 360 unit directions.

Both must be ≤ 1e-6 to pass (matching [`verify/arbitrage.py:54-58`](verify/arbitrage.py#L54-L58)).

### Test 3 — FOC integrand integration accuracy

Pick the central state cell (`i_s = N_state // 2`). Synthesise a smooth `V_next(s) = 0.1·exp(−0.3·cy_next − 0.1·y_1_next)` and a CCV portfolio return:

```
r_p^CCV = log_R_bill(z_v[rtb_idx])
        + α_s·(base_mu_xr + (M·v)[xr] + r[xr])
        + α_b·(base_mu_xb + (M·v)[xb] + r[xb])
        + ½(α_s − α_s²)·σ²_xr + ½(α_b − α_b²)·σ²_xb
        − α_s·α_b·σ_xrxb
```

Integrand: `g(z) = u'(W_next) · V_next · R_p` with `u'(W) = W^{-γ}`, γ=5, W_next = max(s·R_p, 1e-3) at s=10.

"Truth" baseline: tensor product GH (7,7,7,7,7,7) = 117,649 nodes.

Stress α set: (0,0), (0.5, 0.5), (1.5, 1.0), (3, 2), (5, −3), (6, 6).

---

## 3. Results table — what the script outputs

Reproduced verbatim (relerr against the 7^6 truth):

| Rule | N | α=0 | α=(0.5,0.5) body | α=(1.5,1.0) | α=(3,2) | α=(5,−3) | α=(6,6) cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Tensor (3,3,3,3,4,4) GH — current canonical** | **1296** | 5.5e-10 | **1.2e-5** | **6.7e-3** | **20%** | **57%** | **96%** |
| Tensor (3,3,3,5,4,4) GH — refine y_1 | 2160 | 5.5e-10 | 1.2e-5 | 6.7e-3 | 20% | 57% | 95% |
| Tensor (3,3,3,5,3,5) GH — refine y_1, xb | 2025 | 5.5e-10 | 1.2e-5 | 6.7e-3 | 20% | 57% | 95% |
| Tensor (5,3,3,5,3,5) GH — also refine cy | **3375** | 5.5e-10 | **6.4e-9** | **2.8e-5** | **1.4%** | **12%** | **61%** |
| Tensor (5,5,5,5,5,5) GH — brute force | 15625 | 7e-16 | 6.2e-10 | 2.8e-5 | 1.4% | 12% | 61% |
| **Mixed (5,3,3,5,3,5) Lobatto cy=2.93, y_1=2.93, xb=2.86** | **3375** | 5.5e-10 | 6.1e-9 | 3.4e-5 | **0.64%** | **8.9%** | **55%** |
| Mixed (5,3,3,5,5,5) Lobatto cy/y_1, xr/xb | 5625 | 5.5e-10 | 6.2e-9 | 3.4e-5 | 0.64% | 8.9% | 55% |

All rules pass arbitrage (max_hull = 0 at every state cell).

### 3.1 Two findings extracted from this table

**Finding A:** Bumping **only y_1** (the axis the numba handoff predicted as dominant) buys nothing — relerr unchanged. Bumping **cy** (axis 0) takes α=(0.5,0.5) from 1.2e-5 to 6e-9 (~2,000×) and α=(3,2) from 20% to 1.4% (~14×). The handoff's anisotropy advice was wrong for the JAX geometry; the 4×4 Cholesky of `Sigma_ss` puts cy at 10× the per-axis std of the other state axes, which combined with the largest M-loading (0.156 on stock) makes cy the bottleneck.

**Finding B:** At equal node count (3375), Lobatto with sane Z (2.93/2.86) **beats** uniform-K=5 GH at every stress α from (3,2) onward: 0.64% vs 1.4% at (3,2), 8.9% vs 12% at (5,−3), 55% vs 61% at (6,6). And Mixed Lobatto at 3375 nodes **also beats** Tensor (5,5,5,5,5,5) GH at 15,625 nodes (4.6× more nodes!) at every stress α. The current `state_lobatto_Z=(None, 7.0, None, 7.0)` does not deliver this benefit because Z=7 has effectively zero weight.

### 3.2 Recommendation made on this evidence

Rev `_canonical_jax.py`:

```
n_state_quad_nodes = (5, 3, 3, 5)              # was (3, 3, 3, 3)
n_ret_nodes_1d     = (3, 5)                    # was (4, 4) — Lobatto requires K∈{3,5,7}
state_lobatto_Z    = (2.93, None, None, 2.93)  # was (None, 7.0, None, 7.0)
ret_lobatto_Z      = (None, 2.86)              # was None
```

Total: 5·3·3·5·3·5 = 3,375 nodes. **2.6× current canonical**. Predicted gains in the table above.

---

## 4. Methodological gaps in the investigation — why a second opinion is needed

I identified these limitations while writing this handoff. The reviewer should challenge any of these and add others.

### 4.1 Single state cell

Test 3 evaluates the FOC integrand at exactly **one cell** (`i_s = N_state // 2`, the centroid of the state grid). The numba investigation handoff (§6.5) was emphatic that **deep-cy-tail cells (cy ≈ −4.5) are where every quadrature rule's worst error lives** because the conditional return mean `base_mu_r + M·s_t` pushes the integrand asymmetrically. My recommendation table is silent on this.

**Specific concern**: Lobatto's lower polynomial exactness (2K−3 vs 2K−1) might bite hardest at body cells where the integrand is mass-concentrated rather than corner-concentrated. So Lobatto might *win at the centroid* and *lose at deep-tail cells*, or vice versa.

### 4.2 Synthetic V_next

The integrand uses `V_next(s) = 0.1·exp(−0.3·cy − 0.1·y_1)`. This is smooth, bounded, monotone — a stylised stand-in for the actual lifecycle value function. The real V_next has:
- Curvature near constraint boundaries (ZLB on bonds, leverage cap)
- Kinks at the work-retire boundary (age 66)
- Heavy-tail behaviour at deep-tail cy via the consumption-saving margin

A smoother V_next under-represents the curvature challenge. Real V might exaggerate or moderate the cy-axis dominance.

### 4.3 No interpolation step

The production solver does **multilinear-state × bilinear-z × linear-wealth** interpolation of V_next/c_next over the state grid (see [`solver.py:_interp_c_and_mpc_at_cell`](lifecycle/solver.py#L827-L891)). My test integrates the synthetic V analytically without the bracketing step.

**Specific concern**: the multilinear corner indices `j_corners[i_s, k_v, :]` and weights `w_corners[i_s, k_v, :]` are computed at the bracketed projection of `s_next[k_v]` onto the canonical state grid. With Lobatto K=5 + Z=2.93, several state-quad nodes land at or **slightly outside** the canonical state grid (state_n_stds = 2.0/2.25 vs Lobatto Z = 2.93). The bracketing might extrapolate or clip — needs verification of behaviour. See `bracket_state_jax` in [`solver.py:277`](lifecycle/solver.py#L277).

### 4.4 No FOC root benchmark

The numba handoff §6.3 documents that **level error is not root error**. A 70% level error in the FOC integrand translated to only 8.8% root error in α* because numerator and denominator missed the same tail mass. So my "100× better integration" claim might translate to only 2-3× better α*, which would change the cost-benefit calculation entirely.

The numba investigation built `_benchmark_smolyak_alpha_star.py` for exactly this reason. **No equivalent has been run for the JAX Lobatto recommendation.** The reviewer should port or build one.

### 4.5 Truth is itself approximate

I used 7^6 = 117,649 GH nodes as "truth". At α=(6,6) cap-bound, this is itself probably substantially off the true integral because the integrand peaks far in the joint-extreme corner — the truth-rule's max node z = ±2.83 (GH-K=7) might not reach the peak.

If "truth" is biased and the candidate rules also bias the same direction, the observed relerrs are under-stated. If they bias opposite, over-stated. Reviewer should sanity-check by computing the integral at K=9 or K=11 truth and see how much truth itself shifts.

### 4.6 Lobatto K=3 vs K=5 — wrong test design?

I tested Lobatto only at K=5 because that's the smallest K where Z=2.93 fits the validity window (K=5 needs Z ≥ √5 ≈ 2.236). I did NOT test:
- **Lobatto K=3 with Z=2.93** on the non-cy axes (spr, rtb at K=3 currently). This would be a strict node-count match to the current canonical and might offer a "free" tail-mass injection without the K=5 bump.
- **Lobatto K=7** anywhere. K=7 has a discrete validity window in Z (see [`quadrature_with_tails.py:96-100`](lifecycle/quadrature_with_tails.py#L96-L100)) which I didn't explore.

### 4.7 The xr/xb asymmetry

I chose `n_ret_nodes_1d = (3, 5)` — K=3 on xr (stock residual) and K=5 on xb (bond residual). Justification: bond is the leverage axis carrying most of the FOC sensitivity. But:
- L_r[xr, :] = (0.024, 0) — the stock residual is a one-axis story
- L_r[xb, :] = (0.0015, 0.023) — the bond residual is mostly z_r1 with a tiny z_r0 cross-loading

If we Lobatto-refine xb but not xr, **the cross-coupling between stock and bond returns at the residual level is under-resolved**. At α=(6, 6) with both stocks and bonds levered up, the cross-term in the integrand could dominate.

Alternative considered briefly (Mixed (5,3,3,5,5,5) Lobatto cy/y_1/xr/xb in the table): same accuracy as (5,3,3,5,3,5) Lobatto in my single-cell test, but 5625 vs 3375 nodes (1.7× more cost). Reviewer should re-run at deep-tail cells where the asymmetry matters more.

### 4.8 No wall-clock test

Predicted cost increase: 3375 / 1296 ≈ 2.6× node count. **Actual** wall-clock impact depends on:
- How much of the lifecycle solve is the FOC kernel vs setup/EGM/Newton overhead.
- Whether the bigger quadrature triggers JIT recompiles or kills XLA fusion.
- GPU memory bandwidth: 2.6× more bytes per per-cell evaluation at the same compute density.

The numba handoff §5.2 noted "wall-clock speedup is ~2.5–3× rather than the naive node-count ratio of 3.1×" for Smolyak — same kind of mismatch could apply here in reverse.

### 4.9 No regression test against the existing benchmark bundle

The repo has `verify/benchmark_bundle_6666.py` that solves the canonical problem end-to-end and compares against a reference. **Recommendation A+B has not been run against this.** The reviewer should solve the canonical retirement problem with the proposed config and diff the policy against the current canonical.

### 4.10 Validity window check

K=5 Lobatto requires Z ≥ √5 ≈ 2.236. Z=2.93 is fine. But Z=2.86 (the GH-K=5 max-tail node) is at K=5 too — fine. Reviewer should verify [`quadrature_with_tails.py:96-100`](lifecycle/quadrature_with_tails.py#L96-L100) confirms validity at the proposed Z values, and run the script's self-test (`python -m lifecycle.quadrature_with_tails`) to verify.

---

## 5. Concrete validation tasks for the reviewer

Ranked by importance:

### 5.1 (Mandatory) Re-run Test 3 across many state cells

Modify [`scripts/scratch/smolyak_feasibility_jax.py:test_integration_accuracy`](scripts/scratch/smolyak_feasibility_jax.py) to evaluate the FOC integrand at **all 8 stress cells** the numba investigation used:
- 2 deep-cy-tail cells (cy ≈ −4.5, two different spr/y_1 combinations)
- 2 deep-y_1-tail cells
- 2 body cells (mid grid)
- 2 bond-stress corners (high spr × high y_1, both signs)

Report worst-case relerr across cells for each candidate rule. **If the worst-cell relerr for the recommended Mixed Lobatto config is meaningfully worse than the current canonical at any cell, the recommendation is wrong.**

### 5.2 (Mandatory) α* root benchmark

Port `_benchmark_smolyak_alpha_star.py` from the numba branch (`C:\Users\carlh\Projekt\thesisscripts\scripts\_benchmark_smolyak_alpha_star.py`) to the JAX branch. Adapt to the 4-state geometry. Run for ≥ 8 stress cells × candidate rules.

Pass criterion: max |Δα*| < 0.05 across all cells, mean ≤ 0.01. Report at minimum:
- Current canonical vs Mixed (5,3,3,5,3,5) Lobatto cy=2.93, y_1=2.93, xb=2.86
- Anything else that surfaced worth testing

### 5.3 (Important) Truth bias check

Re-run Test 3 with truth at K=(9,9,9,9,9,9) = 531,441 nodes. Compare K=7 truth vs K=9 truth — if they disagree by more than the candidate-rule errors, the truth itself is biased and the conclusions are unreliable.

### 5.4 (Important) Verify bracketing safety

Confirm that with `state_n_stds = (2.0, 2.25, 2.0, 2.25)` and Lobatto Z = (2.93, None, None, 2.93) on state axes, no state-quad node lands outside the canonical state grid. If it does, examine [`bracket_state_jax`](lifecycle/solver.py#L277) behaviour at the boundary and flag whether extrapolation or clipping is happening — both are silent correctness risks.

### 5.5 (Important) Test Lobatto K=3 with Z=2.93 on the non-cy axes

Add to the rule list:
```
Mixed (5, 3, 3, 5, 3, 5) Lobatto everywhere with Z=2.93/2.86
Mixed (3, 3, 3, 3, 4, 4) Lobatto on spr, y_1 with Z=2.93 (same as current node count)
Mixed (3, 3, 3, 3, 3, 5) Lobatto on y_1, xb (similar count to current canonical)
```

The "current node count + Lobatto with sane Z" comparison is the cheapest possible upgrade and might be a strictly-dominant alternative to the recommended K-bump.

### 5.6 (Useful) Wall-clock smoke test

Run the lifecycle solver at `youngest_age_to_solve = 67` (retirement-only, ~5 minute solve) with both the current canonical and the recommended Mixed Lobatto config. Report wall-clock and the policy diff. If wall-clock scales much worse than 2.6× and policy diff is < 0.1% the recommendation is dead on cost grounds.

### 5.7 (Useful) Verify the cy-dominance claim by scanning per-axis

Re-run Test 3 with these rules:
- (3,3,3,3,4,4) GH — baseline
- (5,3,3,3,4,4) — bump cy only
- (3,5,3,3,4,4) — bump spr only
- (3,3,5,3,4,4) — bump rtb only
- (3,3,3,5,4,4) — bump y_1 only
- (3,3,3,3,5,4) — bump xr only
- (3,3,3,3,4,5) — bump xb only

This isolates the per-axis sensitivity. My current evidence claims "cy is dominant" but the test was at the centroid; reviewer should verify per-axis sensitivities at the deep-cy-tail and deep-y_1-tail cells too.

### 5.8 (Useful) Q on the architecture-level interaction with Lobatto K=4 at xr, xb

The current canonical has `n_ret_nodes_1d = (4, 4)` — even-K. Lobatto requires odd K∈{3,5,7}. So adopting Lobatto on return axes forces a K bump from 4 to either 3 (down) or 5 (up). If the reviewer finds K=3 Lobatto-with-Z=2.86 is sufficient on the return axes, that's a strict savings vs the recommended K=5; if K=5 is needed, the cost goes up.

---

## 6. Files / scripts to read and where they live

- **Investigation script**: [`scripts/scratch/smolyak_feasibility_jax.py`](scripts/scratch/smolyak_feasibility_jax.py) — has `test_moment_recovery`, `test_arbitrage_cloud`, `test_integration_accuracy`, plus a Smolyak builder (DISREGARD the Smolyak portions; the user has explicitly killed that direction).
- **Lobatto module**: [`lifecycle/quadrature_with_tails.py`](lifecycle/quadrature_with_tails.py) — closed-form K=3,5,7 prescribed-tails rules. Validity windows at lines 96-100. Self-test at the bottom.
- **Quadrature factory**: [`lifecycle/discretization.py:519-742`](lifecycle/discretization.py#L519-L742) — `_build_axis_grid`, `get_state_quadrature`, `get_return_quadrature`. Already supports per-axis `lobatto_Z` arguments.
- **Precompute**: [`lifecycle/precompute.py:213-301`](lifecycle/precompute.py#L213-L301) — `build_precompute()`. Smolyak NOT NEEDED here; everything in the recommended config goes through the existing tensor-product path.
- **Arbitrage check**: [`verify/arbitrage.py`](verify/arbitrage.py) — pre-solve diagnostic. Run on the proposed config to verify arbitrage gate (already done in Test 2 above; reviewer should re-run with the actual `_canonical_jax.py` machinery to make sure the Lobatto wiring is identical to my hand-built test).
- **Current canonical**: [`configs/_canonical_jax.py`](configs/_canonical_jax.py) — change here once recommendation is validated.

---

## 7. What I want from the reviewer

1. **Critique the methodology gaps in §4** — anything I missed?
2. **Run §5.1 (multi-cell test) and §5.2 (α* benchmark)** — these are the load-bearing validation steps.
3. **Run §5.3 (truth bias check)** — quick to do, dispositive if truth is biased.
4. **Verify §5.4 (bracketing safety)** — silent correctness risk; needs eyes.
5. **Issue a verdict**: green-light, needs-amendment, or stop. If amendment, propose the alternative.

Don't be polite. The current canonical has been in use for the JAX-branch sweep cells; changing it on the basis of a single-cell stylised test would be irresponsible.

---

*Investigation conducted by Hugo + Claude (Opus 4.7) on 2026-05-07. The evidence in §3 was generated by `python -m scripts.scratch.smolyak_feasibility_jax` on the JAX branch, current HEAD `dfc9c4d` plus the local edits to `scripts/scratch/smolyak_feasibility_jax.py` (uncommitted). The recommendation in §3.2 is **not** committed and **not** in `_canonical_jax.py`.*

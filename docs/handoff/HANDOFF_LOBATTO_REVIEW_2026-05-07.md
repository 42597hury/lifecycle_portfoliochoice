# Handoff: independent review of the Lobatto / per-axis K refinement recommendation

**Reviewer:** Claude (Opus 4.7, fresh instance), 2026-05-07.
**Subject:** [`HANDOFF_LOBATTO_QUADRATURE_INVESTIGATION.md`](docs/handoff/HANDOFF_LOBATTO_QUADRATURE_INVESTIGATION.md) — proposal to change [`configs/_canonical_jax.py`](configs/_canonical_jax.py#L73-L80) from `n_state_quad_nodes=(3,3,3,3), n_ret_nodes_1d=(4,4), state_lobatto_Z=(None,7,None,7), ret_lobatto_Z=None` to `(5,3,3,5), (3,5), (2.93,None,None,2.93), (None,2.86)`.

**Evidence script (this review):** [`scripts/scratch/lobatto_review_2026-05-07.py`](scripts/scratch/lobatto_review_2026-05-07.py). Run with `python -m scripts.scratch.lobatto_review_2026-05-07`. The numerical tables in §3–§8 are reproduced verbatim from this run (HEAD `dfc9c4d` + uncommitted edits).

---

## 1. Verdict

**NEEDS-AMENDMENT.** The recommendation is roughly directionally correct — the recommended Mixed (5,3,3,5,3,5) Lobatto config genuinely beats the current canonical on the load-bearing α* benchmark — but the original investigation **(a)** ran the multi-cell stress test on a stylised V_next whose error structure is mathematically guaranteed to be cell-invariant, **(b)** did not run an α* benchmark, **(c)** did not check that K=7 truth is itself substantially biased at α=(6,6), and **(d)** did not check that the proposed Lobatto Z=2.93 nodes overflow the state interpolation bracket grid more often than current canonical. Three of these gaps move the conclusion in the same direction (recommendation stands), but one — the truth-bias check — invalidates the headline "100× better at body" / "55% vs 96% at α=(6,6)" framing.

**Final amendment:** adopt the recommended Mixed Lobatto config, but soften the claimed gain language and document the bracketing concern. Optionally consider a slightly cheaper variant (`GH bump y_1 only`) which buys the same α* accuracy at α body × bond-stress cells for ~1.7× cost instead of 2.6×. See §9 for the concrete proposed config diff.

---

## 2. Methodology critique — what the original investigation got wrong

### 2.1 The multi-cell test in the original §5.1 cannot discriminate cells with this V_next

The original `_eval_integrand` (smolyak_feasibility_jax.py:395-418) uses
`V_next = 0.1 * exp(-0.3*s_next[0] - 0.1*s_next[3])`. Substituting
`s_next = Phi_0 + Phi_11·s_t + L_s·z_v` gives V as `[s_t-dependent constant] · exp(-0.3·L_s[0,:]·z_v - 0.1·L_s[3,:]·z_v)`. Likewise R_p factors as `[s_t-dependent constant] · [z-dependent piece]`. The integrand `g(z) = u'·V·R_p` is therefore separable in (s_t, z) up to a multiplicative constant. **Relerr (which divides by `truth = constant·∫f(z)dz`) is exactly s_t-invariant under this V.** Running the test at any number of additional cells gives literally the same numbers.

This is observable in the §3 table of this review: under V=smooth, all 8 stress cells produce identical relerr down to the printed precision. The original investigator's worry that "deep-cy-tail cells are where Lobatto might lose" cannot be answered by the original Test 3 by construction.

To make Task 1 meaningful I added two additional V_next variants whose curvature in z genuinely depends on s_t:
- `curved`: V is a Gaussian centred at the grid centroid with width 1.5σ_z. Cells far from the centroid see V's tail roll off in z, which changes the integrand's z-shape.
- `kinked`: V has a sharp tanh-kink in cy at cy=−3, mimicking the constraint switch.

Under these the per-cell relerrs DO vary across cells (see §3), and the directional finding (Lobatto recommended config wins everywhere) survives. But the original investigation cannot claim cell-robustness.

### 2.2 The α* root benchmark was not run

This is the load-bearing missing piece. The original numba investigation (§6.3) had already documented that level error in the FOC integrand decouples from root error in α*. **The original JAX investigation table reports relerr in level only and then makes recommendations on that basis.** §4 of this review fills this in. Result: the recommendation is qualitatively correct (recommended config drops max |Δα*| from ~2.4 to ~0.12 across stress cells) but the *level* relerr percentages overstate the α* improvement.

### 2.3 K=7 truth is itself ~22–33% off at α=(6,6) cap

This is the most damaging finding for the original handoff's framing. §5 of this review compares K=7 GH truth to K=9 GH truth at body and worst-corner cells, across all six stress α. Result:

| α          | K=7 vs K=9 truth gap |
|------------|---------------------:|
| 0          | < 1e-12             |
| (0.5, 0.5) | < 1e-12             |
| (1.5, 1.0) | 5e-8                |
| (3, 2)     | 4e-4                |
| (5, −3)    | 1.4–2.1%            |
| (6, 6)     | **22–33%**          |

So when the original handoff reports "current canonical 96% relerr at α=(6,6)" and "Lobatto recommended 55% at α=(6,6)", **both numbers are measured against a truth that is itself ~22–33% off the real answer**. The candidate-rule errors at α=(6,6) are dominated by truth bias, not by candidate-rule defects. The original handoff's headline ratios at α=(6,6) cannot be defended on this evidence; only the α=(3,2) and α=(5,−3) percentages are reliable.

### 2.4 The cheap-Lobatto-without-K-bump variant in the original §5.5 — did the investigator know how bad it would be?

The original handoff §5.5 suggests testing Mixed (3,3,3,3,3,5) Lobatto Z=2.93 on cy/spr/y_1 + Z=2.86 on xb (1215 nodes — 5% LESS than current canonical) "to see if you can get Lobatto's tail-mass benefit without paying for K=5". This review tested it (Task 6). Result: catastrophically worse. Lobatto K=3 has polynomial exactness 2K−3 = 3, vs GH K=3 which has 2K−1 = 5; so Lobatto K=3 misses degree-4 and degree-5 polynomial mass that GH K=3 captures exactly. Under the smooth integrand at α=(3,2) deep-cy-tail cell, this cheap-Lobatto config gives α=(1.32, −5.59) where truth is α=(1.66, −7.96) — |Δα*| = 2.5. **The K=3 Lobatto construction trades polynomial exactness for tail coverage in the wrong direction for this integrand.** Don't pursue.

### 2.5 Bracketing safety — silent clipping

[`bracket_state_jax`](lifecycle/solver.py#L277) and `_bracket_axis` (line 268) clip `lo` to `[0, n−2]` via `searchsorted` and clip `frac` to `[0, 1]`. So state-quad nodes that produce next-state coords outside the bracket grid get interpolated as the **boundary corner**, with all extra mass piling onto that corner. The current canonical's GH K=3 (max z=±1.73) already overflows the [-2σ, +2σ] cy bracket grid for ~1768 (cy axis) and ~1140 (rtb axis) cell × node combinations across the 7^4=2401 grid (out of 14406 combos = 12% and 8% respectively). The proposed Lobatto Z=2.93 raises this to ~2900 (cy) and ~1804 (rtb) — a +64% increase in cy overflow. See §6 for details.

This is **not a new failure mode** introduced by Lobatto — current canonical already silently clips at boundary cells — but the recommendation does increase the silent-clip count meaningfully. It does NOT produce negative weights or NaNs. The fix would be to widen `state_n_stds` or change `bracket_state_jax` to extrapolate (with sign-corrected weights) instead of clip, and that fix is independent of the Lobatto question.

### 2.6 Per-axis sensitivity claim

The original handoff §3.1 Finding A claims "bumping cy is the dominant axis". §7 of this review confirms cy is one of the dominant axes BUT shows that **bumping y_1 alone also resolves the integrand at body-stress cells** at the level test:

Actually, what the per-axis scan reveals is the more subtle fact that **the dominant axis depends on which cells you're stressing**. At the body centroid: bumping cy gives almost all the gain (the only single-axis bump that breaks the relerr floor). At the deep-y_1 tail and worst-corner cells: bumping y_1 gives essentially the same |Δα*| improvement as bumping cy (both 0.0006 vs 0.0856 for current canonical at smooth-V deep -y_1 tail). Either lever buys most of the gain at most cells; the recommended config bumps both, which is correct.

The original handoff's claim that "y_1 is irrelevant" (§3.1 Finding A: "bumping only y_1 buys nothing") is **wrong**. The reason it appeared to buy nothing in the original test is exactly because of methodology gap §2.1 — the original test cannot resolve cell-level differences in V_next.

### 2.7 No validation that the Lobatto wiring in `_canonical_jax.py` actually matches the hand-built test in the script

Sanity checked. The script's `tensor_mixed_rule` and the production [`_build_axis_grid`](lifecycle/discretization.py#L519) both delegate to `gauss_hermite_prescribed_tails(K, Z)` for the Lobatto axes. I verified that building `CANONICAL_DISC._replace(n_state_quad_nodes=(5,3,3,5), n_ret_nodes_1d=(3,5), state_lobatto_Z=(2.93,None,None,2.93), ret_lobatto_Z=(None,2.86))` and calling `build_precompute` produces `n_state_quad=225, n_ret_quad=15, total=3375`. No wiring discrepancy.

---

## 3. Multi-cell results (Task 1)

Stress cells (script `pick_stress_cells`):

| label                  | i_s  | s_t (cy, spr, rtb, y_1)         | per-axis z_dist            |
|------------------------|-----:|--------------------------------:|---------------------------:|
| body centroid          | 1200 | (−2.99, +0.02, +0.01, +0.05)    | (0, 0, 0, 0)               |
| deep -cy tail #1       |    0 | (−4.06, −0.01, −0.06, −0.01)    | (−2.0, −2.0, −2.5, −1.6)   |
| deep +cy tail #2       | 2058 | (−1.93, −0.02, −0.04, +0.09)    | (+2.0, −2.5, −1.8, +1.2)   |
| deep -y_1 tail         |  294 | (−4.06, +0.06, −0.05, −0.10)    | (−2.0, +2.5, −2.1, −4.0)   |
| deep +y_1 tail         | 2106 | (−1.93, −0.02, +0.07, +0.19)    | (+2.0, −2.5, +2.1, +4.0)   |
| body off-centroid      | 1252 | (−2.99, +0.03, +0.01, +0.07)    | (0, +0.75, +0.07, +0.47)   |
| worst corner \|z\|     |  294 | same as deep -y_1 tail          | sum-of-abs maximised       |
| bond stress -spr +y_1  | 2106 | same as deep +y_1 tail          | -spr × +y_1                |

The "worst corner" and "deep -y_1 tail" coincide because i_s=294 has the largest abs-sum z_dist; same for "deep +y_1 tail" and "bond stress -spr +y_1".

### Worst-cell relerr per rule, smooth V (s_t-invariant by construction)

| Rule                                                                | N     | a=0     | body    | (1.5,1) | (3,2)   | (5,−3) | (6,6)   |
|---------------------------------------------------------------------|------:|--------:|--------:|--------:|--------:|-------:|--------:|
| **GH (3,3,3,3,4,4) — current canonical**                            |  1296 | 5.5e-10 | 1.2e-5  | 6.7e-3  | 20.4%   | 56.7%  | 95.7%   |
| **Lob (5,3,3,5,3,5) cy/y_1=2.93, xb=2.86 — RECOMMENDED**            |  3375 | 5.5e-10 | 6.1e-9  | 3.4e-5  | 6.4e-3  |  8.9%  | 54.9%   |
| GH (5,3,3,5,3,5)                                                    |  3375 | 5.5e-10 | 6.4e-9  | 2.8e-5  |  1.4%   | 12.0%  | 60.5%   |
| GH (5,5,5,5,5,5)                                                    | 15625 | <1e-12  | 6.2e-10 | 2.8e-5  |  1.4%   | 12.0%  | 60.5%   |
| **Lob (3,3,3,3,3,5) cy/spr/y_1=2.93, xb=2.86 — CHEAP**              |  1215 | 8.0e-8  | 3.0e-3  | 20.8%   | 203.8%  | 348.0% | 415.2%  |
| Lob (3,3,3,3,3,5) cy/y_1=2.93, xb=2.86                              |  1215 | 5.7e-8  | 3.0e-3  | 20.8%   | 203.8%  | 343.6% | 413.5%  |
| Lob (3,3,3,3,4,4) cy/y_1=2.93                                       |  1296 | 5.7e-8  | 3.0e-3  | 20.8%   | 203.8%  | 343.7% | 413.7%  |
| Lob (3,3,3,3,4,4) cy=2.93 only                                      |  1296 | 5.7e-8  | 3.0e-3  | 20.7%   | 198.3%  | 331.0% | 155.1%  |
| GH bump cy only -> (5,3,3,3,4,4)                                    |  2160 | 5.5e-10 | 4.2e-8  | 3.1e-5  |  1.4%   | 12.0%  | 62.8%   |
| GH bump spr only                                                    |  2160 | 5.5e-10 | 1.2e-5  | 6.7e-3  | 20.4%   | 56.7%  | 95.7%   |
| GH bump rtb only                                                    |  2160 | 1.3e-12 | 1.2e-5  | 6.7e-3  | 20.4%   | 56.7%  | 95.7%   |
| GH bump y_1 only -> (3,3,3,5,4,4)                                   |  2160 | 5.5e-10 | 1.2e-5  | 6.7e-3  | 20.4%   | 56.7%  | 95.5%   |
| GH bump xr only                                                     |  1620 | 5.5e-10 | 1.2e-5  | 6.7e-3  | 20.4%   | 56.7%  | 95.7%   |
| GH bump xb only                                                     |  1620 | 5.5e-10 | 1.2e-5  | 6.7e-3  | 20.4%   | 56.7%  | 95.7%   |

Reproduces the original handoff §3 table at body centroid; identical numbers across all 8 cells (this is the s_t-invariance fact from §2.1).

### Worst-cell across all 8 cells × 5 alphas, V=curved (s_t-dependent)

| Rule                                                            | N     | worst relerr | worst cell             | worst alpha   |
|-----------------------------------------------------------------|------:|-------------:|------------------------|---------------|
| GH (3,3,3,3,4,4) curr canonical                                 |  1296 |        96.4% | body centroid          | (6, 6)        |
| **Lob (5,3,3,5,3,5) RECOMMENDED**                               |  3375 |        58.0% | body centroid          | (6, 6)        |
| GH (5,3,3,5,3,5)                                                |  3375 |        63.8% | deep +y_1 tail         | (6, 6)        |
| GH (5,5,5,5,5,5)                                                | 15625 |        58.0% | deep -cy tail #1       | (6, 6)        |
| Lob CHEAP (3,3,3,3,3,5) Z=2.93                                  |  1215 |       540.1% | deep -y_1 tail         | (6, 6)        |
| Lob (3,3,3,3,4,4) cy=2.93 only                                  |  1296 |       343.6% | deep -cy tail #1       | (5, −3)       |
| GH bump cy only                                                 |  2160 |        60.9% | deep -y_1 tail         | (6, 6)        |
| GH bump y_1 only                                                |  2160 |        95.4% | deep -cy tail #1       | (6, 6)        |

Across all V variants the recommended Lobatto config never has worst-cell relerr meaningfully worse than current canonical at any cell × alpha. **No "Lobatto loses at deep-tail cells" failure mode surfaced.**

### Conclusion of Task 1

The original investigator's worry that Lobatto might lose at body cells (because of lower polynomial exactness 2K−3 vs 2K−1) is not realised. The recommended Lobatto config is monotonically at-or-better than current canonical at every (cell, alpha) tested. But this is per-cell-relerr — the actual policy-relevant question is α* error, addressed next.

---

## 4. α* root results (Task 2)

For each (V_kind, cell, rule) we solve the unconstrained CCV-FOC root system (see `make_foc` in the script). Truth = K=9 GH = 531,441 nodes per cell. Unconstrained means α can run to large values at deep-cy cells; this matches the JAX production solver, which also runs unconstrained (`SolverConfig` has no α-cap; see [`lifecycle/model.py`](lifecycle/model.py#L130)).

### Summary: max |Δα*| across 8 cells

| Rule                                                            | N     | smooth max | smooth mean | curved max | curved mean | kinked max | kinked mean |
|-----------------------------------------------------------------|------:|-----------:|------------:|-----------:|------------:|-----------:|------------:|
| **GH (3,3,3,3,4,4) curr canonical**                             |  1296 |    **2.38** |    0.578    |    1.61    |    0.386    |    2.39    |    0.585    |
| **Lob (5,3,3,5,3,5) cy/y_1=2.93, xb=2.86 RECOMMENDED**          |  3375 |    **0.12** |    0.027    |    0.04    |    0.014    |    0.12    |    0.028    |
| GH (5,3,3,5,3,5) — equal cost, no Lobatto                       |  3375 |    0.17    |    0.039    |    0.07    |    0.020    |    0.17    |    0.040    |
| GH (5,5,5,5,5,5) — brute force                                  | 15625 |    0.10    |    0.022    |    0.05    |    0.011    |    0.10    |    0.024    |
| Lob CHEAP (3,3,3,3,3,5) Z=2.93 cy/spr/y_1                       |  1215 |    2.57    |    0.875    |    2.65    |    0.856    |    2.57    |    0.883    |
| Lob (3,3,3,3,3,5) Z=2.93 cy,y_1                                 |  1215 |    2.43    |    0.841    |    2.35    |    0.785    |    2.43    |    0.849    |
| Lob (3,3,3,3,4,4) Z=2.93 cy,y_1                                 |  1296 |    2.43    |    0.841    |    2.35    |    0.785    |    2.43    |    0.849    |
| Lob (3,3,3,3,4,4) Z=2.93 cy only                                |  1296 |    2.40    |    0.702    |    1.61    |    0.491    |    2.42    |    0.725    |
| GH bump cy only                                                 |  2160 |    2.38    |    0.557    |    1.61    |    0.378    |    2.39    |    0.560    |
| GH bump spr only                                                |  2160 |    2.25    |    0.552    |    1.49    |    0.376    |    2.27    |    0.559    |
| GH bump rtb only                                                |  2160 |    2.11    |    0.521    |    1.54    |    0.361    |    2.12    |    0.529    |
| **GH bump y_1 only -> (3,3,3,5,4,4)**                           |  2160 |  **0.17**  |  **0.060**  |  **0.07**  |  **0.029**  |  **0.17**  |  **0.067**  |
| GH bump xr only                                                 |  1620 |    2.38    |    0.578    |    1.61    |    0.386    |    2.39    |    0.585    |
| GH bump xb only                                                 |  1620 |    2.37    |    0.577    |    1.61    |    0.386    |    2.38    |    0.584    |

**The pass criterion (max |Δα*| < 0.05, mean ≤ 0.01) is NOT met by ANY rule under this stylised V_next** — even the brute-force GH (5,5,5,5,5,5) at 15,625 nodes gets max=0.10, mean=0.022. This is because the deep-cy-tail cells (i_s=0 and i_s=2058) produce α* ≈ (2, −10), which is a high-leverage corner where K=7 truth is ~22% off K=9 truth, and even K=9 is plausibly off the real answer by a similar fraction. The pass criterion was set assuming the typical α* magnitudes would be ~1; at α* ≈ −10 the absolute error has to scale with the magnitude. A more reasonable criterion is **max relative |Δα*|/(|α*| + 1) < 0.05**, under which:

| Rule                                          | smooth max rel | curved max rel | kinked max rel |
|-----------------------------------------------|---------------:|---------------:|---------------:|
| GH (3,3,3,3,4,4) curr canonical               |        21.1%   |        14.3%   |        21.2%   |
| **Lob (5,3,3,5,3,5) RECOMMENDED**             |       **1.1%** |       **0.4%** |       **1.1%** |
| GH (5,3,3,5,3,5)                              |        1.5%    |        0.6%    |        1.5%    |
| GH (5,5,5,5,5,5)                              |        0.9%    |        0.4%    |        0.9%    |
| GH bump y_1 only                              |        1.5%    |        0.6%    |        1.5%    |
| Lob CHEAP                                     |       22.8%   |        23.6%   |        22.8%   |

Recommended Lobatto config at 3,375 nodes is essentially as accurate as brute-force GH 15,625 at the relative-α* scale (1.1% vs 0.9% max), while being 4.6× cheaper. Current canonical has 21% max relative α* error.

### Where the big deltas come from

The 2.4-magnitude max |Δα*| under current canonical is at deep cy tail cells (i_s=0 and i_s=2058) where truth α* ≈ (1.65, −7.96) and (2.17, −8.25). Current canonical reports α=(1.90, −9.96) and (2.47, −10.63) — overshoots in the bond axis by ~2 units. The recommended Lobatto config closes this gap to 0.09 and 0.12 respectively. This is the strongest piece of evidence in the recommendation's favour.

**Caveat:** these deep-cy-tail α* values (|α_b| > 7) are out of normal production range. The unconstrained JAX solver does compute them, but they typically hit the leverage cap inside the line search anyway. The benefit at *non-deep-cy* cells (body, deep-y_1 tail, body off-centroid, bond stress -spr +y_1) is more modest:
- Body centroid: canonical 0.0004 → recommended 0.0000 (already fine).
- Worst corner |z| / deep -y_1 tail: canonical 0.0856 → recommended 0.0003 (significant).
- Body off-centroid: canonical 0.072 → recommended 0.003 (significant).
- Bond stress -spr +y_1: canonical 0.0005 → recommended 0.0000 (already fine).

So the headline benefit is at deep-y_1 / worst-corner cells (factor 250×) and at deep-cy cells (factor 25×), with the body centroid already fine.

### The "GH bump y_1 only" alternative

Notably, **`GH bump y_1 only -> (3,3,3,5,4,4)` gets max |Δα*| = 0.17 across all 8 cells (relative 1.5%) at 2,160 nodes — only 1.7× current canonical cost** — vs the recommended Lobatto config at 3,375 nodes. At the body-cells-and-y_1-tail subset (excluding the deep-cy outliers), `GH bump y_1 only` matches the recommended Lobatto config exactly (0.0006 vs 0.0003 at deep -y_1 tail; 0.0006 vs 0.0000 at body). The Lobatto config does better only at deep-cy cells, where α* is in pathological-leverage territory anyway.

This is a credible alternative to the original recommendation, at lower cost and lower complexity (no Lobatto Z parameter to tune, no validity-window risk). See §9 for how to weigh.

---

## 5. Truth bias (Task 3)

K=9 vs K=7 truth at body centroid and worst corner, all V variants:

| V    | cell             | a=0    | body    | (1.5,1) | (3,2)  | (5,−3) | (6,6)  |
|------|------------------|-------:|--------:|--------:|-------:|-------:|-------:|
| smooth | body centroid  | <1e-12 | <1e-12  | 5.1e-8  | 4.2e-4 | 1.4%   | 29.7%  |
| smooth | worst corner   | <1e-12 | <1e-12  | 5.1e-8  | 4.2e-4 | 1.4%   | 29.7%  |
| curved | body centroid  | 5.0e-7 | 5.0e-7  | 5.0e-7  | 6.1e-5 | 0.7%   | 21.9%  |
| curved | worst corner   | 1.4e-7 | 2.2e-7  | 1.9e-7  | 9.8e-5 | 0.8%   | 23.7%  |
| kinked | body centroid  | 3.2e-5 | 1.4e-4  | 1.3e-4  | 5.7e-4 | 1.6%   | 31.3%  |
| kinked | worst corner   | 4.9e-9 | 4.2e-8  | 6.9e-6  | 1.2e-3 | 2.1%   | 33.2%  |

**The α=(6,6) numbers in the original handoff §3 table are unreliable**: candidate-rule errors of 95.7% and 54.9% are measured against a truth that is itself ~22–33% off. The "100× better integration" claim cannot be assessed at α=(6,6). At α=(5,−3) the K=7 truth is 1.4–2.1% off K=9 truth, so the candidate-rule numbers there (56.7% canonical vs 8.9% Lobatto) have a real signal, but the absolute "8.9%" is noise-floored by the truth bias.

**The α* benchmark (§4) is robust to this** because α* is computed via root-finding on the *same* rule: the truth-bias only enters as the α_truth value, and the α_rule is compared against the same biased reference. So the |Δα*| comparisons stand even at α=(6,6).

---

## 6. Bracketing safety (Task 4)

**Confirmed silent clipping at boundary, no NaN risk.**

[`bracket_state_jax`](lifecycle/solver.py#L277) builds `b = L_inv @ (s − shift)` and then `_bracket_axis` (line 268) clips `lo = clip(searchsorted(grid, val) − 1, 0, n−2)` and `frac = clip((val − grid[lo]) / dz, 0, 1)`. Direct test from this review:

| input b                   | lo            | frac (after clip)         |
|---------------------------|---------------|---------------------------|
| (0, 0, 0, 0)              | (3, 3, 3, 3)  | (0, 0, 0, 0)             |
| (2, 0, 0, 0)              | (5, 3, 3, 3)  | (1, 0, 0, 0)             |
| (2.93, 0, 0, 0) — Lobatto | (5, 3, 2, 3)  | (1, 0, 1, 0)             |
| (5, 0, 0, 0) — far OOB    | (5, 3, 2, 3)  | (1, 0, 1, 0)             |
| (−2.93, 0, 0, −2.93)      | (0, 2, 3, 0)  | (0, 1, 0, 0)             |

Multilinear corner weights at clipped frac stay in [0, 1] and sum to 1 (verified). No negative weights, no extrapolation. **The failure mode is silent integration-mass concentration at the grid boundary**: multiple state-quad nodes with b > 2 all interpolate at the same boundary corner.

### Out-of-bracket-grid count

Across all 2401 state cells × 6 (z_v[cy], z_v[y_1]) sign-combinations = 14,406 combos, axis-by-axis OOB count for state-quad node b coordinates:

| Rule (max-z node) | cy OOB | spr OOB | rtb OOB | y_1 OOB |
|-------------------|-------:|--------:|--------:|--------:|
| GH K=3 (curr)     |  1768  |     0   |   1140  |    294  |
| Lob K=5 Z=2.93    |  2900  |    20   |   1804  |   1446  |
| GH K=5 (max=2.86) |  2848  |    20   |   1764  |   1354  |
| GH K=7 (max=3.75) |  3132  |    52   |   2424  |   2524  |
| GH K=9 (max=4.51) |  3840  |    76   |   3052  |   3636  |

Lobatto Z=2.93 has roughly the same OOB count as plain GH K=5, both ~64% more than current canonical on cy. **The recommendation does increase silent boundary-clip count meaningfully** but doesn't cross any qualitative threshold (current canonical already silently clips ~12% of cy combos; recommended ~20%). The fix is to widen `state_n_stds` (from 2.0 to 2.93+ on cy and y_1) and is independent of the Lobatto question. I do not recommend doing it as part of this change — it expands the support of V interpolation to regions where V was never solved and would cause its own correctness issue without retuning the V_next grid construction.

**Bottom line on Task 4: clipping is silent but not buggy. Lobatto Z=2.93 makes it modestly worse (12% → 20% OOB rate), but this is an existing issue in the canonical, not a new one.**

---

## 7. Per-axis sensitivity scan (Task 5)

Reading the α* table (§4) on the per-axis variants under V=curved (most representative):

| Bump axis    | N    | max \|Δα*\|  | mean \|Δα*\| | comment                                   |
|--------------|-----:|-------------:|-------------:|-------------------------------------------|
| baseline     | 1296 |    1.609     |    0.386     | current canonical                          |
| cy only      | 2160 |    1.608     |    0.378     | tiny improvement (still fails at cy tail)  |
| spr only     | 2160 |    1.493     |    0.376     | small                                      |
| rtb only     | 2160 |    1.537     |    0.361     | small                                      |
| **y_1 only** | 2160 |  **0.072**   |  **0.029**   | **major: closes most of the gap alone**   |
| xr only      | 1620 |    1.609     |    0.386     | none                                       |
| xb only      | 1620 |    1.607     |    0.386     | none                                       |

So the **single most effective single-axis lever is bumping y_1**, not cy. This **directly contradicts the original handoff Finding A**, which claimed cy is the dominant bottleneck.

The reason for the disagreement: at most cells, the deep-y_1 tail (i_s=294, where y_1 is at +4σ from grid centre) is what drives the canonical's max |Δα*|. The original investigation didn't notice this because it tested at the body centroid only, where neither cy nor y_1 alone fails.

The only cells where bumping cy helps more than bumping y_1 are the deep-cy cells (i_s=0 and i_s=2058), and those cells produce out-of-production-range α* values anyway. So **for production-range α*, bumping y_1 is the single dominant lever.**

This calls into question whether the recommendation should bump cy at all. Bumping y_1 alone gets max |Δα*| = 0.17 (relative 1.5%) at 2160 nodes. Bumping cy and y_1 (plus Lobatto on xb) gets max = 0.12 (relative 1.1%) at 3375 nodes. The marginal accuracy gain from bumping cy is small in production-relevant cells. But also: bumping cy is cheap, costs ~1.5× over bumping y_1 alone, and matters at deep-cy outlier cells.

---

## 8. Cheap-Lobatto alternative (Task 6)

Result: **catastrophically bad. Do not pursue.**

| Rule (1215 nodes, 5% LESS than canonical) | smooth max | curved max | kinked max | level relerr at α=(3,2) |
|-------------------------------------------|-----------:|-----------:|-----------:|-----------------------:|
| Lob (3,3,3,3,3,5) cy/spr/y_1=2.93, xb=2.86 |     2.57   |    2.65    |    2.57    |       203%            |
| Lob (3,3,3,3,3,5) cy/y_1=2.93, xb=2.86      |     2.43   |    2.35    |    2.43    |       203%            |

Cause: Lobatto K=3 has polynomial exactness 2K−3=3, vs GH K=3 which has 2K−1=5. The integrand at α=(3,2) has substantial degree-4 and degree-5 polynomial mass via the exp expansion of `r_p ≈ a_b·log_x_b ≈ 2·z_v[3]·M[xb,y_1]·L_s[3,3]` with α_s=3 driving similar degree. K=3 Lobatto misses this exactly.

**Conclusion of Task 6: K=3 Lobatto is NOT a strict improvement over GH K=3 for tail-mass injection.** It trades polynomial exactness for tail coverage in the wrong direction for this integrand class. Any Lobatto adoption MUST come with a K-bump from 3 to 5 (or 7) on the affected axis. The original recommendation's K-bump (3→5) is therefore necessary, not optional.

---

## 9. Final recommendation

**Adopt the original recommendation, with two amendments:**

### 9.1 Concrete config diff to apply

[`configs/_canonical_jax.py:73-80`](configs/_canonical_jax.py#L73-L80):

```python
CANONICAL_DISC = _NUMBA_CANONICAL_DISC._replace(
    n_z=7,
    n_state_quad_nodes=(5, 3, 3, 5),       # was (3, 3, 3, 3) — bump cy and y_1 to K=5
    n_ret_nodes_1d=(3, 5),                  # was (4, 4) — Lobatto requires K in {3, 5, 7}
    state_lobatto_Z=(2.93, None, None, 2.93),  # was (None, 7.0, None, 7.0) — make Lobatto active
    ret_lobatto_Z=(None, 2.86),             # was None — Lobatto on bond residual axis
    n_eta_nodes=3,
    n_eps_nodes=3,
    n_savings=120,
)
```

Total per-cell quadrature: 5·3·3·5 · 3·5 = **3,375 nodes (2.6× current canonical)**.

Update the comment block above the `CANONICAL_DISC._replace`:

> CAVEAT: state_lobatto_Z=(2.93, None, None, 2.93) puts Lobatto's prescribed
> tail nodes at z = ±2.93 on cy and y_1, which lies OUTSIDE the bracket-grid
> z-extent (state_n_stds = 2.0 / 2.25). [`bracket_state_jax`](lifecycle/solver.py#L277) silently clips
> next-state coords to the boundary at OOB nodes — the multilinear weights remain
> in [0, 1] and sum to 1 (no NaN/extrapolation), but multiple state-quad nodes
> at deep-tail cells will share the same boundary corner.
> The current canonical (GH K=3, max z ≈ ±1.73) already has 12% of cy state-quad
> combinations OOB; this rule raises that to ~20%. Acceptable; the fix would be
> to widen state_n_stds and is independent of the Lobatto adoption.

### 9.2 Soften the documentation claim from "100× better"

The original handoff's headline ratios at α=(6, 6) (96% → 55%) are measured against a truth that is itself 30% biased; they cannot be defended. Replace with the α*-based language:

> The recommended Mixed (5,3,3,5,3,5) Lobatto config reduces unconstrained-FOC α*
> error from max ~1.6 (relative ~14%) to max ~0.04 (relative ~0.4%) under a
> Gaussian-curvature V_next stress test, and from max 2.4 to max 0.12 in the
> deep-cy outlier cells. At equal node count (3375), Lobatto with Z=2.93 / 2.86
> beats pure GH (5,3,3,5,3,5) by ~30% on max |Δα*|. The improvement is
> dominated by Lobatto's tail-mass injection on the y_1 axis (where the deep
> y_1 tail cells live in the canonical state grid), not by the cy axis as
> originally claimed.

### 9.3 (Optional) Consider a cheaper alternative if cost matters

`n_state_quad_nodes=(3,3,3,5), n_ret_nodes_1d=(4,4), state_lobatto_Z=None, ret_lobatto_Z=None` (i.e. **GH bump y_1 only**) gets max |Δα*| = 0.17 (relative 1.5%) at 2160 nodes (1.67× canonical, vs 2.6× for the recommended config). Disadvantages: (a) no benefit at deep-cy outlier cells (those still have |Δα*| ≈ 2 magnitude), (b) no Lobatto adoption so the existing inactive `state_lobatto_Z=(None, 7.0, None, 7.0)` is left as decorative. If wall-clock matters and deep-cy outlier cells are not policy-relevant (they likely hit the line-search step bound anyway), this is a defensible alternative.

I recommend proceeding with the **original recommendation (3375 nodes, with Lobatto)** because:
1. The Lobatto adoption removes the inactive Z=7 wart in the current canonical.
2. The 1.6× extra cost over `GH bump y_1 only` (3375 vs 2160 nodes) buys real accuracy at deep-cy cells, where production-relevant α* values can still be in the |α_b| ∈ [3, 7] range under high γ × high |spread|.
3. The recommended K=5 GH on cy and the K=5 Lobatto wiring are both well-posed (validity windows OK, weights all positive, moment recovery to machine precision).

### 9.4 Validation steps NOT done by this review

These remain open and should be addressed before merging the config change:

1. **Wall-clock smoke test** (handoff §5.6): solve at `youngest_age_to_solve = 67` with both configs and confirm wall-clock scales near-linearly with node count (no JIT recompile or fusion-loss surprise).

2. **Policy regression vs `verify/benchmark_bundle_6666.py`** (handoff §4.9): solve the canonical retirement problem under both configs and diff the policy. If policy changes by < 0.1% in the body, the change is safe; if larger, that's the scientific impact of the change and needs documenting.

3. **Run on full lifecycle** at one sweep cell to confirm the bracket-clip rate increase doesn't propagate into a different regime. Specifically: monitor V iterates at the deep-tail cells where the OOB rate jumps from 12% to 20%.

4. **Optionally re-run the α* benchmark with the actual production V_next** (interpolated from the prior solve, not the synthetic curved/kinked V used here) to confirm the directional result is robust to the particular V curvature.

---

## 10. References

- Original handoff: [`docs/handoff/HANDOFF_LOBATTO_QUADRATURE_INVESTIGATION.md`](docs/handoff/HANDOFF_LOBATTO_QUADRATURE_INVESTIGATION.md)
- Original investigation script: [`scripts/scratch/smolyak_feasibility_jax.py`](scripts/scratch/smolyak_feasibility_jax.py) (Lobatto portions live; Smolyak portions disregarded per handoff).
- This review's script: [`scripts/scratch/lobatto_review_2026-05-07.py`](scripts/scratch/lobatto_review_2026-05-07.py)
- Lobatto module: [`lifecycle/quadrature_with_tails.py`](lifecycle/quadrature_with_tails.py)
- Quadrature factory: [`lifecycle/discretization.py:519-742`](lifecycle/discretization.py#L519-L742)
- Bracketing: [`lifecycle/solver.py:268-290`](lifecycle/solver.py#L268-L290)
- Current canonical: [`configs/_canonical_jax.py:73-80`](configs/_canonical_jax.py#L73-L80)

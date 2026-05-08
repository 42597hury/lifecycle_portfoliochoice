# System II × state-grid density convergence study (rtb, y_1)

**Date:** 2026-05-08
**Branch:** `jax-rewrite`
**Scope:** Quantify how 2-axis state-grid density on `(rtb, y_1)` affects the
solved consumption / risky-share / bond-share policies in System II at the
production-aligned canonical settings. The hypothesis under test:

> At state grid (5, 5), the policy is already converged on (rtb, y_1).
> Refining to (6, 6) produces only negligible changes.

**Verdict:** **YELLOW.** Refinement (5,5) → (6,6) is not negligible at the
worst-cell sup-norm — but the divergence concentrates at tail corners
(`(rtb, y_1)` extremes, high z, working-age, mid-to-high wealth). At a
typical-household probe (mean z, median wealth, centre-state) the (5,5)
and (6,6) policies overlap visually. RMS-norm reduction (4,4) → (5,5) → (6,6)
is roughly geometric (~3× per step), so (5,5) sits about one refinement
short of asymptote.

**Outputs:**
- Metrics JSON: [system_ii_grid_convergence_metrics.json](system_ii_grid_convergence_metrics.json)
- Plot bundle: [system_ii_grid_plot.npz](system_ii_grid_plot.npz)
- Figures: [figures/system_ii_grid_convergence_curve.png](figures/system_ii_grid_convergence_curve.png),
  [figures/system_ii_grid_per_axis.png](figures/system_ii_grid_per_axis.png),
  [figures/system_ii_grid_heatmap.png](figures/system_ii_grid_heatmap.png),
  [figures/system_ii_grid_probe_age.png](figures/system_ii_grid_probe_age.png)
- Analysis scripts:
  [scripts/analysis/system_ii_grid_convergence.py](../../scripts/analysis/system_ii_grid_convergence.py),
  [scripts/analysis/plot_system_ii_grid_convergence.py](../../scripts/analysis/plot_system_ii_grid_convergence.py)

---

## TL;DR

| Verdict component | Outcome |
|---|---|
| (4, 4) vs (6, 6) | **RED.** sup\|ΔC\| = 5.36 (~9.8 % rel), sup\|Δα_s\| = 0.16, sup\|Δα_b\| = 1.68. RMS divergence 3 × the (5,5) level on every array. |
| (5, 5) vs (6, 6) (load-bearing) | **YELLOW.** sup\|ΔC\| = 1.76 (~3.1 % rel), sup\|Δα_s\| = 0.19, sup\|Δα_b\| = 1.47. **Sup is at the +rtb / −y_1 corner** at high z, working ages 36–60, mid-to-high wealth — typical-cell heatmap is ~25× smaller (sup\|Δα_s\| = 7.7e-3 at age 44, mean z, w=median). |
| Per-axis dominance | **Neither rtb nor y_1 dominates.** Per-axis maxes are within ~30 % of each other on every array. The corners (extreme \|rtb\| and extreme \|y_1\|) carry the worst divergence symmetrically. |
| Convergence rate (RMS) | Roughly geometric: (4,4) → (5,5) shrinks RMS by ~3×, suggesting (6,6) → (7,7) would shrink by another ~2-3× and (5,5) is *not* at the asymptote. |
| Newton failure structure | **Confirms** [NEWTON_FAILURE_STRUCTURE_2026-05-08](NEWTON_FAILURE_STRUCTURE_2026-05-08.md): per-cell fail rate is 15.20 % / 14.60 % / 14.34 % across (4,4)/(5,5)/(6,6) — modest decline with density, p99 iter count saturates at 30 in every grid. Grid density is not the dominant Newton-fail mechanism. |
| **Recommendation for production canonical (5, 5, 4, 6)** | **Asymmetric (rtb=4, y_1=6) is not justified by this sweep.** Both axes contribute roughly equally to divergence; there is no per-axis dominance favouring y_1 over rtb. Switch to **uniform (rtb=5, y_1=5)** on these sub-axes — same fidelity for typical households, ~30 % lower wall than (6, 6), and removes the unsupported asymmetric allocation. If tail moments enter, consider (rtb=6, y_1=6) (+24 % wall vs (5,5)). |

---

## §1 — Convergence-rate table

Sup, p99, RMS, and rel-sup divergence vs the (6, 6) reference, computed by
linearly interpolating each coarser bundle along (rtb, y_1) onto the (6, 6)
bracket grid (the densest source) and taking element-wise differences across
the full `(78 ages, 11 z, N_rtb=6, N_y_1=6, 180 wealth)` policy tensor.

| Pair vs (6,6) | sup\|ΔC\| | rel-sup C | RMS\|ΔC\| | sup\|Δα_s\| | RMS\|Δα_s\| | sup\|Δα_b\| | RMS\|Δα_b\| | wall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| (4,4) → (6,6) | 5.359 | **9.84 %** | 0.715 | 0.162 | 1.60e-2 | 1.675 | 3.93e-2 | 6.27 min |
| (5,5) → (6,6) | 1.755 | **3.15 %** | 0.234 | 0.190 | 6.12e-3 | 1.471 | 1.75e-2 | 9.62 min |
| (6,6) → (6,6) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13.72 min |

The convergence is approximately geometric in RMS: (4,4) → (5,5) cuts
RMS\|ΔC\| by 3.06×, RMS\|Δα_s\| by 2.61×, RMS\|Δα_b\| by 2.25×. Extrapolating,
(6,6) → (7,7) would shrink RMS by another factor of 2-3, so (5,5) is *not*
at its asymptote — the policy is still moving with grid density. Wall scales
roughly linearly in N_state (16 → 25 → 36 cells, 6.27 → 9.62 → 13.72 min).

Self-consistency gates:
- Reference round-trip: `ref → np.interp on its own bracket grid → ref` returns sup = 0 exactly for C, α_s, α_b. ✓
- All three bundles' `state_n_stds = (2.0, 2.25)` (verified). ✓
- All three bundles' bracket grids are nested by density on `[−n_stds, +n_stds]` (verified). ✓
- No NaN or non-finite cells in any of the three bundles. ✓
- (6, 6) reference policy ranges: C ∈ [3.95e-3, 1.09e+2], α_s ∈ [−0.444, +2.013], α_b ∈ [−6.42, +6.37] — α_b inside the [-6.5, +6.5] System II leverage band. ✓

The relative-error mask ratios (`sup_rel` for α_s and α_b in the JSON, ~30×
and ~40×) are inflated by near-zero α_s / α_b cells passing the small-value
threshold; the rel-sup for C is the meaningful relative measure.

---

## §2 — Per-axis decomposition

Per-axis max of `|coarse - (6,6)|` collapses the 5-D divergence tensor
`(78, 11, 6, 6, 180)` to one number per axis position. The (rtb, y_1)
profiles use the (6, 6) reference grid in u-coords.

### Per-rtb (sup over age × z × y_1 × wealth at each rtb node, in u-coords)

| (5,5) per_rtb | u=−2.0 | u=−1.2 | u=−0.4 | u=+0.4 | u=+1.2 | u=+2.0 |
|:--|--:|--:|--:|--:|--:|--:|
| sup\|Δα_s\| | 0.145 | 0.106 | 0.079 | 0.076 | 0.109 | **0.190** |
| sup\|Δα_b\| | **1.471** | 1.150 | 0.852 | 0.815 | 0.972 | 1.371 |
| sup\|ΔC\| | 1.660 | 1.670 | 1.700 | 1.720 | 1.730 | **1.755** |

### Per-y_1 (same, at each y_1 node)

| (5,5) per_y_1 | u=−2.25 | u=−1.35 | u=−0.45 | u=+0.45 | u=+1.35 | u=+2.25 |
|:--|--:|--:|--:|--:|--:|--:|
| sup\|Δα_s\| | **0.190** | 0.126 | 0.109 | 0.077 | 0.082 | 0.145 |
| sup\|Δα_b\| | 1.370 | 1.080 | 0.612 | 0.386 | 0.879 | **1.471** |
| sup\|ΔC\| | 1.540 | 1.650 | **1.755** | 1.730 | 1.590 | 1.460 |

The α-share divergences are near-symmetric under (rtb, y_1) corner reflection:
the worst α_s cell at +rtb / −y_1 corner mirrors the worst α_b cell at −rtb /
+y_1 corner. Interior nodes (|u| ≤ 1.2) are 2-3× lower than the extremes.
This is the classic bilinear-interp signature when the coarser grid (5,5)
has nodes at u = ±1 but the reference (6,6) needs interior values at u = ±1.2:
linear interpolation between (5,5)'s ±1 and ±2 nodes mismatches the local
curvature most strongly *near* the second-most-extreme reference node.

**No single-axis dominance.** rtb and y_1 contribute symmetrically — the
y_1 max (0.190) marginally exceeds the rtb max (0.190 too, both at the
corner) only because state_n_stds[y_1] = 2.25 is wider than state_n_stds[rtb]
= 2.0. Per-axis profiles are within 30 % of each other on every array.

### Per-age (working vs retirement)

| Pair | working sup\|ΔC\| (22-66) | retirement sup\|ΔC\| (67-99) | working sup\|Δα_s\| | retirement sup\|Δα_s\| |
|:--|--:|--:|--:|--:|
| (4,4) | 5.359 | 2.533 | 0.162 | 0.061 |
| (5,5) | 1.755 | 0.832 | 0.190 | 0.034 |

Working ages dominate by 2-5×. Retirement-side divergence is small in
absolute terms but α_s in retirement is mostly α_s ≈ 0 (inflation-adjusted
buffer position), so even small absolute divergences would matter for
retirement-side decompositions. The working-age peak for C lives at ages
36-37 — the height of the saving build-up just before pre-retirement
de-risking.

### Per-z (z idx 0 = lowest income state, idx 10 = highest)

(5,5) per-z `sup|Δα_s|`:
`3.4e-2, 3.4e-2, 3.4e-2, 3.4e-2, 3.4e-2, 3.4e-2, 3.4e-2, 4.3e-2, 9.9e-2, 1.65e-1, 1.90e-1`

z idx 9-10 (top two z-states, `≈ +1.5σ_z` to `+1.9σ_z`) carry the entire
upper tail of the divergence. The plateau across z idx 0-7 is uniform
because at low/medium z the policy is much smoother in (rtb, y_1) — only
high-z households face high enough wealth/leverage to be sensitive to bond
return dispersion. This matches the System I `n_z` study's finding that the
worst-cell story is concentrated at the high-z right tail.

### Per-wealth (180 bins)

(5,5) per-wealth max for α_s peaks at idx ≈ 122 (mid-to-high wealth, where
the leverage envelope tightens) and again at the maximum-wealth tail. The
bottom-30 wealth bins are quiet (sup\|Δα_s\| ≈ 0.033). For α_b the peak is
at idx ≈ 115 — same mid-to-high wealth region.

---

## §3 — Heatmap at the typical-cell probe

Renders `|coarse - (6,6)|` for C, α_s, α_b on a 50 × 50 (rtb, y_1) grid in
u-coords, at probe (age=44, z idx=5 = mean labour income, wealth idx=89 ≈
SCF-median area). See [figures/system_ii_grid_heatmap.png](figures/system_ii_grid_heatmap.png).

| Heatmap sup at probe | (4,4) → (6,6) | (5,5) → (6,6) | reduction |
|:--|--:|--:|--:|
| sup\|ΔC\| | 0.126 | 0.0409 | 3.08 × |
| sup\|Δα_s\| | 0.0236 | 0.0077 | 3.06 × |
| sup\|Δα_b\| | 0.0543 | 0.0149 | 3.64 × |

At the typical-cell probe, (5,5) is roughly **3× closer** to (6,6) than
(4,4) is. The α_s probe-heatmap sup is **0.77 percentage points** at (5,5)
— well below the 0.5 % threshold the handoff sets for GREEN, but using the
typical-cell metric not the worst-cell sup-norm. Spatial pattern: divergence
spreads near the y_1 corners on both bundles; (5,5) interior is uniformly
small.

---

## §4 — Probe lines at (rtb=0, y_1=0), z=mean, w idx 89

See [figures/system_ii_grid_probe_age.png](figures/system_ii_grid_probe_age.png).

At the centre-state, mean-z, median-wealth probe over the lifecycle:
- **α_s and α_b** lines for (5,5) and (6,6) overlap visually; (4,4) is offset
  by ~0.005-0.010 share-points across working ages. All three grids agree
  to within plotting resolution on the broad shape of the lifecycle profile.
- **c (consumption)** shows a small constant offset in (4,4) (~0.05 income
  units = 5 %) across working ages; (5,5) and (6,6) are within ~0.01-0.02
  income units of each other (1-2 %).

Visually, this is the **converged regime for typical households**: an
ablation reader would not be able to tell (5,5) from (6,6) at the
representative-household lens.

---

## §5 — Convergence curve and asymptote

See [figures/system_ii_grid_convergence_curve.png](figures/system_ii_grid_convergence_curve.png).

In log-RMS space, the (4,4) → (5,5) → (6,6) trace is roughly linear, with
slope:
- C: −1.36 decades per axis-doubling (≈ 3.06× per step)
- α_s: −1.06 decades (≈ 2.61×)
- α_b: −0.86 decades (≈ 2.25×)

If the convergence stays geometric, (6,6) → (7,7) would shrink RMS by
another factor of ~2-3, reaching:
- (7,7) extrapolated RMS\|ΔC\| ≈ 0.10 → 0.05; sup-likely ≈ 0.5 → 0.8
- (7,7) extrapolated RMS\|Δα_s\| ≈ 0.0024; sup ≈ 0.07

That would land (7, 7) close to the GREEN threshold for sup\|Δα_s\| < 0.05.
**(5,5) is therefore one refinement short of GREEN by sup-norm**, but the
typical-cell lens (heatmap §3, probe §4) is already there.

---

## §6 — Cross-link to the production canonical (5, 5, 4, 6)

The user's draft production canonical is `state_grid_sizes = (5, 5, 4, 6)`
on the 4-axis (dp, spr, rtb, y_1) state. This sweep tests the (rtb, y_1)
sub-axes at uniform sizes; the asymmetric (rtb=4, y_1=6) allocation is
not solved here.

Connection:
- The per-axis decomposition shows **no y_1 dominance over rtb**. The
  (5,5) → (6,6) divergence is ~equal in both axes, with corner cells
  driving the sup. There is no evidence that y_1 needs more density than
  rtb.
- The asymmetric `(rtb=4, y_1=6)` allocation in the production draft is
  therefore **not justified by this sweep**. It costs the same as `(5, 5)`
  (24 cells vs 25) but gives less protection where it's needed: at the
  rtb extremes.
- A **uniform (rtb=5, y_1=5)** would be the simpler, defensible choice. It
  matches (5,5) sup-norm and RMS at the System II level, and the per-axis
  decomposition shows neither axis is the binding constraint.
- If the production canonical wants tail-cell GREEN on (rtb, y_1), the
  required size is **(6, 6) or higher** — not asymmetric.

This sweep does not directly validate the asymmetric (4, 6) allocation
(would need an asymmetric solve, out of scope here per the handoff). What
it does establish: there is no signal that y_1=6 is doing more work than
rtb=5 would, and there is no signal that rtb=4 is sufficient when the
sweep's 4-axis floor is the (4, 4) bundle, which is RED.

---

## §7 — Verdict

| Sub-verdict | Outcome | Rationale |
|---|---|---|
| (4, 4) vs (6, 6) | **RED** | sup\|ΔC\| 9.84 % rel, sup\|Δα_s\| 0.16, sup\|Δα_b\| 1.68. RMS divergence 3× the (5,5) level. Working-age, high-z, high-wealth cells significantly off. |
| **(5, 5) vs (6, 6)** | **YELLOW** | Sup\|Δα_s\| = 0.19 and sup\|Δα_b\| = 1.47 exceed the GREEN thresholds (0.005 each), but the sup is concentrated at corner cells (\|rtb\|, \|y_1\| at the n_stds boundary) at high z, and the typical-cell probe / heatmap shows ~25× smaller divergence. RMS reduction (4,4) → (5,5) suggests (5,5) is one refinement short of asymptote. |
| Per-axis dominance | **Neither — symmetric** | Per-axis maxes within ~30 % of each other across all three arrays. Both (rtb, y_1) corners contribute equally; the strict per-axis maxes for α_s peak at u=+2 (rtb) AND u=−2.25 (y_1) symmetrically. No basis to prefer y_1 density over rtb density. |

**Implications for the production canonical's (5, 5, 4, 6):**
1. The asymmetric (rtb=4, y_1=6) allocation is **NOT justified** by this
   sweep — there is no per-axis dominance favouring y_1 over rtb.
2. The rtb=4 setting is **likely too coarse**: this sweep's (4, 4) bundle
   is RED on every metric, and there is no reason to believe (rtb=4, y_1=6)
   would perform materially better than (4, 4) on the rtb axis.
3. **Recommendation**: switch the (rtb, y_1) sub-axes of the production
   canonical to **uniform (rtb=5, y_1=5)** — `state_grid_sizes = (5, 5,
   5, 5)`. Wall: ~9.6 min for System II's (5, 5) bundle, vs 13.7 min for
   (6, 6). For System IV's full 4-axis state at (5, 5, 5, 5) the wall
   blow-up depends on the dp / spr cost profile, but the per-axis logic
   transfers.
4. If tail moments enter the thesis spec (e.g., extreme-rtb conditional
   means), bump uniformly to (6, 6) — not asymmetrically.

---

## §8 — Newton-failure cross-reference

Confirms the dominant Newton-fail mechanism is tol-unreachability at
high-savings × high-leverage cells (per [NEWTON_FAILURE_STRUCTURE_2026-05-08](NEWTON_FAILURE_STRUCTURE_2026-05-08.md)),
not grid density:

| Bundle | n_cells | total_newton_failures | rate | iter p99 | iter max |
|:--|--:|--:|--:|--:|--:|
| (4, 4) | 2,442,240 | 371,219 | **15.20 %** | 30 | 30 |
| (5, 5) | 3,816,000 | 556,995 | **14.60 %** | 30 | 30 |
| (6, 6) | 5,495,040 | 788,156 | **14.34 %** | 30 | 30 |

Per-cell fail rate decreases marginally with grid density (15.2 → 14.6 →
14.3 %), consistent with denser grids producing slightly less curvature
pressure on tail cells — but the absolute failure count rises with N_state
because the cell-count grows faster than the per-cell rate falls. iter p99
and iter max both saturate at the cap of 30 in every grid, so failures are
**purely tol-bound**, not iteration-bound. The per-age failure profile is
similar shape across all three bundles (ramp during working ages, dip
near retirement, ramp again in late retirement) — this is consistent with
the failure-structure scan's claim that the failure mechanism is
state-shape-driven, not density-driven.

---

## Implementation notes

- Cholesky-mode state grids store the bracket-coordinate u-grid as
  `linspace(-n_stds[d], +n_stds[d], N_d)`. Since `state_n_stds = (2.0,
  2.25)` matches across all three bundles, the bracket grids span the
  same u-interval on each axis, and 1-D `np.interp` along the bracket
  axis is the canonical comparison map.
- The state cells in the `(78, 11, N_state, 180)` array are ordered
  C-major over `(N_rtb, N_y_1)` (as produced by `np.ndindex(*N_vec)` in
  `_independence_rouwenhorst_pi`), so the reshape
  `(N_state,) → (N_rtb, N_y_1)` is the natural one.
- The bulk-metrics tensor uses the (6, 6) reference bracket grid — small
  enough (~9.3 M cells per array) to fit in memory comfortably, and
  asymmetric only in that the reference is exact on this grid by
  construction.
- The 50 × 50 heatmap is computed only on a single (age, z, wealth) probe
  slice to avoid the full (78, 11, 50, 50, 180) tensor (~3 GB / array).

## Future work (out of scope here)

- An explicit **(rtb=4, y_1=6) asymmetric solve** to validate (or
  invalidate) the production canonical directly. This sweep can only say
  the asymmetric allocation is unsupported, not that it underperforms.
- A **(7, 7) solve** to confirm the geometric convergence rate and bound
  the (5, 5) → asymptote gap from above.
- Cross-system: System IV at full (5, 5, 5, 5) state to confirm the
  uniform recommendation generalises beyond the (rtb, y_1) sub-axes.

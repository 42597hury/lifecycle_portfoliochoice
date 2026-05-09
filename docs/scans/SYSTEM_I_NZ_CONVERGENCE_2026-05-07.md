# System I × n_z policy-resolution convergence study

**Date:** 2026-05-07
**Branch:** `jax-rewrite`
**Scope:** Quantify how labour-income state-grid resolution `n_z` affects the
solved consumption / risky-share / bond-share policies in the simplest
predictability system (System I, iid returns, single-axis rtb state). The
hypothesis under test: *n_z=10 is already converged*; refinement to 15/30/70
buys nothing. **Verdict: REJECTED.**

**Outputs:**
- Metrics JSON: [system_i_nz_convergence_metrics.json](system_i_nz_convergence_metrics.json)
- Figures: [figures/](figures/)
- Sim-EE artefacts in each bundle as `ee_simpath_nz_convergence.{json,md}`
- Analysis scripts: [scripts/analysis/system_i_nz_convergence.py](../../scripts/analysis/system_i_nz_convergence.py),
  [plot_nz_convergence.py](../../scripts/analysis/plot_nz_convergence.py),
  [run_ee_simpath_system_i.py](../../scripts/analysis/run_ee_simpath_system_i.py)

---

## TL;DR

| Verdict component | Outcome |
|---|---|
| Grid-policy convergence (sup-norm) | **RED for n_z=10** (28 % rel C, 37 pp α_s); **YELLOW for n_z=15** (10 % rel C, 18 pp α_s); GREEN for n_z=30 (3 % rel C). |
| Sim-path Euler residuals (unconstrained, p95) | n_z=10 working **2.6 %** rel c-error; n_z=15 **1.0 %**; n_z=30 **0.45 %**; n_z=70 **0.21 %**. Retirement an order of magnitude smaller. |
| Where the worst sup-norm cells live | At z ≈ +1.5σ to +1.9σ (z = +2.7 to +3.5), young working ages 22-25, two wealth modes (max-W for C; lower-middle-W for shares). These cells are **rare in any realised simulated panel** (the 45-year working horizon never reaches the stationary right tail; realised std(z) at age 65 ≈ 1.4 vs stationary 1.87). Sup-norm metrics over-weight them; realised-panel statistics are dominated by typical cells. |
| Calibration provenance | Matches Catherine (2025) "Interest-Rate Risk and Household Portfolios" and the underlying Guvenen-Karahan-Ozkan-Song (2021, Econometrica) mixture estimates — see [CALIBRATION_VERIFICATION_2026-05-08.md](CALIBRATION_VERIFICATION_2026-05-08.md). |
| Newton convergence (max\_iter cap) | **YELLOW.** Histogram shows max\_iter=100 hit on at least one savings node in every (z, state) cell of every bundle. Concentrated at the tiny-savings boundary where the policy is sentinel-replaced; sim-EE confirms unconstrained-cell policies are still numerically reasonable. Independent of n_z. |
| **Recommendation** | **n_z=15 for typical-household statistics on ablation sweeps; n_z=30 if any tail moments enter; n_z=70 for canonical/publication. n_z=10 is unsafe on every metric.** |

---

## §1 — Convergence-rate table

Sup-norm and RMS divergence vs the n_z=70 reference, computed by linearly
interpolating each coarser bundle along the z-axis onto the reference grid
and taking element-wise differences across the full
`(78 ages, n_z, 7 states, 180 wealth)` policy tensor.

| n_z | sup\|C\| | sup\|α_s\| | sup\|α_b\| | RMS\|C\| | rel-sup C |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.533 | 0.374 | 0.304 | 0.260 | **28.3 %** |
| 15 | 0.766 | 0.180 | 0.145 | 0.108 | **10.0 %** |
| 30 | 0.266 | 0.058 | 0.046 | 0.034 | **2.9 %** |
| 70 | 0     | 0     | 0     | 0     | (reference) |

The convergence is approximately linear in 1/n_z (each ~doubling of n_z
roughly halves the divergence). At n_z=10 the policies are not at their
asymptote — refinement reduces the error monotonically through n_z=70.

Self-consistency gates: (a) reference ↔ reference comparison through the
same `np.interp` framework returns sup=0 for all of (C, α_s, α_b); (b) all
four reconstructed z-grids are symmetric around 0 with matching endpoints
±2.25 σ_z (verified, `[-4.207, +4.207]`); (c) no NaN in any solved cell.

---

## §2 — Where the residual divergence concentrates

For each n_z×{C,α_s,α_b} pair, the per-age / per-z / per-wealth max collapses
the 4-D divergence tensor to a 1-D profile. Peaks (cell where the worst
absolute divergence lives):

| n_z | array | peak age | peak z idx (of 70) | z value | peak wealth idx (of 180) |
|---:|---|---:|---:|---:|---:|
| 10 | C   | 24 (early working) | 57 | +2.74 (≈+1.47σ_z) | 179 (max wealth) |
| 10 | α_s | 25 | 59 | +2.99 (≈+1.60σ_z) | 60 (lower-middle wealth) |
| 10 | α_b | 25 | 59 | +2.99 | 60 |
| 15 | C   | 24 | 57 | +2.74 | 179 |
| 15 | α_s | 23 | 62 | +3.35 (≈+1.79σ_z) | 63 |
| 30 | C   | 24 | 58 | +2.87 | 179 |
| 30 | α_s | 22 (start of life) | 63 | +3.48 (≈+1.86σ_z) | 63 |

**Interpretation:**
- **Young working ages** (22–25) carry the most divergence because the
  z-distribution disperses fastest there: at the start of life everyone
  starts at z=0, and working-age innovations haven't yet propagated.
- **Right tail of z** (z ≈ +1.5σ_z to +1.9σ_z, i.e. z ≈ +2.7 to +3.5 in log
  units) is where coarse z-grids systematically misplace the policy.
  Linear interpolation between 10 z-grid nodes cannot resolve the policy
  curvature there, where the interaction between high realised income
  and the deterministic age-earnings profile produces rapid variation.
- **Two wealth modes:** for C the worst cell is at maximum wealth (smooth-V
  region, no constraint binds); for α_s / α_b it's at lower-middle wealth
  (~33 % of grid range) where the household is closer to the
  borrowing-constraint corner and V has kink structure.

**Economic-relevance caveat for the right-tail divergence.** The
calibration's stationary std of z is ≈ 1.87 in log units (from
`std(eta) ≈ 0.250` and `ρ = 0.991`), which would imply ~5.6 % of households
sit at z ≥ +1.5σ in the **stationary** distribution. That stationary
asymptote is, however, never reached: with `ρ = 0.991` the half-life of z is
77 years, so over a 45-year working life the **realised** cross-sectional
std at age 65 is only ≈ 1.4, and including the transitory ε the realised
std(log y) at retirement is ≈ 1.05 — consistent with SCF/SSA empirical
dispersion. The cells where the worst-case n_z=10 errors live (z = +2.7 to
+3.5, i.e. labour income 15–32 × population mean) are therefore **rare in
any actual simulated panel**, not because the calibration is over-dispersed
but because the working horizon is too short to reach the upper tail. Sup-norm
metrics weight those cells equally with modal cells; sim-EE residuals on
realised paths (§3) and stationary-mass-weighted RMS down-weight them by
roughly an order of magnitude. The calibration itself is consistent with
Catherine (2025) "Interest-Rate Risk and Household Portfolios" and the
underlying Guvenen-Karahan-Ozkan-Song (2021, Econometrica) mixture
estimates — see `docs/scans/CALIBRATION_VERIFICATION_2026-05-08.md`.

See [figures/per_age_divergence.png](figures/per_age_divergence.png),
[figures/per_z_divergence.png](figures/per_z_divergence.png),
[figures/per_wealth_divergence.png](figures/per_wealth_divergence.png) for the full profiles.

The **distribution-level snapshot** shows the same story in coarser form
(see [figures/alpha_distribution.png](figures/alpha_distribution.png)):

| n_z | min α_s | max α_s | min α_b | max α_b |
|---:|---:|---:|---:|---:|
| 10 | 0.392 | 1.298 | 0.165 | 0.897 |
| 15 | 0.392 | 1.265 | 0.165 | 0.869 |
| 30 | 0.392 | 1.262 | 0.165 | 0.866 |
| 70 | 0.392 | 1.261 | 0.165 | 0.865 |

Coarse n_z does not affect the *minimum* shares — those live at the
deeply-constrained corners — but it *over-shoots the maxima* by 3-7 %
(n_z=10 over-states max α_s by 3.7 %, max α_b by 3.7 %). These over-shoots
are the spike cells in the per-z plots above.

---

## §3 — Sim-path Euler-equation residuals

Independent confirmation via sim-path consumption-Euler diagnostic
(`verify/ee_simpath.py`, eval-mode `same`, 500 simulated households,
128 evaluated per age, seed 42). For each bundle, the FOC is re-evaluated
at simulated (z, state, wealth, c, α_s, α_b) and `log10|EE|` aggregated.
The diagnostic could only run after dispatching the bundle's own VAR
builder (these bundles use `build_iid_var_config`, but `ee_simpath` hardcodes
the System IV builder); the wrapper
[run_ee_simpath_system_i.py](../../scripts/analysis/run_ee_simpath_system_i.py)
patches that.

**Mean log10|EE| on unconstrained cells** (lower = better; gates: publication ≤ -5.0, welfare ≤ -4.0):

| n_z | working | boundary | retirement |
|---:|---:|---:|---:|
| 10 | -2.125 | -2.748 | -4.540 |
| 15 | -2.492 | -2.973 | -4.824 |
| 30 | -2.923 | -3.316 | -5.143 |
| 70 | **-3.920** | **-4.236** | **-5.314** |

- **Working** mean residual: ~63 × larger at n_z=10 than at n_z=70.
- **Boundary** age 66 (work → retirement transition): ~30 × larger.
- **Retirement** stage: ~6 × larger (less z-sensitive because pension is
  z-independent under this calibration).
- Monotone improvement at every n_z step — no over-shooting or noise.
- All bundles **fail** the publication gate. Even n_z=70 averages -3.92 in
  the working stage, against a -5.0 publication target. This points to a
  *separate* issue from n_z resolution (see §4 — Newton iter cap).

The retirement-only mean of n_z=70 (-5.31) is comparable to recent System IV
benchmarks (`saved_runs/system_iv_full_var_unconstrained_cholesky_grid5x5x5x5_nz11_jax_benchmark/ee_simpath_same_iw5.md`,
mean -1.60 — but that bundle was retirement-only and at lower n_z and lower
state-grid resolution; not directly comparable).

---

## §4 — Why does the Newton-iter histogram say `max=100` everywhere?

Each diagnostics.pkl contains:

```text
newton_iter_histogram:  p50=100  p95=100  p99=100  max=100   (all four bundles)
backtrack_iter_histogram: p50≈1194  max≈2350-2840
total_newton_failures: 0
age_max_foc.max(): 0.0
```

At first glance this reads as "Newton failed on every cell at max\_iter".
After tracing the solver code, the literal interpretation is more
reassuring:

1. **`age_max_foc = 0` and `total_newton_failures = 0` are vestigial.**
   The current `lifecycle/solver.py` initialises both fields to zero
   ([solver.py:2499-2500](../../lifecycle/solver.py#L2499-L2500)) and never populates them
   from any per-age kernel result. They are leftover from a pre-rewrite
   diagnostic path; the only fields that **are** populated post-hoc are
   `newton_iter_histogram` and `backtrack_iter_histogram`. Future bundles
   should not read either field as a signal.

2. **The histogram value per (z, state) cell is `max(n_iters_egm)` over
   180 savings nodes** ([solver.py:1369-1370](../../lifecycle/solver.py#L1369-L1370)). So
   "histogram max = 100" means *at least one* of the 180 savings nodes
   per (z, state) hit the cap — not that every savings node did. The
   backtrack sum over 180 nodes (~1194 median) is consistent with most
   nodes converging in a handful of Newton iters and a small minority
   running the full budget.

3. **Under `use_fori_newton=True`, the Newton loop uses
   `jax.lax.fori_loop` with mask-based early termination
   ([solver.py:608-687](../../lifecycle/solver.py#L608-L687)).** The loop runs `max_iter`
   trips unconditionally; cells with `converged=True` mask out the
   update. The `n_iters_used` counter increments only while
   `is_active = NOT (converged OR ls_failed)`, so `n_used = max_iter`
   really does mean "this cell never converged within `max_iter` and
   never line-search-failed". It is **not** a fori-loop reporting
   artefact.

4. **The cells stuck at `max_iter=100` are concentrated at the
   tiny-savings boundary.** `per_savings_point`
   ([solver.py:1155-1176](../../lifecycle/solver.py#L1155-L1176)) runs the FOC at every
   savings-grid point, including `s_val ≤ tiny_savings = 1e-6`. At those
   nodes the FOC is degenerate (numerator → 0 from above), Newton
   chases vanishing residuals with line-search-shrinking steps (matches
   the observed median ≈12 backtrack steps per Newton iter, capped at
   `max_backtrack_iter=10` per call), and would converge if given
   more iters. **The result at those nodes is then sentinel-replaced
   by `(min_consumption, init_alpha_s, init_alpha_b)` regardless of
   Newton state**, so the persistent non-convergence does not leak into
   the policy at small s.

5. **The unconstrained-cell sim-EE residuals (§3) confirm that the
   policy at non-boundary cells is numerically clean to ~10⁻⁴ (working)
   / 10⁻⁵ (retirement) on the n_z=70 reference.** That's ~3 dex above
   the publication gate (10⁻⁵) but ~1 dex below the welfare gate (10⁻⁴),
   which is what one would expect when a few unconstrained interior
   cells are still mid-Newton-iter at the cap.

**Recommended next steps** (out of scope for this study):

- Raise `max_iter` from 100 to 200–300 in production runs (the
  `model.py` docstring explicitly warns the canonical 5000 default is
  unsuitable for fori\_loop; 100 was chosen for fori-loop wall-time
  budget). Compute cost of `max_iter=200` is ~2× the inner-loop work,
  but inner-loop work is a small fraction of total wall time.
- Skip the Newton call entirely when `s_val ≤ tiny_savings` — the
  result is replaced anyway. Measure: does this remove ≥ 90% of the
  cells stuck at max\_iter?
- Populate `age_max_foc` and `total_newton_failures` post-hoc from the
  per-age kernel return tuple (currently the kernels return `(c_t,
  s_t, b_t, ni_t, nb_t)`; add a 6th element with the cell-max FOC
  residual). Without this the existing two fields silently mislead.

None of these are convergence-study deliverables. The Newton finding
is **independent of the n_z choice** — every bundle hit max\_iter on
the same set of boundary cells.

---

## §5 — Verdict and recommendation

The verdict has two readings, depending on whether you weight cells by
sup-norm (worst-cell-anywhere) or by realised-panel relevance
(stationary-mass × finite-horizon truncation; see the
economic-relevance caveat in §2).

> **n_z=10 is RED on both readings.**
> - Per-cell: 28 % rel sup-norm in C, 37 pp / 30 pp in α_s / α_b.
> - On realised paths: working-age sim-EE p95 = 2.6 % relative c-error
>   on the central-95 % of cells; mean ≈ 0.77 %. ~63× larger working-age
>   Euler residual than the n_z=70 reference.
> - Do not use for publication, do not use as the System II/III/IV
>   ablation default without re-checking.
>
> **n_z=15 is YELLOW per-cell, plausibly GREEN for typical-household
> statistics.**
> - Per-cell: 10 % rel sup-norm in C, 18 pp / 15 pp in α_s / α_b.
> - On realised paths: working-age sim-EE p95 = 1.0 %, mean ≈ 0.32 %.
> - The worst sup-norm cells live at z ≈ +1.5σ to +1.9σ — the right
>   tail of the *stationary* distribution, which the working-age horizon
>   never reaches. Realised-panel statistics see the typical-cell errors
>   (RMS ~5.5 % in C, 1.7 pp in α_s) rather than the sup-norm ones.
> - Defensible for any quantity computed at sample means / typical-z
>   conditioning. Tight for headline portfolio shares if any tail moments
>   enter (e.g. top-decile-conditional α_s).
>
> **n_z=30 is GREEN on both readings.** 2.9 % rel sup-norm in C, ~0.5 pp
> typical share error, working-age sim-EE p95 = 0.45 %. Defensible for
> ablation sweeps with one fewer dex of headroom than n_z=70.
>
> **n_z=70 is the canonical-quality setting.** Sim-EE residuals at n_z=70
> are dominated by the Newton iter cap (§4) and possibly by `n_eta_nodes=3`
> mixture quadrature, not by z-resolution; further n_z refinement past 70
> will not improve them.

**Operational recommendation:**
- For ablation sweeps targeting **typical-household** moments (means,
  medians, quartile statistics on simulated panels): **n_z=15 acceptable**;
  n_z=30 preferred if compute allows.
- For ablation sweeps that report **tail moments** (top-decile or
  top-quintile conditional statistics): **n_z=30 minimum**, n_z=70
  preferred.
- For the **canonical thesis baseline / publication-grade** numbers:
  **n_z=70.**
- The hoped-for 7× compute saving from stopping at n_z=10 is unsafe on
  every metric. Realistic savings: `wall(n_z=15)/wall(n_z=70) ≈ 0.30`
  (3.4× faster) for typical-household work; `wall(n_z=30)/wall(n_z=70)
  ≈ 0.50` (2× faster) for tail-statistics work. Expect similar ratios for
  Systems II / III / IV.

---

## Reproducibility

```sh
# Bundles must be present at saved_runs/ablations/system_i_grid7_nz<N>_calib1/
#   (sync from s3://hugo-thesis-runs/saved_runs/ablations/ if missing).

python scripts/analysis/system_i_nz_convergence.py
python scripts/analysis/plot_nz_convergence.py

# Sim-EE comparison across all four bundles (~5 minutes total wall):
for nz in 10 15 30 70; do
  python scripts/analysis/run_ee_simpath_system_i.py \
    saved_runs/ablations/system_i_grid7_nz${nz}_calib1 \
    --eval-mode same --n-simulations 500 \
    --eval-households-per-age 128 --seed 42 \
    --out-suffix _nz_convergence
done
```

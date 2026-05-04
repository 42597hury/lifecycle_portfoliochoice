# Grid/Quadrature Policy Convergence Tracker

Last updated: 2026-05-03

## Purpose

Track the partial-solve convergence exercise for lifecycle policy functions as
we vary:

- financial-state grid size
- return quadrature
- state quadrature

The goal is to determine whether the policy functions are stable under finer
discretization, and to identify the smallest configuration that is "good
enough" for the portfolio-choice problem.

This tracker is the bookkeeping layer for:

- what has been run
- what is currently running
- which saved bundles are clean comparison candidates
- which tests we want to run on completed bundles

## Canonical Comparison Rules

To keep convergence comparisons interpretable:

- Hold `state_grid_mode="principal"` fixed.
- Hold `n_z=9`, `n_eta_nodes=3`, `n_eps_nodes=3`, `n_wealth=150`,
  `n_savings=150` fixed unless explicitly testing one of them.
- Keep `state_n_stds` fixed across a comparison family.
- Change one main axis at a time:
  - first return quadrature
  - then state grid size
  - then state quadrature only if needed
- For cross-grid comparisons, do not compare native array entries directly.
  Compare policies on a common probe set.

## Current Sweep Strategy

### Phase 1: Return quadrature on small grid

Use `state_grid_sizes=(5,5,5)` as the screening grid and compare:

- `(3,5,3)` baseline
- `(3,7,5)`
- `(5,9,5)` high-accuracy reference

### Phase 2: Include the difficult pre-retirement ages

For shortlisted quadrature configs, solve ages `65-99`:

- all retirement ages
- working ages `65` and `66`

### Phase 3: Grid refinement

With quadrature frozen at the smallest acceptable setting, compare:

- `(5,5,5)`
- `(7,7,7)`
- `(9,9,9)`

### Phase 4: State quadrature refinement

Only if policy differences still look material after Phases 1-3.

## Known Bundles

### Clean reference bundles

| Label | Bundle | Status | Ages solved | Key discretization notes |
| --- | --- | --- | --- | --- |
| Full small-grid solve | `saved_runs/unconstrained_principal_grid5x5x5_nz9` | `complete` | `22-99` | `state_grid_sizes=(5,5,5)`, `n_ret_nodes_1d=(3,5,3)`, `state_n_stds=(0.6,1.75,2.0)`, `n_state_quad_nodes=2` |
| Large retirement-only partial | `saved_runs/unconstrained_principal_grid9x9x9_nz9_retirement_only` | `stopped_early` | `67-99` | `state_grid_sizes=(9,9,9)`, `n_ret_nodes_1d=(3,7,5)`, `state_n_stds=2.0`, `n_state_quad_nodes=3` |

### Completed partial bundles

| Label | Bundle | Snapshot status | Ages solved in latest checkpoint | Notes |
| --- | --- | --- | --- | --- |
| Small-grid age-65 partial baseline | `saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_v2` | `stopped_early` | `65-99` | Intended partial solve completed as designed. Current diagnostics: `total_calls=5,737,500`, `total_newton_failures=0`, `total_mono_violations=0`, `worst_foc_resid~1.0e-7` |
| Small-grid return-quadrature refinement | `saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2` | `stopped_early` | `65-99` | Same solve window with refined return quadrature `n_ret_nodes_1d=(3,7,5)`. Diagnostics: `total_calls=5,737,500`, `total_newton_failures=0`, `total_mono_violations=0`, `worst_foc_resid~1.0e-7` |
| Small-grid high-accuracy return reference | `saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_kret5x9x5_v2` | `stopped_early` | `65-99` | Same solve window with high-accuracy return quadrature `n_ret_nodes_1d=(5,9,5)`. Diagnostics: `total_calls=5,737,500`, `total_newton_failures=0`, `total_mono_violations=0`, `worst_foc_resid~1.0e-7` |
| Medium-grid refinement | `saved_runs/checkpoints/unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_v2` | `stopped_early` | `65-99` | Grid refinement run completed as intended. Diagnostics: `total_calls=15,743,700`, `total_newton_failures=0`, `total_mono_violations=0`, `worst_foc_resid~1.0e-7` |
| Wide-support retirement check | `saved_runs/checkpoints/unconstrained_principal_grid9x9x9_nz9_from_age70_kret3x7x5_ns2p0x2p25x2p25_v2` | `stopped_early` | `70-99` | Retirement-only large-grid support test with `state_n_stds=(2.0,2.25,2.25)` and `n_state_quad_nodes=(2,2,5)`. Diagnostics: `total_calls=28,540,350`, `total_newton_failures=0`, `total_mono_violations=0`, `worst_foc_resid~1.0e-7` |
| Log1p wealth/savings-spacing test | `saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1` | `stopped_early` | `65-99` | Same `7x7x7` / wide-support age-65 window on the new restretched `log1p` wealth and savings grids. Metadata still shows `n_wealth=150`, `n_savings=150`, so the change is spacing rather than point count. Diagnostics: `total_calls=15,743,700`, `total_newton_failures=0`, `total_mono_violations=0`, `worst_foc_resid~1.0e-7` |

### Next queued run

| Label | Planned bundle pattern | Status | Notes |
| --- | --- | --- | --- |
| 9x9x9 grid refinement | `saved_runs/checkpoints/unconstrained_principal_grid9x9x9_nz9_from_age65_kret3x7x5_v2` | `queued` | Next large-grid confirmation with quadrature frozen at `n_ret_nodes_1d=(3,7,5)` and `n_state_quad_nodes=(2,2,5)` |

### Bundles to treat with caution

| Bundle | Reason |
| --- | --- |
| `saved_runs/unconstrained_principal_grid5x5x5_nz9_v2` | Incomplete save: arrays exist, but no `metadata.json` and `diagnostics.pkl` is empty |
| Any comparison between `unconstrained_principal_grid5x5x5_nz9` and `unconstrained_principal_grid9x9x9_nz9_retirement_only` | Not a clean one-axis convergence comparison because `state_grid_sizes`, `n_ret_nodes_1d`, `state_n_stds`, and `n_state_quad_nodes` all differ |

## Run Ledger

### 2026-04-30

- Saved complete bundle `saved_runs/unconstrained_principal_grid5x5x5_nz9`
  with full lifecycle ages `22-99`.
- Saved large partial bundle
  `saved_runs/unconstrained_principal_grid9x9x9_nz9_retirement_only`
  covering ages `67-99`.
- Created incomplete bundle `saved_runs/unconstrained_principal_grid5x5x5_nz9_v2`
  during an interrupted save.

### 2026-05-01

- Notebook partial-solve cell changed from retirement-only to ages `65-99`.
- Checkpoint bundle now writes to
  `saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_v2`.
- First sweep run finished for the intended partial window `65-99`.
- Final observed metadata for the baseline partial bundle:
  - `solve_status="stopped_early"` on the partial solver path
  - solved ages `65-99` exactly
  - `total_calls=5,737,500`
  - `total_newton_failures=0`
  - `total_mono_violations=0`
  - `worst_foc_resid~1.0e-7`
- Second sweep run also finished for the intended partial window `65-99`.
- Final observed metadata for the `kret3x7x5` partial bundle:
  - `solve_status="stopped_early"` on the partial solver path
  - solved ages `65-99` exactly
  - `total_calls=5,737,500`
  - `total_newton_failures=0`
  - `total_mono_violations=0`
  - `worst_foc_resid~1.0e-7`
- Third sweep run also finished for the intended partial window `65-99`.
- Final observed metadata for the `kret5x9x5` partial bundle:
  - `solve_status="stopped_early"` on the partial solver path
  - solved ages `65-99` exactly
  - `total_calls=5,737,500`
  - `total_newton_failures=0`
  - `total_mono_violations=0`
  - `worst_foc_resid~1.0e-7`
- Ran `scripts.diagnostics._diag_policy_convergence` with
  `kret5x9x5` as the reference bundle and wrote:
  `diagnostics_kret_convergence_report.md`
- First comparison takeaways:
  - all three bundles pass the solver-health gates
  - `kret3x7x5` is extremely close to `kret5x9x5` on median and 95th-percentile
    common-probe drift
  - the coarse baseline `kret3x5x3` is also close in consumption, but has
    noticeably larger working-age portfolio-share outliers relative to
    `kret5x9x5`
  - isolated working-age probe outliers remain even for `kret3x7x5` versus
    `kret5x9x5`, so the quadrature-freeze decision should look at outliers, not
    only medians
- Sweep decision: move on to the state-grid ladder using `kret3x7x5` as the
  frozen return quadrature for the next `7x7x7` and `9x9x9` partial solves.
- Added `scripts/diagnostics/_diag_euler_errors.py`, a simulation-path
  Euler-error diagnostic that:
  - simulates a saved bundle over its solved age window
  - evaluates the Euler equation with an independent evaluation quadrature
  - reports age-by-age `log10|EE|` summaries and publication / welfare gates
- Smoke-tested the Euler diagnostic on
  `saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2`
  and wrote `diagnostics_euler_smoke.md`.
- Important interpretation note from the smoke test:
  - the Euler evaluator matches the solver's own working-age and retirement
    FOC kernels at on-grid states to machine precision
  - but path-level smoke results were dominated by simulation setup issues,
    especially large off-grid wealth mass in the warm-start full-bundle
    simulation
  - so the current Euler script is ready for use, but paper-quality Euler
    tables still need a disciplined simulation design for the age-65 initial
    cross-section and likely a check on wealth-grid tightness
- Ran `scripts.diagnostics._diag_policy_convergence` on the completed
  `7x7x7` bundle using the frozen `5x5x5, kret3x7x5` bundle as the
  comparison and wrote:
  `diagnostics_grid7x7x7_convergence_report.md`
- Main grid-comparison takeaways from `5x5x5 -> 7x7x7`:
  - both bundles pass all solver-health gates cleanly
  - common-probe consumption drift is modest in the center but still material
    in the tails: `C rel %` p95 about `4.99%`, max about `5.49%`
  - common-probe portfolio-share drift remains noticeable:
    `stock share` p95 about `2.34 pp`, `bond share` p95 about `3.71 pp`
  - differences are somewhat larger at the working ages than in retirement
- Ran centered age-65 Euler diagnostics with representative initial
  cash-on-hand `x65 ~ 7.50` (SCF-median wealth schedule plus age-65 income)
  and wrote:
  - `diagnostics_euler_7x7x7_centered_report.md`
  - `diagnostics_euler_5x5x5_centered_report.md`
- Centered Euler comparison takeaways:
  - `7x7x7` improves the working-age mean `log10|EE|` relative to `5x5x5`
    (`-2.94` vs `-2.77`)
  - retirement also improves modestly (`-2.17` vs `-1.95`)
  - neither bundle is close to the publication-grade Euler gates under this
    centered partial-window check, so these numbers should be treated as
    supporting local diagnostics rather than final paper tables
- Ran the strict grid-point parity check for the Euler diagnostic on the
  `7x7x7` bundle with `eval_mode="same"` and exact policy-grid states.
- Parity-check result:
  - tested `180` exact grid points spanning working and retirement ages
  - max relative difference in `e_sum` versus the solver kernels:
    `2.24e-15`
  - median relative difference: `3.90e-16`
  - max implied-consumption difference from `e_sum`: `7.11e-15`
  - interpretation: the Euler diagnostic matches the solver's FOC evaluation
    to machine precision, so the EE estimator itself is confirmed correct
- Ran the gridded-agents exact-grid EE sweep (no simulation) on the `7x7x7`
  bundle and wrote:
  `diagnostics_gridpoint_ee_7x7x7_same.md`
- Exact-grid EE sweep takeaways:
  - mean `log10|EE| = -8.76`
  - p95 `log10|EE| = -4.18`
  - max `log10|EE| = -2.75`
  - retirement exact-grid EE is much cleaner than working-age exact-grid EE;
    the worst points are concentrated at age `65`, high `z`, wealth index
    `134` (`x ~ 46.42`), and extreme financial-state corners
  - those worst points still have age-level `worst_foc_resid ~ 1e-7` and
    zero Newton failures, so this does not look like a diagnostic bug or a
    Newton-solver failure
- Cross-check against the `5x5x5, kret3x7x5` exact-grid EE sweep shows almost
  identical worst-case behavior (`max log10|EE|` about `-2.75` in both cases),
  which suggests this localized EE tail is not primarily driven by
  `state_grid_sizes`; it is more likely tied to the wealth-grid / EGM
  interpolation layer or the still-frozen quadrature choice.
- Ran simulated-state EE stratification on the centered `7x7x7` run and wrote:
  `diagnostics_sim_ee_stratification_7x7x7.md`
- Simulated-state stratification takeaways:
  - overall mean `log10|EE| = -2.31`
  - working mean `log10|EE| = -2.97`
  - retirement mean `log10|EE| = -2.24`
  - wealth-fraction within a cell does not explain the problem; EE is weak
    across the whole within-cell wealth range
  - the centered partial run does not identify a useful `z` margin: almost all
    mass remains in the middle `z` bins
  - the strongest finding is state-grid support failure, not generic
    interpolation depth:
    - `45.4%` of evaluated agent-years lie outside the state-grid support in
      transformed coordinates
    - this is only `3.9%` in working ages but `49.6%` in retirement
    - by ages `95-98`, about `59-62%` of alive households are outside
    - boundary breaches are dominated by transformed state axis 0 with support
      `[-0.6, 0.6]` (outside share about `40.0%` overall, `43.8%` in retirement)
- Interpretation: the current simulated EE tail is being driven importantly by
  state-grid boundary clipping / support width on axis 0, not just by the
  generic fact that policies are evaluated off-grid.
- Ran the wide-support `9x9x9` retirement-only partial solve on ages `70-99`:
  `saved_runs/checkpoints/unconstrained_principal_grid9x9x9_nz9_from_age70_kret3x7x5_ns2p0x2p25x2p25_v2`
- Solver-health outcome for the wide-support run:
  - `solve_status="stopped_early"` as intended for the partial window
  - solved ages `70-99`
  - `total_calls=28,540,350`
  - `total_newton_failures=0`
  - `total_mono_violations=0`
  - `worst_foc_resid~1.0e-7`
- Added `scripts/diagnostics/_diag_bundle_state_clipping.py`, a bundle-level
  simulation diagnostic that:
  - simulates a saved bundle over its solved age window
  - transforms simulated continuous states into principal-grid bracket coords
  - reports joint and per-axis outside-support shares by age and phase
- Ran the clipping diagnostic on the new wide-support bundle and on the older
  narrow-support `9x9x9` retirement bundle, writing:
  - `diagnostics_state_clipping_9x9x9_age70_wide.md`
  - `diagnostics_state_clipping_9x9x9_age67_narrow.md`
- State-clipping comparison takeaways:
  - wide-support bundle: joint outside share `5.37%`
  - narrow-support `9x9x9` bundle: joint outside share `9.37%`
  - the widened support especially cuts axis-1 and axis-2 breaches:
    `u1: 3.79% -> 1.91%`, `u2: 3.52% -> 1.56%`
  - late-retirement clipping is still present, but materially smaller than
    under the older narrow-support configuration
- Ran centered retirement Euler diagnostics on the new wide-support bundle and
  the older narrow-support `9x9x9` retirement bundle, writing:
  - `diagnostics_euler_9x9x9_age70_wide_report.md`
  - `diagnostics_euler_9x9x9_age67_narrow_report.md`
- Retirement Euler comparison takeaways:
  - wide-support bundle: mean `log10|EE| = -2.59`, max `-0.84`
  - narrow-support `9x9x9` bundle: mean `log10|EE| = -1.57`, max `-0.10`
  - this is a meaningful improvement in the local retirement Euler diagnostic,
    even though the bundle still remains short of publication-grade gates

### 2026-05-02

- Completed the new `7x7x7` / wide-support / age-65 partial solve on the
  restretched `log1p` wealth and savings grids:
  `saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age65_kret3x7x5_ns2p0x2p25x2p25_log1p_v1`
- Bundle health:
  - `solve_status="stopped_early"` as intended for the partial window
  - solved ages `65-99`
  - `total_calls=15,743,700`
  - `total_newton_failures=0`
  - `total_mono_violations=0`
  - `worst_foc_resid~1.0e-7`
- Ran the centered simulation-path Euler diagnostic and wrote:
  `diagnostics_euler_log1p_centered_report.md`
- Centered Euler takeaways:
  - working mean `log10|EE| = -3.014`
  - retirement mean `log10|EE| = -2.467`
  - both phases improve relative to the earlier narrow-support `7x7x7`
    centered report, with the biggest gain in retirement
- Ran the state-clipping diagnostic and wrote:
  `diagnostics_state_clipping_log1p_7x7x7.md`
- State-clipping takeaways:
  - overall joint outside share `6.00%`
  - working outside share `0.20%`
  - retirement outside share `6.59%`
  - compared with the earlier centered narrow-support `7x7x7`, this is a very
    large reduction in support failure, especially in retirement
- Ran the exact-grid Euler sweep and wrote:
  `diagnostics_gridpoint_ee_log1p_same.md`
- Exact-grid takeaways:
  - overall mean `log10|EE| = -5.76`
  - p95 `-0.54`
  - max `0.30`
  - however, the entire bad tail is concentrated at the very first wealth node
    `iw=0`, `x=1e-4`
- Ran focused low-wealth follow-up diagnostics and wrote:
  - `diagnostics_log1p_low_wealth_investigation.md`
  - `diagnostics_log1p_low_wealth_sweep.md`
- Low-wealth follow-up takeaways:
  - `iw=0` is the problem node: mean `log10|EE| = -0.875`, max `0.300`
  - by `iw=1` the node is already much better: mean `-2.268`
  - by `iw=2` the node is back to roughly publication-supporting exact-grid
    quality: mean `-4.070`
  - excluding only `iw=0`, the default grid-point sweep improves to:
    mean `-6.986`, p95 `-3.719`, max `-3.182`
- Main interpretation:
  - the new `log1p` spacing is a genuine win where households actually live
    and substantially reduces the off-grid / support problem
  - the remaining weakness is now a highly localized wealth-floor pathology,
    not a broad policy-surface failure
- Important diagnostics caveat discovered during this run:
  - after the `log1p` spacing change, old pre-`log1p` bundles no longer carry
    enough metadata to reconstruct their original wealth/savings grids exactly
  - diagnostics on the new `log1p` bundle are valid
  - but direct reconstructed old-vs-new policy comparisons are not reliable
    until the bundle metadata records the wealth/savings grid spacing mode

### 2026-05-03

Battery of 8 partial sweeps (`system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_*` and one `9x9x9` companion) plus the two full `7x7x7` baselines (`base`, `cap_only`). All terminated as `is_partial=True`. **All 8 sweeps unfortunately used the narrow `state_n_stds=(0.6, 1.75, 2.0)`** — same support as the runs flagged in the Open Questions block below — so absolute EE numbers are dominated by support clipping. The **relative** rankings across configs are still informative.

Sweep grid tested (one axis at a time, holding others fixed):

| label | grid | state-quad | ret-quad | η/ε | cap | youngest age solved |
| --- | --- | --- | --- | --- | --- | --- |
| `base` | 7×7×7 | (2,2,5) | (3,7,5) | 3/3 | ±10 | 22 |
| `cap_only` | 7×7×7 | (2,2,5) | (3,7,5) | 3/3 | ±5 | 22 |
| `state33` | 7×7×7 | (3,3,5) | (3,7,5) | 3/3 | ±10 | 38 |
| `state33_cap` | 7×7×7 | (3,3,5) | (3,7,5) | 3/3 | ±5 | 41 |
| `state44_cap` | 7×7×7 | (4,4,5) | (3,7,5) | 3/3 | ±5 | 53 |
| `inc55_cap` | 7×7×7 | (2,2,5) | (3,7,5) | 5/5 | ±5 | 46 |
| `mid_rich_cap` | 7×7×7 | (3,3,5) | (3,7,5) | 5/5 | ±5 | 58 |
| `tight_cap` | 7×7×7 | (3,3,5) | (3,7,5) | 3/3 | ±3 | 55 |
| `grid9_base` | 9×9×9 | (2,2,5) | (3,7,5) | 3/3 | ±10 | 36 |
| `grid9_state33_cap` | 9×9×9 | (3,3,5) | (3,7,5) | 3/3 | ±5 | 56 |

Full diagnostic battery (A/B/C) run on the priority subset; A and B run on all 8 partials. Per-bundle reports under `diagnostics_reports/diagnostics_gridpoint_<label>_{same,nextfiner}.md` and `diagnostics_simpath_<label>_nextfiner.md`. Cross-bundle comparison report: [`diagnostics_reports/diagnostics_partial_sweeps_comparison.md`](../../diagnostics_reports/diagnostics_partial_sweeps_comparison.md).

#### Headline findings

1. **State quadrature is the dominant solver lever.** Bumping (2,2,5)→(3,3,5) closes ~0.33 orders of Diagnostic-B mean and ~1.4 orders of sim-path retirement max. (3,3,5)→(4,4,5) gives diminishing returns (≈0.04 orders Diag B mean, but still ~1.1 orders on sim-path retirement max). All other levers are smaller in magnitude.

2. **The "extreme leverage tail" in `base` was a state-quad artifact, not real economics.** With (2,2,5) state-quad the policy wants α_s up to 5.5 and α_b up to 6.4 in retirement state corners. Refining to (3,3,5) cuts max α_s to 2.2; (4,4,5) cuts to 2.0. The integrand truncation under coarse state-quad made over-leveraged positions look locally optimal; richer sampling reveals they aren't. **Implication:** the leverage cap intervention was solving an artifact. Once state-quad is in, the cap=±5 stops binding meaningfully.

3. **Cap=±5 is essentially neutral.** sim-path EE between `base` and `cap_only` differs by 0.02 orders. Newton-failure counts in cap-bound runs are KKT slack at the cap, **not** convergence failures. Only `tight_cap` (±3) actually moves sim-path EE — by 1.5 orders on retirement mean — but does so by clipping a *real* part of the policy, with unknown welfare implications.

4. **Income quadrature does not enter retirement directly** (z is frozen, pension is deterministic — confirmed by bit-identical retirement policy arrays between η=3 and η=5 bundles). The apparent retirement-EE benefit of η=5 is **indirect**: better working-age policy → tamer simulated wealth growth → fewer agents end retirement at x values past `wealth_max=200`. Under `cap_only` (η=3) simulated x reaches 2.5M at age 67 and 88M at age 90; under `inc55_cap` (η=5) the same ages cap at 69k and 4.6M. **Working-age policy is calibrated to a slightly wrong income process under η=3, and the bias compounds across 45 years.**

5. **Wealth-grid extrapolation is a hidden source of retirement-EE blow-up.** `wealth_max=200` is being exceeded by simulated agents by orders of magnitude. The "retirement EE failure" in `base`-like bundles is partly the EE script computing residuals at extrapolated (x, c, α) values, not real policy quality. **Bumping `wealth_max` would do real work for sim-path EE in retirement, independent of any quadrature change.** This is a separate diagnostic axis worth a dedicated test.

6. **Grid refinement (9×9×9) is dominated by state-quad refinement.** `grid9_base` (9×9×9, (2,2,5) quad) gives sim-path retirement mean −0.84; `state33` (7×7×7, (3,3,5) quad) gives −1.18 — better by 0.34 orders. **Spending compute on more state grid nodes without refining the integrating quadrature is a near-no-op.**

7. **Income-quad on top of state-quad still helps** (the cleanest A/B test: `state33_cap` η=3 vs `mid_rich_cap` η=5, same retirement window): retirement mean −1.28 → −1.82 (+0.54 orders), retirement max +4.79 → +2.73 (+2.06 orders). Working-age (overlap window only) is more modest: +0.15 mean, +0.60 max. So η/ε=5 is a worthwhile second axis but its effect is **mediated by working-age wealth dynamics**, not by retirement-FOC accuracy.

8. **Tight cap (±3) and η=5 are partial substitutes.** Both clip the dangerous working-age policy tail by different mechanisms. `tight_cap` (state33+cap3, η=3) ≈ `mid_rich_cap` (state33+inc55+cap5) on sim-path retirement mean (−1.89 vs −1.82). **For policy interpretation it's safer to fix the artifact at the source (state-quad refinement + η=5) than to clamp it (cap=±3).**

9. **Same-Q EE is self-grading and badly misleading.** Diagnostic A reported mean ~−5.5 (looks publication-grade) for every bundle; Diagnostic B reported mean ~−2.0 (catastrophic) for the same bundles. **3.5 orders of self-grading bias** under `(2,2,5)` state quadrature. Always run next_finer; never trust same-Q. Refining state-quad to (3,3,5) closes the same-Q vs next_finer gap from 3.5 to 3.3 orders — much of the gap remains because narrow support distorts both.

10. **None of these bundles pass the publication gates** (best is `mid_rich_cap` at retirement max +2.73 vs gate −3.0; ~5.7 orders to close). The narrow `state_n_stds=(0.6, 1.75, 2.0)` is the elephant. Widening to `(2.0, 2.25, 2.25)` is the first required step before any of these refinements can be meaningfully evaluated against the gates.

11. **A large fraction of simulated agents are above the solver's `wealth_max=200` at retirement entry, and the fraction grows through retirement.** Off-grid share at age 67 / 80 / 90 / 99, with `wealth_max=200`:

    | bundle | start | %off-grid @67 | @80 | @90 | @99 |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | `base` | 22 | **30.5%** | 45.3% | 57.4% | **70.3%** |
    | `cap_only` | 22 | 30.0% | 44.8% | 56.9% | 70.3% |
    | `grid9_base` | 36 | 21.2% | 36.1% | 48.7% | 62.8% |
    | `state33` | 38 | 15.8% | 27.1% | 39.6% | 52.2% |
    | `inc55_cap` | 46 | 10.4% | 21.5% | 33.4% | 48.3% |

    Headline: in the full-lifecycle `base` bundle, **30% of agents enter retirement with x > wealth_max, and 70% exceed it by terminal age**. Even refined-quadrature bundles (which only have partial accumulation horizons) leave 50%+ off-grid at age 99. Once x > 200 the policy is iw=149 extrapolated linearly — economically meaningless. **A non-trivial fraction of every sim-path EE result is grading the extrapolation, not the solved policy.** The improvements we attribute to state-quad and η refinement partly reflect those bundles producing tamer working-age leverage and so smaller off-grid mass — the true policy quality on agents who stay on-grid is harder to pin down without a wealth-grid extension.

    This makes `wealth_max` extension an independent, high-priority axis for any subsequent solve. A solve with `wealth_max ∈ [10000, 50000]` (or log1p tail extension) would let us:
    - separate "policy quality" from "extrapolation noise" on sim-path EE,
    - measure `% off-grid` as a first-class bundle-health metric (call it `frac_above_wealth_max[t]`), and
    - reduce the runaway-feedback path where extrapolated policy at high x recommends more leverage which generates higher x next period.

#### Untested axes (open after this battery)

- **Per-axis state-quad sensitivity.** All bumps moved axes 0 and 1 in lockstep. We have not tested `(3,2,5)`, `(2,3,5)`, `(2,2,7)`, or `(3,3,7)`. The three principal-mode axes have different stationary spread (axis-0 needs `1.42` for p90, axis-1 needs `1.59`, axis-2 needs `1.56`), so they probably need different node counts. Three single-axis solves would tell you which axis is binding.
- **Wealth-grid coverage.** The `wealth_max=200` ceiling is too low for the simulated trajectory. A solve with `wealth_max=10000` (or a log1p tail extension) would isolate the extrapolation contribution to retirement-max EE.
- **Wide support × refined state-quad.** The natural next solve: 7×7×7, **`state_n_stds=(2.0, 2.25, 2.25)`**, `state_quad=(3,3,5)`, η/ε=5, cap=±5. This is the candidate for actually clearing publication gates.

## Planned Run Matrix

These are the next natural runs once the current `from_age65` baseline finishes.

| Priority | Solve window | State grid | Return quadrature | State quadrature | Purpose |
| --- | --- | --- | --- | --- | --- |
| 1 | `65-99` | `(5,5,5)` | `(3,5,3)` | `(2,2,5)` | Done: baseline age-65 partial |
| 2 | `65-99` | `(5,5,5)` | `(3,7,5)` | `(2,2,5)` | Done: return-quadrature refinement |
| 3 | `65-99` | `(5,5,5)` | `(5,9,5)` | `(2,2,5)` | Done: high-accuracy return reference |
| 4 | `65-99` | `(7,7,7)` | `(3,7,5)` | `(2,2,5)` | Done: medium-grid refinement |
| 5 | `70-99` | `(9,9,9)` | `(3,7,5)` | `(2,2,5)` | Done: wide-support retirement check with `state_n_stds=(2.0,2.25,2.25)` |
| 6 | `65-99` | `(9,9,9)` | `(3,7,5)` | `(2,2,5)` | Optional next: full large-grid confirmation if we still need working-age evidence |

## Tests To Run On Policy Bundles

Implemented tooling:

- `python -m scripts.diagnostics._diag_policy_convergence ...`
  Compares saved bundles on a common probe set, reports solver-health gates,
  and summarizes policy drift versus a reference bundle.
- Current saved report:
  `diagnostics_kret_convergence_report.md`
- `python -m scripts.diagnostics._diag_euler_errors ...`
  Simulates a bundle over its solved age window and reports simulation-path
  Euler errors with an independent evaluation rule.
- `python -m scripts.diagnostics._diag_bundle_state_clipping ...`
  Simulates a bundle over its solved age window and reports how often
  continuous financial states fall outside the principal-grid support.
- Current smoke-test report:
  `diagnostics_euler_smoke.md`
- Current centered age-65 Euler reports:
  - `diagnostics_euler_5x5x5_centered_report.md`
  - `diagnostics_euler_7x7x7_centered_report.md`
- Current wide-vs-narrow retirement support reports:
  - `diagnostics_state_clipping_9x9x9_age70_wide.md`
  - `diagnostics_state_clipping_9x9x9_age67_narrow.md`
  - `diagnostics_euler_9x9x9_age70_wide_report.md`
  - `diagnostics_euler_9x9x9_age67_narrow_report.md`

### 1. Bundle integrity

For every saved bundle:

- verify `metadata.json` exists
- verify `diagnostics.pkl` loads
- verify `policy_arrays.npz` loads
- verify `solve_status`, `is_partial`, and age coverage
- record whether the bundle is complete, checkpointed, or interrupted

### 2. Solver-health diagnostics

For every candidate comparison bundle, record:

- `total_newton_failures`
- `worst_foc_resid`
- `total_mono_violations`
- `max_newton_iter`
- `avg_newton_iter`
- `total_calls`

Bundles with solver pathologies should not be used as convergence references.

### 3. Common-probe policy comparison

When comparing two bundles, evaluate policies on a common probe set rather than
raw array indices.

Suggested probe ages:

- `65`, `66`, `67`, `75`, `85`, `95`

Suggested probe wealth points:

- low, middle, and high wealth regions
- include points near `wealth_min`
- include points where risky shares often move sharply

Suggested probe income states:

- several `z` values, including low / median / high

Suggested financial-state probes:

- median state
- several off-center states
- if needed, a fixed list of economic states shared across bundles

### 4. Policy metrics to compare

For each pair of bundles, compare:

- consumption `C`
- stock share `S`
- bond share `B`
- implied bill share `1 - S - B`
- consumption ratio `C / W` where meaningful

Recommended summary statistics:

- max absolute difference
- median absolute difference
- 95th percentile absolute difference

### 5. Convergence decision rules

Working rule of thumb:

- If `(3,7,5)` is very close to `(5,9,5)`, keep `(3,7,5)` or even `(3,5,3)`
  if both are materially similar.
- If `(7,7,7)` is close to `(9,9,9)` on the common probe set, prefer `(7,7,7)`
  unless the remaining differences matter in economically relevant regions.
- Focus extra attention on ages `65` and `66`, since those are the difficult
  working-age test points in this sweep.

## Open Questions

### Resolved or substantially answered as of 2026-05-03
- ~~State-quad refinement helps; magnitude unknown~~ → confirmed (3,3,5) gives ~0.33 orders Diag B mean and ~1.4 orders sim-path retirement max over (2,2,5). (4,4,5) gives diminishing returns at the second-order margin.
- ~~Does cap intervention help?~~ → cap=±5 is neutral; cap=±3 helps but distorts the policy. The "leverage tail" was largely a quadrature artifact, not real economics; refining state-quad is the principled fix.
- ~~Does income-quad refinement matter for retirement?~~ → no, not directly (retirement policies are bit-identical between η=3 and η=5). Indirect via working-age wealth dynamics — η=5 still helps sim-path retirement EE through this path.
- ~~Does grid refinement (9×9×9) help?~~ → dominated by state-quad refinement. 7×7×7 + (3,3,5) beats 9×9×9 + (2,2,5) on every metric.

### Still open
- The narrow `state_n_stds=(0.6, 1.75, 2.0)` is the unresolved blocker. Wide support `(2.0, 2.25, 2.25)` confirmed to materially help retirement Euler accuracy in earlier runs, but has not been tested in combination with refined state-quadrature `(3,3,5)`. **This is the priority next solve.**
- **Per-axis state-quad sensitivity**: (3,2,5) vs (2,3,5) vs (2,2,7) vs (3,3,7). Three single-axis bumps would identify which of the three principal-component axes is binding so we don't waste nodes on axes that don't need them.
- **Wealth-grid coverage**: `wealth_max=200` is breached by 30% of `base`-bundle agents at retirement entry and by 70% at terminal age (see 2026-05-03 finding 11). Once off-grid, the policy is iw=149 extrapolated, which feeds back into more leverage and runaway wealth growth. A solve with `wealth_max ∈ [10000, 50000]` (or log1p tail extension) would isolate the extrapolation contribution to retirement EE, and `frac_above_wealth_max` should become a first-class bundle-health metric reported by the solver and the diagnostics.
- The bundle metadata currently does not record the wealth/savings grid spacing rule (`legacy` vs `log1p`). After the restretched-grid change, this makes some historical reconstructed-grid comparisons ambiguous.
- Should the main comparison metric be raw policy differences, portfolio-share differences, or Euler-equation objects implied by the policies?
- Do we want a dedicated comparison script that loads two bundles and emits a markdown summary table for this tracker?

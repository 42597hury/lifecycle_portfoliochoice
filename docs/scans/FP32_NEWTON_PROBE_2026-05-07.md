# FP32 Newton Probe - 2026-05-07

Branch: `jax-rewrite`  
Scope: scratch investigation only. No production solver changes.  
Artifacts:
- `scripts/scratch/probe_fp32_newton.py`
- `docs/scans/fp32_newton_probe_results.json`
- `docs/scans/fp32_newton_probe_results_fullwindow.json`
- `docs/scans/fp32_newton_probe_results_fullwindow_foc64boundary.json`
- tiny EE bundles: `saved_runs/fp32_probe_tiny_baseline`, `saved_runs/fp32_probe_tiny_newton_f32`

## 1. Executive Verdict

**GO, narrowly, for gamma sweeps / calibration exploration if the option is clearly labelled approximate and gated by a same-config fp64 re-solve before conclusions.** The tiny CPU probe gives a repeatable **1.12-1.17x wall speedup** with median alpha drift around **2e-7**. The tail is not zero: the 62-99 smoke window has **p99 alpha-component drift 7.7e-5** and **max drift 6.8e-3**, concentrated in worst cells. That is acceptable for sweep triage, not for final thesis numbers.

**NO-GO for publication-grade runs from this evidence.** The fp32 arithmetic sits directly on the convergence tolerance (`tol=1e-7`) and on cancellation-heavy Jacobian/CCV-return terms. The downstream grid-EE probe shows fp32 Newton barely changes the already-coarse tiny-grid EE distribution, but that is not enough to certify tail cells on 6^4/7^4 production grids.

The best implementation shape, if pursued later, is a solver toggle that keeps the existing `gather_precision="f32"` path and optionally casts only FOC/Jacobian arithmetic to fp32. It should not touch EGM storage, policy storage, or final publication runners.

## 2. Code Sites

| Layer | Current code site | Precision today | Probe treatment |
|---|---|---|---|
| CCV log return and gradients | [`solver.py:694`](../../lifecycle/solver.py#L694) | fp64 | monkey-patched fp32 |
| Terminal FOC/Jacobian | [`solver.py:723`](../../lifecycle/solver.py#L723) | fp64 | monkey-patched fp32 |
| Retirement FOC/Jacobian | [`solver.py:898`](../../lifecycle/solver.py#L898) | fp64 except gather/interp optional fp32 | monkey-patched fp32 after interp |
| Working FOC/Jacobian | [`solver.py:1005`](../../lifecycle/solver.py#L1005) | fp64 except gather/interp optional fp32 | monkey-patched fp32 after interp |
| Newton determinant/step | [`solver.py:595`](../../lifecycle/solver.py#L595), [`solver.py:629`](../../lifecycle/solver.py#L629) | fp64 values | probed both fp32-valued and fp64-boundary variants |
| Backtracking | [`solver.py:532`](../../lifecycle/solver.py#L532) | fp64 values | inherits patched FOC values |
| EGM scan | [`solver.py:1126`](../../lifecycle/solver.py#L1126) | fp64 | unchanged |
| Lift to wealth grid | [`solver.py:1189`](../../lifecycle/solver.py#L1189) | fp64 | unchanged |
| Diagnostics counters | [`solver.py:2499`](../../lifecycle/solver.py#L2499), [`solver.py:2907`](../../lifecycle/solver.py#L2907) | placeholders in this branch | not reliable for true failure counts |

Important diagnostic caveat: `age_max_foc` and `age_newton_fail` are initialized but not populated in the current solver path. Therefore `total_newton_failures=0` and `worst_foc_resid=0.0` are not evidence of convergence in this probe. The useful exposed signal is the iter/backtrack histogram, and with `max_iter=30` it saturates by construction.

## 3. Performance Accounting

For the handoff's 6^4 benchmark-like cell:

`K_v = 3*3*3*5 = 135`, `K_r = 3*3 = 9`, `K_q = 1215`, `K_corners = 16`, `n_savings = 180`, `max_iter = 100`, `max_backtrack = 10`, so each savings point pays `1 + 100*11 = 1101` FOC calls under `fori_loop`.

| Layer | Estimated FLOPs per FOC | Per-cell scale | Memory traffic | Bound | fp32 speed expectation |
|---|---:|---:|---:|---|---|
| `_ccv_log_return_and_grad` | ~25 * K_q = 30k | ~6.0e9 FLOPs/cell/age | ~30 KB tensors reused | compute/SFU | up to 2x scalar, less with `exp` |
| Interp + CRRA + FOC/Jac terms | ~200-230 * K_q = 240k-280k | ~48-56e9 FLOPs/cell/age | retirement c-corners ~3.1 MB fp64, ~1.6 MB fp32 | compute-bound after gather | 1.2-1.8x plausible |
| Newton 2x2 determinant/step | ~50 per Newton iter | <1e6 FLOPs/cell/age | scalar | negligible wall share | little standalone gain |
| Backtracking control | 10 extra FOC calls per iter | dominates by multiplying FOC calls | scalar + repeated FOC | compute-bound through FOC | gain comes from patched FOC |
| EGM inverse and scan wrapper | ~O(n_savings) scalar ops | tiny vs FOC | small arrays | negligible | leave fp64 |
| Lift (`argsort` + 3 `interp`) | O(n_savings log n_savings + 3*n_w) | tiny vs Newton | small arrays | negligible | leave fp64 |

Conclusion: fp32 Newton does not win by saving DRAM. The gather/interp memory win has already shipped via `gather_precision="f32"`. The remaining possible win is scalar fp32 arithmetic inside the repeated FOC/Jacobian evaluations.

## 4. Accuracy Risk

| Layer | Failure mode | Probe evidence | Risk |
|---|---|---:|---|
| CCV log return | Cancellation in `+0.5 alpha*diag(Sigma) - 0.5 alpha*Sigma*alpha`; `R_p=exp(r_p)` then amplifies r_p noise | tail drift appears despite tiny median drift | MEDIUM |
| FOC reductions | `tol=1e-7` is at fp32 epsilon scale; reduction over 1215 production quad points can lose tail digits | p99 alpha drift grows from 5.5e-6 retirement-only to 7.7e-5 with working ages | HIGH |
| Jacobian determinant | `det = J_ss*J_bb - J_sb^2`; fp64 singular threshold is 1e-15, fp32 practical threshold is closer to 1e-7 relative | boundary-cast variant has similar accuracy, so FOC precision dominates this tiny probe | HIGH in theory |
| Backtracking comparisons | near-equal residual comparisons can choose different accepted steps | backtrack hist changes, but policy bulk stable | MEDIUM |
| EGM scan/lift | interpolation monotonicity/endpoints could move if cast | unchanged in probe | LOW if left fp64 |
| Stored policies/warm start | fp32 storage could accumulate age-to-age drift | unchanged in probe | HIGH if changed; do not change |

The current saved 5^4 baseline bundle does not contain useful determinant/final-residual counts: its metadata reports `total_newton_failures=0` and `worst_foc_resid=0.0`, matching the placeholder diagnostics path rather than measured residuals. A production implementation must first add real residual/failure instrumentation.

## 5. Empirical Results

All timing below is CPU/JAX, not H100. It is still useful as an A/B precision differential because the same kernels/configs are compared within each run.

Config for the main smoke window:

- `start_age=62`, `retire_age=67`, `terminal_age=99` -> 38 ages
- `state_grid_sizes=(2,3,2,3)`, `n_state_quad_nodes=(2,3,2,3)`
- `n_z=3`, `n_w=12`, `n_savings=12`, `n_ret_nodes_1d=(2,2)`
- `gather_precision="f32"`, `max_iter=30`, `max_backtrack_iter=10`

| Run | Baseline wall | fp32 wall | Speedup | Median alpha drift | p99 alpha drift | Max alpha drift | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retirement-only 67-99 | 86.44s | 73.94s | 1.169x | 1.81e-7 | 5.52e-6 | 1.65e-3 | 0 |
| Full smoke 62-99, fp32-valued Newton | 184.22s | 157.36s | 1.171x | 1.90e-7 | 7.66e-5 | 6.78e-3 | 0 |
| Full smoke 62-99, FOC fp32 cast back to fp64 before Newton step | 176.42s | 157.50s | 1.120x | 1.90e-7 | 6.70e-5 | 6.33e-3 | 0 |

Per-age wall:

| Window | Baseline median retire age | fp32 median retire age | Baseline median work age | fp32 median work age |
|---|---:|---:|---:|---:|
| Retirement-only | 2.60s | 2.30s | n/a | n/a |
| Full smoke | 2.70s | 2.30s | 18.70s | 15.50s |

Interpretation:

- The measured speedup is stable at about **1.17x** for the aggressive fp32-valued Newton variant.
- Casting fp32 FOC/Jacobian outputs back to fp64 before the Newton step still gives **1.12x** and nearly the same drift. That suggests the accuracy cost mainly enters through fp32 FOC/Jacobian arithmetic, not the 2x2 determinant arithmetic itself, at least on this tiny grid.
- The bulk policy surface is almost unchanged. The worst tail cell is not unchanged.

## 6. EE Downstream Check

Because full-window alpha drift had a non-trivial tail, I saved two tiny bundles and ran:

`python verify/ee_residuals.py saved_runs/fp32_probe_tiny_baseline --metric consumption_ee`  
`python verify/ee_residuals.py saved_runs/fp32_probe_tiny_newton_f32 --metric consumption_ee`

Both bundles fail the absolute publication EE gate because the probe grid is intentionally coarse and `max_iter=30` is a smoke setting. The relevant question is the delta between baseline and fp32 Newton:

| Metric | Baseline | fp32 Newton | Delta |
|---|---:|---:|---:|
| Consumption EE mean log10 abs | -3.8188 | -3.8128 | +0.0060 |
| Consumption EE p95 log10 abs | -1.1470 | -1.1470 | ~0 |
| Consumption EE p99 log10 abs | -0.7944 | -0.7944 | ~0 |
| Consumption EE max log10 abs | -0.3484 | -0.3484 | ~0 |
| Portfolio relative max | 2.55093e-2 | 2.55093e-2 | ~0 |
| Global NaN cells | 0 | 0 | 0 |

So on this tiny smoke, fp32 Newton does **not** measurably worsen the downstream EE distribution. That supports using it for sweep triage, but it does not prove publication safety because the baseline itself is far from publication-grade.

## 7. Production-Scale Wall Projection

Assumptions:

- Anchor from prior scan: 5^4 retirement baseline on GH200 = 273 s/age at `K_v=36`, `K_r=25`, `max_iter=100`.
- Effective fp64 H100 SXM5 throughput = 21 TFLOPS per GPU, using the existing project estimator.
- Multi-GPU scaling is assumed linear for projection only.
- fp32 Newton speed band: measured **1.12-1.17x** on CPU; optimistic GPU band **up to 1.5x** if H100 scalar fp32 throughput is the binding limit. I would budget on **1.15-1.30x** until a real H100 run says otherwise.

| Run size | fp64 estimate | fp32 Newton conservative | fp32 Newton optimistic | Cost baseline | Cost conservative |
|---|---:|---:|---:|---:|---:|
| 6^4 retirement-only, 2x H100, `K_v=135`, `K_r=9`, `max_iter=100` | ~1.15 h | ~1.00 h | ~0.77 h | ~$7.6 | ~$6.6 |
| 7^4 full lifecycle, 8x H100, `K_v=135`, `K_r=9`, `max_iter=100` | ~12.5 h | ~10.9 h | ~8.3 h | ~$329 | ~$287 |

At measured speedups, fp32 Newton is a useful but not transformational cost lever. It saves calibration-cycle time; it does not replace grid/quad/max-iter choices as the primary cost controls.

## 8. Recommendation

If the user wants this implemented later, do it as an explicit solver option, not as a default:

1. Add `newton_precision: "f64" | "f32"` or similar to `SolverConfig`.
2. In `"f32"` mode, cast CCV return inputs, CRRA/bequest arithmetic, FOC sums, and Jacobian terms to fp32 inside `terminal_foc_jac_ccv`, `retirement_foc_jac_ccv`, and `working_foc_jac_ccv`.
3. Keep EGM inversion, lift, policy storage, warm-start storage, and bundle output fp64.
4. Add real diagnostics for final FOC residual, exit code/failure counts, and determinant bands.
5. Validation gates before any sweep use:
   - tiny 62-99 smoke: no NaN/Inf, p99 alpha drift < 1e-4, max drift logged
   - 5^4 retirement: grid EE delta vs fp64 baseline
   - one H100 timing run to replace the CPU speedup band

For publication runs: keep `newton_precision="f64"` until a production-grid, production-quad, same-seed comparison shows EE tails and tail-cell policies survive.

## 9. Unknowns

- No H100/GPU run was performed. CPU speedups are only directional.
- Current solver diagnostics do not expose true Newton failure/residual counts.
- The tiny smoke uses `max_iter=30`, so iteration histograms saturate and residuals are not publication-grade.
- The determinant-risk cell count (`1e-15 < det < 1e-7`) could not be measured from existing diagnostics.
- TF32/tensor-core paths were not investigated; this probe is scalar fp32 vs fp64 only.

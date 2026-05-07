# Handoff: fp32 Newton — Performance vs Accuracy Probe

**Branch:** `jax-rewrite`
**Effort:** 1-1.5 days. Read-only investigation + small empirical test. **No production code changes.**
**Output:** `docs/scans/FP32_NEWTON_PROBE_2026-05-07.md` — written report with measured numbers and a verdict.

**Workflow:** investigate → measure → report → STOP and ask. If verdict is positive, the user dispatches a separate implementation handoff afterward.

---

## Goal

Quantify the **potential gain in wall-time** and **potential loss in economic accuracy** from running the inner Newton arithmetic in fp32 instead of fp64, while keeping the existing mixed-precision gather (`gather_precision="f32"`) plumbing intact. Build a defensible report that lets the user decide:

1. Whether fp32 Newton is suitable for **gamma sweeps and calibration runs** (where some accuracy can be sacrificed for cycle speed).
2. Whether it should ever be used in **publication-grade runs** (probably not, but the report should justify why).
3. Which specific layers (FOC arithmetic, Jacobian inversion, EGM scan, lift) move which fraction of wall vs how much they degrade economic outputs.

This is an **almost-theoretical** investigation — the user wants reasoned analysis with measured numbers, not opinion.

---

## Background

Current mixed-precision configuration (per `SolverConfig.gather_precision`):
- `c_corners` gather to fp32 (memory-bound op)
- Everything else (FOC, Jacobian, Newton step, line search, EGM, lift) in fp64

The "safe half" of mixed precision is shipping. The unsafe half — actually doing arithmetic at fp32 — is what we want to probe.

**Hardware context:** H100 SXM5 fp64 = 33.5 TFLOPS, fp32 = 67 TFLOPS. fp32 tensor cores (TF32) = 989 TFLOPS for matmul-pattern work. So the speedup ceiling depends on which ops dominate.

**Why this matters now:** post-calibration we'll be running gamma sweeps and config-exploration runs. These don't need publication-grade tail accuracy. If fp32 Newton gives 1.5-2× wall at the cost of degraded worst-cell behavior, that's a great trade for sweep workflows.

---

## Investigations (numbered)

### §1. Per-layer performance accounting (theoretical)

Compute the FLOPS distribution across the per-cell solve at our typical config (6⁴ × n_state_quad=(3,3,3,5) × n_ret=(3,3) × max_iter=100 × n_savings=180):

| Layer | FLOPS estimate per cell | Compute-bound or memory-bound? | fp32→ratio |
|---|---|---|---|
| `_ccv_log_return_and_grad` | (formula) | ? | ? |
| FOC sums (`jnp.sum(wmu * dRp_das)` etc.) | ? | ? | ? |
| 2×2 Jacobian inversion + Newton step | ? | ? | ? |
| `_backtracking_fori` (per Newton iter) | ? | ? | ? |
| `_egm_scan_cell` (per savings) | ? | ? | ? |
| `_lift_to_wealth_grid` (jnp.interp × 3) | ? | ? | ? |

For each layer, estimate:
- FLOPS per cell per call
- Memory traffic per cell per call
- arithmetic intensity (FLOPS / bytes)
- Whether dropping to fp32 actually doubles speed (compute-bound) or only saves memory (bandwidth-bound)

The hand-derived FOC + Jacobian arithmetic is closed-form — count ops directly. The reductions are bandwidth-bound for small per-cell tensor sizes.

**Deliverable §1:** filled-in version of the table above with citations to file:line, plus a per-layer % of total wall (estimated, then sanity-checked against actual run wall).

### §2. Per-layer accuracy risk (theoretical)

For each layer, identify the precision-sensitive operation and the failure mode at fp32:

- **`_ccv_log_return_and_grad`**: `r_p` is a quadratic form in `(α_s, α_b)`. fp32 cancellation in the variance-correction term `0.5(α·σ² - α²σ²)` for small alpha. Quantify: what's the smallest |dr/dα| we expect to evaluate, and does fp32 ULP swamp it?
- **2×2 Jacobian inversion**: `det = J_ss * J_bb - J_sb²`. fp64 singular fallback at `det < 1e-15`. fp32 fallback would need `det < 1e-7`. How many cells in the canonical 5⁴ baseline had `1e-15 < det < 1e-7`? (Inspect the bundle's diagnostics or rerun with logging.)
- **Newton tolerance**: `tol=1e-7`, fp32 epsilon ~1.2e-7. Cells where the residual naturally floors near machine precision can never reach `tol` in fp32. How many such cells exist in the 5⁴ baseline? Check by running a fp64 solve and counting cells where final residual < 1e-6 but ≥ 1e-7.
- **EGM scan + lift**: `jnp.interp` is bandwidth-bound. The argsort step (`solver.py:1191`) is robust to fp32 (sort just compares). The interp itself accumulates error proportional to grid spacing; for n_savings=180, grid spacing in EGM-implied wealth is ~ε × wealth_max. fp32 might produce non-monotonic interpolation at extreme grid endpoints.

**Deliverable §2:** a per-layer table with theoretical failure-mode descriptions, cell-count estimates from existing diagnostics, and a hand-graded risk score (LOW / MEDIUM / HIGH) per layer.

### §3. Empirical small-scale test

Write a one-off comparison script in `scripts/scratch/probe_fp32_newton.py`. **Do not commit to the production verify/ folder.**

Goal: solve the same tiny lifecycle window at fp64-Newton (current default) vs a hand-patched fp32-Newton variant. Compare alphas, EE residuals, and Newton failures.

**Implementation approach (read-only friendly):**
1. Take the canonical retirement-only solve at the smallest config that exercises the boundary (e.g. matching `verify/smoke.py`: state_grid=(2,3,2,3), n_z=3, n_w=12, etc., 38 ages).
2. Run baseline at default fp64 Newton → capture (C, S, B, diag).
3. Make a **temporary patch** to the solver: cast `alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b, sigma2_*` to jnp.float32 inside `_ccv_log_return_and_grad` and `terminal_foc_jac_ccv` / `retirement_foc_jac_ccv`. Cast the FOC outputs back to fp64 at the boundary so the rest of the pipeline (EGM, lift) stays unchanged. **Do this on a scratch branch or via monkey-patching, not in main solver.py.**
4. Run the same config with the fp32-Newton patch → capture (C, S, B, diag).
5. Compare:
   - `np.max(np.abs(S_fp64 - S_fp32))`, `np.max(np.abs(B_fp64 - B_fp32))` (worst-cell drift)
   - `np.median(np.abs(S_fp64 - S_fp32))` (typical drift)
   - `diag['total_newton_failures']` for each
   - p99 of the residual at convergence (does fp32 floor higher?)
   - Wall time delta — first-age and steady-state ages separately

**Constraint:** keep the patch contained — do NOT modify production solver.py. Use `monkey-patch + rerun` or maintain on a throwaway branch that's never pushed. The point is to MEASURE, not to leave fp32 Newton in the codebase.

**Deliverable §3:** a table of measured numbers — alpha drift (max, median, p99), Newton failure delta, wall delta, conditions where fp32 broke (if any).

### §4. Sim-EE / Grid-EE downstream impact (optional but valuable)

If §3 shows non-trivial alpha drift, run the existing EE diagnostic on both bundles:
- `verify/ee_residuals.py` (grid-EE, fast on tiny config)
- `verify/ee_simpath.py` (sim-EE, the headline thesis number) at small `--n-simulations` to keep wall down

Compare:
- **Grid-EE**: max log10|EE|, p95, p99, median for fp64 vs fp32 bundles
- **Sim-EE**: same statistics
- The user's hypothesis is sim-EE is unaffected (tail cells get little weight). Test it.

**Deliverable §4:** if pursued, a verdict on whether fp32 Newton degrades the publication-grade EE numbers, separated by metric.

### §5. Wall projection at production scale (theoretical)

From §1's per-layer FLOPS estimates and §3's measured small-scale wall delta, project the wall reduction at:
- 6⁴ retirement-only on 2× H100 (tonight's run scale) → estimated cycle-1 wall in fp32 vs fp64
- 7⁴ canonical full-solve on 8× H100 → projected fp32 vs fp64 wall

Be explicit about the bands. The 5⁴ baseline measured 6.8 TFLOPS effective (~20% of fp64 peak). If we're memory-bound, fp32 Newton might give only 1.1-1.2× wall. If compute-bound, 1.5-1.8×.

**Deliverable §5:** projected wall + cost in $ for both run sizes, fp64 vs fp32, with bands.

---

## Output report structure

`docs/scans/FP32_NEWTON_PROBE_2026-05-07.md` — written for the user + future orchestrator. Sections:

1. **Executive verdict** (1-2 paragraphs): GO for gamma sweeps / NO-GO for publication / GO with caveats.
2. **§1 results**: per-layer performance accounting table.
3. **§2 results**: per-layer accuracy risk table.
4. **§3 results**: empirical small-scale measurements.
5. **§4 results** (if pursued): EE downstream impact.
6. **§5 results**: wall + cost projections at production scale.
7. **Recommendation**: which layers to fp32 (FOC only? FOC + Jacobian? all?) and the implementation scope estimate (LOC, file count, validation gates needed).
8. **Risks / unknowns**: things the probe didn't test, things to verify on real GPU before any production use.

---

## Pause point

After §3 (and §4 if pursued): **stop, write the report, send to user.** Do NOT scope or implement the production fp32 Newton change without explicit approval. The user wants the data first, decision second.

If §3 reveals an unexpected showstopper (fp32 produces NaN, all cells fail to converge, etc.), pause earlier and report.

---

## Out of scope

- **Implementing fp32 Newton in production solver.py.** The probe is investigative; the implementation is a separate handoff dispatched only if the user approves.
- **bf16 or fp16 anywhere.** Investigation is fp32 vs fp64. bf16/fp16 mantissa is too short for FOC interp accuracy.
- **Modifying the gather precision plumbing.** That's already in place and audit-clean.
- **Multi-GPU testing.** Single-device CPU smoke is enough to measure precision differentials. GPU-specific effects can be tested in the production-implementation handoff later.
- **Tensor core / TF32 investigation.** Adjacent question; flag as a follow-up but don't pursue here. (TF32 has fp32 mantissa with fp64 exponent — interesting middle ground, but separate investigation.)
- **Architectural alternatives** (Smolyak-style sparse grids, etc.). Different optimization axis.

---

## Why this matters

- **Gamma sweeps / config exploration**: 2-3× wall reduction × 5+ runs = real money saved in calibration cycles. Even 1.3× helps.
- **7⁴ canonical full-solve cost**: $300-500 per cycle currently. If fp32 saves 1.5×, that's $100-200 per cycle — but only if the publication-grade tail accuracy survives. The probe answers this directly.
- **Strategic clarity**: gives the user a measured tradeoff curve to point to when justifying config choices in the thesis methodology section.

The probe costs ~1 day of agent time. The downstream payoff is decision quality, not direct compute savings.

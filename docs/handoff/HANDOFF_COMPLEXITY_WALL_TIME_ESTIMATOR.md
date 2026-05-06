# Handoff: Complexity Analysis + Wall-Time Estimator

**Branch:** `jax-rewrite`
**Mode:** **REPORT ONLY.** No code edits. Produce one markdown report with a calibrated estimator.

**Output target:** `docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md`. The user will use this to make GO/NO-GO calls on grid sizes, hardware choices, and time budgets.

**Time budget:** 3-5 hours. The estimator is more useful than perfect, but it must reproduce the empirical anchor (§3) within ±30%.

---

## 1. Goal

Produce **a parameterized wall-time formula** for `verify_benchmark_bundle.py`-style runs of the lifecycle solver, plus a **table of projected walls** across the configurations the user is considering for thesis runs. The formula must:

1. Take config knobs as inputs (grid sizes, n_z, n_w, n_s, quadrature counts, max_iter, n_age).
2. Take hardware as input (fp64 TFLOPS, HBM bandwidth, HBM size).
3. Account for known bottlenecks (§6) beyond naive FLOPs counting.
4. Cross-validate against the empirical anchor (§3) within ±30%.
5. Project walls for the configurations in §7.

This is an **engineering estimate**, not a theoretical bound. The goal is "tell the user whether 7⁴ is a 10-hour or 100-hour run, with confidence interval, before they spend the money."

---

## 2. Scope and non-goals

### In scope

- `lifecycle/solver.py` — primary read.
- `lifecycle/precompute.py` — for understanding precompute / one-time costs.
- The four FOC functions (`terminal_foc_jac_ccv`, `retirement_foc_jac_ccv`, `working_foc_jac_ccv`, `_ccv_log_return_and_grad`) and the Newton+backtracking helpers (`_newton_fori`, `_backtracking_fori`).
- The cell-batching pattern (vmap-only on single-GPU, pmap on multi-GPU).
- The empirical anchor data in §3.

### Out of scope

- **Numba reference comparisons.** That's a separate analysis.
- **Memory analysis.** A separate `HANDOFF_MEMORY_ANALYSIS.md` would cover OOM risk; this handoff is purely time.
- **Algorithmic redesign.** Don't propose changes to Newton/EGM/quadrature.
- **GPU profiler runs.** Static read of code + the empirical anchor only.
- **Multi-GPU sharding modeling.** Single-device vmap path only.

### Hard constraints

- **No code edits.** Findings go in the report only.
- Don't run the solver. Static read + arithmetic only.

---

## 3. Empirical anchor — calibrate against this

A 5×5×5×5 retirement-only run on Lambda Labs **GH200 (H200 GPU die, 97 GB HBM3, fp64 ~9.7 TFLOPS effective)** is in flight at the time of writing. Already-observed:

| Config knob | Value |
|---|---|
| `state_grid_sizes` | `(5, 5, 5, 5)` → N_state = 625 |
| `n_z` | 11 |
| `n_w` | 180 |
| `n_s` | 180 |
| `n_state_quad_nodes` | `(2, 3, 2, 3)` → 36 nodes |
| `n_ret_nodes_1d` | `(5, 5)` → 25 ret quad nodes |
| `n_corners` | `2^4 = 16` (multilinear over 4-D state) |
| `max_iter` | 100 (`use_fori_newton=True` so this is wall cost, not just cap) |
| `max_iter_unconstrained` | 100 |
| `youngest_age_to_solve` | 67 → 33 retirement ages |
| Cell-batching | `vmap-only (single-device)` |
| `use_backward_age_warm_start` | `True` |

**Observed per-age wall times:**
- Age 99 (terminal kernel): solved in <1s after compile.
- Age 98 (first retirement age): **277.5s** (dominated by JIT compile cost for the retirement kernel).
- Age 97 (second retirement age): **551.0s elapsed** since terminal → ~273s for age 97 itself.

**Empirical per-age "compiled" wall: ~273 s/age** at this config. **Empirical first-age (with JIT cost): ~277 s.** So the JIT compile cost is small (~5s) — actual compute dominates.

**GPU instrumentation snapshot mid-run:** `100% utilisation, 92.9 / 97.9 GB HBM used`.

This anchor is the calibration point. Your estimator must reproduce **273 ± 80 s/age at this exact config on this exact hardware.** If it predicts 50s or 1000s, your formula is wrong.

---

## 4. Compute drivers — read these specific functions

Walk through these files in this order. For each, identify the **shape of work it does per call**.

### 4.1 The Newton inner loop — `_newton_fori` ([solver.py:523](lifecycle/solver.py#L523))

- Runs `lax.fori_loop(0, max_iter, ...)` — **always max_iter iterations** when `use_fori_newton=True`. Mask-based early termination.
- Each iteration calls:
  - `foc_fn` once (full FOC eval at proposed `(α_s, α_b)`)
  - `_backtracking_fori` once (which itself runs `max_backtrack_iter ≤ 10` extra `foc_fn` calls)
- **Per Newton iter cost: ~1 + max_backtrack_iter ≈ 11 `foc_fn` calls** (worst case if backtracking exhausts; typically 1-2)

### 4.2 The FOC evaluation — `retirement_foc_jac_ccv` and `working_foc_jac_ccv`

For retirement at a single cell with single savings point:
- Build `(n_state_quad, n_ret_quad)` log returns and gradients via `_ccv_log_return_and_grad` ([solver.py:431](lifecycle/solver.py#L431)).
- Bequest mu/mup at sR_p — `O(n_state_quad × n_ret_quad)` ops.
- **Multilinear-state interp at `n_state_quad × n_ret_quad` quad points** via `_interp_c_and_mpc_at_cell` ([solver.py:565](lifecycle/solver.py#L565)). Each interp does:
  - `searchsorted` on `wealth_grid` (n_w points)
  - Read `n_corners` corner values per quad point
  - Reduce: `Σ over corners` of `w_corner × c_corner`
  - **Per quad point: ~`n_corners × 4` FLOPs (multiply + add per corner) × 2 (c and slope)**
- Final reduction over `(n_state_quad × n_ret_quad)`: weighted sum.

**Per `foc_fn` call FLOP estimate (read the code for exact ops):** roughly `n_state_quad × n_ret_quad × (n_corners × C_interp + C_reduce)` where `C_interp ≈ 8-12 FLOPs/corner` and `C_reduce ≈ 30-50 FLOPs/quad-point`.

For working: same shape, but the c_corners is `(n_state_quad, n_z, n_corners, n_w)` — bigger gather, plus the eta/eps quadrature inner loop adds an extra `n_eta × n_eps ≈ 16` factor on top. **~16× more work per FOC eval than retirement.**

### 4.3 The EGM scan — `_egm_scan_cell` ([solver.py:~1090](lifecycle/solver.py#L1090))

- `vmap` over `n_savings = n_s` savings points.
- Each savings point: one full Newton solve (§4.1).
- **Per cell, per age: `n_s × max_iter × ~11 × foc_FLOPs` per cell.**

### 4.4 The cell vmap — `_build_per_age_*_kernel_vmap_only`

- vmap over `n_cells = n_z × N_state` (working/retirement) or `N_state` (terminal — no z dim).
- Each cell: one EGM scan (§4.3).

### 4.5 Total per-age cost (you compose these)

For retirement age:
```
W_age_retire ≈ n_z × N_state × n_s × max_iter × (1 + bt_avg) × foc_FLOPs_retire
```
For working age:
```
W_age_work ≈ n_z × N_state × n_s × max_iter × (1 + bt_avg) × foc_FLOPs_work
                                                                  ↑
                                                        ~16× retire's foc cost
```

Where `bt_avg` ≈ avg backtracking iters per Newton iter (typically 0.2-0.5 with backward-age warm-start).

### 4.6 Total over all ages

Retirement-only (`youngest_age_to_solve=67`): `n_retire_ages = terminal_age - 67 + 1 ≈ 33`.
Full lifecycle: `n_total = 80` ages, but split into retire (33) + work (46) + boundary (1).

```
W_total ≈ W_terminal + n_retire × W_age_retire + n_work × W_age_work
```

(Where `W_terminal` is small — special-cased, no continuation.)

---

## 5. Required formula structure

Your final estimator should look like:

```python
def estimate_wall(
    state_grid_sizes,           # tuple of ints, e.g. (5, 5, 5, 5)
    n_z, n_w, n_s,
    n_state_quad_nodes,         # tuple, e.g. (2, 3, 2, 3)
    n_ret_nodes_1d,             # tuple, e.g. (5, 5)
    max_iter,
    n_retire_ages, n_work_ages,
    fp64_tflops,                # e.g. 9.7 for H200/GH200
    hbm_bandwidth_TB_s,         # e.g. 3.0 for H200
    *,
    bt_avg=0.3,                 # avg backtracking iters per Newton iter
    foc_overhead=2.0,           # multiplier for non-FLOP costs (sched, gather, sync)
    jit_per_kernel_s=10.0,      # one-time per kernel
):
    n_state = len(state_grid_sizes)
    N_state = prod(state_grid_sizes)
    n_state_quad = prod(n_state_quad_nodes)
    n_ret_quad = prod(n_ret_nodes_1d)
    n_corners = 2 ** n_state

    foc_FLOPs_retire = n_state_quad * n_ret_quad * (n_corners * 12 + 40)
    foc_FLOPs_work = foc_FLOPs_retire * 16   # eta/eps inner quadrature

    per_age_retire = n_z * N_state * n_s * max_iter * (1 + bt_avg) * foc_FLOPs_retire * foc_overhead
    per_age_work = n_z * N_state * n_s * max_iter * (1 + bt_avg) * foc_FLOPs_work * foc_overhead

    W_retire = per_age_retire / (fp64_tflops * 1e12)
    W_work = per_age_work / (fp64_tflops * 1e12)

    return {
        "per_age_retire_s": W_retire,
        "per_age_work_s": W_work,
        "total_retire_only_s": jit_per_kernel_s * 2 + n_retire_ages * W_retire,
        "total_full_lifecycle_s": jit_per_kernel_s * 4 + n_retire_ages * W_retire + n_work_ages * W_work,
    }
```

**Calibrate the constants** (`foc_FLOPs` constants 12 and 40, `foc_overhead`, `bt_avg`) by fitting to the empirical anchor in §3. The free parameters give you 1-2 degrees of freedom to match the observed 273 s/age.

You don't need to be exact — but the calibrated formula must give 273 ± 80 s/age at the §3 config. **If your formula naturally predicts >500s or <100s at the anchor, something is structurally wrong.**

---

## 6. Bottlenecks to factor in (beyond raw FLOPs)

Pure FLOPs-over-throughput will overestimate hardware utilisation. The `foc_overhead` multiplier in §5 captures these — but you should explicitly enumerate them and assign rough impact factors so the user sees what's been accounted for.

### 6.1 Memory bandwidth (the dominant bottleneck for this solver)

The c_corners gather pattern reads `(n_state_quad × n_z × n_corners × n_w × 8 bytes)` per cell per FOC eval. At 5⁴/n_z=11/n_w=180/n_corners=16/quad=36: that's 9 MB read per cell per Newton iter per FOC eval. **At 100 max_iter × 6,875 cells = ~6 TB of memory traffic per age.** At H200's 3 TB/s bandwidth, that's a 2-second floor on memory traffic — well below the 273s observed, so memory bandwidth is **not** the binding constraint at current sizes.

**But:** at 9⁴ with full quad, the memory traffic per age is `~20× larger ≈ 120 TB/age`. At that point bandwidth ≈ 40s/age **floor**, comparable to the FLOPs estimate. **Estimator should report which is binding** (memory-bound vs compute-bound) per config.

### 6.2 JIT compile cost (small but mention)

- Per kernel JIT compile: ~5-30 seconds first call, near-zero afterwards.
- Four kernels: terminal, retirement, working, boundary.
- With persistent cache: only paid once per (hardware, config) tuple, then cached.
- **Add `jit_per_kernel_s × n_kernels` as a one-time cost.** Most of your wall is steady-state per-age, so this is <5% of total at canonical scale.

### 6.3 fori_loop unconditional iteration

`use_fori_newton=True` runs **all max_iter iterations** even if a cell converged at iter 5. Backward-age warm-start brings most cells to convergence in <10 iters, but the wall cost is the full max_iter regardless.

**Implication:** doubling max_iter doubles wall, regardless of how many iters cells actually need. **Your formula should treat max_iter as wall cost, not "average cells × avg iters" cost.**

### 6.4 vmap fusion vs materialisation

XLA may materialise the full vmap batched c_corners at peak, costing memory but speeding ops. At canonical sizes this can flip into either regime. The empirical anchor implicitly captures whatever XLA chose for that specific config; **the formula will be more accurate for configs near the anchor than for wildly different ones**. State this as a confidence-band caveat.

### 6.5 GPU SIMT efficiency (newton iter divergence)

Different cells need different real iter counts. Under `vmap` + `fori_loop`, all cells run max_iter iters in lockstep — no divergence penalty (that's the whole point of fori+mask). **Effectively zero cost from this in `use_fori_newton=True` mode.** Under `use_fori_newton=False` (while_loop) divergence costs ~1.5-3×; user is on fori everywhere so ignore.

### 6.6 D→H sync points (small)

Per-age progress probe does 1 `device_get` over a tuple of 3 slices. ~50-200 µs per age × 80 ages = <0.02 s/run. Ignore.

### 6.7 `c_corners` advanced gather efficiency

The `c_next[:, j_corners_i, :]` advanced gather is implemented as a scatter-style op in XLA-CUDA. For the access pattern in this code, performance is **~50-70% of theoretical bandwidth peak** based on similar gather patterns in JAX numerical lit. Fold this into `foc_overhead`.

### 6.8 Hardware-specific notes

- **GH200 == H200 GPU** in compute terms (Grace CPU is irrelevant for fp64 throughput). Same fp64 TFLOPS, same HBM bandwidth.
- **A100 SXM4: ~9.7 TFLOPS fp64, 1.5 TB/s HBM2.** Comparable compute, half memory bandwidth.
- **A100 PCIe: ~9.7 TFLOPS fp64, ~1.3 TB/s HBM2.** Same compute, lower bandwidth, ~10% slower in bandwidth-bound workloads.
- **H100 SXM5: ~30 TFLOPS fp64, 3.4 TB/s HBM3.** ~3× faster compute than A100/H200.
- **B200: ~40 TFLOPS fp64, 8 TB/s HBM3e.** ~4× faster than A100.
- **H200 fp64 is same as H100** despite the bigger HBM.

These are all peak; assume ~70% effective for fp64 scientific compute (the 30% gap = scheduling, dispatch, gather inefficiency).

---

## 7. Configurations to project

Produce a table with at least these rows. Show per-age wall and total wall (retirement-only, 33 ages):

| Config name | `state_grid_sizes` | n_z | n_w | n_s | `n_state_quad_nodes` | max_iter | Hardware |
|---|---|---|---|---|---|---|---|
| **Anchor (5⁴ tonight)** | (5,5,5,5) | 11 | 180 | 180 | (2,3,2,3) | 100 | GH200 |
| 5⁴ on H100 SXM5 | (5,5,5,5) | 11 | 180 | 180 | (2,3,2,3) | 100 | H100 SXM5 |
| 7⁴ "tonight's knobs" | (7,7,7,7) | 11 | 180 | 180 | (2,3,2,3) | 100 | H200/GH200 |
| 7⁴ canonical quad | (7,7,7,7) | 11 | 180 | 180 | (3,4,3,4) | 100 | H200/GH200 |
| 7⁴ full canonical | (7,7,7,7) | 11 | 180 | 180 | (3,4,3,4) | 400 | H200/GH200 |
| 7⁴ full canonical on H100 SXM5 | (7,7,7,7) | 11 | 180 | 180 | (3,4,3,4) | 400 | H100 SXM5 |
| 9⁴ "tonight's knobs" | (9,9,9,9) | 11 | 180 | 180 | (2,3,2,3) | 100 | H200 (memory-permitting) |
| 9⁴ canonical | (9,9,9,9) | 11 | 180 | 180 | (3,4,3,4) | 400 | H200 (memory-permitting) |

For each row report:
- Per-age wall (s)
- Total retirement-only wall (h)
- Whether compute-bound or memory-bound at H200's bandwidth
- Cost at $1.99/hr (GH200), $3.29/hr (H200/H100 SXM5), $9.86/hr (B200)

If a row is "memory-bound" using the §6.1 floor, the wall estimate should be the **max** of (FLOPs-time, bandwidth-time), not just FLOPs-time.

---

## 8. Output format

Create `docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md`:

```markdown
# Complexity Analysis + Wall-Time Estimator — 2026-05-06

**Scanner:** [agent name]
**Branch HEAD:** <git rev-parse HEAD>
**Anchor:** GH200 single-device, 5×5×5×5 + reduced quad (see §3 of handoff). Empirical: 273 s/age.

## 1. Compute driver inventory

[Brief enumeration of the dimensions that drive cost, with file:line refs.]

## 2. Per-age cost formula

[The parameterised formula from §5, with constants you fit to the anchor.]

W_age_retire(config, hardware) = ...

## 3. Calibration

[Show the anchor config plugged into the formula. Predicted vs observed. Should be within ±30%.]

## 4. Bottleneck classification

[Per-config: compute-bound or memory-bound, and at what config the crossover happens.]

## 5. Projected walls

[The §7 table, fully populated.]

## 6. Confidence and caveats

[Where the estimator is most accurate (configs near anchor), where less (wildly different shapes).
What you'd need to measure to reduce uncertainty (e.g. one more datapoint at 7⁴).]

## 7. Recommendations

[Given the table, what's the clearest GO call for thesis runs? What's clearly skip?]
```

Target length: 1500-2500 words. Numbers and tables, not prose.

---

## 9. Workflow

1. Pull latest `jax-rewrite` branch (currently HEAD `93ad086`).
2. Read `lifecycle/solver.py` end-to-end first, especially the four FOC functions, `_newton_fori`, `_egm_scan_cell`, and the kernel builders.
3. Construct the FLOP count for one `foc_fn` call (retirement) — read line by line. Don't guess.
4. Multiply up to per-age cost using §4.5 formulas.
5. Calibrate `foc_overhead` and `foc_FLOPs` constants against the §3 anchor.
6. Add §6 bottleneck factors. Document each.
7. Project §7 configs.
8. Write the report.
9. **Don't commit.** Leave staged for user review.

---

## 10. What "good" looks like

- Calibrated formula reproducing 273 ± 80 s/age at anchor.
- §7 table populated with values.
- Bottleneck classification (compute vs memory) for each row.
- One paragraph of "what config is the right thesis target given X-hour budget."
- Caveats acknowledging that:
  - Constants were fit to one anchor; new hardware or wildly different shape ⇒ ±50% error band.
  - XLA scheduling variance can flip configs between bands (you saw 9⁴ → 7⁴ memory blow-up).
  - max_iter is a hard wall cost under fori_loop, not an "average cells × avg iters" cost.

---

## 11. Out of scope / explicit non-goals

- **Don't propose code changes.** Estimator only.
- **Don't model checkpoint resume.** Assume cold start, no resume.
- **Don't model multi-GPU sharding.** Single-device only.
- **Don't run benchmarks.** Static analysis + arithmetic + the §3 anchor.
- **Don't try to derive constants from first principles.** Calibrate against §3.

---

## 12. Quick reference: known constants

- `n_age` = 80 (typical)
- `n_retire_ages` = 33 (`terminal_age - youngest_age_to_solve + 1`)
- `n_work_ages` = 46 (between `start_age` and `youngest_age_to_solve - 1`)
- `n_eta_nodes` × `n_eps_nodes` = 4 × 4 = 16 (working-age inner quadrature)
- GH200/H200 fp64 effective: ~6.8 TFLOPS (70% of 9.7 TFLOPS theoretical)
- H100 SXM5 fp64 effective: ~21 TFLOPS (70% of 30 theoretical)
- B200 fp64 effective: ~28 TFLOPS

That's enough to start. Read the code, calibrate, project. Single report. Stop when done.

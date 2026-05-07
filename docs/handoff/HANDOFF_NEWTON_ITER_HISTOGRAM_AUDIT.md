# Handoff: Audit + Fix Newton-Iter Histogram Counting Across Ages

**Branch:** `jax-rewrite`
**Effort:** ~3-4 hours. Diagnosis + (conditional) fix + verification.
**Output:**
- Findings note in `docs/scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md` documenting the diagnosis.
- (Conditional) one-line / few-line fix if a counting bug is found.
- Smoke validation that the fix produces realistic histograms.

**Critical constraint:** investigation-first. Don't patch anything until the diagnosis is confirmed.

---

## The suspect signal

Tonight's System I × n_z sweep at 4 different n_z values produced **identical Newton-iter histograms across all four runs**:

```
Newton iters: p50=100  p95=100  p99=100  max=100  (max_iter=100)
```

For every n_z ∈ {10, 15, 30, 70}. This holds even though:
- Policy alphas are stable to 3 decimals across runs (so the underlying solves clearly converged behaviorally)
- `total_newton_failures = 0` (so Newton didn't blow up)
- Wall scales linearly with n_z (suggesting different work per cell)

p50 = p95 = p99 = max = max_iter is suspicious. Either:

1. **Every cell really does need all 100 Newton iters** in System I — possible if the FOC is genuinely flat near the optimum, but then `total_newton_failures` should also be high (cells didn't reach `tol`).
2. **The histogram is counting all iters across all ages, summed per cell, instead of per-age max.** That is, `n_iters_used` is accumulating across ages because the array isn't being reset, OR the histogram aggregator is summing the same cells from multiple ages.
3. **`n_iters_used` is reported as the max-iter cap unconditionally** because of how `_newton_fori`'s mask-and-skip pattern interacts with the accumulator.

User's hypothesis (most plausible): **option 2 or 3** — the count is carrying iters across the solve rather than reflecting per-age-per-cell convergence.

If true, we have **no real Newton-iter calibration data** from any of tonight's bundles. Every bundle's histogram is misleading.

---

## What to investigate

### Step 1 — Read the iter-counting code

Files to read carefully:

- `lifecycle/solver.py`, around the `_newton_fori` function (line ~595 area). Specifically how `n_used` is incremented:
  ```python
  return (..., n_used + jnp.where(is_active, jnp.int32(1), jnp.int32(0)))
  ```
  Confirm: this increments by 1 per ACTIVE iter (cells where Newton hasn't converged yet). When a cell converges, `is_active=False` and `n_used` stops incrementing for that cell. The fori_loop runs to `max_iter` regardless.

  At convergence, `n_used` should equal **the iter at which the cell converged** — capped at `max_iter` if it never converged. `n_used = 100 (=max_iter)` means the cell **never reached `tol`** within the iter budget.

- `lifecycle/solver.py:_egm_scan_cell` and `_solve_*_at_cell` — how `n_iters_egm` is collected from `_newton_fori` calls per savings point:
  ```python
  return ..., n_iters_egm, n_backtrack_egm
  ```
  Verify: per (cell, savings) the n_iters returned is the count for THIS Newton solve, not accumulated.

- The kernel-builder return path: how per-cell `n_iters_max` (or whatever the per-cell scalar is) gets aggregated into `(n_z, N_state)` arrays.

- `_build_iter_histograms` — the aggregator. Confirm it treats per-age arrays correctly (no double-counting cells across ages, no cumulative summation).

- `run_lifecycle_solver`'s per-age append:
  ```python
  newton_iter_per_age.append(np.asarray(ni_t))
  ```
  Verify each age's `ni_t` is a fresh array (not the same array being mutated and reappended).

### Step 2 — Reproduce at tiny scale

Build a tiny config that solves in seconds. Print, per age:
- Mean Newton iters across cells
- Max Newton iters across cells
- Number of cells where Newton hit max_iter

If the mean is wildly close to max_iter for every age (e.g., 99.5/100 for every age), that's the bug — Newton SHOULD have variation across cells.

Compare against expectation: with backward-age warm-start, typical Newton iters per cell should be 5-30. p99 across a real solve should be 20-50, not 100.

If at tiny scale the histogram still shows p50 = p95 = p99 = max = max_iter, the bug is real and reproducible.

### Step 3 — Diagnose

If the bug reproduces, instrument intermediate computations:

1. **Per-cell n_iters_used at one age:** print `np.unique(n_iters_array)` — the distinct values across all cells in one age. If they're all `max_iter`, the bug is in `_newton_fori`'s incrementer (option 3 above). If they vary but the histogram shows uniformly max_iter, the bug is in the aggregator.

2. **Compare two consecutive ages:** print the per-cell n_iters for age 99 and age 98. If they're identical, the array is being reused (option 2). If different but both max_iter, the kernel itself is over-counting.

3. **Compare to backtrack histogram:** if Newton p99=100 but backtrack histogram is much lower (e.g., p99=3), then Newton's counter is bugged and backtrack's isn't — diagnostic narrowing.

### Step 4 — Fix (if bug found)

Likely fix candidates ordered by probability:

a. **`n_iters_per_age` array reuse**: a list of NumPy arrays where the same underlying array gets appended multiple times (Python aliasing). Fix: explicit `.copy()` on append.

b. **`n_iters_used` not zeroed between Newton solves at different cells**: but this is per-cell, not per-age, so unlikely given the vmap.

c. **`_build_iter_histograms` summing instead of taking per-age max**: easy to spot from the function definition.

d. **`is_active` flag never flipping**: would mean Newton's convergence check (residual < tol) never fires. Different bug; would show as actual non-convergence rather than histogram miscount.

The fix is likely a few lines once the bug is located. Don't refactor beyond the minimal change.

### Step 5 — Validate

After the fix:

- **Smoke at tiny config**: histogram should show realistic per-age Newton p99 (somewhere between 5 and max_iter, not at max_iter for every cell).
- **Backtrack histogram should make sense relative to Newton**: e.g., max_backtrack_iter is 10; backtrack p99 should be <= 10×max_iter cumulative if it's a sum, or <= 10 if it's per-iter max.
- **Bit-identity on policies**: histogram is a diagnostic, not a math input. Policies must be bit-identical to pre-change verify/smoke.py output.
- **Total Newton failures still 0**: was 0 before; should remain 0 unless the fix accidentally changes when cells are flagged failed.

### Step 6 — Re-extract histograms from existing bundles (if reasonable)

The 4 n_z bundles + the existing 5⁴ baseline bundle have `diagnostics.pkl` with whatever histogram we wrote. If the bug is in solver-side counting, those histograms are also wrong but the policies are valid. Worth flagging in the findings note that **the histograms in those existing bundles cannot be trusted** if the bug is real; future runs after the fix will produce reliable ones.

---

## Pause points

- **Step 2 doesn't reproduce the bug** at tiny scale: the apparent p50=p95=p99=max=100 might be config-specific (e.g., specific to the System I (rtb,) state where FOC is flat). Pause and surface, don't patch blindly.
- **Diagnosis points to multiple potential bugs**: one fix at a time. Each fix needs its own validation gate.
- **Fix candidate (b) or (d)**: those imply a deeper Newton or convergence bug, not just a histogram bug. Stop and report — those are out of scope for this handoff and need user approval before patching.

---

## Out of scope

- **Lifecycle math changes** (FOC, Newton convergence, EGM, etc.). Only histogram-counting machinery.
- **Refactoring iter-tracking architecture**. Minimal change to fix the specific bug, not a redesign.
- **Inf-horizon's own histogram code path** (in `inf_horizon_solver.py`). Different code path; out of scope for this handoff. If the same bug exists there, flag it for a separate follow-up.
- **Multi-GPU pmap path histogram audit** beyond the existing multi-GPU audit (which the agent should reference if bugs appear at n_dev>1).

---

## Implementation checklist

- [ ] Step 1: read `_newton_fori`, `_egm_scan_cell`, `_build_iter_histograms`, `run_lifecycle_solver` per-age append. Document iter-counting flow.
- [ ] Step 2: build tiny smoke that prints per-age per-cell n_iters distribution. Reproduce or rule out the bug.
- [ ] Step 3: instrument intermediate values to localize the bug if it reproduces.
- [ ] Step 4 (conditional): apply minimal fix.
- [ ] Step 5: validate fix — smoke shows realistic per-age Newton iters, policies bit-identical.
- [ ] Step 6: write `docs/scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md` with:
  - Diagnosis (was the bug real?)
  - Bug location + mechanism (if real)
  - Fix description (if applied)
  - Implications for existing bundles' histograms (trust / don't trust)
- [ ] Commit:
  ```
  diag: fix Newton-iter histogram counting across ages

  Previously the per-cell Newton iter counter saturated at max_iter
  in every reported histogram, making p50=p95=p99=max=max_iter
  uniformly across solves with very different actual Newton work.
  
  Bug: <one-line description>.
  
  Fix: <one-line>.
  
  Verified at tiny config: per-age Newton p99 now reflects the actual
  worst-cell convergence (5-50 typical) rather than always max_iter.
  Policies bit-identical (histogram is diagnostic only, no math change).
  
  Note: existing bundles' newton_iter_histogram fields are unreliable
  due to this bug. Re-running produces correct histograms.
  ```

---

## Why this matters

- **max_iter calibration is blocked** without a correct histogram. Every recommendation we've made about lowering max_iter (e.g., max_iter=30 from 100) is based on the histogram's p99 — if p99 is always reported as max_iter, we have no real signal.
- **Cycle 2 of any calibration loop is impossible** without trustworthy convergence data. We're guessing instead of measuring.
- **Thesis methodology** — claiming "max_iter calibrated from data" requires the data to be real.

This is a small fix with large downstream value.

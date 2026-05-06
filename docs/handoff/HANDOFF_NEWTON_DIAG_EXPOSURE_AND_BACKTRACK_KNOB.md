# Handoff: Expose Newton-iter Histogram + Make `max_backtrack_iter` Configurable

**Branch:** `jax-rewrite`
**Two changes bundled because they're complementary, both small, both prerequisites for measurement-driven Newton tuning.**

**Effort:** half a day total (1-3 hours each, parallelizable).

**Why bundled:** they unlock the same payoff — **2-2.5× wall reduction by setting `max_iter` and `max_backtrack_iter` from data instead of guessing.**

---

## Background

Tonight's run measured 273 s/age at `max_iter=100, max_backtrack_iter=10` under `use_fori_newton=True`. Under fori_loop the wall cost is *literally* `max_iter × per_iter_cost` regardless of cell convergence — there's no early-exit. So:

- **`max_iter=100` is wall cost regardless of whether 99% of cells converge in 5 iters.** Same for `max_backtrack_iter=10` inside `_backtracking_fori`.
- **If we measure the actual iter usage** (median, p99, max), we can set both values to ~1.5× p99 — keeping a safe ceiling for tail cells while halving (or more) the wall cost.

But we currently have no diagnostic exposing actual iter usage. Newton state's `n_iters_used` field is computed correctly inside `_newton_fori` but is not aggregated into `diagnostics` at the end of the solve.

These two handoffs fix that.

---

## Change A: Expose Newton-iter and backtrack-iter histograms in `diag` output

### A.1 Goal

After solve completes, `diagnostics` dict (returned by `run_lifecycle_solver` and saved to bundle) should include:

- `diagnostics["newton_iter_histogram"]`: dict-or-array mapping iter count → cell count, OR percentile stats (p50, p95, p99, max). Per-age and aggregate.
- `diagnostics["backtrack_iter_histogram"]`: same for backtrack inner loop.

### A.2 What's there now

`_newton_fori` already returns `n_iters_used` — the actual iter count for each cell (incremented only when the cell was active in that iter). It's the 6th return value of `newton_2d_with_line_search`. Look at [solver.py:608-651](../../lifecycle/solver.py#L608) — the fori path correctly tracks this.

**But:** the return value is consumed in the per-cell solve and not propagated up to the per-age outputs. The kernel builders return `(c, s, b)` only — `n_iters_used` is dropped on the floor.

For backtrack iters: `_backtracking_fori` doesn't currently return the actual iter count used (just exits at first improvement). Need to add a return value tracking how many halvings actually happened.

### A.3 Implementation

#### A.3.1 In `_backtracking_fori` (solver.py:~501)

Currently returns `(a_s, a_b, fs, fb, ..., found)`. Add `n_backtrack_used`:

```python
def _backtracking_fori(...):
    ...
    init = (
        ..., found_init, jnp.int32(0)   # NEW: backtrack counter
    )

    def body(i, state):
        ..., found, n_used = state
        is_active = jnp.logical_not(found)
        ...
        return (..., new_found, n_used + jnp.where(is_active, jnp.int32(1), jnp.int32(0)))

    final = lax.fori_loop(0, max_backtrack_iter, body, init)
    *fields, found, n_used = final
    return *fields, found, n_used   # one extra return value
```

Caller (`_newton_fori`) collects it and propagates.

#### A.3.2 In `_newton_fori` and `newton_2d_with_line_search` 

Change return signature from `(a_s, a_b, e, exit_code, err_norm, n_iters_used)` to:

```python
return a_s, a_b, e, exit_code, err_norm, n_iters_used, n_backtrack_total
```

Where `n_backtrack_total` is the sum of all backtrack iters over all Newton iters (since fori_loop runs `max_iter` Newton iters, summing the per-iter backtrack count gives total). Update both `_newton_fori` and `_newton_while` for consistency (while_loop path returns the actual variable count which is fine).

#### A.3.3 In `_egm_scan_cell`

Currently returns `(x_egm, c_egm, a_s_egm, a_b_egm)`. Add two more arrays:

```python
return x_egm, c_egm, a_s_egm, a_b_egm, n_iters_egm, n_backtrack_egm
```

Where `n_iters_egm`, `n_backtrack_egm` are shape `(n_savings + 1,)` — one entry per savings point. Pad first entry with 0 (matches the egm_anchor padding).

#### A.3.4 In `_solve_*_at_cell`

Plumb the two extra arrays through. `_lift_to_wealth_grid` will need to handle them too — but they're per-savings-point (not per-wealth-grid). Either lift them onto the wealth grid via the same x_egm interp, or aggregate them per cell (e.g., max iter count over savings points).

**Simplest aggregation:** at the cell level, return `max(n_iters_egm)` and `sum(n_backtrack_egm)` — the worst-case Newton convergence cost for any savings point in that cell.

```python
def _solve_*_at_cell(...):
    ...
    x_egm, c_egm, a_s_egm, a_b_egm, n_iters_egm, n_backtrack_egm = _egm_scan_cell(...)
    n_iters_max = jnp.max(n_iters_egm)
    n_backtrack_total = jnp.sum(n_backtrack_egm)
    c_w, s_w, b_w = _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)
    return c_w, s_w, b_w, n_iters_max, n_backtrack_total
```

#### A.3.5 In kernel builders (`_build_per_age_*_kernel_*`)

Each `call(...)` wrapper needs to return two extra arrays of shape `(n_z, N_state)` (one entry per cell):

```python
def call(c_next_jnp, ..., init_a_s_arr, init_a_b_arr):
    c_pm, as_pm, ab_pm, ni_pm, nb_pm = per_dev_solve(...)
    return collapse(c_pm), collapse(as_pm), collapse(ab_pm), collapse_to_2d(ni_pm), collapse_to_2d(nb_pm)
```

#### A.3.6 In `run_lifecycle_solver` orchestrator

Collect per-age `n_iters_max` and `n_backtrack_total` arrays. Aggregate into the diagnostics dict at end of solve:

```python
ni_per_age = []   # list of (n_z, N_state) arrays
nb_per_age = []
# ... in age loop:
c_t, s_t, b_t, ni_t, nb_t = retirement_kernel(...)
ni_per_age.append(np.asarray(ni_t))
nb_per_age.append(np.asarray(nb_t))
# After loop:
ni_all = np.stack(ni_per_age)   # shape (n_ages_solved, n_z, N_state)
nb_all = np.stack(nb_per_age)
diagnostics["newton_iter_histogram"] = {
    "p50": float(np.median(ni_all)),
    "p95": float(np.percentile(ni_all, 95)),
    "p99": float(np.percentile(ni_all, 99)),
    "max": int(np.max(ni_all)),
    "per_age_p99": [float(np.percentile(ni_all[t], 99)) for t in range(ni_all.shape[0])],
}
diagnostics["backtrack_iter_histogram"] = {
    "p50": ..., "p95": ..., "p99": ..., "max": ...,
}
```

#### A.3.7 Print summary at end of run

In the verbose output of `run_lifecycle_solver`:

```python
if verbose >= 1:
    nih = diagnostics["newton_iter_histogram"]
    print(f"  Newton iters: p50={nih['p50']:.0f}  p95={nih['p95']:.0f}  p99={nih['p99']:.0f}  max={nih['max']}")
    bth = diagnostics["backtrack_iter_histogram"]
    print(f"  Backtrack iters: p50={bth['p50']:.1f}  p95={bth['p95']:.1f}  p99={bth['p99']:.1f}  max={bth['max']}")
```

User reads these numbers off the run, then sets `max_iter` and `max_backtrack_iter` for the next run.

### A.4 Validation

Run smoke (`verify_smoke.py`) with the change. Expected:
- New stats appear in stdout summary and in `diagnostics` dict.
- p99 should be much smaller than `max_iter=100` (probably 10-30 for Newton, 2-5 for backtrack).
- No regression in alphas (math unchanged).

---

## Change B: `SolverConfig.max_backtrack_iter` field

### B.1 Goal

Make `max_backtrack_iter` a configurable SolverConfig field instead of hardcoded 10. Once data shows p99 backtrack count is ~3-5, user can drop default.

### B.2 What's there now

`max_backtrack_iter` is currently a hardcoded `int = 10` field in `SolverConfig`? Check `lifecycle/model.py` — it's already in `SolverConfig` as `max_backtrack_iter: int = 10`. **So actually it's already configurable, just hasn't been tuned.**

Wait — verify by reading the file. If it's already there, this change is just "tune the default to 4 once data supports it." That's a one-line change.

### B.3 Implementation

If already configurable: **this handoff reduces to "verify the value is honored end-to-end and document the recommendation."**

```bash
# Verify it's plumbed through:
grep -n "max_backtrack_iter" lifecycle/model.py lifecycle/solver.py
```

You should see:
- `lifecycle/model.py`: field declaration
- `lifecycle/solver.py`: usage in `_backtracking_fori` via the `static` tuple

If both are present, this handoff is a 5-minute "no-op" check. The work is just **measuring p99 backtrack iters from change A's output** and updating the SolverConfig docstring with a recommendation:

```python
# In SolverConfig docstring:
# Recommended values from production calibration (2026-05-06):
# max_backtrack_iter=4  (p99 observed ≈ 2-3 with backward-age warm-start)
```

If NOT plumbed through (i.e., something hardcodes 10 somewhere): make it configurable. ~5 lines.

---

## Output / verification

For both changes:

1. Run `verify_smoke.py`. Expected stdout addition:
   ```
   Newton iters: p50=4  p95=12  p99=23  max=47
   Backtrack iters: p50=0  p95=2  p99=3  max=7
   ```
2. After tonight's bundle lands and the change is in place, **also run** `python verify_benchmark_bundle.py` (post-change) to get production-scale measurements. The numbers above are smoke-scale; production p99 might be a bit higher (more cells, longer tail).

Once production p99 is known, the user updates:
- `verify_smoke.py` → `max_iter=2 × p99_observed`
- `verify_benchmark_bundle.py` → same
- `SolverConfig` defaults → reflects the new recommendation

---

## Implementation checklist

- [ ] Modify `_backtracking_fori` to return `n_backtrack_used`.
- [ ] Modify `_newton_fori` and `_newton_while` to return `n_backtrack_total`.
- [ ] Modify `newton_2d_with_line_search` dispatcher to return the same.
- [ ] Modify `_egm_scan_cell` to vmap over savings points and return per-savings iter counts.
- [ ] Modify `_solve_*_at_cell` to aggregate per-cell.
- [ ] Modify three kernel builders (`_pmap` and `_vmap_only` variants) to thread the two extra arrays.
- [ ] Modify `run_lifecycle_solver` to collect, aggregate, and add to `diagnostics` dict.
- [ ] Add stdout print of histograms in verbose-1 summary.
- [ ] Verify `max_backtrack_iter` is already a SolverConfig field (likely yes).
- [ ] Run `verify_smoke.py` — confirm stats appear, alphas unchanged.
- [ ] Commit message:
  ```
  diagnostics: expose Newton + backtrack iter histograms in diag output

  After solve completes, diag now includes p50/p95/p99/max for both
  Newton outer iters and backtrack inner iters, per-age and aggregate.
  Lets the user set max_iter / max_backtrack_iter from measurement
  instead of guessing — and under fori_loop both values are wall cost
  regardless of cell convergence, so cutting them is direct wall savings.

  No math change. Two extra arrays propagate from cell-solver up to
  diagnostics aggregator. Default behaviour unchanged.
  ```
- [ ] Push.

---

## Why this is bundled with the EE diagnostic and chunking handoffs

These three handoffs together (EE diagnostic + Newton iter exposure + chunking debug) close the **measurement loop**:

1. **Bundle lands.**
2. **EE diagnostic** tells us if policies are correct.
3. **Newton iter histogram** tells us if `max_iter` was wasted.
4. **chunking** lets us scale up to 7⁴ once 1+2+3 are resolved.

After all three: the user has a calibrated, measured, chunkable solver that can target 7⁴ at production quality. **Tier 1 + 2 of the efficiency stack lands by tomorrow morning.**

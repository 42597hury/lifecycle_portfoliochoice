# Newton-iter histogram counting audit (2026-05-07)

**Branch:** `jax-rewrite`
**Status:** Bug confirmed and fixed. Policies bit-identical. Histograms now reflect real Newton-call distribution.
**Handoff:** [HANDOFF_NEWTON_ITER_HISTOGRAM_AUDIT.md](../handoff/HANDOFF_NEWTON_ITER_HISTOGRAM_AUDIT.md)

## TL;DR

The histogram fields `newton_iter_histogram` and `backtrack_iter_histogram`
in every diagnostics bundle written before this fix are unreliable —
not because Newton was failing, but because the per-cell aggregator
collapsed the per-savings-point Newton-iter array via `jnp.max` BEFORE
the histogram aggregator ever saw it.

With `n_savings = 15` and any single high-savings-point Newton solve
hitting `max_iter`, the cell's reported value is `max_iter`. With
~20–30% of high-savings Newton calls actually saturating, ~95%+ of
cells report `max_iter`, and `p50 = p95 = p99 = max = max_iter`
falls out as a property of max-of-many, not of true Newton work.

The fix exposes the full per-savings array to the histogram (with the
s=0 anchor stripped, since no Newton solve happens there). Policies
are bit-identical because the change is purely diagnostic.

## Diagnosis

### Reproduction at tiny scale

Tiny config (n_w=15, n_savings=15, state grid (2,2,2,2), n_z=4, max_iter=50,
6 ages from 60..65 — see [scripts/scratch/probe_newton_iter_histogram.py](../../scripts/scratch/probe_newton_iter_histogram.py)).

**Pre-fix histogram:** `Newton iters: p50=50  p95=50  p99=50  max=50  (max_iter=50)`

Per-age unique values:

| age | shape | mean | max | unique values |
|-----|-------|------|-----|---------------|
| 60 (WORK) | (4, 16) | 50.00 | 50 | {50} |
| 61 (WORK) | (4, 16) | 50.00 | 50 | {50} |
| 62 (WORK) | (4, 16) | 50.00 | 50 | {50} |
| 63 (RETIRE) | (4, 16) | 49.73 | 50 | {40, 43, 50} |
| 64 (RETIRE) | (4, 16) | 48.98 | 50 | {35, 38, 43, 45, 46, 50} |
| 65 (TERM) | (16,) | 49.44 | 50 | {44, 47, 50} |

The per-cell array shape `(n_z, N_state)` shows that the aggregator gets
already-collapsed scalars (max-over-savings), not per-savings counts.

### Drilling into a single cell

[scripts/scratch/probe_per_savings_iters.py](../../scripts/scratch/probe_per_savings_iters.py)
calls `_egm_scan_cell` directly for terminal-age `i_s ∈ {0,1,2,3}`
and prints the full `n_iters_egm` (shape `(n_savings + 1,)` per cell):

```
i_s=0: n_iters_egm = [0, 37, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 47, 50, 50]
i_s=1: n_iters_egm = [0, 40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 50, 50, 50]
i_s=2: n_iters_egm = [0, 41, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,  7, 50, 50]
i_s=3: n_iters_egm = [0, 35, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 45, 50, 50]
```

Reading per-savings-point:
- index 0 (anchor): 0 by construction (no Newton solve at s=0)
- index 1 (smallest savings): 35–41 iters (cold-start convergence on the
  s≈savings_min FOC, expected)
- indices 2–12 (middle savings): 6–10 iters each — typical Newton work
- indices 13–15 (largest savings): 45–50 iters, with 2–3 of them at exactly
  `max_iter` per cell — Newton isn't converging on the high-savings tail
  within budget

So the underlying counter (`_newton_fori`'s `n_used`) is correct: it
varies per-savings-point as expected. The bug is the **per-cell collapse**
in the per-cell drivers.

### Where the bug lives

[lifecycle/solver.py:1240](../../lifecycle/solver.py#L1240) (and the
analogous lines in `_solve_retirement_at_cell` and `_solve_working_at_cell`):

```python
n_iters_max = jnp.max(n_iters_egm)        # collapses per-savings → per-cell
n_backtrack_total = jnp.sum(n_backtrack_egm)
return c_w, a_s_w, a_b_w, n_iters_max, n_backtrack_total
```

`n_iters_egm` has shape `(n_savings + 1,)`. The `jnp.max` reduction makes
the per-cell return a scalar — by the time `_build_iter_histograms` ever
sees the array, it's already shape `(n_z, N_state)` of cell-maxima, not
the underlying `(n_z, N_state, n_savings)` distribution.

This is a hybrid of options (a) and (c) from the handoff: the counting
machinery is correct, but the per-cell aggregation step pre-collapses
the array in a way that biases the visible histogram toward `max_iter`.

## Fix

Three per-cell drivers updated to return the per-savings array
(s=0 anchor stripped) instead of `jnp.max` / `jnp.sum`:

```python
n_iters_per_s = n_iters_egm[1:]
n_backtrack_per_s = n_backtrack_egm[1:]
return c_w, a_s_w, a_b_w, n_iters_per_s, n_backtrack_per_s
```

Twelve kernel reshape sites updated from `(n_z, N_state)` to
`(n_z, N_state, -1)` to accommodate the savings axis. The chunked runner
paths and pmap collapse already use generic trailing-shape handling
(`r[k].shape[2:]`, `axis=0` concat) so they need no change.

`_build_iter_histograms` ravels per-age arrays unconditionally, so it
absorbs the new savings axis without code changes.

## Validation

### Histogram now realistic

Same tiny config, post-fix:

```
Newton iters:    p50=2  p95=50  p99=50  max=50  (max_iter=50)
```

Per-age (now shape `(n_z, N_state, n_savings)`):

| age | mean | max | p99 | #at-max | #cells | unique vals |
|-----|------|-----|-----|---------|--------|-------------|
| 60 (WORK)   | 18.46 | 50 | 50 | 242/960 | 35 distinct |
| 61 (WORK)   | 16.48 | 50 | 50 | 233/960 | 37 distinct |
| 62 (WORK)   | 16.87 | 50 | 50 | 218/960 | 34 distinct |
| 63 (RETIRE) | 16.20 | 50 | 50 | 209/960 | 32 distinct |
| 64 (RETIRE) | 15.05 | 50 | 50 | 182/960 | 26 distinct |
| 65 (TERM)   | 17.09 | 50 | 50 |  40/240 | 21 distinct |

p50 = 2 (most Newton solves converge in 2 iters thanks to backward warm-start),
mean ~17 (a handful of high-savings calls dominate the average), and ~20–30%
of Newton calls hit `max_iter` (the high-savings tail). p99 still saturates
because of that tail — that's a real signal, not a counting artifact.

### Bit-identity

Stashed solver.py only, ran [scripts/scratch/bit_identity_check.py](../../scripts/scratch/bit_identity_check.py)
to save policies under both baseline and fixed code. Restored the fix and
diffed:

```
C: bit-identical: True  max|delta|=0.000000e+00
S: bit-identical: True  max|delta|=0.000000e+00
B: bit-identical: True  max|delta|=0.000000e+00
```

This is structurally guaranteed: the change touches only diagnostic
outputs (`n_iters_egm` derivatives), never the policy math
(`c_w`, `a_s_w`, `a_b_w`).

## Implications for existing bundles

Every diagnostics bundle written before this commit has unreliable
`newton_iter_histogram` / `backtrack_iter_histogram` fields. Specifically:

- The 4 System I × n_z bundles from the recent sweep (n_z ∈ {10, 15, 30, 70})
- The 5⁴ baseline bundle
- All inf-horizon `inf_horizon_solver.py` bundles, IF that solver shares
  the same per-cell-max collapse pattern (out of scope for this handoff —
  flagged for a separate audit; see [HANDOFF_INF_HORIZON_AUDIT.md](../handoff/HANDOFF_INF_HORIZON_AUDIT.md)
  if it isn't already covered)

**Policies in those bundles are valid** — the bug is purely in the
diagnostic counters. Re-running with the fix produces correct histograms;
no need to re-run for policy correctness.

## What `total_newton_failures = 0` actually means

Side observation surfaced during diagnosis: `age_newton_fail` (initialized
at [lifecycle/solver.py:2500](../../lifecycle/solver.py#L2500)) is never
incremented anywhere in the solver. So `diagnostics["total_newton_failures"]`
is a constant 0 by construction — it doesn't reflect Newton-failure
exit codes. This is a **separate** diagnostic gap (out of scope for
this handoff but worth flagging as a follow-up): the per-cell solvers
do return `exit_code` from `_newton_fori`, but it is currently discarded
inside the per-cell drivers and never aggregated up.

The handoff's pause point #1 ("if Newton genuinely needed all 100 iters,
`total_newton_failures` should be high") therefore doesn't actually rule
that scenario out at the diagnostic level. The per-savings drill above
is what rules it out.

## Files changed

- [lifecycle/solver.py](../../lifecycle/solver.py) — three per-cell driver
  returns + twelve reshape sites + two docstring updates. Net: +38 / −28
  lines of diagnostic-only changes; no math changes.

## Helper scripts (kept under scripts/scratch/ for future probes)

- [probe_newton_iter_histogram.py](../../scripts/scratch/probe_newton_iter_histogram.py)
  — runs a tiny solve and prints per-age per-cell n_iters distribution.
- [probe_per_savings_iters.py](../../scripts/scratch/probe_per_savings_iters.py)
  — calls `_egm_scan_cell` directly to print the full `n_iters_egm`
  array for individual cells.
- [bit_identity_check.py](../../scripts/scratch/bit_identity_check.py)
  — saves C/S/B as NPZ for pre/post-change diffing.

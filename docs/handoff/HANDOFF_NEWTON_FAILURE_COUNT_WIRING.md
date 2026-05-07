# Handoff: Wire Newton Failure Counts Through Solver + Inf-Horizon

**Branch:** `jax-rewrite`
**Effort:** ~2-3 hours. Mechanical surgery, no design. Mirrors the
8bfaec9 histogram-fix pattern almost exactly.
**Output:**
- `lifecycle/solver.py` — `_egm_scan_cell` returns per-savings exit codes;
  three per-cell drivers slice + return them; kernel-collapse reshape sites
  add the new array; aggregator counts failures into `age_newton_fail`.
- `lifecycle/inf_horizon_solver.py` — fixed-point loop receives per-iter
  exit codes; `_build_diagnostics` aggregates and reports
  `total_newton_failures` (drop the hardcoded `0`).
- `tests/test_newton_failure_count.py` (or add to existing test file) —
  bit-identity check pre/post + a small smoke that confirms a deliberately
  under-budgeted Newton (small `max_iter`) registers non-zero failures.
- Commit:
  ```
  diagnostics: wire per-cell Newton exit codes through solver +
  inf-horizon; total_newton_failures and age_newton_fail now reflect
  real per-cell convergence (was constant 0 placeholder).
  ```

**Critical constraint:** diagnostic-only change. Policies must be
bit-identical pre/post. The change touches only the diagnostic plumbing
(`exit_code` propagation), never the policy math (`c_w`, `a_s_w`, `a_b_w`).

---

## Why

`diagnostics["total_newton_failures"]` and `diagnostics["age_newton_fail"]`
are **constant 0 by construction** today. The per-cell Newton solver
([lifecycle/solver.py:528-529](../../lifecycle/solver.py#L528),
[L686-687](../../lifecycle/solver.py#L686)) computes a real `exit_code`
(`EC_INTERIOR` if converged within `tol`, else `EC_NEWTON_FAIL`). It then
threads that exit code back from `_newton_fori` to `_egm_scan_cell`
([L1160](../../lifecycle/solver.py#L1160)) where it's discarded:

```python
(a_s_opt, a_b_opt, V_dot, _exit_code, _err,
 n_iter_used, n_bt_total) = newton_2d_with_line_search(...)
```

The aggregator at [solver.py:2506](../../lifecycle/solver.py#L2506)
initializes `age_newton_fail = np.zeros(n_age, dtype=np.int64)` and never
writes to it. `total_newton_failures` at [L2935](../../lifecycle/solver.py#L2935)
is its sum: always 0.

The inf-horizon solver mirrors this gap with an explicit comment at
[inf_horizon_solver.py:653](../../lifecycle/inf_horizon_solver.py#L653):
```python
0,  # total_newton_failures: kernel doesn't expose per-cell exits
```

This came up during the Newton-iter histogram audit (see
[docs/scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md §"What total_newton_failures
actually means"](../scans/NEWTON_HISTOGRAM_AUDIT_2026-05-07.md)). The
histogram fix (8bfaec9) wired per-savings iter counts through the same
drivers; this handoff extends the same plumbing for exit codes so we have
**real** failure counts for the upcoming inf-horizon state-grid sensitivity
sweep.

---

## What to change

### Step 1 — `_egm_scan_cell` returns exit codes

[lifecycle/solver.py:1155-1186](../../lifecycle/solver.py#L1155). Capture
`exit_code` instead of underscoring it; concatenate the s=0 anchor
placeholder (use `EC_INTERIOR` as the placeholder — anchor isn't a real
failure, and downstream we slice it off at `[1:]` anyway).

```python
(a_s_opt, a_b_opt, V_dot, exit_code, _err,
 n_iter_used, n_bt_total) = newton_2d_with_line_search(...)
...
x_arr, c_arr, a_s_arr, a_b_arr, ni_arr, nb_arr, ec_arr = vmap(per_savings_point)(s_grid)
...
exit_code_egm = jnp.concatenate([
    jnp.array([EC_INTERIOR], dtype=ec_arr.dtype), ec_arr,
])
return (x_egm, c_egm, a_s_egm, a_b_egm,
        n_iters_egm, n_backtrack_egm, exit_code_egm)
```

`per_savings_point` needs an extra return value (`n_iter_used → exit_code`
becomes `n_iter_used, n_bt_total, exit_code`). Make sure `EC_INTERIOR` /
`EC_NEWTON_FAIL` constants are imported at the top of the file (already
defined in this module — verify scope).

### Step 2 — Three per-cell drivers slice + return exit codes

[`_solve_terminal_at_i_s` L1206](../../lifecycle/solver.py#L1206),
[`_solve_retirement_at_cell` L1247](../../lifecycle/solver.py#L1247),
[`_solve_working_at_cell` L1308](../../lifecycle/solver.py#L1308). Each:

```python
(x_egm, c_egm, a_s_egm, a_b_egm,
 n_iters_egm, n_backtrack_egm, exit_code_egm) = _egm_scan_cell(...)
...
n_iters_per_s = n_iters_egm[1:]
n_backtrack_per_s = n_backtrack_egm[1:]
exit_code_per_s = exit_code_egm[1:]
return c_w, a_s_w, a_b_w, n_iters_per_s, n_backtrack_per_s, exit_code_per_s
```

Same shape `(n_savings,)` per cell — same downstream plumbing as the
iter counts.

### Step 3 — Kernel collapse + reshape sites

The histogram fix updated 12 reshape sites from `(n_z, N_state)` to
`(n_z, N_state, -1)`. Find them by grepping for the pattern updated in
8bfaec9:

```bash
git show 8bfaec9 -- lifecycle/solver.py | grep -E '^\+.*reshape|^\+.*concat'
```

Each site that handles `n_iters_per_s` / `n_backtrack_per_s` needs a
parallel handler for `exit_code_per_s`. Final per-age shape should be
`(n_z, N_state, n_savings)` matching the iter count arrays.

The pmap collapse paths (`r[k].shape[2:]`, `axis=0` concat) already use
generic trailing-shape handling — no change needed beyond adding the new
return slot to the kernel signature.

### Step 4 — Aggregator: count failures per age

[lifecycle/solver.py:2506](../../lifecycle/solver.py#L2506) initializes
`age_newton_fail`. After each age's solve produces an
`exit_code_per_age` array of shape `(n_z, N_state, n_savings)`:

```python
n_fail_this_age = int(np.sum(exit_code_per_age != EC_INTERIOR))
age_newton_fail[t] = n_fail_this_age
```

Plug into both the partial-bundle write path ([L2735](../../lifecycle/solver.py#L2735))
and the final write path ([L2785](../../lifecycle/solver.py#L2785)).
`total_newton_failures` at [L2936](../../lifecycle/solver.py#L2936)
already sums `age_newton_fail` — no change there.

### Step 5 — Inf-horizon mirror

[lifecycle/inf_horizon_solver.py:590](../../lifecycle/inf_horizon_solver.py#L590)
receives `(c_new_jnp, s_new_jnp, b_new_jnp, ni_jnp, nb_jnp)` from the
retirement kernel — add `ec_jnp`:

```python
(c_new_jnp, s_new_jnp, b_new_jnp,
 ni_jnp, nb_jnp, ec_jnp) = retirement_kernel(...)
...
exit_code_per_iter.append(np.asarray(ec_jnp))
```

Initialize `exit_code_per_iter = []` alongside the other histories
([L564-565](../../lifecycle/inf_horizon_solver.py#L564)). Sum failures
per iter and accumulate. In `_build_diagnostics`
([L411-446](../../lifecycle/inf_horizon_solver.py#L411)) replace the
hardcoded `0` with `int(sum_of_per_iter_failures)`. Optional: also
expose a per-iter failure count list (parallel to `per_iter_p99` in the
histogram dict) so analysis can see the failure trajectory.

---

## Test plan

### Bit-identity (mandatory)

Reuse [scripts/scratch/bit_identity_check.py](../../scripts/scratch/bit_identity_check.py)
from the histogram fix. Stash `solver.py` + `inf_horizon_solver.py`,
save policies under baseline; restore; save policies under fix; diff:

```
C: bit-identical: True  max|delta|=0.000000e+00
S: bit-identical: True  max|delta|=0.000000e+00
B: bit-identical: True  max|delta|=0.000000e+00
```

This is structurally guaranteed (diagnostic-only change), but verify
explicitly to catch any accidental policy-path edit.

Run for both the lifecycle solver and the inf-horizon solver (the latter
has its own bit-identity reference path — extend the script if needed).

### Failure-count smoke (mandatory)

Tiny config (n_w=15, n_savings=15, state grid (2,2,2,2), n_z=4) with
deliberately tight `max_iter=3` Newton budget so most cells fail to
converge. Run one age (terminal, simplest path). Confirm:
- `diagnostics["age_newton_fail"][terminal_age] > 0`
- `diagnostics["total_newton_failures"] > 0`
- The number is plausible (not all cells, not zero — somewhere between)

Then run the same config with `max_iter=200` (excess budget). Confirm
`total_newton_failures == 0` (or very close — all cells should have time
to converge).

### Inf-horizon failure-count smoke (mandatory)

Tiny inf-horizon config (state grid (2,2,2,2), n_z=1, n_w=20,
`max_iter` Newton=3, outer `max_iter`=5). Confirm
`diagnostics["total_newton_failures"] > 0` and is non-decreasing as
`max_iter` Newton drops.

### Pre-existing tests must still pass

```bash
pytest tests/ -x -q
```

If any pre-existing test reads `total_newton_failures == 0`, that test
was implicitly relying on the bug — update the test to reflect the
fix, and note in the commit message which tests were updated and why.

---

## Pause points

- **Bit-identity fails.** Surface the diff. Likely indicates an accidental
  edit to the policy-math path. Don't proceed until policies are bit-identical.
- **Smoke shows `total_newton_failures` doesn't move with `max_iter`.**
  Indicates the exit codes aren't propagating through the kernel collapse
  correctly. Check the reshape sites (Step 3).
- **`EC_INTERIOR` / `EC_NEWTON_FAIL` import scope** — these are defined
  in [lifecycle/solver.py](../../lifecycle/solver.py) (search for
  `EC_INTERIOR =`). If the kernel collapse / aggregator paths don't have
  them in scope, import explicitly rather than hardcoding `0`/`1`
  comparisons.

---

## Out of scope

- Per-cell exit-code histograms (the `int(np.sum(... != EC_INTERIOR))`
  scalar per age is enough for now; richer per-savings-position breakdown
  is a follow-up if the inf-horizon sweep surfaces interesting patterns).
- Re-running existing bundles. Policies are valid — only the
  `total_newton_failures` field is wrong, and downstream analysis didn't
  trust it anyway.
- Touching the FOC math, the Newton solver, or the line search.

---

## Why this matters now

The next planned work is an **inf-horizon state-grid sensitivity sweep**
(3⁴ → 4⁴ → 5⁴, conservative quad at 4 across the board, `max_iter`=100,
`tol`=1e-5). Without real Newton-failure counts, we can't tell whether
"5⁴ converged differently from 4⁴" reflects real policy divergence or
just a different fraction of cells silently hitting Newton's `max_iter`
without flagging it. The sweep verdict needs trustworthy
`total_newton_failures` to be defensible.

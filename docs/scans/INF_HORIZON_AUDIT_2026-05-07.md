# Audit: `inf_horizon_solver.py` (and `predictability_ablation.py`) for drift

**Branch:** `jax-rewrite`
**Date:** 2026-05-07
**Scope:** Phase A audit only (no fixes applied — handoff says STOP and ask).
**Verdict at a glance:**

| Section | Item | Verdict |
| --- | --- | --- |
| A | `inf_horizon_solver` kernel callable signature | **RED** |
| A | `inf_horizon_solver` 3-tuple unpack of kernel return | **RED** (moot — fails before unpack) |
| A | `inf_horizon_solver` builder argument list | GREEN |
| B.1 | Backward-age warm-start across iterations | **YELLOW** (semantically distinct; design call) |
| B.2 | `cell_vmap_chunks` honored | GREEN (would work via builder once A is fixed) |
| B.3 | Newton-iter / backtrack-iter histogram in diagnostics | **YELLOW** |
| B.4 | `verify_runtime_platform()` warnings | n/a (function not present in repo) |
| B.5 | `solve_control` / checkpointing | n/a (out of scope for inf-horizon) |
| C | Mixed-precision plumbing (`gather_precision`) | GREEN |
| D | All 4 ablation systems build | GREEN |
| E | Tiny System I solve via `run_lifecycle_solver` | GREEN |
| F | Existing exercise / verify coverage for inf_horizon | **YELLOW** (zero coverage) |

`inf_horizon_solver.py` will **TypeError on first call** today. Confirmed at runtime — see §A. The ablation systems and the 1-D-state lifecycle path are GREEN.

---

## A. `inf_horizon_solver.py` — kernel call signature

### A.1 — Builder argument list (GREEN)

`_build_per_age_retirement_kernel` is currently defined as:

```python
def _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state, per_is_tensors):
```
[lifecycle/solver.py:1809](../../lifecycle/solver.py#L1809)

`inf_horizon_solver` calls it correctly:

```python
retirement_kernel = _build_per_age_retirement_kernel(
    pcj, mp, solver_config, n_dev, pc.n_z, pc.N_state, per_is_tensors,
)
```
[lifecycle/inf_horizon_solver.py:479-481](../../lifecycle/inf_horizon_solver.py#L479)
[lifecycle/inf_horizon_solver.py:608-610](../../lifecycle/inf_horizon_solver.py#L608)

7-positional match. **GREEN.**

### A.2 — Returned callable signature (RED)

The vmap-only and pmap-only kernels both expose a 5-arg callable today:

```python
def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
    ...
```
[lifecycle/solver.py:1877](../../lifecycle/solver.py#L1877) (pmap)
[lifecycle/solver.py:1964-1970](../../lifecycle/solver.py#L1964) (vmap-only, K=1)
[lifecycle/solver.py:1987-1992](../../lifecycle/solver.py#L1987) (vmap-only, K>1)

`inf_horizon_solver` calls it with 3 args:

```python
_c, _s, _b = retirement_kernel(jnp.asarray(C_old), pension_zero, psi_one)
```
[lifecycle/inf_horizon_solver.py:488](../../lifecycle/inf_horizon_solver.py#L488) (smoke path)

```python
c_new_jnp, s_new_jnp, b_new_jnp = retirement_kernel(c_old_jnp, pension_zero, psi_one)
```
[lifecycle/inf_horizon_solver.py:505](../../lifecycle/inf_horizon_solver.py#L505) (main loop)

```python
c_new, s_new, b_new = retirement_kernel(jnp.asarray(C_old), pension_zero, psi_one)
```
[lifecycle/inf_horizon_solver.py:622](../../lifecycle/inf_horizon_solver.py#L622) (compile_inner_kernel_smoke_test)

**Runtime confirmation** (12-CPU host, pmap path):

```
TypeError: _build_per_age_retirement_kernel_pmap.<locals>.call() missing
2 required positional arguments: 'init_a_s_arr' and 'init_a_b_arr'
```

**Verdict:** RED. Both `run_infinite_horizon_solver` and `compile_inner_kernel_smoke_test` raise on the first kernel call. `inf_horizon_benchmark.ipynb` and any caller of these functions is broken.

### A.3 — Return-tuple unpack (RED, but moot)

Each kernel now returns a 5-tuple `(c, s, b, n_iters_max, n_backtrack_total)`:
[lifecycle/solver.py:1889-1892](../../lifecycle/solver.py#L1889) (pmap)
[lifecycle/solver.py:1971-1977](../../lifecycle/solver.py#L1971) (vmap-only K=1)
[lifecycle/solver.py:1993-1999](../../lifecycle/solver.py#L1993) (vmap-only K>1)

`inf_horizon_solver` unpacks 3 names: `_c, _s, _b = retirement_kernel(...)` and `c_new_jnp, s_new_jnp, b_new_jnp = retirement_kernel(...)`.

This would fail with a "too many values to unpack" `ValueError` even if A.2 were fixed without also adjusting the unpack — moot in practice because A.2 fails first.

---

## B. `inf_horizon_solver.py` — orchestration drift vs `run_lifecycle_solver`

### B.1 — Warm-start across iterations (YELLOW; design call)

`run_lifecycle_solver` threads the older age's converged policy into the next age's kernel call:

```python
if sc.use_backward_age_warm_start:
    init_a_s_arr = S_list[t + 1]
    init_a_b_arr = B_list[t + 1]
else:
    init_a_s_arr = cold_init_a_s_arr
    init_a_b_arr = cold_init_a_b_arr
...
c_t, s_t, b_t, ni_t, nb_t = retirement_kernel(
    c_next_jnp, pension_next, psi_t, init_a_s_arr, init_a_b_arr,
)
```
[lifecycle/solver.py:2498-2509](../../lifecycle/solver.py#L2498)

`inf_horizon_solver` does fixed-point iteration over the SAME age (Bellman fixed-point), not a backward sweep over ages. Conceptually the natural analogue of "previous age's policy" here is "previous iteration's `S_old`, `B_old`", which is already on the host as NumPy arrays at [inf_horizon_solver.py:504-516](../../lifecycle/inf_horizon_solver.py#L504).

The minimal fix to satisfy the kernel signature is to pass `jnp.asarray(S_old)`, `jnp.asarray(B_old)` per iteration. Whether that is the *right* warm-start remains a design call:

- It is consistent with the lifecycle solver's "previous-step's policy" semantics.
- It implies a host->device copy of the share arrays each iteration in addition to the existing `C_old` upload (currently 3× the size — also includes `S_old`, `B_old`).
- An alternative is to seed the kernel with constant `(sc.init_alpha_s, sc.init_alpha_b)` arrays (what `run_lifecycle_solver` does when `use_backward_age_warm_start=False`). That is the cheapest fix but discards an iteration of warm-start info.

**Recommendation in this audit:** flag and ask. If the intent is "match lifecycle solver behavior for `use_backward_age_warm_start=True`," pass the previous iteration's `S_old`/`B_old`. If the intent is "minimal patch, no semantics change," pass the cold `init_alpha_s/b` constants. Either is a 5-line change if no further refactor is wanted. **YELLOW.**

### B.2 — `cell_vmap_chunks` dispatch (GREEN, conditional on A fix)

Cell-axis chunking is implemented inside `_build_per_age_retirement_kernel*` from `sc.cell_vmap_chunks`:
[lifecycle/solver.py:1735](../../lifecycle/solver.py#L1735)
[lifecycle/solver.py:1920](../../lifecycle/solver.py#L1920)

`inf_horizon_solver` constructs the builder with the same `solver_config`, so chunking is honored implicitly. No orchestration-side change needed. **GREEN once A is fixed.**

### B.3 — Newton-iter histogram (YELLOW)

`run_lifecycle_solver` aggregates per-age `n_iters_max` and `n_backtrack_total` into `diagnostics["newton_iter_histogram"]` / `diagnostics["backtrack_iter_histogram"]`:
[lifecycle/solver.py:2535-2536](../../lifecycle/solver.py#L2535) (per-age capture)
[lifecycle/solver.py:2624-2628](../../lifecycle/solver.py#L2624) (aggregation)

`inf_horizon_solver`'s `_build_diagnostics` does NOT include either histogram. After the A.2 fix, two extra arrays will be returned by every kernel call; the iteration loop should accumulate them per-iteration (across the fixed-point) and the diagnostics dict should expose at minimum a `newton_iter_p99` and `n_backtrack_total_p99`.

The simplest patch: append `int(ni_max)` per iteration to a list and report `np.percentile([...], 99)` in the final diag dict. Ten lines.

**YELLOW — feature parity gap, not a correctness break.**

### B.4 — `verify_runtime_platform()` warnings (n/a)

The handoff names `verify_runtime_platform()` as something `run_lifecycle_solver` calls. **The function does not exist anywhere in the repo as of this audit.** The only platform reporting `run_lifecycle_solver` does is the verbose-1 banner at [lifecycle/solver.py:2289-2300](../../lifecycle/solver.py#L2289). No drift item to report.

### B.5 — `solve_control` / checkpointing (n/a)

`SolveControl` is a finite-horizon abstraction (`youngest_age_to_solve`, age-indexed checkpointing). It has no analog for a stationary-Bellman fixed-point loop, and `inf_horizon_solver` correctly does not consume it. Out of scope.

---

## C. `inf_horizon_solver.py` — mixed-precision plumbing (GREEN)

`gather_precision` is read from the `SolverConfig` *inside* each kernel builder via `_resolve_gather_dtype(sc)`:
[lifecycle/solver.py:1830](../../lifecycle/solver.py#L1830) (pmap path)
[lifecycle/solver.py:1907](../../lifecycle/solver.py#L1907) (vmap-only path)

`inf_horizon_solver` passes the same `solver_config` (default or user-supplied) into the builder, so `f32` mode is honored without any orchestration change.

`mp = ModelParams(...)` constructed at [inf_horizon_solver.py:470-476](../../lifecycle/inf_horizon_solver.py#L470) matches the 5-field namedtuple at [lifecycle/solver.py:1490-1492](../../lifecycle/solver.py#L1490) (`gamma, beta, b_bar, delta, rho`). With `b_bar=0.0` the bequest term vanishes, which is the intended infinite-horizon semantics ("no mortality, no bequest"). **GREEN.**

---

## D. `predictability_ablation.py` — System I-IV buildability (GREEN)

The handoff's example invocation referenced `SYSTEM_I/II/III/IV, build_system_disc_config, build_system_var_config`. Those names do not exist in the module — the actual exported entrypoint is `prepare_predictability_system(system_code, *, csv_path, disc_config_template)` at [lifecycle/predictability_ablation.py:237](../../lifecycle/predictability_ablation.py#L237), which takes the canonical 4-D `disc_config_template` and projects it onto each system's state cardinality.

Smoke-build (12-CPU host, tiny 4-D template `(2,3,2,3)` matching `verify_smoke.py`):

| System | state_names | N_state | Result |
| --- | --- | --- | --- |
| I | `('rtb',)` | 2 | ok |
| II | `('rtb', 'y_1')` | 6 | ok |
| III | `('rtb', 'spr', 'y_1')` | 18 | ok |
| IV | `('cy', 'spr', 'rtb', 'y_1')` | 36 | ok |

All four `build_model` + `build_precompute` calls returned without raising. The state-axis projection from the 4-D template down to (1,), (2,), (3,) state vectors works as designed. **GREEN.**

---

## E. Tiny-solve System I via `run_lifecycle_solver` (GREEN)

Config: `prepare_predictability_system("I", ...)` with the same tiny 4-D template above (projected to `(2,)` for System I), `CANONICAL_SOLVER._replace(max_iter=30, max_iter_unconstrained=30)`, `SolveControl(youngest_age_to_solve=95)`.

```
LIFECYCLE PORTFOLIO SOLVER  (JAX, EGM + 2D Newton)
  Devices: 12 (CPU); pmap+vmap path
  Discretization: state_grid_sizes=(2,), n_z=3, n_w=12  -> 1-D state
  Solve control: youngest_age_to_solve=95  (4 retirement ages: 95-99)

  ages solved: 5/78
  alpha_s range: [0.518, 0.789]
  alpha_b range: [0.136, 0.384]
  NaN check: 0/0/0
  wall: 21.5s
```

Solver completes on a 1-D state. The rtb-as-state migration did NOT break axis-cardinality flexibility. Newton iterations did saturate the aggressive `max_iter=30` cap (this matches `verify_smoke.py`-style smokes; max_iter is set tight on purpose to keep wall time low — not a correctness signal at this granularity). **GREEN.**

---

## F. Exercise / verify-script coverage (YELLOW)

`grep -l "inf_horizon\|run_infinite_horizon\|prepare_predictability_system\|predictability_ablation"` over `verify_*.py` and `scripts/`:

- `verify_benchmark_bundle.py`: only mentions `predictability_ablation` as a metadata-snapshot string for the saved bundle — does NOT call into the module.
- No `verify_*.py` invokes `run_infinite_horizon_solver`, `compile_inner_kernel_smoke_test`, or `prepare_predictability_system`.
- `inf_horizon_benchmark.ipynb` and `main.ipynb` are the only consumers; both are notebooks not exercised by the verify path.

**This audit report is the first regression signal that `inf_horizon_solver` is broken.** Without a `verify_*.py` exercising it, every solver-side refactor since the kernel-signature change has been silently breaking it.

**Recommendation (out of scope for the fix bundle):** add a new `verify_inf_horizon.py` that runs `compile_inner_kernel_smoke_test` plus 5-10 fixed-point iterations on the tiny config used here. That would make the next signature drift fail in the verify harness, not in `inf_horizon_benchmark.ipynb` post-publication. Don't write it as part of the fix bundle (scope creep) — call it out as a follow-up handoff after the fixes land.

---

## Phase B (proposed) — fix shape and ordering

Listed as a recommendation only. **No fixes applied. Handoff says STOP and ask.**

Suggested commit ladder, smallest blast radius first:

1. **Minimal kernel-signature patch.** Update both `retirement_kernel(...)` call sites to pass `init_a_s_arr` and `init_a_b_arr`, and unpack the 5-tuple return.
   - Choice point (B.1): pass `jnp.asarray(S_old)/B_old` (tracks each iteration's policy) OR pass constant `init_alpha_s/init_alpha_b` arrays. **Ask user.**
   - The unused `n_iters_max` / `n_backtrack_total` outputs can be discarded with `_, _` until step 3.
   - Validation: `compile_inner_kernel_smoke_test` runs without TypeError; tiny 5-iteration loop converges.
2. **`verify_smoke.py` regression check.** No code change to the smoke; just confirm the main lifecycle path is untouched.
3. **Newton-iter histogram into diagnostics.** Capture `ni_t/nb_t` per iteration; expose `newton_iter_p99`, `n_backtrack_total_p99` in `diagnostics`. Optional but cheap and aligns with the lifecycle solver's diagnostics surface.
4. **(Out of scope) Add `verify_inf_horizon.py`** to lock in regression coverage. This is its own handoff.

Each commit uses the shape suggested by the handoff: one-line summary, 2-3 sentences explaining drift item + fix + validation, "No math change" or explicit reason if math affected.

**Math-change risk:** B.1 is the only place where math could shift. The current 3-arg call structurally cannot have used warm-start init at all (the kernel had no path for that until init_a_s_arr was added), so any choice in step 1 is a re-introduction of an existing-elsewhere semantics rather than a math change *to inf-horizon*. Worth saying explicitly in the commit message.

---

## Out-of-scope items surfaced during the audit (flag only)

- **Refactoring `inf_horizon_solver` to call `run_lifecycle_solver`** with `terminal_age=youngest_age_to_solve` and re-use its orchestration. Larger structural change, separate handoff.
- **Backward-age warm-start choice across fixed-point iterations** is a design call, not a drift item — flagged in B.1.
- **`verify_inf_horizon.py`** — needed to prevent recurrence, but should be its own handoff after fixes land.
- **`inf_horizon_benchmark.ipynb`** has not been re-executed against current kernels and likely emits stale plots. Not a code path to fix here, but the user should know it cannot be re-run today.

---

## STOP — awaiting approval before Phase B

Per handoff: report sent; do not implement until user approves scope. Specifically requesting input on:

1. **B.1 warm-start semantics:** previous-iteration policy vs cold constants?
2. **B.3 Newton-iter histogram:** include in this fix bundle, or defer?
3. **Any objection to the commit ladder** in the proposed Phase B?

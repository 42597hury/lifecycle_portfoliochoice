# Handoff: Convert Newton `lax.while_loop` to `lax.fori_loop` + Mask (GPU)

**Branch:** `jax-rewrite`
**Status when this doc was written:** [newton_2d_with_line_search](../../lifecycle/solver.py#L304) uses two `lax.while_loop` calls — an outer Newton iteration loop (line 421) and an inner backtracking line-search loop (line 407). Both have data-dependent termination, which is the right pattern on CPU but a known performance footgun on GPU under `vmap`. This change replaces them with `lax.fori_loop` + masked-update bodies.

**Target:** GPU. The branch is committing to GPU as the primary deployment target ("GPU build path is the way"). On CPU this change is a small regression (cells no longer benefit from early termination); a SolverConfig flag `use_fori_newton` keeps the old `while_loop` path available for CPU smoke and dev work.

**Expected payoff:** **1.3-2× total wall** on GPU (Newton portion only sees ~1.5-2.5×; the rest of the per-age work is FOC eval + EGM scan + gather, which this change doesn't touch).

---

## 1. Why this matters on GPU specifically

`lax.while_loop` under `vmap` is implemented as a single `while_loop` whose condition is "ANY element of the batch still needs to iterate." On GPU:

- All cells in a vmap batch execute the body in lockstep (warps run SIMT). Slow-converging cells force the whole batch to keep iterating.
- Each iteration is a kernel launch. Variable iteration counts mean XLA can't pre-schedule the dispatch graph; the GPU stalls between iters waiting for the launch decision.
- With backward-age warm-start (the other handoff), iter variance shrinks but doesn't disappear — cold cells at the boundary still take more.

`lax.fori_loop(0, MAX_ITER, body, init)` runs **exactly** `MAX_ITER` body invocations. Inside the body we mask out the update for cells that have already converged (`jnp.where(active, new, old)`). The "wasted" iters on converged cells run identical math to active cells, but produce identity updates — XLA-CUDA can fuse the loop into a single kernel and dispatch it deterministically.

Trade-off: every cell now runs `MAX_ITER` iters' worth of arithmetic even if it converged in 3. With sensible `MAX_ITER` (3-5× the 99th-percentile observed iter count), the wasted FLOPs are dwarfed by the dispatch + warp-utilization wins.

**On CPU:** vmap batches run sequentially across cells in many cases, so the early-termination of `while_loop` is real savings. fori_loop+mask is a 2-5× regression on CPU. Hence the SolverConfig flag.

---

## 2. Scope

### In scope

- Replace the **outer** `lax.while_loop` ([solver.py:421](../../lifecycle/solver.py#L421)) with a `lax.fori_loop` whose body masks updates for converged or line-search-failed cells.
- Replace the **inner** backtracking `lax.while_loop` ([solver.py:407](../../lifecycle/solver.py#L407)) with a `lax.fori_loop` whose body masks updates after a successful step is found.
- Preserve the existing iteration-count diagnostic (number of iters actually used per cell, not the loop's max).
- Add `use_fori_newton: bool = True` to `SolverConfig`. When `False`, fall back to today's `while_loop` implementation.
- Preserve exit codes (`EC_INTERIOR`, `EC_NEWTON_FAIL`) and the returned `(a_s, a_b, e, exit_code, err_norm, n_iter)` tuple — interface is unchanged.

### Out of scope

- Tuning `max_iter` or `max_backtrack_iter` defaults. Those are runtime SolverConfig values; pick separately based on backward-age warm-start adoption and observed convergence stats.
- Vectorising across the savings dimension differently. The savings vmap is already in place from the warm-start kill commit.
- Replacing the FOC eval (`foc_fn`) internals. Untouched.
- Changing the Newton math itself (Jacobian, gradient fallback, line-search rule). The conversion is purely loop-machinery.

---

## 3. Implementation

### 3.1 `SolverConfig` flag

In [lifecycle/model.py:135-145](../../lifecycle/model.py#L135-L145), add a boolean field next to the Newton iteration knobs:

```python
class SolverConfig(NamedTuple):
    ...
    # --- Newton iteration ---
    tol: float = 1e-7
    max_iter: int = 5000
    max_iter_unconstrained: int = 5000
    use_fori_newton: bool = True   # NEW: fori_loop+mask on GPU; set False for while_loop on CPU
    ...
```

### 3.2 Outer Newton loop conversion

The current state tuple ([solver.py:325-331](../../lifecycle/solver.py#L325-L331)) is:

```python
init_state = (
    init_a_s, init_a_b,
    fs0, fb0, Jss0, Jbb0, Jsb0,
    e0, err0,
    jnp.int32(0),                 # k = iter counter (=== n_iter on exit)
    err0 < tol * scale,           # done flag
)
```

Replace `done` (which conflates "converged" and "line search failed") with two explicit flags, and split `k` into `n_total_steps` (always increments, used as the fori loop var implicitly) and `n_iters_used` (only increments when active):

```python
init_state = (
    init_a_s, init_a_b,
    fs0, fb0, Jss0, Jbb0, Jsb0,
    e0, err0,
    jnp.int32(0),                 # n_iters_used (active iters only)
    err0 < tol * scale,           # converged
    jnp.bool_(False),             # ls_failed (line search couldn't improve)
)
```

The body becomes:

```python
def fori_body(_i, state):
    a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, n_used, converged, ls_failed = state
    is_active = jnp.logical_not(jnp.logical_or(converged, ls_failed))

    # ---- Compute proposed Newton step (same math as today, lines 340-357) ----
    det = Jss * Jbb - Jsb * Jsb
    is_singular = jnp.abs(det) < singular_det
    grad_norm = err + grad_denom_eps
    step_s_grad = grad_step_size * fs / grad_norm
    step_b_grad = grad_step_size * fb / grad_norm
    inv_d = 1.0 / jnp.where(is_singular, 1.0, det)
    step_s_newton = -(Jbb * fs - Jsb * fb) * inv_d
    step_b_newton = -(-Jsb * fs + Jss * fb) * inv_d
    step_s = jnp.where(is_singular, step_s_grad, step_s_newton)
    step_b = jnp.where(is_singular, step_b_grad, step_b_newton)
    slen = jnp.sqrt(step_s * step_s + step_b * step_b)
    cap = jnp.minimum(1.0, line_search_max_step / jnp.where(slen > 0.0, slen, 1.0))
    step_s = step_s * cap
    step_b = step_b * cap

    # ---- Try alpha=1 + backtracking (now also fori_loop, see §3.3) ----
    a_s_full = a_s + step_s
    a_b_full = a_b + step_b
    fs_f, fb_f, Jss_f, Jbb_f, Jsb_f, e_f = foc_fn(a_s_full, a_b_full)
    err_f = jnp.sqrt(fs_f * fs_f + fb_f * fb_f)
    full_improves = err_f < err
    new_a_s, new_a_b, new_fs, new_fb, new_Jss, new_Jbb, new_Jsb, new_e, new_err, found_any = (
        _backtracking_fori(
            a_s, a_b, step_s, step_b,
            a_s_full, a_b_full,
            fs_f, fb_f, Jss_f, Jbb_f, Jsb_f, e_f, err_f, full_improves,
            fs, fb, Jss, Jbb, Jsb, e, err,
            foc_fn, max_backtrack_iter,
        )
    )

    # ---- Compute new convergence/failure flags ----
    new_converged = new_err < tol * scale
    new_ls_failed = jnp.logical_not(found_any)

    # ---- Mask: if not active, hold all fields constant ----
    return (
        jnp.where(is_active, new_a_s, a_s),
        jnp.where(is_active, new_a_b, a_b),
        jnp.where(is_active, new_fs, fs),
        jnp.where(is_active, new_fb, fb),
        jnp.where(is_active, new_Jss, Jss),
        jnp.where(is_active, new_Jbb, Jbb),
        jnp.where(is_active, new_Jsb, Jsb),
        jnp.where(is_active, new_e, e),
        jnp.where(is_active, new_err, err),
        n_used + jnp.where(is_active, jnp.int32(1), jnp.int32(0)),
        jnp.where(is_active, new_converged, converged),
        jnp.where(is_active, new_ls_failed, ls_failed),
    )

final = lax.fori_loop(0, max_iter, fori_body, init_state)
a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, n_used, converged, ls_failed = final
exit_code = jnp.where(converged, EC_INTERIOR, EC_NEWTON_FAIL)
return a_s, a_b, e, exit_code, err / scale, n_used
```

### 3.3 Inner backtracking loop conversion

Refactor the current inline `bt_init`/`bt_body`/`bt_cond`/`while_loop` ([solver.py:366-409](../../lifecycle/solver.py#L366-L409)) into a helper `_backtracking_fori`:

```python
def _backtracking_fori(
    a_s, a_b, step_s, step_b,
    a_s_full, a_b_full,
    fs_f, fb_f, Jss_f, Jbb_f, Jsb_f, e_f, err_f, full_improves,
    fs_old, fb_old, Jss_old, Jbb_old, Jsb_old, e_old, err_old,
    foc_fn, max_backtrack_iter,
):
    # Best-so-far init: full-step result if it improved, else current state.
    init = (
        jnp.float64(1.0),                     # alpha
        jnp.where(full_improves, a_s_full, a_s),
        jnp.where(full_improves, a_b_full, a_b),
        jnp.where(full_improves, fs_f, fs_old),
        jnp.where(full_improves, fb_f, fb_old),
        jnp.where(full_improves, Jss_f, Jss_old),
        jnp.where(full_improves, Jbb_f, Jbb_old),
        jnp.where(full_improves, Jsb_f, Jsb_old),
        jnp.where(full_improves, e_f, e_old),
        jnp.where(full_improves, err_f, err_old),
        full_improves,                         # found
    )

    def bt_body(_i, bt_state):
        alpha, a_s_b, a_b_b, fs_b, fb_b, Jss_b, Jbb_b, Jsb_b, e_b, err_b, found = bt_state
        # Halve step (always; cheap and the math runs anyway).
        new_alpha = alpha * 0.5
        a_s_t = a_s + new_alpha * step_s
        a_b_t = a_b + new_alpha * step_b
        fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = foc_fn(a_s_t, a_b_t)
        err_t = jnp.sqrt(fs_t * fs_t + fb_t * fb_t)
        # Update best-so-far ONLY if (a) we haven't found a step yet AND (b) this trial improves.
        # If found is True, hold everything.
        improved_now = jnp.logical_and(jnp.logical_not(found), err_t < err_old)
        return (
            jnp.where(found, alpha, new_alpha),
            jnp.where(improved_now, a_s_t, a_s_b),
            jnp.where(improved_now, a_b_t, a_b_b),
            jnp.where(improved_now, fs_t, fs_b),
            jnp.where(improved_now, fb_t, fb_b),
            jnp.where(improved_now, Jss_t, Jss_b),
            jnp.where(improved_now, Jbb_t, Jbb_b),
            jnp.where(improved_now, Jsb_t, Jsb_b),
            jnp.where(improved_now, e_t, e_b),
            jnp.where(improved_now, err_t, err_b),
            jnp.logical_or(found, improved_now),
        )

    final = lax.fori_loop(0, max_backtrack_iter, bt_body, init)
    _alpha, a_s_b, a_b_b, fs_b, fb_b, Jss_b, Jbb_b, Jsb_b, e_b, err_b, found = final
    return a_s_b, a_b_b, fs_b, fb_b, Jss_b, Jbb_b, Jsb_b, e_b, err_b, found
```

**Key subtlety**: the original `bt_cond` exits the loop as soon as a successful step is found — it does NOT keep halving once `found=True`. The masked version replicates this by gating the update on `improved_now = NOT found AND err_t < err_old`. After the first success, `found=True` and subsequent iters compute `foc_fn` on a halved alpha but mask out the result. **This wastes some FOC evals**, but the inner loop is bounded at `max_backtrack_iter=10` (default), so the waste is small.

### 3.4 Dispatch on the SolverConfig flag

Wrap both implementations under `newton_2d_with_line_search` with a Python-level branch (the flag is a static boolean known at build time, so the branch happens during JIT trace, not at runtime):

```python
def newton_2d_with_line_search(
    foc_fn,
    init_a_s, init_a_b, scale,
    tol, max_iter, max_backtrack_iter,
    line_search_max_step, singular_det,
    grad_step_size, grad_denom_eps,
    use_fori=True,                              # NEW
):
    if use_fori:
        return _newton_fori(
            foc_fn, init_a_s, init_a_b, scale,
            tol, max_iter, max_backtrack_iter,
            line_search_max_step, singular_det,
            grad_step_size, grad_denom_eps,
        )
    else:
        return _newton_while(
            foc_fn, init_a_s, init_a_b, scale,
            tol, max_iter, max_backtrack_iter,
            line_search_max_step, singular_det,
            grad_step_size, grad_denom_eps,
        )
```

`_newton_while` is the current body of `newton_2d_with_line_search` extracted verbatim; `_newton_fori` is the new implementation from §3.2 + §3.3.

### 3.5 Plumbing the flag into the kernels

The three kernel builders ([_build_per_age_terminal_kernel](../../lifecycle/solver.py#L1125), `_build_per_age_retirement_kernel`, `_build_per_age_working_kernel`) currently call `newton_2d_with_line_search` indirectly via `_egm_scan_cell` → the cell-level `_solve_*_at_cell`. The `sc` (SolverConfig) is already in scope at build time.

Pick one of two threading patterns; **prefer (a)**:

(a) **Bind `use_fori` at builder time** by closing over the bool in the call site to `newton_2d_with_line_search`. This means the chosen path is baked into the JIT'd kernel (no runtime branch). Since the SolverConfig is per-run, this is correct.

(b) Pass the flag through every layer as a runtime arg. More invasive; no benefit.

For (a), pass the flag down via the existing static-args bundle in each kernel builder ([solver.py:1132-1135](../../lifecycle/solver.py#L1132-L1135)) and read it at the `_egm_scan_cell` → `newton_2d_with_line_search` call site. About 4 call sites to update.

---

## 4. Edge cases / gotchas

### 4.1 `foc_fn` runs unconditionally inside the masked body

Every cell calls `foc_fn` at every fori iter, even after convergence. `foc_fn` is the bulk of arithmetic per Newton step (trilinear interp at `n_state_quad × n_ret_quad ≈ 3600` quad points, plus 6 matrix entries). The "wasted" foc_fn calls on converged cells is the entire trade-off — it's accepted because GPU SIMT is in lockstep anyway. **Do not** try to skip foc_fn for converged cells via `lax.cond` — that breaks fusion and reintroduces the dispatch variance.

### 4.2 `n_iters_used` accuracy

The diagnostic `n_iter` returned by `newton_2d_with_line_search` should be the actual count of useful iters, not the fori_loop trip count. The `n_used + jnp.where(is_active, 1, 0)` pattern in §3.2 handles this. The retirement/working/terminal kernels already pipe this into `diagnostics["age_max_foc"]` etc. — verify that flow is unchanged.

### 4.3 `max_iter` becomes the wall cost

In the `while_loop` version, a cell that converges in 3 iters does 3 iters of work. In the `fori_loop` version, it does `max_iter` iters of work (mostly identity-masked but still computing foc_fn). Concrete implication:

- Smoke uses `max_iter=100`. Smoke wall on GPU: barely changes (small problem, dispatch dominates anyway). Smoke wall on CPU **with `use_fori_newton=True`**: 2-5× regression. **Set `use_fori_newton=False` in `verify_smoke.py` if running on CPU.**
- Benchmark uses `max_iter=400`. With `use_fori_newton=True` on GPU: this is the wall cost. Tune `max_iter` to be ~3-5× the 99th-percentile observed iter count. With backward-age warm-start, that's ~30-50; without, ~100-200.
- Canonical default `max_iter=5000` is **catastrophic** under fori_loop+mask. Override to ≤ 200 in any production run that uses `use_fori_newton=True`.

**Add this warning to the `SolverConfig.use_fori_newton` docstring**: "When True, the Newton loop runs `max_iter` iters unconditionally (mask-based early termination). Tune `max_iter` accordingly — the canonical 5000 default is unsuitable for fori_loop mode."

### 4.4 Numerical equivalence

The fori path should produce **bit-identical** results to the while path **modulo Newton tolerance** — same math, same convergence criterion, same line-search rule. Slight differences (~1e-12 to 1e-10) are expected from order-of-operations under XLA fusion; differences > tol are bugs.

The `improved_now = NOT found AND err_t < err_old` gate in `_backtracking_fori` is the ONE place where the math could subtly diverge from the while_loop version. Specifically: the while_loop version exits the backtracking loop on the first success and never evaluates more halvings. The fori version evaluates all halvings but masks them out. **This must produce the same first-success result** because subsequent halvings have `improved_now=False` and the state is held. Verify with a unit test if possible (a simple foc_fn fixture) — though smoke equivalence is the practical check.

### 4.5 Static vs dynamic loop bounds

`lax.fori_loop(0, max_iter, body, init)` accepts `max_iter` as either static (Python int) or dynamic (traced int). Both compile fine. With static `max_iter` XLA can unroll for very small counts (<10), but at the relevant range (50-400) it always emits a tight loop. **Pass `max_iter` as a static value** (it's already in `SolverConfig`, propagated as a Python int through the kernel builders) so XLA gets the strongest hint.

### 4.6 The flag is a build-time choice

Once `use_fori_newton` is baked into a JIT'd kernel, switching it requires a recompile. Don't put the toggle inside a hot loop. The intended pattern is: pick at `run_lifecycle_solver` start, build all four kernels with that choice, run.

### 4.7 Compatibility with backward-age warm-start

Independent change; no interaction beyond what's already covered. Warm-start lowers iter counts → lower waste under fori_loop → fori_loop is *more* attractive after warm-start lands. If the two handoffs land in the same series of commits, do warm-start first.

---

## 5. Verification

### 5.1 Numerical equivalence (must pass before commit)

Run smoke twice — once with `use_fori_newton=False` (today's behavior), once with `True`. Both should produce identical `alpha_s` and `alpha_b` ranges to within 1e-9:

```bash
# Run 1
python -c "
from configs._canonical import CANONICAL_SOLVER
import json
sc = CANONICAL_SOLVER._replace(use_fori_newton=False, max_iter=100)
# ... run smoke ...
"

# Run 2: same with use_fori_newton=True
```

If alpha ranges differ by more than ~1e-9 (typical Newton tolerance noise), there's a bug — most likely in §3.3's `improved_now` gating or in the masked state propagation in §3.2.

### 5.2 Iter-count diagnostic

Confirm `diag.get("total_newton_iters")` (or the equivalent in `_build_diagnostics`) reports a sensible number under the fori path — should match within ±1 of the while path's count for cells that converged, since the mask doesn't increment `n_iters_used` after convergence.

### 5.3 GPU performance check (when GPU access lands)

After this change is in **and** backward-age warm-start has been applied, run the canonical 9×9×9 retirement-only benchmark on A100/H100 with both flag values:

| `use_fori_newton` | Expected wall on A100 | Compared to today's CPU baseline (1342 s) |
|---|---|---|
| False (while_loop) | ~250-500 s | 2.7-5× faster |
| True (fori_loop)   | ~150-300 s | 4.5-9× faster |

If the two are within 20% of each other, GPU dispatch isn't the bottleneck — investigate elsewhere (gather pattern, FOC eval). If `use_fori_newton=True` is **slower** than `False` on GPU, the chosen `max_iter` is much too high relative to typical iter counts; halve and retry.

### 5.4 CPU regression check

Run smoke locally with `use_fori_newton=True` (CPU). Wall should be 2-5× slower than `use_fori_newton=False`. Document the regression in the commit message — confirms behavior and warns future-CPU-users.

---

## 6. Files touched

| File | Change | Lines |
|---|---|---|
| [lifecycle/model.py](../../lifecycle/model.py) | Add `use_fori_newton: bool = True` field to `SolverConfig` with docstring | ~3 |
| [lifecycle/solver.py](../../lifecycle/solver.py) | Split `newton_2d_with_line_search` into dispatcher + `_newton_while` (existing body extracted) + `_newton_fori` (new); add `_backtracking_fori` helper. Update three kernel builders to pass the flag through. | ~250-300 net (mostly the new fori implementation; the while body is moved verbatim into `_newton_while`) |

No other files. No test additions strictly required, but the smoke-level toggle test in §5.1 should be run.

---

## 7. Implementation checklist (for the agent)

- [ ] Add `use_fori_newton: bool = True` to [SolverConfig](../../lifecycle/model.py#L126) with the docstring warning from §4.3.
- [ ] In [solver.py](../../lifecycle/solver.py), rename the current body of `newton_2d_with_line_search` to `_newton_while` (verbatim move, no edits).
- [ ] Add `_backtracking_fori` helper per §3.3.
- [ ] Add `_newton_fori` per §3.2.
- [ ] Replace `newton_2d_with_line_search` with the dispatcher from §3.4.
- [ ] Update the three kernel builders ([_build_per_age_terminal_kernel](../../lifecycle/solver.py#L1125), `_build_per_age_retirement_kernel`, `_build_per_age_working_kernel`) to pass `use_fori=sc.use_fori_newton` through to the cell solvers' Newton call. The cleanest place is in the `static` tuple in each builder.
- [ ] In `_solve_terminal_at_i_s`, `_solve_retirement_at_cell`, `_solve_working_at_cell`, accept `use_fori` and forward to `newton_2d_with_line_search` (or read from a shared static container).
- [ ] Run §5.1 numerical equivalence on smoke locally. Both flag values must produce alpha ranges within 1e-9 (ideally bit-identical).
- [ ] Run §5.4 CPU regression check; record actual wall difference in commit message.
- [ ] Single commit, message:
  ```
  newton: lax.fori_loop + mask path for GPU SIMT efficiency
  
  - SolverConfig.use_fori_newton (default True) selects between today's
    while_loop path and a fori_loop+mask body that runs max_iter iters
    unconditionally with masked updates after convergence.
  - Outer Newton loop and inner backtracking line search both converted.
  - Numerical equivalence verified on smoke: alpha ranges identical to
    1e-12 between the two paths.
  - CPU regression of N×: noted in docstring; pass use_fori_newton=False
    for CPU-only runs (smoke, dev).
  - Pairs with backward-age warm-start: warm-start tightens iter
    variance, making the fori path's wasted iters cheaper.
  
  GPU benchmark deferred until P-quota lands.
  ```
- [ ] Push to `jax-rewrite`. No PR needed unless reviewer requests.

---

## 8. Performance expectations (record in commit + AWS_TRIAL_JAX.md)

| Path | Hardware | Wall vs current | Notes |
|---|---|---|---|
| `use_fori_newton=False` | CPU (hpc8a 192-vCPU) | unchanged | Identity to today |
| `use_fori_newton=True` | CPU (hpc8a) | 2-5× **slower** | Don't run this — for completeness |
| `use_fori_newton=False` | GPU (A100) | 2-3× faster than CPU | Limited by while_loop variance |
| `use_fori_newton=True` | GPU (A100) | **4-7× faster than CPU** | Target operating point |

Combined with backward-age warm-start (the other handoff), expect ~5-10× over today's CPU canonical on A100. Numbers above assume `max_iter=80-150` (tuned for backward-age warm-start) and the canonical 9×9×9 config.

---

## 9. Out of scope / future work

- **Newton math improvements** (BFGS, trust region, etc.): out of scope; this is purely loop-machinery.
- **Mixed-precision FOC eval**: separate optimization, complementary to this one.
- **Per-cell adaptive `max_iter`**: theoretically possible (dynamic max_iter via `lax.dynamic_slice`-style indexing) but breaks XLA fusion. Skip.
- **Removing `_newton_while` entirely**: defer until GPU is the only deployment target. The flag is cheap insurance for now.

# Handoff: Backward-Age Warm-Start for the JAX Solver

**Branch:** `jax-rewrite`
**Status when this doc was written:** the in-savings warm-start has just been removed (commit pending). Every savings point inside `_egm_scan_cell` now starts Newton from the canonical scalar `(SolverConfig.init_alpha_s, SolverConfig.init_alpha_b)` cold init, and the savings sweep is `vmap`-batched instead of `lax.scan`-sequential. This pattern is the prerequisite for the change described here.

**Goal.** Restore an informative Newton init at every cell *without* reintroducing the savings-axis sequential dependency. The init source becomes the previous (older) age's converged policy at the same `(z, state)` cell — i.e. `α_s(t+1, z, state, …)` seeds `α_s(t, z, state, …)`. Backward induction already runs ages T → 0, so this data is in hand by construction.

---

## 1. Why this is the right replacement

- **Smoothness in age dominates smoothness in savings.** Lifecycle policies change very little from age `t+1` to age `t` (especially in retirement: deterministic income, just one less period of utility). Across savings, policies vary more strongly. A converged α at the *same cell, one age newer* is closer to the optimum than the canonical scalar (0.1, 0.4) for every savings point — typically within sub-percent of converged.
- **No sequential dependency on the savings axis.** The age loop is already inherently sequential (age `t` reads `t+1`'s continuation `V`); this change just adds two more arrays (`S_prev`, `B_prev`) to the data already threaded between ages. The savings vmap inside `_egm_scan_cell` stays fully parallel.
- **Pure-JAX.** No mutation, no Python control flow over traced values, no new JIT-cache keys. Same shape signatures.

Expected payoff: Newton converges in ~2–4 iters per cell (vs ~5–15 cold from the canonical init). This compounds over the ~33 retirement ages and ~46 working ages of a canonical run.

---

## 2. Design — pick one variant before implementing

Two viable variants. Pick **A** unless profiling proves it's leaving meaningful Newton iters on the table; B is only worth the wiring cost if A fails to close the iteration-count gap to within 2× of the original (warm-on-savings) baseline.

### Variant A — single scalar init per `(z, state)` cell  *(recommended starting point)*

For each cell `(z_idx, i_s)` at age `t`, the entire savings vmap uses **one** `(α_s_init, α_b_init)` pair, sourced from age `t+1`'s converged policy at that same cell. Concretely, gather a representative wealth slice:

```python
init_a_s = S_prev[z_idx, i_s, w_ref_idx]
init_a_b = B_prev[z_idx, i_s, w_ref_idx]
```

`w_ref_idx = n_w // 2` (mid-grid wealth) is a safe default — for canonical configs the policy at mid-wealth is representative of the whole cell.

- **Pros:** trivial to thread, ~46 MB extra RAM at production-9×9×9 (`n_z × N_state × n_w × 2 × 8 B` ≈ 23 MB per array), identical pmap pattern as today.
- **Cons:** init is uniform across the savings sweep within a cell. Newton at edge savings points (very small / very large `s`) may take 1–2 extra iters than it would with a per-`s` init.

### Variant B — per-savings-point init from age `t+1`'s converged EGM α-grid

`_egm_scan_cell` already produces converged `a_s_egm`, `a_b_egm` of shape `(n_s + 1,)` per cell — these are α at the **savings grid points** for that cell. Plumb them out of every per-age kernel into a stash array of shape `(n_z, N_state, n_s + 1)` per α-component, pass that stash into the next age's kernel as `init_a_s_grid[i_z, i_s, :]`, and feed it as the per-savings init inside `vmap(per_savings_point)`.

- **Pros:** the closest possible init for each savings point — semantically equivalent to the original within-cell warm-start, but propagated *across age* instead of along savings.
- **Cons:** requires modifying every kernel's return contract (currently `(c, S, B)` only over wealth grid; B adds an α-egm-grid output). ~46 MB extra RAM at production-9×9×9, same magnitude as A.
- **When to choose B over A:** only after measuring that A leaves >2× extra Newton iters compared to the warm-on-savings baseline.

**Recommendation: implement A first, ship it, measure, only escalate to B if needed.** The rest of this doc assumes Variant A.

---

## 3. Surfaces that need to change (file-by-file)

### 3.1 `lifecycle/solver.py` — `_egm_scan_cell` (lines 816–870)

**No change in body.** The function already accepts scalar `init_a_s, init_a_b`; the `per_savings_point(s_val)` closure already broadcasts them across the savings vmap. The only thing that changes is *which* scalars get passed in by the callers. Optionally update the docstring to say "init may be a per-cell warm-started scalar from age t+1".

### 3.2 `_solve_terminal_at_i_s`, `_solve_retirement_at_cell`, `_solve_working_at_cell` (lines 890–1025)

Already accept `init_a_s, init_a_b` as scalar args. **No signature change.** Just verify the call sites in section 3.3 thread the per-cell init in correctly.

### 3.3 The three pmap'd kernel builders (lines 1125–1343)

**This is where the actual wiring lives.** Currently each builder closes over `init_a_s = jnp.float64(sc.init_alpha_s)` at build time and bakes it into the pmap'd inner function. Change:

- **Move `init_a_s`, `init_a_b` from build-time closure to runtime args.** Add them to the pmap signature with `in_axes=0` (so each device gets its slice) and the call entrypoint accepts the full `(n_z, N_state, n_w)` arrays.
- **Add the per-cell gather inside `per_cell` / `per_i_s`:**
  - Retirement / working / boundary: `init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]` (and similarly for `a_b`).
  - Terminal: `init_a_s_cell = init_a_s_arr[i_s, w_ref_idx]` (no z dim — see §4).
- **The pmap input arrays** are pre-padded and reshaped exactly like today's `z_pm`, `is_pm`. The init arrays don't need reshaping because they're indexed *inside* the inner function via `[z_idx, i_s, w_ref_idx]` — pass them with `in_axes=None` (broadcast across devices). Keep this in mind: the per-device padding logic does NOT apply to the init arrays.

Pseudocode for the retirement kernel signature change:

```python
@partial(pmap, in_axes=(0, 0, None, None, None, None, None))
def per_dev_solve(
    z_block, is_block, c_next, pension_next_by_z, psi_per_z,
    init_a_s_arr, init_a_b_arr,  # NEW: full (n_z, N_state, n_w) arrays
):
    def per_cell(z_idx, i_s):
        init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
        init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]
        return _solve_retirement_at_cell(
            z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
            ...,
            init_a_s_cell, init_a_b_cell,  # was: init_a_s (closure scalar)
            ...,
        )
    return vmap(per_cell)(z_block, is_block)
```

The `call(...)` wrapper grows two args:

```python
def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
    ...
```

Apply the same pattern to `_build_per_age_terminal_kernel` and `_build_per_age_working_kernel`.

### 3.4 `run_lifecycle_solver` orchestrator (lines 1350–1700)

The age loop already maintains `S_list`, `B_list`. For age `t` we read `S_list[t+1]`, `B_list[t+1]` and pass them as the new `init_a_s_arr`, `init_a_b_arr` args to the kernel call:

```python
for t in reversed(range(n_age - 1)):
    ...
    s_prev = S_list[t + 1]   # (n_z, N_state, n_w)
    b_prev = B_list[t + 1]
    if age >= retire_age:
        c_t, s_t, b_t = retirement_kernel(c_next_jnp, pension_next, psi_t, s_prev, b_prev)
    else:
        if use_pen:
            c_t, s_t, b_t = boundary_kernel(c_next_jnp, income_table, pension_next, psi_t, s_prev, b_prev)
        else:
            c_t, s_t, b_t = working_kernel(c_next_jnp, income_table, pension_next, psi_t, s_prev, b_prev)
```

For the **terminal age**, no warm-start source exists — call `terminal_kernel()` with the canonical scalar fallback (see §4).

### 3.5 `lifecycle/model.py` — `SolverConfig` (lines 126–177)

Add a single boolean toggle so the change is reversible without code edits:

```python
use_backward_age_warm_start: bool = True   # if False, every cell uses (init_alpha_s, init_alpha_b)
```

When `False`, `run_lifecycle_solver` builds and passes constant arrays of shape `(n_z, N_state, n_w)` filled with `(init_alpha_s, init_alpha_b)` — i.e. exactly today's behavior, just expressed through the new signature. This keeps the kernel signatures uniform regardless of toggle.

---

## 4. Terminal kernel — special handling

Terminal age `T` has no continuation value and no "previous age". Three distinct things to handle:

1. **Terminal cold init.** The terminal kernel keeps the canonical scalar `init_a_s = sc.init_alpha_s`, `init_a_b = sc.init_alpha_b` cold init for every cell. The build-time closure pattern is fine here; the runtime-arg pattern from §3.3 is overkill since there's no source data. **Decision: keep terminal as cold-init-only**, do not add `init_a_s_arr` to its signature. The first age that uses backward-age warm-start is `T - 1`.

2. **Z-invariance of the terminal solution.** Today the orchestrator broadcasts terminal output across z:

   ```python
   S_list[-1] = jnp.broadcast_to(s_T[None, :, :], (n_z, N_state, n_w))
   ```

   This is exactly what age `T - 1`'s warm-start needs — a `(n_z, N_state, n_w)` array. The broadcast is lazy/on-device (no copy), so the gather `init_a_s_arr[z_idx, i_s, w_ref_idx]` inside `T - 1`'s pmap reads through the broadcast cleanly. **No special handling required**, but verify the broadcast survives one round-trip into a pmap input — JAX should handle it but worth a test print.

3. **Boundary case: `t == T - 1` reads from terminal.** Terminal's `s_T, b_T` policy is the bequest-driven optimum at the last age. Its α at any wealth is a perfectly fine init for `T - 1`'s portfolio choice (which is bequest + one period of utility). No code path divergence needed.

---

## 5. Edge cases & gotchas

### 5.1 Boundary kernel (work → retirement)

At age `retire_age - 1` (last working age), the warm-start source is age `retire_age` (a retirement age). Their internal FOCs differ (working has labor-income expectation; retirement has pension), but **their output policy arrays are the same shape** `(n_z, N_state, n_w)`. The init gather works without modification. Newton at the boundary may take 1–2 extra iters because the optimal portfolio jumps slightly when income source switches from stochastic labor to deterministic pension — accept this as expected.

### 5.2 Tiny-savings fallback (`s_val <= tiny_savings`)

`_egm_scan_cell` currently has:

```python
tiny = s_val <= tiny_savings
a_s_out = jnp.where(tiny, init_a_s, a_s_opt)
a_b_out = jnp.where(tiny, init_a_b, a_b_opt)
```

After this change, `init_a_s` for tiny savings becomes the per-cell warm-started scalar (good — much more representative than the canonical 0.1). No code change.

### 5.3 Solve-resumption from checkpoint

`solve_control` may resume mid-solve. `_normalize_solve_control` and the checkpoint loader populate `C_list, S_list, B_list` from disk for already-solved ages. **Verify** `S_list[t+1]` is a JAX device array (or convertible via `jnp.asarray`) at the point the next-age kernel call uses it. If checkpoints are loaded as NumPy, do `jnp.asarray(...)` once before entering the resume loop so each kernel call doesn't re-upload `(n_z, N_state, n_w) × 8 B` per age.

### 5.4 JIT cache

Adding two `(n_z, N_state, n_w)` array args to each kernel changes the trace signature once. The kernel will recompile **once** on first call after the change, then be cached. For the canonical 9×9×9 configuration this is a sub-second hit per kernel (terminal/retire/work/boundary = 4 compiles total).

If the toggle in §3.5 (`use_backward_age_warm_start`) is set to `False`, the kernel signatures stay the same (we just pass constant arrays) — there is **no** alternate compile path.

### 5.5 `w_ref_idx` choice

`n_w // 2` is the safe default. For canonical configs (n_w = 180), this lands at wealth ≈ median grid point, which is in the interior of the lifecycle wealth distribution. **Do not** use `w_ref_idx = 0` or `n_w - 1` — those are extreme grid points where policies are most state-sensitive and least representative.

If you want to be principled, gather `S_prev[z, state, :].mean()` instead of `S_prev[z, state, w_ref_idx]`. Costs one reduction per cell (negligible) and gives a more stable init across configurations. This is a minor trade-off; either is fine.

### 5.6 `init_alpha_s = 0.1` is the canonical scalar in `SolverConfig`

The existing default is intentionally conservative. `0.1` for stocks and `0.4` for bonds is far from typical converged values (which often live near `α_s ≈ 0.6–0.9` in retirement). After this change the cold-init is only used at the terminal age — its value matters less. No need to retune.

### 5.7 Don't accidentally re-enable in-savings warm-start

`_egm_scan_cell` was deliberately rewritten to vmap the savings sweep. Do not reintroduce `lax.scan` over `s_grid_rev` "for safety". The vmap pattern is the entire point of this redesign.

---

## 6. Verification plan (do NOT skip)

1. **Smoke (`verify_smoke.py`) — must run before commit.** Toggle `use_backward_age_warm_start = True` (default after this change). Expected output:
   - `Status: complete  (6/6 ages solved)`
   - `Policy sanity: PASS`
   - `alpha_s range`: should match the warm-on-savings baseline `[-1.038, 3.082]` to **at least 1e-5** (better than the current cold-init smoke output of `[-1.038, 3.077]`, because the per-cell init is much closer to the optimum than the canonical scalar).
   - `alpha_b range`: similarly should approach `[-8.996, 9.718]` to within 1e-5.

   If alpha ranges drift more than 1e-3: bump `max_iter` from 100 → 200 and re-run. If they still drift: regression in init plumbing — check w_ref gather indexing and z-broadcast threading.

2. **Toggle test.** Run smoke with `use_backward_age_warm_start=False`. Output must reproduce **today's** post-warm-start-kill cold-init smoke output exactly: `alpha_s: [-1.038, 3.077]`, `alpha_b: [-8.927, 9.718]`. This proves the toggle path is identity.

3. **Newton-iteration count.** Add (or use existing) `total_newton_iters` aggregate to diagnostics. With backward-age warm-start it should drop by **3–5×** vs cold-init smoke. Report the number in the commit message.

4. **Bigger config.** `verify_canonical_small.py` (n_w=40, n_s=40, n_z=5, state=3,3,3) is the right pre-AWS check. Should run in ~5–10 min locally and produce no NaN/Inf.

5. **Bit-comparison against original warm-on-savings.** If you can resurrect commit before the warm-start kill (`git show e21fc50:lifecycle/solver.py`), run smoke against it and compare alpha arrays. Element-wise `max(abs(diff))` should be < 1e-7. (Equivalent inits, different convergence path within Newton tolerance — the difference is just FOC-tolerance noise.)

---

## 7. Performance expectations

- **Local smoke (6-age tiny):** wall time dominated by JIT compile (~3–5 min); the solve itself is too short to show a Newton-iter speedup. Don't read perf signal here.
- **Production 9×9×9 (33 retirement ages):** Newton iters per cell drop ~3–5×. EGM-scan vmap dispatch cost is unchanged. Expected wall improvement vs current cold-init benchmark: **2–4×** on CPU, possibly more on GPU (where Newton dispatch overhead per iter is the bottleneck).
- **Memory:** per-age `(n_z, N_state, n_w)` policy arrays were already retained across the age loop — no new allocation. The kernel just reads them once per age.

---

## 8. Out of scope / explicit non-goals

- **Per-savings-point init (Variant B from §2).** Defer until A is benchmarked. If Variant B becomes necessary, it requires changing the return contract of all four cell solvers — that's a follow-up handoff, not part of this work.
- **Lax-scan over ages.** Wrapping the age loop itself in `lax.scan` would let JAX fuse age-to-age data flow. Out of scope: the four kernel types (terminal/retire/boundary/work) have non-uniform setups that don't fit a single scan body cleanly.
- **Markowitz closed-form init at terminal.** The terminal stays cold-init. Replacing terminal cold-init with a myopic-Markowitz precompute is a separate optimization (small payoff because terminal is one age out of ~33–80).
- **Removing `init_alpha_s`, `init_alpha_b` from `SolverConfig`.** They're still used at terminal and as the toggle-off fallback — keep them.

---

## 9. Implementation checklist (for the agent)

- [ ] Add `use_backward_age_warm_start: bool = True` to `SolverConfig` in `lifecycle/model.py`.
- [ ] Modify `_build_per_age_retirement_kernel`: thread `init_a_s_arr, init_a_b_arr` through pmap (`in_axes=None`) into `_solve_retirement_at_cell`. Gather `[z_idx, i_s, w_ref_idx]` per cell. Update the `call(...)` wrapper to accept the two new args.
- [ ] Same for `_build_per_age_working_kernel` (handles both regular working and the work→retirement boundary; both branches use the same gather).
- [ ] **Do NOT modify `_build_per_age_terminal_kernel`** — keep it cold-scalar.
- [ ] Modify `run_lifecycle_solver`'s age loop:
  - Compute `w_ref_idx = pc.n_w // 2` once before the loop.
  - For each non-terminal age, fetch `S_list[t+1], B_list[t+1]` and pass them to the kernel call.
  - If `sc.use_backward_age_warm_start == False`, build constant arrays `jnp.full((n_z, N_state, n_w), sc.init_alpha_s)` (and `init_alpha_b`) once before the loop and use those instead.
- [ ] Update each kernel's docstring to reflect the new args.
- [ ] Run `verify_smoke.py` and confirm §6 acceptance criteria.
- [ ] Add the smoke alpha ranges + total-Newton-iter count to the commit message.
- [ ] No new tests are required, but if there's an existing toggle-style test, add a case with `use_backward_age_warm_start=False` to it.

---

## 10. Files touched (expected)

- `lifecycle/solver.py` — kernel builders + age loop in `run_lifecycle_solver`
- `lifecycle/model.py` — one new `SolverConfig` field
- (no test file changes required; `verify_smoke.py` and `verify_benchmark_bundle.py` are the verification gates)

That's it. Single-commit-able change. ~80–120 net lines of solver.py edits.

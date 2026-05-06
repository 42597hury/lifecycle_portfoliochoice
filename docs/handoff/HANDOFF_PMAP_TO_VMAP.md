# Handoff: Drop `pmap` Layer on Single-GPU; Keep `pmap` on Multi-Core CPU

**Branch:** `jax-rewrite`
**Status when this doc was written:** the three per-age kernel builders ([_build_per_age_terminal_kernel](../../lifecycle/solver.py#L1323), `_build_per_age_retirement_kernel`, `_build_per_age_working_kernel`) wrap a `vmap`-batched cell solve in an outer `pmap` over `n_dev = len(jax.devices())`. On multi-core CPU with virtual XLA devices this is correct (one virtual device per core gives real multi-threaded parallelism). On single-GPU (`n_dev == 1`) the `pmap` layer is degenerate dead weight: a `(1, n_cells, ...)` reshape sandwich, an extra dispatch barrier, and — most importantly — a fence that prevents XLA-CUDA from fusing the per-age solve into one big kernel.

**Target deployment:** AWS A100 / H100, single-GPU. Multi-GPU sharding is out of scope (called out in §9).

**Effort:** 1.5-3 hours including verification on local CPU smoke. Real GPU benchmark is gated on the GPU env handoff landing first.

**Expected payoff:** **2-5× per-age speedup on single A100/H100** vs the current pmap-degenerate path. Dominant gains come from XLA fusion across the previously-pmap'd boundary, not from removing the dispatch overhead per se.

---

## 1. Why this matters

### What pmap is doing today (CPU — correct)

The CPU pattern relies on `--xla_force_host_platform_device_count=N` (set in [lifecycle/__init__.py](../../lifecycle/__init__.py)) to expose `N` virtual JAX devices, one per core. `pmap` shards `n_z * N_state` cells across those virtual devices. XLA-CPU runs each shard on its own thread → real multi-threaded parallelism. **This is the right pattern for CPU and stays.**

### What pmap is doing today on a single GPU (broken)

`len(jax.devices()) == 1` on single-GPU instances. So:
- `n_dev == 1`, `pad_n == n_cells`, no actual sharding happens.
- Inputs reshape `(n_cells, ...)` → `(1, n_cells, ...)` → `(n_cells, ...)`. XLA fuses these *within* a single kernel but **not across the pmap boundary**.
- `pmap` introduces a kernel-launch fence: the per-age solve becomes "launch input-reshape kernel; launch pmap'd vmap'd kernel; launch output-collapse kernel" instead of one fused mega-kernel.
- Inputs marked `in_axes=None` (`c_next`, `pension_next_by_z`, `psi_per_z`, `init_a_s_arr`, `init_a_b_arr`) are tagged as "pmap-broadcast" memory, which XLA treats more conservatively than plain device-resident arrays.

Net: on single-GPU you pay all of pmap's overhead with none of its benefit, and you lose the cross-layer fusion that's the main XLA-CUDA performance win.

### What multi-GPU would want (also broken — out of scope)

On p4d.24xlarge (8× A100), `n_dev == 8`. `pmap` *does* shard cells across the 8 GPUs — but with `in_axes=None`, every GPU broadcasts a full `c_next` of shape `(n_z, N_state, n_w)`. That's 8× the HBM for the broadcast data, plus inter-device collectives every age. The right pattern is `jax.sharding.NamedSharding` + `jit`. **Out of scope here**; track separately.

---

## 2. Goal

Branch the kernel-builder logic on `n_dev`:

- **`n_dev == 1` (single-GPU):** build a vmap-only kernel. Drop padding, reshape, `pmap`. Plain `vmap` over flat cell indices, plain `jit`.
- **`n_dev > 1`:** keep today's `pmap` + inner `vmap` exactly as-is. Don't touch the multi-core CPU path.

This is **not** a CPU/GPU detection — it's a `n_dev` count check. The same logic also handles the (rare) GPU case where someone explicitly limited `CUDA_VISIBLE_DEVICES=0` on a multi-GPU instance.

---

## 3. Scope and non-goals

### In scope

- Add a `_build_*_vmap_only` variant for each of the three kernel builders (terminal, retirement, working).
- Dispatch at the top of each existing builder: `if n_dev == 1: return _build_*_vmap_only(...)`.
- No changes to `_solve_*_at_cell`, `_egm_scan_cell`, `_precompute_per_is_tensors`, or any FOC eval. These are below the pmap boundary and already vmap-clean.
- Verify smoke produces identical alpha ranges on CPU (the pmap path is unchanged) and on a single-virtual-device CPU run (the vmap path is exercised).

### Out of scope

- **Multi-GPU sharding via `jax.sharding`.** Separate work; high payoff on p4d/p5 but ~1 day of effort and requires careful HBM accounting. Track in a future handoff.
- **Removing `pmap` entirely** on the CPU path. Real perf regression — don't.
- **Auto-detecting GPU vs CPU and using different code paths.** The `n_dev == 1` branch is platform-agnostic and sufficient.
- **Fusing the four kernels** (terminal, retire, work, boundary) into a single trace. Their input shapes differ; the per-age Python loop in `run_lifecycle_solver` orchestrates them as separate JIT calls. Stay out of that.

---

## 4. Implementation

### 4.1 Pattern: extract a small helper for the vmap-only path

In [lifecycle/solver.py](../../lifecycle/solver.py), add a helper that builds the vmap-only call closure. Example for the **retirement** kernel; the same pattern applies (with minor signature differences) to terminal and working/boundary.

#### Current (pmap path) at [solver.py:~1410-1474](../../lifecycle/solver.py#L1410):

```python
def _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state):
    ...
    static = (..., bool(sc.use_fori_newton))
    log_R_bill_all, ... = _precompute_per_is_tensors(pcj)
    n_w = pcj.wealth_grid.shape[0]
    w_ref_idx = n_w // 2

    n_cells = n_z * N_state
    pad_n = math.ceil(n_cells / n_dev) * n_dev
    per_dev = pad_n // n_dev
    cell_idx = np.arange(n_cells, dtype=np.int64)
    cell_idx_padded = np.concatenate([cell_idx, np.full(pad_n - n_cells, cell_idx[-1])])
    z_idx_padded = (cell_idx_padded // N_state).astype(np.int64)
    is_idx_padded = (cell_idx_padded % N_state).astype(np.int64)
    z_pm = jnp.asarray(z_idx_padded.reshape(n_dev, per_dev))
    is_pm = jnp.asarray(is_idx_padded.reshape(n_dev, per_dev))

    @partial(pmap, in_axes=(0, 0, None, None, None, None, None))
    def per_dev_solve(z_block, is_block, c_next, pension_next_by_z, psi_per_z,
                      init_a_s_arr, init_a_b_arr):
        def per_cell(z_idx, i_s):
            init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
            init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]
            return _solve_retirement_at_cell(...)
        return vmap(per_cell)(z_block, is_block)

    def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
        c_pm, as_pm, ab_pm = per_dev_solve(
            z_pm, is_pm, c_next_jnp, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )
        def collapse(a):
            flat = jnp.reshape(a, (pad_n,) + a.shape[2:])
            return jnp.reshape(flat[:n_cells], (n_z, N_state, -1))
        return collapse(c_pm), collapse(as_pm), collapse(ab_pm)

    return call
```

#### New: top-of-function branch + extracted vmap-only helper

```python
def _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state):
    if n_dev == 1:
        return _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state)
    return _build_per_age_retirement_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state)


def _build_per_age_retirement_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state):
    """Existing pmap-over-virtual-devices implementation. Used on multi-core
    CPU and on multi-device hosts where pmap sharding is real."""
    # ... existing body verbatim, no changes ...


def _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state):
    """Single-device path (typically single-GPU). Drops the pmap padding /
    reshape / collect and lets XLA fuse the entire per-age solve into one
    kernel.

    Output shape matches the pmap path: ``(n_z, N_state, n_w)`` per array.
    """
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton))

    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = (
        _precompute_per_is_tensors(pcj)
    )

    n_w = pcj.wealth_grid.shape[0]
    w_ref_idx = n_w // 2

    n_cells = n_z * N_state
    cell_idx = np.arange(n_cells, dtype=np.int64)
    z_idx_arr = jnp.asarray((cell_idx // N_state).astype(np.int64))
    is_idx_arr = jnp.asarray((cell_idx % N_state).astype(np.int64))

    @jit
    def all_cells(c_next, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
        def per_cell(z_idx, i_s):
            init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
            init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]
            return _solve_retirement_at_cell(
                z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
                log_R_bill_all[i_s], log_x_s_all[i_s], log_x_b_all[i_s],
                j_corners_all[i_s], w_corners_all[i_s],
                pcj.weight_kv_kr, pcj.annuity_factors,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s_cell, init_a_b_cell,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )
        return vmap(per_cell)(z_idx_arr, is_idx_arr)   # (n_cells, n_w) per output

    def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
        c_flat, s_flat, b_flat = all_cells(
            c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr,
        )
        return (
            jnp.reshape(c_flat, (n_z, N_state, -1)),
            jnp.reshape(s_flat, (n_z, N_state, -1)),
            jnp.reshape(b_flat, (n_z, N_state, -1)),
        )

    return call
```

**Key differences from the pmap path:**

1. No `pad_n`, no padding of cell indices, no `reshape((n_dev, per_dev) + ...)`. The vmap path operates on `n_cells` exactly.
2. No `@pmap` decorator. The inner function is `@jit`'d directly. (You can also leave `@jit` off — the closure is implicitly traced when `call(...)` runs, and JIT'ing `all_cells` makes the trace explicit. Pick whichever feels cleaner; effect is identical.)
3. Output reshape goes directly from `(n_cells, n_w)` to `(n_z, N_state, n_w)`. No intermediate `(pad_n, ...)` step.
4. Inputs `c_next, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr` are passed as plain JAX arrays — no `in_axes=None` decoration needed (vmap doesn't see them; they're closed over inside `per_cell`).

### 4.2 Same pattern for the **terminal** kernel

The terminal kernel is simpler (no z dimension; pmaps over `N_state`). Pseudocode for the vmap-only variant:

```python
def _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc):
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (...)

    # Pre-build per-i_s log-return tensors (same NumPy → jnp prep as today,
    # but no padding for n_dev). state_grid_np, log_R_bill, log_x_s, log_x_b,
    # ann_p constructed exactly as in the existing builder.
    log_R_bill_jnp = jnp.asarray(log_R_bill)        # (N_state, n_state_quad, n_ret_quad)
    log_x_s_jnp = jnp.asarray(log_x_s)
    log_x_b_jnp = jnp.asarray(log_x_b)
    ann_jnp = jnp.asarray(np.asarray(pcj.annuity_factors))   # (N_state,)

    N_state = log_R_bill_jnp.shape[0]

    @jit
    def all_is():
        def per_i_s(log_Rb, lxs, lxb, A):
            return _solve_terminal_at_i_s(
                log_Rb, lxs, lxb, pcj.weight_kv_kr, A,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s, init_a_b,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )
        return vmap(per_i_s)(log_R_bill_jnp, log_x_s_jnp, log_x_b_jnp, ann_jnp)

    def call(_unused_age_idx=None):
        return all_is()    # already (N_state, n_w) per output — no reshape needed

    return call
```

**Key simplifications:**

- No padding (`pad_n` etc.).
- No reshape to `(n_dev, per_dev, ...)`.
- The output is already `(N_state, n_w)` — no slice `[:N_state]` needed.
- The closure `all_is()` takes no args (everything is closed over) — keeps the call signature `call(_unused_age_idx=None)` matching the pmap path's signature for [run_lifecycle_solver](../../lifecycle/solver.py#L1714) compatibility.

The current pmap terminal kernel does the per-i_s log-return tensor build with NumPy then pads with `pad0(...)`. In the vmap-only variant, just keep the NumPy build and skip the `pad0` calls — go straight from NumPy arrays to `jnp.asarray(...)`.

### 4.3 Same pattern for the **working** kernel (and boundary case)

The working kernel handles two cases via `use_pension_next` (working → retirement boundary vs interior working). Both branches stay; only the pmap layer is removed. Follow the retirement template, with the additional bracket logic and pension/income handling moved inside `per_cell` exactly as today:

```python
def _build_per_age_working_kernel_vmap_only(pcj, mp, sc, n_z, N_state, use_pension_next):
    # ... same setup as retirement-vmap-only but with the working-specific
    # bracket_uniform call and use_pension_next branch in per_cell ...

    @jit
    def all_cells(c_next, income_next_table_z, pension_next_by_z, psi_per_z,
                  init_a_s_arr, init_a_b_arr):
        def per_cell(z_idx, i_s):
            z_now = pcj.z_grid[z_idx]
            z_next = mp.rho * z_now + pcj.eta_nodes
            iz_lo, frac_z = vmap(bracket_uniform, in_axes=(0, None, None, None))(
                z_next, pcj.z_grid[0], pcj.dz, pcj.z_grid.shape[0]
            )

            if use_pension_next:
                pension_at_eta = (
                    (1.0 - frac_z) * pension_next_by_z[iz_lo]
                    + frac_z * pension_next_by_z[iz_lo + 1]
                )
                income_table = pension_at_eta[:, None] * jnp.ones_like(pcj.eps_weights)[None, :]
            else:
                income_table = income_next_table_z[z_idx]

            init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
            init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]

            return _solve_working_at_cell(...)   # same arg list as today

        return vmap(per_cell)(z_idx_arr, is_idx_arr)

    def call(c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
             init_a_s_arr, init_a_b_arr):
        c_flat, s_flat, b_flat = all_cells(
            c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )
        return (
            jnp.reshape(c_flat, (n_z, N_state, -1)),
            jnp.reshape(s_flat, (n_z, N_state, -1)),
            jnp.reshape(b_flat, (n_z, N_state, -1)),
        )

    return call
```

### 4.4 Dispatcher pattern (one branch per builder)

Top of each of the three existing builders:

```python
def _build_per_age_terminal_kernel(pcj, mp, sc, n_dev):
    if n_dev == 1:
        return _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc)
    return _build_per_age_terminal_kernel_pmap(pcj, mp, sc, n_dev)


def _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state):
    if n_dev == 1:
        return _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state)
    return _build_per_age_retirement_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state)


def _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next):
    if n_dev == 1:
        return _build_per_age_working_kernel_vmap_only(pcj, mp, sc, n_z, N_state, use_pension_next)
    return _build_per_age_working_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next)
```

The existing builder bodies move into `_build_*_kernel_pmap` verbatim (rename only). The new vmap-only variants live alongside.

`run_lifecycle_solver` ([solver.py:~1647](../../lifecycle/solver.py#L1647)) keeps its `n_dev = len(jax.devices())` line and passes `n_dev` to all three builders. No orchestrator changes.

### 4.5 Optional: log which path is taken

Add a single print in [run_lifecycle_solver](../../lifecycle/solver.py#L1414) right after `print(f"  Devices: {jax.devices()}")`:

```python
if verbose >= 1:
    n_dev = len(jax.devices())
    pattern = "vmap-only (single-device)" if n_dev == 1 else f"pmap+vmap ({n_dev} devices)"
    print(f"  Cell-batching pattern: {pattern}")
```

Helps catch the case where someone runs on GPU and expects vmap-only but accidentally has multiple devices visible (e.g. forgot to set `CUDA_VISIBLE_DEVICES=0`).

---

## 5. Verification

### 5.1 CPU regression check (must pass)

The CPU path (multi-core, virtual devices, `n_dev == cpu_count`) is **unchanged** — it routes through `_build_*_kernel_pmap` which is the existing body. Smoke must produce **bit-identical** alpha ranges before and after this change:

```bash
# Before this change is applied: capture baseline
git stash
python verify_smoke.py 2>&1 | tee /tmp/smoke_baseline.txt
git stash pop

# After change: re-run
python verify_smoke.py 2>&1 | tee /tmp/smoke_after.txt

# Compare alpha ranges
grep "alpha_s range\|alpha_b range" /tmp/smoke_baseline.txt /tmp/smoke_after.txt
```

The two outputs must match to all printed digits. If they don't: regression in the dispatcher or accidental edit to the pmap path.

### 5.2 Single-device smoke (exercises the new vmap-only path)

The pmap path only kicks in when `n_dev > 1`. To exercise the new vmap-only path on local CPU, force a single virtual device:

```bash
LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 python verify_smoke.py
```

This sets `len(jax.devices()) == 1`, dispatcher routes to vmap-only path. Verify:

1. The new "Cell-batching pattern: vmap-only (single-device)" line appears (if §4.5 was implemented).
2. `Status: complete  (6/6 ages solved)`.
3. `Policy sanity: PASS`.
4. Alpha ranges match the multi-core baseline within ~1e-9. (Floats may differ in the last few decimals due to different op order under XLA fusion — anything larger is a bug.)

### 5.3 Wall-time spot-check on CPU

Single-device CPU smoke should be **much slower** than multi-core CPU smoke (single-threaded XLA-CPU vs N-threaded). This is **expected and not a problem**: that path exists for GPU. On a 12-core dev box, expect single-device smoke to be 10-15× slower than multi-core. If it's roughly the same, XLA-CPU is auto-parallelizing — fine, but unexpected.

### 5.4 GPU benchmark (after the GPU env handoff lands)

Once a GPU instance is up:

1. Confirm the dispatcher logs `Cell-batching pattern: vmap-only (single-device)` on `p4d.24xlarge` with `CUDA_VISIBLE_DEVICES=0`.
2. Run the benchmark. Compare wall to a hypothetical "pmap-on-single-GPU" run (which can be forced by setting `CUDA_VISIBLE_DEVICES=0,0` or similar — JAX will report 1 device but the n_dev==1 branch still triggers; the only way to genuinely force pmap with 1 device is to revert this handoff, so this comparison is academic).

Real comparison is the GPU benchmark vs the `hpc8a.96xlarge` Numba reference (1342 s). Expected on A100 with this handoff + persistent cache + backward-age warm-start applied: **~150-300 s** (4.5-9× over Numba reference, before any GPU-specific further tuning).

### 5.5 Multi-GPU smoke

Out of scope, but if curious: on a multi-GPU instance, the pmap path takes effect. Should still produce identical alpha ranges to single-GPU; performance is suboptimal (the in_axes=None broadcast issue from §1) but correctness is unaffected.

---

## 6. Edge cases / gotchas

### 6.1 `_precompute_per_is_tensors` is shared

[_precompute_per_is_tensors](../../lifecycle/solver.py#L1226) builds `(N_state, n_state_quad, n_ret_quad)` and `(N_state, n_state_quad, 8)` tensors via vmap over the state grid. Used by both pmap and vmap-only paths — no changes needed. Just call it from each `_vmap_only` builder once at build time.

### 6.2 The terminal kernel's NumPy log-return prep

Today's `_build_per_age_terminal_kernel_pmap` constructs `log_R_bill`, `log_x_s`, `log_x_b` arrays in NumPy (not via `_precompute_per_is_tensors`). The vmap-only variant should reuse the same NumPy prep — just skip the `pad0(...)` and `reshape_for_pmap(...)` lines and convert directly to `jnp.asarray(...)`. Don't switch to `_precompute_per_is_tensors` here unless you've verified bit-equivalence first; the two prep paths use slightly different broadcasting orders.

### 6.3 Closure variable capture

In `_build_*_vmap_only`, several arrays are closed over inside `per_cell` (e.g. `log_R_bill_all`, `init_a_s_arr`, `pcj.weight_kv_kr`). With pmap+vmap, these were either passed as `in_axes=None` args (declarative) or closed over similarly. Either approach works inside `vmap`. **Keep `init_a_s_arr` and `init_a_b_arr` as `call(...)` arguments** (passed by `run_lifecycle_solver` per age) so the kernel doesn't re-trace each age. The other arrays are config-shaped and stay closed over.

### 6.4 `jit` placement

Decorating `all_cells` with `@jit` is the conservative choice — makes the trace boundary explicit. The pmap path doesn't have a separate `@jit` because `pmap` itself JITs. Without `@jit` on `all_cells`, JAX will still trace at first call (`call(...)` invokes a vmap'd function whose dispatch is implicitly JIT'd in modern JAX) — same end result. **Use `@jit` for clarity.**

### 6.5 `n_dev` check is at builder time, not runtime

The `if n_dev == 1` branch is a Python-level check during `run_lifecycle_solver` setup. Once the kernel is built, the choice is baked in. This means:

- Resuming a checkpoint on a different host (e.g. dev → AWS) will rebuild kernels with the new `n_dev`. ✓ correct behavior.
- If the user changes `CUDA_VISIBLE_DEVICES` mid-run (don't), the kernel won't know. **Don't.**

### 6.6 Memory: vmap over 8000+ cells on GPU

On A100 (40 GB HBM), the per-cell `c_corners_T` of shape `(48, 11, 8, 180) ≈ 3 MB` × `n_cells = 8019` ≈ **24 GB** if XLA fully materializes the vmap'd batch. **It usually doesn't** — XLA streams the gather through the consumer FOC eval — but this is the single biggest GPU memory risk. Verification approach:

```python
# Run smoke on GPU, capture peak HBM:
import jax
print(jax.devices()[0].memory_stats())   # available on CUDA backend
```

If peak HBM > 30 GB on the smoke, that's a red flag — production canonical at 9×9×9 will OOM. Mitigation is a separate handoff (chunk the cell vmap into N batches, swap c_corners_T to lower precision for the gather only, etc.).

### 6.7 `vmap` axis ordering

`vmap(per_cell)(z_idx_arr, is_idx_arr)` produces outputs with the cell-batch axis at position 0. Reshape to `(n_z, N_state, -1)` is consistent because `cell_idx // N_state` and `cell_idx % N_state` were used to construct the index arrays. **Keep this exact construction order** (z first, then i_s); flipping to `(i_s, n_z)` would silently swap axes in the output reshape.

### 6.8 Don't accidentally convert `init_a_s_arr` to NumPy

`run_lifecycle_solver` passes `S_list[t+1]` (a JAX device array) as `init_a_s_arr`. If anyone changes that to `np.asarray(S_list[t+1])`, the per-age call would upload a 5 MB array per age — measurable overhead on GPU. Currently it's a device array, no upload. Leave it.

---

## 7. Files touched

| File | Change | Approx lines |
|---|---|---|
| [lifecycle/solver.py](../../lifecycle/solver.py) | Rename existing 3 builder bodies to `*_pmap`. Add 3 new `*_vmap_only` builders. Add 3 small dispatcher functions. Optional: 2-line cell-batching pattern log in `run_lifecycle_solver`. | +250 / -0 (additions; pmap bodies just rename) |

No new files. No `model.py` change. No test file changes.

---

## 8. Implementation checklist (for the agent)

- [ ] Rename existing kernel-builder bodies:
  - `_build_per_age_terminal_kernel` → `_build_per_age_terminal_kernel_pmap`
  - `_build_per_age_retirement_kernel` → `_build_per_age_retirement_kernel_pmap`
  - `_build_per_age_working_kernel` → `_build_per_age_working_kernel_pmap`
  
  This is a verbatim move; no code edits inside the bodies. **Verify smoke still passes after the rename + dispatcher additions** (§5.1) before adding the vmap-only variants.

- [ ] Add three new vmap-only builders following the templates in §4.1, §4.2, §4.3.

- [ ] Add three dispatcher functions (the new `_build_per_age_*_kernel` outer functions) per §4.4.

- [ ] (Optional but recommended) Add the cell-batching pattern log line per §4.5.

- [ ] Run `python verify_smoke.py` (multi-core CPU, exercises pmap path). Confirm:
  - `Status: complete`, `Policy sanity: PASS`.
  - Alpha ranges match the pre-change baseline to printed precision.
  - (If §4.5 added) Log shows `Cell-batching pattern: pmap+vmap (N devices)`.

- [ ] Run `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 python verify_smoke.py` (single-device CPU, exercises vmap-only path). Confirm:
  - Same correctness criteria.
  - Alpha ranges match the multi-core run within 1e-9.
  - (If §4.5 added) Log shows `Cell-batching pattern: vmap-only (single-device)`.
  - Wall is much slower than multi-core (expected — single-threaded XLA-CPU).

- [ ] Single commit:
  ```
  solver: drop pmap layer on single-device hosts (GPU-friendly)

  Each of the three per-age kernel builders now branches on n_dev.
  - n_dev > 1: existing pmap-over-virtual-devices pattern (multi-core
    CPU). No change.
  - n_dev == 1: new vmap-only pattern. Drops pmap padding/reshape/
    collect; XLA-CUDA can now fuse the entire per-age solve into one
    kernel. Single-GPU benchmark is the target.

  No changes to _solve_*_at_cell, _egm_scan_cell, FOC eval, or
  _precompute_per_is_tensors. Output shapes unchanged.

  Verified:
  - Multi-core CPU smoke: alpha ranges identical to pre-change baseline.
  - Single-device CPU smoke (LIFECYCLE_DISABLE_VIRTUAL_CPUS=1): same
    alpha ranges to ~1e-12, slower wall (expected, not a regression
    for the GPU target).

  Multi-GPU sharding (jax.sharding.NamedSharding) still TODO; pmap path
  on multi-GPU is functionally correct but suboptimal due to in_axes=None
  broadcast.
  ```

- [ ] Push to `jax-rewrite`. No PR needed.

---

## 9. Out of scope / future work

### Multi-GPU sharding via `jax.sharding`

On `p4d.24xlarge` (8× A100), today's pmap path runs but broadcasts huge arrays. The right fix is `jax.sharding.NamedSharding` + `jit` with explicit cell-axis sharding:

```python
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((n_dev,), ('cells',))
cell_sharding = NamedSharding(mesh, P('cells'))
z_idx_arr = jax.device_put(z_idx_arr, cell_sharding)
is_idx_arr = jax.device_put(is_idx_arr, cell_sharding)
# vmap'd computation now auto-shards across devices
```

Estimated effort: ~1 day. Highest GPU payoff once single-GPU is healthy. **Defer until single-GPU benchmark is done and you've established that multi-GPU is worth it for the production sweep cadence.**

### Removing the pmap path entirely

The CPU path is real users (local dev, smoke runs) — keep it. Once the project has migrated production to GPU, the pmap path can be deleted in a follow-up; for now both paths coexist behind the `n_dev == 1` gate.

### Per-cell HBM chunking

If the c_corners materialization concern from §6.6 turns into an actual OOM at production canonical size, chunk the vmap:

```python
chunk_size = 1000   # cells per chunk
chunks = jnp.array_split(jnp.arange(n_cells), n_cells // chunk_size + 1)
results = [vmap(per_cell)(z_idx_arr[c], is_idx_arr[c]) for c in chunks]
return jnp.concatenate(results)
```

Adds a Python-level loop (no fusion across chunks) but keeps peak HBM bounded. Only do this if profiling shows it's needed.

### Cross-age fusion

Fusing the four kernels (terminal, retire, work, boundary) into a single trace would let XLA fuse across ages too. Their input shapes differ enough that this isn't trivial; skip until single-device performance is well-understood.

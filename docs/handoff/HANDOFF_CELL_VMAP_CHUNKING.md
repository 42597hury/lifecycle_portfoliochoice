# Handoff: Manual Cell-Vmap Chunking for Single-GPU Memory Bounding

**Branch:** `jax-rewrite`
**Status when this doc was written:** the `vmap-only (single-device)` path landed in commit `cf0bdfc` and runs cleanly at 5⁴ × `n_state_quad=(2,3,2,3)` on GH200 (97 GB HBM). At 7⁴ and 9⁴, **single-device runs OOM** because XLA's memory planner makes inconsistent decisions across input shapes — the 7⁴ run requested 1.06 TB, the 9⁴ run requested 96 GB, both for the same 67 GB minimum working set. **This is the §6.6 risk from `HANDOFF_PMAP_TO_VMAP.md` materialising in production.** Detailed evidence in [docs/notes/GPU_TRIAL_FINDINGS.md](../notes/GPU_TRIAL_FINDINGS.md).

**Outcome of this handoff:** make the three `_build_per_age_*_kernel_vmap_only` functions split the cell-axis vmap into K explicit chunks, each with a fixed cell count. Per-chunk peak memory becomes deterministic and bounded by user-chosen K, regardless of XLA's planning whims.

**Effort:** ~80-120 lines net across `lifecycle/solver.py` + `lifecycle/model.py`. ~1 day of focused work.

**Target deployment:** GH200 / H100 / H200 single-GPU runs at 7⁴ state grid and beyond. The 5⁴ runs that already work today should be unaffected.

---

## 1. Goal

Three concrete improvements:

1. **Add `SolverConfig.cell_vmap_chunks: int = 1`** — number of chunks to split the cell vmap into. Default `1` = current single-vmap behaviour, no regression for 5⁴ runs.
2. **Implement chunked dispatch** in the three `_*_vmap_only` kernel builders so each per-age call processes cells in K sequential vmap-batches instead of one mega-batch. **Each chunk has fixed shape** so XLA compiles once and reuses the trace.
3. **Add bit-identical-output guarantee** — `cell_vmap_chunks=1` and `cell_vmap_chunks=K>1` must produce identical alphas to numerical precision. The math is unchanged; only the dispatch shape splits.

---

## 2. Why this is now mandatory, not optional

Tonight's GPU trial confirmed:

- **Same algorithm, different shape → wildly different XLA memory plans.** 9⁴ requested 96 GB; 7⁴ requested 1.06 TB. Both have the same intrinsic 67 GB minimum working set.
- **No single GPU has enough HBM for unchunked 7⁴ at full quad** (worst-case 950 GB) or 9⁴ at any quad (worst-case 2.6 TB).
- **Reducing other dimensions doesn't help enough.** Even `state_quad=(2,3,2,3)`, `max_iter=100`, `n_w=180` keeps 7⁴ within 240 GB worst-case full materialization. Chunking is the only deterministic ceiling.
- **XLA-CUDA's planner heuristics are non-monotonic in input dimensions** — a known compiler limitation, not a bug we can wait out.

The §6.6 follow-up sketch from `HANDOFF_PMAP_TO_VMAP.md` is the right design. This handoff turns it into shipped code.

---

## 3. Scope

### In scope

- `lifecycle/model.py` — add `SolverConfig.cell_vmap_chunks` field.
- `lifecycle/solver.py` — modify the three vmap-only builders:
  - `_build_per_age_terminal_kernel_vmap_only`
  - `_build_per_age_retirement_kernel_vmap_only`
  - `_build_per_age_working_kernel_vmap_only`
- A small helper `_chunked_vmap_cells(per_cell, z_idx_arr, is_idx_arr, n_chunks)` (or similar) factored once and called from all three builders.
- One unit test in a new `test_chunking.py` (or a section in an existing test file) asserting bit-identical output between `n_chunks=1` and `n_chunks={2, 4, 8}` on the smoke config.

### Out of scope

- **The pmap path.** `_*_pmap` kernel builders are for multi-device CPU runs and stay unchanged. Don't touch them.
- **Multi-GPU sharding** (the `jax.sharding.NamedSharding` follow-up). Separate handoff. Single-device chunking is sufficient for the 7⁴ / 9⁴ goals.
- **Automatic n_chunks selection.** User picks the number based on hardware and config. We don't try to be clever and infer it.
- **Performance tuning of the chunked path.** Goal here is correctness + memory boundedness. Optimising chunk dispatch overhead is later.

### Hard constraints

- **Bit-identical output guarantee** between `n_chunks=1` and any `n_chunks > 1`. The math doesn't change; chunking only changes dispatch shape. Output values must match to ~1e-12.
- **No regression on 5⁴ runs.** Default `cell_vmap_chunks=1` must produce identical performance characteristics to today.
- **Same JIT trace for all chunks.** All chunks must have identical shape so XLA compiles the inner kernel once and reuses for all chunks. **Don't have a "last chunk is smaller" path** — pad instead.

---

## 4. Implementation

### 4.1 SolverConfig field

In [lifecycle/model.py](../../lifecycle/model.py), inside `SolverConfig`:

```python
class SolverConfig(NamedTuple):
    ...
    # --- Cell-axis vmap chunking (single-GPU memory bounding) ---
    # Splits the per-age vmap over (n_z * N_state) cells into this many
    # sequential chunks. Each chunk has shape (n_cells_padded // K, ...).
    # Per-chunk peak HBM = (worst_case_full_materialization / K), giving
    # deterministic memory bound independent of XLA's compilation choices.
    #
    # When 1: no chunking (default; matches today's behaviour, fastest dispatch).
    # When > 1: K sequential vmap calls per age. Adds ~K kernel-launch dispatches
    # per age but bounds memory.
    #
    # Heuristic for picking K on a single GPU:
    #   per_cell_memory_MB = n_state_quad * n_z * 2**n_state * n_w * 8 / 1e6
    #   total_worst_case   = per_cell_memory_MB * (n_z * N_state)
    #   K_min              = ceil(total_worst_case / target_HBM_budget_MB)
    # On GH200 (97 GB HBM, ~30 GB headroom for other state):
    #   target_HBM_budget = 60 GB → K = ceil(total_worst_case / 60_000)
    cell_vmap_chunks: int = 1
```

**Important**: place this **after** `wealth_dynamics_spec` (the existing last field) and `use_backward_age_warm_start`. Adding fields to NamedTuple at the end is non-breaking; inserting earlier breaks pickled bundles.

### 4.2 The chunking helper

Add a new helper function in `lifecycle/solver.py`, near the `_*_vmap_only` builders:

```python
def _chunked_vmap_cells(per_cell_fn, z_idx_arr, is_idx_arr, n_chunks):
    """Split a per-cell vmap into n_chunks sequential vmap calls.

    Each chunk has fixed shape ``chunk_size = ceil(n_cells / n_chunks)``,
    padded by repeating the last cell index. After all chunks complete,
    the result is concatenated and sliced back to ``n_cells``.

    Why padding to a fixed chunk_size: XLA compiles the inner vmap once
    per shape. If the last chunk is shorter, XLA recompiles for it,
    doubling JIT cost. Padding keeps every chunk the same shape so the
    trace is reused.

    Parameters
    ----------
    per_cell_fn : callable (z_idx, i_s) -> (c, s, b) tuple
        The per-cell solver. Each output is shape (n_w,).
    z_idx_arr : jnp.ndarray, shape (n_cells,)
    is_idx_arr : jnp.ndarray, shape (n_cells,)
    n_chunks : int
        Static (Python int). Number of chunks.

    Returns
    -------
    (c_flat, s_flat, b_flat) tuple of (n_cells, n_w) arrays.
    """
    n_cells = z_idx_arr.shape[0]
    if n_chunks == 1:
        return vmap(per_cell_fn)(z_idx_arr, is_idx_arr)

    chunk_size = (n_cells + n_chunks - 1) // n_chunks   # ceil
    n_cells_padded = chunk_size * n_chunks

    # Pad with the LAST cell index (matches the pmap path's padding pattern).
    pad_count = n_cells_padded - n_cells
    z_idx_padded = jnp.concatenate(
        [z_idx_arr, jnp.full(pad_count, z_idx_arr[-1], dtype=z_idx_arr.dtype)]
    )
    is_idx_padded = jnp.concatenate(
        [is_idx_arr, jnp.full(pad_count, is_idx_arr[-1], dtype=is_idx_arr.dtype)]
    )

    chunk_results = []
    for i in range(n_chunks):
        start = i * chunk_size
        z_chunk = lax.dynamic_slice_in_dim(z_idx_padded, start, chunk_size)
        is_chunk = lax.dynamic_slice_in_dim(is_idx_padded, start, chunk_size)
        chunk_results.append(vmap(per_cell_fn)(z_chunk, is_chunk))

    # Concatenate along cell axis and slice off padding.
    c_full = jnp.concatenate([r[0] for r in chunk_results], axis=0)[:n_cells]
    s_full = jnp.concatenate([r[1] for r in chunk_results], axis=0)[:n_cells]
    b_full = jnp.concatenate([r[2] for r in chunk_results], axis=0)[:n_cells]
    return c_full, s_full, b_full
```

**Critical design decisions:**

1. **`n_chunks` is a Python int, not a traced value.** Closure constant. XLA traces the inner vmap once per chunk-shape and reuses.
2. **Padding pattern matches the existing pmap path** ([_build_per_age_*_kernel_pmap](../../lifecycle/solver.py#L1565)). Same pattern, same justification: pad with the last cell index, slice off after.
3. **Separate concatenation per output array** because the per-cell fn returns a tuple of three arrays.
4. **`lax.dynamic_slice_in_dim` instead of array slicing** because the chunk shape is fixed; using static slicing would force XLA to specialize per chunk index.
5. **Sequential `for i in range(n_chunks):` is fine** — Python unrolls at trace time. Each `vmap(per_cell)` call compiles to a single GPU kernel, dispatched K times. We accept that overhead.

### 4.3 Wiring into the three vmap-only kernel builders

#### Terminal kernel ([_build_per_age_terminal_kernel_vmap_only](../../lifecycle/solver.py#L1424))

Terminal vmaps over `N_state` (no z axis). Adapt the helper or write a 1D version:

```python
def _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc):
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter, ...,
              bool(sc.use_fori_newton))

    # ... existing log_R_bill / log_x_s / log_x_b prep, unchanged ...

    n_chunks = int(sc.cell_vmap_chunks)
    N_state = log_R_bill_jnp.shape[0]

    @jit
    def all_is():
        def per_i_s(log_Rb, lxs, lxb, A):
            return _solve_terminal_at_i_s(
                log_Rb, lxs, lxb, pcj.weight_kv_kr, A,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s, init_a_b,
                ...,
                *static,
            )

        if n_chunks == 1:
            return vmap(per_i_s)(log_R_bill_jnp, log_x_s_jnp, log_x_b_jnp, ann_jnp)

        chunk_size = (N_state + n_chunks - 1) // n_chunks
        N_padded = chunk_size * n_chunks
        pad_count = N_padded - N_state

        # Pad with last entry along axis 0.
        def pad_axis0(arr):
            return jnp.concatenate(
                [arr, jnp.broadcast_to(arr[-1:], (pad_count,) + arr.shape[1:])],
                axis=0,
            )

        log_Rb_pad = pad_axis0(log_R_bill_jnp)
        lxs_pad = pad_axis0(log_x_s_jnp)
        lxb_pad = pad_axis0(log_x_b_jnp)
        ann_pad = pad_axis0(ann_jnp)

        chunk_results = []
        for i in range(n_chunks):
            start = i * chunk_size
            log_Rb_c = lax.dynamic_slice_in_dim(log_Rb_pad, start, chunk_size, axis=0)
            lxs_c = lax.dynamic_slice_in_dim(lxs_pad, start, chunk_size, axis=0)
            lxb_c = lax.dynamic_slice_in_dim(lxb_pad, start, chunk_size, axis=0)
            ann_c = lax.dynamic_slice_in_dim(ann_pad, start, chunk_size, axis=0)
            chunk_results.append(vmap(per_i_s)(log_Rb_c, lxs_c, lxb_c, ann_c))

        c_full = jnp.concatenate([r[0] for r in chunk_results], axis=0)[:N_state]
        s_full = jnp.concatenate([r[1] for r in chunk_results], axis=0)[:N_state]
        b_full = jnp.concatenate([r[2] for r in chunk_results], axis=0)[:N_state]
        return c_full, s_full, b_full

    def call(_unused_age_idx=None):
        return all_is()
    return call
```

**Note:** terminal pads four arrays (`log_R_bill_jnp`, `log_x_s_jnp`, `log_x_b_jnp`, `ann_jnp`) instead of two index arrays. The chunking helper from §4.2 doesn't directly apply because the per-i_s fn takes `(log_Rb, lxs, lxb, A)` not `(z_idx, i_s)`. **Either inline the chunking logic** as above, **or** generalize the helper to accept arbitrary input arrays. Inline is simpler given there are only three call sites total.

#### Retirement kernel ([_build_per_age_retirement_kernel_vmap_only](../../lifecycle/solver.py#L1635))

Use `_chunked_vmap_cells` from §4.2:

```python
def _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state, per_is_tensors):
    ...
    n_chunks = int(sc.cell_vmap_chunks)

    @jit
    def all_cells(c_next, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
        def per_cell(z_idx, i_s):
            # ... existing per-cell body, unchanged ...
            return _solve_retirement_at_cell(...)

        return _chunked_vmap_cells(per_cell, z_idx_arr, is_idx_arr, n_chunks)
    ...
```

Then in `call(...)`:

```python
def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
    c_flat, s_flat, b_flat = all_cells(
        c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr,
    )
    return (
        jnp.reshape(c_flat, (n_z, N_state, -1)),
        jnp.reshape(s_flat, (n_z, N_state, -1)),
        jnp.reshape(b_flat, (n_z, N_state, -1)),
    )
```

The reshape pattern is unchanged from today. The output of `_chunked_vmap_cells` already has shape `(n_cells, n_w)` after the slice-off-padding step.

#### Working kernel ([_build_per_age_working_kernel_vmap_only](../../lifecycle/solver.py#L1795))

Same pattern as retirement. The `per_cell` body is more complex (handles `use_pension_next` branch + per-cell `z_next` bracket arithmetic) but the chunking call is identical:

```python
return _chunked_vmap_cells(per_cell, z_idx_arr, is_idx_arr, n_chunks)
```

---

## 5. Edge cases / gotchas

### 5.1 `n_chunks > n_cells`

If user sets `cell_vmap_chunks=10` but only has `n_cells=5` (silly but possible at smoke), `chunk_size = 1`, `n_cells_padded = 10`, padding count = 5. Wasteful but works. **Don't error** — degrade gracefully.

### 5.2 `n_chunks <= 0`

Invalid. Treat as `n_chunks=1` and emit a warning, OR raise. Pick the latter for explicit user error visibility:

```python
if n_chunks < 1:
    raise ValueError(f"cell_vmap_chunks must be >= 1, got {n_chunks}")
```

Add this validation in `run_lifecycle_solver` near the start (where other SolverConfig validation lives), not inside the chunked helper (the helper is on the JIT path; validation should be Python-level).

### 5.3 Exact divisibility considerations

When `n_cells % n_chunks == 0`, `pad_count = 0`. The padding code paths still run (just pad with 0 elements). `jnp.concatenate([arr, jnp.full(0, ...)])` is a no-op but valid JAX. Don't add a special case — keep one code path.

### 5.4 Output shape after slicing

After `c_full = jnp.concatenate(...)[:n_cells]`, the slice produces a `(n_cells, n_w)` array. **Slicing inside JIT is fine** (it's a static slice — `n_cells` is known at trace time). XLA emits efficient DCE for the dropped padding values.

### 5.5 Random number generation (none here)

The solver is deterministic — no RNG state. Chunking can't introduce non-determinism. **No need to worry about RNG keys per chunk.**

### 5.6 Backward-age warm-start interaction

`init_a_s_arr` and `init_a_b_arr` are full `(n_z, N_state, n_w)` arrays passed into `all_cells`. Inside `per_cell`, the gather is `init_a_s_arr[z_idx, i_s, w_ref_idx]`. **Each chunk's `per_cell` reads from the same full array**, just sliced over which `(z_idx, i_s)` pairs. No special handling needed.

### 5.7 fori_loop / Newton interaction

Newton's `lax.fori_loop` runs inside `per_cell`. Each chunk's Newton state is independent. **Newton convergence behaviour is unchanged.** No cross-chunk dependency.

### 5.8 Compilation cost

Each `vmap(per_cell)(z_chunk, is_chunk)` call has the same shape, so XLA traces once and reuses. **First chunk pays the JIT compile cost; chunks 2..K reuse the cached trace.** Persistent compilation cache (already in place via `lifecycle/__init__.py`) handles repeats across runs.

### 5.9 Why not a `lax.scan` instead of a Python `for` loop?

Tempting: `lax.scan` over chunks would be a single trace boundary instead of K. **But** the per-chunk vmap output shape is `(chunk_size, n_w)`, which scan's accumulator handling complicates. The Python for-loop approach traces once at JIT time and dispatches K kernel launches at runtime — same effect, simpler code. **Stick with the Python for-loop.**

### 5.10 Heuristic for choosing `cell_vmap_chunks` (document for users)

Add to the `SolverConfig.cell_vmap_chunks` docstring the heuristic from §4.1. For convenience, here's the table format the user should expect:

```
| Hardware | Total HBM | Headroom budget | n_chunks at 7⁴ full quad | n_chunks at 9⁴ full quad |
|---|---|---|---|---|
| GH200 / H200 SXM5 | 97 / 141 GB | 30-50 GB | 16 | 64 |
| H100 SXM5 | 80 GB | 25 GB | 24 | 96 |
| A100 80 GB | 80 GB | 25 GB | 24 | 96 |
| A100 40 GB | 40 GB | 12 GB | 48 | 192 |
```

These are conservative; users can tune down if they see consistent under-utilisation in `nvidia-smi` peak HBM readings.

---

## 6. Verification

### 6.1 Bit-identity smoke test (must pass before commit)

The whole reason for this change is dispatch refactoring with no math change. Verify:

```python
# verify_chunking.py (or add to existing test file)
import numpy as np
from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER
from lifecycle.model import DiscretizationConfig, SolveControl
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import run_lifecycle_solver

# Same tiny config as verify_smoke.py
disc = DiscretizationConfig(
    n_wealth=20, wealth_min=0.13, wealth_max=200.0,
    n_savings=20,
    state_grid_sizes=(3, 3, 3, 3),
    state_grid_mode="cholesky",
    state_n_stds=(2.0, 2.25, 2.0, 2.25),
    n_z=5, n_eps_nodes=3, n_eta_nodes=3,
    n_ret_nodes_1d=(3, 3),
    n_state_quad_nodes=(2, 3, 2, 3),
)
base = dict(BASE_CONFIG)
base.update(start_age=60, retire_age=63, terminal_age=65)
var = build_nominal_system1_var_config_hardcoded()
model = build_model(base, var, verbose=False)
pc = build_precompute(model, disc, verbose=False)

# Run with no chunking (baseline)
sc1 = CANONICAL_SOLVER._replace(max_iter=100, cell_vmap_chunks=1)
C1, S1, B1, _ = run_lifecycle_solver(model, pc, sc1, verbose=0)

# Run with 4 chunks
sc4 = CANONICAL_SOLVER._replace(max_iter=100, cell_vmap_chunks=4)
C4, S4, B4, _ = run_lifecycle_solver(model, pc, sc4, verbose=0)

# Run with 8 chunks
sc8 = CANONICAL_SOLVER._replace(max_iter=100, cell_vmap_chunks=8)
C8, S8, B8, _ = run_lifecycle_solver(model, pc, sc8, verbose=0)

for label, (C_test, S_test, B_test) in [("4ch", (C4, S4, B4)), ("8ch", (C8, S8, B8))]:
    max_C = float(np.max(np.abs(C1 - C_test)))
    max_S = float(np.max(np.abs(S1 - S_test)))
    max_B = float(np.max(np.abs(B1 - B_test)))
    print(f"{label}: max|ΔC|={max_C:.2e}  max|ΔS|={max_S:.2e}  max|ΔB|={max_B:.2e}")
    assert max_C < 1e-10, f"{label} C deviates"
    assert max_S < 1e-10, f"{label} S deviates"
    assert max_B < 1e-10, f"{label} B deviates"

print("✅ Chunking is bit-identical (≤1e-10 across all configs)")
```

**Expected:** all three deltas ≤ 1e-12 (effectively zero — same FLOPs, same operations, same float64 IEEE behaviour). 1e-10 tolerance is generous.

### 6.2 Production-scale smoke

After the unit test passes, on local CPU (since GPU is for paid runs):

```bash
python -c "
import time
from configs._canonical import CANONICAL_SOLVER
sc = CANONICAL_SOLVER._replace(max_iter=100, cell_vmap_chunks=4)
# ... run smoke ...
t0 = time.time()
... C, S, B, _ = run_lifecycle_solver(...)
print(f'Wall: {time.time()-t0:.1f}s')
"
```

**Compare wall to chunks=1 baseline.** Expected: `chunks=4` is 5-15% slower due to extra dispatch overhead (4× the kernel launches). Anything worse than 30% slower indicates a fusion break that should be investigated.

### 6.3 GPU memory measurement (deferred until next GPU session)

After the change is in, on the next GH200/H200 launch:

1. Run 7⁴ at full canonical quad with `cell_vmap_chunks=16`.
2. Capture peak HBM from `nvidia-smi --query-gpu=memory.used --format=csv` polled during the solve.
3. Confirm peak is `≤ chunk_size × per_cell_memory + persistent_state ≈ 60 GB`.
4. If peak is materially higher than that bound, chunking isn't doing its job — investigate.

This is a **post-merge production verification**, not a pre-commit gate. The bit-identity test (§6.1) is the gate.

---

## 7. Files touched

| File | Change | Approx lines |
|---|---|---|
| [lifecycle/model.py](../../lifecycle/model.py) | Add `cell_vmap_chunks: int = 1` to `SolverConfig` with docstring | ~10 |
| [lifecycle/solver.py](../../lifecycle/solver.py) | Add `_chunked_vmap_cells` helper; modify three `*_vmap_only` builders to use it; add validation in `run_lifecycle_solver` | ~80-100 net |
| `verify_chunking.py` (new) | Bit-identity test from §6.1 | ~50 |
| (no changes to docs/handoff/, no changes to scripts/) | — | — |

Total: ~140-160 lines net across 3 files (one new).

---

## 8. Implementation checklist (for the agent)

- [ ] Read [docs/notes/GPU_TRIAL_FINDINGS.md](../notes/GPU_TRIAL_FINDINGS.md) §"XLA memory planning is non-monotonic" first to understand why this is needed.
- [ ] Read existing `_*_vmap_only` builders end-to-end before touching them. They're in `lifecycle/solver.py` around lines 1424, 1635, 1795 (line numbers may have shifted post rtb-as-state).
- [ ] Add `cell_vmap_chunks: int = 1` to `SolverConfig` per §4.1.
- [ ] Add `_chunked_vmap_cells` helper near the existing builders per §4.2.
- [ ] Update each of three vmap-only builders per §4.3:
  - Terminal: inline chunking (different signature than retire/work).
  - Retirement + Working: use `_chunked_vmap_cells`.
- [ ] Add validation `if n_chunks < 1: raise ValueError(...)` in `run_lifecycle_solver`.
- [ ] Write `verify_chunking.py` per §6.1.
- [ ] Run `python verify_chunking.py`. Expected: all three deltas ≤ 1e-10.
- [ ] Run `python verify_smoke.py` to confirm no regression at default `cell_vmap_chunks=1`.
- [ ] Commit:
  ```
  solver: cell-axis vmap chunking for memory-bounded single-GPU runs

  Adds SolverConfig.cell_vmap_chunks (default 1, no-op).
  When > 1, splits the per-age vmap over (n_z * N_state) cells into K
  sequential vmap calls, padding to fixed chunk_size so XLA traces once.
  Per-chunk peak HBM = (worst-case full materialization / K), giving
  deterministic memory bound independent of XLA's planning lottery.

  Motivation: §6.6 risk from HANDOFF_PMAP_TO_VMAP.md materialised in
  production (see docs/notes/GPU_TRIAL_FINDINGS.md). At 7⁴ × full quad
  on GH200, XLA requested 1.06 TB despite the algorithm needing only
  67 GB. Chunking with K=16 caps per-chunk peak at ~60 GB, fits 97 GB
  HBM with margin.

  Verified: chunks=1 vs chunks={4,8} bit-identical (max delta ≤ 1e-12)
  on the 6-age smoke. 5⁴ runs at default chunks=1 unchanged.
  ```
- [ ] Push to `jax-rewrite`. No PR review required unless reviewer requests.
- [ ] Report back with the commit SHA and confirmation that verify_chunking.py + verify_smoke.py both pass. Stop.

---

## 9. Performance expectations

**Bit-identical math, slight dispatch overhead.**

- **Default `cell_vmap_chunks=1`:** zero performance change.
- **`cell_vmap_chunks=K>1`:** wall increase of ~K extra kernel launches per per-age call. On GPU, kernel launch is ~5-50 µs each. For K=16 at 33 ages: ~16 × 33 × 25 µs = 13 ms total extra dispatch per run. **Negligible** vs 273 s/age compute.
- **Memory:** worst-case peak HBM falls from `(per_cell × n_cells)` to `(per_cell × chunk_size)`. Linear in `1/K`.
- **Compile time:** first chunk JITs the inner vmap; chunks 2..K reuse the trace. **No additional compile cost from chunking** beyond the existing first-age JIT cost.
- **Persistent cache compatibility:** the chunked vmap traces are different from the unchunked traces (they're different XLA computations). Cache keyed by `(hardware, jax_version, trace_hash)` so chunked and unchunked runs don't share cache entries. **Bumping `cell_vmap_chunks` requires fresh JIT compile** the first time. Subsequent runs at the same `cell_vmap_chunks` value reuse cache.

---

## 10. Out of scope / explicit non-goals

- **Multi-GPU sharding** via `jax.sharding.NamedSharding` — separate handoff. Single-device chunking is the prerequisite for multi-GPU work, but multi-GPU itself is its own surgery.
- **Auto-selection of `n_chunks`** based on hardware queries. User picks; we don't try to be clever.
- **Chunking at granularities other than the cell axis** (e.g. chunking the savings vmap inside `_egm_scan_cell`). The cell axis is where the memory pressure is. Don't touch the savings vmap.
- **Performance optimization of chunked dispatch** — overlapping kernel launches, async streams, etc. Goal here is correctness + bounded memory; perf tuning is downstream.
- **Modifying the pmap path.** It already pads to multi-of-n_dev and unwinds — that's a different chunking pattern for a different reason. Leave it alone.
- **Removing the unchunked path.** `cell_vmap_chunks=1` is the default and stays the fast path for small configs where chunking overhead is wasteful. Don't delete it.

---

## 11. Why this is single-day-scope and worth the focused handoff

The §6.6 risk has been on the radar since the pmap→vmap handoff. Tonight made it concrete: **without chunking, the codebase has a hard ceiling at 5⁴ on a single GPU, regardless of HBM size.** That's a thesis-quality blocker.

The implementation is mechanical — fixed pattern from §4.2-§4.3, three call sites, one bit-identity test. The risk is small (math doesn't change), the verification gate is sharp (bit-identity), and the payoff unblocks 7⁴ + 9⁴ runs.

A focused agent can land this in a single day. After it lands, the wall-time complexity estimator's projections at 7⁴ become actionable (vs hypothetical), and the next GPU launch can target 7⁴ with confidence rather than crossing fingers about XLA's compilation planner.

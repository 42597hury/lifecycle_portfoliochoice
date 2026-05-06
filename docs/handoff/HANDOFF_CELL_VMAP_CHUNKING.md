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
- A small helper `_chunked_vmap_runner(jit_chunk_fn, z_idx_padded, is_idx_padded, n_cells, chunk_size, n_chunks)` factored once and called from all three builders. Returns a Python-level runner that does the chunk loop *outside* `@jit`. (Earlier draft had `_chunked_vmap_cells(per_cell_fn, ...)` with the loop inside `@jit`; that's the pattern that bit us — see §4.2.)
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

**Critical: the chunk loop must run in Python, *outside* any `@jit` boundary.**
If the loop sits inside an `@jit`'d function, Python unrolls it at trace
time and XLA emits one HLO graph spanning all chunks — its scheduler is then
free to materialise multiple chunks' working memory concurrently, which
produces correct math (output is bit-identical) but **defeats the memory
bound entirely**. Each chunk must be a separate `@jit` call: the for-loop
runs in Python, each iteration calls the per-chunk JIT with that chunk's
slice of the index arrays, and `.block_until_ready()` between chunks forces
the previous chunk's intermediates to be released before the next chunk
schedules its allocations.

The helper in `lifecycle/solver.py` (near the `_*_vmap_only` builders)
returns a *runner* that the kernel builder calls in its `call()` function:

```python
def _chunked_vmap_runner(jit_chunk_fn, z_idx_padded, is_idx_padded,
                          n_cells, chunk_size, n_chunks):
    """Build a Python-level chunk-loop runner around a JIT'd per-chunk vmap.

    The chunk loop runs in Python (outside any @jit boundary). Each iteration
    is a separate JIT call; .block_until_ready() between chunks forces XLA
    to release the previous chunk's intermediates before the next allocates.

    Parameters
    ----------
    jit_chunk_fn : @jit-compiled callable
        Signature: jit_chunk_fn(*kernel_args, z_chunk, is_chunk) -> tuple,
        each output entry has leading axis chunk_size.
    z_idx_padded, is_idx_padded : jnp.ndarray, shape (chunk_size * n_chunks,)
        Pre-padded index arrays from _build_chunked_index_arrays.
    n_cells, chunk_size, n_chunks : int

    Returns
    -------
    runner(*kernel_args) -> tuple of (n_cells, ...) arrays.
    """
    def runner(*kernel_args):
        chunk_results = []
        for i in range(n_chunks):
            start = i * chunk_size
            z_chunk = z_idx_padded[start:start + chunk_size]
            is_chunk = is_idx_padded[start:start + chunk_size]
            out = jit_chunk_fn(*kernel_args, z_chunk, is_chunk)
            out[0].block_until_ready()    # bounds peak HBM at one chunk
            chunk_results.append(out)
        n_outs = len(chunk_results[0])
        return tuple(
            jnp.concatenate([r[k] for r in chunk_results], axis=0)[:n_cells]
            for k in range(n_outs)
        )
    return runner
```

**Critical design decisions:**

1. **Chunk for-loop runs in Python (outside `@jit`).** This is the load-bearing
   choice. An earlier version placed the loop inside `@jit def all_cells(...)`,
   which produced bit-identical math but didn't bound memory at runtime — XLA's
   scheduler fused all chunks into one graph. The fix is `_chunked_vmap_runner`'s
   Python-level loop, with `.block_until_ready()` between chunks.
2. **`n_chunks` is a Python int, not a traced value.** Closure constant. XLA
   traces the inner vmap (`per_chunk`) once and the per-chunk @jit reuses
   that trace for all K iterations.
3. **Padding pattern matches the existing pmap path**
   ([_build_per_age_*_kernel_pmap](../../lifecycle/solver.py#L1565)). Same
   pattern, same justification: pad with the last cell index, slice off after.
4. **Per-builder `per_chunk` is a separate `@jit` function** that wraps the
   per-cell vmap. This is what gets called K times in the Python loop.
5. **Single-chunk fast path shares the same `per_chunk` JIT** as the K>1
   path. When `cell_vmap_chunks=1`, the builder's `call()` invokes
   `per_chunk` once with the full unpadded `(z_idx, is_idx)` slices —
   no chunk-loop wrapper, no inter-chunk block. **Sharing one JIT trace
   between K=1 and K>1 is load-bearing for bit-identity:** an earlier
   draft used a separate `@jit'd all_cells` for the K=1 fast path,
   which produced output that differed from the K>1 path by ~1e-3 on
   alphas (XLA's fusion pass made structurally-different traces
   compute the same algorithm with slightly different op ordering, and
   Newton at `tol=1e-7 * scale` amplified the difference). Use one
   `per_chunk @jit` for both paths.

### 4.3 Wiring into the three vmap-only kernel builders

#### Terminal kernel ([_build_per_age_terminal_kernel_vmap_only](../../lifecycle/solver.py#L1424))

Terminal vmaps over `N_state` (no z axis). It chunks over the padded
`(log_R_bill, log_x_s, log_x_b, ann)` tensors directly, not over index
arrays — so it inlines its own Python-level chunk loop rather than reusing
`_chunked_vmap_runner`:

```python
def _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc):
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter, ...,
              bool(sc.use_fori_newton))

    # ... existing log_R_bill / log_x_s / log_x_b / ann prep + numpy-padding,
    # unchanged. After this block log_R_bill_jnp, log_x_s_jnp, log_x_b_jnp,
    # ann_jnp are jnp arrays of leading axis (chunk_size * n_chunks).

    n_chunks = int(sc.cell_vmap_chunks)
    N_state = log_R_bill.shape[0]

    def per_i_s(log_Rb, lxs, lxb, A):
        return _solve_terminal_at_i_s(
            log_Rb, lxs, lxb, pcj.weight_kv_kr, A,
            pcj.s_grid, pcj.wealth_grid,
            init_a_s, init_a_b,
            ...,
            *static,
        )

    if n_chunks == 1:
        # Fast path: one @jit'd vmap, no chunk-loop wrapper.
        @jit
        def all_is():
            return vmap(per_i_s)(log_R_bill_jnp, log_x_s_jnp, log_x_b_jnp, ann_jnp)
        def call(_unused_age_idx=None):
            return all_is()
        return call

    # Chunked path: per_chunk @jit, Python-level loop in call(), block_until_ready.
    @jit
    def per_chunk(log_Rb_c, lxs_c, lxb_c, ann_c):
        return vmap(per_i_s)(log_Rb_c, lxs_c, lxb_c, ann_c)

    def call(_unused_age_idx=None):
        chunk_results = []
        for i in range(n_chunks):
            start = i * chunk_size
            log_Rb_c = log_R_bill_jnp[start:start + chunk_size]
            lxs_c    = log_x_s_jnp   [start:start + chunk_size]
            lxb_c    = log_x_b_jnp   [start:start + chunk_size]
            ann_c    = ann_jnp       [start:start + chunk_size]
            out = per_chunk(log_Rb_c, lxs_c, lxb_c, ann_c)
            out[0].block_until_ready()      # bound peak HBM at one chunk
            chunk_results.append(out)
        n_outs = len(chunk_results[0])
        return tuple(
            jnp.concatenate([r[k] for r in chunk_results], axis=0)[:N_state]
            for k in range(n_outs)
        )
    return call
```

**Note:** terminal slices four padded jnp tensors with Python-int starts
(static slices, eager). The runner from §4.2 doesn't directly apply because
its signature is geared to (z_idx, i_s) chunking. Inline is simpler given
there are only three call sites total.

#### Retirement kernel ([_build_per_age_retirement_kernel_vmap_only](../../lifecycle/solver.py#L1635))

Use `_chunked_vmap_runner` from §4.2 — the runner is built once at kernel-builder
time and called from `call()` with the per-age kernel args:

```python
def _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state, per_is_tensors):
    ...
    n_chunks = int(sc.cell_vmap_chunks)
    z_idx_padded, is_idx_padded, n_cells, chunk_size = _build_chunked_index_arrays(
        n_z, N_state, n_chunks,
    )

    def per_cell(z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
                  init_a_s_arr, init_a_b_arr):
        # ... existing per-cell body, with kernel args now passed positionally
        # rather than closed-over (so per_chunk can vmap with in_axes=(0,0,None,...)).
        return _solve_retirement_at_cell(z_idx, i_s, c_next, ...)

    # per_chunk: ONE @jit shared by both K=1 fast path and K>1 chunked path.
    # Sharing one trace guarantees bit-identical math. Separate fast-path
    # @jit'd `all_cells` re-introduces drift via XLA's fusion pass.
    @jit
    def per_chunk(c_next, pension_next_by_z, psi_per_z,
                   init_a_s_arr, init_a_b_arr, z_chunk, is_chunk):
        return vmap(
            per_cell, in_axes=(0, 0, None, None, None, None, None),
        )(z_chunk, is_chunk, c_next, pension_next_by_z, psi_per_z,
          init_a_s_arr, init_a_b_arr)

    if n_chunks == 1:
        # Fast path: one per_chunk call with the full unpadded indices.
        z_full = z_idx_padded[:n_cells]
        is_full = is_idx_padded[:n_cells]
        def call(c_next_jnp, pension_next_by_z, psi_per_z,
                  init_a_s_arr, init_a_b_arr):
            c_flat, s_flat, b_flat, ni_flat, nb_flat = per_chunk(
                c_next_jnp, pension_next_by_z, psi_per_z,
                init_a_s_arr, init_a_b_arr, z_full, is_full,
            )
            return (jnp.reshape(c_flat, (n_z, N_state, -1)), ...)
        return call

    # Chunked path: same per_chunk, called K times via the runner.
    runner = _chunked_vmap_runner(
        per_chunk, z_idx_padded, is_idx_padded, n_cells, chunk_size, n_chunks,
    )

    def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
        c_flat, s_flat, b_flat, ni_flat, nb_flat = runner(
            c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr,
        )
        return (
            jnp.reshape(c_flat, (n_z, N_state, -1)),
            jnp.reshape(s_flat, (n_z, N_state, -1)),
            jnp.reshape(b_flat, (n_z, N_state, -1)),
            jnp.reshape(ni_flat, (n_z, N_state)),
            jnp.reshape(nb_flat, (n_z, N_state)),
        )
    return call
```

The output shapes are unchanged from today. The runner returns
`(n_cells, n_w)` arrays with padding sliced off; the reshape is identical.

#### Working kernel ([_build_per_age_working_kernel_vmap_only](../../lifecycle/solver.py#L1795))

Same pattern as retirement. The `per_cell` body is more complex (handles
`use_pension_next` branch + per-cell `z_next` bracket arithmetic) and takes
two extra kernel args (`income_next_table_z`, `pension_next_by_z`), so the
`vmap` `in_axes` tuple grows by two `None`s. The chunking call is otherwise
identical:

```python
runner = _chunked_vmap_runner(
    per_chunk, z_idx_padded, is_idx_padded, n_cells, chunk_size, n_chunks,
)
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

Tempting: `lax.scan` over chunks would be a single trace boundary instead of K. **But** scan runs its body inside `@jit` (it's a JAX primitive), which has the same memory-bounding pathology as putting the for-loop inside `@jit`: XLA emits one HLO graph spanning all K iterations and may schedule their working memory concurrently. **Don't use scan.** The Python for-loop with `.block_until_ready()` between chunks is the only pattern that gives a real per-chunk memory bound.

### 5.11 Chunks-outside-JIT is load-bearing — verify before trusting

If you ever refactor or simplify the chunk dispatch, **the for-loop must stay in Python, outside any `@jit`-traced function.** A clean-looking refactor that moves the loop inside `@jit` (e.g. into an `all_cells` wrapper, or into `lax.scan`/`lax.fori_loop`) will pass `verify_chunking.py`'s bit-identity tests and silently break the memory bound. The runtime memory test in `verify_chunking.py` (post-fix) is the gate that catches this regression — run it at production scale on the next GPU launch and confirm `nvidia-smi` peak HBM tracks `chunk_size * per_cell + persistent_state`, not the unchunked worst case.

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

### 6.1.1 Memory-bound smoke (added by post-fix patch)

Bit-identity proves the math is unchanged but says nothing about whether chunking actually bounds memory. Add a runtime memory test that:

1. Picks a config large enough that K=1 produces a substantial allocation, but small enough to run on local CPU within a few minutes.
2. Runs the solve at K=1 and K=4 with `psutil` (or any available process-RSS probe) sampling peak resident memory.
3. Asserts K=4 peak RSS ≤ ~K=1 peak RSS / 2 (generous margin — the absolute bound is per-chunk-working-memory + persistent-state).
4. As a fallback when `psutil` isn't available, asserts that a config 4× the bit-identity smoke runs successfully at K=4 — proving at minimum that the chunking call path doesn't crash on a non-trivial workload.

This sits in the same `verify_chunking.py` file as the bit-identity tests. Both run on local CPU and form the pre-commit gate.

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
| [lifecycle/solver.py](../../lifecycle/solver.py) | Add `_chunked_vmap_runner` helper; modify three `*_vmap_only` builders so each chunk is its own `@jit` and the chunk loop runs in Python (in `call()`); add validation in `run_lifecycle_solver` | ~80-100 net |
| `verify_chunking.py` (new) | Bit-identity test from §6.1 | ~50 |
| (no changes to docs/handoff/, no changes to scripts/) | — | — |

Total: ~140-160 lines net across 3 files (one new).

---

## 8. Implementation checklist (for the agent)

- [ ] Read [docs/notes/GPU_TRIAL_FINDINGS.md](../notes/GPU_TRIAL_FINDINGS.md) §"XLA memory planning is non-monotonic" first to understand why this is needed.
- [ ] Read existing `_*_vmap_only` builders end-to-end before touching them. They're in `lifecycle/solver.py` around lines 1424, 1635, 1795 (line numbers may have shifted post rtb-as-state).
- [ ] Add `cell_vmap_chunks: int = 1` to `SolverConfig` per §4.1.
- [ ] Add `_chunked_vmap_runner` helper near the existing builders per §4.2 (Python-level chunk loop, `.block_until_ready()` between chunks).
- [ ] Update each of three vmap-only builders per §4.3:
  - Terminal: inline Python chunk loop (different chunked-arg signature than retire/work).
  - Retirement + Working: use `_chunked_vmap_runner`. Restructure `per_cell` to take the kernel args positionally; define `per_chunk` as a separate `@jit` wrapping `vmap(per_cell, in_axes=(0, 0, None, ...))`. Keep the `n_chunks == 1` fast path (single `@jit'd vmap`, no runner wrapper).
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

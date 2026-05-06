# Handoff: Multi-GPU Code Audit + Chunking-Port to pmap Path

**Branch:** `jax-rewrite`
**Effort:** ~1.5 days (half-day audit + 1 day chunking port + validation).
**Output:**
- `docs/scans/MULTI_GPU_AUDIT_2026-05-07.md` — audit findings (Phase A).
- New commit(s) on `jax-rewrite` porting chunking to the three `_pmap` kernel builders (Phase B).

**Pre-confirmed finding:** spot-check on 2026-05-07 confirmed Concern A (chunking-fix coverage) is **RED**. Float's chunking fix landed in `_vmap_only` only — the three `_pmap` builders at [solver.py:1556, 1820, 2015](../../lifecycle/solver.py#L1556) do NOT use `cell_vmap_chunks`. This means 5⁴ on multi-GPU works (per-device cell working set ~31 GB fits any H100+) but 7⁴ canonical OOMs (per-device ~119 GB exceeds H200's 141 GB once XLA scheduler flips). **Phase B fixes this.**

---

## Background

The pmap path is *reportedly* already in place: every kernel builder has a `_pmap` variant that dispatches when `len(jax.devices()) > 1`. Specific evidence:

- [solver.py:1453-1465](../../lifecycle/solver.py#L1453) — header comment for the pmap+vmap section.
- [solver.py:1549](../../lifecycle/solver.py#L1549) — `_build_per_age_terminal_kernel(pcj, mp, sc, n_dev)` dispatches `n_dev==1 → vmap_only`, else `pmap`.
- [solver.py:1809, 2004](../../lifecycle/solver.py#L1809) — same pattern for retirement and working kernels.
- [solver.py:2290-2295](../../lifecycle/solver.py#L2290) — stdout prints `"Cell-batching pattern: pmap+vmap (N devices)"` or `"vmap-only (single-device)"`.
- [`__init__.py:43-49`](../../lifecycle/__init__.py#L43) — `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` opt-out for GPU runs documents the intended workflow.

**But:** several recent code changes (the cell-vmap chunking fix, mixed-precision plumbing, Newton iter exposure, backward-age warm-start arrays) all touched these kernel builders. Each landed touching primarily the `_vmap_only` paths. **It's unclear whether all of them also threaded through the `_pmap` paths.**

The audit's job is to confirm the pmap path is **shippable today** before we spend money on a multi-GPU instance and discover a regression at runtime.

---

## What's at stake

Capturing the multi-GPU speedup unlocks **~2× wall on 2× device, ~6-7× on 8× device** at canonical scale — the difference between "spend $77 on 7⁴ canonical" and "spend $11" on a Lambda 8× instance. Phase B's chunking port is the gating fix to make this work at canonical scale.

If audit (Phase A) comes back with no other reds: Phase B's chunking port is the only code change needed before the Phase C runbook (separate handoff, mechanical: spin instance, run, capture).

If audit surfaces additional reds: list them in the scan report; this handoff still does Phase B (chunking port) but additional reds get their own follow-up handoffs.

---

## Phase A — Audit checklist

### A. Chunking fix coverage (`cell_vmap_chunks` from `SolverConfig`) — KNOWN RED

**Pre-confirmed:** chunking lives in the three `_vmap_only` builders ([solver.py:1735, 1920, 2142](../../lifecycle/solver.py#L1735)) but NOT in the three `_pmap` builders ([solver.py:1556, 1820, 2015](../../lifecycle/solver.py#L1556)). Phase B ports the fix.

The audit report should still document:
- Which exact lines in each pmap builder need the chunking wrapper (so Phase B has clear targets).
- Whether the existing `_chunked_vmap_runner` and `_build_chunked_index_arrays` helpers ([solver.py:1626, 1662](../../lifecycle/solver.py#L1626)) are reusable as-is for the pmap path or need a parallel `_chunked_pmap_runner` wrapper.
- Cell-axis padding interaction: pmap pads to `pad_n = ceil(N_state / n_dev) * n_dev`. Chunking on top must pad to `lcm(n_chunks, n_dev)` or chunk-then-shard to avoid double-padding bookkeeping.

### B. Mixed-precision plumbing

**Concern:** `gather_precision` threading via `_cast_for_gather` and `_resolve_gather_dtype` was added recently. Need to confirm both paths thread it correctly.

**What to verify:**

1. Find the cast sites:
   ```bash
   grep -n "_cast_for_gather\|gather_dtype\|_resolve_gather_dtype" lifecycle/solver.py
   ```
2. For each kernel builder pair (terminal/retirement/working × pmap/vmap_only), confirm:
   - `gather_dtype` is resolved from `mp` config and used identically in both variants.
   - The mixed-fp32-gather+fp64-arithmetic boundary is at the same point in both code paths.
3. The mixed-precision verify script ([verify_mixed_precision.py](../../verify_mixed_precision.py) or [verify_mixed_precision_tiny.py](../../verify_mixed_precision_tiny.py)) currently runs at `n_dev=1`. **Optional: add a multi-device smoke** by setting `XLA_FLAGS=--xla_force_host_platform_device_count=2` and re-running tiny gate; confirm fp32-gather + fp64-FOC bit-identity holds across the pmap dispatch boundary.

### C. Warm-start arrays under sharding

**Concern:** the backward-age warm-start added `init_a_s_arr` and `init_a_b_arr` arrays passed into Newton via the kernel. In the pmap path these need to be sharded across devices the same way `c_next` is.

**What to verify:**

1. In `_build_per_age_retirement_kernel_pmap` and `_build_per_age_working_kernel_pmap`, find where `init_a_s_arr` and `init_a_b_arr` are reshaped or threaded into the per-device kernel. They should be reshaped to `(n_dev, per_dev, ...)` like the cell-axis tensors.
2. Confirm `in_axes` for these arrays in the `pmap(...)` call is `0` (sharded along the cell axis), not `None` (replicated). If they're replicated, every device sees all warm-start values — wrong policy results, no error.
3. Cross-check the call sites in the orchestrator ([solver.py:2330+](../../lifecycle/solver.py#L2330)): the orchestrator passes the warm-start arrays into the kernel; confirm shapes match what the pmap kernel expects.

### D. Cell-axis padding edge cases

**Concern:** [solver.py:1572, 1846, 2040](../../lifecycle/solver.py#L1572) all pad to `pad_n = ceil(n_cells / n_dev) * n_dev`. The padding cells are dummies; their policy outputs must NOT pollute the real policy.

**What to verify:**

1. Confirm the orchestrator strips padding cells in the result collapse. Look at the `(n_dev, per_dev, n_w) -> (pad_n, n_w) -> (n_cells, n_w)` reshape (e.g. [solver.py:1883](../../lifecycle/solver.py#L1883)). The slice should be `[:n_cells]`, not all of `pad_n`.
2. Test edge case: `n_cells = 6875` (5⁴ × 11), `n_dev = 8` → `pad_n = 6880`, 5 padding cells. Trace through what happens to those 5 cells in `init_a_s_arr` (zero? NaN? whatever junk lives there). Padding values being NaN might propagate into Newton via warm-start and contaminate adjacent device's cells via JIT fusion (unlikely but worth checking).
3. **Easy mitigation if needed:** zero-fill padding cells explicitly in the orchestrator before reshape. Add a comment.

### E. Smoke-trace at n_dev=4 (CPU virtual devices)

This is the **single most important sanity check** in Phase A — it actually exercises the pmap path on machines we already have, before paying for a multi-GPU instance.

**What to do:**

1. Unset `LIFECYCLE_DISABLE_VIRTUAL_CPUS` (or set it to `0`).
2. Set `XLA_FLAGS=--xla_force_host_platform_device_count=4`. (Not 2 — odd-vs-even might mask edge cases.)
3. Run [verify_smoke.py](../../verify_smoke.py).
4. Expected stdout:
   ```
   Cell-batching pattern: pmap+vmap (4 devices)
   ```
5. Compare alpha ranges to a single-device smoke run (set `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1`). They should match to bit-identity (same FP, same algorithm, just dispatched differently).

**Pass criteria:** smoke completes, prints `pmap+vmap (4 devices)`, alphas bit-identical to single-device smoke.

**Failure modes to watch for:**
- ImportError or shape mismatch → some kernel signature got out of sync between paths.
- Run completes but alphas differ → sharding bug in warm-start or padding contamination.
- "RuntimeError: pmap requires same shape across devices" → padding logic regressed.

### F. Persistent JIT cache

**Concern:** cache keys include device count. First run on N devices is always cold. Worth knowing the per-device-count cache size implication.

**What to verify (light):**

1. Cache directory: confirm `LIFECYCLE_JAX_CACHE_DIR` points somewhere with enough headroom (`du -sh ~/.cache/jax_lifecycle` if it exists). With multiple device counts compiled (n_dev=1, 2, 4, 8) the cache could grow to 2-4 GB.
2. The compile cache S3 sync ([scripts/_compile_cache_sync.py](../../scripts/_compile_cache_sync.py)) — confirm it doesn't do anything device-count-specific that would break round-tripping a multi-GPU compile.

---

---

## Phase B — Port chunking to the three `_pmap` builders

### B.1 Goal

Make `cell_vmap_chunks > 1` work under the pmap path with the same memory-bounding semantics as the vmap-only path: peak per-device HBM scales with `chunk_size / n_dev`, not with the full per-device cell count. Bit-identity vs vmap-only at `chunks=1`; bit-identity vs pmap-no-chunking at `chunks=1` on multi-device.

### B.2 Architectural choice — chunk OUTSIDE pmap, not inside

**Anti-pattern to avoid:** putting a Python `for chunk in range(...)` loop *inside* the function body that gets pmap'd. pmap implies `@jit` over the function body, so the chunk loop unrolls at trace time and XLA materialises the whole graph anyway — same bug we just fixed in vmap-only.

**Correct pattern:** chunk loop in the outer (non-pmap'd) Python wrapper. Each chunk is its own `pmap`-compiled kernel call:

```python
def chunked_pmap_caller(jit_pmap_kernel, all_cell_indices, n_dev, chunk_size, ...):
    n_cells_padded = ((n_cells + chunk_size - 1) // chunk_size) * chunk_size
    results = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        cell_chunk = all_cell_indices[start:end]      # (chunk_size,)
        # Reshape for pmap dispatch: (n_dev, chunk_size_per_device)
        per_dev_chunk = chunk_size // n_dev           # require chunk_size % n_dev == 0
        sharded = cell_chunk.reshape(n_dev, per_dev_chunk)
        chunk_result = jit_pmap_kernel(sharded, ...)  # block per chunk
        results.append(chunk_result.reshape(chunk_size, ...))  # collapse pmap axis
    return jnp.concatenate(results, axis=0)[:n_cells]  # strip padding
```

This means `chunk_size` must be a multiple of `n_dev`. Easy to enforce: `chunk_size = ceil(n_cells / n_chunks)` then round up to the nearest multiple of `n_dev`. Document the constraint in the SolverConfig docstring so users picking `cell_vmap_chunks` don't pick a value that produces a chunk_size that's not n_dev-aligned (or have the orchestrator round transparently).

### B.3 Implementation steps

1. **Add a helper `_chunked_pmap_runner`** in [solver.py](../../lifecycle/solver.py) near the existing `_chunked_vmap_runner` (line 1662). Mirror its docstring style. This helper takes a pmap-compiled per-device kernel and dispatches it per chunk, blocking between chunks for memory hygiene (use `.block_until_ready()` after each chunk's result, like `_chunked_vmap_runner` does).

2. **Refactor each `_pmap` builder** to:
   - Construct the JIT'd per-device kernel ONCE at build time (lift it out so it can be reused across chunks).
   - Return a Python callable that, when invoked with the age's inputs, runs `_chunked_pmap_runner` over the cell axis.
   - Match the existing `_pmap` builder's return signature so the orchestrator sees the same interface.

3. **Update [solver.py:2330+](../../lifecycle/solver.py#L2330)** (the orchestrator) only if needed — ideally the kernel-builder return signature is unchanged, so the orchestrator doesn't notice.

4. **Update the cell-padding logic.** Currently pmap pads to `pad_n = ceil(N_state / n_dev) * n_dev`. With chunking the pad target becomes `pad_n = ceil(N_state / chunk_size) * chunk_size`, with `chunk_size` a multiple of `n_dev`. Make sure the result-collapse strips the right number of cells.

5. **Mirror mixed-precision plumbing** through the chunked pmap kernel — the `gather_dtype` static config must propagate identically to the vmap-only chunked path.

6. **Mirror Newton-iter histogram plumbing** through the chunked pmap kernel — the `n_iters_max` and `n_backtrack_total` per-cell scalars need to come back from each chunk and be concatenated, same as the policies. Don't break the diagnostics that just landed.

### B.4 Validation gates

Run each gate sequentially. If any fails, do not proceed to the next.

**Gate 1 — single-device chunks=1 bit-identity:**
```bash
LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 python verify_smoke.py
# Capture alphas. Set cell_vmap_chunks=1 explicitly.
```
Compare to baseline (pre-this-change). Bit-identity required.

**Gate 2 — single-device chunks=4 bit-identity vs chunks=1:**
```bash
LIFECYCLE_DISABLE_VIRTUAL_CPUS=1 python verify_chunking.py
```
This script already exercises chunking on the vmap-only path; should still pass post-change.

**Gate 3 — n_dev=4 chunks=1 bit-identity vs single-device chunks=1:**
```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 python verify_smoke.py
# cell_vmap_chunks=1, expect "pmap+vmap (4 devices)" in stdout
```
Alphas should match the Gate 1 single-device run bit-identically.

**Gate 4 — n_dev=4 chunks=4 bit-identity vs n_dev=4 chunks=1:**
Modify verify_smoke.py temporarily (or write a new verify_pmap_chunking.py mirroring verify_chunking.py) to set `cell_vmap_chunks=4`. Run with `XLA_FLAGS=--xla_force_host_platform_device_count=4`. Alphas match Gate 3 bit-identically. **This is the headline gate** — proves chunking under pmap doesn't break correctness.

**Gate 5 — memory test on n_dev=4 chunks=4:**
Optional but recommended: monitor RSS during Gate 4 vs Gate 3. Peak RSS at chunks=4 should be ~lower than chunks=1 (4× chunks = ~1/4 peak working set). Confirms the memory bound actually bites.

### B.5 Out of scope (do NOT do in this handoff)

- **Multi-host dispatch (jax.distributed):** distinct from multi-device-on-one-host. Out of scope.
- **Hierarchical sharding (cell + state-quad axes):** the existing pmap shards the cell axis only; that's enough for the speedup we want. Don't add state-quad sharding here.
- **Phase C runbook (actually running on GPU instance):** separate handoff, dispatched after this one lands.

---

## Validation

After running each audit step in Phase A, write findings into `docs/scans/MULTI_GPU_AUDIT_2026-05-07.md` with one of three verdicts per check:

- **GREEN:** code is correct, no work needed.
- **YELLOW:** code is correct but should be hardened (e.g. add explicit padding zero-fill).
- **RED:** code is broken on multi-GPU, must patch before Phase B.

For any RED, also include:
- File:line of the bug
- Proposed fix (1-3 sentences, no implementation in this handoff)
- Estimated effort

If all checks come back GREEN or YELLOW, the report ends with: *"Cleared for Phase B (multi-GPU runbook)."*

---

## Implementation checklist

### Phase A — Audit
- [ ] (A) Document chunking-coverage gap with exact line refs per builder (RED is pre-confirmed).
- [ ] (B) Audit mixed-precision threading in all three pmap kernel builders.
- [ ] (C) Audit warm-start array sharding (`init_a_s_arr`, `init_a_b_arr`) in retirement + working pmap builders.
- [ ] (D) Audit cell-axis padding strip in collapse logic for all three.
- [ ] (E) Run `XLA_FLAGS=--xla_force_host_platform_device_count=4 python verify_smoke.py`. Confirm `pmap+vmap (4 devices)` prints and alphas match `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` smoke bit-identically.
- [ ] (F) Light check on cache headroom and S3 sync.
- [ ] Write `docs/scans/MULTI_GPU_AUDIT_2026-05-07.md` with per-check verdicts.
- [ ] Commit Phase A separately:
  ```
  docs: multi-GPU code audit — Phase A scan results
  
  Audits the pmap path of every kernel builder against recent changes
  (chunking, mixed-precision, warm-start, padding). Confirms the
  pre-spotted RED on chunking coverage and reports the verdict for
  the other five checks. Phase B (chunking port) lands as a follow-up
  in this same handoff.
  
  Verdict: chunking RED (pre-known); <other checks>: <GREEN | YELLOW | RED>.
  ```

### Phase B — Chunking port
- [ ] Add `_chunked_pmap_runner` helper alongside `_chunked_vmap_runner`.
- [ ] Port chunking into `_build_per_age_terminal_kernel_pmap` ([solver.py:1556](../../lifecycle/solver.py#L1556)).
- [ ] Port chunking into `_build_per_age_retirement_kernel_pmap` ([solver.py:1820](../../lifecycle/solver.py#L1820)).
- [ ] Port chunking into `_build_per_age_working_kernel_pmap` ([solver.py:2015](../../lifecycle/solver.py#L2015)).
- [ ] Update padding logic to `pad_n = ceil(N_state / chunk_size) * chunk_size` with `chunk_size` enforced as multiple of `n_dev`.
- [ ] Thread `gather_dtype` (mixed precision) through the chunked pmap path identically to chunked vmap-only.
- [ ] Thread Newton-iter histogram (`n_iters_max`, `n_backtrack_total`) through chunked pmap path identically.
- [ ] Run **Gate 1**: single-device chunks=1, bit-identity vs pre-change.
- [ ] Run **Gate 2**: single-device chunks=4 via `verify_chunking.py`, bit-identity vs chunks=1.
- [ ] Run **Gate 3**: n_dev=4 chunks=1, bit-identity vs Gate 1.
- [ ] Run **Gate 4**: n_dev=4 chunks=4 (via temporary edit or new `verify_pmap_chunking.py`), bit-identity vs Gate 3. **Headline gate.**
- [ ] (Optional) **Gate 5**: RSS profiling at chunks=4 vs chunks=1, confirm peak working set drops.
- [ ] Commit Phase B:
  ```
  solver: port cell-vmap chunking to the three _pmap kernel builders
  
  Float's chunk-outside-JIT fix landed in _vmap_only only; this commit
  ports it to _pmap so multi-GPU runs at canonical scale (7^4) don't
  OOM under XLA's non-monotonic memory planner. Architecture: chunk
  loop in the outer Python wrapper around pmap, each chunk its own
  pmap-compiled kernel call, .block_until_ready() between chunks.
  Bit-identity verified on n_dev=4 CPU virtual devices at chunks=1
  vs chunks=4. No math change. Prerequisite for Phase C runbook
  (multi-GPU benchmark on Lambda 2x/8x).
  ```
- [ ] Push.

---

## Why this is a separate handoff (not part of any other)

- **Distinct from float's chunking fix:** float was scoped to fix `_chunked_vmap_cells` in the vmap-only path. This handoff explicitly extends it to the pmap path and audits the wider multi-device implications.
- **Distinct from the Phase C runbook (separate, future handoff):** Phase C is mechanical (spin Lambda 2× or 8×, run benchmark, capture wall). It can't start until Phase B (chunking port) lands. Bundling them would conflate code work with compute work and create a runbook that could spend GPU dollars debugging.
- **Distinct from sim-EE work:** sim-EE doesn't touch the solver kernel paths.

---

## Dependencies

- **Must wait for:** float agent's chunk-outside-JIT fix to land on `jax-rewrite`. Audit step (A) is meaningless against pre-fix code.
- **Does NOT block:** sim-EE diagnostic, arbitrage diagnostic, future bundle launches at 5⁴ on single-GPU.

---

## Why this matters

The pmap path was the original (pre-vmap-only) implementation. It worked at small CPU virtual-device scale before recent changes. Recent changes prioritised the single-device path for the GH200 work. Whether the multi-device path still works is an empirical question that's cheap to answer with the n_dev=4 CPU smoke — and very expensive to discover the wrong way (multi-GPU instance billing while debugging).

Confirming green here unlocks **~2× wall reduction at 2× devices, ~6-7× at 8× devices** at canonical bundle scale. That's the difference between "spend $77 on 7⁴ canonical" and "spend $11" on a Lambda 8×. Worth the half-day audit.

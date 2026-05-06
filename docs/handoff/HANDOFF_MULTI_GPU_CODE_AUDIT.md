# Handoff: Multi-GPU Code Audit (Phase A — pre-runbook)

**Branch:** `jax-rewrite`
**Effort:** ~half a day (read-only audit + one CPU smoke at n_dev=4).
**Output:** a written report (markdown in `docs/scans/`) listing what works, what doesn't, and what needs patching before a multi-GPU run.

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

If the audit comes back green: spin up Lambda 2× or 8× instance, run `verify_smoke.py` and `verify_benchmark_bundle.py`, capture wall numbers, done. ~2× wall reduction at 2× device count is genuine compute we want to capture.

If the audit comes back with fixes needed: each fix is small (~1 file, ~10-50 LOC), but they MUST land before any multi-GPU run is meaningful. A multi-GPU run on broken code wastes GPU spend and produces misleading numbers.

---

## Audit checklist

### A. Chunking fix coverage (`cell_vmap_chunks` from `SolverConfig`)

**Concern:** the float agent's "chunk-outside-JIT" fix landed in commit history *after* the pmap path was last modified. The chunk-loop pattern must run in Python (not inside `@jit`) or XLA materialises the whole graph and defeats the memory bound.

**What to verify:**

1. Open [solver.py](../../lifecycle/solver.py) and find every call site that uses `cell_vmap_chunks`. Use:
   ```bash
   grep -n "cell_vmap_chunks\|_chunked_vmap_cells\|_chunk" lifecycle/solver.py
   ```
2. For each kernel builder (`_build_per_age_terminal_kernel_pmap`, `_build_per_age_retirement_kernel_pmap`, `_build_per_age_working_kernel_pmap`):
   - Confirm the chunk loop is in the **outer Python wrapper** (the one that's NOT decorated with `@jit`), not inside the `@jit`'d body.
   - Confirm each chunk dispatches its own `pmap`'d kernel call. The loop must look like:
     ```python
     for chunk_start in range(0, per_dev, chunk_size):
         chunk_result = jit_compiled_kernel(...)  # or pmap'd kernel
         results.append(chunk_result)
     return jnp.concatenate(results, axis=...)
     ```
   - **Anti-pattern to flag:** `for chunk_start in range(...)` *inside* a `@jit`'d function — that gets unrolled at trace time and defeats the whole point.
3. Cross-check: does the same pattern exist in the `_vmap_only` variants? If yes, both paths are clean. If only `_vmap_only` has the fix, **flag for follow-up patch.**

**Likely outcome:** float's fix landed in both, OR landed in `_vmap_only` only and pmap is silently broken. Either way the audit answers definitively.

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

## Validation

After running each audit step, write findings into `docs/scans/MULTI_GPU_AUDIT_2026-05-07.md` with one of three verdicts per check:

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

- [ ] (A) Audit chunking-fix coverage in all three pmap kernel builders.
- [ ] (B) Audit mixed-precision threading in all three pmap kernel builders.
- [ ] (C) Audit warm-start array sharding in retirement + working pmap builders.
- [ ] (D) Audit cell-axis padding strip in collapse logic for all three.
- [ ] (E) Run `XLA_FLAGS=--xla_force_host_platform_device_count=4 python verify_smoke.py`. Confirm `pmap+vmap (4 devices)` prints and alphas match `LIFECYCLE_DISABLE_VIRTUAL_CPUS=1` smoke bit-identically.
- [ ] (F) Light check on cache headroom and S3 sync.
- [ ] Write `docs/scans/MULTI_GPU_AUDIT_2026-05-07.md` with per-check verdicts.
- [ ] Commit:
  ```
  docs: multi-GPU code audit — Phase A pre-runbook scan
  
  Audits the pmap path of every kernel builder against recent changes
  (chunking, mixed-precision, warm-start, padding) to confirm the
  multi-GPU dispatch is shippable before paying for a 2x or 8x
  Lambda instance. Also runs a CPU-side n_dev=4 smoke as a regression
  check using XLA virtual devices.
  
  Verdict: <GREEN | YELLOW | RED with N items>. <one-line summary>.
  ```
- [ ] Push.

---

## Why this is a separate handoff (not part of any other)

- **Distinct from float's chunking fix:** float's job was to fix `_chunked_vmap_cells` correctness, not audit downstream multi-device implications. This handoff explicitly checks the *interaction* of float's fix with the pmap path.
- **Distinct from a Phase B "run multi-GPU benchmark" handoff:** Phase B is mechanical (spin instance, run, capture). It can't start until Phase A clears. Bundling them would conflate "audit work" with "compute work."
- **Distinct from sim-EE work:** sim-EE (already in flight) doesn't touch the solver kernel paths.

---

## Dependencies

- **Must wait for:** float agent's chunk-outside-JIT fix to land on `jax-rewrite`. Audit step (A) is meaningless against pre-fix code.
- **Does NOT block:** sim-EE diagnostic, arbitrage diagnostic, future bundle launches at 5⁴ on single-GPU.

---

## Why this matters

The pmap path was the original (pre-vmap-only) implementation. It worked at small CPU virtual-device scale before recent changes. Recent changes prioritised the single-device path for the GH200 work. Whether the multi-device path still works is an empirical question that's cheap to answer with the n_dev=4 CPU smoke — and very expensive to discover the wrong way (multi-GPU instance billing while debugging).

Confirming green here unlocks **~2× wall reduction at 2× devices, ~6-7× at 8× devices** at canonical bundle scale. That's the difference between "spend $77 on 7⁴ canonical" and "spend $11" on a Lambda 8×. Worth the half-day audit.

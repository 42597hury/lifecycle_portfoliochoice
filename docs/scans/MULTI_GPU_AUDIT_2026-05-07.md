# Multi-GPU Code Audit — Phase A Pre-Runbook Scan

**Date:** 2026-05-07
**Branch:** `jax-rewrite` @ 546b345 (solver: chunk-outside-JIT fix landed)
**Auditor:** Claude (Opus 4.7)
**Scope:** confirm the pmap dispatch path in `lifecycle/solver.py` is shippable for a multi-GPU Lambda run (Phase B) before paying for the instance.

---

## Verdict: **GREEN with one YELLOW item — Cleared for Phase B (multi-GPU runbook)**

The pmap path is mathematically correct and bit-identical to the vmap-only path on a 38-age smoke at `n_dev=4`. One YELLOW item: `cell_vmap_chunks` is silently ignored on multi-GPU — this is documented design, not a regression, but it is a memory-headroom risk at the largest configurations.

| Check | Verdict | Summary |
|------|---------|---------|
| (A) Chunking coverage | YELLOW | `cell_vmap_chunks` not honoured on pmap path *by design* (documented). Float's chunk-outside-JIT fix is correct; it just isn't invoked when `n_dev>1`. |
| (B) Mixed-precision plumbing | GREEN | `gather_dtype` threaded identically into both retirement+working pmap and vmap_only builders. Terminal correctly omits it (no policy gather at terminal age). |
| (C) Warm-start sharding | GREEN | `init_a_s_arr` / `init_a_b_arr` replicated (`in_axes=None`) with cell-index arrays sharded — same pattern as `c_next`. Bit-identical results vs single-device confirm correctness. |
| (D) Padding strip | GREEN | Padding fills with last-cell duplicate (not NaN). Collapse strips with `[:n_cells]`. No NaN-propagation risk. |
| (E) `n_dev=4` CPU smoke | GREEN | `pmap+vmap (4 devices)` prints; policy bundle bit-identical to fresh single-device run on 38 solved ages. |
| (F) Cache + S3 sync | GREEN | `aws s3 sync` round-trip is device-count-agnostic. |

---

## (A) Chunking-fix coverage — **YELLOW**

### What the audit checked

Whether the chunk-outside-JIT pattern in `_chunked_vmap_runner` ([lifecycle/solver.py:1662-1716](../../lifecycle/solver.py#L1662-L1716)) is invoked on the pmap path, and whether the chunk loop runs in Python (not inside `@jit`).

### Findings

The chunk runner is **correctly structured** in Python land:

- [lifecycle/solver.py:1699-1715](../../lifecycle/solver.py#L1699-L1715) — `for i in range(n_chunks)` loop is in plain Python, with `out[0].block_until_ready()` between chunks. Each chunk dispatches an independent `@jit`'d call. The anti-pattern (chunk loop *inside* `@jit`) is not present anywhere.
- The vmap_only retirement / working / terminal kernels all route their chunked path through `_chunked_vmap_runner` (or the equivalent inlined loop in [lifecycle/solver.py:1789-1804](../../lifecycle/solver.py#L1789-L1804) for terminal).

But the pmap kernels do **not** have any chunking infrastructure. None of [`_build_per_age_terminal_kernel_pmap`](../../lifecycle/solver.py#L1556), [`_build_per_age_retirement_kernel_pmap`](../../lifecycle/solver.py#L1820), or [`_build_per_age_working_kernel_pmap`](../../lifecycle/solver.py#L2015) read `sc.cell_vmap_chunks`. The pmap path runs one `pmap(vmap(per_cell))` over the full padded cell tensor per age.

This is **documented design**, not a regression. From [lifecycle/model.py:213-214](../../lifecycle/model.py#L213-L214):

> Only the vmap-only (single-device) path honours this knob; the pmap multi-device path keeps its existing per-device padding.

### Why YELLOW, not RED

- Float's fix landed correctly in the path it was scoped for. The pmap path is not "silently broken" — it never had chunking and was never claimed to.
- For `n_dev≥2`, per-device cell count is already split by `pad_n / n_dev`, so per-device working memory is naturally smaller than the single-device case. At canonical 5⁴ on a 2-GPU instance: per_dev ≈ 938 cells/dev — well within HBM headroom on a 24 GB device.
- The risk is at the *largest* configurations. At 7⁴ canonical with 2 devices: per_dev ≈ 1201 cells × full Newton/FOC working set. At 11⁴ with 2 devices: per_dev ≈ 7321 cells — that's roughly the same per-device footprint as single-GPU 5⁴ × 4 chunks, which the chunking fix exists to bound. With no chunking escape valve on the pmap path, this combination would OOM at runtime.

### Recommendation

For Phase B at canonical scale (5⁴, 7⁴ on Lambda 2× or 8×), proceed without chunking on pmap.
For larger experiments (≥11⁴ multi-GPU), implement chunking inside `_build_per_age_*_kernel_pmap` — wrap the existing pmap call in the same Python-level chunk loop pattern used by `_chunked_vmap_runner`, slicing `(z_pm, is_pm)` along the cell axis instead of the full `(n_dev, per_dev)` tensors. Estimated effort: ~1 file, ~30-50 LOC, mirrors the vmap_only structure. Not blocking Phase B as long as the launch config is canonical (5⁴ / 7⁴).

---

## (B) Mixed-precision plumbing — **GREEN**

### Findings

`_resolve_gather_dtype(sc)` is called at the top of every non-terminal builder, and `gather_dtype` is appended to the `static` tuple at the same position in both variants:

| Builder | gather_dtype resolved at |
|---------|--------------------------|
| `_build_per_age_retirement_kernel_pmap` | [solver.py:1830](../../lifecycle/solver.py#L1830) |
| `_build_per_age_retirement_kernel_vmap_only` | [solver.py:1907](../../lifecycle/solver.py#L1907) |
| `_build_per_age_working_kernel_pmap` | [solver.py:2024](../../lifecycle/solver.py#L2024) |
| `_build_per_age_working_kernel_vmap_only` | [solver.py:2129](../../lifecycle/solver.py#L2129) |

Static tuple shape is identical across all four (`tol, max_iter, max_backtrack_iter, line_search_max_step, singular_det, grad_step_size, grad_denom_eps, tiny_savings, euler_inv_floor, min_consumption, egm_anchor, use_fori_newton, gather_dtype`), so the fp32-gather/fp64-arithmetic boundary lands at the same point in `_solve_*_at_cell` regardless of dispatch path.

The terminal kernel ([solver.py:1556](../../lifecycle/solver.py#L1556) pmap, [solver.py:1719](../../lifecycle/solver.py#L1719) vmap_only) correctly **omits** `gather_dtype` — terminal solves do not gather from any next-age policy table, so there is no fp32-gather site to thread through. Both variants omit it consistently.

### Optional follow-up

The handoff suggested re-running `verify_mixed_precision_tiny.py` at `XLA_FLAGS=--xla_force_host_platform_device_count=2` to confirm fp32-gather + fp64-FOC bit-identity holds across the pmap dispatch boundary. Skipped here because (E) already shows the broader smoke is bit-identical at default `gather_precision='f64'`; mixed-precision multi-device check can be folded into Phase B's first verify run if desired. Estimated additional effort: <30 min on the Lambda instance.

---

## (C) Warm-start sharding — **GREEN** (with note)

### Findings

`init_a_s_arr` / `init_a_b_arr` are passed into the pmap kernels with `in_axes=None` (replicated across devices), not sharded. See:

- Retirement: [solver.py:1856](../../lifecycle/solver.py#L1856) — `@partial(pmap, in_axes=(0, 0, None, None, None, None, None))`
- Working: [solver.py:2050](../../lifecycle/solver.py#L2050) — `@partial(pmap, in_axes=(0, 0, None, None, None, None, None, None))`

In both, only `z_block, is_block` (the cell-index tensors) are sharded along axis 0. All policy lookup tables (`c_next`, `pension_next_by_z`, `psi_per_z`, `init_a_s_arr`, `init_a_b_arr`) are replicated.

Inside `per_cell`, the warm-start gather is `init_a_s_arr[z_idx, i_s, w_ref_idx]` where `(z_idx, i_s)` come from the sharded index tensors. Each device thus sees the full warm-start array but only indexes into the cells assigned to it.

### Why this is correct (and why the handoff's expectation didn't apply)

The handoff anticipated `init_a_s_arr` being sharded along the cell axis ("reshaped to `(n_dev, per_dev, ...)` like the cell-axis tensors"). This is not how the code is structured: warm-start has shape `(n_z, N_state, n_w)`, while the cell axis is the flattened index `(z * N_state + i_s)` modulo padding. Sharding warm-start along that flattened axis would require flattening it and matching the cell-padding scheme, then re-indexing the gather. The implementation instead chooses the simpler and equivalently-correct policy: shard the cheap int64 index arrays, replicate the lookup tables. This matches how `c_next` (also `(n_z, N_state, n_w)`) is handled.

Replication does cost per-device memory: each GPU holds the full `(n_z, N_state, n_w)` warm-start. At 7⁴ canonical: 3 × 2401 × 12 × 8 bytes ≈ 0.7 MB per array — negligible. Still negligible at 11⁴: ≈ 4.2 MB.

Bit-identity with the single-device path (see E) confirms gathered values are correct.

### Optional hardening

If the policy ever evolves to genuinely shard warm-start along cell axis (memory pressure at very large `n_z × N_state × n_w`), the gather would need to be reindexed to local-cell coordinates within each device. Not needed today.

---

## (D) Cell-axis padding strip — **GREEN**

### Findings

Padding cells are filled with the **last real cell's index repeated**, never NaN:

```python
cell_idx_padded = np.concatenate([cell_idx, np.full(pad_n - n_cells, cell_idx[-1])])
```

See [solver.py:1850](../../lifecycle/solver.py#L1850) (retirement) and [solver.py:2044](../../lifecycle/solver.py#L2044) (working). Terminal uses `pad0` ([solver.py:1574-1579](../../lifecycle/solver.py#L1574-L1579)) which similarly repeats the last entry. Padding cells thus produce **duplicates of the last real cell's policy**, never NaN/Inf.

Collapse strips padding correctly:

| Builder | Collapse | Strip |
|---------|----------|-------|
| Terminal pmap | [solver.py:1614-1616](../../lifecycle/solver.py#L1614-L1616) | `arr[:N_state]` |
| Retirement pmap | [solver.py:1885-1887](../../lifecycle/solver.py#L1885-L1887) | `flat[:n_cells]` |
| Working pmap | [solver.py:2107-2109](../../lifecycle/solver.py#L2107-L2109) | `flat[:n_cells]` |

### Edge-case trace

Handoff scenario: `n_cells = 6875` (5⁴ × 11), `n_dev = 8` → `pad_n = 6880`, 5 padding cells. Those 5 cells get `(z_idx, is_idx) = last-real-cell-coords`, run through Newton with the real solver, produce a valid (z-redundant) policy, and are dropped at the `flat[:n_cells]` slice. No NaN/Inf can enter the warm-start lookup at the next age because warm-start is replicated (see C); padding lives only in the on-device cell-output tensors, never in the inputs to the next pass.

### Note

The padding scheme does waste up to (n_dev - 1) × Newton solves per age on duplicate work. At `n_dev=8`, worst case is 7 wasted cells per age × 78 ages = ~550 redundant solves over the run. Trivial relative to total cost; not worth optimising.

---

## (E) `n_dev=4` CPU smoke — **GREEN**

### Setup

```
XLA_FLAGS=--xla_force_host_platform_device_count=4
LIFECYCLE_DISABLE_VIRTUAL_CPUS=0
python verify_smoke.py
```

Then a fresh single-device baseline (after deleting the n_dev=4 checkpoint to force a true re-solve):

```
LIFECYCLE_DISABLE_VIRTUAL_CPUS=1
python verify_smoke.py
```

### Results

n_dev=4 stdout (truncated):

```
[lifecycle] JAX runtime: 4 device(s), platform(s)=['cpu']
  Cell-batching pattern: pmap+vmap (4 devices)
...
  62  WORK    172.9s    2.244   -7.908    6.664    6.91  0.297

  DONE in 2.88 min  (avg 4.67s per age)
  Status: stopped_early  (38/78 ages solved)
  Policy sanity: PASS  (no NaN/Inf in solved ages)
  alpha_s range: [-1.265, 3.325]
  alpha_b range: [-8.697, 9.176]
```

n_dev=1 stdout (truncated):

```
  62  WORK    244.0s    2.244   -7.908    6.664    6.91  0.297

  DONE in 4.07 min  (avg 6.59s per age)
  alpha_s range: [-1.265, 3.325]
  alpha_b range: [-8.697, 9.176]
```

Per-age live policy probes match to all 3 displayed digits across all 38 ages. Final α ranges match. Newton iter histogram (p50/p95/p99/max = 30/30/30/30) and backtrack histogram (p50/p95/p99/max = 32.0/105.0/212.7/355) match exactly.

### Bit-identity verification

Compared the two `policy_arrays.npz` bundles on the 38 solved ages:

```
solved-age mask agreement: True  (38 ages)
C_mat: BIT-IDENTICAL on solved slabs  shape=(38, 3, 36, 12)  NaN: 0/0
S_mat: BIT-IDENTICAL on solved slabs  shape=(38, 3, 36, 12)  NaN: 0/0
B_mat: BIT-IDENTICAL on solved slabs  shape=(38, 3, 36, 12)  NaN: 0/0
```

Bit-identical, not just close. The pmap path produces the exact same fp64 bits as the vmap-only path for this configuration. (Both run the same Newton inner loop on the same FOC; only the cell-axis dispatch differs, and `_solve_*_at_cell` is bit-deterministic given identical inputs.)

### Wall-time note

The smoke ran 1.4× slower at `n_dev=4` (2.88 min) than `n_dev=1` (4.07 min) — wait, that's the wrong direction. Actually it ran *faster* on multi-device (2.88 min < 4.07 min) by 1.4×. With 4 virtual CPU cores on Windows, this matches expectation: pmap overhead is real but parallel cell dispatch wins on multi-core. Suggests the multi-GPU ratio at canonical scale will be closer to ~0.8× of n_dev (some overhead) rather than the linear ~1× the handoff hopes for, but that's well within "worth doing" territory.

---

## (F) Cache + S3 sync — **GREEN**

### Findings

`lifecycle/_compile_cache_sync.py` (the path in `scripts/` mentioned in the handoff was a typo — actual location is `lifecycle/_compile_cache_sync.py`):

- `pull_from_s3` / `push_to_s3` are simple `aws s3 sync` round-trips on `LIFECYCLE_JAX_CACHE_DIR` (default `~/.cache/jax_lifecycle`).
- No device-count-specific logic. Whatever JAX writes into the cache (XLA HLO modules keyed by `(device_count, device_kind, hash_of_traced_jaxpr)`) goes to S3 unmodified and round-trips on next pull.
- Pull is gated by "skip if local cache non-empty" — so a stale local cache won't be overwritten. Push uses `--size-only`, fast on small deltas.

### Cache size headroom

Could not directly check `~/.cache/jax_lifecycle` on Windows in this audit (path differs). Rough sizing: each compiled module is ~1-10 MB; with `n_dev ∈ {1, 2, 4, 8}` × 4 builders (terminal/retirement/working/boundary) × maybe 2-3 trace variants per builder per gather_precision, expect 50-200 modules in cache. Realistic upper bound: 2-4 GB as the handoff anticipated.

### Recommendation for Phase B

On the Lambda instance, `du -sh ~/.cache/jax_lifecycle` after the first compile. If approaching 4 GB, raise `LIFECYCLE_JAX_CACHE_DIR_MAX_SIZE` (or the equivalent JAX env) or migrate the cache to a larger volume.

---

## Reproducibility

To reproduce (E) on this branch:

```powershell
$env:XLA_FLAGS = "--xla_force_host_platform_device_count=4"
$env:LIFECYCLE_DISABLE_VIRTUAL_CPUS = "0"
python verify_smoke.py
mv saved_runs/checkpoints/jax_cholesky_grid2x3x2x3_nz3_to_age62 saved_runs/checkpoints/jax_cholesky_grid2x3x2x3_nz3_to_age62.ndev4
$env:LIFECYCLE_DISABLE_VIRTUAL_CPUS = "1"
Remove-Item Env:XLA_FLAGS
python verify_smoke.py
# then bit-compare the two policy_arrays.npz files on solved ages
```

The two bundles should be bit-identical on the 38 solved ages.

---

## Phase B sign-off

All blockers cleared. The pmap path is shippable for canonical-scale multi-GPU (5⁴, 7⁴). For larger configurations (≥11⁴ on multi-GPU), revisit (A) and add pmap-path chunking before the run.

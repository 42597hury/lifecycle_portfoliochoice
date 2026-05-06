# Performance Scan — 2026-05-06

**Scanner:** Claude (Opus 4.7)
**Branch HEAD:** `389ce0860c0050c4b05489f6812e47330bbac1f4` (`jax-rewrite`)
**Scope:** [lifecycle/solver.py](lifecycle/solver.py) + [lifecycle/__init__.py](lifecycle/__init__.py) + [lifecycle/precompute.py](lifecycle/precompute.py) (build_precompute body)
**Mode:** static read; no benchmarks run; no code edits.

## Summary table

| #  | Item                                       | Verdict | Severity     | Recommended action |
|----|--------------------------------------------|---------|--------------|--------------------|
| 1  | Per-age H→D uploads                        | YES     | MEDIUM       | Hoist `survival`, `pension_table`, `working_income_next_full`, `pension_dummy_z` to single `jnp.asarray` outside the loop; slice on-device. |
| 2  | D→H syncs in age loop                      | YES     | MEDIUM       | Replace 3× `np.asarray(...)` in the verbose probe with one `jax.device_get((...))` over a tuple. |
| 3  | Repeated work in kernel builders           | YES     | MEDIUM       | Hoist `_precompute_per_is_tensors(pcj)` to one call; pass result to all three non-terminal builders. |
| 4  | JIT cache key contamination                | NO      | —            | Clean. No `static_argnums`/`static_argnames`; SolverConfig fields are closure captures. |
| 5  | vmap-able `lax.scan` patterns              | NO      | —            | Clean. Zero `lax.scan` in `solver.py` (one mention in a comment). |
| 6  | vmap axes flat-indexed                     | NO      | —            | Clean. `z_idx_arr`/`is_idx_arr` built at builder time, closed over, output reshape on JAX side. |
| 7  | Broadcast tensor materialisation under vmap | YES    | HIGH         | Already-known risk on working-age path (`c_corners = c_next[:, j_corners_i, :]` under vmap). Profile peak HBM on first GPU run before scaling state grid. |
| 8  | pmap `in_axes=None` broadcast patterns     | INFO    | INFORMATIONAL| Documented below. The vmap-only path used at single-device avoids these entirely. |
| 9  | Persistent cache config sanity             | NO      | LOW (note)   | Defaults look reasonable. Optional: lower `min_compile_time_secs` from 1.0 to 0.5 if first-run compile time matters. |
| 10 | Avoidable `jnp.broadcast_to` materialisation | YES   | LOW          | Stale comment + small one-time materialisation hazard at first retirement age on the new vmap-only path. |
| 11 | `working_income_next_full` upload pattern  | YES     | LOW          | Same fix as Item 1; 110 KB total table, free win + clarity. |
| 12 | Compile-time configuration print           | NO      | —            | Clean. Banner runs once per solve. |

**Tally:** 1 HIGH (already known), 3 MEDIUM, 2 LOW, 1 INFORMATIONAL, 5 NO.

---

## Detailed findings

### Item 1 — Per-age host→device uploads

**Location:** [lifecycle/solver.py:2087](lifecycle/solver.py#L2087), [:2102](lifecycle/solver.py#L2102), [:2110](lifecycle/solver.py#L2110), [:2117-2118](lifecycle/solver.py#L2117-L2118); set up at [:2064-2067](lifecycle/solver.py#L2064-L2067).

**Verdict:** YES.
**Severity:** MEDIUM.

**Description.** The age loop performs four `jnp.asarray(...)` calls per age, all from NumPy arrays held in host memory:

- L2087: `psi_t = jnp.asarray(survival[t, :])` — `(n_z,)` float, ~88 B.
- L2102 / L2110: `pension_next = jnp.asarray(pension_table[t + 1, :])` — `(n_z,)`, ~88 B.
- L2117: `pension_next = jnp.asarray(pension_dummy_z)` — `(n_z,)` zeros, ~88 B (fresh transfer of a constant on every working age).
- L2118: `income_table = jnp.asarray(working_income_next_full[t + 1])` — `(n_z, n_eta, n_eps)`, ~110 KB.

`survival`, `pension_table`, `working_income_next_full`, and `pension_dummy_z` are all immutable for the duration of the solve. None of these slices are used by host code in the loop body — they go straight into the kernel call.

**Why it matters at this scale.** Each `jnp.asarray` from a NumPy array dispatches a host→device transfer; under JAX async dispatch they pipeline, but each adds dispatch overhead and forces a (small) PCIe transaction. Over ~80 ages × 3-4 calls = ~250 dispatch points per solve, plus the same per checkpoint resume. On GPU the total bytes are negligible (<10 MB) but the dispatch latency (~tens of µs each) sums to a measurable but small amount per run; the bigger win is removing GPU pipeline stalls when the kernel is short.

**Recommended fix.** Hoist all four to single device-resident arrays before the loop (around [lifecycle/solver.py:2064-2076](lifecycle/solver.py#L2064-L2076)):

```python
survival_jnp           = jnp.asarray(pc.survival_probs_2d)            # (n_age, n_z)
pension_table_jnp      = jnp.asarray(pc.pension_after_tax)            # (n_age, n_z)
working_income_next_jnp = jnp.asarray(pc.working_income_next)         # (n_age, n_z, n_eta, n_eps)
pension_dummy_z_jnp    = jnp.zeros(n_z, dtype=jnp.float64)
```

Then in the loop:

```python
psi_t          = survival_jnp[t, :]
pension_next   = pension_table_jnp[t + 1, :]                          # retire / boundary
pension_next   = pension_dummy_z_jnp                                  # working
income_table   = working_income_next_jnp[t + 1]                       # working
```

These are on-device slices — no transfer.

**Confidence.** HIGH. Pure hoist; identical numerics; the slices are jnp arrays that the kernels already accept.

---

### Item 2 — D→H syncs in the age loop

**Location:** [lifecycle/solver.py:2137-2139](lifecycle/solver.py#L2137-L2139) (per-age verbose probe). One-time sync at [:2048](lifecycle/solver.py#L2048).

**Verdict:** YES.
**Severity:** MEDIUM (when `verbose >= 1`, the default).

**Description.** Inside the per-age verbose probe block:

```python
s_slice = np.asarray(s_t[i_z_med, i_s_med, :])
b_slice = np.asarray(b_t[i_z_med, i_s_med, :])
c_slice = np.asarray(c_t[i_z_med, i_s_med, :])
```

Each `np.asarray(<jax_array>)` is a separate D→H sync; the host blocks while the dispatch queue completes. With `verbose=1` (the default smoke / canonical setting) this happens every age — typically 80 ages × 3 syncs.

L2048 has `float(c_T.min())` / `float(c_T.max())` after terminal solve; one-time, low priority.

**Why it matters at this scale.** On GPU each sync round-trip is on the order of ~50–500 µs depending on driver / NVLink path; three sequential syncs serialize into ~150 µs–1.5 ms per age × 80 ages ≈ 0.01–0.12 s/run. Small in absolute terms, but the bigger cost is that each sync drains the dispatch pipeline, blocking the GPU from overlapping the next-age kernel launch. Three syncs back-to-back means three pipeline drains.

**Recommended fix.** Merge to a single sync over a tuple:

```python
s_slice, b_slice, c_slice = jax.device_get((
    s_t[i_z_med, i_s_med, :],
    b_t[i_z_med, i_s_med, :],
    c_t[i_z_med, i_s_med, :],
))
```

Or, equivalently, stack on-device first:

```python
probe = jnp.stack([s_t[i_z_med, i_s_med, :],
                   b_t[i_z_med, i_s_med, :],
                   c_t[i_z_med, i_s_med, :]])
s_slice, b_slice, c_slice = np.asarray(probe)
```

Either yields one D→H sync per age.

**Confidence.** HIGH. No semantic change.

---

### Item 3 — Repeated work in kernel builders

**Location:** [lifecycle/solver.py:1573-1575](lifecycle/solver.py#L1573-L1575), [:1638-1640](lifecycle/solver.py#L1638-L1640), [:1704-1706](lifecycle/solver.py#L1704-L1706), [:1797-1799](lifecycle/solver.py#L1797-L1799). Function defined at [:1304-1336](lifecycle/solver.py#L1304-L1336).

**Verdict:** YES.
**Severity:** MEDIUM (one-time builder cost, not per-age).

**Description.** `_precompute_per_is_tensors(pcj)` is called inside each of the four retirement / working builders. At a single solver invocation the orchestrator at [:1944-1947](lifecycle/solver.py#L1944-L1947) builds three kernels using these builders:

- `retirement_kernel` → 1 call to `_precompute_per_is_tensors`
- `working_kernel` → 1 call
- `boundary_kernel` → 1 call

Total: **3 redundant calls per solver run**. The result depends only on `pcj` (which is immutable per solve), so all three returned tuples are identical. The function itself is a `vmap` over `N_state` of `_build_step_state_brackets` + `_build_step_log_returns` — not free; at canonical sizes it traces and runs a non-trivial JAX computation.

The duplicated `static = (...)` tuple construction across the six builders is trivial; mention only.

The terminal builder uses a different path (`_all_is_log_returns_numpy`), so it doesn't share this work — that's fine.

**Why it matters at this scale.** First-time JAX trace + execute of the vmap'd function likely takes a few hundred ms each (depends on `N_state` and `n_state_quad`); doing it 3× rather than 1× adds up to a one-time builder overhead of ~0.5–1 s. Negligible during long solves but visible on smoke / debug runs and obvious to fix.

**Recommended fix.** Hoist out of the builders. In `run_lifecycle_solver`, around L1942, do:

```python
per_is_tensors = _precompute_per_is_tensors(pcj)
```

Change builder signatures so each non-terminal builder accepts `per_is_tensors` and unpacks it instead of recomputing. Trivial threading; no semantic change.

**Confidence.** HIGH. The result is a deterministic function of `pcj`; verified by reading the function body.

---

### Item 4 — JIT cache key contamination

**Verdict:** NO.

Searched for `static_argnums` / `static_argnames` — zero hits. The `static = (...)` tuple in each builder bundles SolverConfig fields (Python floats / ints / a single bool) and is unpacked positionally via `*static` into the inner solver call. Because `sc` is captured at builder time (Python-level closure), all entries are baked into the trace as Python-time constants — JAX/XLA sees them as compile-time literals, not runtime tensor inputs.

`bool(sc.use_fori_newton)` ([:1454](lifecycle/solver.py#L1454), etc.) flows down to `newton_2d_with_line_search(use_fori=...)` at [:1079](lifecycle/solver.py#L1079) where `if use_fori:` selects between `_newton_fori` and `_newton_while` at Python time; closure capture means only one branch is ever traced per builder — exactly as intended.

`@partial(pmap, in_axes=...)` at [:1591](lifecycle/solver.py#L1591) and [:1722](lifecycle/solver.py#L1722) uses fixed integer / `None` axis specs — no per-age contamination. Clean.

---

### Item 5 — `vmap`-able `lax.scan` patterns

**Verdict:** NO.

`grep "lax\.scan" lifecycle/solver.py` returns one hit (a comment at [:1057](lifecycle/solver.py#L1057) describing the warm-start kill). No live `lax.scan` in the hot path. Per-savings-point loop is `vmap`'d at [:1091](lifecycle/solver.py#L1091); per-cell loop is `vmap`'d at [:1666](lifecycle/solver.py#L1666) (retirement) and [:1849](lifecycle/solver.py#L1849) (working). Clean.

---

### Item 6 — `vmap` axes that should be flat-indexed

**Verdict:** NO.

In each `*_vmap_only` builder ([:1646-1648](lifecycle/solver.py#L1646-L1648), [:1805-1807](lifecycle/solver.py#L1805-L1807)):

- `cell_idx = np.arange(n_cells, dtype=np.int64)` — host.
- `z_idx_arr = jnp.asarray((cell_idx // N_state).astype(np.int64))` — uploaded once at builder time.
- `is_idx_arr = jnp.asarray((cell_idx % N_state).astype(np.int64))` — same.

Both arrays are closed over by the inner `@jit all_cells(...)`; they are NOT call arguments, so they can't be re-uploaded per call. The output reshape ([:1673-1675](lifecycle/solver.py#L1673-L1675), [:1860-1862](lifecycle/solver.py#L1860-L1862)) uses `jnp.reshape` — JAX-side. Clean.

---

### Item 7 — Broadcast tensor materialisation under `vmap` (working-age)

**Location:** [lifecycle/solver.py:1230-1231](lifecycle/solver.py#L1230-L1231) inside `_solve_working_at_cell`.

**Verdict:** YES (already-known risk, flagged in pmap→vmap handoff §6.6).
**Severity:** HIGH (per checklist criterion: advanced gather under vmap with no streaming guarantee).

**Description.** The working-age per-cell solve does:

```python
c_corners = c_next[:, j_corners_i, :]                 # (n_z, n_state_quad, 8, n_w)
c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))  # (n_state_quad, n_z, 8, n_w)
```

This is a 4-axis advanced gather (`j_corners_i` is `(n_state_quad, 8)` int → result has those two axes). Under `vmap` over `n_cells`, XLA may choose to materialise the full batched tensor of shape `(n_cells, n_state_quad, n_z, 8, n_w)`. The comment at [:1227-1229](lifecycle/solver.py#L1227-L1229) acknowledges per-cell size (~3 MB) but does not assert that the vmap-batched form streams — only that the inner Newton reads via `dynamic_slice`, which is a different concern.

The retirement path [:1175](lifecycle/solver.py#L1175) is similar but smaller: `c_corners_at_z = c_next[z_idx, j_corners_i, :]` of shape `(n_state_quad, 8, n_w)` per cell — drops the `n_z` axis because retirement policy is z-frozen.

**Why it matters at this scale.** At canonical 9×9×9 state grid (`N_state = 729`), `n_z = 11`, `n_state_quad = 9`, `n_w = 80`: per-cell `(9, 11, 8, 80)` floats = ~507 KB; with `n_cells = n_z × N_state = 8019`, the batched form is ~4.0 GB — within an A100's HBM but pushing into "uncomfortable" territory if the resident `c_next` (~5 MB) ever needs a duplicate copy plus working state. The handoff itself estimates ~24 GB at certain configurations; that's the regime that risks an OOM.

**Recommended fix.** This is out-of-scope for fixing in this scan but worth two profiling actions before scaling:

1. On first GPU run at canonical grid, capture XLA HBM peak via `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `nvidia-smi` polling to confirm the gather is fused / streamed. If peak HBM > 1.5× the resident `c_next` × `n_state_quad`, materialisation is happening.
2. If materialised: defer the transpose ([:1231](lifecycle/solver.py#L1231)) to the consumer (have `working_foc_jac_ccv` accept `c_corners` with `n_z` leading); a single transpose of a 4D tensor under vmap is the kind of op that XLA reorders or eliminates when the consumer's access pattern matches the natural layout. MEDIUM impact at most.
3. Longer-term: replace advanced gather with `lax.dynamic_slice` over the `n_z` axis, paired with the existing `j_corners_i` indices — out-of-scope here.

**Confidence.** MEDIUM that materialisation occurs (XLA's behaviour on advanced gathers under vmap is implementation-dependent). HIGH that profiling will resolve it.

---

### Item 8 — `pmap` `in_axes=None` broadcast patterns

**Verdict:** INFORMATIONAL only (per checklist). On single-device hosts the vmap-only path bypasses these; only relevant on multi-GPU.

**Retirement** ([:1591](lifecycle/solver.py#L1591)): `@partial(pmap, in_axes=(0, 0, None, None, None, None, None))`
- broadcast: `c_next` `(n_z, N_state, n_w)`, `pension_next_by_z` `(n_z,)`, `psi_per_z` `(n_z,)`, `init_a_s_arr` `(n_z, N_state, n_w)`, `init_a_b_arr` `(n_z, N_state, n_w)`.
- At canonical 11 × 729 × 80 × 8 B: each of the 3 large `(n_z, N_state, n_w)` tensors is ~5.1 MB. Per device.

**Working** ([:1722](lifecycle/solver.py#L1722)): `@partial(pmap, in_axes=(0, 0, None, None, None, None, None, None))`
- same plus `income_next_table` `(n_z, n_eta, n_eps)` ~1.4 KB.

**Per-device cost.** ~15–16 MB of broadcast tensors per GPU. On 8× A100 (the worst documented case): 8 × 16 MB = ~125 MB total replicated across the device grid. Tolerable. The fix is `jax.sharding.NamedSharding` (out of scope per the handoff).

---

### Item 9 — Persistent cache config sanity

**Location:** [lifecycle/__init__.py:74-138](lifecycle/__init__.py#L74-L138).

**Verdict:** NO live issue. One sub-LOW note.

- `jax_compilation_cache_dir` set unconditionally when the env var is non-empty. ✓
- `jax_persistent_cache_min_entry_size_bytes = -1` (no size floor). ✓
- `jax_persistent_cache_min_compile_time_secs` defaults to `1.0`. The kernels we care about (terminal / retirement / working `@jit`'d trees) compile in well over 1 second on first run, so they will be persisted. Sub-second helper traces (small jnp.asarray-driven scatters etc.) won't be — fine, they re-trace cheaply on subsequent runs.
- `os.makedirs(cache_dir, exist_ok=True)` is not racy in practice; multiple processes hitting the same directory is the documented use case for JAX's compilation cache, which uses file-level atomicity. ✓

**Optional tuning (sub-LOW):** if the user runs many short solves back-to-back (sweep with small grids that compile in <1 s), lowering `LIFECYCLE_JAX_CACHE_MIN_COMPILE_SECS=0.5` would persist more sweep variants. Not material at canonical sizes.

---

### Item 10 — Avoidable `jnp.broadcast_to` materialisation

**Location:** [lifecycle/solver.py:2042-2044](lifecycle/solver.py#L2042-L2044), comment at [:2039-2041](lifecycle/solver.py#L2039-L2041).

**Verdict:** YES.
**Severity:** LOW.

**Description.**

```python
C_list[-1] = jnp.broadcast_to(c_T[None, :, :], (n_z, N_state, n_w))
S_list[-1] = jnp.broadcast_to(s_T[None, :, :], (n_z, N_state, n_w))
B_list[-1] = jnp.broadcast_to(b_T[None, :, :], (n_z, N_state, n_w))
```

Two issues:

1. **Stale comment.** L2040-2041 says `pmap in_axes=None which materialises the broadcast lazily`. On the new vmap-only path (single-device GPU) there is no `pmap`; the broadcast is consumed by `c_next[z_idx, j_corners_i, :]` inside the retirement kernel, which is an advanced gather over the `n_z` axis. Whether XLA propagates the broadcast through the gather or materialises a `(n_z, N_state, n_w)` tensor first is implementation-dependent. The comment promises a guarantee that no longer applies.

2. **One-time materialisation hazard.** Only the first retirement age reads the broadcasted `c_next` (subsequent ages write a real `(n_z, N_state, n_w)` policy). At canonical sizes `(11, 729, 80)` × 8 B = ~5 MB, so even pessimistic full-materialisation costs ~5 MB device memory once. Not catastrophic.

**Why it matters at this scale.** Genuinely small. Mostly a correctness-of-comments / documentation issue.

**Recommended fix.** Either:

- (Cheap) update the comment at L2040-2041 to acknowledge the vmap-only path and stop claiming a lazy guarantee.
- (Slightly less cheap) special-case the first retirement age: pass the un-broadcast `c_T` `(N_state, n_w)` directly and have `_solve_retirement_at_cell` ignore the `z_idx` dimension when consuming a 2D `c_next`. Adds a code path. Not worth doing for ~5 MB.

**Confidence.** HIGH that the comment is stale; MEDIUM whether materialisation is observable.

---

### Item 11 — `working_income_next_full` size and upload pattern

**Location:** [lifecycle/solver.py:2067](lifecycle/solver.py#L2067), [:2118](lifecycle/solver.py#L2118).

**Verdict:** YES.
**Severity:** LOW.

**Description.** `working_income_next_full = np.asarray(pc.working_income_next)` at L2067 holds a `(n_age, n_z, n_eta, n_eps)` table on host. At canonical: 80 × 11 × 4 × 4 × 8 B ≈ 110 KB. The per-age slice `working_income_next_full[t + 1]` is converted to `jnp.asarray` inside the loop on every working-age iteration (L2118).

**Why it matters at this scale.** Negligible bytes (~46 working ages × ~2.7 KB each = ~125 KB total transferred). Effectively free; the win is clarity and dispatch-pipeline cleanliness (one less host→device call per working age).

**Recommended fix.** Folded into Item 1's hoist:

```python
working_income_next_jnp = jnp.asarray(pc.working_income_next)   # at L2067
income_table = working_income_next_jnp[t + 1]                   # at L2118
```

**Confidence.** HIGH. One-line change, no behaviour change.

---

### Item 12 — Compile-time configuration print

**Location:** [lifecycle/solver.py:1906-1917](lifecycle/solver.py#L1906-L1917).

**Verdict:** NO.

Banner runs once per `run_lifecycle_solver` invocation. `len(jax.devices())` is cheap. `print(f"  Solver: {solver_config}")` produces a long line as `SolverConfig` grows fields, which is a cosmetic concern only — no per-iteration cost. Clean.

---

## Items reported as NO or SKIPPED

- **Item 4** (JIT cache key contamination): no `static_argnums`/`static_argnames` anywhere; SolverConfig is closure-captured at builder time. Clean.
- **Item 5** (vmap-able `lax.scan`): zero `lax.scan` in `solver.py`. Clean.
- **Item 6** (flat-indexed vmap axes): cell-index arrays built once at builder time and closed over; reshapes are JAX-side. Clean.
- **Item 9** (cache config): defaults are reasonable; no demonstrable cache-skip for our kernels.
- **Item 12** (config print): one-time per-solve cost only. Clean.

## Out of scope (noticed but not flagged)

- **`_pc_to_jnp` `(jnp.asarray(np.asarray(...))` double conversion** at [:1427-1428](lifecycle/solver.py#L1427-L1428) for `Phi_0_state` / `Phi_11`. The `np.asarray(..., dtype=np.float64)` is a host-side dtype coercion to guarantee float64; the `jnp.asarray` is the device upload. Not a real perf bug — the inner `np.asarray` on an already-float64 NumPy array is a near-free no-op — but slightly noisy code. Skipped (cosmetic).
- **Verbose probe builds a 3-element NumPy probe and runs `np.interp` per age**. This is host work (no device work), <1 ms per age. Out of scope.
- **Checkpoint resume uploads each solved slab via three separate `jnp.asarray` calls** at [:2003-2005](lifecycle/solver.py#L2003-L2005). One-time at resume; not in scope of "hot path."
- **Newton math, FOC math, EGM scheme, mixed precision**: explicitly out of scope per handoff §2.
- **`_*_pmap` paths**: only used on multi-device hosts; left intact per handoff §5.

---

## End

Total findings: 1 HIGH (already documented elsewhere), 3 MEDIUM, 2 LOW, 1 INFORMATIONAL, 5 NO. Report not committed; staged for user review.

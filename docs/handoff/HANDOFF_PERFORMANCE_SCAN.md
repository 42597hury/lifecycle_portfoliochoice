# Handoff: Performance Scan (Report-Only)

**Branch:** `jax-rewrite`
**Mode:** **REPORT ONLY.** Do not edit any code. Produce one markdown report. Multiple other agents are editing the codebase simultaneously; introducing parallel edits will cause merge conflicts.

**Output target:** Create `docs/scans/PERFORMANCE_SCAN_2026-05-06.md`. The user reviews findings and dispatches fixes as separate work.

**Scope:** Performance / efficiency only. Not correctness. Not code style. Not architecture. There is a separate bug-scan handoff covering correctness.

**Time budget:** 2-3 hours focused. If a check takes >30 min to investigate, mark it "skipped — needs profiler" and move on.

---

## 1. Goal

The lifecycle solver is about to run on a paid GPU instance ($1.99-3.29/hr). Find specific, low-risk efficiency wins **on the hot path** (`run_lifecycle_solver` and the per-cell solve functions in `lifecycle/solver.py`) that the user can act on before launching, so first GPU runs are as cheap as possible.

This scan is **constrained**: a fixed checklist (§3) of specific suspect patterns at specific locations. Not "find inefficiencies" open-ended. Each item has a yes-or-no answer.

---

## 2. Scope

### In scope

- `lifecycle/solver.py` — primary focus.
- `lifecycle/precompute.py` — only the `build_precompute` body (one-time cost, but worth flagging if egregious).
- `lifecycle/__init__.py` — only the persistent cache config block.
- The age-loop orchestrator inside `run_lifecycle_solver` (lines ~1900-2080).

### Out of scope (do not touch / do not analyse)

- `lifecycle/diagnostics.py`, `lifecycle/simulation.py`, `lifecycle/inf_horizon_solver.py`.
- The Newton math itself (`_newton_while`, `_newton_fori`, `_backtracking_fori`). Already deeply optimised.
- The FOC math (`terminal_foc_jac_ccv`, `retirement_foc_jac_ccv`, `working_foc_jac_ccv`, `_ccv_log_return_and_grad`).
- Mixed precision proposals (separate work).
- Algorithmic redesign — EGM scheme, quadrature rules, line-search rule.
- Anything cosmetic (variable names, comments, formatting).
- `verify_*.py` runner scripts (those are configs, not the hot path).

### Hard constraints

- **No code edits.** Findings go in the report only.
- Do not run benchmarks. Static read of code only.
- Do not run smoke or any solver. Only `git grep`, file reads, and analysis.

---

## 3. Checklist (enumerated — answer each item)

For each item: report **YES** (finding present, fix recommended), **NO** (checked, not present), or **SKIPPED** (couldn't determine without profiling). Cite specific file:line locations for YES findings.

### Item 1 — Per-age host→device uploads in the age loop

**Where:** `run_lifecycle_solver`, lines ~1991-2080.

**What to look for:** `jnp.asarray(...)` calls inside the `for t in reversed(range(n_age - 1)):` loop. Each one is a per-age host-to-device transfer. Some are unavoidable (the per-age survival/pension scalars), but check whether any of these tensors could be uploaded ONCE before the loop starts:

- `psi_t = jnp.asarray(survival[t, :])` — small scalar slice, fine, but verify
- `pension_next = jnp.asarray(pension_table[t + 1, :])` — same shape every age, could pre-upload as `pension_table_jnp` and slice on-device
- `income_table = jnp.asarray(working_income_next_full[t + 1])` — could pre-upload the full `(n_age, n_z, n_eta, n_eps)` table once

**Severity criteria:**
- HIGH: any per-age upload >1 MB (none expected at canonical sizes, flag if found)
- MEDIUM: per-age upload that could be hoisted with no logic change
- LOW: <1 KB per age, mention as nicety

### Item 2 — D→H syncs in the age loop

**Where:** Same range. Look for `np.asarray(<jax_array>)` or `float(<jax_scalar>)` calls inside the for-loop body.

**What to look for:** Each such call forces a device→host sync, blocking the GPU. Some are unavoidable (the live progress probe at lines 2061-2069). Check:

- The progress probe block: 3 `np.asarray(...)` calls per age (`s_slice`, `b_slice`, `c_slice`) — could be merged into a single sync via `jax.tree_util.tree_map` plus one `np.asarray` over a tuple
- Any `float(...)` calls in non-probe code paths
- The `print(f"... {float(c_T.min()):.3f}-{float(c_T.max()):.3f}")` after terminal solve (one-time, low priority)

**Severity criteria:**
- MEDIUM: D→H sync inside the per-age loop with a feasible workaround
- LOW: D→H sync at age-loop boundary (one-time per solve)

### Item 3 — Repeated work that could be hoisted out of the kernel builders

**Where:** `_build_per_age_terminal_kernel_pmap` / `_vmap_only`, `_build_per_age_retirement_kernel_*`, `_build_per_age_working_kernel_*`. Lines 1341-1795.

**What to look for:** Operations done at builder time that depend only on the SolverConfig, model, or precompute (not on per-age data). Specifically:

- The `static = (...)` tuple construction — duplicated across 6 builders. Trivial duplication, mention if egregious.
- `_precompute_per_is_tensors(pcj)` is called in EACH retirement/working builder (4 builders × the same call). It returns the same tensors. Could be hoisted into a single call cached in a closure or module-level. **Verify** if this is currently called 4× and the result identical each time.
- NumPy → jnp asarray conversions in the terminal builder (lines 1355-1457) — check whether anything is converted twice.

**Severity criteria:**
- MEDIUM: a non-trivial precompute called 2+ times redundantly
- LOW: trivial duplicated tuple construction

### Item 4 — JIT cache key contamination

**Where:** Each kernel builder produces a `call(...)` closure that gets called per-age. Examine whether any per-age value enters the closure as a STATIC argument (would force re-trace per age).

**What to look for:** `static_argnums` or `static_argnames` in any `@jit` / `@partial(jit, ...)` decorators inside the kernels. If a per-age scalar is in `static_argnums`, JIT recompiles per age — catastrophic.

Specifically:
- `@jit` decorator on `all_cells` in the vmap-only builders (lines 1580, 1739) — verify no static args
- `@partial(pmap, in_axes=...)` decorators — verify no static dimensioning that varies per call
- Whether the `static` tuple's contents (boolean `use_fori_newton` flag) is passed as a runtime arg or a Python-time closure capture. **It must be a closure capture** so JIT only sees one trace.

**Severity criteria:**
- HIGH: any `static_argnums` containing a per-age value
- HIGH: a closure that re-traces because of changed Python-level state
- LOW: static configuration passed via positional args (intended)

### Item 5 — `vmap`-able `lax.scan` patterns

**Where:** Search `lifecycle/solver.py` for `lax.scan`. Currently expected: zero (the warm-start kill removed the only one). If any new ones have appeared, flag them.

**What to look for:**
- Any `lax.scan` in the per-cell solve path. The warm-start kill replaced one with `vmap`. If a new one is back, that's a regression.
- `for ... in <jax-array>:` patterns that should be `vmap`.

**Severity criteria:**
- HIGH: any `lax.scan` over a vmap-able dimension in the hot path
- LOW: `lax.scan` over an inherently sequential dimension (age, Newton iter)

### Item 6 — `vmap` axes that should be flat-indexed

**Where:** The three `*_vmap_only` builders (lines 1424, 1557, 1716).

**What to look for:** Currently the vmap-only path constructs flat 1D `z_idx_arr`, `is_idx_arr` and vmaps `per_cell` over them. This is correct. Verify:

- Both arrays are constructed from `np.arange(n_cells)` and `jnp.asarray`'d — they're at this point JAX device arrays.
- They are NOT re-uploaded per call (should be closed over the `call(...)` closure, not passed in).
- The output reshape from `(n_cells, n_w)` to `(n_z, N_state, n_w)` is on the JAX side, not via NumPy.

**Severity criteria:**
- HIGH: the index arrays are re-uploaded per age call (re-trace risk)
- LOW: cosmetic concerns

### Item 7 — Broadcast tensor materialisation under `vmap`

**Where:** `_solve_retirement_at_cell` (line ~970), `_solve_working_at_cell` (line ~1170). Specifically the `c_corners` and `c_corners_T` gather lines.

**What to look for:** Comments and code at:

```python
c_corners = c_next[:, j_corners_i, :]      # (n_z, n_state_quad, 8, n_w)
c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))   # (n_state_quad, n_z, 8, n_w)
```

This is a 4-axis advanced gather under `vmap` — if XLA materialises the full vmap'd batch (`n_cells × per-cell-tensor`), peak HBM at canonical 9×9×9 is ~24 GB. The pmap→vmap handoff §6.6 flags this as a real risk.

**What to flag:**
- Whether the gather pattern is `c_next[:, j_corners_i, :]` (advanced indexing — risk of materialisation) or `lax.dynamic_slice` (streaming-friendly)
- Whether there's a comment indicating this was profiled and confirmed-not-materialising
- Whether per-cell memory could be reduced by deferring the transpose (have the FOC consume the un-transposed version)

**Severity criteria:**
- HIGH: any `c_next[:, j_corners_i, :]` style advanced gather materialised inside vmap with no streaming guarantee
- MEDIUM: a `transpose` that could be avoided by consumer-side restructuring
- LOW: cosmetic

### Item 8 — `pmap` `in_axes=None` broadcast patterns

**Where:** All three `*_pmap` kernel builders (lines 1341, 1486, 1618).

**What to look for:** `@partial(pmap, in_axes=(0, 0, None, None, None, ...))` decorators. The `None` axes are broadcast across devices — each GPU holds a full copy. Document for each:

- What's broadcast (e.g. `c_next` shape `(n_z, N_state, n_w)`)
- Per-device cost: at canonical 9×9×9, `c_next` is ~92 MB, one copy per GPU
- On 8× A100 with `pmap`: 8 × 92 MB = 736 MB just for c_next broadcasts. Tolerable, but worth knowing.

**Severity criteria:**
- INFORMATIONAL only: this is the documented multi-GPU sub-optimality; the fix is `jax.sharding.NamedSharding` (out-of-scope follow-up). Just enumerate what's broadcast and at what size.

### Item 9 — Persistent cache config sanity

**Where:** `lifecycle/__init__.py:74-138`.

**What to look for:**
- Is the cache write-once or are there cases where it's not consulted? (Should always be consulted on JIT'd functions.)
- Is `jax_persistent_cache_min_compile_time_secs` 1.0s — does that filter out useful smaller-trace caches?
- Is the cache directory creation racy with multiple solver invocations on the same machine?

**Severity criteria:**
- MEDIUM: a setting that demonstrably skips caching for our kernels
- LOW: cosmetic

### Item 10 — Avoidable `jnp.broadcast_to` materialisation

**Where:** `run_lifecycle_solver` line ~1972, the terminal age broadcast:

```python
C_list[-1] = jnp.broadcast_to(c_T[None, :, :], (n_z, N_state, n_w))
```

**What to look for:**
- Does the next kernel that consumes `C_list[-1]` actually go through the broadcast lazily, or does it force materialisation?
- Stale comment at line 1969-1971 says "pmap in_axes=None materialises the broadcast lazily" — that's only true on the pmap path. On the new vmap-only path, there is no pmap.

**Severity criteria:**
- MEDIUM: confirmation that the broadcast is currently materialised (forces ~92 MB copy each retirement-age call). The fix would be to gather only the cell's slice from the non-broadcast `c_T` directly.
- LOW: comment-only (stale documentation)

### Item 11 — `working_income_next_full` size and upload pattern

**Where:** `run_lifecycle_solver` line ~1997, then the per-age slice at line 2048.

**What to look for:**
- Total size: `(n_age, n_z, n_eta, n_eps)` = ~80 × 11 × 4 × 4 = 14080 floats ≈ 110 KB. Trivially small.
- Currently held as `np.asarray(...)` → per-age `jnp.asarray(working_income_next_full[t + 1])` upload. Could be uploaded once as a single `jnp.asarray` and sliced on-device.

**Severity criteria:**
- LOW: bytes are tiny, but it's a free win and a clarity win

### Item 12 — Compile-time configuration print

**Where:** `run_lifecycle_solver` lines ~1836-1847 (verbose==1 banner).

**What to look for:**
- The `print(f"  Solver: {solver_config}")` line prints the entire NamedTuple. With multiple boolean toggle fields added, this gets long. Cosmetic.
- The pattern detection (`pattern = "vmap-only..."` lines 1837-1838) calls `len(jax.devices())` — confirm this is cheap (it is) and not in a hot path (it's not).

**Severity criteria:**
- LOW: only mention if you find something that runs more than once.

---

## 4. Reporting format

Create `docs/scans/PERFORMANCE_SCAN_2026-05-06.md` with this structure:

```markdown
# Performance Scan — 2026-05-06

**Scanner:** [agent name]
**Branch HEAD:** <git rev-parse HEAD>
**Scope:** lifecycle/solver.py + lifecycle/__init__.py + lifecycle/precompute.py (build_precompute body)
**Mode:** static read; no benchmarks run; no code edits.

## Summary table

| # | Item | Verdict | Severity | Recommended action |
|---|---|---|---|---|
| 1 | Per-age H→D uploads | YES / NO / SKIPPED | HIGH/MED/LOW/— | one-line fix sketch |
| 2 | D→H syncs | ... |
| ... | ... |

## Detailed findings

### Item N — [Title]
**Location:** `lifecycle/solver.py:XXXX-YYYY`
**Verdict:** YES (finding present)
**Severity:** MEDIUM
**Description:** [2-3 sentences describing what's there]
**Why it matters at this scale:** [estimated cost or impact]
**Recommended fix:** [one paragraph; do not implement]
**Confidence:** [HIGH/MED/LOW that this fix is safe and correct]

[Repeat for each YES finding.]

## Items reported as NO or SKIPPED

[Brief one-line per item explaining why it's clean or why you couldn't determine.]

## Out of scope (noticed but not flagged)

[Anything you saw that was outside the checklist scope and you chose not to investigate.]
```

---

## 5. Don'ts

- **Don't propose architectural changes.** No "rewrite the kernel builder pattern."
- **Don't propose mixed-precision conversions.** Separate handoff.
- **Don't recommend deleting `_*_pmap` paths** — even if vmap-only is faster on smoke, multi-GPU still needs them.
- **Don't suggest replacing JAX with Numba/PyTorch/etc.** The branch is committed to JAX.
- **Don't run benchmarks** as part of this scan. Static read only.
- **Don't fix anything.** Findings go in the report; the user dispatches fixes.

---

## 6. What "good output" looks like

A YES finding example:

> ### Item 11 — `working_income_next_full` per-age upload
> **Location:** `lifecycle/solver.py:1997, 2048`
> **Verdict:** YES
> **Severity:** LOW
> **Description:** `working_income_next_full` is held as a NumPy array; the per-age slice `working_income_next_full[t + 1]` is converted to `jnp.asarray` inside the age loop on every working-age iteration. Total table size at canonical config: 110 KB.
> **Why it matters at this scale:** Negligible bytes (~46 working ages × 110 KB / 4 chunks/sec on PCIe ≈ 0.25 sec total). Effectively free; the win is clarity.
> **Recommended fix:** At line 1997, replace `working_income_next_full = np.asarray(pc.working_income_next)` with `working_income_next_full = jnp.asarray(pc.working_income_next)`. At line 2048, replace `jnp.asarray(working_income_next_full[t + 1])` with `working_income_next_full[t + 1]`.
> **Confidence:** HIGH (one-line change, no behaviour change).

A NO finding example:

> ### Item 5 — `vmap`-able `lax.scan` patterns
> **Verdict:** NO. Searched `solver.py` for `lax.scan` — zero hits in the hot path. The warm-start kill commit removed the only one. Clean.

---

## 7. Workflow

1. Pull the latest `jax-rewrite` branch (the GPU-prep commits should have just landed).
2. Read the four files in scope at least once end-to-end before scanning.
3. Work through items 1-12 in order. Don't jump around — fresh eyes per item.
4. For each item, decide YES / NO / SKIPPED. Cite specific lines.
5. Compile findings into `docs/scans/PERFORMANCE_SCAN_2026-05-06.md`.
6. **Do not commit the report.** Leave it staged for the user to review and commit.
7. Report back with: total findings count by severity (HIGH/MED/LOW), and the path to the report file. Stop.

Total expected report length: 600-1500 words. Brief is good. The user reads the report; an unfocused dump is worse than three crisp findings.

# Handoff: HLO Fusion Audit on the Retirement Kernel Inner Loop

**Branch:** `jax-rewrite`
**Effort:** 2-3 hours. Read-only inspection + brief report. No code changes.
**Output:**
- `docs/scans/HLO_FUSION_AUDIT_2026-05-07.md` — findings report.
- (Optional artifact) the actual HLO dump under `docs/scans/hlo_dumps/` if useful for the report.

---

## Goal

Confirm that the FOC + Newton step + backtracking + EGM lift inside `_build_per_age_retirement_kernel_vmap_only` compiles into a tightly-fused HLO graph with minimal launch overhead, OR find and document the unfused boundaries.

The hot-path arithmetic (`_ccv_log_return_and_grad`, `retirement_foc_jac_ccv`, the 2D Newton update, backtracking line search, EGM lift) is hand-tuned. We want to verify XLA actually fuses it as expected. An unintentional jit boundary in the inner loop would be a performance bug worth fixing; clean fusion means this layer is done and we can stop micro-optimizing it.

**This is a verification task, not an optimization task.** Don't propose code changes unless you find a clear unfused boundary that obviously shouldn't be there.

---

## Background

`verify/mixed_precision_tiny.py` (commit `106cb18`) already implements the HLO-dumping pattern:
- Builds a tiny config
- Calls `jax.jit(fn).lower(*args).compile().as_text()` (or equivalent) to get the HLO
- Writes to a file for inspection

Reuse that pattern. Don't reinvent the dumping code.

---

## What to inspect

The kernel of interest is the **per-cell solve inside the retirement kernel** — specifically the Newton fori_loop that drives `terminal_foc_jac_ccv` / `retirement_foc_jac_ccv` to convergence at each `(z, i_s, i_savings)` triple.

**Key kernel call sites to dump and inspect:**

1. **`_build_per_age_retirement_kernel_vmap_only`** at [solver.py:1897](../../lifecycle/solver.py#L1897) — the single-device retirement kernel. The inner JIT'd function's HLO is what matters.

2. **The FOC + Newton + EGM stack inside one cell.** Specifically:
   - `_ccv_log_return_and_grad` ([solver.py:694](../../lifecycle/solver.py#L694))
   - `retirement_foc_jac_ccv` (search by name)
   - `_newton_fori` Newton iteration ([solver.py:684](../../lifecycle/solver.py#L684) `lax.fori_loop` call)
   - `_backtracking_fori` backtracking line search (search by name)
   - `_egm_scan_cell` EGM scan (search by name)
   - `_lift_to_wealth_grid` ([solver.py:1189](../../lifecycle/solver.py#L1189))

These should all compile into ONE fused HLO kernel for the per-cell solve (modulo the reduction at the end). If they're separate kernels with intermediate D->H or D->D copies, that's the bug.

---

## Method

1. **Set up a representative tiny config.** Match `verify/smoke.py` scale: `state_grid_sizes=(2,3,2,3)`, `n_z=3`, `n_w=12`, `n_savings=12`, `max_iter=30`, etc. Small enough to dump quickly, large enough to exercise all code paths.

2. **Build the retirement kernel.** Use `_build_per_age_retirement_kernel_vmap_only` directly (or the dispatcher with `n_dev=1`). DO NOT run the full lifecycle solver — just build and lower the kernel.

3. **Lower to HLO.** Call `lowered = retirement_kernel.lower(*sample_args)`, then `compiled = lowered.compile()`, then `compiled.as_text()` to get the HLO. Save to file.

4. **Also dump the StableHLO / pre-XLA representation.** `lowered.as_text()` (without `.compile()`) shows the higher-level graph before XLA fusion. Useful to compare "what JAX submitted" vs "what XLA fused into."

5. **Read the HLO.** Look for:
   - **Number of `fusion` blocks.** The inner Newton loop body should be ONE big fusion. Multiple separate fusions in what should be a single per-cell solve = launch overhead per Newton iter.
   - **Memory traffic patterns.** `bitcast`, `copy`, `transpose`, `reshape` operations that move tensors around without compute. Some are fine; clusters of them between fusions suggest unfused boundaries.
   - **Reduction kernels.** `jnp.sum` over the (n_state_quad, n_ret_quad) integration. Should be fused with the multiplications producing `wmu * dRp_das` etc. If reductions are separate kernels from the multiplies, fusion failed.
   - **Inner-loop structure.** `lax.fori_loop` lowers to a `while` loop in HLO. The body of the while should be one fusion (or as few as possible).
   - **Unexpected `block_until_ready` boundaries.** Should NOT appear in this kernel — only at the chunked-runner level (which we're not testing here).

6. **Report findings.** For each kernel boundary you observe in the HLO:
   - Where it is (which Python function maps to it)
   - Whether it's intentional (e.g., the Newton fori_loop's outer iteration boundary is necessary)
   - Whether it could be eliminated (e.g., a stray `jnp.array(...)` that introduces a materialization)

---

## What "good" looks like

A clean retirement-kernel HLO has:
- ONE big `while` loop for the Newton fori_loop, body = one or two large fusions
- ONE big `while` loop for the EGM scan over savings points
- A handful of small fusions for the lift-to-wealth-grid (`jnp.interp` × 3)
- Nothing else in the per-cell path. No standalone copies, no separate reduction kernels for the FOC sums, no per-iteration broadcast/reshape ops.

Total inner-loop fusion count for one cell solve: **single digits**. If you see 20+ fusion blocks per cell, something's wrong.

---

## What "bad" looks like (and would warrant a follow-up)

- **Reduction kernels separate from FOC arithmetic.** `jnp.sum(wmu * dRp_das)` should fuse into one kernel (multiply + sum). If it's two kernels (multiply, then sum) you're paying memory traffic.
- **Per-Newton-iter materialization of `(n_state_quad, n_ret_quad)` tensors.** These should live in registers / shared memory. If they roundtrip through HBM each iter, perf is leaving 2-5× on the table.
- **Repeated gathers inside the Newton loop.** `c_corners` gather should happen ONCE per cell (outside the Newton iter). If it's inside, that's expensive.
- **`jnp.array(...)` inside `@jit`.** This forces materialization. Look for any `jnp.asarray` or `jnp.array` calls inside hot-path kernel bodies.

---

## Pause point

After producing the HLO dump and reading it: **write the report and stop. Do NOT propose or implement fixes.** The user will decide if any flagged issue warrants a follow-up handoff. Optimization work is downstream of this audit, not part of it.

Exception: if you find a **clear, contained, low-risk fix** (e.g., a stray `jnp.array(...)` inside the Newton body that has obviously no purpose), you may flag it AND propose a one-line patch. Do NOT apply the patch yet; the user reviews first.

---

## Implementation checklist

- [ ] Set up tiny config matching `verify/smoke.py` scale.
- [ ] Build `_build_per_age_retirement_kernel_vmap_only` and lower with sample args.
- [ ] Dump HLO (post-XLA fusion). Optionally also dump pre-fusion StableHLO.
- [ ] Read the HLO. Count fusion blocks in the per-cell solve. Identify any unfused boundaries.
- [ ] Repeat for `_build_per_age_terminal_kernel_vmap_only` and `_build_per_age_working_kernel_vmap_only` if time permits — terminal is simpler, working has eta×eps integration on top.
- [ ] Write `docs/scans/HLO_FUSION_AUDIT_2026-05-07.md` with:
  - Method (config used, how dumped)
  - Per-kernel fusion count summary
  - Any flagged unfused boundaries with file:line of the Python source that produced them
  - Verdict: GREEN (clean fusion) / YELLOW (minor boundary worth fixing) / RED (major fusion broken, real perf bug)
- [ ] Commit:
  ```
  docs: HLO fusion audit on retirement-kernel inner loop
  
  Verifies that FOC + Newton + EGM compile into the expected fused
  HLO graph in the single-device retirement kernel. Method, fusion
  counts, and any flagged boundaries documented.
  
  Verdict: <GREEN | YELLOW | RED>. <one-line summary>.
  ```

---

## Out of scope

- **Profiling on real GPU.** This is HLO inspection, not runtime profiling. Save the GPU profile for a separate handoff if needed.
- **Fixing any boundary you find.** Flag and stop. The user decides next steps.
- **Inspecting pmap kernels.** Single-device kernels are the cleanest test of arithmetic-level fusion. Pmap adds device-axis machinery that obscures the inner-loop structure.
- **Optimizing arithmetic.** The arithmetic is already hand-tuned (closed-form, CSE'd). The audit is about fusion structure, not arithmetic.
- **Adding HLO regression tests.** That's a separate hardening task; not part of this audit.

---

## Why this matters

Tonight's review concluded that `_ccv_log_return_and_grad` + `retirement_foc_jac_ccv` are at ~95% of optimal arithmetic-level efficiency. The remaining wall headroom in this layer comes from confirming (or fixing) that XLA fuses the whole inner loop as expected. If fusion is clean: we move on, knowing this layer is done. If fusion is broken somewhere: we have a concrete, file:line-localized perf bug to fix that could yield 1.3-2× wall on the FOC hot path with minimal regression risk.

Either outcome closes a chapter.

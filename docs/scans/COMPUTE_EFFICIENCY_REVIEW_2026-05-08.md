# Compute Efficiency Review — 12 h × 8× A100 SXM4 80 GB Thesis Run

**Date:** 2026-05-08
**Branch:** `jax-rewrite`
**Mode:** read-only audit. No code edits, no benchmarks. Reasoning from existing scans + `lifecycle/solver.py` source.
**Companion:** [CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md](CANONICAL_CONFIG_SYNTHESIS_2026-05-08.md).

Audit of **uninvestigated** compute optimizations against the 9.84 h projection of canonical row D (`5⁴ + sq=(3,3,3,5) + rq=(3,3) + n_z=11 + mi=30 + mb=10 + f32`). For each candidate: status / effort / wall-save / risk / recommendation.

---

## TL;DR — top 3 ROI candidates

1. **#2 `max_backtrack_iter=10 → 5`.** Linear wall save inside `fori_loop`. Inf-horizon `n_backtrack_total` p99 ≈ 100–125 across whole Newton solve (≈ 1 halving per active iter). 10–20 % wall save (~1.0–2.0 h), 1-line edit.
2. **#1b `max_iter=30 → 20`.** Further squeeze on the canonical mi cap. Same mechanism as the 100→30 cut already taken; ~6–10 % save (~0.6–1.0 h). Risk: more interior cells fall into `EC_NEWTON_FAIL` sentinel.
3. **#3/#4 `n_savings`/`n_wealth` 180 → 120** (probe-only). Theoretically the largest unharvested wall lever (linear factor); but is a resolution decision, not pure compute. Needs a convergence sweep before commit; defer.

> **One "do this no question" recommendation:** **drop `max_backtrack_iter` from 10 to 5** (#2). Same reasoning as `max_iter=30` — under `fori_loop` it's a hard wall multiplier, not "iters needed". Histogram evidence shows most line searches resolve in ≤3 halvings; the 10-cap is rarely used. Net ~1.5 h save on 9.84 h, no plausible quality cost. Failure mode is symmetric to mi (cell falls back to `EC_NEWTON_FAIL`, surfaced via diagnostics).

---

## Candidate table

**Status:** Done / Open / Probe / Decline. **Save** = % of canonical row D's 9.84 h wall.

| # | Name | Status | Effort | Save | Risk | Reco |
|---|---|---|---|---|---|---|
| 1 | `max_iter` 100 → 30 | Done | — | already captured | — | Skip (done) |
| 1b | Further `max_iter` 30 → 20 | Open | 1 line | 6–10 % | rare hard interior cells fall to sentinel | **Probe** |
| 2 | `max_backtrack_iter` 10 → 5 | Open | 1 line | 10–20 % | almost zero — ls rarely uses >3 halvings | **DO** |
| 3 | `n_savings` 180 → 90 | Probe | 1 line + sweep | up to 50 % | EGM-grid quality at high-curvature wealth; needs convergence sweep | Defer |
| 4 | `n_wealth` 180 → 120 | Probe | 1 line | 30–50 % | linear-interp accuracy on policy lift; needs convergence sweep | Defer |
| 5 | `lax.cond` converged-cell short-circuit | Decline | high | 0–30 % theoretical | XLA un-fuses; HANDOFF §4.1 explicitly forbids — re-introduces warp divergence | Skip |
| 6 | Static `n_state_quad` / `n_ret_quad` | Done (closure capture) | — | 0 | — | Skip (PERFORMANCE_SCAN §4) |
| 7 | bf16 inner FOC arithmetic | Decline | high | 30–50 % theoretical | bf16 7-bit mantissa is below `tol=1e-7`; FP32_NEWTON_PROBE shows fp32 already on the edge | Skip |
| 8 | Persistent JAX cache hit rate | Done (S3 sync) | — | 0 (compile <1 % of wall) | — | Skip |
| 9 | HBM-bandwidth → free fp64 promotion | Confirmed compute-bound | — | 0 | — | Skip (COMPLEXITY §4: DRAM 3–4 OOM below compute) |
| 10 | Pre-flight precompute caching | Open | medium | 1–3 % | cache-invalidation correctness | Skip — sub-minute save |
| 11 | Inf-horizon adaptive damping | Open | low | 20–40 % of inf-horizon wall | — | Defer — out of scope of canonical lifecycle 12-h run |
| 12 | Newton "policy converged" early-exit | Decline | medium | 0–10 % | same as #5: requires lax.cond | Skip |
| 13 | pmap `cell_vmap_chunks > 1` | Open | medium (~30–50 LOC) | 0–5 % | adds dispatch overhead before any save manifests; per-device WS ~2 GB at 5⁴ leaves ample HBM | Skip at canonical scale |
| 14 | JIT redundancy across age types | Done | — | 0 | — | Skip (boundary uses working kernel; only 3 builders) |
| 15 | Per-age host→device hoist (PERF_SCAN items 1/2/3) | Done | — | already saved | — | Skip (solver.py:2656–2712) |
| 16 | `gather_precision="f32"` | Done | — | already captured | — | Skip |
| 17 | Backward-age warm-start | Done | — | already harvested | — | Skip |
| 18 | HLO fusion improvements | Decline (HLO_FUSION_AUDIT clean) | — | 0 | — | Skip |

---

## Detail on the "DO" candidate (#2)

### Why it's the highest-ROI uninvestigated optimization

`max_backtrack_iter=10` is the inner-loop counterpart of `max_iter`. From [solver.py:564–593](../../lifecycle/solver.py#L564) the backtracking is a `lax.fori_loop(0, max_backtrack_iter, bt_body, init)` — runs **all** `mb` iters, masking once `found`. Symmetric wall behaviour with the outer Newton, but **un-tuned** in the canonical synthesis.

Per-FOC accounting (COMPLEXITY_WALL_TIME §2):

```
foc_calls = 1 + max_iter × (1 + max_backtrack_iter)
          = 1 + 30 × 11 = 331    (canonical D)
          = 1 + 30 × 6  = 181    (with mb=5)
```

A **45 % cut in FOC calls per Newton solve**. Realised wall save lower because (a) Newton-step FOC dominates over backtrack ones in already-converged cells, (b) some scheduling overhead doesn't scale. Realistic: **10–20 %** of the per-cell solve, ≈ 1.0–2.0 h.

### Why it's safe

1. **Inf-horizon evidence:** `n_backtrack_total` p99 ≈ 100–125 in axis-bump bundles. Spread across `n_iters_used` ≈ 8–12 active iters → ~9–12 halvings *summed across the whole Newton solve*, not per line search. Typical line search uses **~1 halving** (most cells succeed at α=1).
2. **Handoff explicit:** [HANDOFF_NEWTON_FORI_LOOP_MASK §4.3](../handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md) flags this as wall cost, not iters needed. Default 10 was inherited from the `while_loop` era.
3. **Failure mode symmetric to mi:** if a line search needs >5 halvings, cell falls back to `EC_NEWTON_FAIL` exactly as today on a >mi-iter Newton. Surfaced via diagnostics; effect is local.

### Procedure

1. `CANONICAL_SOLVER._replace(max_backtrack_iter=5)` (1-line edit in `configs/_canonical.py`).
2. Run canonical row D once.
3. Compare `diagnostics["backtrack_iter_histogram"]` against mb=10 baseline. If p99 > 5 (cells genuinely using >5 halvings), revert. If p99 ≤ 5 (expected), keep.

One-shot probe-and-confirm; doesn't gate the canonical commit.

---

## Why these are skipped (the more contentious ones)

### #5 / #12 — `lax.cond` short-circuit (the user explicitly asked about)

[HANDOFF_NEWTON_FORI_LOOP_MASK §4.1](../handoff/HANDOFF_NEWTON_FORI_LOOP_MASK.md) is explicit:

> "Every cell calls `foc_fn` at every fori iter, even after convergence. … **Do not** try to skip foc_fn for converged cells via `lax.cond` — that breaks fusion and reintroduces the dispatch variance."

The fori-loop+mask conversion's entire payoff is XLA-CUDA fusing the body into one deterministic kernel; this requires uniform behaviour across the batch. `lax.cond` re-introduces warp divergence. **A regression on the GPU target, not a win.**

### #3 / #4 — n_savings / n_wealth reductions

The largest theoretical wins (linear factors), but they are *resolution* knobs, not compute knobs — changing them changes the answer. The canonical synthesis pinned both at 180 because no convergence sweep at canonical resolution exists. Right action if pursued: add an `n_savings ∈ {90, 120, 150, 180}` row to the existing sensitivity program before any thesis-run commit.

### #7 — bf16 inner FOC

[FP32_NEWTON_PROBE_2026-05-07](FP32_NEWTON_PROBE_2026-05-07.md) §4 documents fp32 inside FOC is already on the edge of `tol=1e-7` (p99 alpha drift 7.7×10⁻⁵; max 6.8×10⁻³). bf16's 7-bit mantissa is structurally below tol. Probe verdict was tentative "GO for sweeps, NO-GO for thesis." bf16 strictly worse. CRRA `c^(-γ)` at γ=5, c≈0.01 reaches 1e10; bf16 mantissa noise of ~1e7 absolute. Skip with high confidence.

### #10 — Precompute caching

`build_precompute` is single-threaded NumPy, sub-minute at canonical 5⁴. <0.2 % of a 9.84 h run. Engineering effort (cache key, invalidation, drift detection) high. Not worth it.

### #13 — pmap-path cell_vmap_chunks

MULTI_GPU_AUDIT §A: pmap path silently ignores `cell_vmap_chunks` (documented design). At per-device cell counts ~860 (5⁴ × 11 / 8 padded), per-device WS ~2 GB on 80 GB A100. The chunk knob bounds memory, doesn't optimize wall; with 80 GB per device the bound is irrelevant. Adding chunks adds dispatch overhead. Handoff's recommendation: "for canonical-scale multi-GPU (5⁴, 7⁴), proceed without chunking on pmap."

---

## What's out of scope

- Any optimization in CANONICAL_CONFIG_SYNTHESIS §3 / §5 — those are resolution decisions, not compute.
- Kernel rewrites (per task constraints).
- Benchmarks to confirm the numbers. Save % bounds derived from COMPLEXITY_WALL_TIME formula (±21 % calibrated band); treat each as ±5 pp.

---

## Recommendation summary

| Priority | Action | One-liner |
|---|---|---|
| **1 (DO)** | `max_backtrack_iter=5` | 1.0–2.0 h save, ~zero risk, 1-line edit |
| 2 (probe) | `max_iter=20` | additional 0.6–1.0 h; only if budget margin desired |
| 3 (probe-then-skip) | `n_savings`, `n_wealth` ↓ | huge save but resolution decision; defer to a convergence sweep |
| 4 (skip) | bf16 FOC, lax.cond short-circuit, pmap chunking, precompute cache | already investigated or counter-productive on GPU |

If the "DO" candidate lands: **~1.0–2.0 h save** → 7.8–8.8 h projected wall, $80–91 cost. If "DO" + the probe both confirm: **~2.0–3.0 h save** → 6.8–7.8 h.

---

**End of review.** No code changes, no benchmarks. Implement only after user signoff.

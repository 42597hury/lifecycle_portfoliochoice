# Handoff: Audit `inf_horizon_solver.py` (and `predictability_ablation.py`) for Drift

**Branch:** `jax-rewrite`
**Effort:** 2-3 hours audit + (conditional) 2-8 hours fixes depending on findings.
**Workflow:** **AUDIT FIRST, ASK BEFORE IMPLEMENTING.** Do not start fixes until findings have been reported and the user has approved scope.
**Output:**
- `docs/scans/INF_HORIZON_AUDIT_2026-05-07.md` — written audit report.
- (Conditional) commit(s) with fixes after user approval.

---

## Background

Tonight's session has piled solver-side improvements on top of the rtb-as-state migration:
- 4-D state vector everywhere (`(cy, spr, rtb, y_1)`)
- Cell-vmap chunking (`cell_vmap_chunks`)
- Mixed precision gather (`gather_precision`)
- Newton-iter histogram exposure (`use_fori_newton`, `n_iters_max`/`n_backtrack_total` returned per cell)
- Backward-age warm-start (`use_backward_age_warm_start`, `init_a_s_arr`/`init_a_b_arr` passed into kernels)
- Sim CCV fix
- pmap/vmap-only dispatch on `n_dev`

`run_lifecycle_solver` was updated for all of these. Two adjacent components have NOT been visited:

1. **`lifecycle/inf_horizon_solver.py`** — calls `_build_per_age_retirement_kernel` directly ([inf_horizon_solver.py:479, 608](../../lifecycle/inf_horizon_solver.py#L479)) and runs its own time loop. Doesn't use `run_lifecycle_solver`. **High drift risk.**

2. **`lifecycle/predictability_ablation.py`** — composes `DiscretizationConfig` for systems I-IV by axis-projecting a 4-D template. Already 4-D aware on its face but never re-tested post-rtb-as-state.

The audit's job is to determine what works, what's silently broken, and what's been left behind by the recent solver work.

---

## Audit checklist

For each item: report **GREEN** (compatible, no work needed), **YELLOW** (works but missing recent improvements / should be hardened), or **RED** (broken — would error or produce wrong results today).

### A. `inf_horizon_solver.py` — kernel call signature

1. Find every call to `_build_per_age_retirement_kernel` and `retirement_kernel(...)` (the returned callable). Compare:
   - **Builder arg count** vs current [solver.py:1809](../../lifecycle/solver.py#L1809) signature: `(pcj, mp, sc, n_dev, n_z, N_state, per_is_tensors)`.
   - **Callable arg count** vs current vmap-only callable signature in [solver.py:1897](../../lifecycle/solver.py#L1897). Specifically: does `inf_horizon_solver` pass `init_a_s_arr` and `init_a_b_arr` (warm-start arrays)? They were added recently — `inf_horizon_solver`'s call at line 488 passes only `(C_old, pension_zero, psi_one)`.
2. **Builder return signature.** Does the kernel now return a 5-tuple `(c, s, b, n_iters_max, n_backtrack_total)` instead of `(c, s, b)`? If yes, `inf_horizon_solver`'s 3-tuple unpacking at line 488 is broken.

**Where to read:**
- [inf_horizon_solver.py:479-490](../../lifecycle/inf_horizon_solver.py#L479)
- [inf_horizon_solver.py:608-620](../../lifecycle/inf_horizon_solver.py#L608)
- [solver.py:1897+](../../lifecycle/solver.py#L1897) — vmap-only kernel `call(...)` definition for the actual callable signature.

### B. `inf_horizon_solver.py` — orchestration features missing from `run_lifecycle_solver`

`run_lifecycle_solver` does the following at the orchestrator level (NOT inside the kernel):
- Threads warm-start init across ages (uses previous age's `S, B` to init Newton at the next-younger age).
- Aggregates per-age `n_iters_max` / `n_backtrack_total` into `diagnostics["newton_iter_histogram"]`.
- Dispatches chunked vs un-chunked path based on `cell_vmap_chunks > 1`.
- Calls `verify_runtime_platform()` warnings.
- Manages `solve_control` (youngest_age_to_solve, checkpointing).

For each: does `inf_horizon_solver` do equivalent work, ignore it, or break in the absence of it? Especially:

1. **Warm-start.** `inf_horizon_solver` has a fixed-point iteration over ages? Or one-shot? If iterative, does each iteration's warm-start init come from the previous iteration's policy?
2. **Chunking dispatch.** Does `inf_horizon_solver` honor `solver_config.cell_vmap_chunks`? If user sets it to 4 expecting memory savings on a large state, does anything actually happen?
3. **Diagnostics.** Does `inf_horizon_solver`'s return include the same Newton-iter histogram a regular run would?

### C. `inf_horizon_solver.py` — mixed precision plumbing

The kernel builders thread `gather_dtype` through (resolved from `sc.gather_precision`). Does `inf_horizon_solver` pass an `mp` config that's compatible? Look for the `mp = ModelPrecompute(...)` construction in [inf_horizon_solver.py:475](../../lifecycle/inf_horizon_solver.py#L475) area — does it match what `run_lifecycle_solver` builds?

### D. `predictability_ablation.py` — system I-IV buildability

Run a smoke-build (no solve, just config + `build_precompute`) for each system:

```python
from lifecycle.predictability_ablation import (
    SYSTEM_I, SYSTEM_II, SYSTEM_III, SYSTEM_IV,
    build_system_disc_config, build_system_var_config,
)
from lifecycle.precompute import build_model, build_precompute
from configs._canonical import BASE_CONFIG, CANONICAL_DISC

for sys_def in (SYSTEM_I, SYSTEM_II, SYSTEM_III, SYSTEM_IV):
    print(f"--- {sys_def.label} ---")
    disc = build_system_disc_config(sys_def, CANONICAL_DISC)
    var = build_system_var_config(sys_def)
    model = build_model(BASE_CONFIG, var, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    print(f"  N_state={pc.N_state}, n_z={pc.n_z}, ok")
```

Each should print without raising. If any system raises (shape mismatch, axis projection failure), that's a RED.

(Function names above are best guesses — adjust to actual exported API in `predictability_ablation.py`.)

### E. `predictability_ablation.py` — solve a tiny System I

After D passes, attempt a tiny solve for System I (1-axis state, smallest case):

```python
from lifecycle.solver import run_lifecycle_solver
from lifecycle.model import SolveControl, SolverConfig
sc = SolverConfig(...)  # canonical-small
sol_ctrl = SolveControl(youngest_age_to_solve=95)  # 4 retirement ages only
C, S, B, diag = run_lifecycle_solver(model, pc, sc, solve_control=sol_ctrl, verbose=1)
print("alphas:", float(S.min()), float(S.max()))
```

If this passes, the 4-D-aware solver still works on collapsed 1-D states. If it fails, the rtb-as-state migration broke axis-cardinality flexibility somewhere.

### F. Exercise scripts / coverage gap

Search for `verify_*.py` or any script that currently runs inf_horizon_solver or predictability_ablation:

```bash
grep -rln "inf_horizon\|predictability_ablation\|infinite_horizon\|SYSTEM_I\|SYSTEM_II\|SYSTEM_III\|SYSTEM_IV" verify_*.py scripts/ 2>/dev/null
```

If nothing exercises them, that's a YELLOW: "no regression coverage exists, recommend a tiny verify script as part of the fix bundle." Don't write the verify script in this handoff (scope creep) but recommend it.

---

## Pause point

After completing the audit (A-F):

1. Write `docs/scans/INF_HORIZON_AUDIT_2026-05-07.md` with per-check verdicts and concrete file:line refs for every RED.
2. **Stop. Send the report. Wait for approval before implementing fixes.**

Reasons to pause:
- Some fixes are local 5-line patches (kernel signature drift). Easy yes.
- Some fixes are bigger structural questions (e.g. "should `inf_horizon_solver` be refactored to call `run_lifecycle_solver` with `terminal_age=youngest_age_to_solve` instead of duplicating orchestration?"). User needs to weigh in.
- A fix that LOOKS small (add `init_a_s_arr` to the call) might cascade into refactoring the inf-horizon time loop to thread warm-start across iterations. The user should know what they're approving.

---

## Conditional implementation

If user approves: implement fixes per the report, run a tiny smoke (System I + retirement-only inf-horizon), commit each logical fix as its own commit. Suggested commit shape:

```
inf_horizon: <one-line summary>

<2-3 sentences explaining the drift item this commit addresses,
the fix, and the validation done.>

No math change. <Or: math affected because <reason>.>
```

For each commit:
- Show the user the diff before pushing if the change is non-trivial (>30 LOC).
- Run `verify_smoke.py` after each fix to confirm regular run_lifecycle_solver still works (don't break the main path while fixing the side path).
- If a fix surfaces ambiguity (e.g. "warm-start init for an inf-horizon iteration is conceptually different from age-step warm-start — what semantics do we want?"), pause and ask.

---

## Out of scope

- **Refactoring `inf_horizon_solver` to share orchestration with `run_lifecycle_solver`.** That's a larger structural conversation; flag it as a recommendation in the report but don't do it here unless explicitly approved.
- **Adding a published-quality benchmark for inf-horizon vs the lifecycle finite-horizon limit.** Out of scope; should be its own handoff.
- **Adding new ablation systems beyond I-IV.** Out of scope.
- **Performance work on inf-horizon.** Audit + fix correctness only.

---

## Implementation checklist

### Phase A — Audit
- [ ] (A) Audit `inf_horizon_solver.py` kernel call signature drift.
- [ ] (B) Audit `inf_horizon_solver.py` for missing orchestration features (warm-start, chunking, diagnostics).
- [ ] (C) Audit `inf_horizon_solver.py` mixed-precision plumbing.
- [ ] (D) Smoke-build all four systems via `predictability_ablation.py`.
- [ ] (E) Tiny-solve System I via `run_lifecycle_solver`.
- [ ] (F) Search for existing exercise / verify scripts; report coverage gap.
- [ ] Write `docs/scans/INF_HORIZON_AUDIT_2026-05-07.md` with per-check verdicts and concrete file:line refs.
- [ ] **STOP. Send report. Wait for user approval.**

### Phase B — Conditional implementation (only after approval)
- [ ] Apply approved fixes one logical commit at a time.
- [ ] Run `verify_smoke.py` after each fix to confirm no regression on the main path.
- [ ] Push.

---

## Why a separate handoff

- Distinct from multi-GPU audit: that one targets the pmap path of `run_lifecycle_solver`. This one targets a different consumer of the same kernels (`inf_horizon_solver`) plus an adjacent module (`predictability_ablation`).
- Distinct from sim-EE / arbitrage / Newton-diag work: those are diagnostics, not solver paths.
- Distinct from rtb-as-state migration: that ran weeks ago; this one checks that the migration was complete in places that weren't actively exercised tonight.

---

## Why this matters

`inf_horizon_solver` and the ablation systems are publication-grade artifacts: System II, III, IV are each a separate experiment in the thesis. If any of them is silently broken by recent solver changes, every cross-system result table downstream is wrong. The audit is cheap; the cost of NOT auditing is finding out post-publication.

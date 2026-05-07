# Handoff: Diagnose + Fix Inf-Horizon Bundle Save on Interrupt

**Branch:** `jax-rewrite`
**Effort:** ~3-4 hours. Diagnosis + fix + smoke verification.
**Output:**
- Modified `lifecycle/inf_horizon_solver.py` (fix the interrupt path; possibly add periodic checkpointing).
- (Optional) modified `verify/benchmark_inf_horizon.py` if the launcher needs adjustment.
- Smoke test demonstrating the fix.

**Critical constraint:** **Do not touch `lifecycle/solver.py` or any lifecycle solver path.** Lifecycle solver behavior must remain bit-identical before/after this change. Verify with `verify/smoke.py`.

---

## Background — what happened tonight

Tonight (2026-05-07) we ran an inf-horizon benchmark on Lambda 2× H100:

```
verify/benchmark_inf_horizon.py at 8^4 × n_z=2 + Lobatto y_1
max_iter=50, tol=1e-5, damping=1.0
```

The run iterated for ~70 minutes (21 fixed-point iters out of 50, contracting ~0.81/iter). User sent SIGINT via `tmux send-keys -t infh C-c`. The tmux session showed `^C === EXIT ===` shortly after.

**Result: NO bundle was saved.** The aborted state — 21 iterations of computed policies — was completely lost. Only the iteration log (`infh.log`) survived, snapshotted to S3 manually.

The script `verify/benchmark_inf_horizon.py` looks correct on the surface:

```python
C, S, B, diag = run_infinite_horizon_solver(model, pc, ...)
# ... print results ...
bundle_path = save_policy_bundle(BUNDLE_DIR, C, S, B, diagnostics=diag, ...)
```

But on `KeyboardInterrupt`, `run_infinite_horizon_solver` raises the exception up to the caller. The script's `save_policy_bundle(...)` line never runs because the exception propagates out of the try-less call.

**Comparison: `lifecycle/solver.py` does this correctly.** It has:

```python
try:
    for t in range(...):
        # backward induction ...
except KeyboardInterrupt:
    solve_status = "interrupted"
    # ... fall through to histogram aggregation + save
```

`run_lifecycle_solver` returns partial policies on interrupt. The script then saves them via `save_policy_bundle`. **The inf-horizon path doesn't do this.**

---

## What to fix

### Step 1 — Audit `lifecycle/inf_horizon_solver.py`

Read the function `run_infinite_horizon_solver` (around line 480). Specifically look at the iteration loop (around lines 547-606). Confirm:

- There is no `try / except KeyboardInterrupt` wrapping the loop.
- On interrupt, the function raises out without returning the (partial) policies that have been computed so far.
- The current iteration's `C_old, S_old, B_old` are valid policies at that point (they survive across iterations on the host as NumPy arrays per the solver's design).

Document the diagnosis in a one-paragraph findings note before patching.

### Step 2 — Add `KeyboardInterrupt` handling in `run_infinite_horizon_solver`

Mirror the pattern used by `run_lifecycle_solver`:

```python
converged = False
n_iter_done = 0
try:
    for it in range(max_iter):
        # ... existing iteration body ...
        n_iter_done = it + 1
        if stop_err < tol:
            converged = True
            break
except KeyboardInterrupt:
    if verbose:
        print(f"\n  Inf-horizon solve interrupted at iter {n_iter_done}. "
              f"Returning partial output.", flush=True)
```

After the try/except, the function continues to its normal post-loop processing (build diagnostics, return). The histograms and policies as of the interrupt point are returned to the caller.

**Critical:** the `C_old, S_old, B_old` arrays at the time of interrupt are the latest converged-up-to-that-iteration policies. They're valid bundle contents. Mark `converged=False` in the diagnostics dict (which the solver already does naturally if max_iter not reached).

### Step 3 — Optional: add periodic checkpointing

The lifecycle solver has `checkpoint_every_n_ages` so partial progress is saved to disk every N ages. Inf-horizon could similarly support `checkpoint_every_n_iters`. **Don't add this unless time permits**; the interrupt fix is the load-bearing thing.

If you do add it: thread `checkpoint_path: str | None = None` and `checkpoint_every_n_iters: int | None = None` into `run_infinite_horizon_solver`'s signature, and call a small helper to save the current `(C_old, S_old, B_old)` plus running diagnostics every N iters. Match the format of `lifecycle/policy_io.py:save_policy_bundle` so the same loader works.

### Step 4 — Verify with smoke

**Gate 1 — Normal completion still works.**

Run a tiny inf-horizon at small config (template state_grid=(2,2,2,2), n_z=3, n_w=10, max_iter=5, tol=1e-2). Should complete, return policies, save bundle. Confirm `converged=True` or `converged=False` (depending on whether 5 iters is enough at this config). Bundle should exist on disk with valid `policy_arrays.npz` and `diagnostics.pkl`.

**Gate 2 — Interrupt now produces a partial bundle.**

Add a small synthetic interrupt test. One approach: use Python's `signal.alarm` to send SIGINT after some seconds:

```python
import signal
signal.signal(signal.SIGALRM, lambda *_: signal.raise_signal(signal.SIGINT))
signal.alarm(3)  # interrupt after 3 seconds
C, S, B, diag = run_infinite_horizon_solver(model, pc, max_iter=50, tol=1e-9, ...)
# Should return partial policies, not raise
assert diag.get("converged") is False
assert C.shape == expected_shape
```

Or run a tiny lifecycle that's slow enough to interrupt (small wealth grid + lots of iterations).

**Gate 3 — Lifecycle solver bit-identity.**

Run `verify/smoke.py` and confirm alpha ranges are bit-identical to a pre-change capture. (The fix should only touch `inf_horizon_solver.py`; lifecycle solver should be untouched.)

### Step 5 — Document in commit message

```
inf_horizon: catch KeyboardInterrupt to enable partial bundle save

Previously run_infinite_horizon_solver raised KeyboardInterrupt out of
the iteration loop without returning, so any caller wrapping it
in save_policy_bundle saw the exception propagate past the save call.
Result: aborted runs lose all computed policies.

Fix: try/except KeyboardInterrupt around the iteration loop, mirror
of lifecycle solver's pattern. On interrupt, mark converged=False,
fall through to diagnostics aggregation, and return partial policies.

Verified at tiny config: (1) normal completion still saves bundle as
before, (2) synthetic SIGINT after N iters returns valid C/S/B with
converged=False and the diagnostics histogram populated up to the
interrupt point. Lifecycle solver bit-identity verified on
verify/smoke.py.

No math change to inf-horizon FOC, Newton, or fixed-point logic.
```

---

## Pause points

The agent must STOP and ask the user when:
- Diagnosis reveals the bundle save issue is **not** a missing interrupt handler — i.e., the function appears to handle interrupts but bundles still aren't saving. There's a different bug; surface it before patching.
- Lifecycle bit-identity (Gate 3) fails after the change. Indicates the fix accidentally leaked into the shared kernel-build path. Stop and report.
- Synthetic SIGINT test (Gate 2) doesn't trigger the interrupt path. Could mean Python's signal model doesn't intersect cleanly with JAX's blocking calls; the agent may need a different test approach.

---

## Out of scope

- **Lifecycle solver changes.** The lifecycle solver already has interrupt + partial-save machinery; don't touch it.
- **Bundle format changes.** `save_policy_bundle` schema is unchanged.
- **Math changes.** No FOC, Newton, fixed-point, or quadrature changes.
- **Performance optimization** beyond what's strictly needed for the fix.
- **Multi-GPU testing.** CPU smoke is enough to verify the interrupt fix (the issue is host-side control flow, not GPU-specific).

---

## Implementation checklist

- [ ] **Step 1**: Audit `inf_horizon_solver.py:run_infinite_horizon_solver`. Confirm the missing `try / except KeyboardInterrupt`. Document in findings.
- [ ] **Step 2**: Add the `try/except` wrapper around the iteration loop. Set `converged=False` on interrupt. Fall through to diagnostics aggregation + return.
- [ ] **Step 3** (optional): periodic `checkpoint_every_n_iters` plumbing. **Skip unless trivial.**
- [ ] **Gate 1**: Tiny inf-horizon completes normally and saves bundle.
- [ ] **Gate 2**: Interrupt during iteration produces a valid partial bundle.
- [ ] **Gate 3**: `verify/smoke.py` (lifecycle) bit-identical to pre-change.
- [ ] Commit + push.

---

## Why this matters

- We just lost 21 iters of compute (~70 min, ~$10) on tonight's inf-horizon run because of this gap. Future inf-horizon runs at production scale (Systems I-IV ablation, calibration cycle 2) might run for 1-3 hours each. **Losing one to an unintended abort is real money.**
- Inf-horizon convergence is asymptotic — at iter 30 we might be at stop=1e-3, by iter 50 at 1e-4. The intermediate state has scientific value even if not fully converged. Right now we throw it away on any interrupt.
- The fix is a pattern proven to work in the lifecycle solver. Just port it.

---

## Related files (for the agent's grep)

- `lifecycle/inf_horizon_solver.py` — the function to patch.
- `lifecycle/solver.py:2746-2749` (approximately) — the `try/except KeyboardInterrupt` pattern to mirror.
- `lifecycle/policy_io.py:save_policy_bundle` — bundle save interface; should not need changes.
- `verify/benchmark_inf_horizon.py` — the launcher that calls `run_infinite_horizon_solver` and then `save_policy_bundle`. Confirm no changes needed here.
- `docs/handoff/HANDOFF_NZ_ONE_FOR_INF_HORIZON.md` — adjacent inf-horizon work (separate handoff; no conflict).

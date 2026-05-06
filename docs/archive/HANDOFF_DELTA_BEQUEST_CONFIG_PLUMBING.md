# HANDOFF — Plumb DELTA_BEQUEST through SolverConfig

## Goal

Make the luxury-bequest shift δ a configurable numerical knob alongside the other solver-side regularizers (`tiny_savings`, `min_consumption`, `euler_inv_floor`, etc.) so that the validation-plan sensitivity sweep over δ ∈ {0.001, 0.005, 0.01, 0.02} can be run from sweep configs without editing module source. Also ensures EE diagnostics use the same δ the bundle was solved with.

## Why SolverConfig

- δ is a **numerical regularization** (the framing the paper will defend), not an economic parameter or a grid choice.
- It belongs in the same category as `tiny_savings`, `min_consumption`, `min_wealth_inv`, `euler_inv_floor` — bounds that prevent solver pathology, all currently fields on `SolverConfig`.
- This keeps the codebase's treatment of numerical knobs consistent.

## Current state

```python
# lifecycle/model.py:273
DELTA_BEQUEST = 0.005

# Used (hard-coded) at three solver call sites:
#   lifecycle/solver.py:906      retirement FOC kernel
#   lifecycle/solver.py:1060     working-age FOC kernel
#   lifecycle/solver.py:1828     terminal-step EGM (delta = DELTA_BEQUEST)
#
# And in two diagnostic call sites:
#   scripts/diagnostics/_diag_euler_errors.py:752,857  EE kernels
#   scripts/diagnostics/_diag_split_rule_sanity.py:268,442  bespoke kernels
```

The kernels already accept `delta` as a function argument (`_shifted_bequest_mu_and_mup(W, A, gamma, b_bar, delta)`). The plumbing job is to thread it from `SolverConfig` → solver Python wrapper → numba kernel parameter.

## Changes

### 1. Add SolverConfig field

[lifecycle/model.py](../../lifecycle/model.py), in `SolverConfig` near the other regularization floors (around the `tiny_savings`, `min_consumption`, `euler_inv_floor` block):

```python
delta_bequest: float = DELTA_BEQUEST   # luxury-bequest numerical shift; see DELTA_BEQUEST
```

`DELTA_BEQUEST` stays as the module-level default so legacy callers still work.

### 2. Thread `delta` through the production FOC kernels

[lifecycle/solver.py](../../lifecycle/solver.py):

- **`compute_foc_jac_retirement_quad`** (around line 805) — add `delta` as a scalar parameter at the end of the kwargs (before `min_wealth_inv`/`min_consumption`/`prob_skip`). Replace the hardcoded `DELTA_BEQUEST` at line 906 with the new `delta` parameter.
- **`compute_foc_jac_working_quad`** (around line 945) — same treatment; replace line 1060.
- **`compute_terminal_foc_jac_shifted`** — already takes `delta` as a parameter. No change needed inside the kernel.

The non-njit callers of these kernels (in `solve_portfolio_2d_*`, `solve_portfolio_unconstrained_*`, the per-Newton helpers, and `solve_terminal_age`) read `solver_config.delta_bequest` and pass it as `delta=...`.

For `solve_terminal_age` (around line 1828): replace `delta = DELTA_BEQUEST` with `delta = solver_config.delta_bequest`.

### 3. Diagnostic kernels — same treatment

[scripts/diagnostics/_diag_euler_errors.py](../../scripts/diagnostics/_diag_euler_errors.py):

- `_compute_euler_sum_retirement_continuous` and `_compute_euler_sum_working_continuous` should take `delta` as a scalar parameter. Replace hardcoded `DELTA_BEQUEST` at lines 752 and 857.
- `_evaluate_age_errors` (the parallel njit wrapper) needs the new `delta` parameter passed through to both branches.
- The Python caller `_load_bundle_context` should populate a `delta_bequest` field on the returned `EulerBundleContext` (read from bundle metadata if present, fallback to `DELTA_BEQUEST` for legacy bundles).
- All callers in `_diag_simpath_worst_cells.py` and `_diag_split_rule_sanity.py` pass `ctx.delta_bequest` to the kernels.

[scripts/diagnostics/_diag_split_rule_sanity.py](../../scripts/diagnostics/_diag_split_rule_sanity.py):

- The bespoke kernels `_foc_and_decomp_kernel` and `_per_node_bequest_kernel` take `delta` as a scalar parameter. Replace hardcoded `DELTA_BEQUEST` at lines 268 and 442.
- The Python `evaluate_at_cell` and `per_node_bequest` wrappers accept `delta` and forward it.
- `main()` reads `delta = ctx.delta_bequest` (from the bundle context, set by `_load_bundle_context`) and passes it to all kernel invocations.

### 4. Bundle metadata — record δ at solve time

[lifecycle/policy_io.py](../../lifecycle/policy_io.py) `save_policy_bundle`: add `delta_bequest` to the `metadata.json` payload, sourced from the solver's run config. This way the EE diagnostic can re-evaluate the Euler equation under the same δ the bundle was solved with — without it, a δ-mismatch between solve and diagnostic produces spurious EE residuals.

[scripts/diagnostics/_diag_policy_convergence.py](../../scripts/diagnostics/_diag_policy_convergence.py) `_extract_disc_config` (or wherever metadata is parsed in `_load_bundle_context`): read `delta_bequest` from metadata; fallback to `DELTA_BEQUEST` for legacy bundles (with a `warnings.warn` so the user knows they're using the default).

### 5. Sweep configs

Pattern for δ-sweep configs (e.g. `configs/run_delta_sensitivity_001.py`):

```python
from configs._canonical import BASE_CONFIG, CANONICAL_DISC, CANONICAL_SOLVER, PREDICTABILITY_SYSTEM

# Override δ for this sweep cell.
SOLVER = CANONICAL_SOLVER._replace(delta_bequest=0.001)
DISC = CANONICAL_DISC
# ... rest of the file follows the canonical pattern
```

The user produces four such configs for δ ∈ {0.001, 0.005, 0.01, 0.02} and launches them as a sweep.

## What does NOT change

- `DELTA_BEQUEST` module constant in [lifecycle/model.py](../../lifecycle/model.py) stays — it's the documented default and the fallback for legacy callers / bundles.
- The shifted-bequest helpers (`bequest_utility`, `bequest_marginal`, `bequest_marginal_inv`, `_shifted_bequest_mu_and_mup`, `_shifted_bequest_mu`) already take `delta` as a parameter. Untouched.
- The simulator stays untouched (no bequest utility computed there).
- `bequest_*` model.py helpers keep the `delta=DELTA_BEQUEST` default kwarg.

## Validation

1. **Smoke**: `from lifecycle.model import SolverConfig; sc = SolverConfig(delta_bequest=0.01); print(sc.delta_bequest)` — confirms field exists and override works.
2. **End-to-end at default**: a small smoke solve with `CANONICAL_SOLVER` (default δ=0.005) should produce a bundle bit-identical to the pre-plumbing solve at the same δ. This confirms threading through the kernels didn't perturb the math.
3. **Override**: a small smoke solve with `SOLVER = CANONICAL_SOLVER._replace(delta_bequest=0.001)` should produce a *different* bundle (since δ changed). The bundle's `metadata.json` should report `delta_bequest: 0.001`.
4. **Diagnostic**: re-load that bundle, run the EE diagnostic. The kernels should pick up `0.001` from the bundle metadata, not the canonical `0.005`. Sanity-check via a print/log of the δ used.

## Out of scope

- Not exposing δ via `BASE_CONFIG` (would imply economic content, weakens the "numerical guardrail" framing).
- Not making δ a `DiscretizationConfig` field (it's not a grid choice).
- Not retroactively re-solving with the new plumbing — existing bundles continue to work via the legacy fallback.

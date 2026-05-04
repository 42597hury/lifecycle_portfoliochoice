# HANDOFF — Propagate Lobatto config from solver to EE diagnostic eval rule

## Scope

The Euler-error diagnostics (`_diag_euler_errors.py`, `_diag_gridpoint_ee.py`, `_diag_invalid_cells.py`, `_diag_simpath_worst_cells.py`, `_diag_arbitrage_quadsweep.py` to varying degrees) construct an "evaluation" Precompute via `_build_eval_disc`, which derives a per-axis quadrature spec from the solver's `DiscretizationConfig` plus an `eval_mode` ("same", "next_finer", "double") that adjusts K per axis. **What it does NOT do is propagate the `ret_lobatto_Z` and `state_lobatto_Z` fields** — so when the solver uses Hermite-Lobatto with explicit ±Zσ endpoints, the eval cloud falls back to pure Gauss-Hermite without those endpoints.

This causes a rule mismatch: the eval cloud has no explicit tail nodes at ±Zσ, so it cannot reproduce the bankruptcy detection the solver was designed against. EE residuals computed against this mismatched eval rule conflate two different things:

1. genuine policy inaccuracy (what we want to measure), and
2. the disagreement between Lobatto and GH about what's "optimal" at the tail.

The fix is: make the eval rule use **the same Lobatto config** as the solver, with K bumped per `eval_mode`. Concretely:

- `same`: identical config (including same Lobatto Z and same K)
- `next_finer`: K+2 per axis on Lobatto axes (or per current convention), Z unchanged on Lobatto axes
- `double`: K*2 — but Lobatto K is restricted to {3, 5, 7}, so this needs careful handling (recommend: clamp to K=7 if 2K > 7, log a warning)

The fix should leave non-Lobatto axes (those with `lobatto_Z[d] is None`) on pure GH as today. Only Lobatto axes get tail-propagation.

## Deliverables

1. A code change that propagates `ret_lobatto_Z` and `state_lobatto_Z` from solver to eval `DiscretizationConfig` inside the diagnostic infrastructure.
2. The change should be configurable: an opt-out flag on the diagnostics (e.g. `--eval-disable-lobatto`) so the user can still run a "GH-only eval" comparison if they want — the rule-mismatch number is itself diagnostically interesting.
3. A short verification: re-run `_diag_euler_errors` (sim-path EE) on the v4_lobatto bundle (`saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v4_lobatto`) with the fix applied. Report the new mean / median / max log10|EE| numbers and compare to the previous run (which had mean=−2.62, median=−2.53, max=−0.81 under the GH-mismatched eval).
4. A short note in `docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md` documenting the change in semantics and what the opt-out flag is for.

## What you need to understand before changing anything

Read these files in this order:

1. **`lifecycle/discretization.py`** — specifically the docstrings of `_build_axis_grid` (≈line 582), `get_return_quadrature` (≈line 634), `get_state_quadrature` (≈line 745), and `_normalize_lobatto_Z` (≈line 533). Understand the K/Z constraint for Lobatto: K odd in {3, 5, 7}, Z windows depend on K (Z>1 for K=3, Z≥√5 for K=5, K=7 has discrete valid windows).

2. **`lifecycle/quadrature_with_tails.py`** — read `gauss_hermite_prescribed_tails(K, Z)` to confirm what Lobatto axes actually produce.

3. **`lifecycle/model.py`** ≈ line 119–127 — the `DiscretizationConfig` dataclass fields `n_ret_nodes_1d`, `n_state_quad_nodes`, `ret_lobatto_Z`, `state_lobatto_Z`. Note that `ret_lobatto_Z` accepts `None`, `float`, or a length-`n_ret` tuple of `None|float`.

4. **`scripts/diagnostics/_diag_policy_convergence.py`** ≈ line 95 — `_build_disc_config(raw)` constructs a `DiscretizationConfig` from a metadata dict. Verify that it ALREADY reads `ret_lobatto_Z` and `state_lobatto_Z` from `raw` — if it doesn't, that's a parallel bug to fix.

5. **`scripts/diagnostics/_diag_euler_errors.py`** ≈ line 88 — `_build_eval_disc(base, n_state, n_ret, mode, ret_override, state_override, eta_override, eps_override)`. This is the function that needs the surgery. Trace what it does for each `mode`. Note that `ret_override` and `state_override` exist but currently only override the K tuples, not the Lobatto Z. Decide whether to add `ret_lobatto_Z_override` and `state_lobatto_Z_override` parameters or to just always propagate the base's Lobatto config.

6. **`scripts/diagnostics/_diag_euler_errors.py`** ≈ line 145 — `_load_bundle_context` calls `_build_eval_disc`. Trace the propagation chain.

7. **`scripts/diagnostics/_diag_arbitrage_quadsweep.py`** — reads `ret_lobatto_Z` from `base_disc` correctly already. Use this as the reference for how propagation should look in `_build_eval_disc`.

8. **`scripts/diagnostics/_diag_gridpoint_ee.py`** and **`_diag_invalid_cells.py`** — these call `_load_bundle_context` from `_diag_euler_errors`. They will inherit the fix automatically. Just verify nothing in them re-strips the Lobatto config.

9. **`scripts/diagnostics/_diag_simpath_worst_cells.py`** — same dependency on `_load_bundle_context`. Should inherit.

## Edge cases to think through

- **K=5 Lobatto + `next_finer` (K+2 = 7)**: 7 is in {3,5,7}, so this works. Z=7 ≥ 3.28 → valid window for K=7. Standard happy path.

- **K=5 Lobatto + `double` (K*2 = 10)**: 10 is even and not in {3,5,7}. Cannot use Lobatto at K=10. Options: (a) clamp K to 7 with a warning, (b) fall back to GH on that axis with a warning, (c) raise an error. Recommendation: clamp to 7 with a warning printed once.

- **K=3 Lobatto + `next_finer` (K+2 = 5)**: 5 is valid. Z>1 satisfied for K=5 if Z≥√5≈2.236; if user's solver Z is below √5 (only possible if they've intentionally chosen K=3 with very tight Z), fall back to GH on that axis with warning.

- **K=7 Lobatto + `next_finer` (K+2 = 9)**: 9 is not in {3,5,7}. Cannot use Lobatto. Same options as above; recommendation: clamp to 7 (no change in K) with a warning, since the user already picked the maximum supported K.

- **Z value validity for new K**: when changing K, re-validate Z against the K's valid windows in `quadrature_with_tails.py`. K=7 Z=5 needs Z in (1.81, 2.86) ∪ [3.28, ∞), so 5 is OK. K=7 Z=2 fails (in the gap). Document the validation logic clearly.

## Implementation-plan deliverable

Before writing code, produce a written implementation plan that covers:

1. **Function signature changes** to `_build_eval_disc`. What new parameters? Which existing parameters are affected?
2. **Default behaviour** when the user doesn't supply Lobatto overrides. Recommendation: propagate solver's Lobatto Z by default (this is the bug-fix behaviour).
3. **Per-axis K-adjustment logic** — how to handle the Lobatto K-restriction issues listed above. Express as a small helper function that, given (K_base, Z_base, eval_mode), returns (K_eval, Z_eval, used_lobatto, fallback_warning).
4. **Argument-parser changes** in the affected diagnostic scripts (which scripts get the new opt-out flag).
5. **Test strategy** — how to verify the fix without re-solving any bundle. Specifically: build two Precompute objects from the same solver disc (one with Lobatto-propagated eval, one with GH-fallback eval), confirm their per-axis nodes/weights match what we expect, run `_diag_euler_errors` on v4_lobatto with both, report the headline EE numbers.
6. **Backwards compatibility** — does this change anyone's saved bundle? No (we're only changing the eval rule construction at diagnostic time). But it does change the EE numbers reported by the diagnostic for any Lobatto bundle. Document that.

Submit the implementation plan first; only proceed with code after the plan is reviewed.

## Why this matters (one-paragraph context for the plan reviewer)

The v4_lobatto bundle (canonical config: K=5 Lobatto Z=7 on the bond/stock return axes and the spr/y_1 state axes) reports residual sim-path EE of mean −2.62, max −0.81 — short of publication grade (mean < −4.5 retirement). Per-cell analysis (`_diag_simpath_worst_cells`) shows the residual concentrates in 167 cells (1% of sim cells) at i₂=6 corners with α_b ≈ +2 — long-bond positions at high y_1-residual states. We suspect the residual is partly artefact: the solver was designed against an explicit ±7σ tail node, but the eval rule has no such node, so the two rules disagree on the optimal α_b at the boundary. The fix above makes them agree. If the residual stays at mean −2.62 after the fix, the policy itself needs more work (smaller Z, smoother bankruptcy clamp, denser state grid). If the residual collapses, the diagnostic was the only thing wrong.

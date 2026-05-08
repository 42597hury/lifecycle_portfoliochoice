# Arbitrage diagnostic post-pivot correctness review

Date: 2026-05-09
Branch: jax-rewrite
Scope: `verify/arbitrage.py` (no other files changed)

## Summary

Reviewed the diagnostic agent's mechanical update in commit `8768652` ("gross-
return cloud now reads bill from `state_grid[:, y_1_idx]`") for correctness
on the new 3-axis real-yields model. **The mechanical update is correct.**
Only docstring cleanup and dead-import removal needed; no math fix and no
redesign required.

## Verification checklist (all PASS)

### 1. State-conditional bill return is read correctly

`y_1_idx = int(model.y_1_index_in_state)` matches the source of truth in
`lifecycle/var.py`:

| System  | state_names               | y_1_index_in_state |
|---------|---------------------------|--------------------|
| Full    | (cape, spr, y_1)          | 2                  |
| 2       | (spr, y_1)                | 1                  |
| 1       | (y_1,)                    | 0                  |

The script reads from `model.y_1_index_in_state` rather than hardcoding `2`,
so it works on all three systems. Bill return is per state cell:
`R_bill[i_s] = exp(state_grid[i_s, y_1_idx])`. Constant across both `k_v` and
`k_r` quadrature axes — there is no shock channel into the bill leg in the
real-yields model. The arbitrage gaps are computed per-state-cell already
(`gap_hull` and `pair_gaps[*]` are arrays of length `N_state`).

### 2. Return-cloud construction matches the solver

`_build_gross_cloud_per_state` reproduces the math in
`lifecycle/solver.py::_build_step_log_returns` line-for-line, vectorised over
`i_s`:

```
log_R_bill[i_s, k_v, k_r] = state_grid[i_s, y_1_idx]                       (deterministic)
log_x_s   [i_s, k_v, k_r] = const_r[xr] + (A_r @ s_t)[xr] + M_v_nodes[k_v, xr] + ret_nodes[k_r, xr]
log_x_b   [i_s, k_v, k_r] = const_r[xb] + (A_r @ s_t)[xb] + M_v_nodes[k_v, xb] + ret_nodes[k_r, xb]
R_stock = R_bill * exp(log_x_s)
R_bond  = R_bill * exp(log_x_b)
```

`const_r = model.Phi_0_ret`, `A_r = model.Phi_21`, `M_v_nodes = v_nodes @
model.M.T` (Cholesky of conditional residual covariance). All three pull
from the new VAR's regression coefficients. No lingering references to `dp`,
`rtb`, or System IV.

### 3. CCV formula consistency

The arbitrage check operates on **gross returns** at quadrature nodes, not
on log-portfolio-returns. It does not need to reconstruct the CCV `r_p`
formula at all — it simply asks whether the discrete (R_bill, R_stock,
R_bond) cloud at each `i_s` admits a separating direction. This is the
correct level of abstraction for a pre-solve diagnostic and is independent
of CRRA / CCV approximations.

### 4. State quadrature consistency

`Phi_0 = model.Phi_0_state` and `Phi_11 = model.Phi_11` carry the new
3-axis VAR's autoregression. `v_nodes`/`v_weights` come from
`pc.v_nodes`/`pc.v_weights` (state-innovation Gauss-Hermite × Cholesky of
`Sigma_state`). Generic over `n_state` via `M_v_nodes` shape; no hardcoded
`n_state == 4`.

### 5. System code references

- `verify/_diag_helpers.build_bundle_var_config` dispatches on `system_code
  ∈ {1, 2, full}` and raises a clear error for legacy nominal bundles
  (System I-IV) with state names from `{dp, cy, rtb}`. Confirmed.
- `_build_pc_from_config` falls back to `build_real_full_var_config_hardcoded()`
  when no bundle metadata is available. Comment notes this is a fallback
  for `.py`-config inputs that don't carry predictability metadata.

### 6. Bundle compatibility / graceful errors

Legacy bundles fail at `build_bundle_var_config` with a multi-paragraph
error message pointing at the pivot doc and offering two recovery paths
(retrospective via pre-pivot revision, forward via re-solve). This is
graceful enough — no further wrapping needed in `arbitrage.py`.

## Changes made (cosmetic only)

- Updated module docstring's "post-rtb-as-state" stale reference and added
  a paragraph documenting the real-yields-pivot semantics of the bill leg.
- Updated the convex-hull memory-budget comment from "9^4" (legacy 4-axis
  nominal) to "7^3" (representative 3-axis dense grid) and recomputed the
  size estimate.
- Removed unused imports (`importlib.util`, `SolverConfig`).

No math or control-flow changes.

## Pause-points and findings

- **Test still meaningful in the new model.** Bill being deterministic
  given the current state simplifies the cloud (it's a pure translation of
  the (X_s, X_b) excess-return cloud at each `i_s`) but the convex-hull
  arbitrage question is unchanged: "is the origin in the hull of the
  (X_s, X_b) discrete distribution?". Axis-aligned pairs involving the bill
  reduce to strict comparisons against a single value, which is the
  strictest possible per-corner statement and remains a useful sanity gate.
- **No downstream bugs found in `_diag_helpers.py`.** The dispatch logic
  matches the post-pivot VAR builders correctly.
- **Three-vs-four-axis generality.** The scan code uses generic
  `pc.state_grid_sizes` and `model.y_1_index_in_state` lookups; nothing
  assumes `n_state == 4`. `_state_index_to_tuple` decodes flat indices for
  any `len(sizes)`. `_format_worst_corner` joins per-axis coordinates
  generically. Works for `n_state ∈ {1, 2, 3}`.

## Verdict

(a) Correctness scan + minimal cleanup. The diagnostic agent's mechanical
update was sufficient on the math; the test's economic structure carries
over to the new model without redesign.

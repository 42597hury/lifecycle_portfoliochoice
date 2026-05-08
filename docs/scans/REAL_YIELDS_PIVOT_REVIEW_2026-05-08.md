# Real-Yields Pivot Correctness Review

**Date:** 2026-05-08
**Branch:** `jax-rewrite`
**Reviewed commits:** `73dee67` (canonical config) ← `ed85cc7` (sim) ← `5fc4182` (solver FOC) ← `0c025eb` (precompute) ← `80ef4a2` (VAR + ablation)
**Scope:** Read-only review of (1) y_1 timing in solver / precompute / simulator and (2) FOC math correctness on the new return structure.

## TL;DR

Both verdicts: **CORRECT.** The pivot is sound on both concerns reviewed; no halt warranted.

1. **y_1 timing — CORRECT.** The bill return is `s_t[y_1_idx]` (current state) at every site (solver `_build_step_log_returns` and `_all_is_log_returns_numpy`; simulator step). State innovation `v_{t+1}` does not enter the bill leg. State evolution is `s_{t+1} = Phi_0_state + Phi_11 @ s_t + v_{t+1}` (solver `_build_step_state_brackets`, sim `s_next`).
2. **FOC math — CORRECT.** The CCV log-portfolio formula and gradient/Hessian terms are unchanged from the validated nominal codebase (`_ccv_log_return_and_grad`); only the upstream `(log_R_bill, log_x_s, log_x_b)` tensors have changed shape/source, and they're sourced consistently. `sigma2_xr / sigma2_xb / sigma_xrxb` come from the new VAR's `Sigma_rr[xr,xb]` block (precompute lines 340–342). `M_v_nodes` is shape `(n_state_quad, n_ret=2)` — collapsed from 4 to 2 return rows along with the rtb-axis removal.

---

## 1. y_1 timing — CORRECT

### Solver

`lifecycle/solver.py:840-842` (per-i_s tensors):
```python
log_R_bill = jnp.broadcast_to(
    state_grid_i[y_1_idx], (n_state_quad, n_ret_quad)
)
```
And the all-i_s NumPy path at `lifecycle/solver.py:1469-1472`:
```python
log_R_bill_is = state_grid_np[:, pcj.y_1_idx]                # (N_state,)
log_R_bill = np.broadcast_to(
    log_R_bill_is[:, None, None], (n_state, n_state_quad, n_ret_quad)
)
```
Both read y_1 from the **current**-period state grid, broadcast across the joint `(k_v, k_r)` quadrature axes — no integration over `v_{t+1}`. The bill leg is correctly deterministic given `s_t`.

### Precompute

`y_1_idx` is sourced from `model.y_1_index_in_state` (`solver.py:1618`), which is set in `build_model` from `var_config["y_1_index_in_state"]`. The Full System VAR (`lifecycle/var.py:441-447`) sets `y_1_index_in_state=2`, matching `state_grid_sizes` axis 2 in `_canonical.py:90`. System 2 sets it to 1 (state names `("spr","y_1")`), System 1 to 0 (just `("y_1",)`). The convention "y_1 at the last state axis" is preserved across all three systems via `predictability_ablation` projection.

### Simulator

`lifecycle/simulation.py:331-337`:
```python
# ----- Real bill rate: deterministic given current state -----
# Real-yields pivot: log_R_bill_{t+1} = s_t[y_1_idx]. No dependence
# on the v^s draw — the bill is risk-free given s_t.
log_R_bill = s_t[y_1_idx]

# ----- Next-period state vector (carries forward to t+1) -----
s_next = Phi_0_state + Phi_11 @ s_t + v_s              # (n_state,)
```
The bill return on savings purchased at t is `s_t[y_1_idx]`, applied to `savings_t` between t and t+1. The next-period state `s_next` is computed independently and carried forward (line 436), so at period t+1 the simulator will read the bill return from what was `s_next` at t, i.e. `y_1_{t+1}` — the rate quoted at t+1 for the t+1→t+2 holding period. This is the canonical timing convention (Catherine 2022; Cocco-Gomes-Maenhout 2005).

### State evolution timing

`solver.py:861`: `s_next = Phi_0_state[None,:] + state_grid_i @ Phi_11.T + v_nodes` — the predictor uses `state_t` and the shock `v_{t+1}` is the integration variable. Likewise `simulation.py:337`. Conditional on `(cape_t, spr_t, y_1_t)`, `y_1_{t+1}` is drawn from the VAR — exactly the predictor block specified in the handoff.

## 2. FOC math — CORRECT

### CCV log-return formula

`solver.py:747-769` (`_ccv_log_return_and_grad`):
```python
r_p = (
    log_R_bill
    + alpha_s * log_x_s
    + alpha_b * log_x_b
    + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
    - 0.5 * (
        alpha_s * alpha_s * sigma2_xr
        + 2.0 * alpha_s * alpha_b * sigma_xrxb
        + alpha_b * alpha_b * sigma2_xb
    )
)
```
This matches the spec exactly. Gradient terms `dr_da_s / dr_da_b` (lines 767-768) and the Jacobian extras (`extra_ss/bb/sb` in retirement_foc / working_foc) propagate the same `sigma2_*` values. The formula is unchanged from the validated pre-pivot code; what changed is upstream:

- `log_R_bill` is now sourced from current state `y_1` rather than from a quadrature node integrating over the bill shock (correct per concern 1).
- `log_x_s` / `log_x_b` are excess-return shocks. From `data/build_var_dataset_real.py:325, 332`: `xr = log(R_stk_{t+1}) - y_1_nom_t` and `xb = r_n_real_{t+1} - y_1_real_t`. Both are excess log returns over the bill known at t. So `R_p = R_bill * exp(α_s · log_x_s + α_b · log_x_b + variance corrections)` is the standard CCV decomposition.

### Predictor sourcing

`solver.py:831-844` (`_build_step_log_returns`):
```python
base_mu_r = const_r + A_r @ state_grid_i              # (n_ret,)
mu_r_per = base_mu_r[None, :] + M_v_nodes              # (n_state_quad, n_ret)
mu_xs = mu_r_per[:, xr_pos]
mu_xb = mu_r_per[:, xb_pos]
```
- `const_r` and `A_r` come from `model.Phi_0_ret` and `model.Phi_21` (precompute lines 319-320). After `partition_var` these are the rows of the full VAR for `ret_idx = (3, 4)` = `(xr, xb)`. So `mu_xs = E[xr_{t+1} | s_t]` and `mu_xb = E[xb_{t+1} | s_t]`. The state vector `s_t = (cape, spr, y_1)` for Full System — exactly the new VAR's predictors. No residual references to `dp` or `rtb` in the predictor.
- `M_v_nodes = v_nodes @ M.T` projects state innovations onto returns (precompute line 321), giving `E[ret | s_t, v_{t+1}]`. Adding `ret_nodes` integrates over the orthogonal residual.

### Variance scalars

`precompute.py:340-342`:
```python
sigma2_xr = float(model.Sigma_rr[xr_pos, xr_pos])
sigma2_xb = float(model.Sigma_rr[xb_pos, xb_pos])
sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])
```
- Indexed by `xr_pos / xb_pos` (looked up by name from `model.ret_names`, line 326-327), so the 2-row collapse from rtb removal does not break the indexing.
- `Sigma_rr` is the unconditional return-block covariance from the new VAR's `Omega_full[ret_idx, ret_idx]` (`var.py:70`). Per the comment at `precompute.py:332-339`, the use of `Sigma_rr` (not `Sigma_r_cond`) is the post-2026-05-06 fix that aligns with CCV w8566 eq. (10) and Markowitz benchmarks.

### `M_v_nodes` size

`M_v_nodes` shape is `(n_state_quad, n_ret) = (n_state_quad, 2)`. `n_ret=2` because the return block now contains only `(xr, xb)` — rtb is gone. The state-quadrature dimension `n_state_quad` is the product of `n_state_quad_nodes` (now 3-tuple, not 4-tuple); precompute gets this generically from `get_state_quadrature`. No 4-axis hardcoding remains.

### Verification of `_validate_state_quadrature` consistency

`precompute.py:526-547` checks `sum_k w_k · mu_r_k == Phi_0_ret + Phi_21 @ s_i` per source state, asserting `< 1e-10`. This sanity check is dimension-generic and would catch any silent shape mismatch. The smoke-test commits report passing, so the consistency holds end-to-end.

---

## What was checked, file-by-file

- `lifecycle/var.py`: real-yields builders (System 1/2/Full); state ordering (cape, spr, y_1) verified; `y_1_index_in_state` correctly assigned per system (0, 1, 2).
- `lifecycle/precompute.py`: `const_r / A_r / M_v_nodes` sourcing; `sigma2_*` from `Sigma_rr` not `Sigma_r_cond`; bequest annuity factor reads y_1 from state with scalar fallback; `_validate_state_quadrature` is dimension-generic.
- `lifecycle/solver.py`: `_ccv_log_return_and_grad` (no change); `_build_step_log_returns` and `_all_is_log_returns_numpy` (read `state_grid_i[y_1_idx]` for bill, predictor uses `state_t`); `terminal_foc_jac_ccv / retirement_foc_jac_ccv / working_foc_jac_ccv` (consume the new tensors via the unchanged signature).
- `lifecycle/simulation.py`: `log_R_bill = s_t[y_1_idx]`; `s_next` from Phi_11 @ s_t + v_s; carry-forward correctly aligned.
- `lifecycle/predictability_ablation.py`: y_1 axis at the last position preserved across all three systems.
- `configs/_canonical.py`: state_grid_sizes axis 2 = y_1, matches `y_1_index_in_state=2` in the Full VAR.

No bugs found. The pivot is internally consistent on both reviewed concerns.

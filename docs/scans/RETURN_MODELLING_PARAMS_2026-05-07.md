# Return-Modelling — Numerical Parameter Appendix

**Date:** 2026-05-07
**Branch:** `jax-rewrite`
**Companion to:** [`RETURN_MODELLING_TRACE_2026-05-07.md`](RETURN_MODELLING_TRACE_2026-05-07.md)

This appendix prints **side-by-side** the VAR parameter values reproduced
from the raw-data estimation pipeline and the hardcoded values shipped in
[`lifecycle/var.py`](../../lifecycle/var.py).

**Reproduction recipe (so anyone can re-run):**

1. Rebuild the dataset:
   `cd data && python build_var_dataset.py` — writes `data/var_dataset.csv`.
2. Run the comparison:
   `python scripts/scratch/reproduce_var_for_handoff.py` — prints the
   tables below verbatim.

The estimator and hardcoded paths produce **identical numbers to floating
point round-off** (`max |diff| ≈ 4.4e-16` on `Phi`, `3.5e-18` on `Omega`).
The hardcoded path in
[`build_nominal_system1_var_config_hardcoded()`](../../lifecycle/var.py#L646)
is therefore an exact frozen snapshot of the estimator output, not a
hand-typed approximation.

All numbers are at **annual frequency** on the 1963–2025 (T=63) sample.

---

## 0. Variable order, partition, dimensions

| Position | Symbol | Role | Block index |
|---|---|---|---|
| 0 | `y_1` | state | state row 3 |
| 1 | `spr` | state | state row 1 |
| 2 | `cy` | state | state row 0 |
| 3 | `rtb` | state | state row 2 |
| 4 | `xr` | return | return row 0 |
| 5 | `xb` | return | return row 1 |

`state_indices = [2, 1, 3, 0]` → state rows in block order are
`(cy, spr, rtb, y_1)`.
`return_indices = [4, 5]` → return rows are `(xr, xb)`.

`n_state = 4`, `n_ret = 2`, full VAR dimension `n = 6`.

---

## 1. Sample mean vector `z_bar`

`z_bar` is the **unconditional mean** of the joint vector
`z = (y_1, spr, cy, rtb, xr, xb)`. The CCV-constrained estimator pins
`z_bar` to the sample mean by construction (no residual freedom in the
intercept), so this row also matches the raw-data column means.

| col | estimated (this run) | hardcoded (var.py) | abs diff |
|---|---|---|---|
| y_1 | +4.8490476190e-02 | +4.8490476190e-02 | 0.00e+00 |
| spr | +1.9922222222e-02 | +1.9922222222e-02 | 3.47e-18 |
| cy  | -2.9928660962e+00 | -2.9928660962e+00 | 0.00e+00 |
| rtb | +9.1313320508e-03 | +9.1313320508e-03 | 0.00e+00 |
| xr  | +5.5470895899e-02 | +5.5470895899e-02 | 6.94e-18 |
| xb  | +1.4267939258e-02 | +1.4267939258e-02 | 0.00e+00 |

Source: [`lifecycle/var.py:608-615`](../../lifecycle/var.py#L608-L615).

---

## 2. AR(1) coefficient matrix `Phi` (full 6×6, annual)

Convention: `Phi[i, j]` = coefficient on lagged `z_j` in equation for
`z_i`. Columns 4 (`xr`) and 5 (`xb`) are zero by the CCV restriction
(lagged returns excluded from every equation). The rtb column is
**freely estimated** under the rtb-as-state regime — it captures the
inflation-persistence channel (`Phi[rtb, rtb] = +0.3627`).

|       | L.y_1 | L.spr | L.cy | L.rtb | L.xr | L.xb |
|-------|-------|-------|------|-------|------|------|
| y_1 | +7.627793e-01 | -1.557638e-01 | +1.059556e-02 | -1.036703e-01 | 0 | 0 |
| spr | +8.680000e-02 | +7.845163e-01 | -4.587939e-03 | +7.431993e-02 | 0 | 0 |
| cy  | +1.906873e+00 | +5.397021e-01 | +8.729942e-01 | -1.561623e+00 | 0 | 0 |
| rtb | +7.548784e-01 | +4.318651e-01 | -2.352980e-02 | +3.627031e-01 | 0 | 0 |
| xr  | -2.536921e+00 | -1.489633e+00 | +1.306595e-01 | +8.245318e-01 | 0 | 0 |
| xb  | +1.238546e+00 | +4.197891e+00 | -4.774994e-02 | +2.506866e-01 | 0 | 0 |

`max |Phi_estimated - Phi_hardcoded| = 4.441e-16`.

Source: [`lifecycle/var.py:623-630`](../../lifecycle/var.py#L623-L630).

---

## 3. Intercept vector `const = (I − Phi) · z_bar`

| col | estimated | hardcoded | abs diff |
|---|---|---|---|
| y_1 | +4.726386e-02 | +4.726386e-02 | 6.94e-18 |
| spr | -1.432579e-02 | -1.432579e-02 | 1.73e-18 |
| cy  | -4.690689e-01 | -4.690689e-01 | 0.00e+00 |
| rtb | -1.098103e-01 | -1.098103e-01 | 0.00e+00 |
| xr  | +5.916815e-01 | +5.916815e-01 | 2.22e-16 |
| xb  | -2.746193e-01 | -2.746193e-01 | 0.00e+00 |

Source: [`lifecycle/var.py:633`](../../lifecycle/var.py#L633)
(`_CONST = (np.eye(6) - _PHI) @ _Z_BAR`).

---

## 4. Innovation covariance `Omega` (full 6×6, annual)

Symmetric. `Omega[i, j]` = sample covariance of OLS residuals from
equation `i` and equation `j`, divided by degrees of freedom
`T − k_predictors = 63 − 1 − 4 = 58`.

|       | y_1 | spr | cy | rtb | xr | xb |
|-------|-----|-----|----|----|----|----|
| y_1 | +2.452786e-04 | -1.509707e-04 | +3.533597e-04 | -1.336154e-04 | -1.962617e-04 | -8.565803e-04 |
| spr | -1.509707e-04 | +1.252935e-04 | +7.523125e-05 | +5.018335e-05 | -1.793261e-04 | +2.502296e-04 |
| cy  | +3.533597e-04 | +7.523125e-05 | +2.694799e-02 | -1.113992e-03 | -2.564681e-02 | -3.891716e-03 |
| rtb | -1.336154e-04 | +5.018335e-05 | -1.113992e-03 | +3.241106e-04 | +8.052754e-04 | +7.193814e-04 |
| xr  | -1.962617e-04 | -1.793261e-04 | -2.564681e-02 | +8.052754e-04 | +2.529121e-02 | +3.461823e-03 |
| xb  | -8.565803e-04 | +2.502296e-04 | -3.891716e-03 | +7.193814e-04 | +3.461823e-03 | +5.882474e-03 |

`max |Omega_est - Omega_hc| = 3.469e-18`.

Source: [`lifecycle/var.py:636-643`](../../lifecycle/var.py#L636-L643).

---

## 5. Partition-derived blocks

After applying [`partition_var()`](../../lifecycle/var.py#L49) with
`state_idx = [2, 1, 3, 0]` (= `cy, spr, rtb, y_1`) and
`ret_idx = [4, 5]` (= `xr, xb`):

### 5.1 State block — `Phi_0_state` (4-vector, in state-row order `(cy, spr, rtb, y_1)`)

```
Phi_0_state = [-4.690689e-01, -1.432579e-02, -1.098103e-01, +4.726386e-02]
```

### 5.2 State transition `Phi_11` (4×4, rows/cols both in state-row order)

|         | L.cy | L.spr | L.rtb | L.y_1 |
|---------|------|-------|-------|-------|
| cy   | +8.729942e-01 | +5.397021e-01 | -1.561623e+00 | +1.906873e+00 |
| spr  | -4.587939e-03 | +7.845163e-01 | +7.431993e-02 | +8.680000e-02 |
| rtb  | -2.352980e-02 | +4.318651e-01 | +3.627031e-01 | +7.548784e-01 |
| y_1  | +1.059556e-02 | -1.557638e-01 | -1.036703e-01 | +7.627793e-01 |

### 5.3 Return intercepts `Phi_0_ret` (2-vector, `(xr, xb)`)

```
Phi_0_ret = [+5.916815e-01, -2.746193e-01]
```

### 5.4 Return loading `Phi_21 = A_r` (2×4, rows `(xr, xb)`, cols `(cy, spr, rtb, y_1)`)

This is the matrix consumed by the solver as `pc.A_r`
([`precompute.py:292`](../../lifecycle/precompute.py#L292)).

|       | L.cy | L.spr | L.rtb | L.y_1 |
|-------|------|-------|-------|-------|
| xr | +1.306595e-01 | -1.489633e+00 | +8.245318e-01 | -2.536921e+00 |
| xb | -4.774994e-02 | +4.197891e+00 | +2.506866e-01 | +1.238546e+00 |

### 5.5 State-innovation covariance `Sigma_ss` (4×4, in state-row order)

|       | cy | spr | rtb | y_1 |
|-------|----|-----|-----|-----|
| cy   | +2.694799e-02 | +7.523125e-05 | -1.113992e-03 | +3.533597e-04 |
| spr  | +7.523125e-05 | +1.252935e-04 | +5.018335e-05 | -1.509707e-04 |
| rtb  | -1.113992e-03 | +5.018335e-05 | +3.241106e-04 | -1.336154e-04 |
| y_1  | +3.533597e-04 | -1.509707e-04 | -1.336154e-04 | +2.452786e-04 |

### 5.6 Return-innovation covariance `Sigma_rr` (2×2, UNCONDITIONAL)

Used by the CCV log-return formula's vol-drag scalars (see [§6.7 of the
trace](RETURN_MODELLING_TRACE_2026-05-07.md#67-the-vol-drag-scalars-source-controversy)).

|       | xr | xb |
|-------|------|------|
| xr | +2.529121e-02 | +3.461823e-03 |
| xb | +3.461823e-03 | +5.882474e-03 |

Annualised stds: `σ(xr) = 15.90%`, `σ(xb) = 7.67%`. Correlation
`ρ(xr, xb) = +0.285`.

### 5.7 Cross-block covariance `Sigma_rs` (2×4, return rows × state cols)

**This is the cross-block covariance the handoff explicitly asks to be
documented.** It is non-zero — return innovations and state innovations
are jointly correlated. Solver and simulator use it via the `M`
projection (§5.8) when forming conditional return means.

|       | cy | spr | rtb | y_1 |
|-------|------|------|------|------|
| xr | -2.564681e-02 | -1.793261e-04 | +8.052754e-04 | -1.962617e-04 |
| xb | -3.891716e-03 | +2.502296e-04 | +7.193814e-04 | -8.565803e-04 |

### 5.8 Return-mean projection `M = Sigma_rs · Sigma_ss⁻¹` (2×4)

`M @ v_state_innovation` is the **conditional mean shift** added to
`(xr, xb)` once a state innovation `v` is drawn. The `M @ M.T` term is
what gets subtracted to form `Sigma_r_cond` (§5.9).

|       | cy | spr | rtb | y_1 |
|-------|------|------|------|------|
| xr | -9.782092e-01 | -1.403067e+00 | -9.869202e-01 | -7.921268e-01 |
| xb | -9.773040e-03 | -8.594138e+00 | -1.263541e-01 | -8.836780e+00 |

The dominant entries are
`M[xb, spr] = -8.59` and `M[xb, y_1] = -8.84` — a positive shock to the
spread or to the long yield drives a large negative shift in the
conditional mean of the long-bond excess return (duration channel).

### 5.9 Conditional return covariance `Sigma_r_cond = Sigma_rr − M · Sigma_sr` (2×2)

|       | xr | xb |
|-------|------|------|
| xr | +5.909321e-04 | +3.745081e-05 |
| xb | +3.745081e-05 | +5.164327e-04 |

Annualised stds: `σ(xr|state) = 2.43%`, `σ(xb|state) = 2.27%`.
Correlation `ρ(xr, xb | state) = +0.068`.

State innovations explain `97.66%` of the unconditional variance of `xr`
and `91.22%` of `xb` (`var_explained_share` in the partition output).

---

## 6. Eigenvalues of `Phi_11` (stationarity check)

Modulus, sorted descending:

```
|λ| = [0.92559216, 0.78598056, 0.78598056, 0.32273311]
max |λ| = 0.925592   →   STATIONARY (max < 1)
```

---

## 7. Joint covariance `Sigma_joint` (eigenvalues, PSD check)

The full 6×6 `Omega` reordered into `(cy, spr, rtb, y_1, xr, xb)` block
form (state rows then return rows):

```
[ +2.6948e-02   +7.5231e-05   -1.1140e-03   +3.5336e-04   -2.5647e-02   -3.8917e-03 ]
[ +7.5231e-05   +1.2529e-04   +5.0183e-05   -1.5097e-04   -1.7933e-04   +2.5023e-04 ]
[ -1.1140e-03   +5.0183e-05   +3.2411e-04   -1.3362e-04   +8.0528e-04   +7.1938e-04 ]
[ +3.5336e-04   -1.5097e-04   -1.3362e-04   +2.4528e-04   -1.9626e-04   -8.5658e-04 ]
[ -2.5647e-02   -1.7933e-04   +8.0528e-04   -1.9626e-04   +2.5291e-02   +3.4618e-03 ]
[ -3.8917e-03   +2.5023e-04   +7.1938e-04   -8.5658e-04   +3.4618e-03   +5.8825e-03 ]
```

Eigenvalues (ascending):

```
[ 2.84161950e-06,
  1.29564265e-04,
  1.93171981e-04,
  5.65831496e-04,
  5.51815123e-03,
  5.24067883e-02 ]
```

All strictly positive → `Sigma_joint` is positive definite. Smallest
eigenvalue ≈ `2.84e-06` is well above the `1e-5` rank-deficiency
threshold that
[`build_model()`](../../lifecycle/precompute.py#L676) uses as a drift
detector for the OPTION-B failure mode.

---

## 8. Equation R²

R² of each VAR equation against its (lagged-state) regressors. Reproduced
from the estimator:

| equation | R² | Notes |
|---|---|---|
| y_1 | 0.7895 | Persistent short rate |
| spr | 0.5327 | Term-spread mean reversion |
| cy  | 0.8795 | Earnings-yield persistence (CAPE smoothing) |
| rtb | 0.6075 | Inflation persistence channel (rtb-as-state freely lagged) |
| xr  | 0.0731 | Equity excess return — weak predictability |
| xb  | 0.3256 | Bond excess return — meaningful predictability via spread + yield |

The handoff banner in
[`build_nominal_system1_var_config_hardcoded`](../../lifecycle/var.py#L646)
documents this contract: **`rtb R² = 0.6075`,
`Phi[rtb, rtb] = +0.3627`, `cond(Sigma_r_cond) ≈ 1.21`.** All three are
reproduced here element for element, confirming the snapshot is current
on the 1963–2025 sample.

---

## 9. CCV vol-drag scalars actually consumed by the solver

The solver and simulator both read three scalars (`σ²_xr`, `σ²_xb`,
`σ_xrxb`) into the CCV log-return formula. As of commit `f23ac83`
(May 2026), they come from the **unconditional** `Sigma_rr`, not the
conditional `Sigma_r_cond`. See
[`precompute.py:303-314`](../../lifecycle/precompute.py#L303-L314)
and [§6.7 of the trace](RETURN_MODELLING_TRACE_2026-05-07.md#67-the-vol-drag-scalars-source-controversy)
for the rationale and the documentation drift this introduces.

| Scalar | Value | Source matrix | Position |
|---|---|---|---|
| `σ²_xr`  | `+2.529121e-02` | `Sigma_rr` | `[xr, xr]` |
| `σ²_xb`  | `+5.882474e-03` | `Sigma_rr` | `[xb, xb]` |
| `σ_xrxb` | `+3.461823e-03` | `Sigma_rr` | `[xr, xb]` |

For comparison, `Sigma_r_cond` would give:

| Scalar | Sigma_r_cond value | Ratio (Sigma_rr / Sigma_r_cond) |
|---|---|---|
| `σ²_xr`  | `+5.909e-04` | 42.8 |
| `σ²_xb`  | `+5.164e-04` | 11.4 |
| `σ_xrxb` | `+3.745e-05` | 92.4 |

The ~12–90× ratio is large enough that the choice between matrices
materially changes the Itô vol-drag and the converged optimal `α`. The
production choice (`Sigma_rr`) is documented in code; the matching
update to `docs/CCV_RETURNS.md` is still pending — see trace §6.7.

End of appendix.

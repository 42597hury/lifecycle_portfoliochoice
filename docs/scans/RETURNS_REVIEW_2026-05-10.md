# Returns Specification & Pipeline Review

**Date:** 2026-05-10
**Branch:** `jax-rewrite`
**Scope:** Read-only verification of the asset-returns pipeline under the
real-yields pivot. State vector `(cape, spr, y_1)`; returns `(xr, xb)`; bill
deterministic given current state via `R_bill_{t+1} = exp(y_1_t)`.
**Method:** Static read of source against existing 2026-05-09 audits, plus
small NumPy probes for partition algebra, stationarity, and CSV-vs-hardcoded
parity. No solver runs, no source modifications, no commits.

---

## §1. Scope

This document re-verifies the seven domains called out in the briefing
against the current source on `jax-rewrite`:

1. The VAR specification and partition arithmetic in
   [lifecycle/var.py](lifecycle/var.py).
2. The state-vector indexing convention in
   [lifecycle/model.py](lifecycle/model.py) and its consumers.
3. The CCV log-wealth dynamics in
   [lifecycle/solver.py](lifecycle/solver.py).
4. The (k_v, k_r) joint quadrature for excess returns and the
   `n_state_quad_nodes=(3,3,5)` y_1-axis K-bump.
5. The AR(1)-matched 10-year RLONG VAR baseline introduced in commit
   `891bb7c` ([data/build_var_dataset_ar1_10y.py](data/build_var_dataset_ar1_10y.py),
   [data/var_dataset.csv](data/var_dataset.csv),
   [data/var_specification.md](data/var_specification.md)).
6. The annuity factor in [lifecycle/model.py:annuity_factor](lifecycle/model.py#L319)
   under the real-yields pivot.
7. The interaction of [lifecycle/discretization.py](lifecycle/discretization.py)
   with the VAR partition (Cholesky vs naive vs lyapunov-axis modes).
8. The per-i_s log-return tensors `_all_is_log_returns_numpy` and
   `_precompute_per_is_tensors` in
   [lifecycle/solver.py](lifecycle/solver.py#L1629).

The 2026-05-09 audits read for verification (not duplication):
- [CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md](docs/scans/CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md)
- [RETURN_TREATMENT_REVIEW_A_2026-05-09.md](docs/scans/RETURN_TREATMENT_REVIEW_A_2026-05-09.md)
- [RETURN_TREATMENT_REVIEW_B_2026-05-09.md](docs/scans/RETURN_TREATMENT_REVIEW_B_2026-05-09.md)
- [STOCK_RETURN_TREATMENT_2026-05-09.md](docs/scans/STOCK_RETURN_TREATMENT_2026-05-09.md)
- [EICHENGREEN_REAL_BOND_YIELDS_REVIEW_2026-05-09.md](docs/scans/EICHENGREEN_REAL_BOND_YIELDS_REVIEW_2026-05-09.md)
- [CP_INFLATION_PIPELINE_REVIEW_2026-05-09.md](docs/scans/CP_INFLATION_PIPELINE_REVIEW_2026-05-09.md)
- [DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md](docs/scans/DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md)
- [ARBITRAGE_PIVOT_REVIEW_2026-05-09.md](docs/scans/ARBITRAGE_PIVOT_REVIEW_2026-05-09.md)
- [BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md](docs/scans/BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md)

The single material change since those audits is the AR(1)-matched 10-year
RLONG dataset (commit `891bb7c`) which replaced the prior CP-Shiller v=0.992
20-year AAA dataset. Several of the older numbers (R²(xb)=0.27, std(xb)=8.93%,
20y AAA bond proxy) are now stale and are flagged where they appear.

---

## §2. VAR partition algebra

### 2.1 Block structure

[lifecycle/var.py:partition_var](lifecycle/var.py#L49) takes the full VAR
`z_{t+1} = (I-Phi) z_bar + Phi z_t + eps`, `Cov(eps) = Omega`, splits the
variable list by `state_idx` and `ret_idx`, and returns the partitioned blocks:

- `Phi_11 = Phi[state_idx, state_idx]` ([var.py:64](lifecycle/var.py#L64))
- `Phi_21 = Phi[ret_idx, state_idx]` ([var.py:65](lifecycle/var.py#L65))
- `Phi_12, Phi_22` are extracted but used only for the restriction-norm
  diagnostic ([var.py:86-87](lifecycle/var.py#L86)).
- `Sigma_ss = Omega[state_idx, state_idx]`,
  `Sigma_rr = Omega[ret_idx, ret_idx]`,
  `Sigma_rs = Omega[ret_idx, state_idx]`,
  `Sigma_sr = Omega[state_idx, ret_idx]` ([var.py:69-72](lifecycle/var.py#L69)).
- `M = Sigma_rs @ inv(Sigma_ss)` ([var.py:74](lifecycle/var.py#L74))
- `Sigma_r_cond = Sigma_rr - M @ Sigma_sr` ([var.py:75](lifecycle/var.py#L75)).
- `Phi_0_full = (I - Phi) z_bar` ([var.py:82](lifecycle/var.py#L82)), then
  partitioned to `Phi_0_state`, `Phi_0_ret`.

### 2.2 Schur complement re-derivation

Under joint Gaussianity of innovations, the conditional return innovation
given the state innovation is

```
eps_r | eps_s = N( M eps_s, Sigma_r_cond ),    M = Sigma_rs Sigma_ss^-1.
```

This is the textbook block-conditional Gaussian: the linear projection of
the return innovation onto the state innovation has slope `M` and residual
covariance `Sigma_r_cond` (Schur complement). The decomposition

```
Sigma_rr = M Sigma_ss M' + Sigma_r_cond                           (2.1)
```

is the variance-of-innovation decomposition into "state-driven" and
"state-orthogonal" parts. In words, the state-driven part is the variance of
`M eps_s = M Sigma_ss M'`, and the state-orthogonal part is the residual
`Sigma_r_cond`.

### 2.3 Numerical verification on the hardcoded fallback

Probe (cpu, fp64) on
[var.py:_PHI / _Z_BAR / _OMEGA](lifecycle/var.py#L529) (`build_real_full_var_config_hardcoded()`,
state_idx=[0,1,2], ret_idx=[3,4]):

```
||M - Sigma_rs @ inv(Sigma_ss)||_inf       = 0.0    (machine zero)
||Sigma_rr - (M Sigma_ss M' + Sigma_r_cond)|| = 6.94e-18
eigvals(Sigma_r_cond)                       = [7.24e-5, 2.33e-3]   (PD)
max |eig(Phi)|                              = 0.92957
max |eig(Phi_11)|                           = 0.92957
```

The Schur identity (2.1) holds at machine precision. `Sigma_r_cond` is
strictly positive definite, so the Cholesky in
[discretization.get_return_quadrature](lifecycle/discretization.py#L654)
is well-defined. Phi and Phi_11 share the same dominant eigenvalue
0.92957 because the restricted estimator zeroes the lagged-return columns
of `Phi` so the return rows do not contribute to the spectral radius
beyond Phi_11's eigenvalues plus two zeros from Phi_22 = 0.

### 2.4 M-matrix structure (post-AR(1)-baseline)

```
M = [[ -0.9379,  +2.0349,  +0.6428],     <- xr
     [ +0.0071,  -6.8306,  -6.9617]]     <- xb
                  cape       spr       y_1
```

The xb row dominance on `(spr, y_1)` is the structural reason the y_1 axis
gets the K-bump in `n_state_quad_nodes` (see §5). Compare the older v=0.992
20y AAA values from
[BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md](docs/scans/BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md):
`M[xb, :] = (+0.022, -13.477, -13.392)`. The new AR(1)/10y numbers are
roughly half the magnitude — duration is shorter (D~10 instead of D~17) and
the bond leg is now the Shiller RLONG real return, not Moody's AAA. The
qualitative message — `M[xb, spr]` and `M[xb, y_1]` are the dominant
entries — is unchanged.

`var_explained_share` = `[0.932, 0.980]` for `(xr, xb)`: 93.2% of `xr`
innovation variance and 98.0% of `xb` innovation variance is mechanically
tied to contemporaneous state innovations (cape, spr, y_1), leaving small
residual shares in `Sigma_r_cond`. This is consistent with §1 of
[CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md](docs/scans/CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md):
the CCV variance correction must use the **full** `Sigma_rr`, not
`Sigma_r_cond`, because the inner solver quadrature integrates over both
the state innovation `v_state` and the residual `eps_ret`.

### 2.5 Restriction enforcement

The restricted, mean-pinned estimator
[var.py:estimate_var1_from_csv](lifecycle/var.py#L191) demeans, regresses
without intercept on lagged state columns only, and back-solves
`const = (I - Phi) z_bar`. Lagged xr/xb columns of Phi are exact zeros by
construction. The hardcoded `_PHI` ([var.py:542-548](lifecycle/var.py#L542))
has these columns set to `0.000000000000000000e+00`, and the partition
diagnostic ([var.py:86-87](lifecycle/var.py#L86)) confirms
`||Phi_12|| = ||Phi_22|| = 0` exactly.

CSV-vs-hardcoded probe:

```
||Phi_csv  - Phi_hardcoded||  = 0.0
||Omega_csv - Omega_hardcoded|| = 0.0
```

So the fallback in `build_real_full_var_config_hardcoded` is byte-equivalent
to the live re-estimation against `data/var_dataset.csv`.

---

## §3. State-vector indexing

### 3.1 Index fields on the model

[lifecycle/model.py:LifecyclePortfolioModel](lifecycle/model.py#L24) carries
four index fields:

- `y_1_index_in_state` — must be set in the real-yields pivot (the bill
  anchor). [model.py:77](lifecycle/model.py#L77).
- `spr_index_in_state` — set to grid index for full / system-2; `None`
  for system-1 with `spr_scalar_fallback` instead. [model.py:78](lifecycle/model.py#L78).
- `rtb_index_in_state` — `None` in the real-yields pivot (no separate rtb
  axis exists; the bill is pinned to y_1). [model.py:79](lifecycle/model.py#L79).
- `y_1_scalar_fallback` / `spr_scalar_fallback` — used only when the
  corresponding index is `None`. [model.py:80-81](lifecycle/model.py#L80).

### 3.2 Index assignment per system

From [var.py:build_real_full_var_config](lifecycle/var.py#L409),
[build_real_system2_var_config](lifecycle/var.py#L452),
[build_real_system1_var_config](lifecycle/var.py#L482):

| System  | state_names         | y_1_index_in_state | spr_index_in_state | rtb_index_in_state |
|---------|---------------------|--------------------|--------------------|--------------------|
| Full    | (cape, spr, y_1)    | 2                  | 1                  | None               |
| 2       | (spr, y_1)          | 1                  | 0                  | None               |
| 1       | (y_1,)              | 0                  | None (fallback=mean) | None             |

The hardcoded fallback ([var.py:578-580](lifecycle/var.py#L578)) sets
`y_1_index_in_state=2`, `spr_index_in_state=1`, `rtb_index_in_state=None`,
matching the Full system live-builder values.

### 3.3 Validation gates in `build_model`

[precompute.py:842-865](lifecycle/precompute.py#L842) enforces:

- `rtb_index_in_state is None` ⇒ `y_1_index_in_state` must be set
  (real-yields setup needs the bill on the grid). Raises if both are
  None.
- `y_1_index_in_state` and `spr_index_in_state`, when both set, must
  be distinct and both within range.
- `rtb_index_in_state`, when set, must be distinct from y_1 and spr.
- `y_1_scalar_fallback` is required when `y_1_index_in_state is None`,
  and similarly for spr.

[solver.py:1757-1761](lifecycle/solver.py#L1757) (`_pc_to_jnp`) re-checks
`y_1_index_in_state is not None` and raises `ValueError` if not. Belt and
braces.

### 3.4 Real-bill anchor in the solver

The contract `log_R_bill_{t+1} = state_t[y_1_idx]` is implemented in two
places, both broadcasting a current-state scalar across both quadrature
axes:

- Per-i_s JAX path
  [solver.py:_build_step_log_returns:914](lifecycle/solver.py#L914):
  `log_R_bill = jnp.broadcast_to(state_grid_i[y_1_idx], (n_state_quad, n_ret_quad))`.
- Host-side NumPy path
  [solver.py:_all_is_log_returns_numpy:1653-1656](lifecycle/solver.py#L1653):
  `log_R_bill_is = state_grid_np[:, pcj.y_1_idx]; log_R_bill = np.broadcast_to(log_R_bill_is[:, None, None], (n_state, n_state_quad, n_ret_quad))`.

Both forms read **state_t** (not s_next) — the bill is in the time-`t`
information set and carries no innovation. The corresponding code in the
arbitrage diagnostic ([ARBITRAGE_PIVOT_REVIEW_2026-05-09.md §1](docs/scans/ARBITRAGE_PIVOT_REVIEW_2026-05-09.md))
does the same. The solver and simulator paths are consistent.

---

## §4. CCV log-wealth dynamics

### 4.1 The formula in code

[solver.py:_ccv_log_return_and_grad:821-843](lifecycle/solver.py#L821):

```
r_p = log_R_bill
      + a_s * log_x_s
      + a_b * log_x_b
      + 0.5 * (a_s * sigma2_xr + a_b * sigma2_xb)
      - 0.5 * (a_s^2 * sigma2_xr
               + 2 * a_s * a_b * sigma_xrxb
               + a_b^2 * sigma2_xb)
R_p = exp(r_p)
dr/da_s = log_x_s + sigma2_xr * (0.5 - a_s) - a_b * sigma_xrxb
dr/da_b = log_x_b + sigma2_xb * (0.5 - a_b) - a_s * sigma_xrxb
```

This matches the Campbell-Viceira w8566 eq. (10) lognormal-corrected
linearised return with `xs = (xr, xb)`, where the convexity corrections
are `+0.5 alpha' diag(Sigma_xx)` (Jensen restoration, leg-wise) and
`-0.5 alpha' Sigma_xx alpha` (lognormal log-of-mean correction for the
portfolio). The 2026-05-09 derivation in
[CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md §1](docs/scans/CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md)
confirms term-by-term.

### 4.2 Sign of the convexity terms

- `+0.5 a_s sigma2_xr` and `+0.5 a_b sigma2_xb` are **positive Jensen
  contributions** that restore `E[exp(x_X)] = exp(mu_X + 0.5 Sigma_xx,XX)`.
- `-0.5 a_s^2 sigma2_xr - a_s a_b sigma_xrxb - 0.5 a_b^2 sigma2_xb` is the
  **negative quadratic vol-drag** that converts the lognormal portfolio's
  `E[exp r_p]` into `exp(E[r_p] + 0.5 Var[r_p])` correctly.
- Cross term `-a_s a_b sigma_xrxb`: with `sigma_xrxb` empirically
  positive but small (~+8.06e-4) and `a_b` typically positive in this
  calibration, this term mildly damps the marginal demand for stocks when
  bonds are also held long. The sign is correct.

### 4.3 Gradient correctness

`dr/da_s = log_x_s + sigma2_xr * (0.5 - a_s) - a_b * sigma_xrxb` is the
analytic partial of `r_p` with respect to `a_s`:

```
d/da_s [ a_s * log_x_s ]                                  = log_x_s
d/da_s [ +0.5 a_s sigma2_xr ]                             = +0.5 sigma2_xr
d/da_s [ -0.5 a_s^2 sigma2_xr ]                           = -a_s sigma2_xr
d/da_s [ -a_s a_b sigma_xrxb ]                            = -a_b sigma_xrxb
sum    : log_x_s + sigma2_xr (0.5 - a_s) - a_b sigma_xrxb
```

Symmetric for `dr/da_b`. ✓

### 4.4 Variance-correction scalars source

[precompute.py:347-358](lifecycle/precompute.py#L347):

```
sigma2_xr  = float(model.Sigma_rr[xr_pos, xr_pos])
sigma2_xb  = float(model.Sigma_rr[xb_pos, xb_pos])
sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])
```

These pull from the **full** `Sigma_rr`, not `Sigma_r_cond`. The 2026-05-06
patch comment is preserved at the call site and is correct: because the
inner solver quadrature integrates over both `v_state` (with weight
`v_weights`) and `eps_ret` (with weight `ret_weights`), the variance term
in (1.5) of the CCV derivation is the unconditional return-block
variance. Substituting `Sigma_r_cond` would double-deduct the
state-projection variance.

For the AR(1)-matched 10y baseline (current canonical):

```
sigma2_xr  = +3.41e-2   (sigma(xr) = 18.5%)
sigma2_xb  = +3.62e-3   (sigma(xb) =  6.0%)
sigma_xrxb = +8.06e-4
```

Note: the bond std fell from the 8.93% reported in
[BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md §3](docs/scans/BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md)
to 6.0% with the move from 20y AAA / v=0.992 to 10y RLONG / AR(1). The
inverse `1/Var(xb)` leverage scale roughly **2.2× larger** under the new
baseline — see §9 finding F-2.

### 4.5 ret_pos lookup robustness

[precompute.py:342-343](lifecycle/precompute.py#L342) and
[solver.py:1770-1771](lifecycle/solver.py#L1770) both use
`ret_names.index("xr")` / `ret_names.index("xb")` rather than hardcoded
indices. Robust to any future re-permutation of the return block.

### 4.6 FOC kernel re-use

`_ccv_log_return_and_grad` is the single source of truth for the log
return; it is called from
- `terminal_foc_jac_ccv` ([solver.py:858](lifecycle/solver.py#L858)),
- `retirement_foc_jac_ccv` ([solver.py:1053](lifecycle/solver.py#L1053)),
- `working_foc_jac_ccv` ([solver.py:1164](lifecycle/solver.py#L1164)),
- the diagnostics path ([diagnostics.py:1258](lifecycle/diagnostics.py#L1258)),
- the simulator (per `RETURN_TREATMENT_REVIEW_A` §2 verification).

No alternative formulation is implemented.

---

## §5. Excess-return quadrature, weight construction, and y_1 K-bump

### 5.1 Building log_x_s and log_x_b at each (i_s, k_v, k_r)

[solver.py:_build_step_log_returns:890-919](lifecycle/solver.py#L890):

```
base_mu_r = const_r + A_r @ state_grid_i              # (n_ret,)
mu_r_per  = base_mu_r[None, :] + M_v_nodes            # (n_state_quad, n_ret)
mu_xs     = mu_r_per[:, xr_pos]
mu_xb     = mu_r_per[:, xb_pos]
res_xs    = ret_nodes[:, xr_pos]
res_xb    = ret_nodes[:, xb_pos]
log_x_s   = mu_xs[:, None] + res_xs[None, :]          # (n_state_quad, n_ret_quad)
log_x_b   = mu_xb[:, None] + res_xb[None, :]
```

Where ([precompute.py:335-337](lifecycle/precompute.py#L335)):

```
const_r   = Phi_0_ret  - M @ Phi_0_state    # (n_ret,)
A_r       = Phi_21    - M @ Phi_11          # (n_ret, n_state)
M_v_nodes = v_nodes @ M.T                   # (n_state_quad, n_ret)
```

So `mu_r_per[k_v, :] = const_r + A_r s_i + M v_nodes[k_v, :]
                     = E[r_{t+1} | s_t = s_i, v_state_{t+1} = v_k]`,
and adding the residual `eps_ret = ret_nodes[k_r, :]` (already
Cholesky-mapped from i.i.d. standard normal) builds the joint quadrature
node `r_{t+1}(i_s, k_v, k_r)`. The state quadrature consistency check
[precompute.py:_validate_state_quadrature:579-600](lifecycle/precompute.py#L579)
asserts `sum_k v_weights[k] mu_r[i_s, k, :] = Phi_0_ret + Phi_21 s_i` to
1e-10 — i.e., averaging out v_state over the state quadrature recovers the
unconditional return mean conditional on `s_i`.

### 5.2 Joint weight tensor

[solver.py:1767](lifecycle/solver.py#L1767):

```
weight_kv_kr = jnp.asarray(pc.v_weights)[:, None] * jnp.asarray(pc.ret_weights)[None, :]
```

This is the tensor product of the two independent quadrature rules. The
factorisation is correct because `v_state` and `eps_ret` are independent
(by construction of the Schur partition: `Cov(v_state, eps_ret) = 0`
after the Cholesky orthogonalisation at the discretisation step,
[discretization.py:get_return_quadrature:639-657](lifecycle/discretization.py#L639)).

### 5.3 The K-bump on y_1

Canonical
[configs/_canonical.py:122](configs/_canonical.py#L122):
`n_state_quad_nodes=(3, 3, 5)`. Axes 0-1 (cape, spr) get standard 3-node
Gauss-Hermite; axis 2 (y_1) gets 5 nodes.

Why y_1 specifically: because of the Cholesky transform `v = L u` with `L`
lower-triangular (
[discretization.py:get_state_quadrature:746-749](lifecycle/discretization.py#L746)),
axis 0 in u-coords is the pure-cape direction, axis 1 is mostly-spr
after orthogonalising cape away, and axis 2 is the **residual y_1
direction** after both cape and spr are orthogonalised away. Refining
this axis is targeted at the y_1 channel specifically.

The xb innovation loads heavily on the y_1 channel. From §2.4,
`M[xb, y_1] = -6.96` with `Var(y_1)_innov = Sigma_ss[2,2] = 3.12e-4`,
contributing `M[xb,y_1]^2 * Sigma_ss[y_1,y_1] = 0.0151` to `Var(xb)` — the
single largest contribution after `M[xb, spr]^2 * Sigma_ss[spr, spr]` (which
is similar in scale at ~0.0101 after the cross-term). At γ=5 with strong
predictability the bond-leg integration error is the dominant accuracy
constraint, and bumping the y_1 axis from 3 to 5 nodes is the cheapest
way to reduce that error (the joint quadrature size grows from 3*3*3=27 to
3*3*5=45 nodes, ~67% cost increase, vs ~5x for a uniform K=5 bump).

### 5.4 Stock-leg and residual-leg refinement

[configs/_canonical.py:120](configs/_canonical.py#L120):
`n_ret_nodes_1d=(4, 4)` — 4 nodes per return-residual axis, totaling 16
joint residual nodes. The Cholesky on `Sigma_r_cond` makes axis 0 the
pure xr residual (after orthogonalisation, ordered as (xr, xb) per
`ret_names`), and axis 1 the pure xb residual. The joint state-quadrature
size therefore is `3 * 3 * 5 * 4 * 4 = 720` per (i_s) cell.

### 5.5 Verification of arithmetic

[ARBITRAGE_PIVOT_REVIEW_2026-05-09.md §2](docs/scans/ARBITRAGE_PIVOT_REVIEW_2026-05-09.md)
derived the same per-state cloud and confirmed it line-for-line against
`_build_step_log_returns`. The arithmetic in the production solver and
the diagnostic agree.

### 5.6 ret_lobatto / state_lobatto

Canonical sets `state_lobatto_Z=None` and `ret_lobatto_Z=None`
([_canonical.py:121, 123](configs/_canonical.py#L121)). Standard
Gauss-Hermite on every axis. The Lobatto path
([discretization.py:_build_axis_grid:546-552](lifecycle/discretization.py#L546))
exists but is not on the production hot path. The branch is gated to
`K in {3,5,7}` and `K odd` to keep the closed-form prescribed-tail rule
valid; standard GH (the production setting) is unrestricted in K.

---

## §6. AR(1) inflation forecast and the new VAR baseline (commit 891bb7c)

### 6.1 What changed

The pre-`891bb7c` baseline ([RETURN_TREATMENT_REVIEW_A_2026-05-09.md §2](docs/scans/RETURN_TREATMENT_REVIEW_A_2026-05-09.md)
and [BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md §2](docs/scans/BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md))
used:

- CP-Shiller EWMA deflator `tau_t = v tau_{t-1} + (1-v) pi_t`, `v = 0.995`
  (then `v=0.992` for one variant).
- Bond leg: 20-year Moody's AAA Long-Term Corporate, real-deflated.
- Stats: `mean(y_1)=+1.531pp`, `mean(spr)=+1.255pp`, `std(xb)=8.93%`,
  `R²(xb)=0.27`, `M[xb, spr]=-13.5`, `M[xb, y_1]=-13.4`.

The post-`891bb7c` baseline (active production) uses
[data/build_var_dataset_ar1_10y.py](data/build_var_dataset_ar1_10y.py):

- AR(1) on December-over-December Shiller log CPI inflation,
  full-sample static fit:
  `pi_{t+1} = +1.289pp + 0.3884 pi_t`, long-run mean `mu = +2.107pp`
  ([build_var_dataset_ar1_10y.py:90-106](data/build_var_dataset_ar1_10y.py#L90)).
- January year-`t` yield uses December year-`t-1` inflation as the AR(1)
  state ([build_var_dataset_ar1_10y.py:124-130](data/build_var_dataset_ar1_10y.py#L124)).
- Real one-year yield: `y_1 = y_1_nom - E[pi_{t+1}|pi_info]`.
- Real 10-year yield: `y_n_real = y_n_nom - E_avg_10[pi]`, where the
  expectation is the AR(1) average over h=1..10
  ([build_var_dataset_ar1_10y.py:109-121](data/build_var_dataset_ar1_10y.py#L109)).
- Bond leg: Shiller RLONG, real-deflated, CLM constant-duration with
  D = (1 - g^{-10}) / (1 - g^{-1}) computed from the realised real
  long yield, n=10 ([build_var_dataset_ar1_10y.py:137-142](data/build_var_dataset_ar1_10y.py#L137)).
- xr stays nominal-data-derived per CCV §2.1.5 invariance:
  `xr_t = log((P_t + D_t)/P_{t-1}) - y_1_nom_{t-1}`
  ([build_var_dataset_ar1_10y.py:166-167](data/build_var_dataset_ar1_10y.py#L166)).
- xb subtracts the lagged real bill:
  `xb_t = r_n_real_t - y_1_{t-1}` ([build_var_dataset_ar1_10y.py:171](data/build_var_dataset_ar1_10y.py#L171)).

### 6.2 Verified sample stats

Probe on the active CSV ([data/var_dataset.csv](data/var_dataset.csv),
T=92, 1920-2011):

| Column | Mean         | Std         | Hardcoded `_Z_BAR` match |
|--------|--------------|-------------|---------------------------|
| cape   | -2.7274      | 0.4364      | exact                     |
| spr    | +0.7181 pp   | 1.832 pp    | exact                     |
| y_1    | +2.0226 pp   | 2.937 pp    | exact                     |
| xr     | +5.1844 pp   | 19.12 pp    | exact                     |
| xb     | +0.6143 pp   | 6.349 pp    | exact                     |

`||z_bar_csv - _Z_BAR|| = 0` exactly. The hardcoded fallback in
[var.py:533-558](lifecycle/var.py#L533) is byte-equivalent to the live
CSV-estimated config, including `Phi` and `Omega`.

### 6.3 Stationarity

```
max |eig(Phi_full)|       = 0.9296    < 1                      STATIONARY
max |eig(Phi_11)|         = 0.9296    < 1                      STATIONARY
|eig(Phi_11)| sorted desc = [0.9296, 0.8839, 0.6040]
```

The dominant eigenvalue 0.9296 corresponds to the cape's slow mean
reversion (cape diagonal 0.876 plus cross-axis amplification through
Phi_11[cape, spr] = -0.516). All three eigenvalues are inside the unit
circle; no near-unit-root warning.

### 6.4 R² by equation

```
cape : 0.7999       slow-state, dominated by own AR(1) at 0.876
spr  : 0.3736       fast-revert with cross-loading on cape
y_1  : 0.6490       persistent (own diagonal 0.880)
xr   : 0.0960       low predictability (typical for stock returns)
xb   : 0.1294       moderate; spr is the dominant predictor (Phi_21[xb, spr] = +1.505)
```

Compare the older 20y AAA / v=0.995 dataset's `R²(xb)=0.31` reported in
[RETURN_TREATMENT_REVIEW_A_2026-05-09.md §6 O4](docs/scans/RETURN_TREATMENT_REVIEW_A_2026-05-09.md);
the 10y RLONG dataset has lower xb predictability (0.13) — see §9 finding
F-2.

### 6.5 What the AR(1) inflation forecast does

The static, full-sample AR(1) is a **modelling choice** ([build_var_dataset_ar1_10y.py:18-19](data/build_var_dataset_ar1_10y.py#L18)),
not a recursive out-of-sample exercise. Its implications:

- The deflator on the bill is matched to the bill's horizon (1-year
  expected inflation).
- The deflator on the long bond is matched to the long bond's horizon
  (10-year expected inflation, AR(1) average).
- This is closer to a **maturity-matched** Fisher decomposition than
  the prior CP-Shiller "uniform tau" simplification (which subtracted the
  same EWMA from both legs). Per
  [RETURN_TREATMENT_REVIEW_A_2026-05-09.md §5 caveat 3](docs/scans/RETURN_TREATMENT_REVIEW_A_2026-05-09.md),
  the uniform-subtract was a known approximation; the AR(1) baseline
  partially addresses it.
- The trade-off: (a) full-sample fit injects look-ahead bias (the AR(1)
  parameters use the entire 1920-2011 sample), but (b) the AR(1)
  parameters are economically stable and the look-ahead is not used
  for trading-strategy claims. (a) is acknowledged in the builder's
  module docstring and is not a bug.

### 6.6 Identity audit (from the builder)

[build_var_dataset_ar1_10y.py:verify:195-222](data/build_var_dataset_ar1_10y.py#L195)
performs five identity checks at build time:

- `T == 92` for 1920-2011.
- Output columns are `[cape, spr, y_1, xr, xb]`.
- No NaN in output.
- `y_1 == y_1_nom - E_pi_1` to 1e-14.
- `spr == y_n_real - y_1` to 1e-14.
- January 1920 row uses December 1919 inflation: `pi_info[1920] == pi_dec[1919]`.
- Row-`t` xb subtracts the lagged real bill:
  `xb[t] == r_n_real[t] - y_1.shift(1)[t]` to 1e-14.

These run on every build. PASS at the time of the production CSV write.

---

## §7. Annuity factor under the real-yields pivot

### 7.1 The function

[lifecycle/model.py:annuity_factor:319-351](lifecycle/model.py#L319):

```
A = sum_{k=1..b_bar} (1 + y(k))^{-k}
where y(k) = y_1 + spr * (k - 1) / (b_bar - 1)
```

Discrete compounding (1+y)^{-k}, not continuous exp(-y k). At y=5%, the
gap is ~12 bp/yr per period and would compound over `b_bar=10` periods to
a non-trivial discrepancy; the discrete form is consistent with the
codebase's other yield-arithmetic conventions (e.g. log1p(R/100) in the
data builder). ✓

### 7.2 Inputs at the active baseline

Canonical `b_bar=10` ([_canonical.py:31](configs/_canonical.py#L31)).
Term structure: linear interpolation between `y_1` (1y yield) and
`y_1 + spr` (10y yield) over k=1..10. With `spr = y_n_real - y_1`
(spec line [data/var_specification.md:91](data/var_specification.md#L91)),
the b_bar-th coupon discount uses the 10-year real yield exactly. ✓

### 7.3 Real vs nominal

[precompute.py:360-377](lifecycle/precompute.py#L360):

```
y_1_idx = model.y_1_index_in_state
spr_idx = model.spr_index_in_state
if y_1_idx is not None and spr_idx is not None:
    _y_1 = state_grid[:, model.y_1_index_in_state]
    _spr = state_grid[:, model.spr_index_in_state]
else:
    if y_1_idx is not None:
        _y_1 = state_grid[:, y_1_idx]
    else:
        _y_1 = np.full(N_state, model.y_1_scalar_fallback, dtype=float)
    if spr_idx is not None:
        _spr = state_grid[:, spr_idx]
    else:
        _spr = np.full(N_state, model.spr_scalar_fallback, dtype=float)
annuity_factors = annuity_factor(_y_1, _spr, model.b_bar)
```

The annuity factor consumes the **state-grid `y_1` and `spr` directly**.
In the real-yields pivot, both are real-deflated quantities (per §6),
so the discounting is in real terms. The bequest utility
[bequest_utility, bequest_marginal](lifecycle/model.py#L366) is in
real consumption units (the per-period-equivalent real consumption out
of the bequest), so the units are consistent: real wealth divided by a
real annuity factor.

### 7.4 Numerical sanity

Probe at the stationary mean (y_1=2.02%, spr=0.72%, b_bar=10):
`A = 8.7574` (matches manual computation). At y_1=spr=0, `A = 10.0`
(sum of ten 1's). At y_1=5%, spr=2pp:
yields ramp 5%→7% linearly; A is monotonically decreasing in either input
through the (1+y)^{-k} factor. ✓

### 7.5 System-1 fallback

System 1 sets `spr_scalar_fallback = mean(spr_csv)` ([var.py:507](lifecycle/var.py#L507)),
which is `+0.0072` (0.72pp) per the CSV. The annuity factor at System 1
uses the state-grid y_1 and the scalar spr-mean — internally consistent
within the System 1 ablation, but loses the bequest-time covariance of
the annuity factor with the spread state. This is a deliberate ablation
(System 1 has dropped spr from the state vector entirely).

---

## §8. Discretisation × VAR

### 8.1 Modes

[discretization.py:build_state_grid:158-299](lifecycle/discretization.py#L158)
supports three modes:

- **`naive`** (Rouwenhorst per axis): per-axis Rouwenhorst on
  `(rho_d, sigma_d) = (Phi[d,d], sqrt(Sigma[d,d]))`. Drops every cross-axis
  dynamic in `Phi_11` (it is a Kronecker product). Used historically for
  the `Pi_state` legacy path.
- **`lyapunov-axis`**: per-axis linspace from `mu_s ± n_stds * sigma_z[d]`,
  where `sigma_z[d] = sqrt(diag(Sigma_z))` is the diagonal of the
  Lyapunov-solved stationary covariance. Cross-axis correlations are
  dropped at the grid-points level (each axis is independently linspaced).
- **`cholesky`** (canonical, [_canonical.py:114](configs/_canonical.py#L114)):
  per-axis linspace in standardised u-coords from `-n_stds[d]` to
  `+n_stds[d]`, mapped to physical state via `s = mu_s + L u` with
  `L L' = Sigma_z` (Cholesky of full stationary cov). Grid points carry
  the cross-axis Cholesky correlation. ✓

### 8.2 Why Cholesky is canonical

Cholesky-mode grid points reproduce the Lyapunov stationary covariance
exactly (modulo the bin-prob discretisation error). Per
[DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md §2.B](docs/scans/DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md):

- `E[state]` recovered exactly to 1e-15.
- The solver-induced lag-1 regression matches `Phi_11` to within 3-5%
  (the worst element is `Phi_11[y_1<-spr]` at 55% relative error, but
  the absolute error is 1e-2 because the true value is small).
- `Var[state]` is **inflated by 22-33%** under the bracket-induced chain;
  this is the single most material discretisation defect in the production
  pipeline. Mechanism: the GH state-quadrature `v_nodes` recovers
  `Sigma_ss` exactly (max error 2.8e-17), but multilinear bracketing onto
  the truncated joint state grid pushes mass into corner cells that absorb
  ~9% of the analytical tail beyond the `±n_stds` bracket.

The audit recommended bumping `state_n_stds` from `(2.0, 2.25, 2.25)` to
`(3.0, 3.0, 3.0)`, which the canonical config has done
([_canonical.py:115](configs/_canonical.py#L115)). At `n_stds=3.0` per
axis, the joint excluded mass drops from ~9.16% to ~0.81% per the audit's
estimate. The 2026-05-10 source has applied this fix.

### 8.3 n_state_quad_nodes interaction with the partition

The state quadrature `v_nodes` integrates against the
**state-innovation covariance `Sigma_ss`** (not `Sigma_z`):
[discretization.py:get_state_quadrature:746-749](lifecycle/discretization.py#L746).
This is correct because `M_v_nodes = v_nodes @ M.T` projects the
**innovation** to a return innovation contribution, and the residual is
already drawn from `Sigma_r_cond` via the Cholesky in
`get_return_quadrature`. Adding the two yields a return draw with covariance
`M Sigma_ss M' + Sigma_r_cond = Sigma_rr` (Schur identity, §2.2),
matching the variance correction applied to `r_p`. ✓

### 8.4 K-bump y_1 is the third Cholesky axis under `(cape, spr, y_1)` ordering

Under Cholesky factorisation of `Sigma_ss`, axis order = state name order =
(cape, spr, y_1). The third Cholesky axis is the **residual y_1 direction**
after cape and spr are orthogonalised away. Refining axis 2 (per
[_canonical.py:122](configs/_canonical.py#L122) `n_state_quad_nodes=(3,3,5)`)
isolates additional resolution to the y_1 channel, which dominates `M[xb, :]`
(see §2.4 / §5.3). ✓

### 8.5 Edge cases

- **N_d=1**: a degenerate axis is supported
  ([discretization.py:266-269](lifecycle/discretization.py#L266)) and
  results in a single grid node at the unconditional mean of that axis.
  Not on the production hot path.
- **Lobatto branch**: prescribed-tails rule with closed-form K∈{3,5,7};
  out of canonical scope.
- **Sigma_r_cond near singular**: `eigvals(Sigma_r_cond) = [7.24e-5, 2.33e-3]`
  on the active VAR. Min eigenvalue is well above the
  `eig_min < 1e-5` warning threshold mentioned in
  [RETURN_TREATMENT_REVIEW_A_2026-05-09.md §4.1](docs/scans/RETURN_TREATMENT_REVIEW_A_2026-05-09.md).

### 8.6 Per-i_s log-return tensors

[solver.py:_precompute_per_is_tensors:1669-1699](lifecycle/solver.py#L1669)
calls `_build_step_log_returns` and `_build_step_state_brackets` once per
i_s under vmap. The host-side mirror
[solver.py:_all_is_log_returns_numpy:1629-1666](lifecycle/solver.py#L1629)
vectorises the same arithmetic in NumPy for the terminal-age kernel
builders. Both paths consume `pcj.y_1_idx`, `pcj.xr_pos`, `pcj.xb_pos`,
`pcj.const_r`, `pcj.A_r`, `pcj.M_v_nodes`, `pcj.ret_nodes` — the same
five inputs — and produce arrays of shape `(N_state, n_state_quad,
n_ret_quad)` for `(log_R_bill, log_x_s, log_x_b)`. Verified by the
[ARBITRAGE_PIVOT_REVIEW_2026-05-09.md §2](docs/scans/ARBITRAGE_PIVOT_REVIEW_2026-05-09.md)
diff against the diagnostic agent's reproduction.

---

## §9. Findings

| ID | Severity | Location | Issue | Fix sketch |
|----|----------|----------|-------|------------|
| F-1 | **MEDIUM (open from prior audit; partially mitigated)** | [discretization.build_state_grid](lifecycle/discretization.py#L158) under cholesky mode + multilinear bracket | Per [DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md](docs/scans/DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md), `Var[state]` inflates 22-33% under the bracket-induced chain even with the `n_stds=(3,3,3)` fix, and `Var[returns]` inflates 40-92% as a consequence. The production solver consumes the analytical `Sigma_rr` (not the bracket-induced one) for the CCV variance correction, so the FOC math is unaffected. The defect propagates only into the implied state-transition distribution, which feeds Bellman expectations through the multilinear interpolation of `c_next`. Effect on converged α_b is empirical (the audit estimated 5-10% of the bond-allocation gap). | Possible mitigations: (a) increase `state_grid_sizes` from (5,5,5) to (7,7,7) — quadruples per-age cell count; (b) finer multilinear interpolation (e.g., PCHIP) — currently linear-only by design ([solver.py:8](lifecycle/solver.py#L8)); (c) accept as a quantified small-bias caveat in the thesis. The canonical has chosen (c) plus partial mitigation via the `(3,3,3)` bracket bump. |
| F-2 | **HIGH (new finding from baseline change)** | [BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md](docs/scans/BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md) is **stale** with respect to the AR(1) 10y baseline | The diagnostic was written against the v=0.992 20y AAA baseline. Under the new AR(1) 10y RLONG baseline: `std(xb)` falls from 8.93% to 6.0%, `M[xb, spr]` falls from -13.5 to -6.83, `M[xb, y_1]` falls from -13.4 to -6.96, `R²(xb)` falls from 0.27 to 0.13, `Phi_21[xb, spr]` falls from +4.39 to +1.50. The leverage scaling `1/Var(xb)` is now ~1.83e+4 (vs the old 1.26e+4), so a +1pp shift in `mu_xb` produces a larger myopic α_b under the new baseline. The diagnostic's "α_b at +0.71σ spr" reasoning still applies qualitatively but the numerical conclusions (g=4 → α_b≈3.9 etc.) are no longer directly applicable. | Re-run the diagnostic against the AR(1) 10y baseline before any α_b interpretation in the thesis. The qualitative finding (probe-cell drift inflates printed α_b under even grid sizes) survives; quantitative numbers must be regenerated. |
| F-3 | **LOW (interpretation, not bug)** | [build_var_dataset_ar1_10y.py:90-106](data/build_var_dataset_ar1_10y.py#L90) | The AR(1) on Dec-over-Dec log CPI inflation is fit on the **full sample** (1920-2011) and then used as if it were the agent's expectation at every t. This injects look-ahead bias: an agent in 1925 cannot estimate a 1920-2011 AR(1). For policy claims this is irrelevant (the agent's information set is being modelled explicitly), but for any out-of-sample / forecasting-style claim it would bias results. | Document explicitly in the thesis. The builder docstring already calls this out ([build_var_dataset_ar1_10y.py:18-19](data/build_var_dataset_ar1_10y.py#L18)). |
| F-4 | **LOW (existing caveat, persists)** | [build_var_dataset_ar1_10y.py:171](data/build_var_dataset_ar1_10y.py#L171) — `xb_t = r_n_real_t - y_1_{t-1}` | This is a "model-implied real bond return" using the AR(1)-deflated yield sequence, not the ex-post realised real bond return (which would be `r_n_nom - pi_realized_{[t-1,t]}`). Internally consistent but a modelling choice. The Sharpe of `xb` (Jensen-adjusted) is ~+0.13 under this convention. | Document in thesis section on bond-return construction. Per `feedback_sensitivity_analysis.md`, it would be worth reporting α_b under both conventions in the robustness appendix. |
| F-5 | **LOW** | [BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md §6.A2](docs/scans/BOND_ALLOCATION_DIAGNOSTIC_2026-05-09.md) suggests moving the live-policy probe from per-axis midpoint to nearest-to-μ_s under even grids. Canonical uses (5,5,5) so the per-axis midpoint **is** μ_s and the issue does not bind. But `verify/test_baseline.py` and other tests may still use even sizes. | Ensure all test probes use odd grid sizes, or fix the probe to find the nearest-to-μ_s cell. Out of scope for this read-only review; flagged for the test author. | Independent fix in the test harness, or change `verify/test_baseline.py` `state_grid_sizes` from (4,4,4) to (5,5,5). |
| F-6 | **INFO** | [precompute.py:347-358](lifecycle/precompute.py#L347) | The `sigma_xrxb` cross-term is +8.06e-4 (correlation +0.073). Small positive cross-term means the FOC's `-a_s a_b sigma_xrxb` term marginally damps simultaneous long stocks + long bonds. The sign is correct and the magnitude is modest. | None. |
| F-7 | **INFO** | [solver.py:_pc_to_jnp:1762-1763](lifecycle/solver.py#L1762) `if n_state < 1 or n_state > 3: raise NotImplementedError` | The JAX solver only supports n_state ∈ {1, 2, 3}. The active System 1, 2, Full all fit. If a 4-axis ablation is ever needed, this gate must be relaxed and `_corner_offsets` / `_grid_strides` extended. | Out of scope; documented limit. |
| F-8 | **INFO** | [var.py:zero_bond_excess_mean_var_config](lifecycle/var.py#L592) and [zero_bond_simple_excess_mean_var_config](lifecycle/var.py#L639) | Experiment knobs that shift the unconditional mean of `xb` to a target (log or simple) value while preserving dynamics and covariance. Not used in the canonical baseline; available for sensitivity sweeps. The `(I-Phi) z_bar` re-solve preserves internal coherence. | None. |
| F-9 | **PARTIAL OBSOLESCENCE OF AUDITS** | [RETURN_TREATMENT_REVIEW_A_2026-05-09.md §2](docs/scans/RETURN_TREATMENT_REVIEW_A_2026-05-09.md), [STOCK_RETURN_TREATMENT_2026-05-09.md](docs/scans/STOCK_RETURN_TREATMENT_2026-05-09.md), [CP_INFLATION_PIPELINE_REVIEW_2026-05-09.md](docs/scans/CP_INFLATION_PIPELINE_REVIEW_2026-05-09.md), [EICHENGREEN_REAL_BOND_YIELDS_REVIEW_2026-05-09.md](docs/scans/EICHENGREEN_REAL_BOND_YIELDS_REVIEW_2026-05-09.md) — all reference the v=0.995 / v=0.992 CP-Shiller deflator and AAA 20y bond proxy. | These reviews remain conceptually correct (the §2.1.5 invariance for xr is still exact, the CCV joint-lognormality argument is unchanged, the deterministic-bill structural property still holds). But all numerical references to bond stats, Sharpe, R², and `Phi`/`Sigma` entries are obsolete by one builder revision. | Add a 1-paragraph "active baseline note" header to each of these scans pointing readers at the `891bb7c` commit and the AR(1) 10y CSV. Not a bug, but a documentation-debt obligation. |
| F-10 | **INFO** | [model.py:annuity_factor:319-351](lifecycle/model.py#L319) and [_canonical.py:b_bar=10](configs/_canonical.py#L31) — bequest horizon | The annuity discounts `(1+y)^{-k}` over k=1..10 with linear yield interpolation. Under the AR(1) 10y baseline, the bequest horizon **exactly matches** the bond horizon (10y), so the annuity is internally aligned with the bond leg used to construct `xb`. Under the previous 20y AAA baseline this alignment was broken (bond was 20y, annuity discounted 10y) — a subtle improvement of the new baseline that the existing audits did not flag. | None — improvement is in. |

### Concrete numerical reference table (active baseline)

For downstream verification of any future audit, the active VAR's
key scalars are:

```
sample              : 1920-2011, T=92 rows, annual Jan-Jan
columns             : [cape, spr, y_1, xr, xb] (state_idx [0,1,2], ret_idx [3,4])
z_bar               : [-2.7274, +0.7181pp, +2.0226pp, +5.1844pp, +0.6143pp]
max |eig(Phi)|      : 0.9296
R^2 by eq           : cape=0.7999, spr=0.3736, y_1=0.6490, xr=0.0960, xb=0.1294
sigma2_xr           : +3.4126e-2     (sigma(xr) = 18.5%)
sigma2_xb           : +3.6210e-3     (sigma(xb) = 6.0%)
sigma_xrxb          : +8.0616e-4     (corr(xr,xb) = +0.073)
M[xr, :]            : (-0.9379, +2.0349, +0.6428)            (cape, spr, y_1)
M[xb, :]            : (+0.0071, -6.8306, -6.9617)
Sigma_r_cond eigs   : [7.24e-5, 2.33e-3]                     (PD)
xr Sharpe (raw)     : +0.281 ; Jensen-adj +5.18 + 0.5*var = +6.89pp
xb Sharpe (raw)     : +0.102 ; Jensen-adj +0.61 + 0.5*var = +0.80pp
```

---

## §10. Verdict

**PASS WITH CAVEATS.**

The asset-returns specification and pipeline are theoretically and
numerically sound under the real-yields pivot. Specifically:

- The VAR partition algebra ([var.py:partition_var](lifecycle/var.py#L49))
  satisfies the Schur identity `Sigma_rr = M Sigma_ss M' + Sigma_r_cond`
  to machine precision, with `M = Sigma_rs Sigma_ss^-1` and
  `Sigma_r_cond` strictly positive definite.
- The CCV log-portfolio formula in
  [_ccv_log_return_and_grad](lifecycle/solver.py#L821) matches the
  Campbell-Viceira w8566 eq. (10) term-by-term, with the variance
  scalars correctly sourced from the **full** `Sigma_rr` (not
  `Sigma_r_cond`) per the 2026-05-06 patch — verified by the independent
  re-derivation in
  [CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md](docs/scans/CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md).
- The bill is correctly anchored to the current state via
  `log_R_bill_{t+1} = state_t[y_1_idx]` in both the JAX and host paths.
- The (k_v, k_r) joint quadrature factorises correctly as
  `weight_kv_kr = v_weights[:, None] * ret_weights[None, :]` because
  the Cholesky-orthogonalisation makes the two innovation sources
  independent, and the y_1-axis K-bump is targeted at the dominant
  M[xb, y_1] channel.
- The new AR(1)-matched 10y RLONG VAR baseline is stationary
  (max |eig(Phi)| = 0.9296), reproduces the hardcoded fallback
  byte-for-byte, satisfies all five build-time identity checks, and
  has aligned the bond horizon with the bequest annuity horizon
  (b_bar = 10).
- The annuity factor consumes real `y_1` and `spr` directly from
  the state grid, with discrete compounding consistent with the rest
  of the codebase.

The caveats are:

1. **Discretised-state covariance inflation** (F-1, carried over from
   [DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md](docs/scans/DISCRETIZED_VAR_FIDELITY_AUDIT_2026-05-09.md)):
   the bracket-induced state chain inflates `Var[state]` by 22-33% even
   under the (3,3,3) bracket bump. The CCV variance correction is
   unaffected (it uses the analytical `Sigma_rr`), but the implied
   state-transition density entering Bellman expectations carries the
   bias. Flagged but accepted as a quantified small bias.
2. **Stale prior audits** (F-9): the 2026-05-09 returns/stock/CP/Eichengreen
   reviews reference numerical values from the v=0.995 / v=0.992 / 20y
   AAA baseline that has been superseded by the AR(1) / 10y RLONG
   baseline (commit 891bb7c). Conceptual conclusions remain valid; the
   numerical headers need an "active baseline" pointer.
3. **Bond-allocation diagnostic stale numbers** (F-2): the BOND_ALLOCATION
   diagnostic's specific α_b values are not directly applicable to the
   new baseline; the qualitative grid-parity insight survives.
4. **Look-ahead in the AR(1)** (F-3): full-sample inflation AR(1) is a
   modelling choice acknowledged in the builder; not a bug.

None of the caveats blocks the production solver run; all are either
pre-existing-and-quantified (F-1) or documentation/numerical-staleness
issues (F-2, F-9). The asset-returns pipeline as instantiated on
`jax-rewrite` at the time of this review (commit family around
`7389238`) is internally consistent, theoretically grounded, and ready
to drive the canonical solve.

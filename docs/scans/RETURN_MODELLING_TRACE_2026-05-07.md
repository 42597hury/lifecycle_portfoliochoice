# Return-Modelling Pipeline — End-to-End Trace

**Date:** 2026-05-07
**Branch:** `jax-rewrite`
**Companion appendix:** [`RETURN_MODELLING_PARAMS_2026-05-07.md`](RETURN_MODELLING_PARAMS_2026-05-07.md)
**Purpose:** Self-contained reference for a theory reviewer to vet that the
return-modelling implementation matches the CCV (Campbell–Chacko–Viceira,
NBER w8566 / Campbell–Viceira 2002) mathematical specification end-to-end.
No code changes; no correctness judgments.

This trace walks the five layers the handoff identifies (raw data →
variable construction → VAR estimation → CCV continuous-rebalancing log
return → discretization & solver consumption) and explicitly calls out
the two design choices that depart from textbook CCV: the **rtb-as-state
migration** (§9) and the **vol-drag scalars sourced from `Sigma_rr`
rather than `Sigma_r_cond`** (§6.7). Both are flagged for theory review.

---

## §0. Reading map

The trace inlines short code excerpts where helpful; every claim about
"what the code computes" cites a `file:line`. For full numerical values
see the companion appendix.

Section index:
- [§1 — Data sources and preprocessing](#1-data-sources-and-preprocessing)
- [§2 — Variable construction](#2-variable-construction)
- [§3 — VAR specification](#3-var-specification)
- [§4 — VAR estimation procedure (reproduced from raw data)](#4-var-estimation-procedure-reproduced-from-raw-data)
- [§5 — VAR parameter values (numerical) → see appendix](#5-var-parameter-values-numerical)
- [§6 — Continuous rebalancing — the CCV log-return formula](#6-continuous-rebalancing--the-ccv-log-return-formula)
- [§7 — Discretization of the return distribution](#7-discretization-of-the-return-distribution)
- [§8 — Solver-side construction of per-cell return tensors](#8-solver-side-construction-of-per-cell-return-tensors)
- [§9 — The rtb-as-state semantics (CRITICAL)](#9-the-rtb-as-state-semantics-critical)
- [§10 — Variable-name dictionary](#10-variable-name-dictionary)
- [§A — Naming caveats and FLAGs collected from the trace](#a-naming-caveats-and-flags-collected-from-the-trace)

---

## §1. Data sources and preprocessing

### 1.1 Raw inputs

All raw series live under [`data/Thesisdata/`](../../data/Thesisdata/).
The single ingestion script
[`data/build_var_dataset.py`](../../data/build_var_dataset.py) is the
authoritative pipeline that turns them into `data/var_dataset.csv`. It
loads four files and reads no others (no GSW yield curve, no Ken French,
no FRED API):

| Series | Source file | Source citation in code | Frequency | Used for |
|---|---|---|---|---|
| `DGS1` (1-year Treasury yield, % p.a.) | `Thesisdata/DGS1.csv` | FRED ticker (loader at [`build_var_dataset.py:24-26`](../../data/build_var_dataset.py#L24-L26)) | daily | `y_1`, `r_1`, `rtb` |
| `AAA` (Moody's seasoned Aaa corporate bond yield, % p.a.) | `Thesisdata/AAA.csv` | FRED ticker (loader at [`build_var_dataset.py:29-31`](../../data/build_var_dataset.py#L29-L31)) | monthly | `y_20` (used as the long yield), `xb` |
| `CPIAUCSL` (CPI, all urban, seasonally adjusted) | `Thesisdata/CPIAUCSL.csv` | FRED ticker (loader at [`build_var_dataset.py:34-36`](../../data/build_var_dataset.py#L34-L36)) | monthly | annual log inflation `pi`, ex-post real bill `rtb` |
| Shiller `ie_data.xls`, sheet `Data` | `Thesisdata/ie_data.xls` | Shiller online data (loader at [`build_var_dataset.py:39-50`](../../data/build_var_dataset.py#L39-L50)) | monthly | nominal stock total return; `cy = -log(CAPE)` |

The other files in `Thesisdata/` (`TB3MS.csv`, `CPIAUCNS.csv`,
`feds200628 (1).csv`, `LW_monthly.xlsx`, `GS20.csv`, `DGS30.csv`) are
**not** consumed by the production pipeline — they are reference data left
in place from earlier iterations.

### 1.2 Frequency, sample, and resampling

- **Frequency:** the VAR is estimated **directly at annual frequency**
  (not aggregated from quarterly, despite what
  [`data/var_specification.md`](../../data/var_specification.md) says —
  see FLAG-A).
- **Resampling:** end-of-year ("YE-DEC", last observation in December)
  for `DGS1`, `AAA`, `CPIAUCSL`. Shiller monthly P, D summed over
  calendar year for stock returns; Shiller `CAPE` taken at December
  ([`build_var_dataset.py:53-82`](../../data/build_var_dataset.py#L53-L82)).
- **Sample period:** 1963–2025 (T = 63 annual observations after the
  one-year lag burn-in for `r_1.shift(1)` and `pi`).
- **No demeaning, no deflating** at the dataset stage; deflation enters
  algebraically via `rtb = log(1 + y_1[T-1]) - pi[T]` and the VAR
  estimator demeans internally.

### 1.3 Verification identities (built into the pipeline)

The pipeline runs four hard-fail identity checks before saving
([`build_var_dataset.py:154-191`](../../data/build_var_dataset.py#L154-L191)).
On the current run (2026-05-07) all four pass at machine precision:

```
V1 (rtb identity):       max |resid| = 3.036e-18  PASS
V2 (bond identity):      max |resid| = 0.000e+00  PASS
V3 (real stock identity): max |resid| = 5.551e-17  PASS
V+ (real bond recovery):  max |resid| = 2.776e-17  PASS
```

The published dataset header confirms the sample:
```
Dataset: years 1963 to 2025, T=63 rows
```

---

## §2. Variable construction

The 6-variable VAR consumes
`columns = [y_1, spr, cy, rtb, xr, xb]`. Algebraic definitions in terms
of raw series, lifted directly from
[`build_var_dataset.py`](../../data/build_var_dataset.py):

### 2.1 `y_1` — 1-year nominal Treasury yield (decimal)

`y_1[T] = DGS1.resample('YE-DEC').last() / 100`
([`build_var_dataset.py:57`](../../data/build_var_dataset.py#L57)).
End-of-year December observation, converted from percent to decimal.

### 2.2 `y_20` — 20-year long-bond yield proxy (decimal, intermediate)

`y_20[T] = AAA.resample('YE-DEC').last() / 100`
([`build_var_dataset.py:62`](../../data/build_var_dataset.py#L62)).
Moody's Aaa corporate bond yield, used as the long yield. **Not the same
as a default-free Treasury 20-year yield** — see FLAG-B.

### 2.3 `spr` — yield spread

`spr[T] = y_20[T] − y_1[T]`
([`build_var_dataset.py:89`](../../data/build_var_dataset.py#L89)).
Long-minus-short. Decimal.

### 2.4 `cy` — log earnings yield

`cy[T] = − log(CAPE[T])` where `CAPE` is the December observation of
Shiller's cyclically-adjusted P/E
([`build_var_dataset.py:92`](../../data/build_var_dataset.py#L92)).

In CCV's notation this is the **negative log dividend-price ratio**
analog: high `cy` = earnings cheap relative to price = low CAPE = low
recent valuation. **Sign convention is opposite to CCV's `log(P/D)`**, so
sign-comparisons against CCV must flip — `verify_var_vs_ccv.py` does this
explicitly ([`scripts/scratch/verify_var_vs_ccv.py:165`](../../scripts/scratch/verify_var_vs_ccv.py#L165)).

### 2.5 `pi` — annual log CPI inflation (intermediate, not in VAR)

`pi[T] = log(CPI_Dec[T] / CPI_Dec[T-1])`
([`build_var_dataset.py:96`](../../data/build_var_dataset.py#L96)).

### 2.6 `r_1` — log gross nominal bill return (intermediate)

`r_1[T] = log(1 + y_1[T])`
([`build_var_dataset.py:100`](../../data/build_var_dataset.py#L100)).
Discrete compounding.

### 2.7 `rtb` — ex-post real bill return

`rtb[T] = r_1[T-1] − pi[T] = log(1 + y_1[T-1]) − pi[T]`
([`build_var_dataset.py:104`](../../data/build_var_dataset.py#L104)).

The bill yield is **lagged by one year**: `r_1[T-1]` is the yield
observed at end of year T-1, which prices a 1-year bill that is then
held over year T. Inflation `pi[T]` is the realised log CPI change over
year T. So `rtb[T]` is the **realised** real return on the bill that was
purchased at the start of year T.

### 2.8 `r_bond` — log gross nominal bond return (intermediate)

`r_bond` uses the CCV loglinear constant-duration approximation
([`build_var_dataset.py:107-114`](../../data/build_var_dataset.py#L107-L114)):

```
n_bond = 20                                          # par maturity (years)
D[T]   = (1 − (1 + y_20[T])^(-n_bond)) / (1 − (1 + y_20[T])^(-1))
y_log_20[T] = log(1 + y_20[T])
r_bond[T]   = D[T-1] · y_log_20[T-1] − (D[T-1] − 1) · y_log_20[T]
```

`D[T]` is the Macaulay duration of a `n_bond`-year par bond at yield
`y_20[T]`. This is exactly the CCV log-return approximation for a
constant-duration bond (Campbell–Viceira 2002, eq. 4.4). Mean realised
duration on the sample is 11.76 years, consistent with the 20-year par
bond convention.

### 2.9 `xb` — excess nominal bond return

`xb[T] = r_bond[T] − r_1[T-1]`
([`build_var_dataset.py:118`](../../data/build_var_dataset.py#L118)).

### 2.10 `xr` — excess nominal stock return

```
nom_ret_m[t] = log( (P[t] + D[t]/12) / P[t-1] )           # Shiller monthly
nominal_stock[T] = sum_{t in calendar year T} nom_ret_m[t]
xr[T] = nominal_stock[T] − r_1[T-1]
```
([`build_var_dataset.py:75-82, 126`](../../data/build_var_dataset.py#L75-L82))

This is a Shiller-CRSP S&P 500 total log return aggregated to annual,
minus the lagged 1-year bill log yield.

### 2.11 Summary statistics on the production sample (T = 63)

From the pipeline's print at
[`build_var_dataset.py:198-206`](../../data/build_var_dataset.py#L198-L206):

```
y_1 : mean=+0.0485 (+4.85%), std=0.0330, min=+0.0010, max=+0.1386
spr : mean=+0.0199 (+1.99%), std=0.0159, min=-0.0141, max=+0.0489
cy  : mean=-2.9929,          std=0.4574, min=-3.7887, max=-2.0583
rtb : mean=+0.0091 (+0.91%), std=0.0278, min=-0.0683, max=+0.0878
xr  : mean=+0.0555 (+5.55%), std=0.1604, min=-0.5310, max=+0.2589
xb  : mean=+0.0143 (+1.43%), std=0.0904, min=-0.2318, max=+0.1720
```

---

## §3. VAR specification

### 3.1 The system equations

The full reduced-form VAR(1) operates on the 6-vector
`z = (y_1, spr, cy, rtb, xr, xb)`:

```
z_{t+1} = const + Phi · z_t + e_{t+1},
e_{t+1} ~ N(0, Omega).
```

After the partition into `state` and `return` blocks (see §3.2):

```
state index set:  S = {2, 1, 3, 0}   →  rows (cy, spr, rtb, y_1)
return index set: R = {4, 5}         →  rows (xr, xb)

s_{t+1}     = Phi_0_state + Phi_11 · s_t + v_state,t+1
xr,xb_{t+1} = Phi_0_ret   + Phi_21 · s_t + v_ret,t+1

Cov[(v_state, v_ret)] = [[Sigma_ss,  Sigma_sr],
                          [Sigma_rs,  Sigma_rr ]]   (block-structured)
```

with the cross-block `Sigma_rs` (= `Sigma_sr.T`) **non-zero by design**
(see §3.4).

### 3.2 Restrictions imposed during estimation

The CCV-constrained restricted estimator at
[`var.py:191-268`](../../lifecycle/var.py#L191-L268) imposes:

1. **Lagged returns excluded from RHS.** Only lagged
   `(y_1, spr, cy, rtb)` enter every equation; `Phi[:, xr]` and
   `Phi[:, xb]` are zero by construction (the CCV 2003 §4.2 restricted
   estimator). Verified post-fit:
   `var_config["max_abs_return_lag_coeff"] = 0.000e+00`
   ([`var.py:348`](../../lifecycle/var.py#L348)).
2. **`z_bar` pinned to the sample mean.** OLS is run on demeaned data
   without an intercept, then the implied intercept is recovered as
   `const = (I − Phi) · z_bar`. This is the CCV constrained estimator
   (CCV 2003); it is not OLS-with-intercept.
3. **No restriction on `rtb` lags**, post the rtb-as-state migration. The
   `rtb` column of `Phi` is freely estimated and captures the inflation
   persistence channel (`Phi[rtb, rtb] = +0.3627`).

### 3.3 Dimensions

Concretely (see partition output):
- `n_state = 4`, `n_ret = 2`, full VAR `n = 6`.
- `Phi` shape `(6, 6)`; `Phi_11` shape `(4, 4)`; `Phi_21 = A_r` shape
  `(2, 4)`.
- `Omega` shape `(6, 6)`; partitioned into
  `Sigma_ss (4, 4)`, `Sigma_rr (2, 2)`, `Sigma_rs (2, 4)`.

### 3.4 Cross-block covariance `Sigma_(state, return)` — KEY DESIGN CHOICE

**The joint innovation covariance has a non-zero off-diagonal block linking
state innovations and return innovations.** This is *not* a textbook
"diagonal-block" simplification — the codebase explicitly implements the
full joint Gaussian.

- The cross-block is `Sigma_rs`
  ([`var.py:71`](../../lifecycle/var.py#L71)) with shape `(n_ret,
  n_state) = (2, 4)`. Numerical values in
  [appendix §5.7](RETURN_MODELLING_PARAMS_2026-05-07.md#57-cross-block-covariance-sigma_rs-2x4-return-rows--state-cols).
- It is consumed via the projection
  `M = Sigma_rs · Sigma_ss⁻¹` ([`var.py:74`](../../lifecycle/var.py#L74))
  and the conditional residual covariance
  `Sigma_r_cond = Sigma_rr − M · Sigma_sr`
  ([`var.py:75`](../../lifecycle/var.py#L75)).

**Conditional structure that the code implements.** Given a state
innovation draw `v_state`, the conditional distribution of the return
innovation is

```
v_ret | v_state ~ N(M · v_state, Sigma_r_cond).
```

This is the textbook conditional Gaussian for a jointly normal pair —
the codebase implements it explicitly:

| Where | Code | Math |
|---|---|---|
| Conditional return *mean* per (state-quad node, base state) | [`solver.py:778-789`](../../lifecycle/solver.py#L778-L789) and [`precompute.py:473-488`](../../lifecycle/precompute.py#L473-L488) | `mu_r_per[k_v, :] = (const_r + A_r · s_t) + M_v_nodes[k_v]`, where `M_v_nodes = v_nodes · M.T` |
| Conditional return *residual* draw | [`discretization.py:643-647`](../../lifecycle/discretization.py#L643-L647) | `ret_nodes = z_quad · L.T` with `L L.T = Sigma_r_cond` (Cholesky of the *conditional* cov) |
| Same conditional structure in simulator | [`simulation.py:329-338`](../../lifecycle/simulation.py#L329-L338) | `v_s = L_ss · z_innov; mu_r = base_mu_r + M · v_s; resid = ret_factor · z_ret` with `ret_factor` = Cholesky of `Sigma_r_cond` ([`simulation.py:97-104`](../../lifecycle/simulation.py#L97-L104)) |

**No part of the pipeline ignores the cross-block.** The Cholesky of
`Sigma_r_cond` (not `Sigma_rr`) for the inner residual draw, combined
with the additive `M · v` shift, is exactly the joint draw from the full
6×6 `Omega` decomposed as
`(v_state, v_ret) = (L_ss z_1, M L_ss z_1 + L_cond z_2)`. The two-step
decomposition is preferred over a single `(6×6)` Cholesky because the
state grid is built around the standardised `v_state` axis and the
returns are integrated *given* the state.

### 3.5 Innovation distribution

Gaussian by assumption everywhere — no heavy-tail or mixture spec on the
return-block side. (The income process uses a mixture-normal for
`(eta, eps)`, but that is independent of the financial VAR.)

### 3.6 Stationarity

`Phi_11` eigenvalue moduli (descending):

```
[0.92559216, 0.78598056, 0.78598056, 0.32273311]   max |λ| = 0.926
```

Strictly < 1 → state sub-VAR is stationary
([`var.py:124-129`](../../lifecycle/var.py#L124-L129) prints this banner;
[`partition_var()`](../../lifecycle/var.py#L49) does not assert it but
[`stationary_covariance()`](../../lifecycle/discretization.py#L65) used
later by `build_state_grid()` does).

---

## §4. VAR estimation procedure (reproduced from raw data)

### 4.1 Where the estimator lives

[`lifecycle/var.py:191-268`](../../lifecycle/var.py#L191-L268) —
`estimate_var1_from_csv()` and the `restricted` wrapper.

Algorithmic summary:

1. Load the dataset (`pd.read_csv(csv_path)`) and select the required
   columns. Drop NA, cast to float.
2. Compute `z_bar` = sample mean over all rows.
3. Demean: `Z = data − z_bar`. Form `Y = Z[1:]` and
   `X = Z[:-1, state_columns]` (state-only RHS for the restricted VAR).
4. OLS without intercept: `coeffs = lstsq(X, Y)`.
5. Build the full `Phi` by zero-padding the return-lag columns. Recover
   `const = (I − Phi) · z_bar`.
6. Residuals `resid = Y − X · coeffs`; covariance
   `Omega = (resid.T · resid) / (T − k_predictors)`, where
   `T = 62, k_predictors = 4`.

This is the **CCV 2003 §4.2 constrained estimator**: the sample mean is
imposed as the unconditional mean (rather than absorbed into a free
intercept), and lagged returns are excluded from every equation.

### 4.2 Reproduction recipe (executed end-to-end)

The script
[`scripts/scratch/reproduce_var_for_handoff.py`](../../scripts/scratch/reproduce_var_for_handoff.py)
runs:

1. `cd data && python build_var_dataset.py` — rebuilds
   `data/var_dataset.csv` (1963–2025, T = 63). All four verification
   identities pass at machine precision (see §1.3).
2. `python scripts/scratch/reproduce_var_for_handoff.py` — calls
   `build_nominal_system1_var_config(csv_path, …)` and
   `build_nominal_system1_var_config_hardcoded()`, prints both, and
   shows element-wise diffs.

### 4.3 Reproduction outcome

Estimator output and the hardcoded snapshot in
[`var.py:608-680`](../../lifecycle/var.py#L608-L680) agree to floating-point
round-off:

```
max |Phi_estimated   - Phi_hardcoded|   = 4.441e-16
max |Omega_estimated - Omega_hardcoded| = 3.469e-18
max |z_bar           difference|        = 6.94e-18
max |const           difference|        = 2.22e-16
```

Per-equation R² (from the estimator, identical to the hardcoded contract
documented in
[`var.py:646-657`](../../lifecycle/var.py#L646-L657)):

```
y_1: R^2 = 0.789525
spr: R^2 = 0.532693
cy : R^2 = 0.879459
rtb: R^2 = 0.607496       ← matches the documented contract (0.6075)
xr : R^2 = 0.073116
xb : R^2 = 0.325587
```

The hardcoded path is therefore a **frozen exact snapshot** of the
estimator output, not a hand-typed approximation. It is gated by the
existence of `data/var_dataset.csv`; if the CSV is missing, the
hardcoded path serves as a fallback that returns the same numbers.

### 4.4 Estimation method, sample, restrictions — summary

| Item | Value |
|---|---|
| Method | OLS on demeaned data (no intercept), CCV constrained estimator |
| Frequency | Annual (estimated directly on annual data; not aggregated from quarterly) |
| Sample | 1963–2025, T = 63 annual observations |
| Effective T for OLS | 62 (one obs lost to lag) |
| Predictors per equation | 4 (lagged `cy, spr, rtb, y_1`) |
| Restrictions | `Phi[:, xr] = Phi[:, xb] = 0`; `z_bar` pinned to sample mean |
| Stationarity | enforced by check; max `|λ(Phi_11)| = 0.926 < 1` |

---

## §5. VAR parameter values (numerical)

→ See [`RETURN_MODELLING_PARAMS_2026-05-07.md`](RETURN_MODELLING_PARAMS_2026-05-07.md)
for all element-by-element tables (`z_bar`, `Phi`, `Omega`, `const`,
`Phi_0_state`, `Phi_11`, `Phi_0_ret`, `Phi_21 = A_r`, `Sigma_ss`,
`Sigma_rr`, `Sigma_rs`, `M`, `Sigma_r_cond`, joint-`Omega` eigenvalues,
`Phi_11` eigenvalues, R² per equation, and the CCV vol-drag scalars
actually consumed by the kernel).

The appendix prints both the estimated and the hardcoded columns
side-by-side with element-wise diffs — agreement is at floating-point
round-off (`max |diff| ≈ 4.4e-16` on `Phi`).

---

## §6. Continuous rebalancing — the CCV log-return formula

### 6.1 The formula in code

[`solver.py:694-716`](../../lifecycle/solver.py#L694-L716) contains the
single canonical implementation:

```python
def _ccv_log_return_and_grad(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                              sigma2_xr, sigma2_xb, sigma_xrxb):
    """Return (R_p, dr/da_s, dr/da_b) for CCV log-wealth dynamics.

    r_p = log_R_bill + a_s*log_x_s + a_b*log_x_b
          + 0.5*(a_s*sigma2_xr + a_b*sigma2_xb)
          - 0.5*(a_s^2*sigma2_xr + 2*a_s*a_b*sigma_xrxb + a_b^2*sigma2_xb)
    """
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
    R_p = jnp.exp(r_p)
    dr_da_s = log_x_s + sigma2_xr * (0.5 - alpha_s) - alpha_b * sigma_xrxb
    dr_da_b = log_x_b + sigma2_xb * (0.5 - alpha_b) - alpha_s * sigma_xrxb
    return R_p, dr_da_s, dr_da_b
```

### 6.2 The formula in math

Per quadrature realisation `(k_v, k_r)` (state-innovation node × return
quadrature node):

```
r_p = log_R_bill
    + α_s · log_x_s + α_b · log_x_b
    + ½ · (α_s · σ²_xr + α_b · σ²_xb)              ← Jensen lift
    − ½ · (α_s² · σ²_xr
           + 2 · α_s · α_b · σ_xrxb
           + α_b² · σ²_xb)                          ← Itô / vol-drag

R_p = exp(r_p)                                      ← gross portfolio return
```

Term-by-term:
- `log_R_bill` — realised log nominal bill return for the period. Under
  rtb-as-state (§9), this is read from the next-period state vector at
  `rtb_index_in_state`, *not* drawn from a return quadrature shock.
- `α_s · log_x_s + α_b · log_x_b` — linear excess-return terms.
- `½ · (α_s σ²_xr + α_b σ²_xb)` — the Jensen lift from
  `E[exp(X)] = exp(E[X] + ½ Var[X])` applied to each individual log
  excess return.
- `½ · α' Σ_xx α` — the Itô / vol-drag from converting the *portfolio* of
  log returns back into an arithmetic combination. At single-asset
  corners the lift and drag cancel and `r_p = log_R_bill + xr` (or
  `xb`). At zero allocation they vanish and `r_p = log_R_bill`.

### 6.3 Source citation

The handoff doc [`docs/CCV_RETURNS.md`](../CCV_RETURNS.md) cites
**Campbell, Chacko, Rodriguez and Viceira (NBER w8566, equation 10;
reprinted in Campbell & Viceira 2002)**
([`docs/CCV_RETURNS.md:46-54`](../CCV_RETURNS.md#L46-L54)).
The code docstring at
[`solver.py:696-700`](../../lifecycle/solver.py#L696-L700) does not name
the equation but the formula matches CCV w8566 eq. (10) symbol for
symbol.

The simulator carries the matching paper-reference comment at
[`simulation.py:349-351`](../../lifecycle/simulation.py#L349-L351):
> "CCV log portfolio return (Campbell-Viceira w8566 eq. 10). Must agree
> node-by-node with solver._ccv_log_return_and_grad — verified in
> verify/ccv_solver_sim_parity.py."

### 6.4 Solver / simulator parity

[`verify/ccv_solver_sim_parity.py`](../../verify/ccv_solver_sim_parity.py)
is a 1000-trial randomised parity check; it asserts solver and simulator
agree on `R_p` to `1e-12` at every random `(α, log_R_bill, log_x_s,
log_x_b, σ²_xr, σ²_xb, σ_xrxb)` realisation. The simulator block
([`simulation.py:352-362`](../../lifecycle/simulation.py#L352-L362))
mirrors `solver._ccv_log_return_and_grad` arithmetically.

### 6.5 Where the gradient is consumed (Hessian-of-V structure)

Under CCV the FOC is the gradient of value, not the asset-pricing moment
condition (`R_j − R_bill` is constant in α; the α-dependence runs through
the variance-quadratic):

```
FOC_j(α) = E[μ_comb(c_{t+1}, sR_p) · R_p · ∂r_p/∂α_j]
```

The Newton Jacobian is the Hessian of V — symmetric, with both an
outer-product term and a `−Σ_jk` correction from the second derivative
of the variance-quadratic
([`docs/CCV_RETURNS.md:130-150`](../CCV_RETURNS.md#L130-L150)). The
terminal-age FOC at
[`solver.py:723-756`](../../lifecycle/solver.py#L723-L756) is the
canonical reference kernel — it shows the `extra_ss/bb/sb`
`−Σ_jk` correction terms explicitly.

### 6.6 Where `σ²_xr`, `σ²_xb`, `σ_xrxb` come from

Constructed once at precompute time
([`precompute.py:303-314`](../../lifecycle/precompute.py#L303-L314)):

```python
xr_pos = model.ret_names.index("xr")
xb_pos = model.ret_names.index("xb")
sigma2_xr  = float(model.Sigma_rr[xr_pos, xr_pos])
sigma2_xb  = float(model.Sigma_rr[xb_pos, xb_pos])
sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])
```

These three scalars are stored as `pc.sigma2_xr / pc.sigma2_xb /
pc.sigma_xrxb` and consumed identically by both the solver and the
simulator
([`simulation.py:720-722`](../../lifecycle/simulation.py#L720-L722)).

### 6.7 The vol-drag scalars source — controversy

This is a **deliberate code-level design choice that contradicts the
companion documentation**, and it should be on the theory reviewer's
list to confirm.

**What the code does (post commit `f23ac83`, May 2026, the patch
referenced in
[`precompute.py:303-311`](../../lifecycle/precompute.py#L303-L311)
and confirmed in
[`scripts/scratch/ccv_audit_numerics.py:115`](../../scripts/scratch/ccv_audit_numerics.py#L115)):**

> Source the CCV vol-drag scalars from the **unconditional** `Sigma_rr`,
> not the conditional `Sigma_r_cond`.

The justification in the precompute comment
([`precompute.py:304-311`](../../lifecycle/precompute.py#L304-L311)):
> "CCV w8566 eq. (10) takes expectations over the FULL VAR innovation
> v_{t+1}, so the constants in r_p are sourced from the unconditional
> return-block covariance Sigma_rr — NOT the residual Sigma_r_cond used
> for the inner-quadrature Cholesky in
> discretization.get_return_quadrature. Sigma_r_cond differs from
> Sigma_rr by the M·Sigma_ss·M' projection term, which can be ~10–30x in
> this calibration; using the wrong matrix shrinks the Itô vol-drag and
> inflates converged alpha_s past sensible levels. Patched 2026-05-06
> (Sigma_rr matches CCV Table 2 vols + Markowitz alpha)."

**What `docs/CCV_RETURNS.md` still says:**
[`docs/CCV_RETURNS.md:200-211`](../CCV_RETURNS.md#L200-L211) and the
formula reference card at
[`docs/CCV_RETURNS.md:459-461`](../CCV_RETURNS.md#L459-L461) document the
*pre-patch* convention:

> "`σ²_xr = pc.sigma2_xr = Sigma_r_cond[1, 1]`,
>  `σ²_xb = pc.sigma2_xb = Sigma_r_cond[2, 2]`,
>  `σ_xrxb = pc.sigma_xrxb = Sigma_r_cond[1, 2]`."

> "`Sigma_r_cond` is the *conditional* covariance of (rtb, xr, xb) after
> projecting out state innovations […] The CCV formula uses the
> conditional one, *not* the unconditional `Sigma_rr`."

**These two statements are direct contradictions of the current
code.** The Appendix-A formula card additionally still indexes a
3-element return block `(rtb, xr, xb)` — pre-rtb-as-state.

**Magnitudes (from the appendix, §9):**

| Scalar | `Sigma_rr` (production) | `Sigma_r_cond` (CCV_RETURNS.md) | Ratio |
|---|---|---|---|
| `σ²_xr`  | `+2.529e-02` | `+5.909e-04` | 42.8× |
| `σ²_xb`  | `+5.882e-03` | `+5.164e-04` | 11.4× |
| `σ_xrxb` | `+3.462e-03` | `+3.745e-05` | 92.4× |

The ratio is large — choice between matrices materially changes the
Itô vol-drag and therefore converged optimal `α`.

**FLAG-C for theory review:**
- Is `Sigma_rr` (current code) or `Sigma_r_cond` (docs) the
  mathematically correct vol-drag input under CCV w8566 eq. (10)?
- The theory question hinges on whether the CCV expectation is over
  `v_{t+1} = (v_state, v_ret)` jointly (favouring `Sigma_rr` via the
  marginal return cov) or over `v_ret | v_state` after the state-grid
  conditioning has been done (favouring `Sigma_r_cond`).
- The codebase contains a previous audit report — see
  [`docs/handoff/HANDOFF_CCV_THEORY_AUDIT_REPORT.md`](../handoff/HANDOFF_CCV_THEORY_AUDIT_REPORT.md)
  §C3.2 — that used the `Sigma_r_cond` convention. The May 2026 patch
  reversed that without updating the audit report or `CCV_RETURNS.md`.

---

## §7. Discretization of the return distribution

### 7.1 Return-residual quadrature (`get_return_quadrature`)

[`discretization.py:571-649`](../../lifecycle/discretization.py#L571-L649)
builds a tensor-product Gauss-Hermite rule on `N(0, Σ_r_cond)`:

1. Per-axis 1D rules: standard Gauss-Hermite of order `K_d` on `N(0, 1)`
   with nodes scaled by `sqrt(2)` and weights divided by `sqrt(π)`
   ([`_build_axis_grid`](../../lifecycle/discretization.py#L519-L544)).
2. Tensor product across the `n_ret = 2` return axes; weights
   multiplied, nodes stacked.
3. Cholesky factor `L`: `L · L^T = Sigma_r_cond` (lower-triangular,
   symmetrised before factorisation).
4. Transform: `ret_nodes = z_quad · L.T` shape `(K_total, n_ret)`,
   `ret_weights` shape `(K_total,)`.

Per-axis labels under Cholesky (with default `ret_names = (xr, xb)`):
- axis 0 (`K[0] = K_xr`) refines `z_0`, the pure xr direction (since `L`
  is lower-triangular).
- axis 1 (`K[1] = K_xb`) refines `z_1`, the **purified xb residual**
  after xr correlation has been orthogonalised away.

This is the **conditional** distribution `v_ret | v_state ~ N(0,
Sigma_r_cond)` — the `M · v_state` shift is added separately at
quadrature time (§8).

### 7.2 State-innovation quadrature (`get_state_quadrature`)

[`discretization.py:682-742`](../../lifecycle/discretization.py#L682-L742)
mirrors the return rule: tensor-product Gauss-Hermite on the
standardised `N(0, I)` then Cholesky-transformed to `N(0, Sigma_ss)`:

```
v_nodes = z_quad · L.T          where L L.T = Sigma_ss
```

Per-axis labels under Cholesky with state ordering
`(cy, spr, rtb, y_1)`:
- axis 0 = pure cy innovation (most-orthogonal axis — mean |ρ| = 0.17)
- axis 1 = mostly spr innovation
- axis 2 = rtb-purified ("inflation surprise" axis)
- axis 3 = y_1-purified residual

Refining `K[3]` (or its Lobatto Z) is the targeted lever for bond-return
integration accuracy because `M[xb, y_1] = -8.84` is the dominant
v-channel for bond returns
([`discretization.py:691-696`](../../lifecycle/discretization.py#L691-L696)).

### 7.3 Lobatto / prescribed-tail rule

[`quadrature_with_tails.py:68-225`](../../lifecycle/quadrature_with_tails.py#L68-L225)
implements `gauss_hermite_prescribed_tails(K, Z)` — a closed-form
modification of Gauss-Hermite that fixes two nodes at `±Z` while
optimally placing the `K-2` interior nodes for polynomial exactness on
the *full* (untruncated) Gaussian.

- Closed-form solutions for `K ∈ {3, 5, 7}` only.
- Polynomial exactness is `2K − 3` (vs `2K − 1` for pure Gauss-Hermite at
  the same `K`). The trade-off buys guaranteed extreme-tail coverage:
  `K=5, Z=4` puts a node at exactly 4σ, vs Hermite-`K=5`'s max node ≈
  2.86σ.
- Validity windows are enforced
  ([`quadrature_with_tails.py:96-101`](../../lifecycle/quadrature_with_tails.py#L96-L101))
  and reject Z below threshold with explicit error messages.

The motivation
([`quadrature_with_tails.py:21-26`](../../lifecycle/quadrature_with_tails.py#L21-L26)):
spurious arbitrages in finite quadrature happen when the discrete node
set fails to span the true return distribution support. Adding explicit
far-tail nodes closes that gap — the bond-tail discrete-free-lunch case
in particular.

### 7.4 Production discretization (canonical)

From [`configs/_canonical.py:60-85`](../../configs/_canonical.py#L60-L85):

```python
state_grid_sizes = (7, 7, 7, 7)         # 4D state grid (cy, spr, rtb, y_1)
state_grid_mode  = "cholesky"
state_n_stds     = (2.0, 2.25, 2.0, 2.25)    # u-space half-widths

n_ret_nodes_1d   = (5, 5)               # GH order per return axis
ret_lobatto_Z    = (7.0, 7.0)           # Lobatto on both xr and xb axes at Z=7σ

n_state_quad_nodes = (3, 5, 3, 5)       # GH per state-innov axis
state_lobatto_Z    = (None, 7.0, None, 7.0)  # Lobatto on spr-axis and y_1-axis only
```

So the production return quadrature is `5×5 = 25` joint nodes with
`±7σ` Lobatto tails on both axes; the state quadrature is
`3×5×3×5 = 225` joint nodes with `±7σ` Lobatto tails on the spr- and
y_1-purified axes (the strong drivers of `xb` via the `M` projection).

### 7.5 State-grid construction (interpolation grid, separate from
quadrature)

[`discretization.py:158-299`](../../lifecycle/discretization.py#L158-L299)
in `cholesky` mode (the canonical setting):

1. Solve the unconditional mean `mu_s = (I − Phi)⁻¹ · mu_intercept` and
   the stationary covariance `Sigma_z` via `solve_discrete_lyapunov`.
2. Cholesky-factor `Sigma_z = L · L.T`.
3. Per-axis grid `state_bracket_grids[d] = linspace(-n_stds_d, +n_stds_d,
   N_d)` in u-coordinates.
4. Tensor product to flat indices `state_indices`; transform
   `state_grid[i] = mu_s + L · u_i` for each grid index.
5. Stationary marginal probabilities derived from per-axis normal CDFs
   ([`discretization.py:239-246`](../../lifecycle/discretization.py#L239-L246)).
6. Bracket transform `b = L_inv · (s − mu_s)` is what the solver inverts
   on the fly to bracket arbitrary continuous state values onto the grid
   ([`solver.py:794-823`](../../lifecycle/solver.py#L794-L823)).

The Rouwenhorst Markov chain `Pi_state` is built but only consumed by
the simulator's stationary-init draw and a few diagnostic helpers; the
solver uses the **continuous quadrature `(v_nodes, v_weights)`** for
all integrations.

---

## §8. Solver-side construction of per-cell return tensors

### 8.1 `_build_step_log_returns` ([`solver.py:763-791`](../../lifecycle/solver.py#L763-L791))

For a fixed source state `state_grid[i_s]` and a fixed pre-computed
`s_next` (= next-period state at every state-quadrature node `k_v`), the
function returns three tensors of shape
`(n_state_quad, n_ret_quad)`:

```python
def _build_step_log_returns(state_grid_i, M_v_nodes, ret_nodes,
                             const_r, A_r, s_next, rtb_idx, xr_pos, xb_pos):
    base_mu_r = const_r + A_r @ state_grid_i              # (n_ret,)
    mu_r_per  = base_mu_r[None, :] + M_v_nodes            # (n_state_quad, n_ret)
    mu_xs = mu_r_per[:, xr_pos]
    mu_xb = mu_r_per[:, xb_pos]
    res_xs = ret_nodes[:, xr_pos]
    res_xb = ret_nodes[:, xb_pos]
    log_R_bill_kv = s_next[:, rtb_idx]                    # (n_state_quad,)
    log_R_bill = jnp.broadcast_to(log_R_bill_kv[:, None],
                                   (log_R_bill_kv.shape[0], n_ret_quad))
    log_x_s = mu_xs[:, None] + res_xs[None, :]
    log_x_b = mu_xb[:, None] + res_xb[None, :]
    return log_R_bill, log_x_s, log_x_b
```

Arithmetic, term by term:

```
base_mu_r[k]          = const_r[k] + (A_r @ s_t)[k]                       (n_ret)
mu_r_per[k_v, k]      = base_mu_r[k] + (M_v_nodes)[k_v, k]                (n_state_quad, n_ret)
                      = (Phi_0_ret − M Phi_0_state)[k]
                          + (Phi_21 − M Phi_11)[k] · s_t
                          + M[k] · v_state[k_v]
                      = E[r_{t+1}^k | s_t, v_state[k_v]]                  (textbook conditional mean)
log_x_s[k_v, k_r]     = mu_r_per[k_v, xr_pos] + ret_nodes[k_r, xr_pos]
log_x_b[k_v, k_r]     = mu_r_per[k_v, xb_pos] + ret_nodes[k_r, xb_pos]
log_R_bill[k_v, k_r]  = s_next[k_v, rtb_idx]   (broadcast over k_r)
```

So the realised log returns at quadrature node `(k_v, k_r)` are the
joint-Gaussian draw

```
log_x_s, log_x_b ~ N(M · v_state[k_v], Sigma_r_cond)   (the (k_r) draw)
log_R_bill        = (Phi_0_state + Phi_11 · s_t + v_state[k_v])[rtb_idx]
                    (deterministic given (s_t, k_v))
```

The `mu_r_per` formulation is verified element-by-element in
[`precompute._validate_state_quadrature()`](../../lifecycle/precompute.py#L491-L512)
to recover the unconditional return-mean exactly under the state
quadrature weights (max error `< 1e-10`).

### 8.2 `_build_step_state_brackets` ([`solver.py:794-824`](../../lifecycle/solver.py#L794-L824))

For the same fixed source state, this returns the multilinear bracketing
of `s_next[k_v]` onto the policy grid:

```python
s_next = Phi_0_state[None, :] + state_grid_i @ Phi_11.T + v_nodes
```

For each `k_v`:
1. `bracket_state_jax(s_next_kv, axis_grids, shift, L_inv)` transforms
   `s_next_kv` into the Cholesky-decorrelated u-coordinate
   `b = L_inv · (s_next_kv − mu_s)`, then brackets each axis onto its
   1D grid (`lo[d]`, `frac[d]`).
2. Multilinear weights `w[c] = prod_d (frac[d] if offset[c, d] else
   1 − frac[d])` for each of `2^n_state` corners.
3. Flat indices `j[c] = sum_d (lo[d] + offset[c, d]) · stride[d]`.

Returns `(s_next, j_corners, w_corners)` of shapes
`(n_state_quad, n_state)`, `(n_state_quad, 2^n_state)`,
`(n_state_quad, 2^n_state)`. The next-step value/policy lookup
then reads `c_next[:, j_corners, :]` and applies the multilinear
weights `w_corners` for interpolation.

### 8.3 Joint draw structure — independent or conditional?

**Conditional.** The state innovation `v_state[k_v]` is drawn first
(through `v_nodes` from `Sigma_ss`'s Cholesky), and the return residual
`ret_nodes[k_r]` is drawn from `Sigma_r_cond`'s Cholesky **plus the
mean shift `M · v_state[k_v]`**. This is exactly the joint Gaussian
draw decomposed via the law of total covariance:

```
v_state ~ N(0, Sigma_ss)
v_ret | v_state ~ N(M · v_state, Sigma_r_cond)
```

with marginal cov `Cov(v_ret) = M · Sigma_ss · M.T + Sigma_r_cond =
Sigma_rr`. The two Cholesky factors (`L_ss` for state, `L_cond` for
return-residual) together constitute the block-Cholesky of the full
6×6 `Omega`. **The cross-block covariance is fully consumed.**

The solver does not factor the full `(n_state + n_ret) × (n_state +
n_ret)` `Omega` directly — that approach would not allow the state grid
to be built around a clean per-axis Cholesky basis and would not let
the inner return integration share a Cholesky across all `(z, i_s)`
cells. The two-step decomposition gives the same draws with a structure
that matches the algorithmic decomposition (state grid → return
quadrature inside each grid cell).

Simulator does the same:
[`simulation.py:329-347`](../../lifecycle/simulation.py#L329-L347).
The `M @ v_s` shift is at line 337, the Cholesky-of-`Sigma_r_cond`
factor is `ret_factor`
([`simulation.py:97-104`](../../lifecycle/simulation.py#L97-L104)).

### 8.4 Where these tensors enter the FOC kernels

The terminal-age FOC at
[`solver.py:723-756`](../../lifecycle/solver.py#L723-L756) consumes the
three tensors plus `weight_kv_kr = outer(v_weights, ret_weights)` and
the `(σ²_xr, σ²_xb, σ_xrxb)` triple, and integrates the CCV `R_p` and
its gradient over the quadrature cloud. Same pattern at the working-age
and shifted-bequest kernels (referenced in
[`docs/CCV_RETURNS.md:222-226`](../CCV_RETURNS.md#L222-L226)).

The inf-horizon solver re-uses the same arithmetic at
[`inf_horizon_solver.py:142-170`](../../lifecycle/inf_horizon_solver.py#L142-L170)
(cold-start Markowitz) and
[`inf_horizon_solver.py:321-345`](../../lifecycle/inf_horizon_solver.py#L321-L345)
(stability proxy `β · E[R_p^{1−γ}]`).

---

## §9. The rtb-as-state semantics (CRITICAL)

### 9.1 What changed

Prior to the rtb-as-state migration (commits `b9c8b37`, `c17ebf5`,
`389ce08`, `dfdb390`):

- The return block was 3-vector `(rtb, xr, xb)`.
- `log_R_bill` was *drawn* from the return-block component at
  every return-quadrature node — i.e. `log_R_bill[k_r] =
  mu_rtb + ret_nodes[k_r, rtb_pos]`.
- `Sigma_rr` was 3×3.

After the migration (current code, this trace):

- The return block is 2-vector `(xr, xb)`.
- `log_R_bill` is read from the next-period state vector at the
  `rtb_index_in_state` position:
  `log_R_bill = s_next[k_v, rtb_idx]`
  ([`solver.py:785-788`](../../lifecycle/solver.py#L785-L788),
  [`simulation.py:333`](../../lifecycle/simulation.py#L333)).
- `Sigma_rr` is 2×2 (just `(xr, xb)`).

### 9.2 Migration commits (chronological)

```
b9c8b37  var: rtb-as-state migration, VAR layer
c17ebf5  model: rtb_index_in_state field + 4D state grid support
389ce08  solver: 4D state grid + log_R_bill from state_{t+1}[rtb_idx]
dfdb390  sim/diag/configs: 4D state plumbing + rtb-from-state log_R_bill
```

The docstring of
[`build_nominal_system1_var_config`](../../lifecycle/var.py#L381-L440)
(specifically lines 424–428) records the timeline:

> "Migration history: pre-2026-04-30 used (y_1, spr, cy) with
> state_indices=(0, 1, 2). 2026-04-30 to 2026-05-06 used (cy, spr, y_1)
> with state_indices=(2, 1, 0); rtb was a return-block variable. Post
> 2026-05-06 (this version) is the rtb-as-state migration: rtb joins the
> state block at position 2."

### 9.3 Mathematical implications

**(a) `log_R_bill` is now deterministic given `(s_t, v_state[k_v])`,
not stochastic given a return-quadrature draw.**

Concretely:
```
log_R_bill = s_next[k_v, rtb_idx]
           = (Phi_0_state + Phi_11 · s_t + v_state[k_v])[rtb_idx]
```

There is no `k_r` index in `log_R_bill_kv` — the same realisation is
broadcast across all return-quadrature nodes
([`solver.py:787-788`](../../lifecycle/solver.py#L787-L788)). The
randomness in `log_R_bill` lives entirely in the state-innovation
quadrature axis (which does have non-trivial weight on `rtb`-direction
nodes via the Cholesky of `Sigma_ss`).

**(b) The conditional return distribution is now
`p(xr, xb | s_t, s_{t+1})`, not `p(xr, xb | s_t)`.**

Because `s_{t+1}` is sampled by the state quadrature, conditioning on
`s_{t+1}` (i.e. on `v_state[k_v]`) gives a Gaussian residual via the
`(M, Sigma_r_cond)` decomposition. The downstream consumer (solver FOC)
sees a return distribution that depends on `(s_t, s_{t+1})` jointly.

**(c) `rtb` enters the FOC mechanics via two distinct channels.**

1. As `log_R_bill` in the CCV portfolio return formula (the riskless
   leg). This is the new rtb-as-state channel.
2. As a state-vector component that determines the **next-period state**
   for value-function lookup via the multilinear bracket on the policy
   grid. This is the standard state-vector channel.

Both channels are driven by the same `v_state[k_v]` draw, so they are
*not* independent — a positive `rtb` shock simultaneously raises
`log_R_bill` and shifts the next-period state to a high-rtb policy cell.

### 9.4 Consequences elsewhere in the code

- `Sigma_rr` is 2×2 in production; `pc.sigma2_xr/sigma2_xb/sigma_xrxb`
  read from the (xr, xb) sub-block.
- The dataset still has `rtb` as a column — the rtb-as-state migration
  changed the *partition*, not the dataset.
- The **iid System I**
  ([`build_iid_var_config`](../../lifecycle/var.py#L519-L577)) is the
  one configuration that does not lag-predict anything. Even there
  rtb lives in the **state block** because the solver/simulator
  unconditionally read `log_R_bill` from `s_next[rtb_idx]`. System I
  encodes "no predictability" by setting `Phi = 0` so rtb becomes an
  iid draw around its sample mean
  ([`var.py:519-541`](../../lifecycle/var.py#L519-L541)).

### 9.5 Departure from textbook CCV

In Campbell-Viceira (2002) ch. 4 / CCV w8566, the riskless rate
`r_{f, t+1}` is observed at time `t` (locked in at the start of the
period). Under rtb-as-state in this codebase, **the realised real bill
return** `rtb_{t+1}` is part of the time-`t+1` state — only its
*conditional mean given `s_t`* is known at time `t`. The realised
`log_R_bill` in `R_p` is the *ex post* real bill return, not the *ex
ante* time-`t` real rate.

This is consistent with the dataset definition of
`rtb[T] = log(1 + y_1[T-1]) - pi[T]` (§2.7) — it is the realised real
return on a *one-year* bill held over year `T`, with the inflation
draw `pi[T]` realised at the end of year `T`. The lag in `r_1[T-1]`
captures the "locked-in nominal yield, random ex-post inflation"
structure of a one-period nominal bill held over a horizon with
stochastic inflation.

**FLAG-D for theory review:**
- Is the CCV portfolio formula sound when `log_R_bill` is the *ex-post*
  realised real bill return rather than the *ex-ante* known real rate?
- Specifically: in the Jensen / Itô variance correction terms of CCV w8566
  eq. (10), `r_bill` appears additively but is implicitly treated as
  non-stochastic (no `σ²_rtb` term, no `σ_rtb,xr` cross-term in the
  vol-drag). Under rtb-as-state the quantity `log_R_bill` has variance
  `Sigma_ss[rtb, rtb] ≈ 3.24e-4` (annual std ≈ 1.8%) and non-trivial
  covariance with `(xr, xb)` via the cross-block. Whether this should
  contribute to the vol-drag is a theory question. The current code
  *does not* add an `α_bill² · σ²_rtb` or `α_s · σ_xr,rtb` etc. term —
  the bill leg is treated as non-stochastic in the variance correction
  even though its realisation is random.

---

## §10. Variable-name dictionary

| Paper / theory symbol | Code identifier | File:line | Definition / role |
|---|---|---|---|
| `s_t` (state vector) | `state_grid` row, `s_t` in solver | [`precompute.py:138`](../../lifecycle/precompute.py#L138) | Joint state vector in production order `(cy, spr, rtb, y_1)`; row `i` of `state_grid` |
| `n_state`, `n_ret` | `model.n_state`, `model.ret` | [`model.py:58-59`](../../lifecycle/model.py#L58-L59) | Production: `n_state = 4`, `n_ret = 2` |
| `Φ_0` (state drift) | `Phi_0_state` | [`var.py:83-84`](../../lifecycle/var.py#L83-L84), [appendix §5.1](RETURN_MODELLING_PARAMS_2026-05-07.md#51-state-block--phi_0_state-4-vector-in-state-row-order-cy-spr-rtb-y_1) | Additive constant in state transition |
| `Φ_11` (state transition) | `Phi_11` | [`var.py:64`](../../lifecycle/var.py#L64), [appendix §5.2](RETURN_MODELLING_PARAMS_2026-05-07.md#52-state-transition-phi_11-4x4-rowscols-both-in-state-row-order) | Multiplicative state→state |
| `A_r` (return loading) | `Phi_21`, also `A_r = pc.A_r` | [`var.py:65`](../../lifecycle/var.py#L65), [`precompute.py:292`](../../lifecycle/precompute.py#L292), [appendix §5.4](RETURN_MODELLING_PARAMS_2026-05-07.md#54-return-loading-phi_21--a_r-2x4-rows-xr-xb-cols-cy-spr-rtb-y_1) | State→return predictive loading |
| `c_r` (return drift) | `Phi_0_ret`, also `pc.const_r` | [`var.py:84`](../../lifecycle/var.py#L84), [`precompute.py:291`](../../lifecycle/precompute.py#L291), [appendix §5.3](RETURN_MODELLING_PARAMS_2026-05-07.md#53-return-intercepts-phi_0_ret-2-vector-xr-xb) | Return-equation intercept |
| `Σ_ss` (state innov cov) | `Sigma_ss` | [`var.py:69`](../../lifecycle/var.py#L69), [appendix §5.5](RETURN_MODELLING_PARAMS_2026-05-07.md#55-state-innovation-covariance-sigma_ss-4x4-in-state-row-order) | 4×4 |
| `Σ_rr` (return innov cov, unconditional) | `Sigma_rr` | [`var.py:70`](../../lifecycle/var.py#L70), [appendix §5.6](RETURN_MODELLING_PARAMS_2026-05-07.md#56-return-innovation-covariance-sigma_rr-2x2-unconditional) | 2×2 post rtb-as-state; **source of the production CCV vol-drag scalars** (FLAG-C) |
| `Σ_rs` (cross-block) | `Sigma_rs` | [`var.py:71`](../../lifecycle/var.py#L71), [appendix §5.7](RETURN_MODELLING_PARAMS_2026-05-07.md#57-cross-block-covariance-sigma_rs-2x4-return-rows--state-cols) | Cross covariance (return rows × state cols) |
| `M = Σ_rs Σ_ss⁻¹` | `M`, also `pc.M_v_nodes = v_nodes @ M.T` | [`var.py:74`](../../lifecycle/var.py#L74), [`precompute.py:293`](../../lifecycle/precompute.py#L293) | State-innov → return-mean projection |
| `Σ_r│s` (conditional return cov) | `Sigma_r_cond` | [`var.py:75`](../../lifecycle/var.py#L75), [appendix §5.9](RETURN_MODELLING_PARAMS_2026-05-07.md#59-conditional-return-covariance-sigma_r_cond--sigma_rr--m--sigma_sr-2x2) | Used as Cholesky basis for `ret_nodes` |
| `ε_t^state` (state innovation) | `v_state` (sim), `v_nodes` (solver quadrature) | [`simulation.py:329`](../../lifecycle/simulation.py#L329), [`precompute.py:149`](../../lifecycle/precompute.py#L149) | Discretised via Gauss-Hermite + Lobatto, Cholesky transform |
| `ε_t^r` (return innovation residual) | `ret_resid` (sim), `ret_nodes` (solver) | [`simulation.py:341`](../../lifecycle/simulation.py#L341), [`discretization.py:647`](../../lifecycle/discretization.py#L647) | Drawn from `N(0, Sigma_r_cond)` |
| `r_{f, t+1}` / `log R_bill` | `log_R_bill = s_next[:, rtb_idx]` | [`solver.py:785`](../../lifecycle/solver.py#L785), [`simulation.py:333`](../../lifecycle/simulation.py#L333) | rtb-as-state: realised log gross nominal bill return per state-quad node |
| `xr_{t+1}`, `xb_{t+1}` (excess log returns) | `log_x_s`, `log_x_b` | [`solver.py:789-790`](../../lifecycle/solver.py#L789-L790) | `(state_quad, ret_quad)` shape; conditional mean + residual |
| `α_s`, `α_b` (portfolio shares) | `alpha_s`, `alpha_b` | [`solver.py:694-716`](../../lifecycle/solver.py#L694-L716) | Stock and bond shares; `α_bill = 1 − α_s − α_b` |
| `r_p` (CCV log portfolio return) | `r_p` (local), `R_p = exp(r_p)` | [`solver.py:702-713`](../../lifecycle/solver.py#L702-L713) | The CCV w8566 eq. (10) formula |
| `σ²_xr`, `σ²_xb`, `σ_xrxb` | `pc.sigma2_xr`, `pc.sigma2_xb`, `pc.sigma_xrxb` | [`precompute.py:312-314`](../../lifecycle/precompute.py#L312-L314), [appendix §9](RETURN_MODELLING_PARAMS_2026-05-07.md#9-ccv-vol-drag-scalars-actually-consumed-by-the-solver) | Vol-drag scalars; **sourced from `Sigma_rr` post commit `f23ac83`** (FLAG-C) |
| Dataset variables | `y_1`, `spr`, `cy`, `rtb`, `xr`, `xb` | [`data/build_var_dataset.py:133-140`](../../data/build_var_dataset.py#L133-L140) | See §2 for algebraic definitions |

---

## §A. Naming caveats and FLAGs collected from the trace

### Naming caveats — internal nomenclature is overloaded

- **"System I" overloaded.**
  [`predictability_ablation.py:28-35`](../../lifecycle/predictability_ablation.py#L28-L35)
  uses "System I / II / III / IV" to label four progressively richer
  ablations: I = iid, II = (rtb, y_1), III = (rtb, spr, y_1), IV = full
  `(cy, spr, rtb, y_1)`. By contrast, the function name
  `build_nominal_system1_var_config` (§3, used as the production
  System IV) refers to "System 1" in the older
  [`data/var_specification.md`](../../data/var_specification.md) sense
  ("VAR System 1: Nominal Bond System"). **The production canonical
  config uses `PREDICTABILITY_SYSTEM = "IV"`**
  ([`configs/_canonical.py:19`](../../configs/_canonical.py#L19)). These
  refer to the same thing but the names look like they conflict.
- **`y_20` is AAA, not Treasury.** The variable named `y_20` in the
  dataset is the Moody's Aaa corporate bond yield, used as a long-yield
  proxy for the bond return CCV log-linear approximation. It is *not* a
  20-year Treasury yield (FLAG-B).
- **Documentation drift between dataset spec and code.**
  [`data/var_specification.md`](../../data/var_specification.md)
  describes a 5-variable VAR with quarterly→annual aggregation, GSW
  zero-coupon yield curve for the bond, FRED `TB3MS` (3-month bill) for
  the short rate, and Shiller `dp` (log dividend-price). The actual
  code uses a 6-variable VAR estimated **directly at annual frequency**,
  with AAA for the long bond, DGS1 (1-year Treasury) for the short, and
  `cy = -log(CAPE)` for valuation. The spec doc predates the current
  pipeline and should not be relied on for variable definitions
  (FLAG-A).

### FLAGs (theory-reviewer punch list)

**FLAG-A: dataset spec doc is stale.**
[`data/var_specification.md`](../../data/var_specification.md) describes
a different VAR (5-var, quarterly aggregated, GSW yield curve, dp ratio,
3-mo bill) than the production pipeline (6-var, native annual, AAA
corporate, CAPE-based `cy`, 1-yr Treasury). **Source of truth for
variable definitions is
[`data/build_var_dataset.py`](../../data/build_var_dataset.py)**, not
the spec doc. Do not use `var_specification.md` when checking the
mapping from raw series to model variables.

**FLAG-B: AAA used as the long yield.** The bond return uses Moody's
Aaa corporate bond yield (FRED `AAA`) as a 20-year par-bond yield via
the CCV constant-duration log-linear approximation. AAA carries credit
risk (~50–100 bp credit spread); a Treasury constant-maturity yield
(e.g. GSW SVENY10 or SVENY20, both available in
`data/Thesisdata/feds200628 (1).csv`) would be the default-free choice.
Theory reviewer should confirm whether the credit-spread contamination
of `xb` is acceptable for the modelled "default-free long bond" object.

**FLAG-C: vol-drag scalars sourced from `Sigma_rr` (unconditional),
contradicting `docs/CCV_RETURNS.md`.**
[`precompute.py:303-314`](../../lifecycle/precompute.py#L303-L314)
sources `(σ²_xr, σ²_xb, σ_xrxb)` from `Sigma_rr` (the unconditional
return-block covariance) per a May 2026 patch. The companion docs
([`docs/CCV_RETURNS.md:200-211, 459-461`](../CCV_RETURNS.md#L200-L211))
still document the pre-patch convention (`Sigma_r_cond`). The two
matrices differ by 11–93× in this calibration. Theory reviewer should
adjudicate which is the correct CCV w8566 eq. (10) input. See §6.7 for
detailed argument from the precompute comment.

**FLAG-D: rtb-as-state and the bill-leg variance.** Under rtb-as-state
(§9), the realised `log_R_bill` is part of the random next-period state,
yet the CCV vol-drag formula does not contain an `α_bill² · σ²_rtb`
term or any `σ_rtb,xr / σ_rtb,xb` cross-term. Whether this is the
correct "ex-post real bill, no bill-side variance correction"
specification or an oversight is a theory question. The dataset
construction (`rtb[T] = log(1 + y_1[T-1]) − pi[T]`, ex-post real
return on a one-period nominal bill) is consistent with the
"locked-in nominal, ex-post inflation surprise" interpretation, which
*could* justify treating the bill leg as non-stochastic in the
*α-decision-relevant* variance — but this should be confirmed against
the CCV derivation.

**FLAG-E: `_diag_euler_errors.py` reference card is stale.** Per
[`docs/CCV_RETURNS.md:272-290`](../CCV_RETURNS.md#L272-L290) the
diagnostic carries its own copy of the CCV formula and threads
`(ret_nodes, sigma2_xr, sigma2_xb, sigma_xrxb, use_ccv)` through five
helper functions. The reference card at
[`docs/CCV_RETURNS.md:459-466`](../CCV_RETURNS.md#L459-L466) still
documents a 3-element return block `(rtb, xr, xb)` and indexes
`sigma2_xr = Sigma_r_cond[1,1]`, etc. Post rtb-as-state these indices
no longer apply — `Sigma_r_cond` is 2×2 and indexed `[xr, xb]`. The
production code does the indexing correctly via name lookup
(`ret_names.index("xr")`); the doc just hasn't been updated. Cosmetic
flag rather than a theory issue.

End of trace.

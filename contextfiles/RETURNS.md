# RETURNS AND FINANCIAL STATE VARIABLES — VALIDATION DOCUMENT

**Purpose:** Consolidate the specification, data construction, estimation, and
validation status of the VAR(1) financial state system: the six variables,
their construction from raw data, and the CCV constrained VAR estimation that
feeds the lifecycle solver.

**Frequency:** The VAR is estimated directly at **annual frequency**. Raw data
(daily yields, monthly CPI, monthly Shiller) is resampled to end-of-December
values before constructing annual variables. No quarterly intermediate step.

**Code references:** `data/build_var_dataset.py` (raw data → annual `var_dataset.csv`),
`var.py` (VAR estimation, partition, hardcoded fallbacks),
`model.py` (stores partitioned annual parameters on `LifecyclePortfolioModel`),
`precompute.py` (builds state grid via Rouwenhorst, state quadrature arrays,
conditional return means),
`discretization.py` (Rouwenhorst multivariate, Gauss-Hermite state/return quadrature),
`solver.py` (bracket/trilinear interpolation, FOC with quadrature integration).

**Data references:** `data/var_dataset.csv` (clean annual dataset, 63 obs),
`HANDOFF_VAR_RESTRUCTURE_new.md` (full specification of the restructure),
`data/Thesisdata/` (raw source files).

---

## 0. Variable Definitions

The VAR has 6 variables at **annual frequency**, partitioned into 3 state
variables (known at decision time) and 3 return variables (uncertain):

| # | Symbol | Type | Definition | Source | Units |
|---|--------|------|-----------|--------|-------|
| 0 | `y_1` | State (level) | 1-year nominal Treasury yield, end-of-Dec value | DGS1 (FRED) | Annual decimal |
| 1 | `spr` | State (level) | Yield spread: `y_20 - y_1` (AAA minus 1yr) | AAA (FRED) - DGS1 | Annual decimal |
| 2 | `cy` | State (level) | Log earnings yield: `-log(CAPE)` | Shiller ie_data | Log ratio |
| 3 | `rtb` | Return | Real bill return: `log(1+y_1_t) - pi_{t+1}` | DGS1 + CPIAUCSL | Annual log return |
| 4 | `xr` | Return | Excess nominal stock return: `nominal_stock - log(1+y_1)` | Shiller P + D + DGS1 | Annual log excess |
| 5 | `xb` | Return | Excess nominal bond return: `r_bond - log(1+y_1)` | AAA + DGS1 | Annual log excess |

**Partition into state variables and returns:**

| Role | Variables | VAR indices | Within-block indices |
|------|-----------|-------------|---------------------|
| State variables (grid + quadrature) | `y_1`, `spr`, `cy` | 0, 1, 2 | 0, 1, 2 |
| Return variables (integrated out) | `rtb`, `xr`, `xb` | 3, 4, 5 | 0, 1, 2 |

**Key design choices:**
- **No riskless asset.** `rtb` is now a return variable (uncertain), not a state
  variable. The nominal bill yield `y_1` is known at decision time, but the real
  return `rtb = log(1+y_1) - pi` is uncertain because inflation is unknown.
- **All excess returns are nominal minus nominal** (CCV convention): `xr` and `xb`
  subtract `r_1 = log(1+y_1)`, the log nominal bill return. Inflation appears in
  exactly ONE variable (`rtb`).
- **Recovery identities:** `R_bill = exp(rtb)`, `R_stock = exp(rtb + xr)`,
  `R_bond = exp(rtb + xb)`. All three use the SAME quadrature node (joint draw).
- **`spr = y_20 - y_1`** rather than `y_20` directly: more orthogonal to `y_1`,
  better-conditioned estimation, more efficient Rouwenhorst grid. Matches CCV
  Table A specification.
- **`cy = -log(CAPE)`** rather than `dp`: cyclically-adjusted earnings yield is a
  stronger long-horizon equity predictor and reduces one-year earnings noise.
- **20-year Moody's AAA par bond** replaces the 10-year GSW zero-coupon. Bond
  return uses the CCV loglinear approximation. AAA includes a credit spread
  component (~100 bp) — documented caveat, consistent with CCV sample.

---

## 1. Variable Construction

### 1.1 State variables (levels, end of year t)

**y_1 — 1-year nominal Treasury yield:**
```
y_1[t] = DGS1(last trading day of Dec year t) / 100
```
DGS1 is the FRED 1-year constant maturity Treasury yield in % p.a. Dividing by
100 gives an annual decimal (e.g., 0.05 = 5%).

**spr — yield spread:**
```
spr[t] = y_20[t] - y_1[t]
y_20[t] = AAA(last observation in Dec year t) / 100
```
Moody's Seasoned AAA Corporate Bond Yield. The spread isolates the term premium
+ credit component orthogonal to the short rate.

**cy — log earnings yield (cyclically adjusted):**
```
cy[t] = -log(CAPE[t])
```
CAPE is from Shiller's online dataset (December value). `cy` is the log of the
inverse CAPE, i.e., the log earnings yield. Log form is used for Gaussian VAR
plausibility and consistency with the predictability literature.

### 1.2 Return variables (realized during year t+1)

**pi — annual log inflation (intermediate, not stored in dataset):**
```
pi[t+1] = log(CPI_Dec[t+1] / CPI_Dec[t])
```
CPIAUCSL (seasonally adjusted) from FRED.

**rtb — real bill return:**
```
rtb[t+1] = log(1 + y_1[t]) - pi[t+1]
```
The log nominal bill return `r_1 = log(1+y_1)` is known at t; realized inflation
`pi` is not. This makes `rtb` uncertain despite `y_1` being a state variable.

**r_bond — nominal bond return (CCV loglinear approximation):**
```
Y[t]      = y_1[t] + spr[t]           # AAA yield in decimal = y_20
D[t]      = (1 - (1+Y[t])^{-20}) / (1 - (1+Y[t])^{-1})    # par bond duration
y_log[t]  = log(1 + Y[t])             # log gross yield

r_bond[t+1] = D[t] * y_log[t] - (D[t] - 1) * y_log[t+1]
```
Buy at end of year t at yield Y[t], sell at end of year t+1 at yield Y[t+1].
The CCV approximation treats y(19,t+1) ≈ y(20,t+1) (flat yield curve locally).

**xr — excess nominal stock return:**
```
nom_ret_m[t] = log((P[t] + D[t]/12) / P[t-1])     # monthly nominal return from Shiller P, D
nominal_stock[T] = sum of nom_ret_m for Jan-Dec of year T   (12 months required)
xr[T]            = nominal_stock[T] - log(1 + y_1[T-1])
```
Uses Shiller's nominal price (P) and dividend (D) columns directly — no CPI involved.
This avoids the mismatch between Shiller's non-seasonally-adjusted CPI and FRED's SA CPIAUCSL.

**xb — excess nominal bond return:**
```
xb[t+1] = r_bond[t+1] - log(1 + y_1[t])
```

### 1.3 Timing convention

```
  decision date t                                   realisation date t+1
  ─────────────┬─────────────────────────────────────┬──────────────
  KNOWN at t                       UNKNOWN at t (becomes known at t+1)

  y_1[t]   ──►  yield locked in for year t+1
  spr[t]   ──►  (state)
  cy[t]    ──►  (state)
                                   pi[t+1]
                                   rtb[t+1]   = log(1+y_1[t]) - pi[t+1]
                                   xr[t+1]    = nominal_stock[t+1] - log(1+y_1[t])
                                   xb[t+1]    = r_bond[t+1]        - log(1+y_1[t])
```

In the dataset, **the row labelled year T** contains:
- levels (`y_1, spr, cy`) at end of year T
- returns (`rtb, xr, xb`) realised during year T (t+1 of diagram, with t = T-1)

The VAR `z_T = c + Phi @ z_{T-1} + eps_T` then has `y_1[T-1]` on the RHS of
the return equations, which is mechanically correct: the bill yield that
determines `rtb[T]` is `y_1[T-1]`.

### 1.4 Dataset assembly

Only calendar years with complete data are included. DGS1 starts 1962, so the
first complete return year is 1963 (needs y_1[1962] for the shift). Final sample:
**1963–2025, T=63 annual observations**.

Output: `data/var_dataset.csv` with columns `[year, y_1, spr, cy, rtb, xr, xb]`.

---

## 2. Data Sources and Sample

| File | Content | Original source | Coverage |
|------|---------|----------------|----------|
| `DGS1.csv` | 1-year constant maturity Treasury yield (% p.a.) | FRED | 1962–2026 |
| `AAA.csv` | Moody's Seasoned AAA Corporate Bond Yield (% p.a.) | FRED | 1919–2026 |
| `CPIAUCSL.csv` | CPI-U seasonally adjusted (level) | FRED | 1947–2026 |
| `ie_data.xls` | S&P 500 P (nominal price), D (nominal dividends), CAPE | Shiller online data | 1871–2026 |

**Annual VAR sample:** 1963–2025 (63 annual observations). Binding constraint
is DGS1 (starts 1962; first return year is 1963 after the shift).

**Raw data files no longer needed:** TB3MS.csv (replaced by DGS1),
`feds200628 (1).csv` (GSW yield curve — replaced by AAA).

---

## 3. VAR Estimation

### 3.1 CCV Constrained VAR(1)

The VAR is estimated using the CCV (2003, §4.2) constrained estimator:

1. Compute `z_bar = sample mean` of the full dataset (all T rows).
2. Demean: `z_tilde_t = z_t - z_bar`.
3. Regress `z_tilde_{t+1}` on `z_tilde_t` **without intercept**, using only
   state columns as regressors (lagged returns excluded).
4. Recover `const = (I - Phi) @ z_bar`.

This guarantees `(I - Phi)^{-1} @ const = z_bar = sample_mean` **exactly**,
eliminating the grid-centering drift that plagued the unconstrained estimator
(old Section 6.11).

**Restriction:** Only lagged state variables (`y_1`, `spr`, `cy`) enter each
equation. Return-lag columns of Phi are zero by construction:
`Phi[:, 3] = Phi[:, 4] = Phi[:, 5] = 0`.

**Implementation:** `estimate_var1_from_csv()` in `var.py` with
`state_indices=[0,1,2]`. Wrapper: `build_nominal_system1_var_config()`.

### 3.2 State-return partition

`partition_var()` in `var.py` splits the estimated VAR into:

- **State sub-VAR:** Phi_11 (3×3), Phi_0_state (3,), Sigma_ss (3×3)
- **Return equations:** Phi_21 (3×3), Phi_0_ret (3,)
- **Conditioning matrix:** M = Sigma_rs @ Sigma_ss^{-1} (3×3)
- **Residual return covariance:** Sigma_r_cond = Sigma_rr - M @ Sigma_sr (3×3)

The conditional return distribution given state s_t and state innovation v^s is:

```
E[r_{t+1} | s_t, v^s_{t+1}] = Phi_0_ret + Phi_21 @ s_t + M @ v^s_{t+1}
                              = const_r  + A_r @ s_t    + M @ v^s_{t+1}
Var[r_{t+1} | s_t, v^s_{t+1}] = Sigma_r_cond   (constant)
```

where `const_r = Phi_0_ret` and `A_r = Phi_21` (since v_nodes are zero-mean
innovations, not next-state values).

### 3.3 Direct annual estimation (no compounding)

The VAR is estimated directly on annual data (T=62 regression observations).
Returns are calendar-year quantities; levels are end-of-December values. No
quarterly-to-annual compounding step is needed.

---

## 4. Key Parameter Values

### 4.1 Annual (estimated on 1963–2025, T=63, CCV constrained)

```
z_bar = [+0.04849, +0.01992, -2.99287, +0.00913, +0.05547, +0.01427]
         [ y_1      spr       cy        rtb       xr        xb      ]

Annual means: y_1=4.85%  spr=1.99%  cy=-2.99  rtb=0.91%  xr=5.55%  xb=1.43%

Phi_11 diagonal (state persistence, annual):
  y_1: 0.670   spr: 0.872   cy: 0.919

Phi_21 (return equations, 3×3):
         L.y_1       L.spr       L.cy
  rtb   +1.079      +0.857      -0.034
  xr    -1.801      -0.523      +0.107
  xb    +1.462      +4.492      -0.055

Innovation std devs (annual):
  y_1: 1.57%   spr: 1.12%   cy: 16.69%   rtb: 1.98%   xr: 15.89%   xb: 7.63%

Equation R²:
  y_1: 0.784   spr: 0.521   cy: 0.873   rtb: 0.518   xr: 0.059   xb: 0.322

M matrix (3×3, return × state):
         y_1         spr         cy
  rtb   -0.936      -0.621      -0.036
  xr    +0.075      -0.959      -0.933
  xb    -8.718      -8.514      -0.005

Key derived quantities:
  M[xb, y_1]  = -8.72    (100bp ↑y_1 → -8.7pp xb)
  M[xb, spr]  = -8.51    (100bp ↑spr → -8.5pp xb)
  M[xr, cy]   = -0.93    (mechanical CAPE/price relationship)
  M[rtb, y_1] = -0.94    (Fisher effect: ↑y_1 → ↑expected pi → ↓rtb)

Variance explained by state conditioning:
  rtb: 39.1%   xr: 96.2%   xb: 91.2%

Residual return std (after conditioning):
  rtb: 1.54%   xr: 3.10%   xb: 2.26%
```

---

## 5. State Transition: Gauss-Hermite Quadrature

### 5.1 Overview

The solver integrates over next-period state innovations using **Gauss-Hermite
quadrature** rather than a discrete Markov transition matrix. The state grid
(built via Rouwenhorst) is retained for policy function storage and
interpolation, but transitions between grid points are continuous.

### 5.2 Mathematical structure

The state sub-VAR is:

```
s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s_{t+1},    v^s ~ N(0, Sigma_ss)
```

The quadrature approximates expectations over v^s:

```
E[f(v^s)] ≈ sum_{k=1}^{K^3} w_k * f(v_k)
```

where (v_k, w_k) are tensor-product Gauss-Hermite nodes and weights,
transformed from standard normal via Cholesky: v_k = L_ss @ xi_k.

**Conditional return mean** at state s_i with innovation v_k:

```
mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ v_k
       = (const_r + A_r @ s_i) + M_v_nodes[k]
```

where `const_r = Phi_0_ret`, `A_r = Phi_21`, `M_v_nodes = v_nodes @ M.T`
(precomputed). `mu_r_k` has **3 components**: [rtb, xr, xb].

### 5.3 Return quadrature (3-dimensional)

In addition to state quadrature, the solver integrates over return residuals
using Gauss-Hermite quadrature on `Sigma_r_cond` (3×3). With K nodes per
dimension, this produces K^3 return quadrature nodes. At K=2, that's 8 nodes;
at K=3, 27 nodes.

The gross real returns at each return quadrature node k_r are:

```
R_bill[k_r]  = exp(mu_rtb  + ret_nodes[k_r, 0])
R_stock[k_r] = exp(mu_rtb  + ret_nodes[k_r, 0] + mu_xr + ret_nodes[k_r, 1])
R_bond[k_r]  = exp(mu_rtb  + ret_nodes[k_r, 0] + mu_xb + ret_nodes[k_r, 2])
```

**Critical invariant:** all three returns for a single period use the SAME
return quadrature node k_r — they are components of one joint draw.

### 5.4 Precomputed arrays

| Array | Shape | Content |
|-------|-------|---------|
| `v_nodes` | (K_s^3, 3) | State innovation nodes |
| `v_weights` | (K_s^3,) | State tensor-product weights, sum to 1 |
| `M_v_nodes` | (K_s^3, 3) | `v_nodes @ M.T` — return contribution per state node |
| `const_r` | (3,) | `Phi_0_ret` |
| `A_r` | (3, 3) | `Phi_21` |
| `exp_ret_bill` | (K_r^3,) | `exp(ret_nodes[:, 0])` |
| `exp_ret_stock` | (K_r^3,) | `exp(ret_nodes[:, 1])` |
| `exp_ret_bond` | (K_r^3,) | `exp(ret_nodes[:, 2])` |
| `ret_nodes` | (K_r^3, 3) | Return residual nodes |
| `ret_weights` | (K_r^3,) | Return tensor-product weights |

### 5.5 Simulation

In simulation, state transitions are fully continuous:

```
z_std ~ N(0, I_3)
v^s = L_ss @ z_std
s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s
mu_r = const_r + A_r @ s_t + M @ v^s      # 3-vector [rtb, xr, xb]
```

Return residuals are drawn from N(0, Sigma_r_cond) via its Cholesky factor.

---

## 6. Validation

### 6.1 Data construction checks

- [x] **V1: rtb identity holds exactly** — `rtb[T] + pi[T] = log(1+y_1[T-1])`
      max |resid| = 3.0e-18 across all 63 observations.
- [x] **V2: bond return identity holds exactly** — reconstructed xb from
      y_20 and CCV duration formula matches dataset xb to max |resid| = 0.0.
- [x] **V3: real stock identity** — `rtb + xr = nominal_stock - pi`
      max |resid| = 5.6e-17. Machine precision because nominal_stock is
      constructed from Shiller P+D (CPI-free), and pi uses FRED CPIAUCSL
      consistently with rtb.
- [x] **V+: real bond recovery** — `rtb + xb = r_bond - pi` max |resid| = 2.8e-17.
- [x] **CPI series is seasonally adjusted** — CPIAUCSL (not CPIAUCNS) used.
- [x] **No interior NaN gaps** — all six series have zero interior NaNs after
      dropping first year (lost to shift). 63 clean rows.
- [x] **y_1 range sanity** — min 0.10% (2020 COVID low), max 13.86% (1981 Volcker).
- [x] **spr range sanity** — mean +1.99%, min -1.41% (inverted curve),
      max +4.89%. Negative spreads occur during tight monetary policy.
- [x] **cy range sanity** — mean -2.99, min -3.79 (2025), max -2.06 (1974).
      Consistent with CAPE range ~8 to ~44.
- [x] **rtb range sanity** — mean +0.91%, min -6.83%, max +8.78%.
- [x] **xr range sanity** — mean +5.55% (equity premium), std 16.0%.
      Min -53.1% (2008), max +25.9%.
- [x] **xb range sanity** — mean +1.43%, std 9.0%. Min -23.2%, max +17.2%.
- [x] **AAA bond duration** — mean 11.76 years for 20-year par bond at mean
      AAA yield of 6.84%. Duration rises as yields fall (currently ~14 at 5%).

### 6.2 Cross-validation

Not applicable. This is a new dataset with new variable definitions. Historical
cross-validation against the old 5-variable quarterly dataset is not meaningful.

### 6.3 VAR estimation checks

- [x] **V5: sample-mean restriction** — `(I-Phi)^{-1} @ const = sample_mean`
      max |diff| = 1.1e-16. Exact by construction (CCV constrained estimator).
- [x] **State sub-VAR is stationary** — Phi_11 eigenvalues: [0.936, 0.775, 0.775].
      Max |eigenvalue| = 0.936, well below 1.0.
- [x] **Restriction correctly imposed** — `||Phi[:, 3:6]|| = 0.0` exactly.
      Return-lag columns zero by construction.
- [x] **Annual magnitudes sensible** — z_bar: y_1=4.85%, spr=1.99%, cy=-2.99,
      rtb=0.91%, xr=5.55%, xb=1.43%. All economically reasonable.
- [x] **Bond duration mechanism intact** — M[xb, y_1] = -8.72, M[xb, spr] = -8.51.
      A 100bp rise in y_1 reduces xb by ~8.7pp; a 100bp rise in spr reduces xb
      by ~8.5pp. Both channels reflect the ~12-year duration of the 20-year AAA
      par bond at historical average yields.
- [x] **State conditioning explains most bond return variance** — 91.2% of xb
      innovation variance explained by state conditioning. Residual std = 2.26%.
- [x] **Stock return conditioning very strong** — 96.2% of xr variance explained
      (driven by M[xr, cy] = -0.93: mechanical CAPE/price relationship). Residual
      std = 3.10%.
- [x] **rtb conditioning moderate** — 39.1% of rtb variance explained. Residual
      std = 1.54%. Expected: inflation has substantial unpredictable component.
- [x] **Residual correlations correct** — xb/y_1 = -0.71 (yield up → bond loss),
      xr/cy = -0.98 (mechanical CAPE identity), rtb/y_1 = -0.49 (Fisher effect:
      higher y_1 → higher expected inflation → lower real return).

### 6.4 Annual estimation characteristics

- [x] **Sufficient observations** — T=63 annual observations, k=3 state
      predictors (no intercept in demeaned regression). T/k = 21.
- [x] **V4: dual-regression identity** — rtb + pi regression slope sums:
      y_1 = 0.951 (expected ~1; gap = y_1²/2 ≈ 0.05 at mean 5% yield),
      spr = -3.1e-3, cy = -5.0e-4. The spr/cy sums are not machine-zero
      because rtb + pi = log(1+y_1) ≠ y_1 exactly; the quadratic residual
      leaks through correlations. This is correct behavior — not a timing bug.
- [x] **State persistence reasonable** — Phi_11 diagonal: y_1=0.670,
      spr=0.872, cy=0.919. All below 1.0.
- [x] **z_bar matches sample means exactly** — by construction (constrained
      estimator pins z_bar = sample_mean, verified to 1.1e-16).
- [x] **Hardcoded fallbacks updated** — `_Z_BAR`, `_PHI`, `_OMEGA` in `var.py`
      now contain full-precision estimates from 1963–2025 (T=63). Verified
      max |live - hardcoded| < 1e-15 for Phi, z_bar, const; < 1e-15 for Omega.

### 6.5 State quadrature checks

- [x] **Weights sum to 1** — state quad err = 1.1e-16, return quad err = 0.0.
- [x] **All weights positive** — verified at K=3 (state) and K=2 (return).
- [x] **Mean zero** — state: max|wmean| = 1.4e-18. Return: max|wmean| = 0.0.
- [x] **Covariance exact** — state: max|cov - Sigma_ss| = 1.0e-17.
      Return: max|cov - Sigma_r_cond| = 3.9e-18.
- [x] **Node count correct** — state: 27 (K=3, 3³). Return: 8 (K=2, 2³).
      Return quadrature is now 3-dimensional (rtb, xr, xb).
- [x] **M_v_nodes consistency** — `M_v_nodes = v_nodes @ M.T`: max err = 0.0.
- [x] **Return formula identity** — `const_r = Phi_0_ret` and `A_r = Phi_21`:
      both max err = 0.0. Unconditional return mean at 20 grid points: max
      err = 8.3e-17.
- [x] **exp_ret precomputation** — `exp_ret_bill = exp(ret_nodes[:,0])`,
      `exp_ret_stock = exp(ret_nodes[:,1])`, `exp_ret_bond = exp(ret_nodes[:,2])`:
      all max err = 0.0.
- [x] **All quadrature arrays C-contiguous** — v_nodes, v_weights, M_v_nodes,
      const_r, A_r, exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_nodes,
      ret_weights, state_grid, wealth_grid: all C_CONTIGUOUS float64.

### 6.6 Solver integration checks

- [x] **Terminal condition** — VERIFIED. Terminal solver replaced with `@njit`
      2D Newton (no scipy dependency). Uses state quadrature consistent with
      rest of solver. Verified by 12 automated tests (`test_terminal_correctness.py`):
      - CRRA homogeneity: c/W constant across wealth (CV = 3e-16)
      - z-independence: policy identical across income states (spread = 0)
      - KKT conditions at solution (violation = 6e-13)
      - Newton matches 201x201 brute-force grid search (moment err < 1e-6)
      - Moment E[R^{1-gamma}] finite and positive for all states
      - Consumption in (0, W) with c/W in [0.08, 0.20]
      - Analytic FOC matches finite-difference gradient (rel err = 5e-8)
      - Terminal return construction bit-identical to retirement (err = 6e-16)
      - Quadrature mean reproduces Phi_0_ret + Phi_21 @ s_i (err = 8e-17)
      - Quadrature covariance reproduces full Omega_rr (err = 1e-17)
      - Return correlation structure exact (bill-stock-bond correlations match)
      - State-return cross-covariance Cov(v^s, e_r) = Sigma_sr (err = 9e-16)
      - Two-layer vs single-layer quadrature agree (rel err = 2e-4)
- [ ] **Full lifecycle solve** — no NaN/Inf. (Skipped — too slow for validation.)

### 6.7 Quadrature convergence (K) and cross-validation

(To be run after solver integration is verified with new parameters.)

### 6.8 Boundary, stress, and performance checks

- [ ] **Boundary hit rates** — per-dimension and overall, at target grid size.
- [ ] **Corner-state FOC finite** — FOC at min/max grid corners.
- [ ] **Tiny savings (s=1e-8)** — FOC returns finite values.

### 6.9 Economic mechanism checks

(To be run after full solver integration.)

### 6.10 Duration matching / immunization

- [ ] **Analytical immunizing bond share** — computed from annuity factor
      duration and M[xb, ...] at unconditional mean.
- [ ] **Bond share direction correct** — higher yields → more bonds.

### 6.11 z_bar / grid centering

**Status: FIXED by CCV constrained estimator.**

The grid is now centered on `z_bar = sample_mean` by construction. The
constrained estimator (Section 3.1) pins `(I-Phi)^{-1} @ const = z_bar`
exactly, eliminating the drift between implied stationary mean and sample
mean that affected the old unconstrained estimator.

### 6.12 Open items (quadrature)

- [ ] **Determinism test** — 2 full solves with bit-exact comparison.
- [ ] **Grid convergence** — policies at 5^3 vs 7^3 vs 9^3.
- [x] **Terminal age uses Pi_state, not quadrature** — FIXED. Terminal
      solver now uses v_nodes/v_weights state quadrature, consistent with
      retirement and working-age solvers.
- [x] **Terminal solver uses scipy** — FIXED. Replaced `solve_portfolio_2d_terminal_exact`
      (scipy trust-constr + SLSQP) and `solve_portfolio_unconstrained_terminal_exact`
      (scipy trust-ncg + BFGS) with `@njit` Newton solvers. Removed scipy.optimize
      import from solver.py. CRRA concavity guarantees Newton convergence without
      multi-start or trust regions.

### 6.13 Bugs found and fixed

- [x] **const_r/A_r algebra error** (`precompute.py`) — fixed to
      `const_r = Phi_0_ret`, `A_r = Phi_21`.
- [x] **z_bar grid centering drift** — fixed by CCV constrained estimator.
      Old unconstrained OLS produced implied z_bar diverging from sample mean
      by up to 1.3pp for persistent variables. Now exact by construction.
- [x] **r_bill treated as riskless** — fixed by moving rtb to return block.
      All three returns (bill, stock, bond) now uncertain and integrated via
      3D return quadrature.

### 6.14 Open items (data/estimation)

- [ ] **TIPS system (System 2)** — `feds200805.csv` not yet in `data/Thesisdata/`.
- [ ] **Ken French data for exact CCV replication** — current `xr` uses Shiller
      P+D nominal returns (CPI-free). CCV originally use CRSP via Ken French.

### 6.15 Known caveats

- **AAA credit spread.** The spread `spr = AAA - y_1` includes ~100bp of credit
  risk. The annuity factor uses `y_20 = y_1 + spr` as the discount rate, which
  overstates the rate for a riskless income stream by ~100bp. With b_bar=10,
  the annuity is biased down by ~5%. Consistent with CCV (who use AAA for
  the same sample period).
- **CPI consistency (resolved).** Nominal stock returns are constructed directly
  from Shiller's P and D columns (no CPI involved). Inflation enters only via
  FRED CPIAUCSL in the rtb definition. This avoids the Shiller NSA-CPI vs
  FRED SA-CPIAUCSL mismatch entirely. V3 identity holds to machine precision.

# RETURNS AND FINANCIAL STATE VARIABLES — VALIDATION DOCUMENT

**Purpose:** Consolidate the specification, data construction, estimation, and
validation status of the VAR(1) financial state system: the five variables,
their construction from raw data, and the restricted VAR estimation that feeds
the lifecycle solver.

**Frequency:** The VAR is estimated directly at **annual frequency**. Quarterly
raw data is aggregated to annual (returns summed, levels at Q4) before estimation.
No quarterly-to-annual compounding is needed.

**Code references:** `data/data_construction.ipynb` (raw data → annual `var_dataset.csv`),
`var.py` (VAR estimation, partition, hardcoded fallbacks),
`model.py` (stores partitioned annual parameters on `LifecyclePortfolioModel`),
`precompute.py` (builds state grid via Rouwenhorst, state quadrature arrays,
conditional return means),
`discretization.py` (Rouwenhorst multivariate, Gauss-Hermite state quadrature),
`solver.py` (bracket/trilinear interpolation, FOC with quadrature integration).

**Data references:** `data/var_dataset.csv` (clean annual dataset, 64 obs),
`data/var_specification.md` (variable construction specification),
`data/Thesisdata/` (raw source files).

---

## 0. Variable Definitions

The VAR state vector has 5 variables at **annual frequency**:

| # | Symbol | Definition | Source | Units |
|---|--------|-----------|--------|-------|
| 0 | `rtb` | Ex-post real bill rate: sum of 4 quarterly log returns | TB3MS (FRED) + CPIAUCSL (FRED) | Annual log return |
| 1 | `xr` | Excess real stock return: sum of 4 quarterly log excess returns | Shiller RTRP | Annual log excess return |
| 2 | `xb` | Excess nominal bond return: sum of 4 quarterly log excess returns | GSW feds200628 (Fed) | Annual log excess return |
| 3 | `y_nom` | 10-year nominal yield (continuously compounded), Q4 value | GSW SVENY10 / NSS params | Annual yield decimal (SVENY10/100) |
| 4 | `dp` | Log dividend-price ratio (trailing 12-month dividends / price), Q4 value | Shiller ie_data | Log ratio (level) |

**y_nom units:** `y_nom` is stored as `SVENY10 / 100` (annual decimal, e.g.
0.05 = 5% p.a.). This is consistent with all other variables being in annual
units. The annuity pricing code uses `y_nom` directly as the annual discount rate.

**Partition into state variables and returns:**

| Role | Variables | VAR indices | Within-state indices |
|------|-----------|-------------|---------------------|
| State variables (grid + quadrature) | `rtb`, `y_nom`, `dp` | 0, 3, 4 | 0, 1, 2 |
| Return variables (integrated out) | `xr`, `xb` | 1, 2 | — |

---

## 1. Variable Construction

### 1.1 Real bill rate (`rtb`)

```
r_bill_t = TB3MS_{t-1} / 400       rate SET at end of previous quarter
pi_q_t   = log(CPI_t / CPI_{t-1})  log inflation DURING quarter t
rtb_t    = r_bill_t - pi_q_t       ex-post real return
```

**Timing:** `r_bill` uses the lagged T-bill rate (known at the start of the
quarter). CPI is end-of-quarter (seasonally adjusted, CPIAUCSL). This makes
`rtb` an ex-post realised return, not an ex-ante expectation.

### 1.2 Excess stock return (`xr`)

```
xr_t = log(RTRP_t / RTRP_{t-1}) - rtb_t
```

`RTRP` is Shiller's Real Total Return Price index — a cumulative real total
return index (dividends reinvested, deflated by CPI). The log difference is
therefore already a real return; subtracting `rtb` gives the excess real return.

**Note:** RTRP proxies for the CRSP value-weighted market portfolio. For exact
CCV replication, replace with Ken French monthly factors (Mkt-RF + RF).

### 1.3 Excess nominal bond return (`xb`)

```
r_bond_t = -9.75 * y(9.75)_t / 100  +  10 * y(10)_{t-1} / 100
xb_t     = r_bond_t - r_bill_t
```

The bond return is the quarterly log return on a 10-year zero-coupon nominal
bond: buy at end of quarter t-1 at yield y(10), sell one quarter later as a
9.75-year bond at yield y(9.75). Both yields are in % p.a. (continuously
compounded), so dividing by 100 gives log prices.

**y(9.75) from NSS formula.** The GSW dataset provides Nelson-Siegel-Svensson
parameters (BETA0–BETA3, TAU1, TAU2) daily. The 9.75-year yield is evaluated as:

```
y(n) = β₀ + β₁·((1-e^{-n/τ₁})/(n/τ₁))
     + β₂·((1-e^{-n/τ₁})/(n/τ₁) - e^{-n/τ₁})
     + β₃·((1-e^{-n/τ₂})/(n/τ₂) - e^{-n/τ₂})
```

Before 1980, TAU2 is missing and BETA3 = 0, reducing to 3-parameter
Nelson-Siegel. The same formula is used for y(10), which extends coverage
back to 1961Q2 (the SVENY10 column only starts 1971Q3).

### 1.4 Nominal yield (`y_nom`)

```
y_nom_t = y(10)_t / 100
```

SVENY10 is in % p.a.; dividing by 100 gives an annual decimal (e.g. 0.05 = 5%).
This enters the model directly as the annuity pricing yield for bequest utility.

### 1.5 Dividend-price ratio (`dp`)

```
dp_t = log(D_t / P_t)
```

`D` is trailing 12-month dividends, `P` is the S&P 500 price level, both from
Shiller's online dataset. This is a log level, not a return.

### 1.6 Quarterly resampling and annual aggregation

All raw series are first resampled to **end-of-quarter** dates: last observation
in March, June, September, December. The GSW daily data uses `resample("QE-DEC").last()`.

The quarterly variables are then aggregated to **annual frequency**:

| Variable type | Aggregation rule | Examples |
|---------------|------------------|----------|
| Returns | Sum of 4 quarterly log returns (Q1+Q2+Q3+Q4) | `rtb`, `xr`, `xb` |
| Levels | Q4 (end-of-year) value | `y_nom`, `dp` |

Only calendar years with all 4 quarters present are included. The first full
year is **1962** (quarterly data starts 1961Q3, so 1962 is the first year with
Q1–Q4 available).

---

## 2. Data Sources and Sample

| File | Content | Original source | Coverage |
|------|---------|----------------|----------|
| `TB3MS.csv` | 3-month T-bill rate (% p.a.) | FRED | 1934–2026 |
| `CPIAUCSL.csv` | CPI-U seasonally adjusted | FRED | 1947–2026 |
| `ie_data.xls` | S&P 500 P, D, RTRP | Shiller online data | 1871–2025 |
| `feds200628 (1).csv` | GSW nominal yield curve (NSS params + SVENY) | Federal Reserve | 1961–2026 |

**Quarterly intermediate sample:** 1961Q3 – 2026Q1 (259 quarters). Binding
constraint is the GSW yield curve (NSS parameters start 1961-06-14).

**Annual VAR sample:** 1962 – 2025 (64 annual observations). Calendar years
with all 4 quarters of clean quarterly data. This is the sample used for VAR
estimation.

**Previous approach:** Estimated at quarterly frequency (258 obs) and compounded
to annual via `annualize_var_config()`. Now replaced by direct annual estimation.

---

## 3. VAR Estimation

### 3.1 Restricted VAR(1)

The VAR is estimated by equation-by-equation OLS with the restriction that
lagged return variables are excluded from all equations:

```
z_{t+1} = c + Phi @ z_t + eps_{t+1}

Restriction: Phi[:, return_cols] = 0    (columns 1 and 2)
```

This means only lagged state variables (`rtb`, `y_nom`, `dp`) predict all
five variables. Return variables (`xr`, `xb`) enter no equations as predictors.
The residual covariance Omega is computed from the restricted residuals
(i.e. residuals of the restricted regression where only state columns
appear in X).

**Implementation:** `estimate_restricted_var1_from_csv()` in `var.py`. The
restriction is imposed structurally (only state columns appear in X), not
via post-estimation zeroing. R² is computed per equation.

### 3.2 State-return partition

`partition_var()` in `var.py` splits the estimated VAR into:

- **State sub-VAR:** Phi_11 (3×3), Phi_0_state (3,), Sigma_ss (3×3)
- **Return equations:** Phi_21 (2×3), Phi_0_ret (2,)
- **Conditioning matrix:** M = Sigma_rs @ Sigma_ss^{-1} (2×3)
- **Residual return covariance:** Sigma_r_cond = Sigma_rr - M @ Sigma_sr (2×2)

The conditional return distribution given state transition s_i → s_j is:

```
E[r_{t+1} | s_t=i, s_{t+1}=j] = const + A @ s_i + M @ s_j
Var[r_{t+1} | s_t=i, s_{t+1}=j] = Sigma_r_cond   (constant)
```

### 3.3 Direct annual estimation (no compounding)

The VAR is estimated directly on annual data (T=63). Returns are calendar-year
sums of quarterly log returns; levels are Q4 values. This avoids the
quarterly→annual compounding step (`annualize_var_config()` in `var.py` is
retained for reference but no longer used in the pipeline).

**Trade-off:** 63 annual observations vs 258 quarterly observations means less
statistical power, but avoids temporal aggregation assumptions and is standard
in the lifecycle portfolio choice literature (CCV, Campbell & Viceira 2004).

---

## 4. Key Parameter Values

### 4.1 Annual (estimated on 1962–2025, T=63)

```
z_bar = [+0.0008, +0.0501, +0.0216, +0.0450, -3.9236]
         [  rtb      xr       xb     y_nom      dp   ]

Annual means: rtb=+0.08%  xr=+5.01%  xb=+2.16%  y_nom=4.50% p.a.

Phi_11 diagonal (state persistence, annual):
  rtb:   0.432    (moderate persistence)
  y_nom: 0.871    (high persistence)
  dp:    0.874    (high persistence)

Innovation std devs (annual):
  rtb:   1.65%
  xr:   15.96%
  xb:   10.49%
  y_nom: 1.02%
  dp:    16.82%

Equation R²:
  rtb: 0.612   xr: 0.077   xb: 0.138   y_nom: 0.871   dp: 0.860

Key derived quantities:
  M[xb, y_nom] = -10.23    (bond duration: 1pp rise in y_nom → -10.2pp in xb)
  M[xr, dp]    = -0.89     (stock return predictability from D/P)

Variance explained by state conditioning:
  xr: 89.0%
  xb: 97.8%
```

---

## 5. State Transition: Gauss-Hermite Quadrature

### 5.1 Overview

The solver integrates over next-period state innovations using **Gauss-Hermite
quadrature** rather than a discrete Markov transition matrix. The state grid
(built via Rouwenhorst) is retained for policy function storage and
interpolation, but transitions between grid points are continuous.

**Why quadrature?** A discrete Markov chain (Pi_state) with N^3 states requires
an N^3 × N^3 transition matrix. At 7^3 = 343 states, Pi_state is 343 × 343
(~118k entries per row of the FOC sum). Gauss-Hermite quadrature with K=3
nodes per dimension uses only K^3 = 27 integration points, independent of
grid size. This decouples grid resolution from integration cost.

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
transformed from standard normal via Cholesky: v_k = L_ss @ xi_k, where
L_ss = chol(Sigma_ss) and xi_k are standard GH nodes in R^3.

**Conditional return mean** at state s_i with innovation v_k:

```
mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ v_k
       = (const_r + A_r @ s_i) + M_v_nodes[k]
```

where `const_r = Phi_0_ret`, `A_r = Phi_21`, `M_v_nodes = v_nodes @ M.T`
(precomputed).

### 5.3 Policy lookup via trilinear interpolation

For each quadrature node k, the next state is:

```
s_next = Phi_0_state + Phi_11 @ s_i + v_k
```

This generally falls between grid points. The policy at s_next is obtained
by trilinear interpolation across the 8 corners of the enclosing grid cell,
using `bracket_state_3d` to find cell indices and fractional positions.

For working-age periods, Catmull-Rom cubic interpolation in the z (log income)
dimension is nested inside the trilinear state interpolation.

### 5.4 Precomputed arrays

| Array | Shape | Content |
|-------|-------|---------|
| `v_nodes` | (K^3, 3) | Innovation nodes in original coordinates |
| `v_weights` | (K^3,) | Tensor-product weights, sum to 1 |
| `M_v_nodes` | (K^3, 2) | `v_nodes @ M.T` — return contribution per node |
| `const_r` | (2,) | `Phi_0_ret` |
| `A_r` | (2, 3) | `Phi_21` |
| `exp_ret_stock` | (n_ret,) | `exp(ret_nodes[:, 0])` |
| `exp_ret_bond` | (n_ret,) | `exp(ret_nodes[:, 1])` |

### 5.5 Simulation

In simulation, state transitions are fully continuous:

```
z_std ~ N(0, I_3)
v^s = L_ss @ z_std
s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s
mu_xr = const_r + A_r @ s_t + M @ v^s
```

Policy lookup uses nearest grid point (no interpolation in simulation).

---

## 6. Validation

### 6.1 Data construction checks

- [x] **NSS y(10) matches SVENY10** — on the 1971–2026 overlap (220 quarters),
      NSS-computed y(10) matches the SVENY10 column to max 0.00005 pct pts and
      mean 0.00002. The NSS formula is exact (same computation GSW uses to
      produce SVENY10).
- [x] **NSS handles pre-1980 Nelson-Siegel gracefully** — before 1980, TAU2 is
      missing and BETA3 = 0 in the GSW file. The `nss_yield()` function falls
      back to 3-parameter Nelson-Siegel. No NaN gaps in the computed yields
      from 1961Q2 onward.
- [x] **Bond return formula correct** — manual verification for 2020Q1 (COVID
      yield crash): y(10) dropped from 1.96% to 0.79%, bond return = +12.1%.
      `p_sell - p_buy = -9.75 * 0.7703/100 + 10 * 1.9611/100 = 0.121001`,
      matching the series value exactly.
- [x] **r_bill timing correct** — `r_bill` uses TB3MS from the previous quarter
      (the rate known at the start of the quarter). Verified at 2020Q1
      (`r_bill = 1.54%/400 = 0.00385`, TB3MS from 2019Q4), 2008Q4, 2000Q2.
- [x] **pi_q formula correct** — `log(CPI_t / CPI_{t-1})` manually verified at
      2020Q1 (mild deflation: -0.21%) and 2022Q2 (high inflation: +2.50%).
- [x] **Identities hold exactly** — `rtb = r_bill - pi_q` and `xb = r_bond - r_bill`
      hold to max absolute error 0.0 across all observations.
- [x] **CPI series is seasonally adjusted** — `CPIAUCSL` (not `CPIAUCNS`)
      used, avoiding artificial quarterly seasonality in `rtb`. Standard choice
      for portfolio choice models (CCV, Campbell & Viceira).
- [x] **No interior NaN gaps** — all five series have zero interior NaNs between
      their first and last valid observations.
- [x] **y_nom range sanity** — min 0.93% (0.0093) at 2020 (COVID low), max
      13.51% (0.1351) at 1981 (Volcker peak). Both are correct historical
      extremes. y_nom is in annual decimal: SVENY10 / 100.
- [x] **dp range sanity** — min -4.4564 at 2025 (recent low D/P),
      max -2.9248 at 1974 (highest D/P in sample). Correct for annual
      dataset (1962–2025).

### 6.2 Cross-validation (historical, now superseded)

The quarterly-era dataset was previously cross-validated against an older
`var/var_dataset.csv` file (now deleted). Those checks confirmed
consistency at the quarterly level. After the switch to annual frequency,
the quarterly cross-validation no longer applies. The annual dataset is
validated by the sanity checks in Sections 5.1, 5.3, and 5.4.

### 6.3 VAR estimation checks

- [x] **State sub-VAR is stationary** — max eigenvalue of Phi_11 (annual) =
      0.933, well below 1.0.
- [x] **Restriction correctly imposed** — Phi[:, 1] = Phi[:, 2] = 0 exactly
      (return-lag columns). `||Phi_12|| = 0`, `||Phi_22|| = 0`.
- [x] **Annual magnitudes sensible** — mean xr = 5.01% (expect 5–7% equity
      premium), std xr = 16.2% (expect 15–20%). Mean xb = 2.16% (expect 1–3%),
      std xb = 10.9% (expect 8–12%).
- [x] **Direct annual vs compounded quarterly comparison** — Phi_11 diagonal
      within ~0.04 of compounded quarterly estimates; Omega_ss diagonal ratios
      within 0.95–1.05. Differences are expected (direct estimation captures
      within-year dynamics differently from compounding).
- [x] **Bond duration mechanism intact** — `M[xb, y_nom] = -10.23` (annual).
      A 100bp (0.01) rise in y_nom reduces annual xb by ~10.2 percentage points.
      Economically: the modified duration of a 10-year zero-coupon bond is
      ~9.75. With y_nom in annual decimal (SVENY10/100), M directly reflects
      bond duration without scaling artifacts.
- [x] **State conditioning explains most bond return variance** — 97.8% of xb
      innovation variance explained by state conditioning (driven by M[xb, y_nom]).
- [x] **Stock return conditioning weaker but substantial** — 89.0% of xr
      variance explained. The residual std (~5.3% annually) represents
      idiosyncratic stock market risk not captured by the state variables.
- [x] **Residual correlations correct** — xb/y_nom = -0.987 (mechanical bond
      pricing), xr/dp = -0.941 (mechanical D/P identity). Both correct signs
      and expected magnitudes.

### 6.4 Annual estimation characteristics

The direct annual VAR on 1962–2025 (T=63) has these properties:

- [x] **Sufficient observations** — T=63 annual observations with k=3 state
      predictors + intercept (4 parameters per equation). T/k ≈ 16, adequate
      for OLS estimation.
- [x] **State persistence reasonable** — Phi diagonal: rtb=0.43, y_nom=0.87,
      dp=0.87. Comparable to CCV Table 2 estimates.
- [x] **z_bar(rtb) near zero** — +0.08% p.a. Consistent with the full post-war
      real bill rate being close to zero on average.
- [x] **z_bar(xr) ≈ 5%** — the equity premium over the full 1962–2025 sample.
- [x] **M[xb, y_nom] ≈ -10.2** — close to the 10-year ZCB modified duration
      of 9.75. With y_nom in annual decimal, M has a clean economic
      interpretation.
- [x] **Hardcoded fallbacks updated** — `_Z_BAR`, `_PHI`, `_OMEGA` in `var.py`
      now reflect the annual estimates from 1962–2025.

### 6.5 State quadrature checks

- [x] **Weights sum to 1** — verified for K=1..5, error < 1e-14.
- [x] **All weights positive** — verified for K=2,3,4.
- [x] **Mean zero** — `sum_k w_k v_k = 0` to < 1e-13 for K=2,3,4.
- [x] **Covariance exact** — `sum_k w_k v_k v_k' = Sigma_ss` to < 1e-12
      for K=2,3,4.
- [x] **Third moments zero** — max |sum_k w_k v_ki v_kj v_kl| < 1e-11 at K=3.
      Consistent with Gaussian symmetry.
- [x] **Node count correct** — K^3 nodes for 3-dimensional state (verified K=1..4).
- [x] **M_v_nodes consistency** — `M_v_nodes = v_nodes @ M.T` to < 1e-14.
- [x] **Return formula identity** — `const_r + A_r @ s_i + M @ v = Phi_0_ret + Phi_21 @ s_i + M @ v`
      verified at 100 random (s_i, v) points, error < 1e-13.
- [x] **Unconditional return mean (P7)** — `sum_k w_k mu_r_k = Phi_0_ret + Phi_21 @ s_i`
      at all grid points, error < 1e-12.
- [x] **Conditional state mean (P4)** — `sum_k w_k (Phi_0 + Phi_11 s_i + v_k) = Phi_0 + Phi_11 s_i`
      verified, error < 1e-12.
- [x] **Conditional state covariance (P5)** — `sum_k w_k v_k v_k' = Sigma_ss`
      verified at 5 grid points, error < 1e-12.
- [x] **State-return innovation covariance (P8)** — `sum_k w_k v_k (M v_k)' = Sigma_rs'`
      error < 1e-12.
- [x] **Next-state/return covariance (P9)** — `Sigma_ss @ M' = Sigma_rs'` error < 1e-12.
- [x] **Flat index ordering** — `j = i0*N1*N2 + i1*N2 + i2` matches `np.ndindex`
      row-major for both square (5,5,5) and non-square (5,7,3) grids.
- [x] **exp_ret precomputation** — `exp_ret_stock[k] = exp(ret_nodes[k,0])` exact.
- [x] **All quadrature arrays C-contiguous** — v_nodes, v_weights, M_v_nodes,
      const_r, A_r, exp_ret_stock, exp_ret_bond, state_grid, ret_nodes,
      ret_weights, wealth_grid all C_CONTIGUOUS with dtype float64.
- [x] **Numba nopython compilation** — `bracket_state_3d` and `_interp_z_wealth`
      compile in nopython mode without TypingError.
- [x] **Bracket helper correct** — exact grid points, midpoint fractions,
      and out-of-bounds clamping all verified.
- [x] **Trilinear exact for linear functions** — interpolation of
      `f(s) = a + b's` recovers exact values at 100 random interior points,
      error < 1e-12.

### 6.6 Solver integration checks

- [x] **Terminal condition** — finite, positive consumption; stock share in [0,1].
- [x] **Quad vs Markov retirement (1 period)** — max relative error < 10%,
      mean < 5%. Finite and positive.
- [x] **Quad vs Markov working-age (1 period)** — max relative error < 10%,
      mean < 5%. Finite and positive.
- [x] **Full lifecycle solve** — no NaN/Inf in C, S, B. C non-negative.
      Portfolio shares satisfy 0 <= alpha_s, alpha_b and alpha_s + alpha_b <= 1.
- [x] **Wealth monotonicity** — consumption increasing in wealth, < 5% violations.
- [x] **Zero Newton failures** — 3,368,750 calls, 100% convergence.
      Worst FOC residual 5.14e-07, RMS 9.97e-09.
- [x] **Zero monotonicity violations** — EGM monotonicity fully preserved.
- [ ] **Deterministic** — not yet verified (test written, not yet run).
- [x] **Stock share declines with age** — verified at median state/wealth.

### 6.7 Quadrature convergence (K) and cross-validation

- [x] **FOC-level convergence** — euler, foc_s, foc_b all converge monotonically:
      |diff K=3→4| < |diff K=2→3| across 5 test points.
- [x] **K=3 vs K=5 euler < 0.02%** — relative difference 1.2e-4. K=3 is
      sufficient for the integration.
- [x] **K=1 FOC inaccurate** — foc_s at K=1 is ~2× the converged value.
      The variance of the integration matters for portfolio decisions even
      though it barely affects the euler sum level.
- [x] **Policy-level K convergence** — 5 periods solved at K=2,3,4.
      Consumption diff shrinks 21% (K=2→3 to K=3→4), stock share diff
      shrinks 51%. K=3 vs K=4 consumption relative diff 0.26%.
- [x] **Monte Carlo cross-check** — FOC at median state vs 30k MC draws from
      N(0, Sigma_ss). Euler rel diff 0.06%, FOC_s/FOC_b normalized diff < 0.04%.
      GH K=3 (27 nodes) reproduces MC to within sampling noise.

### 6.8 Boundary, stress, and performance checks

- [x] **Boundary hit rates** — rtb: 20.8%, y_nom: 13.3%, dp: 13.5%.
      Any dimension: 39.2%. All below thresholds (25% per dim, 50% overall).
      At 5^3 grid with K=3 quadrature.
- [x] **Corner-state FOC finite** — FOC evaluated at both min and max grid
      corners with mid-wealth portfolio. All values finite, euler > 0.
- [x] **Tiny savings (s=1e-8)** — FOC returns finite values, no NaN.
- [x] **Timing: quad 3.7× faster than Markov at 7^3** — retirement period:
      Markov 3.74s vs quad 1.00s. K=3 quad uses 27 nodes vs Markov's 343
      next-states.

### 6.9 Economic mechanism checks

- [x] **State sensitivity correct** — at age 30, median z and wealth:
      rtb: higher → more stocks (+0.30).
      y_nom: higher → zero stocks (-1.00, corner solution).
      dp: higher → 100% stocks (+1.00, corner solution).
- [x] **Portfolio regime breakdown** — ~80% corner solutions (expected with
      constrained optimization on coarse grid). ~2–3% interior solutions.
- [x] **Age profile** — stock share ~58% at 22, declining to ~43% mid-career.
      Bill allocation near zero when young, rising with age.

### 6.10 Duration matching / immunization

- [x] **Analytical immunizing bond share = 0.137** — modified duration of A(y_nom)
      at the unconditional mean (y_nom=0.91%) is 5.38 years.
      `alpha_b_immunize = D_mod / |M[xb, y_nom]| = 5.38 / 39.35 = 0.137`.
      This is the allocation that makes W/A(y_nom) insensitive to yield shocks.
- [x] **Solver exceeds immunizing floor** — bond shares at median state, high
      wealth range from 0.26 (age 59, 40 periods to terminal) to 0.34 (age 98,
      1 period to terminal). The +0.13 to +0.21 excess over the immunizing
      allocation is consistent with the 2.16% p.a. bond risk premium driving
      additional demand.
- [x] **Bond share increases near terminal** — bond allocation rises from 0.26
      at age 59 to 0.34 at age 98, as bequest utility (which creates the
      liability) becomes dominant near terminal age.
- [x] **y_nom grid sensitivity shows correct economic direction** — at
      near-terminal age, bond share varies from 0.000 at low y_nom to 1.000
      at high y_nom. High y_nom → high bond risk premium → agent loads bonds.
      Low y_nom → expected yield reversion / capital loss → agent sheds bonds.
      Corner solutions (0% and 100%) are expected with constrained optimization
      on a coarse 5-point y_nom grid.
- [x] **Quadrature preserves the M[xb, y_nom] channel** — the conditional return
      formula `mu_r = const_r + A_r @ s + M @ v` correctly transmits yield
      innovations to bond returns, enabling duration matching. The analytical
      immunizing calculation and the solver's bond demand are consistent in
      direction and magnitude.

### 6.11 z_bar / grid centering problem (OPEN — affects all results)

**Status: BUG IDENTIFIED, NOT YET FIXED.**

The Rouwenhorst state grid is centered on `z_bar = (I - Phi)^{-1} @ const`,
the VAR's implied stationary mean. For the 1962–2025 annual sample, z_bar
diverges significantly from the sample mean:

| Variable | Sample mean | z_bar (VAR) | Difference |
|----------|-------------|-------------|------------|
| rtb      | +0.70%      | +0.08%      | -0.62%     |
| y_nom    | 5.80%       | 4.50%       | -1.30%     |
| dp       | -3.66       | -3.92       | -0.27      |

**Root cause:** y_nom and dp exhibit secular trends over the 62-year sample
(y_nom: 14% → 4%; dp: -3.0 → -4.5). The VAR's implied stationary mean
is pulled away from the sample average by these trends operating through
the cross-equation dynamics. The rtb equation has a large coefficient on
y_nom (+0.553), so the y_nom trend propagates into a depressed rtb
equilibrium. This is not a coding bug — the OLS estimation is correct
(mean fitted rtb = 0.685% matches the data). It is a property of
`(I - Phi)^{-1} @ const` when the data is not stationary.

**Why this matters:** The grid center determines the "typical" state the
agent expects to inhabit over a lifetime. With the grid centered at
rtb = 0.08%, the agent's median-state bill return is an order of magnitude
below the historical average (0.70%). This systematically makes bills look
unattractive relative to stocks (excess 5.0%) and bonds (excess 2.2%),
producing near-zero bill allocation across most of the lifecycle.

**What is NOT affected:** The VAR coefficients (Phi, Sigma), the
conditional return formula, the quadrature integration, and the solver
FOC are all correct. The Phi_0 intercepts and the transition dynamics
faithfully reproduce the OLS fit. At any given state grid point, the
conditional expectations are right.

**What IS affected:** The grid placement and therefore which states
the agent visits and how much probability mass sits where. The grid
wastes resolution on unrealistic low-rtb / low-y_nom / low-dp states
and has fewer points covering the historically typical region.

**Fix options under consideration:**
1. Center grid on sample means instead of z_bar
2. Demean data before VAR estimation; use sample means as grid centers
3. Use the Campbell-Viceira convention where the grid is centered on
   sample means and the VAR intercept is adjusted accordingly

### 6.12 Open items (quadrature)

- [ ] **Determinism test** — 2 full solves with bit-exact comparison. Low
      priority: likely to show benign floating-point ordering noise from
      `prange` thread scheduling, not actual bugs.
- [ ] **Grid convergence** — policies at 5^3 vs 7^3 vs 9^3 (not yet implemented).
- [ ] **Terminal age uses Pi_state, not quadrature** — `solve_terminal_age()`
      (`solver.py:973`) uses the discrete Markov transition matrix `Pi_state`
      for state transitions, while all other ages (both working and retirement)
      use Gauss-Hermite quadrature over `N(0, Sigma_ss)`. This is an asymmetry:
      the terminal-age portfolio allocation is computed under the independence-
      Rouwenhorst approximation (which ignores cross-correlations in state
      innovations) while every other age uses the exact covariance structure.
      Impact is likely small (terminal age has minimal remaining horizon so
      portfolio choice matters less), but should be unified for consistency.
      Fix: rewrite `solve_terminal_age` to use `get_state_quadrature` nodes
      instead of `Pi_state` rows, mirroring the retirement FOC approach.

### 6.13 Bugs found and fixed

- [x] **const_r/A_r algebra error** (`precompute.py:168-172`) — original
      `const_r = Phi_0_ret - M @ Phi_0_state`, `A_r = Phi_21 - M @ Phi_11`
      assumed v_nodes were next-state values, not zero-mean innovations.
      Produced incorrect conditional return means. Fixed to
      `const_r = Phi_0_ret`, `A_r = Phi_21`.
- [x] **regenerate_savings_grid keyword** (`solver.py:3446`) — called with
      `n_points=n_s_points` but method expects positional arg. Fixed.
- [x] **run_lifecycle_solver argument order** (`solver.py:3413`) — `n_s_points`
      came before `solver_config`, so positional calls assigned SolverConfig
      to wrong parameter. Fixed by swapping order.
- [x] **Farmer-Toda removal** — `_validate_conditional_returns` in precompute.py
      blocked (7,7,7) grids with a RuntimeError. This check was specific to the
      old independence-Rouwenhorst Markov approach; the quadrature solver does
      not use Pi_state for transitions. Removed the validation, the farmer_toda
      discretization code, and related config fields (`state_discretization_method`,
      `consistency_tol_warn`, `consistency_tol_error`). Old code archived in
      `archive/`.

### 6.14 Cleared as correctly implemented

The following components were audited end-to-end during the bill-allocation
investigation and are confirmed correct:

1. **Data construction** — rtb = TB3MS(t-1)/400 - log(CPI_t/CPI_{t-1}),
   annual = sum of 4 quarterly log returns. Sample mean 0.70%, range
   [-6.9%, +7.2%]. Timing, signs, and magnitudes all verified.

2. **VAR estimation** — restricted OLS (only state lags as regressors)
   is mechanically correct. Fitted values reproduce the sample mean.
   Phi, const, and Omega match direct OLS verification.

3. **Partition (Phi_0, Phi_11, Phi_21, M, Sigma)** — all derived
   quantities are algebraically correct given the estimated Phi and Omega.

4. **State quadrature** — Gauss-Hermite nodes reproduce mean (0 to 1e-13),
   covariance (Sigma_ss to 1e-12), and third moments (0 to 1e-11).
   M_v_nodes = v_nodes @ M.T is exact. All 27 nodes include non-zero
   rtb innovations.

5. **Conditional return formula** — `mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ v_k`
   is correct. At any given state, the expected excess returns match
   the VAR's conditional predictions.

6. **Solver FOC and timing** — R_bill = exp(rtb) at the current state
   (known at decision time). R_stock = R_bill * exp(mu_xr + residual).
   Portfolio return R_p = alpha_s*R_s + alpha_b*R_b + (1-alpha_s-alpha_b)*R_bill.
   FOC: E[mu(c_next) * (R_k - R_bill)] = 0. Follows CCV convention.

7. **Solver convergence** — zero Newton failures across 3.4M calls,
   worst FOC residual 5.1e-7. EGM monotonicity fully preserved.

**The problem is isolated to grid centering (Section 6.11).** Everything
downstream of the grid placement is correct.

### 6.15 Open items (data/estimation)

- [x] ~~**Hardcoded fallback parameters in var.py**~~ — updated to annual
      frequency estimates (1962–2025, T=63). Old quarterly + compounded-annual
      constants removed.
- [ ] **TIPS system (System 2)** — `feds200805.csv` not yet in `data/Thesisdata/`.
      The `build_tips_system2_var_config()` function exists but the data
      pipeline for `xtips` and `y_real` has not been rebuilt in the new notebook.
- [ ] **Ken French data for exact CCV replication** — current `xr` uses Shiller
      RTRP as a proxy. For published results, consider replacing with the
      Fama/French monthly factors (Mkt-RF + RF).

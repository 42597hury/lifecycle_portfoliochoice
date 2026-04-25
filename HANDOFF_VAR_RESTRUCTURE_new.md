# HANDOFF: VAR Restructure — No Riskless Asset

## Problem

The current implementation treats the real bill return (`rtb`) as riskless — known at
decision time with zero variance. This is economically incoherent: `rtb = nominal_rate -
inflation`, and while the nominal rate is known in advance, inflation is not. CCV (2003)
are explicit: "the nominal return on a nominal bill is riskless, but the real return is
not because it is subject to short-term inflation risk."

The fix splits the short rate into a **nominal yield** (state variable, known) and a
**real return** (uncertain, integrated over). This also switches the bond from a 10-year
GSW zero-coupon to a 20-year Moody's AAA par yield, using the CCV loglinear return
approximation, and reparametrises the state vector so that **all three state variables
have a yield interpretation** (`y_1`, `spr`, `cy`). This last change is for estimation
quality and interpretability, not for the no-riskless-bill fix itself.

## New VAR Specification

**Old (5 variables):**
```
State:   rtb(0),  y_nom(3), dp(4)     — 3 state variables
Returns: xr(1),   xb(2)               — 2 return variables
```

**New (6 variables):**
```
State:   y_1(0),  spr(1),   cy(2)     — 3 state variables (known at end of year t)
Returns: rtb(3),  xr(4),    xb(5)     — 3 return variables (uncertain, realized year t+1)
```

| # | Name | Type | Definition | Source |
|---|------|------|-----------|--------|
| 0 | `y_1` | State (level) | 1-year nominal Treasury yield, Q4 value / 100 | DGS1 (FRED) |
| 1 | `spr` | State (level) | Yield spread: `y_20 - y_1`, Q4 value | AAA - DGS1 |
| 2 | `cy` | State (level) | Log Shiller earnings yield: `log(E10/P) = -log(CAPE)`, Q4 value | Shiller ie_data.xls |
| 3 | `rtb` | Return | Real bill return: `log(1+y_1_t) - pi_{t+1}` | DGS1 + CPIAUCSL |
| 4 | `xr` | Return | Excess nominal stock return over nominal bill: `nominal_stock - log(1+y_1)` | Shiller RTRP + CPIAUCSL + DGS1 |
| 5 | `xb` | Return | Excess nominal bond return over nominal bill: `r_bond - log(1+y_1)` | AAA + DGS1 |

Key points:
- `spr = y_20 - y_1`: yield spread between 20-year AAA and 1-year Treasury. More orthogonal
  to `y_1` than `y_20` itself, better-conditioned estimation, more efficient Rouwenhorst grid,
  and more mean-reverting (lower persistence). Matches CCV Table A specification.
- `y_20` is recovered as `y_1 + spr` wherever needed (bond return formula, annuity pricing).
- `cy = -log(CAPE) = log(E10/P)`, where `E10` is Shiller's trailing 10-year average real
  S&P 500 earnings and `P` is the end-of-December S&P 500 price. The log form (rather
  than `1/CAPE`) is used because (a) the VAR's homoskedastic Gaussian innovations are
  far more plausible in logs than levels, (b) the predictability literature regresses
  on `log(E10/P)` (Campbell-Shiller 1988, Campbell-Vuolteenaho 2004), and (c) it matches
  the log form of the other state variables. Cyclically-adjusted earnings (10-year
  smoothed) reduce one-year accounting noise in the numerator and give a stronger
  long-horizon equity-return forecaster than `dp`. Units are "% per year" — a yield.
- `rtb_{t+1} = log(1+y_1_t) - pi_{t+1}`: log nominal bill return (known at t) minus realized log inflation (unknown at t).
- All excess returns are **nominal minus nominal** (CCV convention): `xr = nominal_stock - r_1`, `xb = r_bond_nominal - r_1`, where `r_1 = log(1+y_1)` is the log nominal bill return.
- Inflation appears in exactly ONE variable (`rtb`). The solver recovers real returns via `exp(rtb + xr)` and `exp(rtb + xb)`.
- `nominal_stock = log(RTRP_{t+1}/RTRP_t) + pi_{t+1}` (Shiller's real RTRP + inflation added back).
- `r_1 = log(1 + y_1)`: the log gross nominal bill return. Used as the subtracted leg in all excess returns.

## Recovery identities the solver relies on

After the VAR draws joint `(rtb, xr, xb)`, the solver reconstructs **gross real
returns** as:

```
R_bill_real  = exp(rtb)
R_stock_real = exp(rtb + xr)        # because rtb + xr = nominal_stock - pi = real_stock
R_bond_real  = exp(rtb + xb)        # because rtb + xb = r_bond_nominal  - pi = real_bond
```

These three identities are what makes the nominal-minus-nominal convention clean:
inflation enters exactly once (`rtb`), and adding `rtb` to any excess return strips
the bill leg, leaving the real gross return.

**Critical invariant.** All three of `rtb`, `xr`, `xb` for a single decision period
must come from the **same quadrature node** — not three independent draws. They are
the three components of one joint outcome of the period-t+1 innovation. See "Solver
invariant" in §4 below.

## Bond Return: CCV Loglinear Approximation

Replace the exact ZCB formula with the CCV (2003) / Campbell, Lo & MacKinlay (1997) Ch.10
loglinear approximation for par/coupon bonds:

```
r_bond_{t+1} ≈ D_t * y_t - (D_t - 1) * y_{t+1}
```

where `y_t = log(1 + Y_t)` is the log gross yield, `Y_t` is the AAA yield in decimal
(i.e., `Y_t = y_1 + spr` recovered from the state), and duration is:

```
D_t = (1 - (1 + Y_t)^{-n}) / (1 - (1 + Y_t)^{-1})
```

with `n = 20`. CCV approximate `y_{n-1,t+1} ≈ y_{n,t+1}` (flat yield curve locally).

All data is at **annual frequency** — no quarterly intermediate step.

**AAA caveat.** AAA mixes credit and duration risk. CCV's annual sample uses Shiller's
long yield series for 1890–1953 and updates with Moody's AAA thereafter (CCV 2003, §4.1).
Our 1962–present sample is in the AAA-update period, so this is exactly the series
CCV would use for this window. Document the credit component as a known caveat.

**AAA and the annuity factor.** The spread `spr = AAA - y_1` includes the AAA credit
spread, so `y_20 = y_1 + spr` is a corporate rate, not a Treasury rate. When used to
discount the retirement annuity (a riskless income stream), this overstates the discount
rate and understates the present value of the annuity. With AAA spreads of ~100 bp and
b_bar = 10, the annuity factor is biased down by ~5%. This is a known approximation:
fixing it would require a separate Treasury long-yield series (e.g., DGS20, which starts
only in 1993). For now, document the bias and keep the single-spread design for
consistency with the bond return.

## Data Construction Changes (`data/data_construction.ipynb`)

### New raw data files
- `data/Thesisdata/DGS1.csv` — 1-year constant maturity Treasury yield (FRED), daily, starts 1962-01-02
- `data/Thesisdata/AAA.csv` — Moody's Seasoned AAA Corporate Bond Yield (FRED), monthly, starts 1919-01

### Raw data files no longer needed
- `data/Thesisdata/TB3MS.csv` — replaced by DGS1
- `data/Thesisdata/feds200628 (1).csv` — replaced by AAA (no longer need GSW yield curve)

### Variables to construct (all annual)

All levels use the **last available observation in December** (end-of-year value).
All returns are for calendar year t+1 (realized during year t+1).

**y_1 (state, level — end of year t):**
```python
dgs1 = pd.read_csv("Thesisdata/DGS1.csv", index_col="observation_date", parse_dates=True)
y_1 = dgs1.resample("YE-DEC").last()["DGS1"] / 100    # annual decimal
```

**y_20 (intermediate, used for spr and bond return — end of year t):**
```python
aaa = pd.read_csv("Thesisdata/AAA.csv", index_col="observation_date", parse_dates=True)
y_20 = aaa.resample("YE-DEC").last()["AAA"] / 100      # annual decimal
```

**spr (state, level — end of year t):**
```python
spr = y_20 - y_1    # yield spread: 20-year AAA minus 1-year Treasury
```

**cy (state, level — end of year t):**
```python
# Preferred: use Shiller's CAPE column directly when available (avoids unit ambiguity).
cy = -np.log(CAPE_annual)             # log earnings yield = -log(CAPE)

# Equivalent (verify E10 and P are on the same nominal/real base in your local file):
# cy = np.log(E10_annual / P_annual)
```
**Unit warning:** Some Shiller spreadsheets store `P` nominal and `E10` real; CAPE
is computed using `P / CPI_normalised`. Computing `log(E10/P)` directly in that case
gives a constant offset error. **Use `cy = -log(CAPE)` from the CAPE column** unless
you have audited the columns.

**pi — annual log inflation (realized during year t+1):**
```python
cpi = pd.read_csv("Thesisdata/CPIAUCSL.csv", index_col="observation_date", parse_dates=True)
cpi_dec = cpi.resample("YE-DEC").last()["CPIAUCSL"]
pi = np.log(cpi_dec / cpi_dec.shift(1))    # log(CPI_Dec_{t+1} / CPI_Dec_t)
```

**rtb (return — realized during year t+1):**
```python
# Use log(1+y_1) for log-linear consistency with the bond return formula
# (which also uses y_log = log(1+Y)).  CCV define r_1 = log(1+Y_1).
# Difference from using y_1 directly: ~y_1^2/2 ≈ 12 bp at y_1=5%.
rtb = np.log(1 + y_1.shift(1)) - pi    # log nominal bill return minus inflation
# rtb for year t+1 = log(1+y_1_t) - pi_{t+1}
```

**r_bond (annual nominal bond return — realized during year t+1):**
```python
# CCV loglinear approximation (Campbell, Lo & MacKinlay 1997, Ch. 10)
Y = y_20                                                    # AAA yield in decimal
n = 20
D = (1 - (1 + Y)**(-n)) / (1 - (1 + Y)**(-1))             # par bond duration
y_log = np.log(1 + Y)                                       # log gross yield

# Buy at end of year t at yield Y_t, sell at end of year t+1 at yield Y_{t+1}
# Approximate y(19, t+1) ≈ y(20, t+1)  (CCV convention)
r_bond = D.shift(1) * y_log.shift(1) - (D.shift(1) - 1) * y_log
# r_bond for year t+1 uses D_t, y_t (purchase) and y_{t+1} (sale)
```

**r_1 (log nominal bill return — intermediate, used for xr and xb):**
```python
# CCV define r_1 = log(1+Y_1).  All excess returns subtract this, not y_1 directly.
# This keeps the recovery identities exact: rtb + xr = real_stock, rtb + xb = real_bond.
r_1 = np.log(1 + y_1)    # log gross nominal bill return
```

**xr (return — realized during year t+1):**
```python
# CCV convention: all excess returns are nominal - nominal.
# Construct nominal stock return from Shiller's real RTRP by adding back inflation.
real_stock = np.log(RTRP_annual / RTRP_annual.shift(1))    # real return from Shiller
nominal_stock = real_stock + pi                              # add back inflation
xr = nominal_stock - r_1.shift(1)                           # excess nominal stock over nominal bill
# xr for year t+1 = nominal_stock_{t+1} - log(1+y_1_t)
```

**xb (return — realized during year t+1):**
```python
# Excess nominal bond return over nominal bill (same convention as xr)
xb = r_bond - r_1.shift(1)    # r_bond_{t+1} - log(1+y_1_t)
```

### Why nominal - nominal for all excess returns

CCV define all excess returns as nominal log return minus log nominal bill return
`r_1 = log(1+Y_1)`. This ensures:
1. Inflation appears in exactly ONE variable: `rtb = log(1+y_1) - pi`
2. The excess returns `xr` and `xb` contain no inflation component
3. The solver recovers real returns via `exp(rtb + xr)` and `exp(rtb + xb)`,
   where `rtb` provides the inflation adjustment for all assets

If `xr` were instead defined as `real_stock - rtb` (real - real), it would be
numerically identical (because `(real_stock - real_bill) = (nominal_stock - nominal_bill)`
when both legs use the same period's inflation), but it mixes labels and is inconsistent
with how `xb` is defined.

---

## Timing convention — explicit diagram

```
  decision date t                                   payoff/realisation date t+1
  ─────────────┬─────────────────────────────────────┬──────────────
  KNOWN at t                       UNKNOWN at t (becomes known at t+1)

  y_1[t]   ──►  enters as the yield locked in for year t+1
  spr[t]   ──►  (state)
  cy[t]    ──►  (state)
                                   pi[t+1]                   ◄── pulls down rtb
                                   nominal_stock[t+1]
                                   r_bond[t+1]               (uses y_log[t+1])

                                   rtb[t+1]   = log(1+y_1[t]) - pi[t+1]
                                   xr[t+1]    = nominal_stock[t+1] - log(1+y_1[t])
                                   xb[t+1]    = r_bond[t+1]        - log(1+y_1[t])
```

In the dataset, **the row labelled by year T contains**:
- levels (`y_1, spr, cy`) at end of year T
- returns (`rtb, xr, xb`) realised during year T (i.e., the t+1 of the diagram, with t = T-1)

When the VAR `z_T = c + Phi @ z_{T-1} + eps_T` is estimated, the levels at T-1
appear on the RHS of the rtb/xr/xb equations. This is mechanically correct: the
bill yield that determines `rtb[T]` is `y_1[T-1]`, which is the y_1 component of `z_{T-1}`.

---

## Verification — run BEFORE handing off to estimation

The four most common timing bugs are caught by deterministic identities that hold
to machine precision. Run all four after constructing the dataset; hard-fail on any
assertion.

### V1. The rtb identity

```python
# Must be exactly zero (within float64 noise).
resid_rtb = (df["rtb"] + pi - np.log(1 + df["y_1"].shift(1))).dropna()
assert resid_rtb.abs().max() < 1e-10, f"rtb timing bug: {resid_rtb.abs().max():.3e}"
```
If this fails: wrong shift on `y_1` (off by one year) OR inflation defined over a
different window than rtb.

### V2. The bond identity

```python
n = 20
D     = (1 - (1 + y_20)**(-n)) / (1 - (1 + y_20)**(-1))
y_log = np.log(1 + y_20)
r_bond_check = D.shift(1) * y_log.shift(1) - (D.shift(1) - 1) * y_log
xb_check     = r_bond_check - np.log(1 + df["y_1"].shift(1))
assert (xb_check - df["xb"]).dropna().abs().max() < 1e-10
```
If this fails: AAA in % rather than decimal (xb ~100x too large), wrong duration
date (D[T] instead of D[T-1]), or swapped purchase/sale yields.

### V3. The real-stock recovery identity

```python
real_stock = np.log(RTRP_annual / RTRP_annual.shift(1)).reindex(df.index)
recovered  = df["rtb"] + df["xr"]
resid_v3   = (recovered - real_stock).dropna()
assert resid_v3.abs().max() < 1e-4, f"V3 recovery fail: {resid_v3.abs().max():.3e}"
```
This confirms the solver's `R_stock_real = exp(rtb + xr)` formula will produce
numerically correct real returns.

**Tolerance note.** If `pi` is built from FRED CPIAUCSL and `RTRP` is Shiller's real
total-return series (deflated by Shiller's own monthly CPI), the two CPI series are not
bit-identical. The residual will be ~1e-4, not 1e-12. To get machine precision, deflate
Shiller's nominal price+dividend series using the same CPI used for `pi`, or accept
1e-4 as the tolerance.

### V4. The dual-regression identity (post-estimation)

The natural impulse — "Phi[rtb, y_1] should equal 1 because rtb = y_1.shift(1) - pi" —
is **wrong** as a numerical test. By OLS algebra, the rtb coefficient on lagged y_1 is
`1 - g_1`, where `g_1` is the coefficient of `pi` on the same state regressors. Because
the Fisher effect makes nominal short rates predict future inflation, `g_1` is
substantially positive (≈ 0.3–0.5 in US annual data even after controlling for spr and
cy), so the rtb regression will deliver a coefficient on y_1 of roughly **0.5–0.7**, not
1. This is correct VAR behaviour: the rtb equation's loadings on the state should
absorb whatever inflation predictability the state offers.

The deterministic identity that *does* hold to machine precision is the dual-regression
sum. The identity `rtb + pi = log(1+y_1).shift(1)` is exact for every observation.
Projecting both sides onto the same regressors, the coefficients must sum to the
projection of `log(1+y_1).shift(1)` — i.e., `(0, 1, 0, 0)` when the regressors are
`(1, y_1, spr, cy)` lagged. (Strictly: the y_1 coefficient sums to ~1 because
`log(1+y_1) ≈ y_1` for small yields; the identity is exact in log yields.)

**Critical:** both regressions must use the **same** estimation approach. The constrained
VAR demeanes by `z_bar` (full-sample mean of all T rows) and regresses without intercept.
If you run the pi regression with standard OLS (intercept, subsample centering), the
slopes differ by O(1/T) and the 1e-10 assertions fail. Run the pi regression with the
same constrained demeaning:

```python
import numpy.linalg as la

# Use the SAME z_bar and demeaning as the constrained VAR estimator
z_bar = var_config["z_bar"]
state_idx = [0, 1, 2]  # y_1, spr, cy

# Demean the full dataset by z_bar, then form lag/lead pairs
Z = df[["y_1", "spr", "cy", "rtb", "xr", "xb"]].to_numpy() - z_bar
X_state = Z[:-1, state_idx]     # demeaned lagged state, shape (T-1, 3)

# rtb coefficients from the constrained VAR (slopes only, no intercept)
i_rtb = 3
beta_rtb_slopes = var_config["Phi"][i_rtb, :3]   # from demeaned regression

# pi regression with the SAME demeaning (demean by z_bar, no intercept)
# pi is not in the VAR, so demean by its own full-sample mean
pi_full = pi.reindex(df.index).to_numpy()
pi_bar  = np.nanmean(pi_full)
pi_tilde = pi_full[1:] - pi_bar      # align with X_state (T-1 rows)
beta_pi_slopes, *_ = la.lstsq(X_state, pi_tilde, rcond=None)

# Slope sums (the identity)
s_slopes = beta_rtb_slopes + beta_pi_slopes
# Note: sum ≈ 1 for y_1 because log(1+y_1) ≈ y_1.  For exact 1.0,
# the VAR would need to store log(1+y_1) as the state variable.
assert abs(s_slopes[0] - 1.0) < 1e-6, f"y_1 slopes: rtb+pi = {s_slopes[0]:.8f} (expected ~1)"
assert abs(s_slopes[1])       < 1e-10, f"spr slopes: rtb+pi = {s_slopes[1]:.3e} (expected 0)"
assert abs(s_slopes[2])       < 1e-10, f"cy  slopes: rtb+pi = {s_slopes[2]:.3e} (expected 0)"

# Intercept sum: const_rtb + pi_bar_intercept should ≈ 0
# With constrained estimator: const[i] = z_bar[i] - Phi[i,:3] @ z_bar[:3]
const_rtb = z_bar[i_rtb] - beta_rtb_slopes @ z_bar[:3]
const_pi  = pi_bar - beta_pi_slopes @ z_bar[:3]
assert abs(const_rtb + const_pi) < 1e-6, f"intercepts: rtb+pi = {const_rtb+const_pi:.3e} (expected ~0)"
```

If the slope sums on spr and cy fail (they should hold to machine precision because
both regressions use identical demeaning and regressors), it's a row-alignment bug
between the dataset and the inflation series. The y_1 and intercept sums hold to ~1e-6
rather than 1e-10 because `rtb + pi = log(1+y_1)_{t-1} ≈ y_1_{t-1}` (the log-vs-level
gap is ~y_1^2/2).

A weak directional sanity check (not deterministic, but useful): `Phi[rtb, y_1]` should
land between roughly 0.3 and 0.9. Outside that range — or worse, near zero or above 1 —
suggests a timing bug somewhere upstream that V1 should have caught but didn't.

### V5. The sample-mean restriction (post-estimation)

The implied unconditional mean of the VAR must equal the sample mean **exactly**:

```python
z_bar_implied = np.linalg.solve(np.eye(6) - var_config["Phi"], var_config["const"])
z_bar_sample  = df[["y_1", "spr", "cy", "rtb", "xr", "xb"]].mean().values
assert np.allclose(z_bar_implied, z_bar_sample, atol=1e-10), (
    f"VAR implied mean does not match sample mean.\n"
    f"  implied = {z_bar_implied}\n"
    f"  sample  = {z_bar_sample}\n"
    f"  diff    = {z_bar_implied - z_bar_sample}"
)
```

If this fails, the VAR was estimated with **unconstrained** OLS and the implied
`z_bar = (I - Phi)^{-1} @ const` is drifting away from the sample mean because of
finite-sample noise in `const` amplified by `(I - Phi)^{-1}`. For variables with high
persistence (annual short rates have `Phi ≈ 0.9`, so `(1 - Phi)^{-1} ≈ 10`), even small
noise in `const` produces large errors in `z_bar`. This is the bug we hit previously
where the implied short-rate mean was off by several percentage points from the
sample average.

The fix is **constrained OLS** (CCV 2003, §4.2): pin `z_bar` to the sample mean and
estimate Phi on demeaned data without an intercept. See §"VAR estimation procedure"
below.

### Sample
- DGS1 starts 1962 → annual VAR sample starts **1963** after dropping the row lost to `.shift(1)` (binding constraint)
- AAA starts 1919, CPI starts 1947, Shiller (E10, P, RTRP, CAPE) starts 1881 → none binding
- GSW and TB3MS are no longer needed

### Output
Save as `data/var_dataset.csv` with columns: `[y_1, spr, cy, rtb, xr, xb]`
(or keep whatever column order is convenient, but update all index references).

---

## VAR estimation procedure: pin z_bar to the sample mean

CCV (2003, §4.2): *"We estimate the VAR imposing the restriction that the unconditional
means of the variables implied by the VAR coefficient estimates equal their full-sample
arithmetic counterparts."* This is not a stylistic choice — it materially affects every
quantity downstream of the VAR (conditional return means, the state-grid centring,
the implied bond/equity premia at each grid point). With unconstrained OLS, the implied
`z_bar` for a variable with persistence ~0.9 can drift several percentage points from
the sample mean purely due to finite-sample noise.

### The constraint

Define `z̄ := sample_mean(data)`. Impose:

```
const = (I - Phi) @ z̄
```

This guarantees `z_bar_implied = (I - Phi)^{-1} @ const = z̄` exactly. Phi is then the
only free parameter (apart from the restriction that lagged returns don't predict).

### Implementation (replaces the current `estimate_restricted_var1_from_csv`)

```python
def estimate_restricted_var1_constrained(csv_path, columns, state_indices):
    """
    Restricted VAR(1) with z_bar pinned to the full-sample mean (CCV 2003).

    Steps:
      1. Compute z_bar = column-wise sample mean of the FULL dataset (no row drop).
      2. Demean the data: z_tilde_t = z_t - z_bar.
      3. Regress z_tilde_{t+1} on z_tilde_t WITHOUT intercept, using state columns
         only (lagged returns excluded).
      4. Recover const = (I - Phi) @ z_bar.

    Returns var_core dict with z_bar, Phi, Omega, const, ...
    """
    import numpy as np, pandas as pd

    data = _load_var_dataset(csv_path=csv_path, columns=columns)
    n = len(columns)

    # 1. Sample mean over ALL rows (the restriction target).
    z_bar = data.mean(axis=0).to_numpy()

    # 2. Demean.
    Z = data.to_numpy() - z_bar          # shape (T, n)
    Y = Z[1:, :]                          # z_tilde_{t+1}, shape (T-1, n)
    state_idx = np.asarray(state_indices, dtype=int)
    X = Z[:-1, state_idx]                 # z_tilde_t for state cols only

    # 3. OLS WITHOUT intercept (it's pinned by the constraint).
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (n_state, n)

    Phi = np.zeros((n, n), dtype=float)
    for k, j in enumerate(state_idx):
        Phi[:, j] = coeffs[k, :]
    # Return columns of Phi are zero by the restriction.

    # 4. Recover const so the implied mean equals z_bar by construction.
    const = (np.eye(n) - Phi) @ z_bar

    # Residuals and Omega from the constrained model
    Y_hat = X @ coeffs
    resid = Y - Y_hat
    dof   = (Y.shape[0]) - X.shape[1]
    if dof <= 0:
        raise ValueError("Not enough observations for restricted constrained VAR")
    Omega = (resid.T @ resid) / dof

    return {
        "z_bar":    z_bar,
        "Phi":      Phi,
        "Omega":    Omega,
        "const":    const,
        "variable_names": list(columns),
        "residual_correlation": _safe_residual_correlation(resid),
        "equation_r2":         _compute_r2_per_equation_demeaned(Y, Y_hat, columns),
        "estimation": "restricted_constrained",
        "trend": "c",
    }
```

### Equivalence note

This is mathematically identical to writing the regression as
`(z_{t+1} - z̄) = Phi (z_t - z̄) + ε` and is the standard "constrained least-squares"
in CCV's footnote. It is **not** the same as estimating with an intercept and then
solving for `z_bar` afterwards — that's the unconstrained OLS we're replacing.

### Why this matters here specifically

Three of the five variables affected by this constraint matter for the lifecycle solver:

1. **`y_1` short-rate mean.** Drives `R_bill_real = exp(rtb)` via `rtb = log(1+y_1) - pi`. An
   error of 1pp in `z_bar(y_1)` shifts the unconditional bill return by ~1pp,
   propagating into wealth accumulation across the entire lifecycle.

2. **`xr` equity-premium mean.** The Jensen-adjusted equity premium `E[xr] + 0.5·Var(xr)`
   is the central object the model is calibrated against. A drift in `z_bar(xr)` directly
   distorts portfolio choice.

3. **`xb` bond-premium mean.** Same logic for the bond demand.

`spr` and `cy` are state predictors only, but their means still enter the conditional
return formulas, so they should also be pinned for consistency.

---

## Code Changes by File

### 1. `var.py`

**`estimate_restricted_var1_from_csv()` — REPLACE with constrained version:**
The current implementation estimates Phi and `const` jointly by OLS, then computes
`z_bar = (I - Phi)^{-1} @ const`. This is the unconstrained estimator and produces
an implied `z_bar` that drifts from the sample mean — fatally so for the persistent
short-rate variables in this VAR. Replace with the CCV constrained estimator that
pins `z_bar` to the sample mean (full code in §"VAR estimation procedure" above).

V5 must pass after this change: `(I - Phi)^{-1} @ const == sample_mean` to machine
precision. If V5 fails, the constrained estimator was not used.

**`estimate_var1_from_csv()` (unrestricted variant):** Apply the same constraint —
demean, regress without intercept, recover const from `(I - Phi) @ z_bar`. The
unrestricted version is only used for diagnostic comparison and not by the solver,
but applying the constraint there too keeps both estimators on the same footing.

**`build_var_config_from_dataset()` and convenience wrappers:**
- New column list: `["y_1", "spr", "cy", "rtb", "xr", "xb"]`
- `state_indices = [0, 1, 2]` (y_1, spr, cy)
- `return_indices = [3, 4, 5]` (rtb, xr, xb)
- Remove `bill_rate_index_in_state` parameter — no bill rate in state
- Remove `annuity_yield_index_in_state` parameter — annuity uses both y_1 and spr
- Add `y_1_index_in_state = 0` and `spr_index_in_state = 1` to var_config
- (No `cy_index_in_state` field needed — `cy` does not have a special downstream role
  beyond being a state predictor.)

**Hardcoded fallbacks (`_Z_BAR`, `_PHI`, `_OMEGA`, etc.):**
- All need re-estimation after new data construction
- Dimensions change from 5×5 to 6×6

**`partition_var()`:**
- No changes needed — it's generic over any state/return split
- Output `M` will be (3×3) instead of (2×3)
- Output `Sigma_r_cond` will be (3×3) instead of (2×2)

### 2. `model.py`

**`LifecyclePortfolioModel` (NamedTuple):**

Remove:
```python
bill_rate_index_in_state: int          # line 81
annuity_yield_index_in_state: int      # line 82
```

Add:
```python
y_1_index_in_state: int                # index of y_1 in state vector (= 0)
spr_index_in_state: int                # index of spr in state vector (= 1)
```

Update:
```python
n_ret: int    # changes from 2 to 3
```

**`annuity_factor()` function (line 194):**

Current signature: `annuity_factor(y_ann, b_bar)` — single discount rate.

New signature: `annuity_factor(y_1, spr, b_bar)` — short yield and spread, interpolated term structure.

New implementation:
```python
def annuity_factor(y_1, spr, b_bar):
    """
    Annuity factor with linearly interpolated term structure.

    Recovers y_20 = y_1 + spr, then interpolates discount rates between
    y_1 (1-year yield) and y_20 (20-year yield).

    A = sum_{k=1}^{b_bar} (1 + y(k))^{-k}
    where y(k) = y_1 + spr * min(k - 1, 19) / 19

    For k=1:   y(1)  = y_1.
    For k=20:  y(20) = y_1 + spr = y_20.
    For k>=20: y(k)  = y_20 (capped — do NOT extrapolate).

    Uses DISCRETE compounding (1+y)^{-k} to match the existing codebase
    convention.  Do NOT use exp(-y*k) — that's continuous compounding and
    gives a ~12 bp/yr gap at y=5%, accumulating over b_bar periods.

    Capping (rather than extrapolating) avoids unbounded discount rates if
    b_bar > 20. With Catherine's b_bar = 10 the cap never binds, but the
    defensive code documents intended behaviour.

    Parameters: y_1, spr can be scalars or aligned arrays (one per grid point).
    """
    y_1, spr = np.asarray(y_1, dtype=float), np.asarray(spr, dtype=float)
    A = np.zeros_like(y_1)
    for k in range(1, b_bar + 1):
        frac = min(k - 1, 19) / 19.0
        y_k  = y_1 + spr * frac
        A   += (1.0 + y_k) ** (-k)
    return A
```

### 3. `precompute.py`

**`Precompute.__init__():`**

State grid comment (line 134): update from `[rtb, y_nom, dp]` to `[y_1, spr, cy]`.

Note: wherever `y_20` is needed (bond return formula in data construction, annuity pricing),
recover it as `y_20 = y_1 + spr`. The state grid stores `spr` directly; `y_20` is never
stored as a separate array.

Conditional return means (lines 161-171): `const_r`, `A_r`, `M_v_nodes` all change
dimension from n_ret=2 to n_ret=3. No code change needed if they read from the model,
which will have the right dimensions. But verify:
- `self.const_r` shape: (3,) instead of (2,)
- `self.A_r` shape: (3, 3) instead of (2, 3)
- `self.M_v_nodes` shape: (K^3, 3) instead of (K^3, 2)

Return quadrature (lines 146-150): `get_return_quadrature` uses `model.Sigma_r_cond`
which is now (3×3). The quadrature will automatically produce `n_nodes^3` joint nodes
instead of `n_nodes^2`. No code change needed in `get_return_quadrature` itself (it's
already generic over `n_ret`).

Precomputed exp arrays (lines 173-175): Currently:
```python
self.exp_ret_stock = np.exp(self.ret_nodes[:, 0])  # xr residuals
self.exp_ret_bond  = np.exp(self.ret_nodes[:, 1])  # xb residuals
```

Change to (note: ret_nodes columns are now [rtb_resid, xr_resid, xb_resid]):
```python
self.exp_ret_bill  = np.exp(self.ret_nodes[:, 0])  # rtb residuals
self.exp_ret_stock = np.exp(self.ret_nodes[:, 1])  # xr residuals
self.exp_ret_bond  = np.exp(self.ret_nodes[:, 2])  # xb residuals
```

**Remove `r_bill_grid` (line 177-179):** Delete entirely. No bill rate in state.

**Annuity factors (lines 187-188):** Change from:
```python
_y_ann = self.state_grid[:, model.annuity_yield_index_in_state]
self.annuity_factors = annuity_factor(_y_ann, model.b_bar)
```
To:
```python
_y_1 = self.state_grid[:, model.y_1_index_in_state]
_spr = self.state_grid[:, model.spr_index_in_state]
self.annuity_factors = annuity_factor(_y_1, _spr, model.b_bar)
```

**`build_model()` (line 409):**

Remove:
```python
bill_rate_index_in_state=bill_rate_index_in_state,    # line 471
annuity_yield_index_in_state=annuity_yield_index_in_state,  # line 472
```

Add:
```python
y_1_index_in_state=int(var_config["y_1_index_in_state"]),
spr_index_in_state=int(var_config["spr_index_in_state"]),
```

Remove the validation checks for `bill_rate_index_in_state` and
`annuity_yield_index_in_state` (lines 423-429). Add range checks for
`y_1_index_in_state` and `spr_index_in_state` (in-range, distinct).

**Docstring / comments:** Update the solver input reference (lines 51-67) to remove
`r_bill_grid` and update return dimensions.

### 4. `solver.py`

This is the most invasive change. The core pattern is the same everywhere:
`R_bill` must come from the return quadrature, not from the state.

**Solver invariant.** All three of `R_bill`, `R_stock`, `R_bond` for a single
quadrature evaluation must use the **same `k_r`** index. They are the three
components of one joint outcome of period-t+1 returns. Sampling them from
different `k_r` values would re-introduce the timing inconsistency this
refactor fixes.

**A. Retirement FOC: `compute_foc_jac_retirement_quad()` (line 322)**

Current signature includes `R_bill` as a scalar parameter (line 333).

Change: Remove `R_bill` from parameters. Add `exp_ret_bill` array parameter.

Current inner loop (lines 388-405):
```python
mu_r_stock = base_mu_r_i[0] + M_v_nodes[k_v, 0]
mu_r_bond  = base_mu_r_i[1] + M_v_nodes[k_v, 1]
exp_mu_s = exp(mu_r_stock)
exp_mu_b = exp(mu_r_bond)

for k_r in range(n_ret_quad):
    R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
    R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
    R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
    Rex_s = R_s - R_bill
    Rex_b = R_b - R_bill
```

New inner loop:
```python
# base_mu_r_i is now (3,): [rtb_mean, xr_mean, xb_mean]
mu_r_bill  = base_mu_r_i[0] + M_v_nodes[k_v, 0]
mu_r_stock = base_mu_r_i[1] + M_v_nodes[k_v, 1]
mu_r_bond  = base_mu_r_i[2] + M_v_nodes[k_v, 2]
exp_mu_bill = exp(mu_r_bill)
exp_mu_s = exp(mu_r_stock)
exp_mu_b = exp(mu_r_bond)

for k_r in range(n_ret_quad):
    R_bill = exp_mu_bill * exp_ret_bill[k_r]         # UNCERTAIN — same k_r as below
    R_s    = R_bill * exp_mu_s * exp_ret_stock[k_r]  # = exp(rtb + xr)
    R_b    = R_bill * exp_mu_b * exp_ret_bond[k_r]   # = exp(rtb + xb)
    R_p    = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
    Rex_s  = R_s - R_bill
    Rex_b  = R_b - R_bill
```

Note: `R_bill` now varies per `(k_v, k_r)` node. The rest of the FOC accumulation
(marginal utility, Jacobian) is unchanged — it already uses `R_p`, `Rex_s`, `Rex_b`
which now correctly reflect the uncertain bill return.

**B. Working-age FOC: `compute_foc_jac_working_quad()` (line 459)**

Identical pattern. Same change: remove `R_bill` parameter, add `exp_ret_bill`,
compute `R_bill` inside the `k_r` loop.

**C. Retirement step solver: `_solve_retirement_age_step_quad()` (line ~1680)**

Line 1694: `R_bill = exp(r_bill_grid[i_s])` — DELETE this line.

Line 1699-1701: `base_mu_r_i` computation. Change from (2,) to (3,):
```python
base_mu_r_i = np.empty(3)
base_mu_r_i[0] = const_r[0] + A_r[0,0]*s_i[0] + A_r[0,1]*s_i[1] + A_r[0,2]*s_i[2]
base_mu_r_i[1] = const_r[1] + A_r[1,0]*s_i[0] + A_r[1,1]*s_i[1] + A_r[1,2]*s_i[2]
base_mu_r_i[2] = const_r[2] + A_r[2,0]*s_i[0] + A_r[2,1]*s_i[1] + A_r[2,2]*s_i[2]
```

Update all calls to `compute_foc_jac_retirement_quad()` within this function to pass
`exp_ret_bill` instead of `R_bill`.

**D. Working-age step solver: `_solve_working_age_step_quad()` (line ~1855)**

Line 1870: `R_bill = exp(r_bill_grid[i_s])` — DELETE.

Line 1874-1876: Same `base_mu_r_i` expansion to (3,).

Update all calls to `compute_foc_jac_working_quad()`.

**E. Terminal age solver: `solve_terminal_age()` (line 973)**

Line 988: `R_bill = exp(r_bill_grid[i_s])` — DELETE.

This function currently uses `Pi_state` (discrete Markov) for state transitions,
not quadrature. It needs a deeper rewrite to use quadrature, which was already
an open TODO (RETURNS.md section 6.12). With the new system, `R_bill` varies
across return nodes, so the terminal solver must integrate over return uncertainty
including `rtb`.

The terminal portfolio problem becomes: for each return quadrature node, compute
R_bill, R_stock, R_bond, then evaluate E[R_p^{1-gamma}] across all joint
(state × return) nodes.

Remove `r_bill_grid` from the function signature. Add `exp_ret_bill` and the
state quadrature arrays. The functions `_terminal_prepare_scenarios`,
`_terminal_portfolio_moment`, `_terminal_portfolio_grad`, `_terminal_portfolio_hess`
all need `R_bill` to become an array varying per scenario rather than a scalar.

**F. Newton solver wrapper: `solve_portfolio_2d_newton_retirement_quad()` (line ~1029)**

Remove `R_bill` from signature. Add `exp_ret_bill`. Pass through to FOC function.

**G. `solve_portfolio_2d_newton_working_quad()` — same pattern.**

**H. Function signatures throughout solver.py that pass `r_bill_grid`:**

Search for `r_bill_grid` in solver.py and remove from all signatures. The step
solvers (`_solve_retirement_age_step_quad`, `_solve_working_age_step_quad`) and
the top-level `run_lifecycle_solver` all pass `r_bill_grid` — remove everywhere
and replace with `exp_ret_bill`.

### 5. `simulation.py`

**`simulate_lifecycle_core()` (line 347):**

Remove `r_bill_grid` from signature (line 358).

Line 491: `R_bill = np.exp(r_bill_grid[state_idx])` — DELETE.

Lines 509-511: Conditional return means. Expand from 2 to 3:
```python
mu_rtb = const_r[0] + A_r[0,0]*s_cur[0] + A_r[0,1]*s_cur[1] + A_r[0,2]*s_cur[2] \
       + M_matrix[0,0]*v_s_0 + M_matrix[0,1]*v_s_1 + M_matrix[0,2]*v_s_2
mu_xr  = const_r[1] + A_r[1,0]*s_cur[0] + A_r[1,1]*s_cur[1] + A_r[1,2]*s_cur[2] \
       + M_matrix[1,0]*v_s_0 + M_matrix[1,1]*v_s_1 + M_matrix[1,2]*v_s_2
mu_xb  = const_r[2] + A_r[2,0]*s_cur[0] + A_r[2,1]*s_cur[1] + A_r[2,2]*s_cur[2] \
       + M_matrix[2,0]*v_s_0 + M_matrix[2,1]*v_s_1 + M_matrix[2,2]*v_s_2
```

Lines 537-549: Return draws. Expand residual draw from 2 to 3 dimensions:
```python
if use_mc_returns:
    rtb_res = 0.0; xr_res = 0.0; xb_res = 0.0
    for k in range(n_ret):                          # n_ret is now 3
        shock_k = normal_draws[i, t, k]
        rtb_res += ret_factor[0, k] * shock_k
        xr_res  += ret_factor[1, k] * shock_k
        xb_res  += ret_factor[2, k] * shock_k
    R_bill  = np.exp(mu_rtb + rtb_res)
    R_stock = R_bill * np.exp(mu_xr + xr_res)
    R_bond  = R_bill * np.exp(mu_xb + xb_res)
else:
    ret_idx = draw_discrete(ret_weights, uniform_draws[i, t, 3])
    R_bill  = np.exp(mu_rtb + ret_nodes[ret_idx, 0])
    R_stock = R_bill * np.exp(mu_xr + ret_nodes[ret_idx, 1])
    R_bond  = R_bill * np.exp(mu_xb + ret_nodes[ret_idx, 2])
```

Lines 551-552: Portfolio return (unchanged structurally):
```python
R_port = alpha_s_t * R_stock + alpha_b_t * R_bond + alpha_bill_t * R_bill
```

**`simulate_lifecycle()` wrapper (line 628):** Remove `r_bill_grid` from the call
to `simulate_lifecycle_core`. Update `normal_draws` shape to accommodate 3 return
shocks + 3 state shocks (currently 2 + 3).

**`ret_factor`**: Currently the Cholesky of `Sigma_r_cond` with shape (2, 2).
Now (3, 3). This is passed from Precompute — verify it's computed from the new
(3×3) `Sigma_r_cond`.

### 6. `discretization.py`

**`get_return_quadrature()` (line 220):**

Already generic — it reads `model.n_ret` and `model.Sigma_r_cond`. With n_ret=3
and a (3×3) Sigma_r_cond, it will produce `n_nodes^3` tensor-product nodes
automatically. **No code change needed**, but verify:
- `n_ret_quad = n_nodes ** 3` (was `n_nodes ** 2`)
- the Cholesky / spectral factor is computed from the (3×3) `Sigma_r_cond`,
  not a hard-coded (2×2) anywhere
- tensor-product indexing handles 3D correctly (uses `np.meshgrid` with `n_ret`
  repetitions — should work)

**`get_state_quadrature()` — no change** (still 3 state dimensions).

**`rouwenhorst_multivariate()` — no change** (still 3 state dimensions, different variables).

### 7. `main.ipynb` (and any other notebooks)

Update calls to:
- `build_var_config_from_dataset()` or convenience wrappers with new column names and indices
- Remove `bill_rate_index_in_state` and `annuity_yield_index_in_state` from configs
- Add `y_1_index_in_state=0` and `spr_index_in_state=1`
- Rename plot/diagnostic labels: `state_var_labels = ["y_1", "spr", "cy"]`,
  `rate_dims = {model.y_1_index_in_state, model.spr_index_in_state}`
  (both yields; the new annual VAR has no `*4` quarterly→annual scaling)

---

## Common pitfalls

1. **Off-by-one on `y_1` in rtb.** If you write `rtb = np.log(1+y_1) - pi` (forgetting `.shift(1)`),
   V1 fails. V4 (post-estimation dual-regression sum) also fails — but the symptom is a
   nonzero residual in the sum identity, not a "wrong" Phi coefficient. (The Phi
   coefficient itself isn't a clean test; see V4.)

2. **AAA in % not decimal.** `Y = 0.05` for 5%, not `Y = 5.0`. The bond formula
   compounds this nonlinearly via `(1+Y)^{-n}`; resulting `xb` is wildly large
   and `Sigma_r_cond` explodes. V2 catches this.

3. **Mixing real and nominal in xr/xb.** If you define `xr = real_stock - rtb`
   (numerically equal to nominal-minus-nominal so the constructed value is fine)
   but then forget that and *also* deflate elsewhere, you subtract inflation twice.
   Stick to nominal-minus-nominal in construction; recover real via `rtb + xr` in
   the solver.

4. **Sampling `R_bill` and `R_s`/`R_b` from different `k_r` draws.** This is the
   bug the whole refactor exists to fix. After the change, the inner loop must
   use **one** `k_r` for all three returns. See "Solver invariant" above.

5. **Shiller column unit conventions.** Some `ie_data.xls` versions store P
   nominal and E10 real; CAPE is then computed using `P / CPI_normalised`. Computing
   `log(E10/P)` directly in that version gives a constant offset. **Use
   `cy = -log(CAPE)` from the CAPE column** unless you have audited the columns.

6. **`cy` sign.** `cy = -log(CAPE)`. With CAPE ~30 in 2024, `cy ≈ -3.4`. If your
   `cy` values are positive and ~30, you stored CAPE itself, not its log inverse.

7. **`y_20` recovery.** Wherever you need `y_20` (bond formula, annuity), compute
   `y_20 = y_1 + spr` on the fly from the state. Do NOT add a separate `y_20_grid`.

8. **Unconstrained OLS for the VAR.** The current `var.py` computes
   `z_bar = (I - Phi)^{-1} @ const` from unconstrained OLS estimates. With persistent
   variables (`Phi ≈ 0.9`) and T=63, this drifts the implied means away from the sample
   means by several percentage points. Use constrained OLS (CCV 2003) to pin
   `z_bar = sample_mean` exactly. V5 catches this.

---

## Computational Impact

- Return quadrature: K_r^3 instead of K_r^2 nodes. At K_r=2: 8 vs 4. At K_r=3: 27 vs 9.
- Total inner loop evaluations per grid point: K_state^3 × K_ret^3 (e.g., 27 × 8 = 216 at K=3,K_r=2)
- State grid unchanged (3D, same sizes)
- Annuity factor precomputation: trivial (N_state × b_bar exp calls)
- Overall: expect ~2-3× slower solve, still fast

## What Does NOT Change

- Rouwenhorst discretization (still 3 state dims)
- Trilinear interpolation / `bracket_state_3d`
- FOC structure: still 2 equations (alpha_s, alpha_b), 2 unknowns
- EGM / Newton solver logic
- Income process (z_grid, Pi_z, eps quadrature, eta quadrature)
- Mortality tables
- Utility function (Epstein-Zin)
- Gauss-Hermite quadrature machinery
- Portfolio constraint handling (simplex projection)

## Implementation Order

1. **Data construction** — new `data_construction.ipynb` producing the 6-variable annual dataset.
   **Run all five V1–V5 checks; hard-fail on assertion.** (V5 only meaningful after estimation.)
2. **`var.py`** — replace the unconstrained estimator with the CCV constrained version
   (pins `z_bar` to the sample mean). Re-estimate, verify partition dimensions.
   Confirm V4 (dual-regression identity) and V5 (sample-mean restriction).
3. **`model.py`** — update NamedTuple fields, new `annuity_factor()`.
4. **`precompute.py`** — update `Precompute.__init__()` and `build_model()`.
5. **`solver.py`** — update all FOC functions and step solvers (most invasive).
   Keep one git commit per FOC function.
6. **`simulation.py`** — update return draws and portfolio computation.
7. **Validate** — re-run sanity checks: FOC convergence, wealth monotonicity, terminal condition.
   K_r=1 smoke test (see below).

## Key Invariant to Verify After Implementation

At every point in the solver and simulation where portfolio returns are computed,
ALL THREE returns (R_bill, R_stock, R_bond) must come from the SAME quadrature
node draw. They must be for the same period (t+1) and conditioned on the same
innovation realization. The old code had R_bill from the current state and
R_stock/R_bond from the next-period draw — this timing inconsistency is the
entire reason for this refactor.

**K_r=1 smoke test.** With one return node (the conditional mean):
- `R_bill` should equal `exp(mu_rtb)` exactly
- `R_stock` should equal `exp(mu_rtb + mu_xr)` exactly
- `R_bond` should equal `exp(mu_rtb + mu_xb)` exactly
- `Rex_s = R_stock - R_bill` must equal `R_bill * (exp(mu_xr) - 1)` exactly

If any of these fail at K_r=1, the rest of the pipeline is moot.

# VAR Data Specification

## Overview

Build one quarterly dataset with all variables below. Two VARs will be estimated on different sample windows from this dataset.

---

## VAR System 1: Nominal Bond System (CCV replication)

**Sample: 1952Q2 – 2025Q4 (approx 295 observations)**

State vector (6 variables, ordered):

| # | Variable | Symbol | Definition | Units |
|---|----------|--------|------------|-------|
| 1 | Real bill rate | `rtb` | Ex post real return on 3-month T-bill: nominal bill rate minus realized CPI inflation over the quarter | Quarterly log return (not annualized) |
| 2 | Excess stock return | `xr` | Log total return on CRSP value-weighted market portfolio minus log gross bill return | Quarterly log excess return |
| 3 | Excess nominal bond return | `xb` | Log total return on constant-maturity 10-year nominal zero-coupon bond minus log gross bill return | Quarterly log excess return |
| 4 | Nominal bill yield | `y_nom` | 3-month Treasury bill yield (annualized in raw data, convert to quarterly) | Quarterly yield level |
| 5 | Log dividend-price ratio | `dp` | Log of trailing 12-month dividends divided by current price for S&P 500 / CRSP VW index | Log ratio (level, not return) |
| 6 | Yield spread | `spr` | 10-year nominal yield minus 3-month bill yield | Quarterly spread (level) |

---

## VAR System 2: TIPS System (new)

**Sample: 2003Q1 – 2025Q4 (approx 92 observations)**

State vector (5 variables, ordered):

| # | Variable | Symbol | Definition | Units |
|---|----------|--------|------------|-------|
| 1 | Real bill rate | `rtb` | Same as System 1 | Quarterly log return |
| 2 | Excess stock return | `xr` | Same as System 1 | Quarterly log excess return |
| 3 | Excess TIPS return | `xtips` | Log total return on constant-maturity 10-year real zero-coupon bond minus log gross bill return | Quarterly log excess return |
| 4 | Log dividend-price ratio | `dp` | Same as System 1 | Log ratio (level) |
| 5 | Real long yield | `y_real` | 10-year zero-coupon TIPS yield from GSW fitted curve | Quarterly yield level |

---

## Variable Construction Details

### 1. Real bill rate (`rtb`)

```
Source: FRED TB3MS (monthly), FRED CPIAUCSL (monthly)
Construction:
  - Nominal quarterly bill return: r_bill = TB3MS / 400 (approximate, or compound monthly)
  - Quarterly CPI inflation: pi_q = log(CPI_t / CPI_{t-1}) using end-of-quarter months (Mar, Jun, Sep, Dec)
  - Real bill rate: rtb = r_bill - pi_q
Note: This is the ex post real return, not the ex ante real rate.
```

### 2. Excess stock return (`xr`)

```
Source: Ken French data library, "Fama/French 3 Factors" file
  - Mkt-RF is the value-weighted market excess return (already in excess of risk-free)
  - RF is the risk-free rate
Construction:
  - Monthly data: compound to quarterly
  - xr = sum of monthly log(1 + Mkt-RF/100 + RF/100) - sum of monthly log(1 + RF/100)
  - Or equivalently: quarterly log market return minus quarterly log bill return
Note: Ken French reports in percentage points. Divide by 100 first.
```

### 3. Excess nominal bond return (`xb`) — System 1 only

```
Source: GSW nominal yield curve (feds200628.csv from Federal Reserve)
  - Use SVENY10 (10-year zero-coupon nominal yield, in percent per annum)
Construction:
  - Log price of n-year zero-coupon bond: p(n,t) = -n * y(n,t)/100
  - Quarterly return: buy 10-year bond, sell as 9.75-year bond one quarter later
  - r_bond = -(10 - 0.25) * y(9.75, t+1)/100 + 10 * y(10, t)/100
  - For y(9.75): use NSS parameters (BETA0-BETA3, TAU1, TAU2) to compute yield at 9.75
  - NSS formula: y(n) = beta0 + beta1*((1-exp(-n/tau1))/(n/tau1))
                  + beta2*((1-exp(-n/tau1))/(n/tau1) - exp(-n/tau1))
                  + beta3*((1-exp(-n/tau2))/(n/tau2) - exp(-n/tau2))
  - Excess return: xb = r_bond - r_bill
  - Use end-of-quarter dates (last business day of Mar, Jun, Sep, Dec)
```

### 4. Excess TIPS return (`xtips`) — System 2 only

```
Source: GSW TIPS yield curve (feds200805.csv from Federal Reserve)
  - Use TIPSY10 (10-year zero-coupon real yield, in percent per annum)
  - Use BETA0-BETA3, TAU1, TAU2 for non-integer maturities
Construction:
  - Identical method to xb but using real yields:
  - r_tips = -(10 - 0.25) * y_real(9.75, t+1)/100 + 10 * y_real(10, t)/100
  - Excess return: xtips = r_tips - r_bill
  - Use end-of-quarter dates
Note: Check for missing values pre-2003. Start sample where TIPSY10 is reliably available.
```

### 5. Nominal bill yield (`y_nom`) — System 1 only

```
Source: FRED TB3MS (monthly, secondary market 3-month T-bill rate, percent per annum)
Construction:
  - Use end-of-quarter value (March, June, September, December)
  - Convert to quarterly: y_nom = TB3MS / 400
  - Or keep in annualized percent if preferred (just be consistent)
```

### 6. Log dividend-price ratio (`dp`)

```
Source: Robert Shiller's online data (http://www.econ.yale.edu/~shiller/data.htm)
  - Contains S&P 500 price and 12-month trailing dividends, monthly
  - OR construct from CRSP: trailing 12-month sum of dividends / current index level
Construction:
  - dp = log(D12_t / P_t) where D12 is trailing 12-month dividend sum
  - Use end-of-quarter values
```

### 7. Yield spread (`spr`) — System 1 only

```
Source: GSW nominal yield curve SVENY10 and FRED TB3MS
Construction:
  - spr = SVENY10/100 - TB3MS/100 (both in same units, either quarterly or annualized)
  - Use end-of-quarter values
  - Keep in same units as y_nom for consistency
```

### 8. Real long yield (`y_real`) — System 2 only

```
Source: GSW TIPS yield curve TIPSY10 (percent per annum)
Construction:
  - Use end-of-quarter value
  - Convert to quarterly: y_real = TIPSY10 / 400
  - Or keep in annualized percent (be consistent with y_nom choice)
Cross-check: Compare to FRED DFII10 series during overlap period
```

---

## Date Alignment

All variables use **end-of-quarter** timing:
- Q1 = last business day of March
- Q2 = last business day of June
- Q3 = last business day of September
- Q4 = last business day of December

For the VAR `z_{t+1} = Phi_0 + Phi_1 * z_t + v_{t+1}`:
- `z_t` is observed at end of quarter t
- Returns (`rtb`, `xr`, `xb`, `xtips`) dated t+1 are realized over quarter t+1
- State variables (`y_nom`, `dp`, `spr`, `y_real`) dated t+1 are observed at end of quarter t+1

This means row t of the dataset has:
- Returns earned during quarter t
- State variable levels observed at end of quarter t

The VAR regresses row t+1 on row t.

---

## Unit Convention

Keep everything in **natural quarterly units** (not annualized, not in percent):
- Returns: quarterly log returns as decimals (e.g., 0.02 = 2% per quarter)
- Yields: quarterly yields as decimals (e.g., 0.01 = 1% per quarter = 4% annualized)
- dp: log ratio, no conversion needed
- spr: quarterly spread as decimal

This matches CCV's convention. Standard deviations and VAR coefficients will be directly comparable to CCV Table 2 after accounting for the quarterly scaling.

---

## Output

Produce a single CSV file: `var_dataset.csv`

Columns: `date, rtb, xr, xb, y_nom, dp, spr, xtips, y_real`

- `xb`, `y_nom`, `spr` will have values from 1952 onward
- `xtips`, `y_real` will have values from ~2003 onward (NaN before)
- All other columns available for full sample

This single file supports both VARs:
- System 1: select columns `[rtb, xr, xb, y_nom, dp, spr]`, drop rows with NaN
- System 2: select columns `[rtb, xr, xtips, dp, y_real]`, drop rows with NaN

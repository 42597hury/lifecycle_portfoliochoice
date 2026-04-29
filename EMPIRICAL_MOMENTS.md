# Empirical Moments — Bond and Bill Return Series

Quick-reference document for in-sample expected returns, standard deviations,
and covariances across all candidate bond and bill series. Use this to compare
alternatives for the lifecycle VAR model.

All returns are **annual log returns** unless noted otherwise.

---

## 1. Summary Tables

### 1.1 Bond Excess Returns (xb = r_bond − log(1+y_1))

Ranked by Sharpe ratio. "Full" = maximum available sample. "1963–2025" = matched to current VAR window.

| # | Series | Source | Method | n | Full sample | N | E[xb] % | std(xb) % | Sharpe | Eff. dur. |
|---|--------|--------|--------|---|-------------|---|---------|-----------|--------|-----------|
| 1 | AAA 20yr | `AAA.csv` | CCV | 20 | **1963–2025** | **63** | **+1.43** | 9.04 | 0.158 | 11.8 |
| 2 | GS20 20yr | `GS20.csv` | CCV | 20 | 1963–2025† | 56 | +0.37 | 10.80 | 0.034 | 12.7 |
| 3 | DGS30 30yr | `DGS30.csv` | CCV | 30 | 1978–2025 | 48 | +1.39 | 14.92 | 0.093 | 15.5 |
| 4 | DGS30 n=20 | `DGS30.csv` | CCV | 20 | 1978–2025 | 48 | +1.57 | 11.95 | 0.131 | 12.6 |
| 5 | ZC 10yr | `SVENY10/09` | ZC exact | 10 | 1972–2025 | 54 | +2.22 | 10.85 | 0.205 | 10.0 |
| 6 | ZC 20yr | `SVENY20/19` | ZC exact | 20 | 1982–2025 | 44 | +5.98 | 18.93 | 0.316 | 20.0 |
| 7 | ZC 30yr | `SVENY30/29` | ZC exact | 30 | 1986–2025 | 40 | +5.43 | 26.27 | 0.207 | 30.0 |
| 8 | Par 10yr | `SVENPY10` | CCV | 10 | 1972–2025 | 54 | +1.13 | 7.57 | 0.150 | 7.9 |
| 9 | Par 20yr | `SVENPY20` | CCV | 20 | 1982–2025 | 44 | +2.92 | 11.29 | 0.259 | 12.9 |
| 10 | Par 30yr | `SVENPY30` | CCV | 30 | 1986–2025 | 40 | +2.33 | 14.63 | 0.159 | 16.6 |

† GS20 has NaN gap 1987–1992 (Treasury stopped issuing 20yr bonds). N=56 out of 63 possible years.

**Key observations:**
- Only **AAA 20yr** and **GS20 20yr** cover the full 1963–2025 VAR window.
- All GSW series (Par and ZC) start 1972 at earliest (10yr) or 1982+ (20yr, 30yr).
- ZC bonds have duration = maturity, making them a fundamentally different (more aggressive) asset than par bonds of the same maturity. Their higher E[xb] is compensation for higher duration, not a "better" estimate.
- Par 20yr (GSW Treasury) has a higher Sharpe than AAA 20yr, but on 19 fewer years that happen to coincide with the secular bond bull market.

### 1.2 Bond Total Real Returns (rtb + xb = r_bond − π)

| Series | Sample | N | E[rtb+xb] % | E[exp(rtb+xb)]−1 % |
|--------|--------|---|-------------|---------------------|
| AAA 20yr | 1963–2025 | 63 | +2.34 | +2.90 |
| GS20 20yr | 1963–2025 | 56 | +1.12 | +1.88 |
| DGS30 30yr | 1978–2025 | 48 | +2.43 | +3.76 |
| ZC 10yr | 1972–2025 | 54 | +3.01 | +3.86 |
| ZC 20yr | 1982–2025 | 44 | +7.12 | +9.61 |
| Par 10yr | 1972–2025 | 54 | +1.93 | +2.37 |
| Par 20yr | 1982–2025 | 44 | +4.06 | +4.95 |

### 1.3 Bill Returns

| Series | Source | Construction | Sample | N | E[nominal] % | E[real] % | std(nominal) % |
|--------|--------|-------------|--------|---|-------------|----------|----------------|
| DGS1 | `DGS1.csv` | log(1 + y_1_Dec) | 1962–2026 | 65 | 4.65 | +0.91‡ | 3.09 |
| TB3MS compounded | `TB3MS.csv` | discount→BEY→monthly HPR→sum 12 months | 1934–2025 | 92 | 3.48 | +0.64§ | 3.13 |
| TB3MS compounded | (overlap) | same | 1962–2025 | 64 | 4.52 | +0.76‡ | — |

‡ Real return sample 1963–2025 (N=63). § Real return sample 1948–2025 (N=78).

**DGS1 vs TB3MS on overlap (1963–2025, N=63):**
- Mean difference: TB3MS is **−0.16%** lower (3m→1yr term premium)
- Correlation: 0.95
- TB3MS extra years (1934–1961): mean nominal 1.12%, mean real +0.08% — the WWII rate-pegging era with near-zero rates

### 1.4 Bond–Stock–Bill Correlations (1963–2025)

Using AAA 20yr as the bond series (current model):

| | rtb | xr | xb |
|---|-----|-----|-----|
| rtb | 1.000 | | |
| xr | 0.031 | 1.000 | |
| xb | 0.357 | 0.254 | 1.000 |

Variance–covariance (×10⁻⁴):

| | rtb | xr | xb |
|---|------|-------|------|
| rtb | 7.79 | 3.23 | 8.96 |
| xr | 3.23 | 256.0 | 36.8 |
| xb | 8.96 | 36.8 | 81.6 |

---

## 2. Data Sources

All files in `data/Thesisdata/`.

| File | Column | Freq. | Range | Description |
|------|--------|-------|-------|-------------|
| `DGS1.csv` | `DGS1` | Daily | 1962–2026 | 1-year Treasury constant-maturity yield (% p.a.) |
| `GS20.csv` | `GS20` | Monthly | 1953–2026 | 20-year Treasury CM yield (% p.a.). **NaN gap 1987–1992.** |
| `DGS30.csv` | `DGS30` | Daily | 1977–2026 | 30-year Treasury CM yield (% p.a.) |
| `AAA.csv` | `AAA` | Monthly | 1919–2026 | Moody's AAA corporate bond yield (% p.a.). ~0.78% credit spread. |
| `TB3MS.csv` | `TB3MS` | Monthly | 1934–2026 | 3-month T-bill secondary market rate (% p.a., **discount basis, 360-day**). |
| `CPIAUCSL.csv` | `CPIAUCSL` | Monthly | 1947–2026 | CPI-U, seasonally adjusted (index level). |
| `feds200628 (1).csv` | Many | Daily | 1961–2026 | Gürkaynak-Sack-Wright fitted yield curve. Skip first 9 rows. |
| `ie_data.xls` | Sheet "Data" | Monthly | 1871–2025 | Shiller: S&P 500 P, D, CAPE, GS10. |

### GSW yield curve columns

| Mnemonic | Compounding | Example | Available maturities |
|----------|-------------|---------|---------------------|
| `SVENYXX` | Continuously compounded zero-coupon yield | `SVENY10` | 01–30 |
| `SVENPYXX` | Coupon-equivalent par yield | `SVENPY10` | 01–30 |

Data start dates: `SVENY10`/`SVENPY10` from 1971-08, `SVENY20`/`SVENPY20` from 1981-07, `SVENY30`/`SVENPY30` from 1985-11.

---

## 3. Return Construction Methods

### 3.1 CCV Loglinear Approximation (coupon-bearing / par yields)

Applies to: DGS1, GS20, DGS30, AAA, SVENPY*.

These yields are **coupon-equivalent** (bond-equivalent) yields on par bonds.

```
D_t = (1 - (1 + Y_t)^{-n}) / (1 - (1 + Y_t)^{-1})       Macaulay duration
r_bond[T] = D[T-1] * log(1 + Y[T-1]) - (D[T-1] - 1) * log(1 + Y[T])
xb[T] = r_bond[T] - log(1 + y_1[T-1])
```

- Y in decimal (divide % by 100)
- n = maturity in years
- D ≈ 12–14 for n=20 at typical yields, ≈ 15–16 for n=30

### 3.2 Exact Zero-Coupon Return (SVENY* CC yields)

Applies to: SVENY01 through SVENY30.

```
r_zc[T] = n * y_cc_n[T-1] - (n-1) * y_cc_{n-1}[T]
xb[T] = r_zc[T] - log(1 + y_1[T-1])
```

- Requires two yield series: maturity n (lagged) and maturity n−1 (current)
- SVENY* are in percent in the CSV — divide by 100
- Duration of ZC bond = maturity (not comparable to par bond of same maturity)

### 3.3 TB3MS Discount Rate → Annual Log Return

TB3MS is a **discount rate** on a **360-day basis**, not a yield.

```
Step 1:  Price = 100 * (1 - d * 91/360)           d = TB3MS/100
Step 2:  BEY = (100/Price - 1) * (365/91)         bond-equivalent yield
Step 3:  monthly_HPR = (1 + BEY * 91/365)^(1/3) - 1
Step 4:  annual_log_return = sum of 12 monthly log(1 + monthly_HPR)
```

### 3.4 Common Definitions

```
y_1 = DGS1 end-of-December / 100                  1-year yield (decimal)
r_1 = log(1 + y_1)                                 log nominal bill return
pi = log(CPI_Dec_T / CPI_Dec_{T-1})                annual log inflation
rtb[T] = r_1[T-1] - pi[T]                          real bill return
xb[T] = r_bond[T] - r_1[T-1]                       excess bond return
```

Timing: row labelled year T contains yields at end of T and returns realized during T.

---

## 4. Verification (all pass)

- **Return identity:** rtb + xb = r_bond − π holds to machine precision (< 1e-16) for all 10 series.
- **Duration sanity:** Mean D = 11.8 (AAA 20yr), 12.7 (GS20), 15.5 (DGS30 30yr), 7.9 (Par 10yr), 12.9 (Par 20yr), 16.6 (Par 30yr).
- **Sign checks:** xb large and positive in 2008/2019/2020 (yields fell), large and negative in 2022 (yields rose) — correct for all series.
- **CCV vs ZC cross-check:** Correlations at matched maturity: 0.987 (10yr), 0.969 (20yr), 0.968 (30yr). ZC returns are more volatile due to higher effective duration.

---

## 5. Interpretation for the VAR Model

The lifecycle model VAR uses `xb` as the bond excess return variable. The choice of bond series determines:

1. **E[xb]** — the unconditional bond premium (z_bar[5])
2. **Var(xb)** — the Σ_rr[xb,xb] entry and all cross-covariances
3. **Sample length** — more observations for VAR estimation

**Current model:** AAA 20yr, E[xb] = +1.43%, the only series with full 1963–2025 coverage (N=63).

**Why AAA is conservative:** The ~0.78% credit spread raises the yield level → lower Macaulay duration (~11.8 vs ~12.9 for a Treasury par 20yr) → lower sensitivity to yield changes → lower measured excess return. The spread also doesn't track Treasuries 1:1.

**GS20 is worse:** E[xb] = +0.37% with NaN gap losing 7 years. The gap biases the sample.

**GSW Par 20yr looks better** (E[xb] = +2.92%, Sharpe 0.259) **but only has N=44 years starting 1982** — the secular bond bull market. Not directly comparable to AAA's 63-year estimate.

**ZC series are a different asset class:** Duration = maturity means ZC 20yr (duration 20) is not comparable to a 20yr par bond (duration ~13). Their higher E[xb] is duration compensation, not a free lunch.

**Bill choice:** DGS1 is preferred over TB3MS. The extra TB3MS years (1934–1961) are the rate-pegging era with near-zero real returns (+0.08%). TB3MS is ~16bp lower than DGS1 on overlap (3m→1yr term premium).

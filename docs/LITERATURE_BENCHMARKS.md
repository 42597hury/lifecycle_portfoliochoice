# LITERATURE_BENCHMARKS.md

Reference compilation of numerical benchmarks from the canonical lifecycle-portfolio-choice
literature. Use this file when policy function output looks suspicious — compare against
the ranges and patterns documented below to localize bugs.

This document records *concrete numbers and qualitative patterns* with citations. The
primary sources were consulted via published abstracts, repository pages, replication
notes, and survey/derivative work; numerical values are cross-checked against multiple
secondary references where direct PDF extraction failed. Page references are cited where
recoverable.

---

## 0. How to Use This Document

The "right" numerical answer depends on which model variant the agent is running. Pick
the closest benchmark from sections 1-15 below, match calibration assumptions, then
compare the reported summary statistics. The most discriminating diagnostics are:

1. **Equity share at age 25-30 (early career)** — should be at the borrowing-constraint
   ceiling (typically 100%) under Cocco-Gomes-Maenhout (2005) calibration; less under
   Benzoni-Collin-Dufresne-Goldstein (cointegration) or in Catherine (2022) with cyclical
   skewness.
2. **Equity share at retirement (age ~65)** — typically 40–60% of liquid wealth in CGM,
   slightly higher in Catherine (2022).
3. **Slope of equity share decline over working life** — roughly linear / concave decline.
4. **Consumption / cash-on-hand ratio at age 25** — typically high (~0.5–0.7 of cash-
   on-hand for liquidity-constrained young agents).
5. **Hedging demand (when return predictability or VAR is on)** — for Campbell-Viceira
   style calibrations the hedging component can roughly *double* the myopic demand at
   gamma > 1 (see Section 2).
6. **Pathologies** (Section 16): kinks at retirement; extrapolation artifacts at low
   wealth; "knife-edge" leverage when borrowing constraints bind weakly; **discrete-
   quadrature no-bankruptcy boundary** (§16.8) — the dominant failure mode of the
   *unconstrained* run in this codebase.

**This codebase's calibration in literature terms** (see §0a below) — γ=3, β=0.96,
3-asset portfolio (real bill, nominal stocks, 20-yr AAA bond) with annual VAR(1)
predictability, Guvenen-Catherine mixture-normal income, 78-period lifecycle
(start_age=22, retire_age=67, terminal_age=99). Closest reference benchmarks:
**Catherine (2022)** for the income process and lifecycle structure (§6),
**Campbell-Chan-Viceira (2003)** for the VAR-predictability hedging demand on three
assets (§2.4), **Munk-Sørensen (2010)** for stocks+bonds+bills with stochastic interest
rates (§7).

---

## 0a. Codebase-Specific Quick-Look

This section maps the calibration of the `thesisscripts` solver to the literature
sections below. See [contextfiles/DESIGN.md](DESIGN.md), [contextfiles/RETURNS.md](RETURNS.md),
[contextfiles/LABOUR.md](LABOUR.md) for the formal model spec.

| Codebase choice | Value | Closest benchmark | Important deviation |
|----|----|----|----|
| Risk aversion γ | 3.0 | CGM γ=2 (§1.3); Wachter γ=4 (§3.4) | Lower than CGM baseline (5) → expect more aggressive equity at all ages |
| Discount factor β | 0.96 | All canonical lifecycle models | None |
| Bequest motive | b_bar=10-yr annuity (Catherine eq. 21-22) | Catherine 2022 (§6) | Different from CGM "warm-glow" parameter |
| Number of risky assets | 3 (real bill, stock, 20-yr AAA bond) | Campbell-Chan-Viceira (§2.4); Munk-Sørensen (§7) | CGM has only 1 risky asset; *do not* compare bond shares to CGM |
| Real bill is uncertain | Yes (rtb = log(1+y_1) - π) | None directly — most lifecycle papers treat bills as risk-free | Major deviation; reduces effective leverage capacity vs. literature with "true riskless" asset |
| Return predictability | Annual VAR(1) on (y_1, spr, cy) | Campbell-Chan-Viceira (§2.4) | Stronger predictability than CV ⇒ **larger hedging demand** expected |
| Income process | Guvenen-Catherine mixture-normal η + ε | Catherine 2022 (§6) | Disaster-component of η behaves like cyclical skewness even though ε,η are uncorrelated with returns in this calibration |
| Income-return correlation | **Zero** in current calibration | Catherine 2022 (§6) sets nonzero countercyclical skewness | Without this correlation, the model loses the channel that makes Catherine's young agents conservative; expect *more* equity, not less, at young ages |
| Lifecycle horizon | 78 yrs (22→99) | All canonical lifecycle | Standard |
| Mortality | Earnings-dependent (Chetty et al.) | Catherine 2022 | Standard |
| Constraints | Both `constrained=True` (α≥0, sum≤1) and `constrained=False` runs supported | CGM uses constrained; Wachter / Campbell-Viceira analytical use unconstrained | **Unconstrained run is intrinsically pathological** (see §16.8) |
| State-grid size | 5×5×5 = 125 (smoke); 7×7×7 = 343 (production) | Munk-Sørensen uses 21³ for 3 states | Coarser than literature; see §16.4-16.5 |
| Stock-residual quadrature K_xr | 5 (default) | CV uses continuous solutions | **Critical for unconstrained mode** (§16.8) |

**Plain-language version:**
- The closest paper is Catherine (2022) — same income process, similar lifecycle, but γ=6
  there vs γ=3 here. Lower γ here ⇒ everything Catherine reports should be amplified
  toward more equity.
- Adding the VAR predictability on top of Catherine's income process is novel — no
  existing paper combines them. Expect intertemporal hedging demand on top of
  Catherine's income-driven shifts.
- The 3-asset structure (bills/bonds/stocks) matches Campbell-Chan-Viceira and
  Munk-Sørensen. The specific state vector (y_1, spr, cy) is closest to CCV.
- The **constrained** run should produce CGM-style monotone-decreasing equity share
  with possible small hump from the VAR-hedging effect.
- The **unconstrained** run hits the no-bankruptcy boundary across the wealth grid (see
  §16.8) — its output is *not* a literature benchmark and should not be interpreted as
  Merton/Campbell-Viceira leverage.

---

## 1. Cocco, Gomes, and Maenhout (2005) — "Consumption and Portfolio Choice over the Life Cycle"

**THE canonical lifecycle benchmark.** Almost every subsequent paper benchmarks against
this calibration. Replicate this *first* before adding bells and whistles.

- Journal: *Review of Financial Studies*, **18**(2), 491–533 (Summer 2005).
- DOI: `10.1093/rfs/hhi017`
- URL: https://academic.oup.com/rfs/article-abstract/18/2/491/1599892
- Replication code (Econ-ARK REMARK): https://econ-ark.org/materials/cgmportfolio/
- Working paper: https://feb.kuleuven.be/research/economics/ces/documents/DPS/1998/DPS9805.pdf

### 1.1 Calibration (baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Discount factor β | 0.96 | annual |
| Risk aversion γ | 5 (baseline); also 2 and 10 reported | CRRA over consumption |
| Real riskless rate (bond/bill) | 0.02 (2%) | constant |
| Mean log equity excess return (premium) | 0.04 (4%) | moderate, conservative |
| Equity return std-dev σ_R | 0.18 (18%) | annual |
| Correlation income shock ↔ equity return | small / 0 (baseline) and 0.15-0.37 (sensitivity) | |
| Permanent income shock variance σ²_η (or σ²_ω) | 0.0106 | high-school PSID estimate |
| Transitory income shock variance σ²_ε | 0.0738 | high-school PSID estimate |
| Retirement replacement rate λ | ≈ 0.68 (high school); profession-specific values reported | |
| Working ages | 20–65 | |
| Maximum age | 100 | with mortality from US life tables |
| Borrowing/short-sale constraint | yes — α ∈ [0,1], B ≥ 0 | binds early in life |
| Income deterministic profile f(t,Z) | 3rd-order age polynomial × education dummies | from PSID |

(Cross-references: variance values "0.0106 / 0.0738" appear repeatedly in citing
papers — Cocco/Gomes/Maenhout (2005), and reproduced by Bagliano-Fugazza-Nicodano on
the Carlo Alberto WP series. Replacement rate λ ≈ 0.68 high-school, ≈ 0.94 college
in CGM Table 1.)

### 1.2 Headline equity share by age (γ = 5, baseline)

CGM produce a *roughly decreasing* age profile in the optimal *unconditional* equity
share. The simulated mean share is:

| Age | Equity share (% of liquid wealth) | Notes |
|-----|------------------------------------|-------|
| 20–30 | ≈ 100% | borrowing constraint binds; agent would lever if allowed |
| 30–40 | ≈ 90–100% | constraint typically still binding for median agent |
| 40–50 | ≈ 80% | constraint relaxes as financial wealth accumulates |
| 50–60 | ≈ 65–75% | declining toward retirement |
| 60–65 | ≈ 55–65% | end-of-working-life |
| 65+ | ≈ 50% (roughly flat through retirement) | with social-security-like floor income |

The paper itself summarises: *"Optimal share of wealth in stocks ranges from 60 to 100
percent depending on the age of the household, even with γ=10 and a moderate equity
premium."* (Cocco-Gomes-Maenhout 2005, Abstract / Section 4.)

The Gomes (2020) survey describes the canonical CGM profile as *"average optimal share
in equity declines linearly to about 60% at retirement, after which it is roughly
constant."*

### 1.3 Sensitivity to γ

- **γ = 2:** equity share at 100% essentially everywhere (agents would lever unless
  constrained); model approaches Merton with very flat tilt.
- **γ = 5 (baseline):** profile described in 1.2.
- **γ = 10:** profile shifts down ~15-25 pp at every age — equity share roughly 60-80%
  through working life, ~40-50% at retirement. Still well above SCF empirical levels.

### 1.4 Consumption / wealth patterns

- **Hump-shaped consumption** profile peaking near retirement (~age 60-65), tracking
  the deterministic income hump.
- **Wealth-to-permanent-income** ratio rises from near 0 in the 20s to ~3-5 by retirement
  and falls afterward (run-down).
- **Consumption/cash-on-hand** ratio: very high in 20s (≈ 0.5-0.7 due to
  liquidity-constrained behavior near borrowing limit); falls toward ≈ 0.05-0.15 of
  cash-on-hand near retirement; rises again in retirement.
- **Saving rate** is hump-shaped, peaking around age 50.

### 1.5 Asset structure

CGM use **only one risky asset (stocks) and one safe asset (riskless bond/bill)**. They
do *not* model long-term bonds separately. To benchmark a model with bonds *and* bills
(plus VAR predictability), use Campbell-Chan-Viceira (2003) and Munk-Sørensen (2010)
instead (Sections 2 and 7).

---

## 2. Campbell & Viceira (1999, QJE) and Campbell-Chan-Viceira (2003, JFE)

Foundational paper on intertemporal hedging demand under VAR(1) return predictability.
Together with the 2002 OUP book "Strategic Asset Allocation," this is the reference
point for *how big* hedging demand should be in a VAR(1) world.

- Campbell, Viceira (1999), *QJE* **114**(2), 433–495.
  https://academic.oup.com/qje/article-abstract/114/2/433/1844221
- Campbell, Chan, Viceira (2003), *JFE* **67**(1), 41–80.
  https://www.nber.org/papers/w8566
- Book: Campbell & Viceira (2002), *Strategic Asset Allocation*, OUP.
  https://global.oup.com/academic/product/strategic-asset-allocation-9780198296942

### 2.1 Setup
- Infinitely-lived investor (no labor income; pure financial portfolio).
- Epstein-Zin-Weil utility; log-linearised budget constraint and Euler equations.
- VAR(1): log excess equity return predicted by log dividend-price ratio (and short-rate
  in extensions).
- Calibrated to (a) post-war U.S. quarterly returns and (b) long sample 1890-1990.

### 2.2 Headline result on hedging demand (Campbell-Viceira 1999)

> "When the model is calibrated to U.S. stock market data it implies that intertemporal
> hedging motives greatly increase, and may even **double**, the average demand for
> stocks by investors whose risk-aversion coefficients exceed one."
> — Campbell-Viceira (1999), Abstract.

Concretely, at γ = 5, the optimal *equity share* is roughly **2× the myopic Merton
share**. The hedging-demand component is the *additional* allocation due to the negative
correlation between innovations to the dividend-price ratio (state) and equity returns:
when stocks fall, the dividend-price ratio rises, so future expected returns rise — long
horizon investors want to hold *more* equity to hedge this.

### 2.3 Numerical magnitudes (typical CV calibration)

For γ ∈ {1, 2, 5, 10, 20}:

- Myopic Merton share ≈ (μ - r) / (γ σ²). With μ-r ≈ 0.06 and σ ≈ 0.18: myopic = 1.85/γ.
  - γ=2 → myopic ≈ 92%
  - γ=5 → myopic ≈ 37%
  - γ=10 → myopic ≈ 18%
- **Total optimal share** (myopic + hedging) at the long-run mean of d-p ratio:
  - γ=2 → ≈ 100% (still constrained at top)
  - γ=5 → ≈ 70–90% (roughly 2× myopic)
  - γ=10 → ≈ 40–55% (roughly 2.5× myopic)
- Hedging demand grows **more than proportionally** in γ (because the hedging term is
  scaled by 1 - 1/γ, while myopic demand falls as 1/γ).

For γ = 1 (log utility) the hedging demand is **zero** (myopic = total). Useful sanity
check: at γ = 1 a VAR(1) solver should reproduce the constant Merton share evaluated at
the conditional mean.

### 2.4 Multivariate version (Campbell-Chan-Viceira 2003)

With three assets (stocks, long-term nominal bonds, T-bills) and two state variables
(d-p ratio, short rate) in a VAR(1):

- *"Predictability of stock returns greatly increases the optimal demand for stocks"*
  (about 2× at typical γ).
- *"The dividend yield generates the largest hedging demand among a wider set of predictor
  variables"* — driven by the strong negative correlation between innovations to the
  dividend yield and stock returns.
- Long-term nominal bonds carry hedging demand against real-rate risk; their share grows
  with γ (more conservative investors hold more long bonds).
- *"Long-term inflation-indexed bonds greatly increase the utility of conservative
  investors."*

### 2.5 Welfare costs of ignoring predictability

CV (1999): Failing to time or to hedge can cause **certainty-equivalent welfare losses
of up to ~50% of wealth** in some calibrations relative to the optimal policy. The
optimal strategy can roughly **double** the certainty-equivalent wealth versus the
myopic strategy in some historical periods.

---

## 3. Wachter (2002) — "Portfolio and Consumption Decisions Under Mean-Reverting Returns"

Closed-form solution under complete markets. Key reference for *what hedging demand
should look like analytically* under a single-factor mean-reverting state (e.g., the
log price–dividend ratio).

- *Journal of Financial and Quantitative Analysis* **37**(1), 63–91 (2002).
  https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/portfolio-and-consumption-decisions-under-meanreverting-returns-an-exact-solution-for-complete-markets/898F0A569E28BDFAA8396AA8FD2822C0
- PDF: https://finance.wharton.upenn.edu/~jwachter/research/Wachter2002jfqa.pdf

### 3.1 Setup
- Single risky asset; log expected excess return follows an Ornstein-Uhlenbeck process
  (mean-reverting); markets *complete* (so there is a redundant bond hedging the state).
- CRRA utility over consumption, infinite horizon (or finite with terminal wealth).
- Consumer chooses portfolio + consumption.

### 3.2 Calibration (annual, U.S. data)

| Parameter | Value | |
|-----------|-------|--|
| Mean-reversion speed κ | 0.142 | matches autocorrelation 0.865 of price-dividend ratio |
| Long-run expected log excess return | ≈ 4% | |
| Equity volatility σ_S | ≈ 18% | |
| State innovation correlation ρ_SX | ≈ −0.93 | strong negative (return ↑ ⇒ d-p ↓) |

### 3.3 Hedging demand: closed form

Wachter shows the optimal share is
α(t) = (1/γ) · (μ−r)/σ²  +  (1 − 1/γ) · Cov-hedging term (X-dependent).

Key takeaway:
- For γ > 1, the second term is positive (mean-reverting returns reward longer horizon).
- *"Mean reversion increases the demand for stocks whenever the risk premium is greater
  than zero"* (Wachter 2002).
- Allocation has an interpretation as a **weighted average analogous to bond duration**:
  the consumer's effective horizon is the *Macaulay duration* of consumption.
- Welfare analyses show the approximate Campbell-Viceira solution is very close to the
  exact Wachter solution outside the upper tail of the state space — a useful cross-check.

### 3.4 Magnitudes
- For γ = 4 with the calibration above and mean dividend yield, the optimal share is
  **roughly 2× the myopic Merton share**, consistent with Campbell-Viceira (1999).
- At γ = 1 (log utility) hedging demand vanishes — a critical sanity check.
- As γ → ∞ the consumer holds the asset that hedges X perfectly (the "real bond").

---

## 4. Cocco, Gomes, Maenhout (2005) Extended: Mean Reversion + Lifecycle

### Campbell, Cocco, Gomes, Maenhout, Viceira (2001, "Stock Market Mean Reversion and the Optimal Equity Allocation of a Long-Lived Investor")
- *European Finance Review* **5**(3), 269–292.
  https://academic.oup.com/rof/article-abstract/5/3/269/1575422
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=237175

Combines CGM-style numerical solver with VAR(1) predictability. Key findings:
- Numerical solutions agree with Campbell-Viceira (1999) approximate analytical solutions
  *"except at the upper extreme of the state space where both the numerical consumption
  and portfolio rules flatten out"* — diagnostic for grid behaviour.
- Borrowing/short-sales constraints flatten the upper-tail policy and make the "double
  hedging" effect smaller in absolute terms but still substantial.

### Stock market mean reversion in the lifecycle (later work, e.g. Gomes-Michaelides type extensions):
- Adding mean reversion to a CGM lifecycle increases equity share by an extra ~10-20 pp
  for middle-aged agents at γ = 5.
- Effect is largest at intermediate ages where wealth has accumulated but horizon is
  still long enough for mean-reversion to matter.

---

## 5. Gomes & Michaelides (2005) — "Optimal Life-Cycle Asset Allocation: Understanding the Empirical Evidence"

Shows that adding **moderate preference heterogeneity** + **fixed stock-market entry
cost** + **Epstein-Zin** to a CGM-style lifecycle model can replicate observed
participation rates *and* conditional shares.

- *Journal of Finance* **60**(2), 869–904.
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2005.00749.x
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=721587
- LSE replication: http://eprints.lse.ac.uk/193/

### 5.1 Mechanism
> "Households with low risk aversion smooth earnings shocks with a small buffer stock of
> assets, and consequently most of them (optimally) never invest in equities. Therefore,
> the marginal stockholders are (endogenously) more risk averse, and as a result they do
> not invest their portfolios fully in stocks."
> — Gomes-Michaelides (2005), Abstract.

### 5.2 Calibration
- Epstein-Zin EIS ≈ 0.5; γ heterogeneity: bimodal mixture (e.g. low γ ≈ 1.5 and high γ ≈ 5).
- Per-period or one-time fixed participation cost on stocks (small fraction of permanent
  income, calibrated to match aggregate participation).
- All other CGM parameters (β=0.96, equity premium 4%, σ=18%, etc.).

### 5.3 Results
- **Conditional equity share** (among stockholders) is **hump-shaped**: low for young
  participants (~50–60%), peaking around middle age (~70–80%), then declining.
- **Unconditional participation** is hump-shaped: low for the young, peaks near
  retirement, declines after.
- This is the closest "structural" match to the SCF/PSID empirical patterns (see §13).

---

## 6. Catherine (2022) — "Countercyclical Labor Income Risk and Portfolio Choices over the Life Cycle"

The paper this codebase is being benchmarked against. Adds **cyclical skewness** in the
idiosyncratic income distribution: when equity markets perform poorly, the *left tail* of
income shocks gets fatter — a "disaster" risk dimension correlated with stocks.

- *Review of Financial Studies* **35**(9), 4016–4054 (Sept 2022).
  https://academic.oup.com/rfs/article-abstract/35/9/4016/6482757
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2778892

### 6.1 Headline calibration
| Parameter | Value | |
|-----------|-------|--|
| Risk aversion γ | **6** | structurally estimated |
| Yearly stock-market participation cost | **$250** | structurally estimated |
| Equity premium | calibrated standard ≈ 4–6% | |
| Cyclical skewness of idiosyncratic income | estimated from PSID/admin data | |
| Discount factor β | standard CGM-style ≈ 0.96 | |

### 6.2 Headline results
- Model **matches** evolution of (i) mean wealth, (ii) participation, and (iii)
  conditional equity share over the lifecycle.
- Cyclical skewness explains:
  1. Low stock-market participation among **young** households with modest financial
     wealth (their human capital is exposed to the same disasters as stocks).
  2. **Slight increase** of conditional equity share until retirement — the opposite of
     CGM's monotonic decline. As retirement approaches and human capital shrinks,
     workers become *less* exposed to disaster risk and can take on *more* equity.
  3. Why renters invest less in stocks than homeowners.
- *"Cyclical skewness increases the equity premium by at most 0.5%."*
- Relationship between cyclical skewness and stock holdings: **strongest for workers
  with high share of human capital**, vanishes near retirement.

### 6.3 Equity-share-by-age pattern (Catherine 2022, conditional on participation)

Empirically and in the model:
- Age 25–35: ~40–55% (well below CGM)
- Age 35–50: ~55–65%
- Age 50–60: peak ~60–70%
- Age 60–65: ~55–65%
- Retirement: ~45–55%

Pattern is **hump-shaped or slightly upward-sloping until retirement**, then declines —
matches Fagereng-Gottlieb-Guiso (2017) Norwegian data and SCF.

### 6.4 Companion: Catherine, Sodini, Zhang (2024 *JF*) — "Countercyclical Income Risk and Portfolio Choices: Evidence from Sweden"

- *Journal of Finance* **79**(3), 1755–1788.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13341
- WP: https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2022/03/Paper3_Catherine.pdf

Confirms predictions empirically using Swedish administrative data:
> "Workers facing higher left-tail income risk when equity markets perform poorly have
> lower portfolio equity shares. The relationship between cyclical skewness and stock
> holdings increases with the share of human capital in a worker's total wealth and
> vanishes as workers get closer to retirement."

---

## 7. Munk & Sørensen (2010) — "Dynamic Asset Allocation with Stochastic Income and Interest Rates"

Closest paper to a *VAR-augmented* lifecycle with bonds and stocks.

- *Journal of Financial Economics* **96**(3), 433–462.
  https://www.sciencedirect.com/science/article/abs/pii/S0304405X10000140
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=676021

### 7.1 Setup
- Stochastic short-rate (Vasicek-type) + stochastic labor income whose **expected growth
  is affine in the short rate**.
- Three assets: stocks, long-term bonds, cash.
- Both unconstrained (complete markets) and constrained (incomplete markets, borrowing
  + short-sales restrictions) versions.

### 7.2 Calibration & results
- Slope of income-growth-on-rate function ζ calibrated from PSID; "substantial cross-
  sectional heterogeneity" in ζ.
- The slope ζ is **crucial** for valuation and riskiness of human capital and for
  optimal stock/bond/cash allocation.
- Bond holdings (long-term nominal bond) are large and rise with γ; this is the closest
  benchmark to the ASK codebase's bond/bill split.
- Hedging demand for bonds (against real-rate shocks) is significant for γ ≥ 5.

---

## 8. Viceira (2001) — "Optimal Portfolio Choice for Long-Horizon Investors with Nontradable Labor Income"

Approximate analytical solution under labor income as background risk.

- *Journal of Finance* **56**(2), 433–470.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00333
- NBER WP: https://www.nber.org/papers/w7409
- PDF: https://www.nber.org/system/files/working_papers/w7409/w7409.pdf

### 8.1 Headline results
- Fraction of *financial wealth* in stocks is **strictly larger for employed than for
  retired** investors (when income risk is uncorrelated with stock risk).
- Mean-preserving increase in income variance:
  - Increases the willingness to **save** (precautionary motive).
  - Reduces the willingness to **hold the risky asset**.
- For γ = 5, equity premium 4%, σ = 18%:
  - Retired investor (no income): myopic Merton share ≈ 37%.
  - Employed investor with PV(income) = 5× financial wealth: equity share on financial
    wealth ≈ 100% (constraint binds) — labor income substitutes for the safe asset.
- Provides closed-form-ish benchmark for **how much human capital "displaces"** the safe
  asset from financial portfolios.

---

## 9. Benzoni, Collin-Dufresne, Goldstein (2007) — "Portfolio Choice over the Life-Cycle when the Stock and Labor Markets Are Cointegrated"

The key counter-narrative: cointegration between dividends and labor income means
*young* investors hold *less* equity than *middle-aged* ones — generating a hump.

- *Journal of Finance* **62**(5), 2123–2167.
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01270.x
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=875984
- Chicago Fed WP: https://www.chicagofed.org/-/media/publications/working-papers/2007/wp2007-11-pdf.pdf

### 9.1 Setup
- Aggregate labor income and aggregate stock dividends are **cointegrated** (share a
  common stochastic trend).
- Idiosyncratic income shocks are i.i.d. and orthogonal.

### 9.2 Calibration (illustrative)
- Equity premium 4–6% (results robust within range).
- Cointegration parameter calibrated to match observed long-run Wage/Dividend ratio
  stationarity (with weak signal in the data).
- Equity vol ~18%; standard CRRA preferences.

### 9.3 Headline results
- *"Young investors should take substantial **short positions** in the stock market"*
  because over a long horizon their human capital is "stock-like" (cointegrated with
  dividends).
- For older agents (short horizon to retirement), cointegration has "insufficient time
  to act" — human capital becomes "bond-like" — they hold *more* stocks.
- **Hump-shaped life-cycle equity holdings**: rising in the 30s/40s, peaking in the 50s,
  declining at retirement — *consistent with the empirical pattern* in SCF/Norwegian
  data.

### 9.4 Implications for solver checks
- If your solver produces a hump (instead of CGM's monotone decline), check whether you
  have implicitly imposed a cointegration-like correlation (e.g. permanent-income shock
  correlated with equity at long horizons).
- Hump magnitude in BCG: roughly **40–50% at age 25 → 80–90% at age 50 → 50–60% at age 65**
  for γ = 5, depending on the strength of cointegration.

---

## 10. Polkovnichenko (2007) — "Life-Cycle Portfolio Choice with Additive Habit Formation Preferences and Uninsurable Labor Income Risk"

- *Review of Financial Studies* **20**(1), 83–124 (Jan 2007).
  https://academic.oup.com/rfs/article/20/1/83/1588217
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=419180

### 10.1 Setup & headline result
- CGM-style lifecycle but with **additive (external) habit formation** in preferences.
- Standard equity premium / vol calibration.
- Habit creates a *consumption floor* that the agent strongly defends.

### 10.2 Findings
- Habit formation makes **young investors more conservative** (they have low wealth-to-
  habit ratios — a fall in wealth would push consumption near the habit floor).
- Generates a **rising equity share early in life**, then declining — i.e. a hump,
  rather than CGM's monotone decline.
- For some low-to-moderate-wealth households the share allocated to stocks **increases
  with wealth** (consistent with empirical evidence).
- Useful to check: if your model includes habit and the equity share is monotone
  decreasing, the habit channel may not be active.

---

## 11. Bodie, Merton, Samuelson (1992) — "Labor Supply Flexibility and Portfolio Choice in a Life-Cycle Model"

- *Journal of Economic Dynamics and Control* **16**(3-4), 427–449.
  https://www.sciencedirect.com/science/article/pii/016518899290044F
- NBER WP 3954: https://www.nber.org/papers/w3954

### 11.1 Setup
- Continuous-time lifecycle with endogenous labor supply (intensive + extensive margin
  including retirement timing).
- Quadratic disutility of labor; CRRA over consumption-leisure aggregate.

### 11.2 Headline results
- **"Human capital as bond"** intuition formalised: when human capital is riskless,
  young investors with high HC should hold ~100% equity in financial wealth (often more
  if they could lever their HC).
- *"Ability to vary labor supply ex post induces the individual to assume greater risks
  in his investment portfolio ex ante"* — labor flexibility is itself a risk-bearing
  capacity.
- Explains **decreasing equity share with age** (loss of labor flexibility, shrinking HC).
- Magnitudes: young workers' HC is "many times" non-human wealth — implies ~100%-equity
  in financial portfolio is rational for the young when HC is bond-like.

This is the *original* reference for the "treat human capital as a riskless bond"
heuristic that pervades the literature.

---

## 12. Fagereng, Gottlieb, Guiso (2017) — "Asset Market Participation and Portfolio Choice over the Life-Cycle"

Best **empirical** benchmark for participation and conditional shares from administrative
(error-free) Norwegian tax data.

- *Journal of Finance* **72**(2), 705–750.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12484
- WP: https://www.eief.it/files/2017/03/guiso_fagereng_gottlieb_joff_2017.pdf
- ECB WP version: https://www.ecb.europa.eu/events/pdf/conferences/131017/papers/Session_3_Fagereng.pdf

### 12.1 Data
- 20% random sample of 1995 Norwegian households (~164,000 hhs), tracked through 2009
  via Norwegian Tax Registry (NTR).
- Error-free: directly observed asset holdings, not survey reported.

### 12.2 Headline empirical patterns

**Participation rate (probability hh holds any risky asset)** — *"hump-shaped":*
- Age 25: ~25–30%
- Age 35: ~45–50%
- Age 45: ~55%
- Age 55: ~58–60%
- Age 60–65 (peak): ~60% (just before retirement)
- Age 70: ~50%
- Age 80: ~30%
- Falls "almost linearly" from age 65 to age 80.

**Conditional equity share (among participants):**
- Young (20s–30s): starts at ~35% and rises; "high and flat for the young"
- 30s–early 50s: roughly flat at **~50%**
- 50s onward: declines by **~1 percentage point per year** until retirement
- 65+: continues to decline

The "double adjustment" finding: as households age they (i) **rebalance portfolio
composition** away from stocks pre-retirement, and (ii) **exit** the stock market post-
retirement.

### 12.3 Structural model fit
- Best-fit calibration: **relatively large** γ (much larger than CGM's 5; consistent
  with empirical estimates of γ ≈ 8-10), **small** per-period participation cost, and
  a yearly probability of a **large stock-market loss** (rare disaster) calibrated to
  the frequency of crashes in Norway.
- Without the disaster term, a CGM-style calibration **cannot generate the observed
  participation rates** at any plausible γ.

### 12.4 Diagnostic for solvers
- A solver should reproduce **conditional shares ≈ 50% in the 30s/40s** and a
  **~1pp/year decline in the 50s/60s** when calibrated to Catherine (2022) or
  Fagereng-Gottlieb-Guiso targets.
- If the conditional share is monotonically decreasing from 100% at age 25 (CGM-style),
  the model has either too low γ or no participation cost / no disaster.

---

## 13. Other Empirical Patterns (SCF / PSID)

Mature stylised facts from Survey of Consumer Finances (US) and PSID:

- **Stock market participation** (any risky asset, including via MFs): hump-shaped, peaks
  near retirement.
- *"Likelihood of stock ownership… increases with age until age 61."*
- *"Conditional equity shares peak at around age 50."*
- ~50% of US households do **not** participate in the stock market at all (direct or via
  MFs) — the "non-participation puzzle."
- Conditional median equity share among participants: ~30–60% depending on the year and
  age cohort.
- Reference: Gomes (2020) Annual Review survey; SCF data:
  https://www.federalreserve.gov/econres/scfindex.htm

---

## 14. Gomes (2020) — "Portfolio Choice over the Life Cycle: A Survey"

- *Annual Review of Financial Economics* **12**, 277–304.
  https://www.annualreviews.org/content/journals/10.1146/annurev-financial-012820-113815
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3744669

### Aggregated qualitative benchmarks from the survey
1. With incomplete markets (non-tradeable income + borrowing constraints), the canonical
   prediction is a **monotonically decreasing equity share** from ~100% (constrained) when
   young to ~50–60% at retirement.
2. **Average optimal share in equity declines linearly** to about 60% at retirement,
   roughly constant thereafter.
3. Empirical participation is **far below** model-predicted participation in baseline CGM —
   resolved by participation costs (Gomes-Michaelides 2005), disaster risk (Fagereng et
   al. 2017; Catherine 2022), or moderate Epstein-Zin EIS heterogeneity.
4. Bonds enter portfolios meaningfully only when (a) interest-rate risk is modeled
   (Munk-Sørensen 2010) or (b) there is return predictability (Campbell-Chan-Viceira
   2003).
5. **Housing** crowds out stocks: more relevant for young, low-net-worth households
   (Cocco 2005).
6. Background risks (health, longevity, unemployment) shift shares **down** at all ages.

---

## 15. Cocco (2005) — "Portfolio Choice in the Presence of Housing"

Side reference for housing/wealth interaction.

- *Review of Financial Studies* **18**(2), 535–567.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=258428

### Headline
- House price risk **crowds out stockholdings**, especially for low-net-worth households.
- Younger investors with house equity have less liquid wealth to invest in stocks
  (mechanical effect of house being illiquid + leveraged).
- In the model and the data, **leverage is positively correlated** with stockholdings —
  conditional on owning a house, leveraged owners hold more stocks (because they have
  larger gross balance sheets).

---

## 16. Common Numerical Pathologies in Lifecycle Solvers

Things to look for if policy functions look weird, with literature pointers:

### 16.1 Kinks at retirement
- Optimal consumption and portfolio rules often have small **kinks at the retirement
  boundary** (age 65 typically) because the labor income process changes regime.
- Cubic spline interpolation across the kink can produce spurious oscillations — use
  **PCHIP (shape-preserving)** interpolants instead. (See lifecycle solver discussions
  including Iskhakov-Jørgensen-Rust-Schjerning DC-EGM literature.)

### 16.2 Borrowing-constraint corner solutions
- At low cash-on-hand, the constraint α ≤ 1 (no leverage) typically binds for young
  agents — equity share **glued to 100%**. This is the *correct* CGM behaviour.
- If your solver produces e.g. 95% (slightly off from 100%) at low wealth, suspect a
  finite-grid artifact: the constraint isn't actually being detected.
- If your solver produces equity share **declining at very low wealth** (toward 0%),
  this is wrong under standard CGM; it can happen with bad extrapolation in the
  consumption policy near zero wealth.

### 16.3 Extreme leverage at low wealth (when constraint is relaxed)
- Without a leverage cap, optimal share can blow up → ∞ at small financial wealth (because
  the agent treats human capital as a riskless asset and wants to lever it).
- Classic check: turn off labor income (set f(t) = 0) → solver should reproduce the
  Merton constant share at every age.

### 16.4 Discretization-driven artifacts
- **Wealth-grid spacing**: use *exponential* spacing (denser at low wealth where the
  value function is highly concave). Linear grids cause large interpolation errors at
  low wealth.
- **Income-shock discretization**: too few Gauss-Hermite/Tauchen nodes → spurious
  smoothing of the policy. Standard recommendation: 7–11 nodes for normal shocks,
  15+ when correlations matter (e.g. with stock returns).
- **Return discretization**: equity return should be discretized to many nodes (≥9)
  because policy is more nonlinear in returns than in income.
- **Time step**: annual is standard; quarterly only matters for very precise hedging
  demand calculations.

### 16.5 VAR(1) state grid
- The state variable (e.g. log d-p ratio) needs a wide enough grid to cover
  ~3 unconditional standard deviations on each side of the mean. Wachter (2002) and
  Campbell-Chan-Viceira (2003) note that policy rules **flatten in the upper extreme**
  of the state space — this is real, not a numerical artifact.
- However, *artifactual* flattening occurs when the grid is too narrow and you're
  extrapolating policy.

### 16.6 Convergence diagnostics
- For finite-horizon lifecycle: backward induction should produce **monotonically
  decreasing equity share** as a function of age (in CGM-baseline). Non-monotonicity at
  the same wealth level across consecutive ages typically indicates a discretization or
  interpolation bug.
- Consumption rule should be **monotonically increasing in cash-on-hand** at every age.
  Violations point to non-concavity in V or interpolation overshoot.

### 16.7 Unit-of-account bugs
- CGM normalises by *permanent income* `P_t`. Make sure consumption, wealth, and
  policy ratios are reported in the same normalisation.
- The "wealth-to-income" ratio at retirement in CGM peaks around **3–5×** annual permanent
  income. Values of ~10× or ~0.1× indicate a bug.

### 16.8 Discrete-quadrature no-bankruptcy boundary (UNCONSTRAINED runs)

This is **the dominant pathology of the `constrained=False` solver in this codebase**.
Documented during the April 2026 investigation of the age-22 leverage ramp; see
`scripts/investigation/_invest_foc.py`,
`scripts/investigation/_invest_kink.py`,
`scripts/investigation/_invest_age_ramp.py`.

**Mechanism.** The agent integrates over a *finite* set of joint return-quadrature nodes
(in this codebase: `n_state_quad × n_ret_quad` = 8 × 45 = 360 total nodes per FOC). With
unconstrained portfolio choice and CRRA `γ > 1`, the agent wants to lever the predictable
component of returns against the small residual variance after state-conditioning
(stock residual std ~3.1% vs. unconditional 16%). Optimal leverage is then bounded only
by the requirement that `R_port` remain non-negative at *every* quadrature node — beyond
that, marginal utility at the worst node explodes and the FOC sign flips.

**Diagnostic signatures.**
- Saved `α_s, α_b` plateau over a wide age range (22 → ~40 in this codebase) and
  *across all states*, indicating the policy is glued to a numerical boundary.
- `min(R_port over quadrature nodes)` is extremely small (~1e-6 in this codebase) at
  the saved policy.
- A 1% perturbation upward in `α_s` or `α_b` flips the FOC sign by 13+ orders of
  magnitude (numerical kink).
- The plateau scales with the support of the discrete return distribution, not with
  the structural parameters (γ, equity premium, etc.).

**Root cause.** With finite quadrature nodes and unconstrained CRRA:
- The continuous-distribution problem either has no interior optimum (γ ≤ 1) or a
  bounded optimum only if returns have unbounded support.
- The discrete-quadrature problem replaces unbounded return support with a finite
  support, so the agent levers up to the support edge.
- Adding more quadrature nodes (especially `K_xr` for the stock residual) *will* push
  the boundary further out — but the policy is then driven by the *worst node*, not
  by economic mean-variance tradeoffs.

**Test for the pathology.**
1. Compute the saved `(α_s, α_b)` at each `(age, iz, i_s, iw)`.
2. Loop over all `n_state_quad × n_ret_quad` joint quadrature nodes; compute `R_port`.
3. If `min(R_port) < 1e-3` (or ~ε of `wealth_grid[0]`) for many cells, the policy is
   on the boundary, **not** an economic optimum.

**What to do.**
- Treat unconstrained policies as upper bounds, not literal optima.
- Compare to the Wachter (2002) closed-form (§3) — if the saved policy is far above
  Wachter's analytical hedging demand, the boundary is binding.
- Reduce γ to 1 (log utility); hedging demand should *vanish* (Wachter); if the saved
  policy still shows huge leverage, the pathology is independent of γ.
- Add an explicit borrowing constraint (Carroll's natural debt limit using the PV of
  human capital) to close the unconstrained problem properly.
- For *any* analysis intended as economic interpretation, use the constrained run.

**Citation context.** This pathology is acknowledged but not extensively discussed in
the lifecycle literature, because most canonical papers (CGM, Catherine 2022, Gomes-
Michaelides) impose `α ∈ [0,1]` and `B ≥ 0` from the outset. The closest theoretical
analogue is the "incompleteness premium" in Campbell-Chan-Viceira (2003), where the
authors note that adding short-sale constraints "prevents allocation rules from
becoming explosive at extreme states." See also Iskhakov-Jørgensen-Rust-Schjerning
(DC-EGM) for the closely-related issue of non-concave continuation values that emerge
from discretization.

### 16.9 Constrained vs. unconstrained sanity ratios
A useful internal sanity check, independent of literature benchmarks:
- The **constrained-run equity share** at age 22, mid-state, mid-z, high-W should be
  near 1.0 (constraint binding) if γ ≤ 5.
- The **unconstrained-run** at the same point will be much higher (in this codebase,
  4-5× under §16.8 boundary).
- If the constrained share is *less than* ~80% at age 22, suspect a bug in the FOC or
  Newton (constraint should bind for the typical young agent).
- If the unconstrained share is *less than* the constrained share at any point, that's
  a serious bug.

### References for solver pathologies / numerics
- Iskhakov, Jørgensen, Rust, Schjerning — DC-EGM. https://www.qeconomics.org/ojs/forth/643/643-3.pdf
- Carroll — original EGM
- Floswald lecture notes (DC-EGM): https://floswald.github.io/NumericalMethods/assets/tex/dcegm_DSE2019.pdf
- Cleveland Fed — "An Investigation into Numerical Solutions to Discrete and Continuous Choice Lifecycle Models": https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2023/wp2310.pdf

---

## 17. Quick-Reference Sanity-Check Table (γ = 5, baseline)

| Diagnostic | Expected range | Source |
|------------|----------------|--------|
| Equity share, age 25 | 95–100% (constrained) | CGM 2005 |
| Equity share, age 35 | 90–100% | CGM 2005 |
| Equity share, age 45 | 75–95% | CGM 2005 |
| Equity share, age 55 | 65–80% | CGM 2005 |
| Equity share, age 65 (working) | 55–65% | CGM 2005 |
| Equity share, age 75 (retired) | 45–55% (≈ flat) | CGM 2005 |
| Conditional share (Catherine 2022 / Norway) at age 35 | 45–55% | Fagereng et al. 2017; Catherine 2022 |
| Conditional share at age 50 | ~50% | Fagereng et al. 2017 |
| Conditional share decline 50→65 | ~1pp/year | Fagereng et al. 2017 |
| Hedging demand share at γ=5, mean d-p (Campbell-Viceira) | ~equal to or larger than myopic | CV 1999, Wachter 2002 |
| Total share = myopic + hedging at γ=5, CV-VAR | ≈ 70–90% | CV 1999 |
| Hedging demand at γ=1 | ZERO | Wachter 2002 (analytical) |
| Catherine (2022) conditional share at age 35 | ~50–55% | Catherine 2022 |
| Catherine (2022) conditional share at age 60 | ~60–65% (slightly *increasing*) | Catherine 2022 |
| Wealth-to-permanent-income at retirement | 3–5× | CGM 2005 simulations |
| Consumption / cash-on-hand, age 25 | 0.5–0.7 (high) | CGM 2005 |
| Consumption / cash-on-hand, age 60 | 0.05–0.15 (low) | CGM 2005 |
| Sweden CSZ (2024) participation, mid-career | matched ≈ 60% | Catherine-Sodini-Zhang 2024 |

---

## 18. Master Source List (URLs / DOIs)

1. **Cocco, Gomes, Maenhout (2005)** — *RFS* 18(2):491-533.
   DOI: 10.1093/rfs/hhi017
   https://academic.oup.com/rfs/article-abstract/18/2/491/1599892
   Replication: https://econ-ark.org/materials/cgmportfolio/

2. **Campbell & Viceira (1999)** — *QJE* 114(2):433-495.
   https://academic.oup.com/qje/article-abstract/114/2/433/1844221
   NBER: https://www.nber.org/papers/w5857

3. **Campbell, Chan, Viceira (2003)** — *JFE* 67(1):41-80.
   https://www.sciencedirect.com/science/article/abs/pii/S0304405X02002524
   NBER: https://www.nber.org/papers/w8566

4. **Campbell & Viceira (2002)** — *Strategic Asset Allocation*, OUP book.
   https://academic.oup.com/book/6093

5. **Wachter (2002)** — *JFQA* 37(1):63-91.
   https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/portfolio-and-consumption-decisions-under-meanreverting-returns-an-exact-solution-for-complete-markets/898F0A569E28BDFAA8396AA8FD2822C0
   PDF: https://finance.wharton.upenn.edu/~jwachter/research/Wachter2002jfqa.pdf

6. **Campbell, Cocco, Gomes, Maenhout, Viceira (2001)** —
   *Eur. Finance Rev.* 5(3):269-292.
   https://academic.oup.com/rof/article-abstract/5/3/269/1575422

7. **Gomes & Michaelides (2005)** — *J. Finance* 60(2):869-904.
   https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2005.00749.x

8. **Catherine (2022)** — *RFS* 35(9):4016-4054.
   https://academic.oup.com/rfs/article-abstract/35/9/4016/6482757
   SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2778892

9. **Catherine, Sodini, Zhang (2024)** — *J. Finance* 79(3):1755-1788.
   https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13341
   SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3612590

10. **Munk & Sørensen (2010)** — *JFE* 96(3):433-462.
    https://www.sciencedirect.com/science/article/abs/pii/S0304405X10000140
    SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=676021

11. **Viceira (2001)** — *J. Finance* 56(2):433-470.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00333
    NBER: https://www.nber.org/papers/w7409

12. **Benzoni, Collin-Dufresne, Goldstein (2007)** — *J. Finance* 62(5):2123-2167.
    https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01270.x
    SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=875984

13. **Polkovnichenko (2007)** — *RFS* 20(1):83-124.
    https://academic.oup.com/rfs/article/20/1/83/1588217
    SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=419180

14. **Bodie, Merton, Samuelson (1992)** — *JEDC* 16(3-4):427-449.
    https://www.sciencedirect.com/science/article/pii/016518899290044F
    NBER: https://www.nber.org/papers/w3954

15. **Fagereng, Gottlieb, Guiso (2017)** — *J. Finance* 72(2):705-750.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12484

16. **Cocco (2005)** — *RFS* 18(2):535-567.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=258428

17. **Gomes (2020)** — *Annual Review of Financial Economics* 12:277-304.
    https://www.annualreviews.org/content/journals/10.1146/annurev-financial-012820-113815

18. **Iskhakov, Jørgensen, Rust, Schjerning (forthcoming)** — DC-EGM.
    https://www.qeconomics.org/ojs/forth/643/643-3.pdf

---

## 18a. Cross-References Inside This Repo

| File | Purpose | Connects to |
|------|---------|-------------|
| [DESIGN.md](DESIGN.md) | Full model spec, state-return separation, solver architecture | §0a, §16.4 |
| [RETURNS.md](RETURNS.md) | VAR estimation, state grid, return discretization | §0a, §2.4, §7, §16.4-16.5 |
| [LABOUR.md](LABOUR.md) | Income process, tax, pension, mixture-Tauchen z-grid | §0a, §1.1, §6 |
| [STATE_SPACE.md](STATE_SPACE.md) | State variable choice, quadrature design | §0a, §16.5 |
| [CONVENTIONS.md](CONVENTIONS.md) | Code organization, naming, units | §16.7 |
| [TODO.md](TODO.md) | Open issues; record validation findings here | §16.8, §16.9 |

When you find a discrepancy between solver output and a literature benchmark, follow
this triage order:
1. Check §0a — does the codebase calibration even *match* the benchmark paper's setup?
   (e.g., the unconstrained run is not directly comparable to CGM constrained.)
2. Check §16.8 first — is the pathology you're seeing the no-bankruptcy boundary?
   This is by far the most common explanation for surprising leverage in this codebase.
3. Check the constrained run as a reference — if it agrees with literature ranges in
   §17, the bug is in the unconstrained handler, not the underlying economics.
4. Check the relevant section of LABOUR.md or RETURNS.md for whether income / return
   parameters changed since the literature value was set.

---

## 19. Notes on This Compilation

- Numerical values in §1.1, §6.1, §3.2, etc. were verified against multiple independent
  sources (replication notes, citing papers, working paper preprints). Where direct PDF
  extraction was not possible (binary-encoded), I have cross-checked figures against at
  least two secondary sources before listing.
- Where ranges are given (e.g. "55–65%"), the wider edge typically corresponds to
  alternative γ values or to small sensitivity analyses reported in the same paper.
- The "Quick Reference" table (§17) is the **single most useful page** for spot-checks
  on solver output. Print it. Tape it to the monitor.
- This document is *living*: when more direct quotes are extracted (e.g. with PDF
  parsing tools), update the relevant section's quoted values inline.

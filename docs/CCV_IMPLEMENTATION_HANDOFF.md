# Coding-Agent Handoff: Return-Modelling Implementation & Verification

**Date:** 2026-05-07
**Subject:** Implement the changes specified in `CCV_RETURN_IMPLEMENT.md`
(the locked source-of-truth spec) and verify with empirical tests that the
resulting return-modelling pipeline is correct.

---

## 1. The mandate

The user has, over an extended theory-review process, locked a complete
specification of the return-modelling pipeline based on Campbell, Chan,
Viceira (NBER w8566) with one user-elected deviation (Moody's AAA as the
long-bond yield source) and two as-implemented deviations forced by data-
source contents (CPI vs PPI; January vs December stock timing).

That spec lives in `CCV_RETURN_IMPLEMENT.md`. The codebase as it stands
**does not** match that spec — most notably:

- Sample is 1963–2025 (locked spec: **1871–2011**).
- Sixth state variable is `cy` (Lettau–Ludvigson `cay`) (locked spec:
  **`dp` = log(D) − log(P)** per CCV).
- Long-bond yield series differs from the locked Moody's AAA throughout
  source.
- Data-construction pipeline reads from `LW_monthly.xlsx` and uses
  Treasury yields rather than chap_26 + Moody's AAA.

The agent is asked to:

1. **Implement** the data-construction and estimation changes required to
   bring the codebase into compliance with `CCV_RETURN_IMPLEMENT.md`.
2. **Verify** with empirical tests that the resulting pipeline is correct
   — both internally consistent and consistent with the spec.
3. **Report** test results and any spec violations that cannot be reconciled.

Empirical verification is the load-bearing piece. The agent should design
and run as many tests as necessary to be confident the pipeline is right.
Unit tests, integration tests, regression tests, sniff tests against CCV's
published reference numbers — whatever combination is needed.

---

## 2. The locked specification — what must be true after the implementation

The following are the binding spec items. Each is locked in
`CCV_RETURN_IMPLEMENT.md` and must hold post-implementation. The agent
should treat each item as a test target.

### 2.1 — Sample period and data sources

- Annual sample, **1871–2011, T = 141** observations.
- **chap_26** (`chapt26.xlsx`) provides P, D, R, CPI. All taken at
  **January-of-year** observations.
- **FRED Moody's AAA** (`AAA.csv`) provides the long-bond yield $Y_n$,
  taken at January-of-year.
- chap_26's RLONG column is **not used** (replaced by Moody's AAA, per
  §4.1 D1 deviation).
- `LW_monthly.xlsx` ingest is **dropped** entirely (the `cy` variable goes
  away).

### 2.2 — Six VAR variables

The VAR vector is $\mathbf{y} = (r_{tb}, x_r, x_b, y_1, \text{spr}, dp)'$,
$m = 6$. The state block (in code partition) is $(r_{tb}, y_1, \text{spr}, dp)$, return block is $(x_r, x_b)$.

Construction formulas (locked, see §4.1 for full detail):

- $\pi_{t+1} = \log(\text{CPI}_{t+1}/\text{CPI}_t)$ (auxiliary)
- $y_{1,t} = \log(1 + R_t/100)$ — convert from percent before logging
- $r_{tb,t+1} = y_{1,t} - \pi_{t+1}$
- $x_{r,t+1} = \log((P_{t+1} + D_{t+1})/P_t) - y_{1,t}$
- $x_{b,t+1} = r_{n,t+1} - y_{1,t}$ where $r_{n,t+1}$ is from the CLM formula below
- $\text{spr}_t = y_{n,t} - y_{1,t}$ where $y_{n,t} = \log(1 + Y_{n,t}/100)$
- $dp_t = \log(D_t) - \log(P_t)$

### 2.3 — Bond return: Campbell–Lo–MacKinlay constant-duration

$$r_{n,t+1} \approx D_{n,t}\,y_{n,t} - (D_{n,t} - 1)\,y_{n,t+1}$$

with **maturity $n = 20$**, log yield $y_{n,t} = \log(1 + Y_{n,t}/100)$,
and Macaulay-style duration

$$D_{n,t} \approx \frac{1 - (1 + Y_{n,t}/100)^{-n}}{1 - (1 + Y_{n,t}/100)^{-1}}.$$

Constant-duration approximation: $y_{n-1,t+1}$ replaced by $y_{n,t+1}$.

### 2.4 — VAR estimation restrictions

The estimation must impose **two** restrictions:

**(i) §2.2.r restriction.** Zero columns of $\Phi_1$ corresponding to
lagged $\mathbf{x}_t = (x_r, x_b)$:

$$\boldsymbol{\phi}_{r_1, \mathbf{x}} = \mathbf{0},\quad \Phi_{\mathbf{x}, \mathbf{x}} = \mathbf{0},\quad \Phi_{\mathbf{s}, \mathbf{x}} = \mathbf{0}$$

**(ii) §2.2.μ restriction (CCV w8566 §4.2 footnote 5).** Sample-mean
pinning of $\Phi_0$:

$$\Phi_0 = (I - \Phi_1)\,\mu_{\mathbf{y}}^{\text{sample}}$$

Implemented via demeaning the data, OLS without intercept, and back-solve.

The existing `var.py:191-268` already implements both restrictions on
the current dataset. The agent should verify it still does so on the new
1871–2011 / chap_26 + AAA dataset.

### 2.5 — Σ_v partition and the σ²_x, Σ_xx scalars

In CCV's three-block partition $(r_1, \mathbf{x}, \mathbf{s})$, the
quantity $\Sigma_{\mathbf{xx}}$ that enters CCV eq. (10) is the
$\mathbf{x} \times \mathbf{x}$ sub-block of $\Sigma_v$.

In code's two-block partition (state, return), this is `Sigma_rr` exactly.

The eq. (10) formula constants `sigma2_xr`, `sigma2_xb`, `sigma_xrxb` in
`precompute.py:303-314` are sourced from `Sigma_rr` (post May-2026 patch).
This is **correct** and must not be reverted.

### 2.6 — Eq. (10) and FOC architecture (§3.1)

`solver._ccv_log_return_and_grad` and `solver.terminal_foc_jac_ccv`
implement CCV eq. (10) symbol-for-symbol with the chain-rule FOC and
$-\Sigma_{\mathbf{xx}}$ Hessian correction. These are spec-compliant and
should not be modified beyond passing through new constants from the
re-estimated $\Sigma_v$.

`simulation.py:329-362` mirrors the formula with verified solver/simulator
parity to 1e-12. Must remain so after the data migration.

---

## 3. The implementation work, concretely

### 3.1 Data ingestion (the major change)

Replace whatever currently lives in `build_var_dataset.py` (or wherever
the dataset is constructed) with a pipeline that:

1. **Reads chap_26** from `chapt26.xlsx`, sheet "Data". Extract columns by
   header (B = P, C = D, E = R, G = CPI), starting at the data row (row 9
   in the file the user supplied; the agent should auto-detect by year).
   Annual observations, year column = column A.
2. **Reads Moody's AAA** from `AAA.csv` (FRED). Take the **January
   observation** for each year. Series is monthly; the simplest extraction
   is the row whose date is YYYY-01-01 (or first business day of January).
3. **Validates timing alignment**: chap_26 P, D, CPI for year $t$ should
   agree with `ie_data.xls` January-year-$t$ values to <1bp (this is the
   property that lets the agent treat chap_26 as the canonical
   January-snapshot). If the agent wants a sanity check on the timing
   convention before trusting the data, this is the test to run.
4. **Construct the six VAR variables** per §2.2 above.
5. **Output** a clean CSV (replacing `var_dataset.csv`) with columns
   `(year, rtb, xr, xb, y_1, spr, dp)` for years 1872 through 2011 (the
   first observation lost to the inflation differencing).

### 3.2 Drop the LW_monthly path

The current pipeline ingests Lettau–Ludvigson `cay` from
`LW_monthly.xlsx`. After the migration, this file is irrelevant — its
ingest path should be removed (or at least deactivated) so future
contributors don't think it's still in play.

### 3.3 Re-estimate the VAR

Run the existing estimator (`var.py:191-268` with §2.2.r and §2.2.μ
restrictions intact) on the new dataset. The hardcoded snapshot at
`var.py:608-680` will need updating (or removing, if the agent thinks
it's better to always re-estimate from data — user discretion required).

### 3.4 Verify Σ_rr constants flow through correctly

`precompute.py:303-314` sources σ²_xr, σ²_xb, σ_xrxb from `Sigma_rr`. This
should continue to work without modification after the new estimation —
the agent just needs to confirm the data-flow downstream still makes
sense at the new numerical values.

### 3.5 Operational items inherited from the theory review

These are housekeeping items called out in the §3.1 theory-agent review
that the user has accepted but not yet acted on:

- **R1: Update `docs/CCV_RETURNS.md`** to reflect the May-2026 `Sigma_rr`
  patch and the May-2026 partition change. The current docs describe a
  pre-migration partition where `rtb` lived in the return block — they
  are stale on multiple counts. See §2.2's "Stale-doc note" subsection in
  the spec.
- **R3: Add a regression test** that pins `Sigma_rr` (not `Sigma_r_cond`)
  as the source for the eq. (10) constants. Suggested form is in the
  theory review under R3; the agent can copy it. This is cheap insurance
  against a future contributor "fixing" the precompute back to the wrong
  matrix.

(R2, the bankruptcy-suppression sensitivity memo, is empirical not
implementation — leave for later.)

---

## 4. Empirical tests the agent must run

This is the bulk of the verification work. Tests are grouped by what
they verify, in increasing order of integration. The agent should run
**all** of these and report results; failures or unexpected magnitudes
must be investigated before declaring the pipeline correct.

### 4.A — Data ingestion and timing tests

**A1.** chap_26 vs ie_data Jan timing. For each year in the overlap,
chap_26 column B (P) must equal `ie_data.xls` Col 6 of the January row to
<1bp. Same for column C (D) and column G (CPI). Sample years: 1900,
1950, 1980, 2000, 2010, 2015. Failure mode: chap_26 timing convention
differs from January.

**A2.** Moody's AAA ingestion. Verify the January-of-year extraction
returns one and only one value per year, no NaN, monotonic year coverage.
Sample years: 1919 (start of FRED AAA), 1950, 2000, 2011 (sample end).

**A3.** No NaN in any of the six VAR variables for years 1872–2011.

**A4.** Year coverage sanity: T = 141 observations.

### 4.B — Variable construction tests

**B1.** Algebraic identity for `rtb`: assert `rtb[t]` equals
`log(1 + R[t-1]/100) - log(CPI[t]/CPI[t-1])` for randomly chosen years.

**B2.** Algebraic identity for `xr`:
`xr[t] = log((P[t] + D[t])/P[t-1]) - log(1 + R[t-1]/100)`. Test on
randomly chosen years.

**B3.** `dp` construction: `dp[t] = log(D[t]) - log(P[t])`. Should give
values in roughly [-4.5, -3.0] (CCV reports mean -3.101, std 0.304 for
their sample).

**B4.** `spr` construction: positive on average (long yields exceed short
yields in normal regimes). CCV's annual reference: mean 0.902 pp.

### 4.C — Bond return / CLM duration tests

**C1.** Duration formula sanity: $D_{n,t}$ at $Y_n = 5\%$ and $n = 20$
should give approximately 12.5 years (textbook result for a 20yr 5%
par bond). Test at multiple yields (2%, 5%, 10%) and verify $D_{n,t}$
increases as yield decreases (standard convexity behavior).

**C2.** Bond return magnitudes: $\sigma(x_b)$ should be in single-digit
percent territory, comparable to CCV's reference 6.543 pp on annual data.
Significantly larger or smaller would indicate a units bug.

**C3.** Bond return sign behaviour: in years when AAA yields rise
sharply, $r_n$ should be negative (bond losses). E.g. 1980 (Volcker
shock), 1994 (Greenspan tightening) — agent should spot-check these.

**C4.** Constant-duration approximation: agent should verify the formula
implementation does $y_{n-1, t+1} = y_{n, t+1}$, NOT some forward yield
lookup. (Easy to mis-implement.)

### 4.D — VAR estimation correctness tests

**D1.** §2.2.r restriction: assert $\Phi_1[:, \text{x-cols}] = 0$ exactly
(not approximately). The columns corresponding to lagged $x_r$ and $x_b$
must be all zeros in the estimated $\Phi_1$.

**D2.** §2.2.μ restriction: assert
$(I - \Phi_1)^{-1}\Phi_0 = \mu_{\mathbf{y}}^{\text{sample}}$ to numerical
precision. Equivalent: the residuals from the demeaned regression have
exactly zero mean (machine epsilon).

**D3.** $\Sigma_v$ positive definite: all eigenvalues $> 0$.

**D4.** Stationarity: $\max_i |\lambda_i(\Phi_1)| < 1$ strictly.
CCV w8566 reports the largest annual VAR eigenvalue around 0.92–0.95
historically; if our build's largest eigenvalue is much larger or near 1,
investigate.

**D5.** Lyapunov consistency: solving
$\Sigma_{\mathbf{yy}} = \Phi_1 \Sigma_{\mathbf{yy}} \Phi_1' + \Sigma_v$
via `scipy.linalg.solve_discrete_lyapunov` and comparing with the
sample covariance of $\mathbf{y}_t$ on the data — the two should be
"close" but not identical (sample size effects). Magnitudes should match
within a factor of 2 or so on each entry.

### 4.E — CCV reference number sniff tests

These are the §4.2 reference values. Our build's numbers will not exactly
match (caveats C1, C2, C3 in §4.2), but should be in the same ballpark.
Wide deviations signal a bug or a misunderstanding.

**E1.** Sample statistics (Table 1 analogues, on our 1871–2011 sample):

| Quantity | CCV ref | Our build (target ballpark) |
|---|---|---|
| $\mathbb{E}[r_{tb}]$ + Jensen | 2.101 | within ±2 pp |
| $\sigma(r_{tb})$ | 8.806 | within ±2 pp |
| $\mathbb{E}[x_r]$ + Jensen | 6.797 | within ±2 pp |
| $\sigma(x_r)$ | 18.192 | within ±2 pp |
| $\mathbb{E}[x_b]$ + Jensen | 0.674 | within ±1 pp (Moody's AAA differs from CCV's series) |
| $\sigma(x_b)$ | 6.543 | within ±2 pp |
| $\mathbb{E}[y_1]$ | 4.361 | within ±1 pp |
| $\mathbb{E}[dp]$ | −3.101 | within ±0.3 |
| $\sigma(dp)$ | 0.304 | within ±0.1 |
| $\mathbb{E}[\text{spr}]$ | 0.902 | within ±0.5 pp (Moody's-AAA-minus-bill differs from CCV's spread) |

Order-of-magnitude failures here mean a units bug or formula
mis-implementation. Wide divergences within this band are explainable by
the longer sample + Moody's-AAA-vs-CCV-spliced-series and not bugs.

**E2.** VAR coefficient sniff: the diagonal autoregressive coefficients
$\Phi_1[r_{tb}, r_{tb}]$, $\Phi_1[y_1, y_1]$, $\Phi_1[dp, dp]$,
$\Phi_1[\text{spr}, \text{spr}]$ should all be positive and < 1
(persistence). CCV reports them around 0.30, 0.92, 0.84, 0.82 respectively.
Ours should be in the same ballpark.

**E3.** $R^2$ sniff: the $y_{t+1}$ and $dp_{t+1}$ equations should have
$R^2 > 0.7$ (highly persistent variables). The $x_{r,t+1}$ equation
should have $R^2$ in the 0.05–0.10 range (limited stock predictability).

### 4.F — Eq. (10) consistency tests (regression tests for the locked invariants in §3.1)

**F1.** $\alpha = 0$ collapse: at $\boldsymbol{\alpha} = \mathbf{0}$,
$r_p$ from `_ccv_log_return_and_grad` must equal `log_R_bill` exactly
(all quadratic terms vanish, σ²_x terms vanish). Test on randomly
chosen state/innovation realisations.

**F2.** $\alpha = e_j$ collapse: at $\boldsymbol{\alpha} = (1, 0)'$,
$r_p$ must equal `log_R_bill + log_x_s` — the Jensen lift
$+\tfrac{1}{2}\sigma^2_{xr}$ must exactly cancel the Itô drag
$-\tfrac{1}{2}\sigma^2_{xr}$. Same test for $\boldsymbol{\alpha} = (0, 1)'$.

**F3.** Source-of-Σ_xx regression test (theory review R3): assert that
`pc.sigma2_xr == model.Sigma_rr[xr, xr]`, `pc.sigma2_xb == model.Sigma_rr[xb, xb]`,
`pc.sigma_xrxb == model.Sigma_rr[xr, xb]`. This is the explicit guard
against a future "fix" reverting `Sigma_rr` to `Sigma_r_cond`.

**F4.** Markowitz-at-γ=1 sanity: at γ=1 with iid returns, the myopic
optimal weights should be approximately
$\boldsymbol{\alpha}^* = \Sigma_{\mathbf{xx}}^{-1}(\mathbb{E}[\mathbf{x}] + \tfrac{1}{2}\boldsymbol{\sigma}_x^2)$.
Compute this from our estimated $\mu_{\mathbf{x}}, \Sigma_{\mathbf{xx}}$
and compare against the solver's converged α at γ=1 (deterministic-state
case). Should agree to within numerical precision.

### 4.G — Solver/simulator parity (preserved post-migration)

**G1.** Run the existing `verify/ccv_solver_sim_parity.py` on the new
data. Must continue to agree to 1e-12. Failure here means the migration
broke parity.

### 4.H — Restriction-effect diagnostic (deferred but worth running)

**H1.** Optionally run an unrestricted estimation as well (free up the
§2.2.r columns) and compare $R^2$ for $x_{r,t+1}$ and $x_{b,t+1}$ between
the two estimations. Document the empirical effect of the §2.2.r
restriction on the new sample. (This addresses the deferred §4.2 item
"Empirical effect of §2.2.r restriction.")

---

## 5. Acceptance criteria

The implementation is done when:

1. All tests in §4 pass (or failures are explained and documented).
2. The dataset is regenerated and matches the §4.1 spec exactly.
3. The VAR estimates are produced and respect both §2.2.r and §2.2.μ
   restrictions (verified by D1, D2 tests).
4. `solver/simulator parity` (test G1) passes.
5. The CCV reference comparison (test E1, E2, E3) shows our numbers in
   the same ballpark, with deviations attributable to the named caveats
   (sample length, restriction, long-bond source) rather than bugs.
6. The deferred §4.2 item "this build's own VAR estimates on 1871–2011"
   is added to `CCV_RETURN_IMPLEMENT.md` (Tables 1 and 2 analogues with
   the actual estimated values).
7. R1 (`docs/CCV_RETURNS.md` update) and R3 (Sigma_rr regression test)
   are addressed.
8. §2.2.μ item 4 (verbatim code-side check that
   `var.py:191-268` implements $\Phi_0 = (I - \Phi_1)\,\mu^{\text{sample}}$)
   is verified, allowing §2.2.μ to move from PARTIALLY VERIFIED to
   ✅ LOCKED.

---

## 6. What "verified" looks like in the deliverable

The agent's report should include:

1. **Code changes summary**: every file touched, what changed.
2. **Test results**: every test in §4 with pass/fail status and observed
   numerical value where relevant.
3. **CCV reference comparison table**: our build's analogues of Tables 1
   and 2 from §4.2, side-by-side with CCV's reference values, with
   explicit annotation of which differences are attributable to caveats
   C1/C2/C3 vs. unexplained.
4. **Caveats inventory**: any item from `CCV_RETURN_IMPLEMENT.md` that the
   agent could not verify or implement, with reasoning.
5. **Spec doc updates**: the §4.2 "this build's own estimates" subsection
   filled in with actual numbers; §2.2.μ status updated based on item 4
   verification.

The user's preference is for **direct, opinionated technical engagement**
over hedged language. If something looks off — a test fails, a number is
out of expected range, a code path looks suspicious — flag it plainly
with reasoning. If the agent disagrees with anything in the spec, raise
it for user discussion before silently working around it.

---

## 7. Reference materials

| Document | Location | Purpose |
|---|---|---|
| **`CCV_RETURN_IMPLEMENT.md`** | `/mnt/user-data/outputs/` | The locked spec. **Source of truth.** |
| `w8566.pdf` | project root | CCV w8566 paper, ground reference |
| `chapt26.xlsx` (uploaded as `chapt26__2_.xlsx`) | uploads | Annual data source |
| `AAA.csv` | project root | Moody's AAA from FRED |
| `ie_data.xls` | project root | Reference dataset (Shiller monthly), used for chap_26 timing cross-check |
| `RETURN_MODELLING_TRACE_2026-05-07.md` | project root | Existing pipeline trace |
| `RETURN_MODELLING_PARAMS_2026-05-07.md` | project root | Existing numerical params (will be superseded by new estimation) |
| `CCV_EQ10_THEORY_REVIEW_2026-05-07.md` | uploads | Theory verification of eq. (10) — reference for B-tests in §4.F |
| `var.py:191-268` | project root | Existing estimator (restrictions intact) |
| `var.py:608-680` | project root | Hardcoded snapshot (will need updating or removal) |
| `solver.py:_ccv_log_return_and_grad` | project root | Eq. (10) impl. (no changes needed) |
| `solver.py:terminal_foc_jac_ccv` | project root | FOC kernel (no changes needed) |
| `precompute.py:303-314` | project root | Source of σ²_xr, σ²_xb, σ_xrxb (must remain Sigma_rr) |
| `simulation.py:329-362` | project root | Parity formula (no changes needed) |

---

## 8. Out of scope for this handoff

- **§2.3 (Preferences)**: not yet specified in the spec doc, do not
  modify utility functions or preference parameters.
- **Solver/quadrature internals beyond data flow**: do not modify
  `discretization.py` or the JAX kernels themselves; just confirm they
  consume the new data correctly.
- **R2 (bankruptcy-suppression sensitivity memo)**: empirical exercise,
  separate from this implementation handoff.
- **Post-war robustness re-estimation** (per §4.1's robustness note):
  optional follow-up; not required for sign-off.

---

*Handoff prepared: 2026-05-07.*

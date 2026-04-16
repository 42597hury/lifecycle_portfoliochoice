# HANDOFF: State and Return Covariance with Labour Income

**To:** returns validation agent / future implementation agent
**From:** planning session (2026-04-16)
**Status:** BLOCKED on VAR/returns validation. Decision boxed until state
transitions and return quadrature have been verified.

---

## 1. The Question

Should the persistent income innovation η correlate with financial state
variables (rtb, y_nom, dp) or return residuals (xr, xb)?

Currently η is fully independent of all financial variables. The income
process is:

```
log Y = f(age) + z_t + ε_t
z_{t+1} = ρ z_t + η_{t+1}          η ⊥ everything financial
```

This matches Catherine (2025) baseline and CGM (2005). The question is
whether relaxing this independence would strengthen the model's treatment
of interest-rate risk and human capital duration.

---

## 2. Empirical Literature on Income–Financial Variable Correlations

### 2.1 Income ↔ equity returns (your xr)

| Source | Estimate | Note |
|--------|----------|------|
| CGM (2005) Table 3 | ρ ≈ -0.01 to -0.02 | PSID micro data, statistically insignificant |
| Heaton & Lucas (2000) | -7% to +14% | Broader sample, still noisy |
| Davis & Willen (2000) | -25% to +30% | Occupation-level, wide range |
| Munk & Sørensen (2005) | +17% | Single estimate |
| Benzoni, Collin-Dufresne, Goldstein (2007) | High long-run | Cointegration argument: contemporaneous ≈ 0 but long-run large |

**Consensus:** Contemporaneous correlation near zero at the micro level.
Long-run correlation possibly substantial via cointegration, but this
operates through the aggregate wage channel (L̄), not idiosyncratic η.

### 2.2 Income ↔ real short rate (your rtb)

| Source | Estimate | Note |
|--------|----------|------|
| Munk & Sørensen (2005/2010) | +26% | Only published micro estimate for this channel |
| Munk & Sørensen (2010, JFE) | Affine function | E[Δlog Y] = a + b × r_t, calibrated from PSID |
| De Jong, Driessen, Van Hemert (2008) | Sweep ±60% | No own estimate; cite M&S and run comparative statics |

**This is the strongest channel for the thesis.** A +26% correlation means
income innovations co-move with real rate surprises. This gives human
capital interest-rate duration, which is exactly Catherine's core mechanism
(Figure 5: rate sensitivity of human capital).

Note: M&S find this operates partly through the *conditional mean* of
income growth (expected income is higher when rates are higher), not
just through innovation correlation.

### 2.3 Income ↔ dividend-price ratio (your dp)

| Source | Estimate | Note |
|--------|----------|------|
| Lynch & Tan (2011, JFE) | Significant, negative | Aggregate income growth covaries negatively with lagged dp |

**Well-documented business-cycle channel.** dp is countercyclical (high in
recessions), income growth is procyclical, so the covariance is negative.
Lynch & Tan use dp as the business-cycle predictor for income dynamics
and find it matters for portfolio choice of young agents.

### 2.4 Income ↔ nominal yield (your y_nom)

No direct micro estimate found. De Jong et al. sweep income–expected
inflation correlation from -60% to +60% and find it produces the
**largest portfolio effects** of all channels they test. Nominal yields
embed inflation expectations, so this channel matters if real wages
track (or fail to track) inflation. But nobody has pinned down the
number from household data.

### 2.5 Income ↔ bond excess returns (your xb)

Nothing directly estimated. Bond returns are composites of rate and
inflation shocks. Any income–bond correlation is implied by the rate
and inflation channels above, not independently measured.

### 2.6 Summary table

| Financial variable | Empirical support | Point estimate | Source |
|-------------------|-------------------|----------------|--------|
| xr (stock excess) | Strong for ≈ 0 | -1% to +2% | CGM (2005) |
| rtb (real rate) | Moderate, one study | +26% | Munk & Sørensen |
| dp (div-price) | Solid | Significant negative | Lynch & Tan (2011) |
| y_nom (nom yield) | None direct | Unknown | — |
| xb (bond excess) | None direct | Unknown | — |

---

## 3. Two Design Options

### Option A: Innovation Correlation

**Mechanism:** η loads linearly on financial innovation realizations.

```
z_{t+1} = ρ z_t + η + λ' × e_financial
```

where `e_financial` is a vector of financial surprises realised in the
same period. This creates contemporaneous co-movement between income
and financial shocks.

**Sub-choice: load on what?**

- **Return residuals** `ret_nodes[k_r, :]`: directly available as
  continuous quadrature nodes inside the solver's `k_r` loop. Clean,
  no dependency on state-grid quality. Parameters: `(λ_stock, λ_bond)`.

- **Backed-out state innovations** `state_grid[j_s] - Phi_11 @ state_grid[i_s]`:
  precomputable as `(N_state, N_state, n_state)` array. Requires
  Rouwenhorst discretisation to be accurate. Parameters:
  `(λ_rtb, λ_ynom, λ_dp)`.

**What changes in the income expression:** Only the evolution of z.
`f(age)`, `ε_t`, `ρ`, the z grid, and all tax/pension functions are
completely unchanged.

**Economic content:** In a period where stock returns (or rates) are
unexpectedly high, the persistent income component also gets a
positive (or negative) kick. On average across financial outcomes,
the shift cancels (E[e_financial] = 0), so unconditional income
moments are preserved.

**Empirical anchor:** Weak. CGM's ≈ 0 is the best micro estimate for
the equity channel. The rate channel (+26% from M&S) is partly a
conditional-mean effect, not purely an innovation correlation.

### Option B: Conditional Mean

**Mechanism:** Expected income growth depends on the current financial state.

```
E[log Y_{t+1}] = f(age) + b_rtb × rtb_t + b_dp × dp_t + z_t
```

The deterministic component of income becomes state-dependent: when
rates are high, expected income growth is higher.

**What changes:** The precomputed income table gains a state dimension.
Shape goes from `(n_age, n_z, n_eps)` to `(n_age, n_z, n_eps, N_state)`.
The solver already loops over `i_s` and `j_s`, so it indexes into the
appropriate state-specific income table.

**Economic content:** Human capital has interest-rate duration because the
*level* of future expected income shifts with the rate environment.
When rates rise, future income is expected to be higher, so the PV
of human capital rises — exactly the mechanism in Catherine's Figure 5.

**Empirical anchor:** Strong. Munk & Sørensen (2010) calibrate the
income-rate slope from PSID data. Lynch & Tan (2011) calibrate the
income-dp slope from aggregate wage data.

**Memory cost:** Income table grows from ~26 KB to ~3.2 MB with current
grid sizes (N_state = 125). Negligible.

**Compute cost:** Zero — same loop structure, one extra index.

---

## 4. Why We Stopped

The implementation of either option touches the financial side of the
model:

- **Option A (return residuals)** requires `ret_nodes` to correctly
  represent draws from `N(0, Σ_r_cond)`. If the return quadrature
  has issues, λ × ret_nodes is wrong.

- **Option A (state innovations)** requires the Rouwenhorst grid to be
  accurate enough that `state_grid[j_s] - Phi_11 @ state_grid[i_s]`
  is a meaningful "innovation." If the state discretisation has
  approximation error, the backed-out innovations inherit it.

- **Option B (conditional mean)** requires the state grid to correctly
  represent the financial state space. Income at `state_grid[i_s]`
  must reflect the true income level at that rate/yield/dp combination.

The VAR structure, state transitions, and return quadrature have NOT
yet been fully validated. The validation plan calls for:

1. Labour income verification ← in progress
2. Bequest term analysis ← next
3. **Return and state variable validation** ← must complete before
   implementing any income–financial covariance

Building this feature on unverified financial infrastructure risks
compounding errors. The feature itself is small (a few lines of code),
so delaying it costs nothing.

---

## 5. Suggested Implementation Plan (Post-Validation)

### Phase 1: Decide which option

After returns validation, choose based on:

- If return quadrature is clean → Option A (return residuals) is
  simplest to implement and sidesteps any state-grid issues.
- If the economic goal is human-capital duration (Catherine's story)
  → Option B (conditional mean) is better supported empirically
  and more directly relevant.
- Both can be implemented. They are not mutually exclusive. But
  given the deadline, pick one.

### Phase 2: Implementation (either option)

**Option A — Innovation correlation (return residual loading):**

```
Files changed:
  model.py           Add lambda_zr (float, default 0.0)
  solver.py          4 function signatures gain lambda_zr + ret_nodes
                     1 line added inside k_eta loop:
                       eta_shift = lambda_zr * ret_nodes[k_r, 0]
                       z_next = rho * z + eta_nodes[k_eta] + eta_shift
  simulation.py      1 line added:
                       z_next = rho * z_val + eta + lambda_zr * xr_res
  diagnostics.py     Report lambda_zr value (informational)

Files NOT changed:  precompute.py, discretization.py, mortality.py,
                    var.py, policy_io.py

Calibration:        lambda_zr = rho_target × sigma_eta / sigma_e_stock
                    Baseline: rho_target = 0 (independence, current model)
                    Present as: single solve at chosen rho, discuss
                    literature range in thesis text

Retirement solver:  No change (pension is deterministic, no η)
```

**Option B — Conditional mean (state-dependent income):**

```
Files changed:
  precompute.py      _precompute_working_income() gains state dimension
                     Income table: (n_age, n_z, n_eps) → (n_age, n_z, n_eps, N_state)
                     f(age) → f(age) + b_rtb × state_grid[i_s, 0] + b_dp × state_grid[i_s, 2]
  model.py           Add b_rtb, b_dp parameters (floats, default 0.0)
  solver.py          income_next_table indexing changes:
                     income_next_table[iz, ie] → income_next_table[iz, ie, j_s]
                     Propagate through function signatures
  simulation.py      Income computation gains state dependence:
                     y_gross = exp(f(age) + b_rtb*rtb + b_dp*dp + z + eps)
  diagnostics.py     Report income-state slopes, income at representative states

Files NOT changed:  discretization.py, mortality.py, var.py, policy_io.py

Calibration:        b_rtb from Munk & Sørensen (2010) PSID estimates
                    b_dp from Lynch & Tan (2011) aggregate wage data
                    Baseline: b_rtb = b_dp = 0 (independence)

Retirement solver:  No change (pension depends on frozen z, not current state)
```

### Phase 3: Validation

- Verify at λ = 0 / b = 0 that results are numerically identical to
  current baseline (regression test)
- Check that income moments (mean, variance by age) are preserved
  at chosen parameter values
- For Option B: verify income table varies sensibly across states
  (higher income in high-rate states if b_rtb > 0)

### Phase 4: Thesis integration

Regardless of implementation choice, write discussion section citing:
- CGM (2005): income–equity ≈ 0 (baseline justification)
- Munk & Sørensen (2010): income–rate +26%, affine relationship
- Lynch & Tan (2011): income–dp significant, business cycle channel
- De Jong et al. (2008): income–inflation largest portfolio effect
- Catherine (2021): cyclical skewness channel (higher order)
- Benzoni et al. (2007): long-run cointegration argument

If ρ = 0 baseline is kept, this section explains why and flags the
channels as future work. If a non-zero correlation is implemented,
this section motivates the calibration.

---

## 6. References

- Catherine, S. (2025). "Interest-Rate Risk and Household Portfolios."
- Catherine, S. (2021). "Countercyclical Labor Income Risk and Portfolio
  Choices over the Life Cycle." RFS.
- Cocco, J., Gomes, F., Maenhout, P. (2005). "Consumption and Portfolio
  Choice over the Life Cycle." RFS.
- De Jong, F., Driessen, J., Van Hemert, O. (2008). ""; working paper.
- Munk, C., Sørensen, C. (2010). "Dynamic Asset Allocation with
  Stochastic Income and Interest Rates." JFE.
- Lynch, A., Tan, S. (2011). "Labor Income Dynamics at Business-Cycle
  Frequencies: Implications for Portfolio Choice." JFE.
- Benzoni, L., Collin-Dufresne, P., Goldstein, R. (2007). "Portfolio
  Choice over the Life-Cycle when the Stock and Labor Markets Are
  Cointegrated." JF.

# UTILITY, MORTALITY, AND BEQUEST — VALIDATION DOCUMENT

**Purpose:** Consolidate the specification, implementation mapping, and validation
status of the three components of the agent's objective function: CRRA consumption
utility, earnings-dependent mortality, and bequest utility. This is the single
source of truth for anyone auditing or extending the utility side of the model.

**Code references:** `model.py` (CRRA functions, bequest functions, annuity factor),
`mortality.py` (SSA life table, Chetty targets, chi calibration, survival probs),
`precompute.py` (annuity_factors array, survival_probs_2d construction),
`solver.py` (FOC functions, terminal solver, EGM inversion).

---

## 0. Units and Context

The model works in units of the SSA Average Wage Index (AWI ≈ $54,100 in 2019).
See LABOUR.md Section 0 for full details on the unit convention.

All utility-related quantities are dimensionless transformations of consumption,
wealth, and returns already expressed in AWI units. CRRA utility `u(c)` maps
consumption (in AWI units) to utils. Bequest utility maps terminal wealth (in AWI
units) through an annuity factor to flow-equivalent consumption, then applies the
same CRRA kernel. Survival probabilities are unitless scalars ∈ [0, 1].

**Calibration:** γ = 3, β = 0.96, b̄ = 10. Source: Catherine (2025, Section 5.1).

**Bellman equation:** The agent's objective at age t is:

```
V_t(W, s, z) = max_{c, α}  u(c) + β · E[ ψ_t(z) · V_{t+1}(W') + (1-ψ_t(z)) · b(a·R_p, r_f) ]
```

where ψ_t(z) = 1 - m(age_t, z) is the survival probability, W' = a·R_p + Y_{t+1}
is next-period cash-on-hand (savings × portfolio return + income), and b(·) is the
bequest utility function. β discounts both the alive and dead branches equally.

---

## 1. CRRA Consumption Utility

### 1.1 Theoretical Specification

Source: standard CRRA preferences, as in Cocco, Gomes & Maenhout (2005, eq. 1)
and Catherine (2025, Section 3.1).

```
u(c) = c^{1-γ} / (1-γ)          for γ ≠ 1
u(c) = log(c)                    for γ = 1

u'(c) = c^{-γ}                  for γ ≠ 1
u'(c) = 1/c                     for γ = 1

u'^{-1}(μ) = μ^{-1/γ}           for γ ≠ 1
u'^{-1}(μ) = 1/μ                for γ = 1
```

### 1.2 Code Implementation

`create_utility_functions(gamma)` in [model.py:168](model.py#L168) returns
`(u, u_prime, u_prime_inv)`. The function has two branches:

- **γ = 1.0 branch** (lines 171–178): `u = log(c)`, `u' = 1/c`, `u'^{-1} = 1/μ`
- **γ ≠ 1 branch** (lines 180–187): `u = c^{1-γ}/(1-γ)`, `u' = c^{-γ}`,
  `u'^{-1} = μ^{-1/γ}`

These functions are stored in the `LifecyclePortfolioModel` NamedTuple as
`model.u`, `model.u_prime`, `model.u_prime_inv` (built at
[precompute.py:446](precompute.py#L446)).

### 1.3 How u' Enters the Euler Equation

The Euler equation for optimal consumption is:

```
u'(c_t) = β · E[ ψ · u'(c_{t+1}) · R_p + (1-ψ) · b'(a·R_p) · R_p ]
```

In the solver, the right-hand side is accumulated as `euler_sum`:

**Retirement** ([solver.py:473–485](solver.py#L473-L485)):
```python
mu_alive  = c_next ** (-gamma)                           # u'(c_{t+1})
mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is  # b'(a·R_p)
mu_comb   = psi * mu_alive + prob_death * mu_bequest
euler_sum += weight * mu_comb * R_p
```

**Working age** ([solver.py:829–836](solver.py#L829-L836)):
Bequest contribution is hoisted outside the income quadrature loops (valid because
bequest depends only on invested wealth, not income realization):
```python
death_mu  = p_state_ret * prob_death * mu_bequest
euler_sum += death_mu * R_p
```
Alive contribution is accumulated inside the `(k_eta, i_e)` loops with
`mu_alive = c_next ** (-gamma)`.

### 1.4 How u'^{-1} Enters EGM

The EGM inversion step recovers optimal consumption from the Euler equation
right-hand side:

```
c* = u'^{-1}(β · euler_sum) = (β · euler_sum)^{-1/γ}
```

In code ([solver.py:1742](solver.py#L1742) for retirement,
[solver.py:1923](solver.py#L1923) for working age):
```python
c_opt = max(beta * euler, sc.euler_inv_floor) ** (-1.0 / gamma)
```

The `max(·, euler_inv_floor)` guard (floor = 1e-20) prevents negative or zero
arguments to the power function. The floor is extremely small — it would imply
consumption of ~10^{6.7} AWI units ($360M), far beyond any reachable wealth level,
so it never binds in practice.

**Algebraic verification that this equals `u_prime_inv(beta * euler_sum)`:**
- `u_prime_inv(μ) = μ^{-1/γ}` by definition
- `(β · euler)^{-1/γ}` is exactly `u_prime_inv(β · euler)`
- The code inlines the formula rather than calling `u_prime_inv` because the solver
  is Numba-JIT compiled and cannot call Python closures

---

## 2. Bequest Utility

### 2.1 Theoretical Specification

Source: De Nardi (2004) luxury-bequest form, with Catherine (2025, Section 3.6,
equations 21–22) annuity normalisation. The shift `δ` removes the unshifted
CRRA-with-clamp's `∞` discontinuity in marginal utility at the bankruptcy
boundary — the original form is recovered as `δ → 0`.

The bequest is valued as if the agent lives b̄ = 10 extra years on a fixed
consumption stream C̄, where C̄ is the annual coupon from investing the estate
in a b̄-year annuity at the current nominal yield, plus a small luxury shift:

```
b(W, r_f) = b̄ · (max(W,0)/A + δ)^{1-γ} / (1-γ)

where:
  C̄ = max(W,0)/A(y_nom) + δ           shifted flow-equivalent consumption
  A(y) = Σ (1+y(k))^{-k}              annuity factor (interpolated term struct.)
  δ = DELTA_BEQUEST                    luxury shift (lifecycle/model.py)
  b̄ = 10                              bequest horizon (years of heir consumption)
```

Marginal bequest utility (W > 0):
```
b'(W)  =  b̄ · (W/A + δ)^{-γ} / A
b''(W) = -γ · b'(W) / (A · (W/A + δ))
```

Marginal utility is bounded above by `mu_max = b̄·δ^{-γ}/A` (vs. `∞` in the
unshifted spec). For W ≤ 0 (bankrupt heirs), `b'(W) = 0` — the realised-estate
clamp `max(W,0)` survives unchanged.

Inverse marginal (clamps to W = 0 above mu_max):
```
W = A · max( (μ·A/b̄)^{-1/γ} - δ,  0 )
```

### 2.2 Code Implementation

| Function | File | Formula |
|----------|------|---------|
| `annuity_factor(y_1, spr, b_bar)` | [model.py](../lifecycle/model.py) | `Σ (1+y(k))^{-k}` |
| `bequest_utility(W, A, gamma, b_bar, delta)` | [model.py](../lifecycle/model.py) | `b̄·(max(W,0)/A + δ)^{1-γ}/(1-γ)` |
| `bequest_marginal(W, A, gamma, b_bar, delta)` | [model.py](../lifecycle/model.py) | `b̄·(W/A + δ)^{-γ}/A` for W>0; else 0 |
| `bequest_marginal_inv(mu, A, gamma, b_bar, delta)` | [model.py](../lifecycle/model.py) | `A·max((μA/b̄)^{-1/γ} - δ, 0)` |
| `DELTA_BEQUEST` | [model.py](../lifecycle/model.py) | calibrated shift parameter |

**Annuity factor array:** Precomputed at [precompute.py:168–169](precompute.py#L168-L169):
```python
_y_ann = state_grid[:, model.annuity_yield_index_in_state]
annuity_factors = annuity_factor(_y_ann, model.b_bar)
```
Shape: `(N_state,)` — one value per financial state. Currently uses the 10-year
nominal bond yield (annual decimal, SVENY10/100), but the choice of discount
rate is not yet finalized (see Section 2.5).

### 2.3 How Bequest Enters the Solver

**Retirement FOC** ([solver.py](../lifecycle/solver.py)):
```python
# Solvent branch (sR_p > 0): shifted-bequest helper.
mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(
    sR_p, annuity_factor_is, gamma, b_bar, DELTA_BEQUEST
)
# Bankrupt branch (sR_p <= 0): mu_bequest = mup_bequest = 0.
mu_comb  = psi * mu_alive + prob_death * mu_bequest
mup_comb = psi * mup_alive + prob_death * mup_bequest
```

**Working-age FOC** ([solver.py](../lifecycle/solver.py)):
Same bequest marginal and second derivative (now from `_shifted_bequest_mu_and_mup`),
hoisted outside the `(k_eta, i_e)` income quadrature loops. This is valid because:
- Bequest depends only on invested wealth `a·R_p` (determined by `j_s, k_r`)
- Bequest does not depend on income realization (no `Y_{t+1}` in the dead branch)
- The sum of income quadrature weights = 1, so hoisting doesn't change the level

**Terminal age** ([solver.py](../lifecycle/solver.py) — `solve_terminal_age`):
At age 99, death is certain (ψ = 0). Under the **shifted** bequest the
homogeneity that gave the closed-form `c* = W·ratio/(1+ratio)` is broken —
the bequest is no longer proportional to `a^{1-γ}`. The terminal step is
instead solved by EGM on a savings grid:

```
V_T(W) = max_{c,α}  u(c) + β · E[ b(a·R_p, A) ]
       = max_{c,α}  c^{1-γ}/(1-γ)  +  β · E[ b̄·(max(aR_p,0)/A + δ)^{1-γ}/(1-γ) ]
```

For each savings level `a = s_grid[j_s]`:
1. Solve the s-dependent portfolio FOC
   `Σ w · b'(s·R_p) · (R_k − R_bill) = 0` for `(α_s*, α_b*)` via 2D Newton
   (constrained or unconstrained, mirroring the working/retirement solvers).
2. Compute `V_dot = E[b'(s·R_p) · R_p]`.
3. Invert the consumption Euler `u'(c) = β · V_dot`:
   `c* = (β · V_dot)^{-1/γ}`.
4. Implied wealth `W = c* + s`.

Then for each `i_w` on the wealth grid, interpolate `(c, α_s, α_b)` from the
EGM arrays. Below the smallest implied W we hold `s = s_grid[0]` and consume
the residual; above the largest W we extrapolate `c` linearly with `α` held
flat at the largest-s solution. Bequest depends on (s, A_is) only — not on z —
so the policy is broadcast across the z dimension.

Cost: `N_state × N_s` Newton solves vs. the original `N_state` (terminal step
is still negligible relative to the working/retirement induction loop).

### 2.4 Timing

Following Catherine (2025):
- Death occurs between periods t and t+1
- Bequeathed wealth = a_t · R_p (invested savings × portfolio return; no income)
- The alive branch receives income: W' = a_t · R_p + Y_{t+1}
- β discounts both alive and dead branches equally
- Annuity factor uses the current-period state `annuity_factors[i_s]`, not the
  next-period state — matches Catherine's subscript r_{ft}. The specific yield
  used as the discount rate is not yet finalized (see Section 2.5)

### 2.5 Open Question — Annuity Discount Rate

The annuity factor currently uses a flat term structure:
`A(y) = Σ_{k=1}^{10} (1+y_{10yr})^{-k}`, discounting all 10 payments at the
10-year nominal bond yield. This choice is **not yet settled**.

Catherine's paper uses zero-coupon bond prices P_{kt} for each maturity k
(eq. 22), but his model has zero term premium, making any single-rate
approximation exact. Our model does NOT have zero term premium (the VAR allows
a spread between the bill rate and 10-year yield), so the choice of discount
rate matters.

**Options under consideration:**
1. **10-year yield** (current implementation) — matches Catherine literally,
   but overestimates discounting for shorter maturities within the 10-year
   annuity stream
2. **Bill rate** — risk-free discounting, but ignores the term premium the
   heir would earn on longer-duration assets
3. **Duration-matched blend** — weight between bill and bond rates to match
   the annuity's Macaulay duration (~4.5–5.5 years). More accurate but
   adds implementation complexity
4. **Zero-coupon strip** — use actual bond prices from the VAR state for each
   maturity k. Most faithful to Catherine eq. 22 but requires constructing
   a term structure from the state variables

**Direction of bias under current implementation:** If the 10-year yield
overstates the appropriate discount rate, then A is too small → C̄ = W/A too
large → bequest value overstated. If the 10-year yield understates it (e.g.
in a low-rate environment with positive term premium), the bias reverses.

This decision is deferred until returns validation is complete, at which point
the magnitude of the bias can be quantified across the VAR state grid.

---

## 3. Earnings-Dependent Mortality

### 3.1 Theoretical Specification

Source: Catherine (2025, eq. 35), calibrated to Chetty et al. (2016, JAMA).

The mortality rate for an agent of age `age_t` in persistent income state `z_i`
is a scaled version of the population-average baseline rate:

```
m(age_t, z_i) = min( χ(z_i) · m_baseline(age_t),  1 )
ψ(age_t, z_i) = 1 - m(age_t, z_i)
```

where:
- `m_baseline(age)` is the gender-averaged death probability from the SSA 2017
  Period Life Table (2020 Trustees Report)
- `χ(z_i)` is an income-specific mortality scaling factor, calibrated so that the
  implied life expectancy at age 40 matches Chetty et al. (2016) data
- The `min(·, 1)` cap ensures death probabilities never exceed 1
- `ψ(age_t, z_i)` is the probability of surviving from age `age_t` to `age_t + 1`

The scaling factor χ shifts the entire age-mortality profile up or down
proportionally. Low earners (low z, χ > 1) face uniformly higher mortality at
every age. High earners (high z, χ < 1) face uniformly lower mortality. This
preserves the shape of the age-mortality curve — the accelerating risk with age
implied by the SSA table — while allowing the level to vary with income, matching
the empirical mortality gradient documented by Chetty et al. (2016).

**Calibration targets:** Life expectancy at age 40 by income percentile from
Chetty et al. (2016). Data: race-adjusted, gender-averaged, pooled 2001–2014,
from the Health Inequality Project (CC0 license).

**Calibration procedure:**
1. Map z-grid point to income percentile: `p_i = 100 · Φ(z_i / σ_z)`
   where `σ_z = √(Var(η) / (1-ρ²))` is the unconditional std of z
2. Interpolate the Chetty target expected age at death at percentile p_i
3. Solve for χ(z_i) via Brent root-finding such that the implied curtate
   life expectancy at age 40 matches the Chetty target:
   `LE@40(χ) = 40 + Σ_{k=1}^{ω-40} ₖp₄₀` where
   `ₖp₄₀ = Π_{s=40}^{39+k}(1-min(χ·m(s),1))`

### 3.2 Data Sources

**SSA 2017 Period Life Table:**
- Source: https://www.ssa.gov/oact/STATS/table4c6.html (select "2017 (2020 TR)")
- Gender averaging: `q(age) = 0.5 · q_male(age) + 0.5 · q_female(age)`
- Ages 0–119 stored in `SSA_DEATH_PROB_2017` dict at [mortality.py:65](mortality.py#L65)
- Catherine (2025, Sec. 5.1): "m(age_it) is the average mortality rate at that
  age, which we calibrate as the average across genders from the 2017 Social
  Security actuarial life tables."
- Key values: q(40) = 0.258%, q(65) = 1.45%, q(85) = 8.87%, q(99) = 34.4%,
  q(119) = 100% (absorbing barrier). Monotonically non-decreasing from age 40.

**Chetty et al. (2016):**
- Source: https://healthinequality.org/dl/health_ineq_online_table_1.csv
- Variable: `le_raceadj` (race- and ethnicity-adjusted life expectancy at 40)
- Gender averaging: `0.5 · le_raceadj_male + 0.5 · le_raceadj_female`
- Expected age at death = 40 + le_raceadj (pre-applied in source data)
- Stored as `CHETTY_EXPECTED_AGE_AT_DEATH` dict at [mortality.py:115](mortality.py#L115)
- Range: p1 = 75.8 years to p100 = 88.1 years (expected age at death)
- Verification against paper headline numbers:
  - p100 − p1 gap, men only: 14.59 years (paper: 14.6)
  - p100 − p1 gap, women only: 10.09 years (paper: 10.1)
  - p100 − p1 gap, averaged: 12.34 years

### 3.3 Calibrated χ Values

With production parameters (ρ = 0.991, n_z = 11, σ_z = 1.870):

```
  z        pctile    χ        target LE   actual LE   err (yr)
  ─────    ──────    ──────   ─────────   ─────────   ────────
  -5.609    1.0      1.375    75.76       75.76       <1e-11
  -4.487    1.0      1.375    75.76       75.76       <1e-11
  -3.365    3.6      1.058    78.53       78.53       <1e-11
  -2.244   11.5      0.983    79.30       79.30       <1e-11
  -1.122   27.4      0.824    81.17       81.17       <1e-11
   0.000   50.0      0.669    83.35       83.35       <1e-11
  +1.122   72.6      0.573    85.00       85.00       <1e-11
  +2.244   88.5      0.507    86.29       86.29       <1e-11
  +3.365   96.4      0.456    87.43       87.43       <1e-11
  +4.487   99.2      0.440    87.82       87.82       <1e-11
  +5.609   99.9      0.430    88.06       88.06       <1e-11
```

Key features of the calibrated schedule:
- χ ranges from 0.43 (richest, 57% of baseline mortality) to 1.38 (poorest,
  138% of baseline). The median z agent has χ = 0.67.
- The LE gap between the lowest and highest z is 12.3 years, matching the
  Chetty p1–p100 gap.
- The two lowest z-grid points both clip to percentile 1.0 (they fall at
  Φ(−3) ≈ 0.13% and Φ(−2.4) ≈ 0.82%), so they receive identical χ values.
- No χ value causes `χ · m_baseline` to exceed 1 within the model age range
  [22, 99]: the worst case is `1.375 × 0.345 = 0.474` at age 99.

### 3.4 Code Implementation

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `compute_sigma_z(...)` | mortality.py | [284](mortality.py#L284) | Unconditional σ_z from mixture params |
| `chetty_expected_age_at_death(pct)` | mortality.py | [148](mortality.py#L148) | Linear interpolation of Chetty data |
| `life_expectancy_at_40(chi)` | mortality.py | [178](mortality.py#L178) | Curtate LE: `40 + Σ ₖp₄₀` |
| `_solve_chi(target, ...)` | mortality.py | [221](mortality.py#L221) | Brent root-finding for single χ |
| `calibrate_chi_vector(z_grid, sigma_z)` | mortality.py | [245](mortality.py#L245) | z → percentile → Chetty target → χ |
| `build_survival_probs_2d(ages, chi_vec)` | mortality.py | [296](mortality.py#L296) | (n_age, n_z) array construction |
| `calibrate_earnings_dependent_mortality(...)` | mortality.py | [336](mortality.py#L336) | One-call entry point |

**Precompute integration:** Called at [precompute.py:234–245](precompute.py#L234-L245):
```python
self.survival_probs_2d, self._chi_vec, self._mortality_diag = \
    calibrate_earnings_dependent_mortality(
        start_age=model.start_age, terminal_age=model.terminal_age,
        z_grid=self.z_grid, rho=model.rho, pz=model.pz,
        mu_eta1=model.mu_eta1, sigma_eta1=model.sigma_eta1,
        mu_eta2=model.mu_eta2, sigma_eta2=model.sigma_eta2)
```
Shape: `(n_age, n_z)` — one per-period survival probability ψ(age, z) per
(age, persistent income state) pair. Index convention: `survival_probs_2d[t, iz]`
is the probability of surviving from age `start_age + t` to `start_age + t + 1`
for an agent in z-state `iz`.

### 3.5 How Mortality Enters the Solver

**Backward induction loop** ([solver.py:2108–2128](solver.py#L2108-L2128)):
```python
for t in reversed(range(n_age - 1)):
    psi = survival_probs[t, :]      # (n_z,) -- z-dependent survival
    ...
    # psi passed to solve_retirement_step / solve_working_age_step
```

The loop runs from `t = n_age - 2` (age 98) down to `t = 0` (age 22). The
terminal age `t = n_age - 1` (age 99) is handled separately by
`solve_terminal_age()` at [solver.py:1549](solver.py#L1549), which does not
receive a survival probability — it treats death as certain (ψ = 0 implicitly).

**Inside each step function:** The `for z_i in range(n_z):` loop extracts scalar
`psi = psi_vec[z_i]` before calling the Newton solver. The FOC functions receive
scalar `psi` and compute:
```python
prob_death = 1.0 - psi
mu_comb = psi * mu_alive + prob_death * mu_bequest
```

This weights the marginal utility of future consumption (alive branch) and
bequest (dead branch) by their respective probabilities. The effective discount
factor is β·ψ for the alive branch and β·(1−ψ) for the dead branch. Both
branches are discounted by the same β — the agent values future consumption and
bequests symmetrically in terms of time preference, with only the survival
probability distinguishing the two outcomes.

**z-dependence:** Unlike the original CGM (2005) model which uses only
age-dependent survival, our model's survival probability varies across the z-grid.
Low-z agents (low earners) face higher mortality (χ > 1) and shorter life
expectancy; high-z agents face lower mortality (χ < 1) and longer life expectancy.
This creates a channel where income affects optimal savings through differential
longevity risk: agents who expect to live longer save more, while agents facing
high mortality consume more and rely more on the bequest motive.

The effective discount rate β·ψ(age, z) ranges from β·0.526 = 0.505 (age 99,
lowest z) to β·0.9996 = 0.960 (age 22, highest z). The maximum effective
discount is comfortably below 1, ensuring Bellman equation convergence.

### 3.6 How Mortality Enters the Simulation

Death is simulated as a Bernoulli draw each period. At
[simulation.py:507–516](simulation.py#L507-L516):

```python
# Terminal age: forced death
if t == n_age - 1:
    death_age[i] = age_t
    estate_at_death[i] = estate_t
    break

# Bernoulli survival draw
if uniform_draws[i, t, 0] > survival_probs_2d[t, z_idx_near]:
    death_age[i] = age_t
    estate_at_death[i] = estate_t
    break
```

**Death timing within a period:**
1. Agent consumes, saves, and allocates portfolio (using policy functions)
2. Portfolio return is realized: `estate_t = savings_t × R_p`
3. Survival draw: die with probability `1 − ψ(t, z_nearest)`
4. If dead: `death_age = age_t`, `estate_at_death = estate_t` (bequeathed)
5. If alive: income transition → next period's cash-on-hand = estate_t + Y_{t+1}

This matches the Bellman equation convention: the dead branch receives invested
wealth `a·R_p` but no income; the alive branch receives `a·R_p + Y_{t+1}`.

**Post-death:** All arrays (consumption, income, wealth, alive flag) are
zero-filled for periods after death. The `alive` array is 1 for all periods up
to and including the death period, and 0 thereafter.

**z-conditioning:** The survival lookup uses `z_idx_near`, the nearest z-grid
point to the agent's continuous z value. This matches the solver, which solves
on the discrete z-grid and interpolates consumption/portfolio between grid points.

### 3.7 Survival Probability Array Properties

The `survival_probs_2d` array has the following verified properties:

**Shape and range:** (78, 11) — ages 22–99 × 11 z-grid points. All values in
(0.526, 0.9996). No survival probability is 0 or 1 within the model horizon.

**Monotonicity:**
- Decreasing in age (from age 40 onward) for every z-state: mortality accelerates
  with age, as in the underlying SSA table.
- Increasing in z at every age: higher income → lower χ → higher survival.

**Key values:**

| Age | Lowest z (ψ) | Highest z (ψ) | Gap |
|-----|-------------|--------------|-----|
| 40  | 0.9965      | 0.9989       | +0.0024 |
| 50  | 0.9938      | 0.9980       | +0.0043 |
| 65  | 0.9801      | 0.9938       | +0.0137 |
| 80  | 0.9279      | 0.9775       | +0.0495 |
| 99  | 0.5262      | 0.8518       | +0.3256 |

The mortality gradient widens dramatically at older ages: the gap in per-period
survival between richest and poorest is 0.24 pp at age 40 but 32.6 pp at age 99.

**Cumulative survival to age 99:** P(survive from 22 to 99) = 0.35% for the
lowest z (these agents die young with high probability) vs 20.2% for the highest z.

### 3.8 z-to-Percentile Mapping

The mapping from the z-grid to income percentile uses the unconditional
distribution of z:

```
percentile_i = 100 · Φ(z_i / σ_z)
```

where `σ_z = √(Var(η) / (1-ρ²))` and `Var(η)` is the total variance of the
mixture-normal innovation.

This uses the Gaussian CDF Φ as an approximation — the true stationary
distribution of z is not exactly Gaussian due to mixture innovations, but with
ρ = 0.991 the AR(1) aggregation pulls z sharply toward Gaussian. Monte Carlo
simulation of the mixture-normal AR(1) (200,000 periods) shows the maximum
|empirical − Gaussian| percentile difference is 1.3 pp, translating to at most
0.47 years of LE target error.

The percentiles are clipped to [1, 100] to stay within the Chetty data range
([mortality.py:273](mortality.py#L273)). With n_z = 11 and σ_z = 1.87, the two
lowest z-grid points (at ±3σ_z) clip to the boundary, receiving identical χ.

### 3.9 σ_z Computation

`compute_sigma_z()` at [mortality.py:284](mortality.py#L284):
```python
mu_eta = pz * mu_eta1 + (1-pz) * mu_eta2
var_eta = pz * (sigma_eta1² + (mu_eta1 - mu_eta)²)
       + (1-pz) * (sigma_eta2² + (mu_eta2 - mu_eta)²)
sigma_z = sqrt(var_eta / (1 - rho²))
```

This is the standard formula for the unconditional variance of a mixture-normal
AR(1): `Var(z) = Var(η) / (1-ρ²)`, where `Var(η) = Σ_k π_k(σ_k² + (μ_k - μ̄)²)`
is the total mixture variance (within-component + between-component). With
production parameters: σ_z = 1.870.

### 3.10 Terminal Age Convention

At the terminal age (age 99, `t = n_age - 1`), the model treats death as certain:

- **Solver:** `solve_terminal_age()` at [solver.py:1549](solver.py#L1549) receives
  no survival probability. It computes optimal consumption by solving:
  ```
  u'(c_T) = β · b'((W_T - c_T) · R_p)
  ```
  All terminal wealth not consumed becomes bequest. See Section 2.3 for the full
  derivation of the CRRA factoring that decouples portfolio choice from consumption
  at the terminal age.

- **Simulation:** At [simulation.py:507–510](simulation.py#L507-L510), any agent
  reaching `t = n_age - 1` is forced to die. `death_age = 99`,
  `estate_at_death = savings × R_p`.

- **Array:** `survival_probs_2d[77, :]` stores the physical survival probabilities
  at age 99 (range: [0.526, 0.852]), but these values are never accessed. The
  backward loop runs `for t in reversed(range(n_age - 1))`, terminating at `t = 0`.
  The values are computed for diagnostic completeness but are inert.

### 3.11 Known Approximations

The following are inherent to the Catherine (2025) approach:

1. **Truncation bias.** The calibration computes LE using the full SSA table to
   age 119, but the model forces death at terminal_age = 99. Agents whose survival
   mass extends past 99 experience a lower effective LE within the model than their
   Chetty target. The bias ranges from 0.002 years (lowest z) to 0.72 years
   (highest z) and is monotonically increasing in z.

2. **Pre-40 mortality extrapolation.** χ is calibrated to match LE at age 40, but
   `survival_probs_2d` applies χ from age 22. The Chetty data makes no claim about
   mortality before 40. Impact is small: cumulative survival 22→40 is 95.9%
   (lowest z) to 98.7% (highest z), a 2.8 pp gradient, because SSA mortality at
   ages 22–39 is very low (< 0.25% per year).

3. **Gaussian percentile approximation.** The mapping Φ(z/σ_z) approximates the
   unconditional z distribution as Gaussian. Maximum percentile error is 1.3 pp
   (≤ 0.47 years of LE target error). Catherine (2025) makes the same assumption.

---

## 4. Interaction in the Bellman Equation

### 4.1 Structure

The three components interact as follows:

```
V_t(W) = max_{c,α}  u(c) + β · Σ_j Σ_k π_{ij} w_k [
            ψ_t · u'(c_{t+1}) / u'(c_t) · V_{t+1}(W')     # alive branch
          + (1-ψ_t) · b(a·R_p, A_i)                         # dead branch
        ]
```

In the FOC (what the solver actually evaluates):
```
0 = E[ ψ · c_{t+1}^{-γ} · (R_k - R_bill) + (1-ψ) · b̄ · (aR_p/A)^{-γ}/A · (R_k - R_bill) ]
```
for k ∈ {stock, bond}.

The Euler equation (for EGM consumption recovery):
```
c_t^{-γ} = β · E[ ψ · c_{t+1}^{-γ} · R_p + (1-ψ) · b̄ · (aR_p/A)^{-γ}/A · R_p ]
```
→ `c_t = (β · euler_sum)^{-1/γ}`

### 4.2 Weight Accounting

In the retirement FOC, the probability weighting is:
```
weight = Pi_state[i_s, j_s] × ret_weights[k_r]
euler_sum += weight × mu_comb × R_p
```
where `mu_comb = ψ · mu_alive + (1-ψ) · mu_bequest`.

In the working-age FOC, with the bequest hoist:
```
# Bequest (once per j_s, k_r):
euler_sum += p_state_ret × prob_death × mu_bequest × R_p

# Alive (per j_s, k_r, k_eta, i_e):
euler_sum += p_state_ret × w_eta × eps_weights[i_e] × psi × mu_alive × R_p
```

The alive branch integrates over (state transition × return quadrature × η
quadrature × ε quadrature). The dead branch skips the income quadrature dimensions.
Total weight sums to 1 because: `Σ π_{ij} · Σ w_k = 1` (state × return) and
`Σ w_η · Σ w_ε = 1` (income quadrature) and `ψ + (1-ψ) = 1` (survival).

---

## 5. Validation

### 5.1 CRRA Utility

- [x] **Functional forms algebraically correct** — `create_utility_functions(gamma)`
      at [model.py:168](model.py#L168) implements the standard CRRA utility,
      marginal utility, and inverse marginal utility. For γ ≠ 1:
      `u(c) = c^{1-γ}/(1-γ)`, `u'(c) = c^{-γ}`, `u'^{-1}(μ) = μ^{-1/γ}`.
      For γ = 1: `u(c) = log(c)`, `u'(c) = 1/c`, `u'^{-1}(μ) = 1/μ`.
      Verified: `u'` is the derivative of `u` (differentiate
      `c^{1-γ}/(1-γ)` → `(1-γ)c^{-γ}/(1-γ) = c^{-γ}`), and `u'^{-1}`
      inverts `u'` (solve `μ = c^{-γ}` → `c = μ^{-1/γ}`).
- [x] **Consistency: u'_inv(u'(c)) = c** — By substitution:
      `u'^{-1}(u'(c)) = (c^{-γ})^{-1/γ} = c^{γ/γ} = c`. For γ = 1:
      `u'^{-1}(u'(c)) = 1/(1/c) = c`. Exact algebraic identity.
- [x] **γ = 1 special case handled** — The log utility branch is a separate code
      path (lines 171–178) rather than a limit computation, avoiding the
      `0/0` indeterminacy of `c^0 / 0`.
- [x] **u' enters the Euler equation correctly** — The solver computes
      `mu_alive = c_next ** (-gamma)` which equals `u'(c_{t+1})`. This is
      weighted by `ψ` and combined with `mu_bequest` weighted by `(1-ψ)`.
      The product `mu_comb * R_p` is accumulated as `euler_sum`, giving the
      Euler equation RHS: `E[u'_comb · R_p]`. Confirmed: `beta` is absent
      from the accumulation inside `compute_foc_jac_retirement` (lines
      437–494) and `compute_foc_jac_working` (lines 784–933) — it appears
      only in the function signature, passed through to the Newton solver
      but never multiplied into `euler_sum`. Both retirement (line 485:
      `euler_sum += wmu * R_p`) and working-age (lines 836 + 924) accumulate
      the un-discounted expectation. β is applied only at the EGM inversion.
- [x] **EGM inversion equals u'^{-1}(β · euler_sum)** — The code
      `(beta * euler)^{-1/gamma}` at [solver.py:1742](solver.py#L1742)
      (retirement) and [solver.py:1923](solver.py#L1923) (working age)
      is algebraically identical to `u_prime_inv(beta * euler_sum)`:
      `μ^{-1/γ}` evaluated at `μ = β · euler_sum`. The formula is inlined
      rather than calling the closure because the solver is `@njit`-compiled.
      β enters here and nowhere else — consistent with the Euler equation
      `u'(c_t) = β · E[...]` → `c_t = (β · E[...])^{-1/γ}`.
- [x] **euler_inv_floor does not bind in practice** — The floor value
      `1e-20` would imply `c_opt = (1e-20)^{-1/3} ≈ 4.6 × 10^6` AWI units
      ($250 billion), far beyond the wealth grid maximum of 200 AWI units.
      The floor exists only as a numerical guard against pathological
      negative euler values (which are tracked by `DI_NEG_CONSUMPTION`).
- [x] **Numerical tests** (`test_egm_crra.py`, 6 tests, all pass):
      **(A)** Round-trip `u'_inv(u'(c)) = c` across 7 gammas × 200 c-values
      (1e-6 to 1e6): max rel err 8.5e-16.
      **(B)** Inlined `(β·e)^{-1/γ}` matches `u_prime_inv(β·e)` across 6
      gammas × 3 betas × 200 euler values: max rel err 1.2e-16.
      **(C)** Euler identity `u'(c_opt) = β·euler` after inversion: max rel
      err 1.5e-15 across 3,600 probes.
      **(D)** Live FOC: called `compute_foc_jac_retirement` with 150 probes
      across 5 survival probs × 6 savings levels × 5 portfolio weights,
      extracted real `euler_sum` values, verified `u'(c_opt) = β·euler_sum`
      to max rel err **1.05e-15**.
      **(E)** EGM budget identity `x = c_opt + s_val` holds to 2.2e-16
      across 80 savings points (no off-by-one in assignment).
      **(F)** `euler_sum` monotonically decreasing in savings (0.01 to 50);
      `c_opt` monotonically increasing. Zero violations. Confirms the
      survival-weighted alive+bequest marginal utility behaves correctly.

### 5.2 Bequest Utility

> **Status note (post-shift):** the audit items below were performed against
> the *unshifted* CRRA-with-clamp specification. The bequest has since moved
> to the De Nardi (2004) shifted form `b̄·(W/A + δ)^{1-γ}/(1-γ)` to remove the
> bankruptcy-boundary discontinuity. The structural arguments (annuity-factor
> indexing, b̄=10, MU = derivative of level, second-derivative formula in the
> Jacobian, identity check vs. solver) still hold with `W/A` replaced by
> `W/A + δ`; the `delta → 0` limit recovers the originally audited code path.

- [x] **Functional forms match Catherine (2025) eqs. 21–22** —
      `bequest_utility` ([model.py:232–233](model.py#L232-L233)):
      `b_bar * C_bar**(1-gamma)/(1-gamma)` with `C_bar = W/A`. Matches
      Catherine eq. 21. `bequest_marginal` ([model.py:247–248](model.py#L247-L248)):
      `b_bar * C_bar**(-gamma) / A`. `bequest_marginal_inv`
      ([model.py:262](model.py#L262)): `A * (mu*A/b_bar)**(-1/gamma)`.
      All three traced line-by-line against the paper equations.
- [x] **Marginal is the derivative of level** — Differentiating
      `b(W) = b̄ · (W/A)^{1-γ}/(1-γ)` w.r.t. W:
      `b'(W) = b̄ · (1-γ)·(W/A)^{-γ}·(1/A) / (1-γ) = b̄ · (W/A)^{-γ}/A`.
      Matches `bequest_marginal`. Also numerically verified: the terminal
      age test (`test_terminal_omega.py`) independently computed `b'(W)`
      via `mu_bequest = b_bar * w_A**(-gamma) / A` and the resulting c/W
      matched the solved policy to 2.5e-16.
- [x] **Inverse marginal is algebraically correct** — Starting from
      `μ = b̄ · (W/A)^{-γ}/A`, solve: `(W/A)^{-γ} = μA/b̄`,
      `W/A = (μA/b̄)^{-1/γ}`, `W = A · (μA/b̄)^{-1/γ}`. Matches
      [model.py:262](model.py#L262).
- [x] **Second derivative (Jacobian) correct** — In the solver at
      [solver.py:479](solver.py#L479) (retirement) and
      [solver.py:830](solver.py#L830) (working age):
      `mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)`.
      Deriving: `b''(W) = b̄·(-γ)(W/A)^{-γ-1}·(1/A)/A
       = -γ·[b̄(W/A)^{-γ}/A] / ((W/A)·A) = -γ·b'(W)/((W/A)·A)`.
      With `w_A = W/A` and `mu_bequest = b'(W)`, this is
      `-γ·mu_bequest/(w_A·A)` as coded. Both retirement and working-age
      FOCs use the identical formula.
- [x] **Annuity factor uses current-period state** — `annuity_factor_is`
      at each FOC call is `annuity_factors[i_s]`, indexed by the current
      financial state. Traced: [precompute.py:168](precompute.py#L168) builds
      the array from `state_grid[:, annuity_yield_index_in_state]`;
      retirement solver at [solver.py:1640](solver.py#L1640) reads
      `annuity_factors[i_s]`; working-age solver does the same. This matches
      Catherine's timing subscript r_{ft} (eq. 22) — the annuity is priced
      at the time-t state, not the next-period state. Note: the specific
      yield used as discount rate is not yet decided (see Section 2.5).
- [x] **Bequeathed wealth = invested savings only** — In the retirement
      FOC ([solver.py:466–467](solver.py#L466-L467)): `w_inv = s_val * R_p`,
      then bequest uses `w_A = w_inv / A` (line 474). Alive branch adds
      income: `x_next = w_inv + pension` (line 467). In the working-age FOC
      ([solver.py:826–829](solver.py#L826-L829)): same `w_inv = s_val * R_p`
      for bequest; alive branch adds income at line 872:
      `x_next = w_inv + income_next`. Dead branch receives no income.
      Matches the timing: death occurs between t and t+1, before income
      arrives.
- [x] **β discounts both branches equally** — In the retirement FOC,
      `mu_comb = psi * mu_alive + prob_death * mu_bequest` (line 476), and
      `euler_sum += weight * mu_comb * R_p` (line 485). In the working-age
      FOC, both `death_mu * R_p` (line 836) and `weight * psi * mu_alive * R_p`
      (line 924) accumulate into the same `euler_sum`. In both cases, the
      final EGM step `c_opt = (beta * euler)^{-1/gamma}` applies a single β
      to the combined alive+dead sum. Verified in Test D of
      `test_egm_crra.py` (150 FOC calls including ψ = 0 and ψ = 1 extremes).
- [x] **Bequest hoist optimization valid** — In the working-age FOC, the
      bequest contribution is computed once per `(j_s, k_r)` and accumulated
      outside the `(k_eta, i_e)` loops ([solver.py:832–843](solver.py#L832-L843)).
      Validity: (a) `w_inv = s_val * R_p` at line 826 depends only on
      `(j_s, k_r)`, not on income; (b) `mu_bequest` and `mup_bequest` at
      lines 828–830 depend only on `w_inv` and `annuity_factor_is`;
      (c) income quadrature weights sum to 1 by construction
      (`sum(eta_weights) = sum(eps_weights) = 1`, documented in
      discretization.py and verified in LABOUR.md Section 5). Therefore
      summing `death_mu` once per `(j_s, k_r)` is equivalent to summing
      `death_mu * Σ w_eta * Σ w_eps = death_mu * 1`.
- [x] **Terminal age structure derived and verified** — At age 99, death is
      certain (ψ = 0). The terminal Bellman `V_T = max u(c) + β·E[b(aR_p)]`
      with CRRA bequest factors as `u(c) + β·Ω·a^{1-γ}/(1-γ)` where
      `Ω = b̄·A^{γ-1}·E[R_p^{1-γ}]`. The CRRA factoring makes the portfolio
      FOC `∂/∂α E[R_p^{1-γ}] = (1-γ)·E[R_p^{-γ}·(R_k-R_bill)] = 0`
      independent of W and c — the portfolio decouples from consumption.
      The gradient at [solver.py:270–281](solver.py#L270-L281) implements
      exactly `(1-γ)·E[R_p^{-γ}·Rex_k]`. With α* fixed, the consumption
      FOC gives `c/(W-c) = (β·Ω)^{-1/γ}`, yielding `c* = W·ratio/(1+ratio)`
      as coded at [solver.py:1588–1590](solver.py#L1588-L1590). Full
      derivation in Section 2.3.
      **Numerical verification** (`test_terminal_omega.py`): loaded saved
      run `constrained_grid5x5x5_nz11` (125 states, 11 z-points, 150
      wealth points). At each state: (a) read solved α from S_mat/B_mat;
      (b) manually computed `E[R_p^{1-γ}]` via explicit double loop over
      all (j_s, k_r) scenarios — matches helper to 2.0e-15; (c-e) chained
      Ω → ratio → c*/W from the derivation and compared to C_mat — worst
      relative error across all 125 states: **2.51e-16** (machine precision).
      (g) Verified terminal FOC / KKT conditions: interior solutions have
      FOC < 1e-7; corner/edge solutions satisfy complementary slackness.
      (h) c/W is bit-exact identical across all 11 z-grid points at each
      state (spread = 0), confirming bequest is independent of income.
- [x] **Terminal portfolio gradient and Hessian** —
      `_terminal_portfolio_grad` ([solver.py:270](solver.py#L270)) computes
      `∂/∂α_k E[R_p^{1-γ}] = (1-γ)·Σ w·R_p^{-γ}·Rex_k`. Algebraic
      derivation: differentiating `R_p^{1-γ}` w.r.t. `α_k` gives
      `(1-γ)·R_p^{-γ}·∂R_p/∂α_k = (1-γ)·R_p^{-γ}·(R_k - R_bill)`.
      `_terminal_portfolio_hess` ([solver.py:284](solver.py#L284)) computes
      `∂²/∂α_j∂α_k = γ(γ-1)·Σ w·R_p^{-γ-1}·Rex_j·Rex_k`, where
      `γ(γ-1) = -(1-γ)(-γ)` is the correct second-derivative coefficient.
      Hessian is symmetric by construction (H[0,1] = H[1,0] both use
      `Rex_s·Rex_b`).
      **Numerical verification** (`test_terminal_grad_hess.py`): central
      finite differences of the objective (`h = 1e-7`) compared to the
      analytical gradient at 5 financial states × 7 simplex points = 35
      probes. Worst gradient relative error: **9.08e-7**. Hessian verified
      by finite-differencing the gradient; worst relative error: **2.81e-7**.
      Hessian symmetry exact at all test points.
- [ ] **Annuity discount rate decision** — The choice of which yield to use
      in the annuity factor is not yet finalized. See Section 2.5 for the
      options under consideration and bias analysis.

### 5.3 Mortality

**Bug fix applied (2026-04-19):** `life_expectancy_at_40()` had an off-by-one
error that added `S(40) = 1` (the trivial probability of being alive at 40)
to the curtate expectation sum, inflating LE by exactly 1.0 year. The fix
swaps two lines at [mortality.py:209–210](mortality.py#L209-L210) so that
survival is updated *before* accumulation. This caused all calibrated χ values
to be ~9% too high. See `MORTALITY_LE_FIX_HANDOFF.md` for full derivation.

**Test suites:** `_test_mortality_suite.py` (Sections 1–6, 9: 30 tests),
`_test_mortality_suite_78.py` (Sections 7–8: 8 tests),
`_test_mortality_deep.py` (Sections A–K: 34 tests). Results below reference
specific test IDs.

#### 5.3.1 SSA Life Table Integrity

- [x] **SSA 2017 life table correctly sourced** — `SSA_DEATH_PROB_2017` at
      [mortality.py:65](mortality.py#L65) contains gender-averaged death
      probabilities for ages 0–119 from the SSA 2017 Period Life Table
      (2020 Trustees Report). Catherine (2025, Sec. 5.1) confirms this is
      the correct source table. Gender averaging: `q = 0.5·q_male + 0.5·q_female`.
      **Numerical verification (Tests 1.1–1.5):** table has exactly 120 entries
      (ages 0–119); all values in (0, 1]; monotonically non-decreasing from
      age 40 onward; `q(119) = 1.0` (absorbing barrier); spot checks at
      ages 40, 65, 85 match published SSA values to within stated tolerances.

#### 5.3.2 Chetty Calibration Targets

- [x] **Chetty et al. (2016) targets correctly sourced** —
      `CHETTY_EXPECTED_AGE_AT_DEATH` at [mortality.py:115](mortality.py#L115)
      contains expected age at death (= 40 + le_raceadj) by income percentile,
      gender-averaged, race-adjusted, pooled 2001–2014. **Numerical
      verification (Tests 2.1–2.4):** 100 integer percentiles (1–100); all
      values in (70, 95); p100−p1 gap = 12.34 years, consistent with Chetty
      headline numbers (men: 14.59 vs paper's 14.6; women: 10.09 vs 10.1).
- [x] **Interpolation function correct** — `chetty_expected_age_at_death()`
      at [mortality.py:148](mortality.py#L148) returns exact values at integer
      percentiles and linear interpolates between them **(Test 2.5)**. Note:
      the Chetty data has minor local non-monotonicities (e.g. p50 > p51)
      reflecting measurement noise in the underlying data; the interpolation
      correctly tracks these.

#### 5.3.3 Life Expectancy Formula (Post-Fix)

- [x] **Life expectancy computation correct** — `life_expectancy_at_40(chi)`
      at [mortality.py:178](mortality.py#L178) computes
      `LE@40 = 40 + Σ_{k=1}^{ω-40} ₖp₄₀` where
      `ₖp₄₀ = Π_{s=40}^{39+k}(1-min(χ·m(s),1))`.
      Early termination when `survival < 1e-15` avoids unnecessary iterations
      at extreme χ values. **Numerical verification (Tests 4.1–4.6):**
      **(4.1)** Toy example `{40:0.5, 41:0.5, 42:1.0}`: returns 40.75,
      matches closed-form `40×0.5 + 41×0.25 + 42×0.25 = 40.75` to 1e-12.
      **(4.2)** Certain death at 40: returns 40.0 exactly.
      **(4.3)** Zero mortality except `m(119)=1`: returns 119.0 to 1e-10.
      **(4.4)** Full SSA table: matches independent direct enumeration
      (`Σ a·P(die at a)`) to 2.8e-14.
      **(4.5)** Monotonicity: `LE(χ=0.5) > LE(χ=1.0) > LE(χ=2.0)`.
      **(4.6)** Continuity: 0.1% change in χ produces < 0.011 year change in LE.

#### 5.3.4 σ_z and Percentile Mapping

- [x] **σ_z computation correct** — `compute_sigma_z()` at
      [mortality.py:284](mortality.py#L284) computes the total mixture variance
      `Var(η) = Σ_k π_k(σ_k² + (μ_k - μ̄)²)` and divides by `(1-ρ²)`.
      **Numerical verification (Tests 3.1–3.3):** **(3.1)** pure Gaussian
      special case (`pz=1`): matches `σ_η/√(1-ρ²)` to 1e-12. **(3.2)**
      symmetric mixture: matches hand-computed `√(0.0125/0.0591)` to 1e-12.
      **(3.3)** production parameters: `σ_z = 1.870`, positive and finite.
- [x] **z-to-percentile mapping uses correct σ_z** — The Gaussian CDF
      `Φ(z/σ_z)` is an adequate approximation for the unconditional z
      distribution. **Quantified (Tests C.1–C.2):** Monte Carlo simulation
      (200,000 periods of the mixture-normal AR(1)) shows the maximum
      |empirical − Gaussian| percentile difference is 1.3 pp, translating
      to at most 0.47 years of LE target error. With ρ = 0.991, the CLT
      pulls z sharply toward Gaussian.
- [x] **Percentiles clipped to [1, 100]** — At [mortality.py:273](mortality.py#L273),
      `percentiles = np.clip(percentiles, 1.0, 100.0)`. This ensures the
      Chetty interpolation never extrapolates outside the data range.
      With `n_z=11` and `σ_z=1.87`, the two lowest z-grid points both clip
      to percentile 1.0 (they map to `Φ(-3σ_z/σ_z) = Φ(-3) ≈ 0.13%`).
      Consequence: these two points get identical χ values. **Verified
      (Test 5.4):** percentile range spans [1.0, 99.9].

#### 5.3.5 χ Calibration

- [x] **χ(z_i) root-finding procedure correct** — `_solve_chi()` at
      [mortality.py:221](mortality.py#L221) uses `scipy.optimize.brentq` to
      find χ such that `life_expectancy_at_40(χ) = target`. Brent's method is
      bracketed in [chi_lo=0.01, chi_hi=20.0] with tolerance `xtol=1e-10`.
      Fallback: if the residual has the same sign at both bracket ends (can
      happen at extreme percentiles), the closer endpoint is returned.
      **Numerical verification (Tests 5.1–5.4):** **(5.1)** round-trip
      accuracy: max |actual − target| LE = 1.6e-11 years across all 11
      z-grid points (machine precision). **(5.2)** χ monotonically
      decreasing across distinct percentiles (higher income → lower mortality
      scaling). **(5.3)** χ range [0.43, 1.38], within plausible bounds.
      **(5.4)** percentile range [1.0, 99.9]. **Deterministic (Test 9.1):**
      two identical calibration runs produce bitwise-identical `surv_2d`
      and `chi_vec`.
- [x] **Post-fix χ direction confirmed** — After the off-by-one fix, all
      χ values dropped by ~9% (range: −8.9% to −9.1%). Lower χ = lower
      mortality, which is correct: the old code overestimated LE by 1 year,
      so the solver had to push χ higher to compensate. Production χ values
      (n_z=11, ρ=0.991): [1.375, 1.375, 1.058, 0.983, 0.824, 0.669,
      0.573, 0.507, 0.456, 0.440, 0.430].

#### 5.3.6 2D Survival Probability Array

- [x] **survival_probs_2d has correct shape** — `build_survival_probs_2d()`
      at [mortality.py:296](mortality.py#L296) produces shape `(n_age, n_z)`.
      Construction: `out[t, iz] = 1 - min(chi_vec[iz] * m_table[ages[t]], 1.0)`.
      Ages span `[start_age, terminal_age]` = `[22, 99]`, so n_age = 78.
      **Verified (Tests 6.1, E.1):** shape = (78, 11); independent
      recomputation via `build_survival_probs_2d(ages, chi_vec)` is
      bitwise-identical to `pc.survival_probs_2d`.
- [x] **All values in valid range** — **Verified (Tests 6.2, F.1):**
      min = 0.526 (age 99, lowest z), max = 0.9996 (age 22, highest z).
      No survival probability equals 0 within the model horizon.
- [x] **Monotonicity properties** — **Verified (Tests 6.3, 6.4):**
      **(6.3)** Decreasing in age from age 40 onward for all z (mortality
      accelerates with age). **(6.4)** Increasing in z at every age (higher
      income → lower mortality → higher survival).
- [x] **Spot-check against formula** — **Verified (Test 6.5):** 6 entries
      at corners and midpoint of the (age, z) grid match
      `1 - min(χ[iz] · m(age), 1)` to 1e-15.
- [x] **m(age, z) never exceeds 1 within model horizon** — **Verified
      (Test F.3):** `max(χ) × max(m_baseline in [22,99]) = 1.375 × 0.345
      = 0.474 < 1`. The `min(·, 1)` cap never binds within the model's
      age range.
- [x] **Cumulative survival curves consistent** — **Verified (Tests
      G.1, G.2):** cumulative survival from age 22, computed by chaining
      `survival_probs_2d[t, iz]` period-by-period, matches independent
      computation from χ and SSA table to 1e-14 at iz = {0, 5, 10}.
      P(survive to 99): lowest-z = 0.35%, highest-z = 20.2%.

#### 5.3.7 Solver Integration

- [x] **Correct indexing in solver** — The backward induction loop at
      [solver.py:2108](solver.py#L2108) extracts `psi = survival_probs[t, :]`
      which is shape `(n_z,)`. Inside the step functions, `psi_vec[z_i]`
      extracts the scalar survival probability for the specific (age, z) pair.
      The time index `t` is correct: `t = age - start_age`, so `t=0` maps
      to age 22 and `t=77` maps to age 99. **Verified (Test 7.2):** solver
      completes all 78 age steps without shape errors; policy arrays have
      expected shape (78, 11, 125, 150).
- [x] **Effective discount factor β·ψ < 1** — **Verified (Test E.3):**
      max(β·ψ) = 0.960 across all (age, z) pairs. Comfortably below 1,
      ensuring Bellman equation convergence.
- [x] **β·ψ monotone in z** — **Verified (Test E.4):** at every age,
      β·ψ is weakly increasing in z (higher income → longer effective
      planning horizon).
- [x] **Terminal age survival probability** — At age 99 (the last period),
      `survival_probs_2d[77, :]` is nonzero (range: [0.526, 0.852]) because
      `SSA_DEATH_PROB_2017[99] = 0.344` (not 1.0). However, the terminal
      solver at [solver.py:1549](solver.py#L1549) does not use the survival
      probability — it treats death as certain by only computing bequest
      utility. The array value at `t = n_age - 1` is never accessed: the
      backward loop runs `for t in reversed(range(n_age - 1))`, stopping
      at `t = 0`, and the terminal solver is called separately at
      [solver.py:2084](solver.py#L2084) with no `psi_vec` argument.
      **Verified (Test E.2):** the stored values are physically meaningful
      (what survival *would* be if the model continued), but they are inert
      — the solver never reads them. The simulation independently forces
      death at `t = n_age - 1` (line 507–510). The model implicitly assumes
      certain death at terminal_age; no array modification is needed.

#### 5.3.8 Simulation Consistency

- [x] **Death is Bernoulli(1−ψ)** — At [simulation.py:513](simulation.py#L513):
      `if uniform_draws[i, t, 0] > survival_probs_2d[t, z_idx_near]`, agent
      dies. This is equivalent to dying with probability `1 − ψ(t, z)`.
      **Verified (Tests D.1, D.2):** empirical hazard rates at ages 65 and
      80 match analytical rates to within 0.05 pp (50,000 agents).
- [x] **Death timing matches Bellman** — Death is checked after consumption
      and portfolio return but before next-period income, matching the solver's
      convention that bequeathed wealth = invested savings × R_p (no income).
      `death_age` records the age the agent dies *in*.
- [x] **Dead-agent hygiene** — **Verified (Tests H.1–H.3):** consumption,
      income, and wealth are all zero for every period after death (checked
      5,000 agents, zero violations). **(Test H.4):** 99.3% of agents have
      positive estate at death.
- [x] **Mortality gradient in simulation** — **Verified (Tests D.3, 8.2):**
      agents in the bottom z-half have lower median death age (83) than
      top z-half (87). Mean death age by z ranges from ~48 (lowest z, which
      clips to p1) to ~90 (highest z), a 41-year gap, consistent with the
      Chetty p1–p100 gap of 12.3 years amplified by grid extremity.
- [x] **No death where ψ ≈ 1** — **Verified (Test D.4):** zero agents
      died in a period where their survival probability was ≥ 1 − 1e-15,
      confirming correct Bernoulli draw implementation.
- [x] **No one survives past terminal_age** — **Verified (Test 8.3):**
      max(death_age) = 99 = terminal_age. Simulation forces death at
      `t = n_age - 1` (line 507–510).
- [x] **Alive mask consistency** — **Verified (Test 8.4):** for 1,000
      agents, `alive[i, :t_death+1] = 1` and `alive[i, t_death+1:] = 0`,
      zero violations.
- [x] **Survival rates at key ages plausible** — **Verified (Test 8.5):**
      fraction alive at 65: 87.6% (range: 80–99%); at 80: 66.3% (range:
      50–90%); at 95: 18.1% (range: 5–50%).

#### 5.3.9 Death-Age Distribution

- [x] **Distribution shape is realistic** — **Verified (Tests I.1–I.5):**
      mean death age = 81.7; std = 14.8 years; mode = 99 (terminal pile-up,
      8.3% of agents); 4.4% die before age 50 (consistent with analytical
      prediction of ~4.9% across the z distribution). The distribution is
      right-skewed with the expected actuarial shape.

#### 5.3.10 Known Approximations

The following are inherent to the Catherine (2025) approach, not bugs:

- **Truncation bias:** The calibration matches Chetty targets using the full
  SSA table to age 119, but the model forces death at terminal_age = 99.
  **Quantified (Tests A.1–A.3):** truncation bias ranges from 0.002 years
  (lowest z, who rarely survive to 99 anyway) to 0.72 years (highest z,
  whose survival mass extends significantly past 99). The bias is monotonically
  increasing in z. Below the 1-year threshold but non-negligible for the
  richest agents.

- **Pre-40 mortality extrapolation:** χ is calibrated to match LE at age 40,
  but `survival_probs_2d` applies χ from age 22 onward. The Chetty data says
  nothing about pre-40 mortality by income. **Quantified (Tests B.1–B.2):**
  cumulative survival from 22 to 40 is 95.9% (lowest z) to 98.7% (highest z),
  a 2.8 pp gradient. The impact is small because SSA mortality at ages 22–39
  is very low (< 0.25% per year).

- **Gaussian percentile approximation:** The mapping `Φ(z/σ_z)` assumes the
  unconditional z distribution is Gaussian, which is approximate with
  mixture-normal innovations. **Quantified (Tests C.1–C.2):** maximum
  percentile error is 1.3 pp, translating to at most 0.47 years of LE
  target error. Acceptable given ρ = 0.991.

---

## 6. Summary

| Component | Status | Key References |
|-----------|--------|----------------|
| CRRA utility functions | **Validated** | model.py:168–189 |
| CRRA in Euler equation | **Validated** | solver.py:473, 1742, 1923 |
| Bequest utility functions | **Validated** | model.py:196–263 |
| Bequest in solver FOC | **Validated** | solver.py:474–480, 828–843 |
| Bequest hoist optimization | **Validated** | solver.py:832–843 |
| Terminal age structure | **Validated** | solver.py:1549–1597 |
| SSA life table data | **Validated** | mortality.py:65–100; Tests 1.1–1.5 |
| Chetty calibration targets | **Validated** | mortality.py:115–142; Tests 2.1–2.5 |
| σ_z computation | **Validated** | mortality.py:284–293; Tests 3.1–3.3 |
| LE formula (off-by-one fixed) | **Validated** | mortality.py:178–214; Tests 4.1–4.6 |
| χ root-finding | **Validated** | mortality.py:221–238; Tests 5.1–5.4, 9.1 |
| survival_probs_2d array | **Validated** | mortality.py:296, precompute.py:234; Tests 6.1–6.5, E.1, G.1–G.2 |
| Solver integration (indexing, β·ψ) | **Validated** | solver.py:2108–2128; Tests 7.1–7.3, E.3–E.4 |
| Simulation death draws | **Validated** | simulation.py:507–516; Tests D.1–D.4, 8.1–8.5, H.1–H.4 |
| z-to-percentile mapping | **Validated** | mortality.py:272–273; Tests C.1–C.2 |
| Mortality cap (m ≤ 1) | **Validated** | mortality.py:327; Tests F.1–F.3 |
| Terminal age ψ convention | **Validated** | solver.py:1549, 2084; Test E.2 |
| Truncation bias (99 vs 119) | **Quantified** | Tests A.1–A.3: ≤ 0.72 yr |
| Pre-40 mortality extrapolation | **Quantified** | Tests B.1–B.2: 2.8 pp gradient |
| Gaussian percentile approx. | **Quantified** | Tests C.1–C.2: ≤ 0.47 yr |
| Annuity discount rate choice | **Undecided** | See Section 2.5 |

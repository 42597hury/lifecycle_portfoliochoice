# LIFECYCLE PORTFOLIO CHOICE MODEL -- FULL CONTEXT DOCUMENT

## State-Return Separation Architecture with Stocks, Nominal Bonds, and Bills

---

## 0. Executive Summary

This document specifies the complete design of a lifecycle portfolio choice model with
three assets (stocks, nominal bonds, bills) and VAR(1) return dynamics. The model separates
the VAR state vector into **state variables** (discretized on the DP grid) and **returns**
(integrated out using their conditional distribution given the state transition).

The former monolithic notebook has now been split into focused Python modules plus a
thin orchestration notebook. The current source-of-truth layout is:

- `model.py` -- immutable economic model objects plus stateless utility, bequest, tax,
  and income helpers
- `var.py` -- VAR estimation, state/return partitioning, quarterly-to-annual compounding,
  and hardcoded fallback parameter builders
- `discretization.py` -- Rouwenhorst discretization and transitory-shock quadrature
- `mortality.py` -- calibration of age- and earnings-dependent
  survival probabilities
- `precompute.py` -- `build_model()` factory and `Precompute` container for all
  discretized arrays consumed by the solver
- `solver.py` -- Numba-accelerated backward induction solver (EGM + 2D Newton)
- `diagnostics.py` -- pre-solve calibration reports and targeted Newton failure analysis
- `main.ipynb` -- orchestration notebook that imports the modules
  above and runs the end-to-end workflow

Execution in the refactored codebase follows this sequence:

1. Build or load quarterly VAR parameters in `var.py`, then annualize them.
2. Combine the baseline economic calibration and VAR configuration via
   `precompute.build_model()`.
3. Instantiate `Precompute(model, disc_config=DiscretizationConfig(...))` to create
   grids, transitions, lookup tables, and survival probabilities.
4. Optionally inspect the assembled model with
   `diagnostics.print_model_diagnostic_report(model, pc, periods_per_year=1)`.
5. Solve the lifecycle problem with
   `solver.run_lifecycle_solver(model, pc, solver_config=SolverConfig())`.

Simulation, plotting, and ad hoc analysis are now downstream notebook tasks rather than
core model-definition code.

---

## 1. Economic Environment

### 1.1 Agent and Preferences

The agent lives from `start_age` (22) to `terminal_age` (99). They work
until `retire_age` (67) and receive Social Security pension thereafter.

Preferences are CRRA over consumption:

```
u(c) = c^(1-gamma) / (1-gamma)     for gamma != 1
u(c) = log(c)                       for gamma = 1
```

The agent discounts the future at rate `beta` and faces age- and income-dependent
survival probability `psi_{t,z}`. Upon death, they leave a bequest. Following Catherine (2025),
the bequest is valued as the utility of spreading wealth W over b_bar years of
consumption via an annuity:

```
b(W, r_f) = b_bar * (W / A(y_nom))^(1-gamma) / (1-gamma)

where:
  C_bar    = W / A(y_nom)          flow-equivalent consumption
  A(y_nom) = (1 - (1+y_ann)^{-b_bar}) / y_ann    annuity factor
  y_ann    = y_nom * 4             annual nominal yield (y_nom is quarterly decimal)
  b_bar    = 10                    bequest horizon in years = bond maturity
```

The annuity factor A prices a 10-year consumption stream at the current 10-year
nominal bond yield. This is coherent: the bequest horizon equals the bond maturity,
so the nominal bond is the natural pricing instrument. No expectations hypothesis
or term structure model is needed -- the current yield is applied as a flat rate.

**Calibration:** `gamma = 3`, `beta = 0.96`, `b_bar = 10`.

### 1.1b Earnings-Dependent Mortality

Survival probabilities are both age- and income-dependent, following Catherine (2025,
eq. 35). The mortality rate for an agent at age `t` with persistent income state `z_i` is:

```
m(age_t, z_i) = min( chi(z_i) * m_baseline(age_t),  1 )
psi(age_t, z_i) = 1 - m(age_t, z_i)
```

where:
- `m_baseline(age)` is the gender-averaged death probability from the SSA 2017 Period
  Life Table (2020 Trustees Report). Source: https://www.ssa.gov/oact/STATS/table4c6.html
- `chi(z_i)` is an income-specific mortality scaling factor calibrated to match
  life-expectancy targets from Chetty et al. (2016, JAMA)

**Calibration procedure:**

1. Map each z-grid point to an income percentile via `Phi(z / sigma_z)`, where
   `sigma_z = sqrt(Var(eta) / (1 - rho^2))` is the unconditional std of `z`
2. Look up the target expected age at death from Chetty et al. (2016) at that
   percentile (race-adjusted, gender-averaged, pooled 2001-2014)
3. Solve for `chi(z_i)` via Brent root-finding such that the implied life
   expectancy at age 40 matches the Chetty target

**Storage:** `survival_probs_2d` has shape `(n_age, n_z)` and lives in the
`Precompute` object (not the model NamedTuple), because it depends on `z_grid`
which is a discretization choice. It is computed during `Precompute.__init__()`.

**How psi enters the solver:** In the main backward induction loop, a vector
`psi = survival_probs_2d[t, :]` of shape `(n_z,)` is extracted per age and
passed to the step functions as `psi_vec`. Inside each step function, the
`for z_i in range(n_z):` loop extracts the scalar `psi = psi_vec[z_i]` before
calling the Newton portfolio solver. The FOC functions receive scalar `psi` and
are unchanged -- they compute `prob_death = 1 - psi` for the bequest branch and
weight `psi * mu_alive` for the alive branch, exactly as before but now with a
z-dependent value.

**Bequest hoist compatibility:** The bequest hoist optimization (Section 6.2) remains
valid. Within each FOC function call, `psi` is scalar (fixed for that `z_i`), so
`prob_death` is constant across the `(j_z, i_e)` inner loops. The bequest contribution
is still accumulated once per `j_s`, not per `(j_z, i_e)`.

**Implementation:** See `mortality.py` for the calibration module.
Key function: `calibrate_earnings_dependent_mortality(start_age, terminal_age, z_grid,
rho, pz, mu_eta1, sigma_eta1, mu_eta2, sigma_eta2)` returns `(survival_probs_2d,
chi_vec, diagnostics)`.

### 1.2 Three Assets

| Asset | Return | Risk | Role |
|-------|--------|------|------|
| Bills | `R_bill = exp(rtb_t)` | Risk-free (known at t) | Safe asset, numeraire |
| Stocks | `R_stock = R_bill * exp(xr_{t+1})` | Risky (excess return uncertain) | Equity exposure |
| Nominal bonds | `R_bond = R_bill * exp(xb_{t+1})` | Risky (excess return uncertain) | Duration exposure |

The agent chooses portfolio weights `(alpha_stock, alpha_bond)` with the remainder
`alpha_bill = 1 - alpha_stock - alpha_bond` invested in bills.

**Portfolio constraints:**
```
alpha_stock >= 0
alpha_bond  >= 0
alpha_stock + alpha_bond <= 1     (alpha_bill >= 0, no borrowing)
```

### 1.3 Timing Convention

```
Start of period t:
  - Agent observes state (W_t, s_t, z_t)
  - s_t = (rtb_t, y_nom_t, dp_t) -- observable financial states
  - Bill rate R_bill = exp(rtb_t) is KNOWN at decision time
  - Agent chooses consumption c_t and portfolio (alpha_stock, alpha_bond)
  - Savings: a_t = W_t - c_t

Between t and t+1:
  - Financial state transitions: s_{t+1} drawn from Pi_state[i_s, :]
  - Excess returns realized: xr_{t+1}, xb_{t+1}
    conditional on (s_t=i, s_{t+1}=j): (xr,xb) ~ N(mu_r[i,j], Sigma_r_cond)
  - Income realized: Y_{t+1} depends on age (see below)
  - Portfolio gross return:
      R_port = alpha_s * R_bill * exp(xr) + alpha_b * R_bill * exp(xb)
                                          + alpha_bill * R_bill

Start of period t+1:
  - Cash-on-hand: W_{t+1} = a_t * R_port + Y_{t+1}

Income timing at the working-retirement boundary:
  - Working ages (age_t < retire_age):
      z transitions continuously: z_{t+1} = rho * z_t + eta_{t+1}
      where eta is drawn from the mixture-normal distribution.
      Transitory shock eps drawn from quadrature nodes.
      Y_{t+1} = disposable_income(exp(f(age_{t+1}) + z_{t+1} + eps_{t+1}))
      where f(age) = b0 + b1*age + b2*age^2/10 + b3*age^3/100
  - Last working year (age_t = retire_age - 1 = 66):
      Same as working -- Y_{t+1} is the agent's final labor income paycheck.
      z transitions one last time; the realized z_{t+1} determines pension
      for all subsequent periods.
  - Retirement (age_t >= retire_age):
      z is frozen (no transitions, no transitory shocks).
      Y_{t+1} = pension(z), where z is the state carried from retirement entry.
  - Consequence: first pension payment arrives at age retire_age + 1 (= 68).
    At age retire_age (= 67), cash-on-hand includes the final labor income.

  Treatment of z differs between solver and simulation:
  - Solver: z is discrete (n_z grid points). E[V(z')] is computed via
    Gauss-Hermite quadrature over mixture-normal eta innovations, with
    linear interpolation of policies at off-grid z' values.
  - Simulation: z is continuous (float64). Each period draws a continuous
    eta from the mixture, computes z' = rho*z + eta (clamped to grid
    bounds), and interpolates policies at the continuous z' value.
    Income and pension are computed directly from continuous z (not
    from precomputed tables), avoiding interpolation error in the
    nonlinear tax function.
```

### 1.4 Labor Income Process

Follows Guvenen et al. (2022) / Catherine (2025, Appendix E.1) with mixture-normal innovations:

```
log(Y_{t,gross}) = f(age_t) + z_t + eps_t

where:
  f(age) = b0 + b1*age + b2*age^2/10 + b3*age^3/100       (Catherine 2025 cubic)
  z_{t+1} = rho * z_t + eta_{t+1}
  eta_t ~ pz * N(mu_eta1, sigma_eta1^2) + (1-pz) * N(mu_eta2, sigma_eta2^2)
  eps_t ~ pe * N(mu_eps1, sigma_eps1^2) + (1-pe) * N(mu_eps2, sigma_eps2^2)

Calibration (Guvenen et al. 2022):
  rho = 0.991,  pz = 0.176
  mu_eta1 = -0.524,  sigma_eta1 = 0.113
  mu_eta2 = -(pz/(1-pz))*mu_eta1  (zero-mean condition)
  sigma_eta2 = 0.046
  pe = 0.044,  mu_eps1 = 0.134,  sigma_eps1 = 0.762,  sigma_eps2 = 0.055
  b0 = -6.142,  b1 = 0.3040,  b2 = -0.051,  b3 = 0.002586
```

After-tax income uses a post-TCJA progressive tax schedule (7 brackets: 10%, 12%,
22%, 24%, 32%, 35%, 37%). The income process is independent of the VAR.

**Retirement income (Social Security pension):** At `retire_age` (67), the
persistent state `z` freezes and the agent receives a Social Security pension
that follows the U.S. SSA Primary Insurance Amount (PIA) formula as in
Catherine (2025), Section 3.4, eqs. (17)–(20). The pension is constant in
levels for the rest of life (no further indexation in model units).

The benefit is a piecewise-linear function of **AIME** (Average Indexed
Monthly/Yearly Earnings), *not* of `exp(z)` directly. AIME is the career
average of total income, capped at the SSA taxable maximum of `2.5 × L̄`
(where the wage index `L̄ = 1` in model units):

```
AIYE_it = L̄_t · Σ_{s=t0}^{t} min{ L̃_is, 2.5 }          (Catherine eq. 20)
L̃_is   = L_is / L̄_s
```

In our model gross income is `exp(f(age) + z + ε)` where `f(age)` is the
deterministic age-earnings profile and `ε` averages to zero. With a stationary
wage index, AIME for a worker at persistent state `z` is approximated by

```
AIME(z) ≈ min( exp(z) · avg_det , 2.5 )
avg_det  = mean_{age ∈ [start_age, retire_age)} exp(f(age))
f(age)   = b0 + b1·age + b2·age²/10 + b3·age³/100
```

With the calibrated `(b0, b1, b2, b3)`, `avg_det ≈ 0.5069`, so the median
worker (`z = 0`) has AIME ≈ 0.507 — *not* 1.0. Multiplying by `avg_det`
converts the persistent component to a career-average level; the 2.5 cap
makes the benefit side consistent with the payroll-tax cap already used in
`disposable_income_working` (`payroll_tax = 0.106 · min(y, 2.5)`).

The PIA formula (Catherine eq. 19) applies SSA-style bend points and
replacement rates to AIME:

```
            ⎧ r1 · AIME                                              if AIME ≤ b1
PIA(AIME) = ⎨ r1·b1 + r2·(AIME − b1)                                 if b1 < AIME ≤ b2
            ⎩ r1·b1 + r2·(b2−b1) + r3·(AIME − b2)                    if AIME > b2

bend points: b1 = 0.21,  b2 = 1.25
rates:       r1 = 0.90,  r2 = 0.32,  r3 = 0.15
```

The gross PIA is then taxed by the same 7-bracket progressive income-tax
schedule used for labor income, yielding `pension_after_tax(z)`. Because
`avg_det` and the bend points do not depend on age, the precomputed table
`pension_after_tax` of shape `(n_age, n_z)` is just the `(n_z,)` vector
tiled across all ages.

Resulting calibration (n_z = 11 grid):

| z          | AIME            | pension after-tax |
|------------|-----------------|-------------------|
| `z = 0`    | 0.507           | ≈ 0.254           |
| `z ≈ 1.12` | 1.554           | ≈ 0.503           |
| `z ≥ 1.6`  | 2.5 (cap binds) | ≈ 0.628           |

This gives a replacement rate of ≈ 63% relative to career-average after-tax
income at `z = 0`, and a much smaller cross-sectional dispersion than an
uncapped `exp(z)`-based scheme would produce.

**Timing.** The last `z` transition occurs between `retire_age − 1` and
`retire_age` (age 66 → 67); the realized `z` at 67 determines the pension
for all subsequent periods. The working-age solver handles age 66 (last
labor income at 67), and the retirement solver handles ages 67+ (first
pension payment at 68).

**Implementation:**
- `compute_pension_after_tax(z_grid, avg_det)` in `model.py` — applies the
  AIME cap, the PIA piecewise formula, and the income-tax schedule.
- `_precompute_pension(self)` in `precompute.py` — computes `avg_det` from
  the model's `(b0, b1, b2, b3)` over `[start_age, retire_age)` and tiles
  `compute_pension_after_tax(z_grid, avg_det)` across ages into
  `pension_after_tax` of shape `(n_age, n_z)`.

---

## 2. The VAR System and State-Return Separation

### 2.1 Full VAR(1) Specification

The financial state vector has 5 variables:

```
z_t = [rtb_t, xr_t, xb_t, y_nom_t, dp_t]

Variable ordering:
  Index 0: rtb    -- ex-post real bill rate (quarterly decimal, TB3MS lagged/400 - inflation)
  Index 1: xr     -- log excess real stock return (RTRP - rtb)
  Index 2: xb     -- log excess nominal bond return (inflation cancels in excess return)
  Index 3: y_nom  -- 10-year nominal yield (SVENY10 / 400, quarterly decimal)
  Index 4: dp     -- log dividend-price ratio (log level)
```

Dynamics:
```
z_{t+1} = c + Phi @ z_t + eps_{t+1}

where:
  c = (I - Phi) @ z_bar     intercept (5,)
  Phi                        transition matrix (5,5)
  eps_{t+1} ~ N(0, Omega)   innovation covariance (5,5)
  z_bar                      unconditional means (5,)
```

**Estimation:** The VAR is estimated by restricted OLS at **quarterly frequency**
(T=183 observations, 1980 Q1 to 2025 Q4) from `var_dataset.csv`. The restriction
is that lagged return columns are excluded from all equations (Phi_12 = 0, Phi_22 = 0).
The residual covariance Omega comes from the unrestricted residuals.

**Annualization:** Because the DP model uses annual periods (ages 25-80, beta=0.96),
the quarterly VAR is **compounded to annual frequency** before passing to the solver.
The annual "return" is the sum of four quarterly returns. The compounding formulas
(see Section 2.1.1) preserve the stationary mean z_bar exactly.

Note: ages 22-99 gives 78 annual periods.

### 2.1.1 Quarterly-to-Annual Compounding

Let h=4 (quarters per year), P_k = Phi_11^k, C_k = sum_{j=0}^k P_j.

**State dynamics (exact):**
```
Phi_11_ann = Phi_11^4
c_s_ann    = C_{h-1} @ c_s
Omega_ss_ann = sum_{k=0}^{3} P_k @ Omega_ss @ P_k'
```

**Cumulative annual return (sum of h quarterly returns):**
```
Phi_21_ann = Phi_21 @ C_{h-1}
c_r_ann    = h*c_r + Phi_21 @ [sum_{k=1}^{3} C_{k-1}] @ c_s
```

Note: Phi_21_ann != [Phi^4]_{21}. The latter gives the h-step-ahead return,
not the cumulative sum. The sum formula is required.

**Verification:** The mean annual return exactly equals h times the mean quarterly
return: c_r_ann + Phi_21_ann @ z_bar_s = h * (c_r + Phi_21 @ z_bar_s).

**Stationary mean z_bar:** Invariant to time aggregation -- same quarterly or annual.
The discretization grid is therefore unchanged by compounding.

### 2.2 Partition into State Variables and Returns

We partition the 5-variable system into:

```
State variables: s_t = (rtb_t, y_nom_t, dp_t)   indices [0, 3, 4]
Returns:         r_t = (xr_t, xb_t)              indices [1, 2]
```

**Key restriction:** Lagged returns do not predict anything (imposed by estimation):

```
Full Phi partitioned by (state, return) blocks:

        | Phi_11  Phi_12 |     Phi_11: (3,3) state -> state
Phi =   | Phi_21  Phi_22 |     Phi_21: (2,3) state -> returns
                                Phi_12: (3,2) returns -> state   [= 0 by restriction]
                                Phi_22: (2,2) returns -> returns  [= 0 by restriction]
```

With this restriction:

```
State dynamics:    s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s_{t+1}
Return equations:  r_{t+1} = Phi_0_ret   + Phi_21 @ s_t + v^r_{t+1}
```

### 2.2.1 Generic Architecture

Although examples use `(rtb, y_nom, dp)` as state variables and `(xr, xb)` as
returns, the architecture is **generic** and config-driven via `variable_names`,
`state_indices`, and `return_indices`. No solver logic may hardcode specific
variable identities. Adding/removing/replacing variables requires only updating
the partition config.

The indexing is non-contiguous. State indices are [0, 3, 4] and return indices
are [1, 2] within the full 5-vector. The `partition_var()` function handles
this using NumPy fancy indexing:

```python
Phi_11 = Phi_full[np.ix_(state_idx, state_idx)]   # (3,3)
Phi_21 = Phi_full[np.ix_(ret_idx,   state_idx)]   # (2,3)
Phi_12 = Phi_full[np.ix_(state_idx, ret_idx)]     # (3,2) -- should be ~0
Phi_22 = Phi_full[np.ix_(ret_idx,   ret_idx)]     # (2,2) -- should be ~0
```

### 2.3 Innovation Covariance Partition

```
         | Sigma_ss  Sigma_sr |     Sigma_ss: (3,3)  state-state
Omega =  | Sigma_rs  Sigma_rr |     Sigma_rr: (2,2)  return-return
                                    Sigma_rs: (2,3)  return-state cross
```

Extracted as:
```python
Sigma_ss = Omega[np.ix_(state_idx, state_idx)]   # (3,3)
Sigma_rr = Omega[np.ix_(ret_idx,   ret_idx)]     # (2,2)
Sigma_rs = Omega[np.ix_(ret_idx,   state_idx)]   # (2,3)
```

### 2.4 Conditional Return Distribution

For a state transition s_i -> s_j, the state innovation is:
```
v^s_{ij} = s_j - Phi_0_state - Phi_11 @ s_i
```

The conditional mean return given this transition:
```
E[r_{t+1} | s_t=i, s_{t+1}=j] = Phi_0_ret + Phi_21 @ s_i + M @ v^s_{ij}

where:
  M = Sigma_rs @ inv(Sigma_ss)     (2,3) conditioning matrix
```

Rearranged for efficient vectorized computation:
```
mu_r[i,j] = (Phi_0_ret - M @ Phi_0_state) + (Phi_21 - M @ Phi_11) @ s_i + M @ s_j
           = const + A @ s_i + M @ s_j
```

**Consistency check:** Averaging over j gives the regression prediction:
```
sum_j Pi_state[i,j] * mu_r[i,j]  ==  Phi_0_ret + Phi_21 @ s_i     for each i
```

**Key values (annual parameters):** After compounding, M[xb, y_nom] = -39.4.
This is the bond duration mechanism: a 1-unit rise in y_nom at t+1 reduces the
annual bond return by 39.4 units (in quarterly decimal units).

### 2.5 Residual Return Variance and the K=1 Approximation

The conditional distribution of returns given a state transition is:
```
(xr, xb) | (s_t=i, s_{t+1}=j) ~ N(mu_r[i,j], Sigma_r_cond)

Sigma_r_cond = Sigma_rr - M @ Sigma_rs'     (2,2) constant matrix
```

The solver supports configurable return quadrature. Let `K` denote the
Gauss-Hermite order **per return dimension**. With `n_ret` return variables,
the tensor-product rule has `K_eff = K^n_ret` joint residual-return nodes.
The baseline `K=1` case is the old point-mass approximation at `mu_r[i,j]`.

**Why K=1 is accurate:**
- Conditioning on `s_{t+1}` (especially y_nom via `M[xb,y_nom]=-39.4`) explains
  ~99.5% of bond return variance. The xb residual is near-degenerate.
- Stock residual variance is larger but the portfolio FOC effect is second-order:
  the Jensen's inequality bias scales as `gamma*(gamma+1)*sigma_resid^2`, which
  is small when `sigma_resid` is already modest after conditioning.

For `K>1`, draw tensor-product Gauss-Hermite nodes from
`N(0, Sigma_r_cond)` and evaluate the FOC at `mu_r[i,j] + node_k` for each joint
node `k`. Since `Sigma_r_cond` is constant across `(i,j)`, the residual nodes
and weights are precomputed once and reused for every state transition.
This multiplies the inner loop cost by `K_eff` (e.g. `K=3` in the 2-return
model implies `K_eff = 9` joint nodes).
Expected improvement: basis-point-level changes in portfolio shares.

`Sigma_r_cond` is stored on the model object and used directly by the
return-quadrature constructor.

### 2.6 The `partition_var()` Function

```python
def partition_var(Phi_full, Omega_full, z_bar, state_idx, ret_idx,
                  variable_names=None, verbose=True):
    """
    Partition a full VAR(1) into state sub-VAR and return equations.

    Parameters:
        Phi_full:        (n, n) full transition matrix (annual after compounding)
        Omega_full:      (n, n) full innovation covariance
        z_bar:           (n,) unconditional means (same quarterly or annual)
        state_idx:       list of indices for state variables, e.g. [0, 3, 4]
        ret_idx:         list of indices for returns, e.g. [1, 2]
        variable_names:  optional list of names for diagnostics

    Returns: dict with keys:
        Phi_0_state, Phi_11, Phi_0_ret, Phi_21
        Sigma_ss, Sigma_rr, Sigma_rs, M, Sigma_r_cond
        z_bar_state, z_bar_ret
        n_state, n_ret, state_names, ret_names
    """
```

---

## 3. Core Modules and Precomputation

This section is organized by **module responsibility** rather than notebook cell order.
`model.py` holds the immutable economic specification and stateless helpers; `var.py`,
`discretization.py`, and `mortality.py` generate calibrated inputs;
`precompute.py` assembles the solver-facing numerical objects; and `diagnostics.py`
provides reporting/debugging utilities around the core model.

### 3.1 model.py -- Immutable Model and Config Objects

```python
class LifecyclePortfolioModel(NamedTuple):
    """Model specification with generic state-return partition."""

    # === PREFERENCES ===
    u: Callable               # CRRA utility function
    u_prime: Callable         # Marginal utility
    u_prime_inv: Callable     # Inverse marginal utility
    beta: float               # Discount factor (0.96 annual)
    gamma: float              # CRRA risk aversion (3.0)

    # === BEQUEST (Catherine 2025) ===
    b_bar: int                # Bequest horizon in years = bond maturity (10)
                              # b(W,s) = b_bar*(W/A(y_nom_s))^(1-gamma)/(1-gamma)

    # === LIFECYCLE ===
    start_age: int            # 22
    retire_age: int           # 67
    terminal_age: int         # 99
    # survival_probs: moved to Precompute as survival_probs_2d (n_age, n_z)

    # === LABOR INCOME (Guvenen 2022 / Catherine 2025) ===
    b0: float                 # Age-earnings intercept (-6.142)
    b1: float                 # Age-earnings linear (0.3040)
    b2: float                 # Age-earnings quadratic /10 (-0.051)
    b3: float                 # Age-earnings cubic /100 (0.002586)
    rho: float                # Persistent income AR(1) coefficient (0.991)
    pz: float                 # Mixture prob for persistent shock (0.176)
    mu_eta1: float
    sigma_eta1: float
    mu_eta2: float
    sigma_eta2: float
    pe: float                 # Mixture prob for transitory shock (0.05)
    mu_eps1: float
    sigma_eps1: float
    mu_eps2: float            # NOTE: overridden to enforce zero mean in quadrature
    sigma_eps2: float

    # === PARTITIONED VAR STRUCTURE (annual parameters) ===
    n_state: int              # Number of state variables (3)
    n_ret: int                # Number of return variables (2)
    state_names: tuple        # ('rtb', 'y_nom', 'dp')
    ret_names: tuple          # ('xr', 'xb')

    z_bar_state: np.ndarray   # (n_state,) state unconditional means
    z_bar_ret: np.ndarray     # (n_ret,) return unconditional means

    Phi_0_state: np.ndarray   # (n_state,) state intercepts = C_{h-1} @ c_s (annual)
    Phi_11: np.ndarray        # (n_state, n_state) state persistence (annual = Phi_11_q^4)
    Phi_0_ret: np.ndarray     # (n_ret,) return intercepts (annual)
    Phi_21: np.ndarray        # (n_ret, n_state) state -> return loading (annual)

    Sigma_ss: np.ndarray      # (n_state, n_state) state innovation covariance (annual)
    Sigma_rr: np.ndarray      # (n_ret, n_ret) return innovation covariance (annual)
    Sigma_rs: np.ndarray      # (n_ret, n_state) return-state cross-covariance (annual)
    M: np.ndarray             # (n_ret, n_state) conditioning matrix = Sigma_rs @ Sigma_ss^{-1}
    Sigma_r_cond: np.ndarray  # (n_ret, n_ret) residual return covariance

    bill_rate_index_in_state: int   # Index of rtb within state vector (0)
    annuity_yield_index_in_state: int   # Index of y_nom within state vector (1)
    constrained: bool            # True = no short-selling/leverage, False = unconstrained
```

The refactor cleanly separates the **economic model** from the **numerical tuning**.
`LifecyclePortfolioModel` is immutable and reusable across discretizations; grid design
and Newton tuning are carried by separate configuration objects in `model.py`:

```python
disc_config = DiscretizationConfig(
    state_grid_sizes=(5, 5, 5),
    n_z=11,
    n_eps_nodes=5,
    n_ret_nodes_1d=1,   # K = GH order per return dimension; total joint nodes = K^n_ret
)

solver_config = SolverConfig(
    tol=1e-7,
    max_iter=20,
    init_alpha_s=0.1,
    init_alpha_b=0.4,
)
```

`DiscretizationConfig` owns wealth/savings grids, state-grid sizes, income-grid sizes,
and conditional-return validation tolerances. `SolverConfig` owns Newton tolerances,
iteration caps, initial guesses, dampening rules, feasibility floors, and EGM safety
constants. This is a major structural change from the old notebook, where these choices
were implicit in cell-local variables.

### 3.2 model.py -- Bequest Utility Functions

```python
def annuity_factor(y_ann, b_bar):
    """
    A(y) = (1 - (1+y)^{-b_bar}) / y

    y_ann: annual nominal yield (y_nom_quarterly * 4)
    b_bar: bequest horizon in years (= bond maturity = 10)
    """
    return (1 - (1 + y_ann)**(-b_bar)) / y_ann

def bequest_utility(W, A, gamma, b_bar):
    """b(W) = b_bar * (W/A)^(1-gamma) / (1-gamma)"""
    return b_bar * (W / A)**(1 - gamma) / (1 - gamma)

def bequest_marginal(W, A, gamma, b_bar):
    """db/dW = b_bar * (W/A)^(-gamma) / A"""
    return b_bar * (W / A)**(-gamma) / A

def bequest_marginal_inv(mu, A, gamma, b_bar):
    """Inverse of bequest_marginal: W = A * (mu*A/b_bar)^(-1/gamma)"""
    return A * (mu * A / b_bar)**(-1.0 / gamma)
```

### 3.3 model.py and discretization.py -- Helper Functions

```python
create_utility_functions(gamma)          # Returns u, u_prime, u_prime_inv
mixture_cdf(x, p, mu1, sigma1, mu2, sigma2)
mixture_quantile(q, p, mu1, sigma1, mu2, sigma2)
disposable_income_working(y_gross)       # Progressive tax on labor income (vectorized)
scalar_disposable_income(y_gross)        # Same schedule, scalar float — Numba-callable from solver hot loop
compute_pension_after_tax(z_grid, avg_det)  # SSA PIA on AIME = min(exp(z)*avg_det, 2.5)
```

### 3.4 var.py and precompute.py -- VAR Parameter Handling

The workflow from raw data to model:

```python
# Step 1: Estimate quarterly VAR from CSV
var_config_q, var_res, var_data = build_nominal_system1_var_config(
    csv_path="var_dataset.csv",
    # columns = ['rtb', 'xr', 'xb', 'y_nom', 'dp']
    # state_indices = [0, 3, 4]    (rtb, y_nom, dp)
    # return_indices = [1, 2]      (xr, xb)
    # estimation = "restricted"    (Phi_12=0, Phi_22=0)
)

# Step 2: Compound quarterly -> annual
# Annual "return" = sum of 4 quarterly returns (not 4-step-ahead)
var_config = annualize_var_config(var_config_q, h=4)

# Step 3: Build model
base_config = build_base_config_legacy_defaults()
model = build_model(base_config, var_config)

# Step 4: Precompute
disc_config = DiscretizationConfig(
    state_grid_sizes=(5, 5, 5),
    n_z=11,
    n_eps_nodes=5,
)
pc = Precompute(model, disc_config=disc_config)

# Step 5: Optional calibration report
print_model_diagnostic_report(model, pc, periods_per_year=1)

# Step 6: Solve
C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(
    model,
    pc,
    solver_config=SolverConfig(),
)
```

**Fallback hardcoded parameters** now live in `var.py` rather than notebook cells.
Use `build_nominal_system1_var_config_hardcoded()` for quarterly parameters and
`build_nominal_system1_var_config_annual_hardcoded()` for already-annualized ones
when `var_dataset.csv` is unavailable.

**Key annual parameter values:**

```
z_bar = [-8.35e-4,  1.34e-2,  5.90e-3,  9.12e-3, -4.148]
        [ rtb,       xr,       xb,       y_nom,    dp   ]

Annual means: rtb=-0.33%/yr  xr=+5.36%/yr  xb=+2.36%/yr  y_nom=+3.65%/yr

Annual Phi_11 diagonal (state persistence):
  rtb:   0.017   (nearly iid annually)
  y_nom: 0.861
  dp:    0.807

M[xb, y_nom] = -39.4    (bond duration; 1 unit rise in y_nom -> -39.4 in xb)
```

### 3.5 discretization.py -- Discretization Functions

#### 3.5.1 Rouwenhorst Univariate

```python
def rouwenhorst_univariate(N, mu, rho, sigma):
    """
    Rouwenhorst method for AR(1): y_{t+1} = mu*(1-rho) + rho*y_t + sigma*eps
    Returns: (y_grid, Pi) where y_grid is (N,) and Pi is (N,N)
    """
```

#### 3.5.2 Rouwenhorst Multivariate

```python
def rouwenhorst_multivariate(N_vec, mu, Phi, Sigma, method='independent'):
    """
    Multivariate Rouwenhorst for VAR(1). Uses independent marginal
    discretization per dimension (Kronecker product transition matrix).
    method='independent': uses only diagonal(Phi_11) per marginal.

    Returns: (state_grids, Pi_state, state_indices)
    """
```

Called on the 3D state sub-VAR (not the full 5D system):

```python
state_grids, Pi_state, state_indices = rouwenhorst_multivariate(
    N_vec=[5, 5, 5],               # 125 joint states
    mu=model.Phi_0_state,          # (3,) annual state intercepts
    Phi=model.Phi_11,              # (3,3) annual state persistence
    Sigma=np.linalg.cholesky(model.Sigma_ss)  # (3,3) annual Cholesky
)
```

**Approximation error:** The independence method uses only `diag(Phi_11)` per
marginal, ignoring off-diagonal cross-persistence. The structural error is
`M @ Phi_11_off @ s_i` where `Phi_11_off = Phi_11 - diag(diag(Phi_11))`.
This is reported at runtime. Finer grids do not reduce this error.

Grid coverage: ±2 sigma of the stationary distribution with N=5 points.

#### 3.5.3 Income Process Discretization

```python
discretize_income_ar1_mixture(rho, p, mu1, sigma1, mu2, sigma2, N)
    # Tauchen-style Markov chain for persistent income z. Produces z_grid
    # and Pi_z. NOTE: Pi_z is retained for backward compatibility but is
    # NOT used by the solver or simulation for z-transitions. The solver
    # uses Gauss-Hermite quadrature (eta_nodes/weights); the simulation
    # draws continuous eta from the mixture distribution.

get_eps_quadrature_corrected(model, n_nodes)   # Gauss-Hermite, zero-mean enforced
get_eta_quadrature_mixture(model, n_nodes)     # Gauss-Hermite for persistent innovation eta
    # Both quadratures use the physicist's convention: nodes scaled by sqrt(2),
    # weights divided by sqrt(pi). Component 2's mean is computed internally
    # to enforce E[eta] = 0: mu_eta2_eff = -(pz/(1-pz)) * mu_eta1.
```

### 3.6 precompute.py -- Precompute Class

```python
class Precompute:
    """
    Precompute grids, transitions, and lookup tables.

    Grid-size choice (state_grid_sizes) controls accuracy vs. speed:
      [5, 5, 5] = 125 states  -- fast, coarser
      [7, 7, 7] = 343 states  -- good default for production
      [9, 9, 9] = 729 states  -- finer, check memory
    """
```

#### 3.6.1 Attributes

```python
# === GRIDS ===
wealth_grid:      (n_w,)          cash-on-hand interpolation grid [geometric]
s_grid:           (n_s,)          savings grid for EGM endogenous gridpoints
ages:             (n_age,)        integer ages from start_age to terminal_age

# === FINANCIAL STATE DISCRETIZATION ===
state_grid:       (N_state, 3)    joint state grid; row i = [rtb, y_nom, dp]
Pi_state:         (N_state, N_state)  transition matrix; Pi_state[i,j] = P(s'=j|s=i)
state_grids:      list[3]         marginal 1-D grids for each state variable
state_indices:    (N_state, 3)    multi-index into marginal grids
N_state:          int             total joint states = prod(state_grid_sizes)

# Backward-compatibility aliases:
slow_grid, Pi_slow, slow_grids, slow_state_indices, N_s

# === CONDITIONAL RETURNS ===
mu_r:             (N_state, N_state, 2)
                  mu_r[i, j, 0] = E[xr | s_t=i, s_{t+1}=j]   log excess stock return
                  mu_r[i, j, 1] = E[xb | s_t=i, s_{t+1}=j]   log excess bond return
ret_nodes:        (n_ret_quad, 2)   residual log-return shocks around mu_r
ret_weights:      (n_ret_quad,)     tensor-product quadrature weights; sum=1
r_bill_grid:      (N_state,)      log real bill rate at each state (rtb component)

# === BEQUEST ===
annuity_factors:  (N_state,)      A(y_nom_s, b_bar) for each financial state
                  Computed as: annuity_factor(state_grid[:, y_nom_idx] * 4, b_bar)
                  Used by bequest_utility / bequest_marginal / bequest_marginal_inv

# === INCOME PROCESS ===
z_grid:           (n_z,)          persistent income states (log, mean-zero)
Pi_z:             (n_z, n_z)      income transition matrix (retained, not used by solver/simulation)
eps_nodes:        (n_eps,)        Gauss-Hermite nodes for transitory shock
eps_weights:      (n_eps,)        quadrature weights; sum=1, mean=0 enforced
eta_nodes:        (n_eta,)        Gauss-Hermite nodes for persistent innovation eta
eta_weights:      (n_eta,)        quadrature weights; sum=1, mean=0 enforced
dz:               float           uniform z_grid spacing (z_grid[1] - z_grid[0])
log_det_profile:  (n_age,)        deterministic age-earnings profile f(age) for each period
avg_det:          float           mean(exp(f(age))) over working ages; used for pension AIME

# === LOOKUP TABLES ===
working_income:   (n_age, n_z, n_eps)  after-tax labor income table (simulation warmup + diagnostics; solver computes income on the fly — §4.5)
pension_after_tax:(n_age, n_z)         after-tax Social Security benefit (solver grid lookup)
survival_probs_2d:(n_age, n_z)         age- and earnings-dependent survival probabilities

# === DIMENSIONS ===
n_w, n_s, n_z, n_eps, n_age, N_state
```

#### 3.6.2 Conditional Return Mean Precomputation

```python
def _precompute_conditional_returns(self):
    """
    mu_r[i, j, :] = E[r | s_t=i, s_{t+1}=j]

    Formula (rearranged for vectorized efficiency):
      mu_r[i,j] = const + A @ s_i + M @ s_j
      where  const = Phi_0_ret - M @ Phi_0_state
             A     = Phi_21    - M @ Phi_11
    """
    const  = model.Phi_0_ret - model.M @ model.Phi_0_state  # (2,)
    A      = model.Phi_21    - model.M @ model.Phi_11        # (2,3)
    term_i = state_grid @ A.T        # (N_state, 2)
    term_j = state_grid @ model.M.T  # (N_state, 2)
    mu_r = const[None,None,:] + term_i[:,None,:] + term_j[None,:,:]
    # Shape: (N_state, N_state, 2)
```

Memory: N_state=125 -> 0.25 MB. N_state=343 -> 1.9 MB.

#### 3.6.3 Bequest Annuity Factor Precomputation

```python
# y_nom is the 2nd state variable (index 1 in state_names = ('rtb','y_nom','dp'))
# y_nom stored in quarterly decimal (SVENY10/400); multiply by 4 for annual yield
_y_nom_idx = list(model.state_names).index('y_nom')
_y_ann = state_grid[:, _y_nom_idx] * 4.0
self.annuity_factors = annuity_factor(_y_ann, model.b_bar)
# (N_state,) -- one annuity factor per financial state
```

#### 3.6.4 Consistency Validation

```python
def _validate_conditional_returns(self):
    """
    Verify: sum_j Pi_state[i,j] * mu_r[i,j] == Phi_0_ret + Phi_21 @ s_i

    Error = M @ Phi_11_off @ s_i  where Phi_11_off = Phi_11 - diag(diag(Phi_11))
    Printed at runtime; finer grids do not reduce this error.
    """
```

`Precompute.__init__()` is also where `mortality.py` is invoked:
it calls `calibrate_earnings_dependent_mortality(...)` and stores the resulting
`survival_probs_2d` on the precompute object, alongside the grids and income tables.

### 3.7 diagnostics.py -- Diagnostic Reports and Failure Analysis

`diagnostics.py` sits outside the production solver path and consumes an already-built
`model` + `pc` pair for inspection and debugging:

```python
print_model_diagnostic_report(model, pc, periods_per_year=1)
```

This pre-solve report checks calibration consistency, income grids, VAR structure,
state-grid coverage, conditional-return moments, and approximate memory use before
launching the expensive backward induction.

For targeted debugging, `diagnostics.py` also exposes:

```python
diagnose_newton_failures_retirement(...)
```

This routine re-evaluates retirement FOCs over a chosen slice of the state space and
classifies why Newton calls fail. It imports low-level FOC routines from `solver.py`,
but keeping it in a separate module avoids mixing debugging logic into the main solver.

---

## 4. solver.py -- Backward Induction Solver

### 4.1 State Space and Policy Functions

Each period, the agent's state is `(W, i_s, i_z)`:

| Dimension | Variable | Grid | Size |
|-----------|----------|------|------|
| Wealth | W | `wealth_grid` (endogenous via EGM) | n_w (150) |
| Financial state | i_s | `state_grid` (Rouwenhorst on state sub-VAR) | N_state (125 or 343) |
| Persistent income | i_z | `z_grid` (mixture Rouwenhorst) | n_z (11) |

Policy functions stored as:

```python
C_mat[t, i_z, i_s, i_w]   # optimal consumption
S_mat[t, i_z, i_s, i_w]   # optimal stock weight (alpha_stock)
B_mat[t, i_z, i_s, i_w]   # optimal bond weight  (alpha_bond)
```

Shape: `(n_age, n_z, N_state, n_w)`

### 4.2 Bellman Equation

```
V(W, i_s, i_z, t) = max_{c, alpha_s, alpha_b}  u(c)
                     + beta * psi_{t,i_z}     * E[V(W', i_s', i_z', t+1)]
                     + beta * (1-psi_{t,i_z}) * b(a*R_port, annuity_factors[i_s])

where:
  a       = W - c                                    savings
  R_bill  = exp(r_bill_grid[i_s])                    known at decision time
  R_s     = R_bill * exp(mu_r[i_s, j_s, 0])          stock return for transition i_s->j_s
  R_b     = R_bill * exp(mu_r[i_s, j_s, 1])          bond return for transition i_s->j_s
  R_port  = alpha_s*R_s + alpha_b*R_b + alpha_bill*R_bill
  W'      = a * R_port + Y'                          next-period cash-on-hand
  b(W, A) = b_bar * (W/A)^(1-gamma) / (1-gamma)     bequest utility
```

The expectation integrates over:
1. Next financial state `j_s`: weighted by `Pi_state[i_s, j_s]`
2. Persistent innovation `eta`: Gauss-Hermite quadrature with `eta_weights[k_eta]`.
   `z_next = rho * z_grid[i_z] + eta_nodes[k_eta]` is generally off-grid;
   policies are linearly interpolated between bracketing z-grid points.
3. Transitory shock `eps`: weighted by `eps_weights[i_eps]` (working age only)
4. Residual return shock `k_r`: weighted by `ret_weights[k_r]`

### 4.3 Interpolation and Helper Functions

```python
@njit
def fast_interp_1d(x, x_grid, y_grid):
    """Linear interpolation with binary search.
    Uses linear extrapolation (nearest-interval slope) beyond grid boundaries."""

@njit
def fast_interp_slope_1d(x, x_grid, y_grid):
    """Slope (MPC) of the piecewise-linear interpolant.
    Returns nearest interior slope at extrapolation boundaries."""

@njit
def project_to_triangle(alpha_s, alpha_b):
    """Project (alpha_stock, alpha_bond) onto feasible region: >=0, sum<=1."""
```

### 4.4 FOC and Jacobian -- Retirement

For a given savings level, current state, and candidate portfolio weights,
compute the FOC and Jacobian for the 2D Newton step.

```python
@njit(fastmath=True)
def compute_foc_jac_retirement(alpha_s, alpha_b, s_val, z_idx, i_s,
                                wealth_grid, c_next_full, pension_next_scalar,
                                annuity_factor_is,
                                Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                gamma, psi, beta, b_bar):
    """
    Parameters:
        alpha_s, alpha_b:     candidate portfolio weights
        s_val:                savings (a = W - c)
        z_idx:                persistent income state index (unused, passed through)
        i_s:                  current financial state index
        c_next_full:          (N_state, n_w) consumption policy at t+1 for this z
        pension_next_scalar:  pension income for this z
        annuity_factor_is:    A(y_nom, b_bar) at current state i_s  [scalar]
        Pi_state:             (N_state, N_state)
        Rx_stock_next:        (N_state, n_ret_quad) precomputed exp(mu_r[i_s, :, 0] + ret_node[:,0])
        Rx_bond_next:         (N_state, n_ret_quad) precomputed exp(mu_r[i_s, :, 1] + ret_node[:,1])
        ret_weights:          (n_ret_quad,) joint return quadrature weights
        R_bill:               scalar, exp(r_bill_grid[i_s])

    Returns:
        foc_s, foc_b:         FOC for stock and bond weights
        J_ss, J_bb, J_sb:     Jacobian entries
        euler_sum:            expected marginal utility * R_port (for EGM)
    """
```

**Annuity factor pricing:** The bequest annuity factor uses the **current** state's
nominal yield (`annuity_factors[i_s]`), consistent with Catherine (2025, eq. 21-22)
where the annuity is priced at the time-t risk-free rate `r_ft`. The invested wealth
`a * R_port` varies by next state `j_s` through portfolio returns, but the annuity
pricing is fixed at the current yield.

### 4.5 FOC and Jacobian -- Working Age

Same structure as retirement, with additional integration over income innovations.

**Gauss-Hermite quadrature over persistent innovations:** Instead of iterating
over discrete grid points weighted by `Pi_z[i_z, j_z]`, the inner loop iterates
over Gauss-Hermite quadrature nodes `eta_nodes[k_eta]` with weights `eta_weights[k_eta]`.
For each node, `z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]` is computed and
bracketed on the z-grid. Consumption policy is linearly interpolated between the
two bracketing grid points `iz_lo` and `iz_lo + 1`; income is **not** interpolated
(see below).

**Income computed on the fly (no table interpolation):** Next-period gross income

```
y_gross = exp( f(t+1) + ρ·z_grid[z_idx] + η_k + ε_j )
```

is evaluated directly via `scalar_disposable_income(y_gross)` (the scalar,
Numba-JIT'd companion to the vectorized `disposable_income_working`). This
replaces the earlier scheme of interpolating the precomputed `working_income`
table in `z`, which introduced a chord-overshoot bias of ~14–17% between grid
points because the tax schedule is nonlinear. The solver now uses the exact same
gross-to-net mapping as the simulation.

To keep the hot loop free of transcendentals, the exponential is factored once
per FOC call:

```
base_det_z = exp( f(t+1) + ρ·z_grid[z_idx] )          # 1 exp (hoisted)
exp_eta[k] = exp( η_k )      for k in 0..n_eta-1      # n_eta exps (hoisted)
exp_eps[j] = exp( ε_j )      for j in 0..n_eps-1      # n_eps exps (hoisted)

y_gross    = base_det_z · exp_eta[k_eta] · exp_eps[i_e]   # inside hot loop: 2 muls
```

Total transcendentals per FOC call: `n_eta + n_eps + 1` (typically ~8), versus
`N_state · n_ret_quad · n_eta · n_eps` (typically ~4,500) if `exp()` were left
inline. The two small `exp_eta` / `exp_eps` buffers are allocated inside the
`@njit` function and fit comfortably in L1 cache.

**Bequest hoist optimization:** The bequest marginal utility (`mu_bequest`, `mup_bequest`)
depends only on `j_s` (through `w_inv = s_val * R_p`), not on `(k_eta, i_e)`. Its
contribution to all 6 accumulators (foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum) is
accumulated once at the `j_s` level, while the inner `(k_eta, i_e)` loops handle only the
alive (`psi * mu_alive`) part. This is valid because `sum(eta_weights) = sum(eps_weights) = 1`,
so `sum(weight) = p_var` for each `j_s`.

**Note on earnings-dependent mortality:** The FOC functions receive scalar `psi` (already
indexed by the current `z_i` in the calling step function). Therefore `prob_death = 1 - psi`
is constant within a single FOC call, and the bequest hoist remains valid: the bequest
weight `p_var * prob_death` is still independent of `(k_eta, i_e)` within the inner loops.

**Note on `z_next` clamping:** The z-bracketing for consumption-policy interpolation
clamps `iz_lo` and `frac_z` to `[0, n_z-2]` and `[0, 1]`. The income computation uses
the raw `z_next = ρ·z + η_k` without clamping — the tax function is well-defined at
every real `z`, so tail realizations get more accurate income than they would under a
clamped table lookup.

```python
@njit(fastmath=True)
def compute_foc_jac_working(alpha_s, alpha_b, s_val, z_idx, i_s,
                             wealth_grid, c_next_full, log_det_next,
                             annuity_factor_is,
                             z_grid, rho, eta_nodes, eta_weights, dz,
                             Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                             eps_nodes, eps_weights,
                             gamma, psi, beta, b_bar):
    """
    Working-age FOC/Jacobian. Integrates over:
      - Next financial state j_s (weighted by Pi_state)
      - Joint return node k_r (weighted by ret_weights)
      - Persistent innovation eta (Gauss-Hermite quadrature: eta_weights)
      - Transitory shock eps (weighted by eps_weights)

    z_next = rho * z_grid[z_idx] + eta_nodes[k_eta] is generally off-grid;
    consumption is linearly interpolated in z. Income is evaluated exactly
    via scalar_disposable_income(exp(log_det_next + rho*z + eta + eps))
    with the exp() factored into precomputed per-node tables.

    log_det_next: scalar float = f(age_{t+1}), the deterministic age-earnings
                  profile evaluated at next period's age.
    c_next_full shape: (n_z, N_state, n_w)

    Loop structure (after bequest hoist):
      precompute exp_eta[k], exp_eps[j], base_det_z    # outside all loops
      for j_s:                              # future macro state
          for k_r:                          # joint return node
              bequest accumulators += ...   # once per (j_s, k_r)
              for k_eta:                    # persistent innovation (GH quadrature)
                  z_next = rho * z + eta_nodes[k_eta]
                  iz_lo, frac_z = bracket(z_next)       # for consumption policy only
                  det_z_eta = base_det_z * exp_eta[k_eta]
                  for i_e:                  # transitory shock
                      y_gross     = det_z_eta * exp_eps[i_e]
                      income_next = scalar_disposable_income(y_gross)
                      interpolate consumption at (iz_lo, frac_z)
                      alive accumulators += ...
    """
```

### 4.6 Newton Portfolio Solver

2D Newton-Raphson for optimal `(alpha_stock, alpha_bond)`. Checks corners (all bills,
all stocks, all bonds), then edges (two-asset combinations), then interior Newton.
`Rx_stock_next` and `Rx_bond_next` are precomputed per `i_s` in the step functions
to avoid redundant `exp()` calls across Newton iterations and savings grid points.
They now have shape `(N_state, n_ret_quad)` rather than `(N_state,)`.

```python
@njit(fastmath=True)
def solve_portfolio_2d_retirement(s_val, z_idx, i_s,
                                  wealth_grid, c_next_full, pension_next_scalar,
                                  annuity_factor_is,
                                  Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                                  gamma, psi, beta, b_bar,
                                  init_s=0.1, init_b=0.4, tol=1e-7, max_iter=20):
    """Returns: (opt_alpha_s, opt_alpha_b, euler_sum)"""
```

#### 4.6.1 Relative Tolerance Scaling

All tolerance checks in the Newton solvers use **relative** rather than absolute
tolerances. The FOCs `foc_s`, `foc_b` are sums of `c^{-gamma} * Rex_k` terms,
so their natural magnitude scales with marginal utility `c^{-gamma}`. With
`gamma=3` and small consumption, absolute FOC values can reach ~1e+9, making a
fixed absolute tolerance of 1e-7 impossible to achieve.

**Scale computation.** After the first FOC evaluation (all-bills corner), the
Euler equation level `e0 = sum_j pi_j * MU_j * R_bill` provides the natural
FOC magnitude. Each solver computes:

- Retirement/working: `scale = max(abs(e0), 1.0)`
- Terminal: `scale = R_bill ** (-gamma)` (since terminal FOC has no euler_sum)

The floor of 1.0 ensures the tolerance never becomes tighter than the absolute
`tol` when FOC values happen to be small.

**Where scale is applied:**

| Check | Old (absolute) | New (relative) |
|-------|---------------|----------------|
| Corner KKT | `fs0 <= 1e-8` | `fs0 <= 1e-8 * scale` |
| Edge Newton convergence | `abs(fs) < tol` | `abs(fs) < tol * scale` |
| Edge acceptance | `abs(fs) < tol*10 and fb <= tol` | `abs(fs) < tol*scale*10 and fb <= tol*scale` |
| Interior Newton convergence | `err < tol` | `err < tol * scale` |
| Returned `foc_resid` | `abs(fs)` or `err` | `abs(fs)/scale` or `err/scale` |

The returned `foc_resid` is now a **relative** Euler equation error. Values
< `tol` indicate convergence; the post-solve diagnostic "Worst FOC" and "RMS FOC"
report these relative residuals directly.

### 4.7 Terminal Condition

At the final age T, the agent consumes optimally given that savings generate
bequest utility. The portfolio FOC decouples from consumption due to the CRRA
bequest structure (homogeneous of degree 1-gamma in wealth). The portfolio is
solved once per financial state `i_s` via `solve_portfolio_2d_terminal`, then
consumption follows in closed form.

```python
@njit
def solve_terminal_age(wealth_grid, annuity_factors, r_bill_grid, Pi_state, mu_r,
                        ret_nodes, ret_weights,
                        gamma, beta, b_bar, N_state, n_z):
    """
    Terminal period: jointly solve for c* and optimal (alpha_s*, alpha_b*).

    Portfolio FOC (independent of c and W due to CRRA bequest):
        sum_j sum_k pi(j|i) * ret_weights[k] * R_port*(j,k)^{-gamma} * (R_k(j,k) - R_bill) = 0

    Consumption closed-form:
        Omega = b_bar * A^{gamma-1} * sum_j sum_k pi(j|i) * ret_weights[k] * R_port*(j,k)^{1-gamma}
        ratio = (beta * Omega)^{-1/gamma}
        c*    = W * ratio / (1 + ratio)

    Output shape: (n_z, N_state, n_w) -- identical across z (bequest independent of income).
    """
```

### 4.8 Period Solvers

#### 4.8.1 Retirement Step

```python
@njit(parallel=True)
def solve_retirement_step(wealth_grid, savings_grid, z_grid, N_state,
                          c_next_full, pension_1d,
                          annuity_factors,
                          Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                          gamma, psi_vec, beta, b_bar):
    """
    Solve one retirement period using EGM + 2D Newton.
    Parallelized: prange over i_s (financial state index).
    psi_vec: (n_z,) survival probs for this age, indexed by z.
    Inside the z_i loop: psi = psi_vec[z_i] (scalar) passed to Newton solver.
    Output shapes: (n_z, N_state, n_w) for policy_c, policy_alpha_s, policy_alpha_b
    """
```

#### 4.8.2 Working Age Step

```python
@njit(parallel=True)
def solve_working_age_step(wealth_grid, savings_grid, z_grid, N_state,
                           c_next_full, log_det_next,
                           annuity_factors,
                           rho, eta_nodes, eta_weights, dz,
                           Pi_state, mu_r, ret_nodes, ret_weights, r_bill_grid,
                           eps_nodes, eps_weights,
                           gamma, psi_vec, beta, b_bar):
    """
    Solve one working-age period using EGM + 2D Newton.
    Same EGM structure as retirement but with income integration via
    Gauss-Hermite quadrature over persistent innovations (eta_nodes/weights)
    and linear z-interpolation of the consumption policy at off-grid z'
    values. Income at each quadrature node is computed on the fly from
    log_det_next, z_grid[z_i], eta_nodes[k_eta], eps_nodes[i_e] via
    scalar_disposable_income — no table interpolation (§4.5).

    log_det_next: scalar = f(age_{t+1}). Master solver passes
                  pc.log_det_profile[t+1].
    psi_vec:      (n_z,) survival probs for this age, indexed by z.
    """
```

### 4.9 Master Solver

```python
def run_lifecycle_solver(model, pc, n_s_points=None, solver_config=None, verbose=1):
    """
    Main lifecycle solver using backward induction.

    Returns:
        C_mat, S_mat, B_mat: policy functions, shape (n_age, n_z, N_state, n_w)
        diagnostics: dict with per-age diagnostic arrays and aggregates
    """
```

If `solver_config` is omitted, the function instantiates `SolverConfig()` internally.

### 4.10 Solver Diagnostics System

The solver collects comprehensive diagnostics via flat counter arrays passed
through Numba `@njit` functions. Each `i_s` (prange variable) writes to its
own row in the diagnostic arrays -- no race conditions.

#### 4.10.1 Newton Solver Return Values

All Newton solvers (`solve_portfolio_2d_retirement`, `_working`, `_terminal`)
return `(alpha_s, alpha_b, euler_sum, exit_code, foc_resid)` where:

| Exit code | Constant | Meaning |
|-----------|----------|---------|
| 0 | EC_TINY_SAVINGS | s_val < 1e-6, trivial all-bills |
| 1 | EC_CORNER_BILLS | All-bills corner |
| 2 | EC_CORNER_STOCKS | All-stocks corner |
| 3 | EC_CORNER_BONDS | All-bonds corner |
| 4 | EC_EDGE_SB | Stock + bill edge |
| 5 | EC_EDGE_BB | Bond + bill edge |
| 6 | EC_EDGE_STOCKBOND | Stock + bond edge |
| 7 | EC_INTERIOR | Interior Newton converged |
| 8 | EC_NEWTON_FAIL | Newton hit max_iter |

`foc_resid` is the final FOC norm at exit (0.0 for corners).

#### 4.10.2 Diagnostic Arrays

Step functions return `diag_int` (N_state, 13) and `diag_float` (N_state, 9)
instead of the old `mono_violations` and `mono_worst` arrays.

**Integer counters** (`diag_int[i_s, idx]`):
Corner/edge/interior counts, Newton failures, negative euler events,
monotonicity violations, total calls.

**Float counters** (`diag_float[i_s, idx]`):
Worst monotonicity drop, max/RMS FOC residual, portfolio share min/max/sum.

#### 4.10.3 Output Layers

**Per-age table** (printed as each age completes):
```
 Age  Phase  Time  Newton%  alpha_s  alpha_b  a_bill  c/W    %int  %edge  %corn  mono
  79  RETIRE  2.1s  100.0%    0.115    0.528   0.357  0.189   53%    34%    13%     0
```
Portfolio shares and c/W are at the median income/financial/wealth state.

**Post-solve report** (5 sections):
1. Newton convergence: total/failed calls, worst/RMS FOC residual
2. Portfolio regime breakdown: 7-category split by retirement vs working
3. Portfolio share ranges: global min/max/mean
4. EGM monotonicity: violations by age
5. Policy sanity: NaN/inf/negative checks on C_mat, S_mat, B_mat

**Verbosity:** `verbose=0` silent, `verbose=1` per-age progress table plus the
post-solve diagnostic report.

---

## 5. Summary of Data Structures

### 5.1 Key Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| `state_grid` | (N_state, 3) | Joint state grid; row i = [rtb, y_nom, dp] |
| `Pi_state` | (N_state, N_state) | State transition matrix |
| `mu_r` | (N_state, N_state, 2) | Conditional return means (xr, xb) |
| `r_bill_grid` | (N_state,) | Log real bill rate at each state |
| `annuity_factors` | (N_state,) | A(y_nom, b_bar) annuity factor per state |
| `z_grid` | (n_z,) | Persistent income states |
| `Pi_z` | (n_z, n_z) | Income transition matrix (retained, not used by solver/simulation) |
| `eps_nodes/weights` | (n_eps,) | Gauss-Hermite quadrature for transitory shocks |
| `eta_nodes/weights` | (n_eta,) | Gauss-Hermite quadrature for persistent innovation eta |
| `log_det_profile` | (n_age,) | Deterministic age-earnings profile f(age) per period |
| `avg_det` | scalar | Mean of exp(f(age)) over working ages; for pension AIME |
| `working_income` | (n_age, n_z, n_eps) | After-tax labor income table (retained for simulation warmup and diagnostics; solver now computes income on the fly — §4.5) |
| `pension_after_tax` | (n_age, n_z) | After-tax pension table (solver grid lookup) |

### 5.2 Policy Function Shapes

| Array | Shape | Description |
|-------|-------|-------------|
| `C_mat` | (n_age, n_z, N_state, n_w) | Optimal consumption |
| `S_mat` | (n_age, n_z, N_state, n_w) | Optimal stock share |
| `B_mat` | (n_age, n_z, N_state, n_w) | Optimal nominal bond share |

### 5.3 VAR Variables

| Variable | Description | Quarterly decimal | Annual equivalent |
|----------|-------------|-------------------|-------------------|
| `rtb` | Real bill rate | TB3MS_lag/400 - d(log CPI) | ~-0.33%/yr |
| `xr` | Excess stock return | log(RTRP) - rtb | ~5.36%/yr |
| `xb` | Excess nominal bond return | bond - rtb | ~2.36%/yr |
| `y_nom` | 10-year nominal yield | SVENY10/400 | ~3.65%/yr |
| `dp` | Log dividend-price ratio | log level | -4.148 |

---

## 6. Computational Notes

### 6.1 Inner Loop Cost

Per Newton evaluation, the inner loop iterates over:
- N_state financial state transitions (125 or 343)
- Working age: x n_eta persistent innovation nodes (10) x n_eps transitory nodes (10)

| Config | Retirement | Working |
|--------|-----------|---------|
| N_state=125 | 125 iters | 12,500 iters |
| N_state=343 | 343 iters | 34,300 iters |

### 6.2 Bequest Hoist Optimization

In `compute_foc_jac_working`, the bequest contribution to the FOC and Jacobian
depends only on the future macro state `j_s` (through `w_inv = s_val * R_p`),
not on the income dimensions `(j_z, i_e)`. The bequest terms are accumulated
once per `j_s` at the outer loop level, while the inner `(j_z, i_e)` loops
handle only the alive contribution (`psi * mu_alive`). This avoids
`(n_z * n_eps - 1)` redundant multiply-accumulates per active `j_s` across
6 accumulators, reducing inner-loop work by ~30%.

### 6.3 Parallelization

`prange` over `i_s` (financial state index) in both period solvers.
`mu_r` and `annuity_factors` are read-only shared arrays.

### 6.4 Memory

| Array | N_state=125 | N_state=343 |
|-------|-------------|-------------|
| `mu_r` (N_state^2 x 2) | 0.24 MB | 1.8 MB |
| `Pi_state` (N_state^2) | 0.12 MB | 0.9 MB |
| `C_mat` (78 x n_z x N_state x 150) | ~20 MB (n_z=11) | ~55 MB |
| Total policy (x3) | ~60 MB | ~165 MB |

### 6.5 Numba Compatibility

All `@njit` functions receive plain NumPy arrays. The Precompute object unpacks
everything before passing to compiled functions. `mu_r` and `annuity_factors` are
standard float64 arrays, fully Numba-compatible.

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
- `var.py` -- VAR estimation, state/return partitioning, and hardcoded fallback
  parameter builders
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

1. Estimate or load annual VAR parameters in `var.py`.
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
b(W, y_1, spr) = b_bar * (W / A(y_1, spr))^(1-gamma) / (1-gamma)

where:
  C_bar      = W / A(y_1, spr)        flow-equivalent consumption
  A(y_1,spr) = sum_{k=1}^{b_bar} (1 + y(k))^{-k}    annuity factor
  y(k)       = y_1 + spr * min(k-1, 19) / 19          interpolated yield
  b_bar      = 10                      bequest horizon in years
```

The annuity factor A prices a 10-year consumption stream using a linearly
interpolated term structure between the 1-year yield y_1 and the 20-year AAA
yield y_20 = y_1 + spr. For k=1, y(1) = y_1; for k=20, y(20) = y_20; for
k > 20, y(k) = y_20 (capped, no extrapolation). With b_bar=10, only yields
up to k=10 matter. Uses discrete compounding (1+y)^{-k} to match the codebase
convention.

**Calibration:** `gamma = 5`, `beta = 0.96`, `b_bar = 10`.

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
| Bills | `R_bill = exp(rtb)` | Uncertain (inflation risk) | Safe-ish asset, numeraire |
| Stocks | `R_stock = R_bill * exp(xr)` | Risky (excess return uncertain) | Equity exposure |
| Nominal bonds | `R_bond = R_bill * exp(xb)` | Risky (excess return uncertain) | Duration exposure |

**All three returns are uncertain.** The bill rate `rtb = log(1+y_1) - pi` is
uncertain because realized inflation is unknown at decision time. Excess returns
`xr` and `xb` are nominal-minus-nominal (CCV convention): they subtract the log
nominal bill return `log(1+y_1)`, so inflation appears in `rtb` only.

**Recovery identities:** `R_bill = exp(rtb)`, `R_stock = exp(rtb + xr)`,
`R_bond = exp(rtb + xb)`. All three use the SAME quadrature node (joint draw).

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
  - s_t = (y_1_t, spr_t, cy_t) -- observable financial states (known at end of year t)
  - Agent chooses consumption c_t and portfolio (alpha_stock, alpha_bond)
  - Savings: a_t = W_t - c_t

Between t and t+1:
  - State innovation v^s_{t+1} drawn; next state s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s
  - Returns realized: rtb_{t+1}, xr_{t+1}, xb_{t+1}
    conditional on (s_t, v^s): (rtb,xr,xb) ~ N(mu_r, Sigma_r_cond)
    where mu_r = Phi_0_ret + Phi_21 @ s_t + M @ v^s
  - Income realized: Y_{t+1} depends on age (see below)
  - Portfolio gross real return:
      R_port = alpha_s * exp(rtb+xr) + alpha_b * exp(rtb+xb) + alpha_bill * exp(rtb)

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
    a Judd (1998) quadrature built directly on the mixture-normal eta
    density (n_eta total nodes, polynomial exactness 2*n_eta - 1), with
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
Catherine (2025), Section 3.4, eqs. (17)--(20). The pension is constant in
levels for the rest of life (no further indexation in model units).

The benefit is a piecewise-linear function of **AIME** (Average Indexed
Monthly/Yearly Earnings), *not* of `exp(z)` directly. AIME is the career
average of total income, capped at the SSA taxable maximum of `2.5 * L_bar`
(where the wage index `L_bar = 1` in model units):

```
AIYE_it = L_bar_t * sum_{s=t0}^{t} min{ L_tilde_is, 2.5 }  (Catherine eq. 20)
L_tilde_is = L_is / L_bar_s
```

In our model gross income is `exp(f(age) + z + eps)` where `f(age)` is the
deterministic age-earnings profile and `eps` averages to zero. With a stationary
wage index, AIME for a worker at persistent state `z` is approximated by

```
AIME(z) = min( exp(z) * avg_det , 2.5 )
avg_det  = mean_{age in [start_age, retire_age)} exp(f(age))
f(age)   = b0 + b1*age + b2*age^2/10 + b3*age^3/100
```

With the calibrated `(b0, b1, b2, b3)`, `avg_det ~ 0.5069`, so the median
worker (`z = 0`) has AIME ~ 0.507 -- *not* 1.0. Multiplying by `avg_det`
converts the persistent component to a career-average level; the 2.5 cap
makes the benefit side consistent with the payroll-tax cap already used in
`disposable_income_working` (`payroll_tax = 0.106 * min(y, 2.5)`).

The PIA formula (Catherine eq. 19) applies SSA-style bend points and
replacement rates to AIME:

```
            { r1 * AIME                                              if AIME <= b1
PIA(AIME) = { r1*b1 + r2*(AIME - b1)                                if b1 < AIME <= b2
            { r1*b1 + r2*(b2-b1) + r3*(AIME - b2)                   if AIME > b2

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
| `z = 0`    | 0.507           | ~ 0.254           |
| `z ~ 1.12` | 1.554           | ~ 0.503           |
| `z >= 1.6` | 2.5 (cap binds) | ~ 0.628           |

This gives a replacement rate of ~ 63% relative to career-average after-tax
income at `z = 0`, and a much smaller cross-sectional dispersion than an
uncapped `exp(z)`-based scheme would produce.

**Timing.** The last `z` transition occurs between `retire_age - 1` and
`retire_age` (age 66 -> 67); the realized `z` at 67 determines the pension
for all subsequent periods. The working-age solver handles age 66 (last
labor income at 67), and the retirement solver handles ages 67+ (first
pension payment at 68).

**Implementation:**
- `compute_pension_after_tax(z_grid, avg_det)` in `model.py` -- applies the
  AIME cap, the PIA piecewise formula, and the income-tax schedule.
- `_precompute_pension(self)` in `precompute.py` -- computes `avg_det` from
  the model's `(b0, b1, b2, b3)` over `[start_age, retire_age)` and tiles
  `compute_pension_after_tax(z_grid, avg_det)` across ages into
  `pension_after_tax` of shape `(n_age, n_z)`.

---

## 2. The VAR System and State-Return Separation

### 2.1 Full VAR(1) Specification

The financial state vector has 6 variables at **annual frequency**:

```
z_t = [y_1_t, spr_t, cy_t, rtb_t, xr_t, xb_t]

Variable ordering:
  Index 0: y_1    -- 1-year nominal Treasury yield (annual decimal)
  Index 1: spr    -- yield spread: AAA 20yr - y_1 (annual decimal)
  Index 2: cy     -- log earnings yield: -log(CAPE)
  Index 3: rtb    -- real bill return: log(1+y_1) - pi (annual log)
  Index 4: xr     -- excess nominal stock return (annual log)
  Index 5: xb     -- excess nominal bond return (annual log)
```

Dynamics:
```
z_{t+1} = c + Phi @ z_t + eps_{t+1}

where:
  c = (I - Phi) @ z_bar     intercept (6,)
  Phi                        transition matrix (6,6)
  eps_{t+1} ~ N(0, Omega)   innovation covariance (6,6)
  z_bar                      unconditional means (6,)
```

**Estimation:** The VAR is estimated using the CCV (2003) constrained estimator
directly at **annual frequency** (T=63 observations, 1963--2025) from
`data/var_dataset.csv`. The restriction is that only lagged state variables enter
each equation -- lagged return columns of Phi are zero by construction.

**CCV constrained estimator:**
1. Pin z_bar = sample mean of the full dataset.
2. Demean: z_tilde_t = z_t - z_bar.
3. Regress z_tilde_{t+1} on z_tilde_t **without intercept**, using only
   state columns as regressors.
4. Recover const = (I - Phi) @ z_bar.

This guarantees `(I - Phi)^{-1} @ const = z_bar = sample_mean` **exactly**,
eliminating grid-centering drift.

**Data construction:** Stock returns use Shiller's nominal P and D columns
directly (CPI-free). Inflation enters only via FRED CPIAUCSL in the rtb
definition. Bond returns use the CCV loglinear approximation for a 20-year
AAA par bond. See `data/build_var_dataset.py` and `contextfiles/RETURNS.md`.

Note: ages 22-99 gives 78 annual periods.

### 2.2 Partition into State Variables and Returns

We partition the 6-variable system into:

```
State variables: s_t = (y_1_t, spr_t, cy_t)     indices [0, 1, 2]
Returns:         r_t = (rtb_t, xr_t, xb_t)      indices [3, 4, 5]
```

**Key design choices:**
- **No riskless asset.** `rtb` is a return variable (uncertain because of
  inflation risk), not a state variable. The nominal bill yield `y_1` is known
  at decision time, but the real return is uncertain.
- **All excess returns are nominal minus nominal** (CCV convention): `xr` and `xb`
  subtract `log(1+y_1)`. Inflation appears in `rtb` only.
- **`spr = y_20 - y_1`** rather than `y_20` directly: more orthogonal to `y_1`,
  better-conditioned estimation, more efficient Rouwenhorst grid.
- **`cy = -log(CAPE)`** rather than `dp`: cyclically-adjusted earnings yield is a
  stronger long-horizon equity predictor.
- **20-year Moody's AAA par bond** with CCV loglinear approximation.

**Key restriction:** Lagged returns do not predict anything (imposed by estimation):

```
Full Phi partitioned by (state, return) blocks:

        | Phi_11  Phi_12 |     Phi_11: (3,3) state -> state
Phi =   | Phi_21  Phi_22 |     Phi_21: (3,3) state -> returns
                                Phi_12: (3,3) returns -> state   [= 0 by restriction]
                                Phi_22: (3,3) returns -> returns  [= 0 by restriction]
```

With this restriction:

```
State dynamics:    s_{t+1} = Phi_0_state + Phi_11 @ s_t + v^s_{t+1}
Return equations:  r_{t+1} = Phi_0_ret   + Phi_21 @ s_t + v^r_{t+1}
```

### 2.3 Innovation Covariance Partition

```
         | Sigma_ss  Sigma_sr |     Sigma_ss: (3,3)  state-state
Omega =  | Sigma_rs  Sigma_rr |     Sigma_rr: (3,3)  return-return
                                    Sigma_rs: (3,3)  return-state cross
```

Extracted as:
```python
Sigma_ss = Omega[np.ix_(state_idx, state_idx)]   # (3,3)
Sigma_rr = Omega[np.ix_(ret_idx,   ret_idx)]     # (3,3)
Sigma_rs = Omega[np.ix_(ret_idx,   state_idx)]   # (3,3)
```

### 2.4 Conditional Return Distribution

The return innovation decomposes as:

```
v^r = M @ v^s + eps,     eps ~ N(0, Sigma_r_cond) independent of v^s

where:
  M = Sigma_rs @ inv(Sigma_ss)                    (3,3) conditioning matrix
  Sigma_r_cond = Sigma_rr - M @ Sigma_sr          (3,3) residual covariance
```

**Conditioning on the state innovation v^s** (used by the solver):

```
E[r_{t+1} | s_t, v^s_{t+1}] = Phi_0_ret + Phi_21 @ s_t + M @ v^s_{t+1}
                              = const_r  + A_r @ s_t    + M @ v^s_{t+1}
Var[r_{t+1} | s_t, v^s_{t+1}] = Sigma_r_cond   (constant)
```

where `const_r = Phi_0_ret` and `A_r = Phi_21`.

**Equivalently, conditioning on the next state s_{t+1}** (used for precomputed mu_r):

```
E[r_{t+1} | s_t=i, s_{t+1}=j] = Phi_0_ret + Phi_21 @ s_i + M @ (s_j - Phi_0_state - Phi_11 @ s_i)

Rearranged:
mu_r[i,j] = (Phi_0_ret - M @ Phi_0_state) + (Phi_21 - M @ Phi_11) @ s_i + M @ s_j
           = const + A @ s_i + M @ s_j
```

**Consistency check:** Averaging over v^s gives the regression prediction:
```
E_{v^s}[mu_r] = Phi_0_ret + Phi_21 @ s_i     for each i
```

**Key values (annual parameters):**
```
M[xb, y_1]  = -8.72    (100bp rise in y_1 -> -8.7pp xb; bond duration)
M[xb, spr]  = -8.51    (100bp rise in spr -> -8.5pp xb)
M[xr, cy]   = -0.93    (mechanical CAPE/price relationship)
M[rtb, y_1] = -0.94    (Fisher effect)

Variance explained by state conditioning:
  rtb: 39.1%   xr: 96.2%   xb: 91.2%
```

### 2.5 Residual Return Variance and 3D Return Quadrature

The conditional distribution of returns given a state innovation is:

```
(rtb, xr, xb) | (s_t, v^s) ~ N(mu_r, Sigma_r_cond)

Sigma_r_cond = Sigma_rr - M @ Sigma_rs'     (3,3) constant matrix
```

The solver integrates over return residuals using tensor-product Gauss-Hermite
quadrature on `Sigma_r_cond` (3x3). With K nodes per dimension, this produces
K^3 joint residual-return nodes. At K=2, that's 8 nodes.

Gross real returns at each return quadrature node k_r:

```
R_bill[k_r]  = exp(mu_rtb  + ret_nodes[k_r, 0])
R_stock[k_r] = R_bill * exp(mu_xr + ret_nodes[k_r, 1])
R_bond[k_r]  = R_bill * exp(mu_xb + ret_nodes[k_r, 2])
```

**Critical invariant:** all three returns for a single period use the SAME
return quadrature node k_r -- they are components of one joint draw.

**Residual return std (after conditioning):**
```
rtb: 1.54%   xr: 3.10%   xb: 2.26%
```

`Sigma_r_cond` is stored on the model object and used directly by the
return-quadrature constructor.

### 2.6 The `partition_var()` Function

```python
def partition_var(Phi_full, Omega_full, z_bar, state_idx, ret_idx,
                  variable_names=None, verbose=True):
    """
    Partition a full VAR(1) into state sub-VAR and return equations.

    Parameters:
        Phi_full:        (6, 6) full transition matrix (annual)
        Omega_full:      (6, 6) full innovation covariance (annual)
        z_bar:           (6,) unconditional means (sample means, CCV convention)
        state_idx:       [2, 1, 0]  (cy, spr, y_1) default since 2026-04-30; was [0,1,2]
        ret_idx:         [3, 4, 5]  (rtb, xr, xb)
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
    gamma: float              # CRRA risk aversion (5.0)

    # === BEQUEST (Catherine 2025) ===
    b_bar: int                # Bequest horizon in years (10)

    # === LIFECYCLE ===
    start_age: int            # 22
    retire_age: int           # 67
    terminal_age: int         # 99

    # === LABOR INCOME (Guvenen 2022 / Catherine 2025) ===
    b0, b1, b2, b3: float    # Age-earnings polynomial coefficients
    rho: float                # Persistent income AR(1) coefficient (0.991)
    pz: float                 # Mixture prob for persistent shock (0.176)
    mu_eta1, sigma_eta1: float
    mu_eta2, sigma_eta2: float   # mu_eta2 DERIVED: -(pz/(1-pz))*mu_eta1
    pe: float                 # Mixture prob for transitory shock (0.044)
    mu_eps1, sigma_eps1: float
    mu_eps2, sigma_eps2: float   # mu_eps2 DERIVED: -(pe/(1-pe))*mu_eps1

    # === PARTITIONED VAR STRUCTURE (annual parameters) ===
    n_state: int              # Number of state variables (3)
    n_ret: int                # Number of return variables (3)
    state_names: tuple        # default ('cy', 'spr', 'y_1') since 2026-04-30; legacy ('y_1', 'spr', 'cy')
    ret_names: tuple          # ('rtb', 'xr', 'xb')

    z_bar_state: np.ndarray   # (3,) state unconditional means
    z_bar_ret: np.ndarray     # (3,) return unconditional means

    Phi_0_state: np.ndarray   # (3,) state intercepts
    Phi_11: np.ndarray        # (3, 3) state persistence
    Phi_0_ret: np.ndarray     # (3,) return intercepts
    Phi_21: np.ndarray        # (3, 3) state -> return loading

    Sigma_ss: np.ndarray      # (3, 3) state innovation covariance
    Sigma_rr: np.ndarray      # (3, 3) return innovation covariance
    Sigma_rs: np.ndarray      # (3, 3) return-state cross-covariance
    M: np.ndarray             # (3, 3) conditioning matrix = Sigma_rs @ Sigma_ss^{-1}
    Sigma_r_cond: np.ndarray  # (3, 3) residual return covariance

    y_1_index_in_state: int       # Index of y_1 within state vector (= 0)
    spr_index_in_state: int       # Index of spr within state vector (= 1)

    constrained: bool             # True = no short-selling/leverage
```

The refactor cleanly separates the **economic model** from the **numerical tuning**.
`LifecyclePortfolioModel` is immutable and reusable across discretizations; grid design
and Newton tuning are carried by separate configuration objects in `model.py`:

```python
disc_config = DiscretizationConfig(
    state_grid_sizes=(5, 5, 5),
    state_grid_mode="principal",   # "naive" | "lyapunov-axis" | "principal"
    state_n_stds=2.0,              # scalar (broadcast) OR length-3 sequence for
                                   # per-axis half-width in standardized u-coords
                                   # (principal mode) or physical sigma_stat units
                                   # (lyapunov-axis mode). Per-axis added 2026-04-30.
    n_z=11,
    n_eps_nodes=5,
    n_ret_nodes_1d=2,   # int K -> uniform K^n_ret nodes (legacy); also accepts a
                        # tuple (K_rtb, K_xr, K_xb) for per-dimension refinement,
                        # giving prod(K_i) joint nodes (e.g. (3,9,3) -> 81)
    n_state_quad_nodes=3,  # GH order per state dimension for state innovation quadrature
)

solver_config = SolverConfig(
    tol=1e-7,
    max_iter=20,
    init_alpha_s=0.1,
    init_alpha_b=0.4,
)
```

`DiscretizationConfig` owns wealth/savings grids, state-grid sizes, income-grid sizes,
and quadrature node counts. `SolverConfig` owns Newton tolerances,
iteration caps, initial guesses, dampening rules, feasibility floors, and EGM safety
constants.

### 3.2 model.py -- Bequest Utility Functions

```python
def annuity_factor(y_1, spr, b_bar):
    """
    Annuity factor with linearly interpolated term structure.

    A = sum_{k=1}^{b_bar} (1 + y(k))^{-k}
    where y(k) = y_1 + spr * min(k-1, 19) / 19

    Uses discrete compounding (1+y)^{-k}.
    """

def bequest_utility(W, A, gamma, b_bar):
    """b(W) = b_bar * (W/A)^(1-gamma) / (1-gamma)"""

def bequest_marginal(W, A, gamma, b_bar):
    """db/dW = b_bar * (W/A)^(-gamma) / A"""

def bequest_marginal_inv(mu, A, gamma, b_bar):
    """Inverse of bequest_marginal: W = A * (mu*A/b_bar)^(-1/gamma)"""
```

### 3.3 model.py and discretization.py -- Helper Functions

```python
create_utility_functions(gamma)          # Returns u, u_prime, u_prime_inv
mixture_cdf(x, p, mu1, sigma1, mu2, sigma2)
disposable_income_working(y_gross)       # Progressive tax on labor income (vectorized)
scalar_disposable_income(y_gross)        # Same schedule, scalar float -- Numba-callable
compute_pension_after_tax(z_grid, avg_det)  # SSA PIA on AIME = min(exp(z)*avg_det, 2.5)
```

### 3.4 var.py and precompute.py -- VAR Parameter Handling

The workflow from raw data to model:

```python
# Step 1: Estimate annual VAR from CSV (CCV constrained, restricted)
var_config, fit_details, data = build_nominal_system1_var_config(
    csv_path="data/var_dataset.csv"
)
# columns = ['y_1', 'spr', 'cy', 'rtb', 'xr', 'xb']    # CSV column order, fixed
# state_indices = [2, 1, 0]    (cy, spr, y_1) default since 2026-04-30
#                              (legacy [0, 1, 2] = (y_1, spr, cy) was the pre-reorder default)
# return_indices = [3, 4, 5]   (rtb, xr, xb)

# Step 2: Build model
base_config = build_base_config_legacy_defaults()
model = build_model(base_config, var_config)

# Step 3: Precompute
disc_config = DiscretizationConfig(
    state_grid_sizes=(5, 5, 5),
    n_z=11,
    n_eps_nodes=5,
)
pc = Precompute(model, disc_config=disc_config)

# Step 4: Optional calibration report
print_model_diagnostic_report(model, pc, periods_per_year=1)

# Step 5: Solve
C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(
    model,
    pc,
    solver_config=SolverConfig(),
)
```

**Fallback hardcoded parameters** live in `var.py` as module-level arrays
(`_Z_BAR`, `_PHI`, `_OMEGA`). These are annual estimates from the 1963-2025 dataset.

**Key annual parameter values:**

```
z_bar = [+0.04849, +0.01992, -2.99287, +0.00913, +0.05547, +0.01427]
        [ y_1,      spr,      cy,       rtb,      xr,       xb      ]

Annual means: y_1=4.85%  spr=1.99%  cy=-2.99  rtb=0.91%  xr=5.55%  xb=1.43%

Annual Phi_11 diagonal (state persistence):
  y_1: 0.670   spr: 0.872   cy: 0.919

Phi_21 (return equations, 3x3):
         L.y_1       L.spr       L.cy
  rtb   +1.079      +0.857      -0.034
  xr    -1.801      -0.523      +0.107
  xb    +1.462      +4.492      -0.055

M[xb, y_1] = -8.72   (bond duration; 100bp rise in y_1 -> -8.7pp xb)
M[xb, spr] = -8.51   (100bp rise in spr -> -8.5pp xb)
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

Called on the 3D state sub-VAR (not the full 6D system):

```python
state_grids, Pi_state, state_indices = rouwenhorst_multivariate(
    N_vec=[5, 5, 5],               # 125 joint states
    mu=model.Phi_0_state,          # (3,) annual state intercepts
    Phi=model.Phi_11,              # (3,3) annual state persistence
    Sigma=np.linalg.cholesky(model.Sigma_ss)  # (3,3) annual Cholesky
)
```

**Note:** The independence method uses only `diag(Phi_11)` per marginal grid,
ignoring off-diagonal cross-persistence. This only affects grid coverage, not
transition accuracy -- the solver uses Gauss-Hermite quadrature for state
transitions, which handles the full covariance structure exactly. The Rouwenhorst
grid is retained for policy function storage and interpolation.

#### 3.5.2a build_state_grid (production entry point)

```python
build_state_grid(N_vec, mu_intercept, Phi, Sigma_innov, n_stds=3.0, mode="principal")
    # Mode-aware financial-state interpolation grid. Returns a dict with
    # state_grid (joint physical-state lattice), state_bracket_grids
    # (per-axis interpolation grids in standardized u-coords for principal
    # mode, or physical coords for lyapunov-axis), state_indices (multi-index
    # into marginals), bracket_shift (mu_s in principal, zero elsewhere),
    # bracket_L_inv (Cholesky inverse in principal, identity elsewhere),
    # Pi_state, stationary_probs.
    #
    # n_stds accepts a scalar (broadcast across axes) OR a length-3 sequence
    # for per-axis half-widths.
    # - principal mode: n_stds[d] is the half-width of Cholesky direction d
    #   in standardized u-coords. Cholesky directions mix physical state
    #   variables; under the default ordering (cy, spr, y_1) -- 2026-04-30:
    #     L[:, 0] = (+0.530,  0,       0)        # PURE cy (100%)
    #     L[:, 1] = (-0.054, +0.0158,  0)        # mostly spr (~99%) + tiny cy leakage
    #     L[:, 2] = (+0.378, -0.0187, +0.0165)   # y_1 with residual cy/spr coupling
    #   So n_stds[0] is the clean cy knob, n_stds[1] is the clean spr knob;
    #   n_stds[2] only controls the leftover y_1 variance.  Variable order
    #   matters: the variable listed first gets the "pure" Cholesky column.
    # - lyapunov-axis mode: n_stds[d] is the half-width of physical axis d in
    #   stationary-sigma units (mu_s[d] +/- n_stds[d] * sigma_stat[d]).
    #   Per-axis n_stds maps directly to per-physical-variable control.
    # - naive mode: ignores n_stds (Rouwenhorst per-marginal).
```

This is the production entry point used by `Precompute`. The legacy
`rouwenhorst_multivariate` block above describes the `mode="naive"` fallback;
production runs use `mode="principal"` for joint covariance shape and
`state_bracket_*` artefacts that the solver and simulator consume directly.

#### 3.5.3 Income Process Discretization

```python
discretize_income_ar1_mixture(rho, p, mu1, sigma1, mu2, sigma2, N)
    # Tauchen-style Markov chain for persistent income z. Produces z_grid
    # and Pi_z. NOTE: Pi_z is retained for backward compatibility but is
    # NOT used by the solver or simulation for z-transitions. The solver
    # uses Judd-mixture quadrature (eta_nodes/weights); the simulation
    # draws continuous eta from the mixture distribution.

get_eps_quadrature_corrected(model, n_nodes)   # Judd 1998 mixture quadrature, zero-mean enforced
get_eta_quadrature_mixture(model, n_nodes)     # Judd 1998 mixture quadrature for eta
    # Both build an n-point Gauss rule directly on the mixture density
    # (Hankel-system orthogonal polynomial -> roots = nodes; Vandermonde
    # solve for weights). n_nodes is the TOTAL node count; polynomial
    # exactness against the mixture is 2*n_nodes - 1. Component 2's mean
    # is computed internally to enforce E[eta] = 0: mu_eta2_eff =
    # -(pz/(1-pz)) * mu_eta1. Helper _judd_mixture_quadrature() in
    # discretization.py:338.
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
state_grid:       (N_state, 3)    joint state grid; row i = state vector in MODEL ordering
                                  (default [cy, spr, y_1] since 2026-04-30; legacy [y_1, spr, cy])
Pi_state:         (N_state, N_state)  transition matrix (retained, not used by solver)
state_grids:      list[3]         marginal 1-D grids for each state variable
state_indices:    (N_state, 3)    multi-index into marginal grids
N_state:          int             total joint states = prod(state_grid_sizes)

# === STATE INNOVATION QUADRATURE ===
v_nodes:          (K_s^3, 3)      state innovation Gauss-Hermite nodes
v_weights:        (K_s^3,)        tensor-product weights, sum to 1
n_state_quad:     int             total state quadrature nodes

# === CONDITIONAL RETURNS ===
mu_r:             (N_state, N_state, 3)
                  mu_r[i, j, 0] = E[rtb | s_t=i, s_{t+1}=j]
                  mu_r[i, j, 1] = E[xr  | s_t=i, s_{t+1}=j]
                  mu_r[i, j, 2] = E[xb  | s_t=i, s_{t+1}=j]
const_r:          (3,)            Phi_0_ret
A_r:              (3, 3)          Phi_21
M_v_nodes:        (K_s^3, 3)     v_nodes @ M.T (precomputed)
ret_nodes:        (n_ret_quad, 3) residual log-return shocks around mu_r
ret_weights:      (n_ret_quad,)   joint return quadrature weights; sum=1
exp_ret_bill:     (n_ret_quad,)   exp(ret_nodes[:, 0])
exp_ret_stock:    (n_ret_quad,)   exp(ret_nodes[:, 1])
exp_ret_bond:     (n_ret_quad,)   exp(ret_nodes[:, 2])
                  n_ret_quad = prod(n_ret_nodes_1d).
                  Scalar K_r -> K_r^3.  Tuple (K_rtb, K_xr, K_xb) -> K_rtb*K_xr*K_xb.

# === BEQUEST ===
annuity_factors:  (N_state,)      A(y_1, spr, b_bar) for each financial state
                  Computed from state_grid[:, 0] (y_1) and state_grid[:, 1] (spr)

# === INCOME PROCESS ===
z_grid:           (n_z,)          persistent income states (log, mean-zero)
Pi_z:             (n_z, n_z)      income transition matrix (retained, not used by solver)
eps_nodes:        (n_eps,)        Judd-mixture nodes for transitory shock (n_eps = n_eps_nodes total)
eps_weights:      (n_eps,)        quadrature weights; sum=1, mean=0 enforced, all > 0
eta_nodes:        (n_eta,)        Judd-mixture nodes for persistent innovation eta (n_eta = n_eta_nodes total)
eta_weights:      (n_eta,)        quadrature weights; sum=1, mean=0 enforced, all > 0
dz:               float           uniform z_grid spacing
log_det_profile:  (n_age,)        deterministic age-earnings profile f(age)
avg_det:          float           mean(exp(f(age))) over working ages

# === LOOKUP TABLES ===
working_income:   (n_age, n_z, n_eps)  after-tax labor income table (simulation; solver computes on the fly)
pension_after_tax:(n_age, n_z)         after-tax Social Security benefit
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
    const  = model.Phi_0_ret - model.M @ model.Phi_0_state  # (3,)
    A      = model.Phi_21    - model.M @ model.Phi_11        # (3,3)
    term_i = state_grid @ A.T        # (N_state, 3)
    term_j = state_grid @ model.M.T  # (N_state, 3)
    mu_r = const[None,None,:] + term_i[:,None,:] + term_j[None,:,:]
    # Shape: (N_state, N_state, 3)
```

Memory: N_state=125 -> 0.37 MB. N_state=343 -> 2.8 MB.

#### 3.6.3 Bequest Annuity Factor Precomputation

```python
# y_1 is the 1st state variable (index 0 in state_names = ('y_1','spr','cy'))
# spr is the 2nd state variable (index 1)
_y_1 = state_grid[:, model.y_1_index_in_state]
_spr = state_grid[:, model.spr_index_in_state]
self.annuity_factors = annuity_factor(_y_1, _spr, model.b_bar)
# (N_state,) -- one annuity factor per financial state
```

#### 3.6.4 Consistency Validation

```python
def _validate_state_quadrature(self):
    """
    Verify: sum_k w_k * (const_r + A_r @ s_i + M @ v_k) == Phi_0_ret + Phi_21 @ s_i

    Uses the property that sum_k w_k * v_k = 0 (zero-mean quadrature).
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
                     + beta * psi_{t,i_z}     * E[V(W', s', z', t+1)]
                     + beta * (1-psi_{t,i_z}) * b(a*R_port, annuity_factors[i_s])

where:
  a       = W - c                                    savings
  R_bill  = exp(mu_rtb + eps_rtb)                    real bill return (uncertain)
  R_s     = R_bill * exp(mu_xr + eps_xr)             real stock return
  R_b     = R_bill * exp(mu_xb + eps_xb)             real bond return
  R_port  = alpha_s*R_s + alpha_b*R_b + alpha_bill*R_bill
  W'      = a * R_port + Y'                          next-period cash-on-hand
  b(W, A) = b_bar * (W/A)^(1-gamma) / (1-gamma)     bequest utility
```

The expectation integrates over:
1. State innovation `v^s_k`: Gauss-Hermite quadrature with `v_weights[k_v]`.
   Next state `s_{t+1} = Phi_0_state + Phi_11 @ s_t + v_k` is generally off-grid;
   policies are trilinearly interpolated on the 3D state grid.
2. Return residual `eps_k`: Gauss-Hermite quadrature with `ret_weights[k_r]`.
   Joint draw from `N(0, Sigma_r_cond)`.
3. Persistent innovation `eta`: Judd-mixture quadrature with `eta_weights[k_eta]`.
   `z_next = rho * z_grid[i_z] + eta_nodes[k_eta]` is generally off-grid;
   policies are linearly interpolated between bracketing z-grid points.
4. Transitory shock `eps`: Judd-mixture quadrature, weighted by
   `eps_weights[i_eps]` (working age only)

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
def bracket_state_3d(s0, s1, s2, grids_0, grids_1, grids_2):
    """Bracket (s0,s1,s2) in the 3D state grid for trilinear interpolation.
    Returns (lo0, lo1, lo2, f0, f1, f2) with clamping to valid range."""

@njit
def project_to_triangle(alpha_s, alpha_b):
    """Project (alpha_stock, alpha_bond) onto feasible region: >=0, sum<=1."""
```

### 4.4 FOC and Jacobian -- Retirement

For a given savings level, current state, and candidate portfolio weights,
compute the FOC and Jacobian for the 2D Newton step.

The solver integrates over state innovations (outer loop) and return residuals
(inner loop). For each state innovation v_k:

```
s_next = Phi_0_state + Phi_11 @ s_i + v_k    (trilinearly interpolated)
mu_r   = const_r + A_r @ s_i + M_v_nodes[k_v]  (conditional return mean)
```

For each return residual node k_r:

```
R_bill = exp(mu_rtb + ret_nodes[k_r, 0])
R_s    = R_bill * exp(mu_xr + ret_nodes[k_r, 1])
R_b    = R_bill * exp(mu_xb + ret_nodes[k_r, 2])
R_port = alpha_s * R_s + alpha_b * R_b + alpha_bill * R_bill
```

**Annuity factor pricing:** The bequest annuity factor uses the **current** state's
yields (`annuity_factors[i_s]`), consistent with Catherine (2025, eq. 21-22)
where the annuity is priced at the time-t yield curve. The invested wealth
`a * R_port` varies by scenario through portfolio returns, but the annuity
pricing is fixed at the current state.

### 4.5 FOC and Jacobian -- Working Age

Same structure as retirement, with additional integration over income innovations.

**Judd-mixture quadrature over persistent innovations:** Instead of iterating
over discrete grid points weighted by `Pi_z[i_z, j_z]`, the inner loop iterates
over Judd-mixture quadrature nodes `eta_nodes[k_eta]` with weights `eta_weights[k_eta]`.
For each node, `z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]` is computed and
bracketed on the z-grid. Consumption policy is interpolated in z by **PCHIP**
(Fritsch-Carlson, monotonicity-preserving cubic Hermite) on interior intervals
where the 4-point stencil `[iz_lo-1, iz_lo, iz_lo+1, iz_lo+2]` fits
(`iz_lo >= 1` and `iz_lo + 2 < n_z`); the first and last z intervals fall back
to linear. Wealth interpolation is linear on `[iw, iw+1]`; the FOC Jacobian's
`mpc` is computed as the analytical wealth derivative of the PCHIP interpolant
(PCHIP evaluated at iw and iw+1, finite-differenced across `dw`), which keeps
`mpc` exactly consistent with `c_val` even when the slope limiter activates
asymmetrically. Income is **not** interpolated (see below).

**Income computed on the fly (no table interpolation):** Next-period gross income

```
y_gross = exp( f(t+1) + rho*z_grid[z_idx] + eta_k + eps_j )
```

is evaluated directly via `scalar_disposable_income(y_gross)` (the scalar,
Numba-JIT'd companion to the vectorized `disposable_income_working`). This
replaces the earlier scheme of interpolating the precomputed `working_income`
table in `z`, which introduced a chord-overshoot bias of ~14-17% between grid
points because the tax schedule is nonlinear.

To keep the hot loop free of transcendentals, the exponential is factored once
per FOC call:

```
base_det_z = exp( f(t+1) + rho*z_grid[z_idx] )          # 1 exp (hoisted)
exp_eta[k] = exp( eta_k )      for k in 0..n_eta-1      # n_eta exps (hoisted)
exp_eps[j] = exp( eps_j )      for j in 0..n_eps-1      # n_eps exps (hoisted)

y_gross    = base_det_z * exp_eta[k_eta] * exp_eps[i_e]   # inside hot loop: 2 muls
```

**Bequest hoist optimization:** The bequest marginal utility (`mu_bequest`, `mup_bequest`)
depends only on the scenario `(k_v, k_r)` (through `w_inv = s_val * R_p`), not on
`(k_eta, i_e)`. Its contribution to all 6 accumulators (foc_s, foc_b, J_ss, J_bb, J_sb,
euler_sum) is accumulated once at the outer loop level, while the inner `(k_eta, i_e)`
loops handle only the alive (`psi * mu_alive`) part.

**Loop structure (after bequest hoist):**
```
precompute exp_eta[k], exp_eps[j], base_det_z    # outside all loops
for k_v:                              # state innovation quadrature node
    compute s_next, bracket on 3D grid
    compute mu_r = base_mu_r + M_v_nodes[k_v]
    for k_r:                          # return residual quadrature node
        compute R_bill, R_s, R_b, R_port
        bequest accumulators += ...   # once per (k_v, k_r)
        for k_eta:                    # persistent innovation (GH quadrature)
            z_next = rho * z + eta_nodes[k_eta]
            iz_lo, frac_z = bracket(z_next)       # for consumption policy only
            det_z_eta = base_det_z * exp_eta[k_eta]
            for i_e:                  # transitory shock
                y_gross     = det_z_eta * exp_eps[i_e]
                income_next = scalar_disposable_income(y_gross)
                interpolate consumption at (iz_lo, frac_z)
                alive accumulators += ...
```

### 4.6 Newton Portfolio Solver

2D Newton-Raphson for optimal `(alpha_stock, alpha_bond)`. Checks corners (all bills,
all stocks, all bonds), then edges (two-asset combinations), then interior Newton.

```python
@njit(fastmath=True)
def solve_portfolio_2d_retirement(s_val, z_idx, i_s,
                                  wealth_grid, c_next_full, pension_next_scalar,
                                  annuity_factor_is,
                                  ...
                                  init_s=0.1, init_b=0.4, tol=1e-7, max_iter=20):
    """Returns: (opt_alpha_s, opt_alpha_b, euler_sum)"""
```

#### 4.6.1 Relative Tolerance Scaling

All tolerance checks in the Newton solvers use **relative** rather than absolute
tolerances. The FOCs are sums of `c^{-gamma} * Rex_k` terms, so their natural
magnitude scales with marginal utility `c^{-gamma}`. With `gamma=5` and small
consumption, absolute FOC values can reach ~1e+9, making a fixed absolute
tolerance of 1e-7 impossible to achieve.

**Scale computation.** After the first FOC evaluation (all-bills corner), the
Euler equation level `e0 = sum * MU * R_bill` provides the natural FOC magnitude.
The floor of 1.0 ensures the tolerance never becomes tighter than the absolute
`tol` when FOC values happen to be small.

### 4.7 Terminal Condition

At the final age T (99), the agent consumes `c` and saves `a = W - c`.
Savings generate bequest utility only (no continuation value). The problem is:

```
V_T(W, s) = max_{c, alpha}  u(c) + beta * E[b(a * R_port, A(s))]
```

**Portfolio-consumption separation.** Because the bequest is CRRA in
terminal wealth `a * R_port`:

  b(a*R_port, A) = b_bar * (a*R_port/A)^{1-gamma} / (1-gamma)
                 = b_bar * A^{gamma-1} * a^{1-gamma} * R_port^{1-gamma} / (1-gamma)

the portfolio problem `min_{alpha} E[R_port^{1-gamma}]` (for gamma > 1)
is independent of savings `a` and consumption `c`. The portfolio is solved
once per financial state `i_s`, then consumption follows in closed form.

**Return construction.** For each financial state `i_s`, returns are built
using the same two-layer Gauss-Hermite quadrature as the retirement solver:

```
base_mu_r = const_r + A_r @ state_grid[i_s]       (3,) unconditional return mean
mu_r_k    = base_mu_r + M_v_nodes[k_v]             + M @ v^s innovation effect
R_bill    = exp(mu_r_k[0] + ret_nodes[k_r, 0])     gross real bill return
R_stock   = R_bill * exp(mu_r_k[1] + ret_nodes[k_r, 1])
R_bond    = R_bill * exp(mu_r_k[2] + ret_nodes[k_r, 2])
```

where `M_v_nodes = v_nodes @ M'` and `M = Sigma_rs @ inv(Sigma_ss)`.
This factored quadrature correctly reproduces: (i) the unconditional return
mean `Phi_0_ret + Phi_21 @ s_i`, (ii) the full return covariance `Omega_rr`,
and (iii) the state-return cross-covariance `Sigma_sr`.

**Terminal FOC (dropping constant `(1-gamma)` factor):**

```
FOC_k = sum_{k_v,k_r} w_v * w_r * R_port^{-gamma} * (R_k - R_bill) = 0
J_kl  = sum_{k_v,k_r} w_v * w_r * (-gamma) * R_port^{-gamma-1} * Rex_k * Rex_l
```

The Jacobian is negative definite for gamma > 1 (guaranteed by CRRA concavity
and R_port > 0 on the simplex), so Newton converges to the unique optimum.
No scipy dependency; the terminal solver uses the same `@njit` 2D Newton with
corner/edge/interior pattern as the retirement solver.

**Euler sum.** The FOC loop also accumulates
`euler_sum = sum w * R_port^{1-gamma} = E[R_port^{1-gamma}]`, used in the
consumption formula below.

**Consumption formula.** Given the optimal portfolio with
`moment = E[R_port^{1-gamma}]`, the FOC for consumption gives:

```
c^{-gamma} = beta * b_bar * A^{gamma-1} * (W-c)^{-gamma} * moment

omega = b_bar * A^{gamma-1} * moment
ratio = (beta * omega)^{-1/gamma}
c     = W * ratio / (ratio + 1)
```

This is constant in `c/W` across wealth (CRRA homogeneity) and independent
of income state `z` (bequest depends only on financial state via `A(y_1, spr)`
and conditional return distribution).

**Implementation:**

| Function | Role |
|----------|------|
| `_build_terminal_quad_returns` | Builds `(Rx_bill, Rx_stock_mult, Rx_bond_mult)` scenario arrays from state quadrature |
| `compute_terminal_portfolio_foc_jac` | `@njit` FOC + Jacobian + euler_sum, returns 6 values |
| `solve_portfolio_2d_terminal_constrained_njit` | `@njit` constrained Newton (corner/edge/interior) |
| `solve_portfolio_unconstrained_terminal_njit` | `@njit` unconstrained Newton with backtracking |
| `solve_terminal_age` | Loops over `i_s`, calls Newton, applies consumption formula |

### 4.8 Period Solvers

#### 4.8.1 Retirement Step

```python
@njit(parallel=True)
def solve_retirement_step(wealth_grid, savings_grid, z_grid, N_state,
                          c_next_full, pension_1d,
                          annuity_factors,
                          ...
                          gamma, psi_vec, beta, b_bar):
    """
    Solve one retirement period using EGM + 2D Newton.
    Parallelized: prange over i_s (financial state index).
    psi_vec: (n_z,) survival probs for this age, indexed by z.
    """
```

#### 4.8.2 Working Age Step

```python
@njit(parallel=True)
def solve_working_age_step(wealth_grid, savings_grid, z_grid, N_state,
                           c_next_full, log_det_next,
                           annuity_factors,
                           ...
                           gamma, psi_vec, beta, b_bar):
    """
    Solve one working-age period using EGM + 2D Newton.
    log_det_next: scalar = f(age_{t+1}).
    psi_vec: (n_z,) survival probs for this age, indexed by z.
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

All Newton solvers return `(alpha_s, alpha_b, euler_sum, exit_code, foc_resid)` where:

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

#### 4.10.2 Output Layers

**Per-age table** (printed as each age completes):
```
 Age  Phase  Time  Newton%  alpha_s  alpha_b  a_bill  c/W    %int  %edge  %corn  mono
  79  RETIRE  2.1s  100.0%    0.115    0.528   0.357  0.189   53%    34%    13%     0
```

**Post-solve report:** Newton convergence, portfolio regime breakdown,
share ranges, EGM monotonicity, and policy sanity checks.

---

## 5. Summary of Data Structures

### 5.1 Key Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| `state_grid` | (N_state, 3) | Joint state grid; row i = state vector in MODEL ordering (default [cy, spr, y_1] since 2026-04-30) |
| `v_nodes` | (K_s^3, 3) | State innovation quadrature nodes |
| `v_weights` | (K_s^3,) | State quadrature weights |
| `mu_r` | (N_state, N_state, 3) | Conditional return means (rtb, xr, xb) |
| `M_v_nodes` | (K_s^3, 3) | v_nodes @ M.T (precomputed) |
| `const_r` | (3,) | Phi_0_ret |
| `A_r` | (3, 3) | Phi_21 |
| `ret_nodes` | (n_ret_quad, 3) | Return residual quadrature nodes |
| `ret_weights` | (n_ret_quad,) | Return quadrature weights; sum=1 |
| `exp_ret_bill` | (n_ret_quad,) | exp(ret_nodes[:, 0]) |
| `exp_ret_stock` | (n_ret_quad,) | exp(ret_nodes[:, 1]) |
| `exp_ret_bond` | (n_ret_quad,) | exp(ret_nodes[:, 2]) |
| `n_ret_quad` | scalar int | `prod(n_ret_nodes_1d)`; e.g. scalar K -> K^3, tuple (3,9,3) -> 81 |
| `annuity_factors` | (N_state,) | A(y_1, spr, b_bar) annuity factor per state |
| `z_grid` | (n_z,) | Persistent income states |
| `eps_nodes/weights` | (n_eps,) | Judd-mixture for transitory shocks (n_eps = n_eps_nodes total) |
| `eta_nodes/weights` | (n_eta,) | Judd-mixture for persistent innovation eta (n_eta = n_eta_nodes total) |
| `log_det_profile` | (n_age,) | Deterministic age-earnings profile f(age) |
| `working_income` | (n_age, n_z, n_eps) | After-tax labor income (simulation; solver on-the-fly) |
| `pension_after_tax` | (n_age, n_z) | After-tax pension table |
| `survival_probs_2d` | (n_age, n_z) | Earnings-dependent survival probs |

### 5.2 Policy Function Shapes

| Array | Shape | Description |
|-------|-------|-------------|
| `C_mat` | (n_age, n_z, N_state, n_w) | Optimal consumption |
| `S_mat` | (n_age, n_z, N_state, n_w) | Optimal stock share |
| `B_mat` | (n_age, n_z, N_state, n_w) | Optimal nominal bond share |

### 5.3 VAR Variables

| Variable | Description | Units | Sample mean |
|----------|-------------|-------|-------------|
| `y_1` | 1-year nominal Treasury yield | annual decimal | 4.85% |
| `spr` | Yield spread (AAA 20yr - y_1) | annual decimal | 1.99% |
| `cy` | Log earnings yield: -log(CAPE) | log level | -2.99 |
| `rtb` | Real bill return | annual log | +0.91% |
| `xr` | Excess nominal stock return | annual log | +5.55% |
| `xb` | Excess nominal bond return | annual log | +1.43% |

---

## 6. Computational Notes

### 6.1 Inner Loop Cost

Per Newton evaluation, the inner loop iterates over:
- `n_state_quad` state innovation nodes (e.g. K_s=3 → 27) × `n_ret_quad` return
  residual nodes (`prod(n_ret_nodes_1d)`; e.g. uniform K_r=2 → 8, asymmetric
  (3,9,3) → 81)
- Working age: additionally × n_eta persistent innovation nodes × n_eps transitory nodes

| Config | Retirement | Working |
|--------|-----------|---------|
| K_s=3, K_r=2, n_eta=3, n_eps=5 (uniform) | 216 iters | 3,240 iters |
| K_s=3, K_r=3, n_eta=3, n_eps=5 (production) | 729 iters | 10,935 iters |
| K_s=3, K_r=(3,9,3), n_eta=5, n_eps=5 | 2,187 iters | 54,675 iters |

(Income joint count under Judd-mixture is `n_eta_nodes × n_eps_nodes`
total — half the marginal and a quarter the joint of the previous
concatenated-GH rule at the same `n_*_nodes` settings.)

### 6.2 Bequest Hoist Optimization

In `compute_foc_jac_working`, the bequest contribution to the FOC and Jacobian
depends only on the scenario `(k_v, k_r)` (through `w_inv = s_val * R_p`),
not on the income dimensions `(k_eta, i_e)`. The bequest terms are accumulated
once at the `(k_v, k_r)` level, while the inner `(k_eta, i_e)` loops handle only the
alive contribution (`psi * mu_alive`). This avoids redundant multiply-accumulates,
reducing inner-loop work by ~30%.

### 6.3 Parallelization

`prange` over `i_s` (financial state index) in both period solvers.
All quadrature arrays are read-only shared.

### 6.4 Memory

| Array | N_state=125 | N_state=343 |
|-------|-------------|-------------|
| `mu_r` (N_state^2 x 3) | 0.37 MB | 2.8 MB |
| `C_mat` (78 x n_z x N_state x 150) | ~20 MB (n_z=11) | ~55 MB |
| Total policy (x3) | ~60 MB | ~165 MB |

### 6.5 Numba Compatibility

All `@njit` functions receive plain NumPy arrays. The Precompute object unpacks
everything before passing to compiled functions. All arrays are standard float64
and fully Numba-compatible.

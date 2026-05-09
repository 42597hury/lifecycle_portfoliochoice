# CONVENTIONS — TIMING, INDEXING, UNITS, AND SIGN CONVENTIONS

**Purpose:** Single source of truth for every convention that cuts across modules.
Any agent working on the codebase should read this first and cross-check against
the code before assuming a convention.

**Code references:** `model.py`, `precompute.py`, `solver.py`, `simulation.py`,
`var.py`, `data/build_var_dataset.py`.

---

## 0. Time Index ↔ Age Mapping

```
t = age - start_age

start_age    = 22      →  t = 0
retire_age   = 67      →  t = 45    (retire_age_idx)
terminal_age = 99      →  t = 77    (n_age - 1)
n_age        = 78      (terminal_age - start_age + 1)
```

`ages = np.arange(start_age, terminal_age + 1)` lives on Precompute.
`ages[t]` converts any index back to calendar age.

**Working vs retirement:**
- Working phase: `t = 0..44`  (age 22..66), solver calls `solve_working_age_step_quad`
- Retirement phase: `t = 45..76` (age 67..98), solver calls `solve_retirement_step_quad`
- Terminal: `t = 77` (age 99), solver calls `solve_terminal_age`

The solver backward loop is `for t in reversed(range(n_age - 1))`, i.e. t = 76
down to 0. The boundary test is `age >= retire_age` → retirement solver.

---

## 1. Period Timing (Within-Period Sequence)

Each period t follows this exact sequence. This is the timing convention for
both the Bellman equation (solver) and the forward simulation.

```
START OF PERIOD t
│
│  Agent OBSERVES state:
│    x_t  = cash-on-hand (wealth + income already received)
│    s_t  = (cape, spr, y_1) financial state under the current baseline
│    z_t  = persistent income state (log, mean-zero)
│
│  Agent CHOOSES:
│    c_t       = consumption
│    α_s, α_b  = portfolio shares (stock, bond; bill = 1 - α_s - α_b)
│
│  SAVINGS:
│    a_t = x_t - c_t
│
│  ── between t and t+1 ──
│
│  1. State innovation drawn:  v^s_{t+1} ~ N(0, Σ_ss)
│     s_{t+1} = Φ_0_state + Φ_11 @ s_t + v^s_{t+1}
│
│  2. Returns realized (conditional on s_t and v^s_{t+1}):
│       μ_r = Φ_0_ret + Φ_21 @ s_t + M @ v^s_{t+1}
│       (xr, xb) ~ N(μ_r, Σ_r_cond)
│       R_bill  = exp(y_1)
│       R_stock = R_bill · exp(xr)
│       R_bond  = R_bill · exp(xb)
│       R_port  = α_s · R_stock + α_b · R_bond + α_bill · R_bill
│
│  3. Estate:  estate_t = a_t · R_port
│
│  4. Survival draw:  die with prob (1 - ψ(t, z_t))
│     If dead → estate_t becomes bequest; agent exits
│
│  5. Income realized (if alive):
│     Working (t < retire_age_idx):
│       z_{t+1} = ρ · z_t + η_{t+1}     (η from mixture-normal)
│       Y_{t+1} = disposable_income(exp(f(age_{t+1}) + z_{t+1} + ε_{t+1}))
│     Retirement (t ≥ retire_age_idx):
│       z_{t+1} = z_t (frozen)
│       Y_{t+1} = pension_after_tax(z_t)
│
│  6. Next-period cash-on-hand:
│       x_{t+1} = estate_t + Y_{t+1}
│
START OF PERIOD t+1
```

**Key implications of this timing:**

1. **x_t includes current-period income.** When the agent enters period t, they
   already have income in hand. At t=0 this is `x_0 = initial_wealth + Y_0`.

2. **Survival is checked AFTER portfolio returns.** The bequest is `a_t · R_port`,
   not `a_t`. Dead agents' estates include the realized portfolio return.

3. **Income Y_{t+1} is realized AFTER survival.** Only surviving agents receive
   next-period income. This matches Catherine (2025): the agent must be alive to
   work/collect pension.

4. **pension_table[t+1, :]** is used in the solver at step t, because the
   retirement solver needs next-period income to form x_{t+1}.

5. **log_det_profile[t+1]** = f(age_{t+1}) is similarly forward-looking.

---

## 2. Retirement Boundary — Exact Timing

```
Age 66 (t=44):  Last WORKING period.
  - Solver: solve_working_age_step_quad
  - z transitions one final time: z_{67} = ρ·z_{66} + η
  - Y_{67} = disposable_income(exp(f(67) + z_{67} + ε))   ← last paycheck
  - x_{67} = a_{66} · R_port + Y_{67}

Age 67 (t=45):  First RETIREMENT period.
  - Solver: solve_retirement_step_quad
  - z is frozen at z_{67} for the rest of life
  - Agent consumes and invests from x_{67}
  - Y_{68} = pension_after_tax(z_{67})   ← first pension payment
  - x_{68} = a_{67} · R_port + Y_{68}

Age 99 (t=77):  TERMINAL period.
  - No continuation value; all savings generate bequest utility only
  - Portfolio-consumption separation applies (CRRA homogeneity)
```

**Consequence:** The first pension payment arrives at age 68. At age 67, cash-on-hand
still includes the final labor income paycheck (earned between age 66 and 67).

---

## 3. Survival Probability Indexing

```
survival_probs_2d[t, iz] = P(survive from period t to period t+1)
                         = 1 - min(χ(z_iz) · m_baseline(age_t), 1)
```

- Shape: `(n_age, n_z)` = `(78, 11)`
- `t` indexes the period, NOT the transition target
- The solver at step t uses `psi = survival_probs[t, :]`
- The simulation checks `uniform > survival_probs_2d[t, z_idx]` → die
- At terminal age (t=77), all remaining agents die unconditionally

**Bequest pricing uses current-state annuity factor.** The annuity factor
`A(y_1, spr)` is evaluated at the CURRENT financial state `i_s`, not the
next-period state. This is consistent with Catherine (2025, eq. 21-22): the
annuity is priced at the time-t yield curve.

---

## 4. Unit Conventions

### 4.1 Model Units (Income, Wealth, Consumption)

All quantities are in units of the **SSA Average Wage Index (AWI)**.
1 model unit = AWI ≈ $54,100 (2019 dollars).

```
Examples:
  Income 0.216  = $11,700 (entry wage at age 22)
  Income 0.65   = $35,000 (median 45-year-old)
  Wealth 0.1    = $5,400  (initial wealth)
  Wealth 24     = $1.3M
  Pension 0.25  = $13,500/year
  Tax bracket 0.72 = $39,475 (TCJA 12%→22% threshold)
  Payroll cap 2.5  = $132,900
```

The model itself never uses dollars. The $54,100 conversion is purely for
sanity-checking.

### 4.2 Financial Variables — Level vs Log

| Variable | Domain | Units | Example value |
|----------|--------|-------|---------------|
| `cape` | State | Log level (= -log(CAPE)) | -2.73 |
| `spr` | State | Real log-yield spread | 0.0072 (= 0.72%) |
| `y_1` | State | Annual real log return | 0.0202 (= 2.02%) |
| `xr` | Return | Annual log excess return | 0.0518 (= 5.18%) |
| `xb` | Return | Annual log excess return | 0.0061 (= 0.61%) |

**Key distinctions:**
- `y_1` is the real one-year log yield and the deterministic bill log return.
- `spr = y_10_real - y_1` is a real log-yield spread.
- `cape = -log(CAPE)` is a log ratio, NOT a percentage.
- `xr` and `xb` are LOG EXCESS RETURNS (not levels, not percent).
- There is no separate `rtb` return variable in the active real-yields model.

**Compounding convention:** The annuity factor uses DISCRETE compounding
`(1+y)^{−k}`, matching the codebase. Do NOT use `exp(−y·k)`.

### 4.3 Income Process Variables

| Variable | Domain | Units |
|----------|--------|-------|
| `z` | z_grid | Log deviation from mean (mean-zero) |
| `η` (eta) | innovation | Log (additive to z) |
| `ε` (eps) | transitory | Log (additive to log income) |
| `f(age)` | log_det_profile | Log (deterministic age-earnings profile) |
| Gross income | exp(f + z + ε) | Model units (AWI) |

`z_grid` is MEAN-ZERO by construction. The unconditional mean of income is
`E[exp(z)] · E[exp(ε)] · exp(f(age))`, where both expectations ≈ 1 by the
zero-mean property of the mixture innovations.

### 4.4 Portfolio Shares

```
α_s = share of SAVINGS in stocks       (S_mat)
α_b = share of SAVINGS in bonds        (B_mat)
α_bill = 1 − α_s − α_b                (residual in bills)
```

These are shares of SAVINGS `a_t = x_t − c_t`, NOT of total wealth `x_t`.
The gross portfolio return on savings is:
```
R_port = α_s · R_stock + α_b · R_bond + α_bill · R_bill
```

---

## 5. Data Timing and Sampling

Raw financial data is sampled at **annual January** frequency. Inflation
expectations use Shiller December-over-December CPI inflation, with January
year `t` using December `t-1` information. The VAR is estimated directly at
**annual frequency** (no quarterly intermediate step). Sample: 1920-2011
(T=92 observations).

**Variable timing within a calendar year t:**
- State variables `(cape_t, spr_t, y_1_t)` are observed in January year t.
- Return variables `(xr_{t+1}, xb_{t+1})` are flows realized from January t to
  January t+1:
  - `log_R_bill,t+1 = y_1_t`
  - `xr_{t+1} = log((P_{t+1}+D_{t+1})/P_t) - y_1_nom,t`
  - `xb_{t+1}` is the CLM constant-duration return on the constructed 10-year
    real yield minus `y_1_t`

**CCV constrained estimation:** Pins `z̄ = sample_mean` exactly, ensuring
the Rouwenhorst grid is centered on the unconditional mean. Only lagged STATE
variables enter each equation (Φ_12 = 0, Φ_22 = 0 by restriction).

---

## 6. Frequency

**Everything is annual.** The model operates at annual frequency throughout:
- VAR parameters (Φ, Ω, z̄) are annual
- Income innovations (η, ε) are annual
- Survival probabilities are annual
- Tax brackets and pension formula use annual income
- All grids and quadrature nodes reflect annual distributions
- One model period = one calendar year

There is no sub-annual time step anywhere in the model.

---

## 7. Sign and Direction Conventions

| Convention | Details |
|-----------|---------|
| Bond duration | `M[xb, y_1] = -6.96`: a +100bp y_1 state innovation maps to about -7.0pp xb |
| Spread duration | `M[xb, spr] = -6.83`: a +100bp spread innovation maps to about -6.8pp xb |
| CAPE link | `M[xr, cape] = -0.94`: stock returns covary strongly with valuation-state innovations |
| Spread sign | `spr = y_10_real - y_1` is usually positive |
| cape sign | `cape = -log(CAPE) < 0` always (CAPE > 1 historically). More negative = more expensive market. |
| Utility | `u(c) = c^{1−γ}/(1−γ)` is NEGATIVE for γ > 1 (= 3). This is standard and correct. |
| Bequest weight | `b̄ = 10` enters multiplicatively. Higher b̄ = stronger bequest motive. |

---

## 8. Solver vs Simulation Treatment of z

| Aspect | Solver | Simulation |
|--------|--------|------------|
| z representation | Discrete grid `z_grid[i_z]` | Continuous float64 |
| z transitions | Judd-mixture quadrature over η, interpolate policies at off-grid z' | Draw continuous η from mixture, clamp z' to grid bounds |
| Income computation | On-the-fly `scalar_disposable_income(exp(f + ρz + η + ε))` | Same: direct from continuous z |
| Pension computation | Table lookup `pension_table[t, i_z]` | Direct from continuous z: `_scalar_pension_after_tax(z, avg_det)` |
| Policy lookup | Direct indexing by `i_z` | Linear interpolation between bracketing z-grid points |
| Financial state | Direct indexing by `i_s` | Nearest-neighbor snap to grid (after continuous VAR propagation) |

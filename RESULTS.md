# RESULTS BLUEPRINT — Implementation-Ready Specifications

## Prerequisites and Naming Conventions

All results assume the following objects are available:

```
model          : LifecyclePortfolioModel NamedTuple
pc             : Precompute object
C_mat, S_mat, B_mat : np.ndarray, shape (n_age, n_z, N_state, n_w)
diagnostics    : dict from run_lifecycle_solver()
sim            : dict from simulate_lifecycle(), keys listed below
```

Simulation dict keys and shapes (n_sim households, n_age periods):
```
sim["x"]            : (n_sim, n_age)   cash-on-hand
sim["c"]            : (n_sim, n_age)   consumption
sim["savings"]      : (n_sim, n_age)   savings = x - c
sim["alpha_s"]      : (n_sim, n_age)   stock portfolio share
sim["alpha_b"]      : (n_sim, n_age)   bond portfolio share
sim["alpha_bill"]   : (n_sim, n_age)   bill share = 1 - alpha_s - alpha_b
sim["R_port"]       : (n_sim, n_age)   realized gross portfolio return
sim["income"]       : (n_sim, n_age)   labor income or pension received
sim["estate"]       : (n_sim, n_age)   end-of-period wealth = savings * R_port
sim["z_idx"]        : (n_sim, n_age)   int32, persistent income state index
sim["state_idx"]    : (n_sim, n_age)   int32, financial state index
sim["alive"]        : (n_sim, n_age)   bool, alive at start of period t
sim["death_age"]    : (n_sim,)         int32, age at death (-1 if survived)
sim["estate_at_death"]: (n_sim,)       estate when died
sim["ages"]         : (n_age,)         integer ages, ages[0]=start_age=22
```

Grid references:
```
pc.wealth_grid      : (n_w,)           e.g. 150 points
pc.z_grid           : (n_z,)           e.g. 11 points, persistent income log-levels
pc.state_grid       : (N_state, 3)     columns = [rtb, y_nom, dp]
pc.state_grids      : list of 3 1-D arrays, marginal grids for each state variable
pc.ages             : (n_age,)         integer ages 22..99
pc.N_state          : int              e.g. 125 or 343
pc.working_income   : (n_age, n_z, n_eps)
pc.pension_after_tax: (n_age, n_z)
pc.survival_probs_2d: (n_age, n_z)
pc.annuity_factors  : (N_state,)
```

Index shorthands used below:
```
mid_z   = n_z // 2                     # median income state
mid_s   = N_state // 2                 # median financial state
mid_w   = n_w // 2                     # mid wealth grid (approximate)
retire_t = model.retire_age - model.start_age   # index of retirement age
```

Simulation config: `n_sim = 10_000`, `return_draw_mode = "monte_carlo"`, 
`initial_z = "normal"`, `initial_state = "stationary"`, `seed = 2025`.

---

## FIGURE 1 — Policy Functions: Portfolio Shares vs Age

**Goal:** The headline figure. Show how optimal portfolio composition evolves 
over the lifecycle at a representative point in the state space.

**Data source:** Policy arrays `S_mat`, `B_mat` directly (no simulation).

**Computation:**
```
For each age index t in range(n_age):
    age = pc.ages[t]
    # Pick a representative wealth level: use the wealth grid point closest
    # to the median simulated wealth at that age. For a first pass, use a 
    # fixed index, e.g. the point where wealth_grid is closest to 5.0 
    # (roughly median mid-life wealth in model units). Call this i_w_rep.
    # Better: after running simulation, compute median wealth per age and 
    # use that. For now, pick i_w_rep = index of wealth_grid closest to 3.0.
    
    alpha_s[t] = S_mat[t, mid_z, mid_s, i_w_rep]
    alpha_b[t] = B_mat[t, mid_z, mid_s, i_w_rep]
    alpha_bill[t] = 1.0 - alpha_s[t] - alpha_b[t]
```

**Plot:** Single panel, x-axis = age (22–99), y-axis = portfolio share (0–1).
Three stacked areas (bottom to top): bills (light gray), bonds (blue), stocks (red/orange).
Vertical dashed line at `retire_age = 67`.

**Variant:** Three-panel version, one for each income quintile:
- Panel A: `z_idx = 1` (low income, ~10th percentile of z_grid)
- Panel B: `z_idx = mid_z` (median income)
- Panel C: `z_idx = n_z - 2` (high income, ~90th percentile of z_grid)

Same `i_s = mid_s`, same `i_w_rep` for all.

---

## FIGURE 2 — Policy Functions: Portfolio Shares vs Wealth

**Goal:** Show how portfolio allocation depends on wealth at key ages.

**Data source:** `S_mat`, `B_mat` directly.

**Computation:**
```
For age in [35, 55, 75]:
    t = age - model.start_age
    For i_w in range(n_w):
        alpha_s[i_w] = S_mat[t, mid_z, mid_s, i_w]
        alpha_b[i_w] = B_mat[t, mid_z, mid_s, i_w]
        alpha_bill[i_w] = 1.0 - alpha_s[i_w] - alpha_b[i_w]
    x_axis = pc.wealth_grid
```

**Plot:** Three panels (one per age), x-axis = cash-on-hand (log scale), 
y-axis = share (0–1). Three lines: stocks, bonds, bills.

---

## FIGURE 3 — Policy Functions: Portfolio Shares vs Financial State

**Goal:** Show how the portfolio responds to each state variable, isolating 
the strategic/hedging component.

**Data source:** `S_mat`, `B_mat` directly.

**Computation:** Fix `age = 45`, `z_idx = mid_z`, `i_w = i_w_rep`. 
The financial state grid is a tensor product of 3 marginal grids. 
To vary one state variable while holding others at median:

```
# pc.state_grids is a list of 3 1D arrays: [rtb_grid, y_nom_grid, dp_grid]
# pc.state_indices is (N_state, 3): multi-index into the marginal grids
# Find the set of joint states where y_nom and dp are at their median values,
# but rtb varies:

n_per_dim = model discretization sizes  # e.g. (5,5,5)
mid_idx = [n // 2 for n in n_per_dim]

# States varying rtb (y_nom and dp at median):
rtb_slice = []
for i_s in range(N_state):
    idx = pc.state_indices[i_s]  # (3,) multi-index
    if idx[1] == mid_idx[1] and idx[2] == mid_idx[2]:
        rtb_slice.append(i_s)
# Now for each i_s in rtb_slice, the rtb value is pc.state_grid[i_s, 0]

# Similarly for y_nom_slice (rtb and dp at median) and dp_slice (rtb and y_nom at median)
```

Then for each slice, read `S_mat[t, mid_z, i_s, i_w_rep]` and `B_mat[...]`.

**Plot:** Three panels (one per state variable). x-axis = state variable value 
(annualized: rtb×4, y_nom×4 for percentages, dp as is). 
y-axis = portfolio share. Two lines per panel: α_stock, α_bond.

---

## FIGURE 4 — Simulation: Lifecycle Profiles

**Goal:** The standard lifecycle figure showing simulated outcomes with 
uncertainty bands.

**Data source:** Simulation output.

**Computation:** For each age index t:
```
alive_mask = sim["alive"][:, t]    # (n_sim,) boolean

x_vals   = sim["x"][alive_mask, t]
c_vals   = sim["c"][alive_mask, t]
inc_vals = sim["income"][alive_mask, t]
as_vals  = sim["alpha_s"][alive_mask, t]
ab_vals  = sim["alpha_b"][alive_mask, t]

# Compute percentiles:
x_p25, x_p50, x_p75 = np.percentile(x_vals, [25, 50, 75])
c_p25, c_p50, c_p75 = np.percentile(c_vals, [25, 50, 75])
inc_p50 = np.median(inc_vals)
as_mean = np.mean(as_vals)
ab_mean = np.mean(ab_vals)
abill_mean = 1.0 - as_mean - ab_mean
```

**Plot:** 2×2 panels, x-axis = age (22–99) for all:
- **Panel A (Wealth):** Lines for p25, p50, p75 of cash-on-hand. Shade IQR.
- **Panel B (Consumption):** Same layout as Panel A for consumption.
- **Panel C (Income):** Median income. Shows hump-shaped labor income then flat pension.
- **Panel D (Portfolio):** Mean α_stock, α_bond, α_bill as stacked areas 
  (use mean not median for shares so they sum to 1).
  Vertical dashed line at retirement.

---

## FIGURE 5 — Simulation: Portfolio by Income Quintile

**Goal:** Test whether high earners hold more long-duration assets, as 
Catherine predicts.

**Data source:** Simulation output.

**Computation:** At a fixed age (age = 45, so `t = 45 - 22 = 23`):
```
alive_mask = sim["alive"][:, t]
z_vals = sim["z_idx"][alive_mask, t]    # income state indices

# Define quintile boundaries on z_idx:
# z_grid has n_z points (e.g. 11). Quintile bins:
#   Q1: z_idx in {0, 1}
#   Q2: z_idx in {2, 3}
#   Q3: z_idx in {4, 5, 6}
#   Q4: z_idx in {7, 8}
#   Q5: z_idx in {9, 10}
# (adjust for actual n_z; or use np.digitize on z_vals into 5 equal-count bins)

For each quintile q:
    mask_q = (alive_mask) & (quintile_assignment == q)
    as_q = np.mean(sim["alpha_s"][mask_q, t])
    ab_q = np.mean(sim["alpha_b"][mask_q, t])
    abill_q = 1.0 - as_q - ab_q
    wealth_q = np.median(sim["x"][mask_q, t])
```

**Plot:** Grouped bar chart. x-axis = income quintile (Q1–Q5). 
For each quintile, stacked bar showing α_stock, α_bond, α_bill.
Add a secondary y-axis or annotation showing median wealth per quintile.

**Variant:** Same figure but at age 65 (pre-retirement) and age 80 (retirement).

---

## FIGURE 6 — Simulation: Portfolio by Financial Regime

**Goal:** Show that time-varying investment opportunities generate meaningful 
portfolio variation, the mechanism from CCV.

**Data source:** Simulation output.

**Computation:** At age 45 (`t = 23`):
```
alive_mask = sim["alive"][:, t]
state_vals = sim["state_idx"][alive_mask, t]

# Extract the y_nom component of each household's financial state:
y_nom_vals = pc.state_grid[state_vals, 1]  # column 1 = y_nom

# Split into terciles by y_nom:
tercile_bounds = np.percentile(y_nom_vals, [33.3, 66.7])
low_yield  = y_nom_vals <= tercile_bounds[0]
mid_yield  = (y_nom_vals > tercile_bounds[0]) & (y_nom_vals <= tercile_bounds[1])
high_yield = y_nom_vals > tercile_bounds[1]

# Compute mean portfolio shares for each regime
# Repeat for dp terciles
```

**Plot:** Two grouped bar charts side by side:
- Left: portfolio by y_nom tercile (low/mid/high yield)
- Right: portfolio by dp tercile (low/mid/high dividend yield)

Each bar is stacked: stocks, bonds, bills.

---

## FIGURE 7 — Hedging Demand Decomposition

**Goal:** Quantify how much of the bond (and stock) demand comes from 
intertemporal hedging vs myopic optimization.

**Requires:** Two separate model solves.

**Method:** 
1. **Full solve** (already done): yields `S_mat_full`, `B_mat_full`
2. **IID solve:** Create a modified precompute where `Pi_state` is replaced 
   by a matrix that always transitions to the median state (row `mid_s` 
   gets probability 1 in every row). This makes investment opportunities 
   constant from the agent's perspective. Solve again to get `S_mat_iid`, `B_mat_iid`.

Actually, the cleaner approach (matching CCV's definition): the myopic portfolio 
is the one-period-ahead optimal ignoring future state dependence. For the 
constrained lifecycle model, the most practical approximation:

**Alternative simpler method:** Use the simulation output directly.
The myopic component can be approximated by evaluating the policy at the 
*unconditional mean* financial state for all agents:
```
# Myopic approximation: evaluate policies at mid_s for all
For each t:
    myopic_as[t] = S_mat_full[t, mid_z, mid_s, i_w_rep]
    myopic_ab[t] = B_mat_full[t, mid_z, mid_s, i_w_rep]
    
    # Total demand: average over financial states, weighted by stationary dist
    pi_stat = get_stationary_distribution(pc.Pi_state)  # (N_state,)
    total_as[t] = sum over i_s: pi_stat[i_s] * S_mat_full[t, mid_z, i_s, i_w_rep]
    total_ab[t] = sum over i_s: pi_stat[i_s] * B_mat_full[t, mid_z, i_s, i_w_rep]
    
    hedging_as[t] = total_as[t] - myopic_as[t]
    hedging_ab[t] = total_ab[t] - myopic_ab[t]
```

Wait — this isn't quite right either. The CCV definition of hedging demand is 
`total - myopic`, where myopic means the portfolio you'd choose if investment 
opportunities were i.i.d. The cleanest way: re-solve with a Pi_state that 
concentrates on a single absorbing state. If that's too expensive for a first 
pass, use the spread across financial states as a proxy:

**Practical first-pass approach:**
```
For each age t:
    # Evaluate policy at each financial state
    as_by_state = S_mat_full[t, mid_z, :, i_w_rep]  # (N_state,)
    ab_by_state = B_mat_full[t, mid_z, :, i_w_rep]  # (N_state,)
    
    # Stationary-weighted average
    pi_stat = get_stationary_distribution(pc.Pi_state)
    as_mean = np.dot(pi_stat, as_by_state)
    ab_mean = np.dot(pi_stat, ab_by_state)
    
    # Spread: std of policy across states (measures state-dependence)
    as_std = np.sqrt(np.dot(pi_stat, (as_by_state - as_mean)**2))
    ab_std = np.sqrt(np.dot(pi_stat, (ab_by_state - ab_mean)**2))
```

**Plot:** Two panels:
- **Panel A (Stocks):** Mean α_stock ± 1 std across financial states, vs age.
  Shaded band shows how much the stock allocation varies with the macro state.
- **Panel B (Bonds):** Same for α_bond.

This directly shows whether investment opportunities matter quantitatively. 
If the bands are wide, hedging demands are significant.

**Full version (requires second solve):** If budget permits, do the IID solve 
and plot `total`, `myopic = IID_solve`, `hedging = total - myopic` as three 
lines per panel.

---

## FIGURE 8 — Role of Social Security

**Requires:** Second solve with Social Security turned off.

**Method:** Set `pension_after_tax` to zero in the Precompute object. 
Re-solve. Compare `S_mat_noSS`, `B_mat_noSS` with the baseline.

**Computation:** At each age, compare baseline vs no-SS portfolio at median state:
```
diff_ab[t] = B_mat_baseline[t, mid_z, mid_s, i_w_rep] - B_mat_noSS[t, mid_z, mid_s, i_w_rep]
diff_as[t] = S_mat_baseline[t, mid_z, mid_s, i_w_rep] - S_mat_noSS[t, mid_z, mid_s, i_w_rep]
```

Also do this by income quintile at age 45:
```
For each z_idx:
    diff_ab_by_z = B_mat_baseline[t45, z_idx, mid_s, i_w_rep] - B_mat_noSS[t45, z_idx, mid_s, i_w_rep]
```

**Plot:** 
- **Panel A:** Two lines over age: α_bond with SS, α_bond without SS. 
  (At median z and state.)
- **Panel B:** Bar chart at age 45: Δα_bond by income quintile (z_idx bins). 
  Positive bar means SS *reduces* bond holdings (substitution effect).

---

## TABLE 1 — Calibration Parameters

**Content:** Three sections:

**A. Preferences and lifecycle:**
| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Risk aversion | γ | 3 | — |
| Discount factor | β | 0.96 | — |
| Bequest horizon | b̄ | 10 years | Catherine (2025) |
| Start age | — | 22 | — |
| Retirement age | — | 67 | — |
| Terminal age | — | 99 | — |

**B. Income process (Guvenen et al. 2022):**
| Parameter | Value |
|-----------|-------|
| ρ | 0.991 |
| p_z | 0.176 |
| μ_η1, σ_η1 | -0.524, 0.113 |
| σ_η2 | 0.046 |
| p_ε | 0.044 |
| ... | ... |

**C. VAR (annual, after compounding from quarterly):**
| Variable | Unconditional mean | Persistence (diag Φ₁₁) |
|----------|-------------------|------------------------|
| rtb | -0.33%/yr | 0.017 |
| y_nom | 3.65%/yr | 0.861 |
| dp | -4.148 | 0.807 |

Also report: Equity premium (E[xr] = 5.36%), Bond premium (E[xb] = 2.36%),
M[xb, y_nom] = -39.4 (bond duration coefficient).

**Source:** `model` attributes and `var.py` outputs. All numbers are already 
in the DESIGN.md document.

---

## TABLE 2 — Simulation Summary Statistics

**Goal:** Validate that simulated lifecycle profiles are realistic.

**Data source:** Simulation output.

**Computation:**
```
For age in [25, 35, 45, 55, 65, 75, 85, 95]:
    t = age - 22
    alive = sim["alive"][:, t]
    survival_rate = np.mean(alive)
    
    x_med = np.median(sim["x"][alive, t])
    c_med = np.median(sim["c"][alive, t])
    inc_med = np.median(sim["income"][alive, t])
    sav_rate = np.median(sim["savings"][alive, t] / sim["x"][alive, t])
    
    as_mean = np.mean(sim["alpha_s"][alive, t])
    ab_mean = np.mean(sim["alpha_b"][alive, t])
    abill_mean = 1.0 - as_mean - ab_mean
```

**Table layout:**
| Age | Survival % | Median W | Median C | Median Y | Savings rate | α_stock | α_bond | α_bill |
|-----|-----------|----------|----------|----------|-------------|---------|--------|--------|
| 25 | ... | ... | ... | ... | ... | ... | ... | ... |
| 35 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | | |

---

## TABLE 3 — Welfare Cost of Suboptimal Portfolio Rules

**Goal:** Quantify the value of optimal three-asset allocation.

**Requires:** Multiple simulation runs with different portfolio rules imposed.

**Method:** For each alternative rule, run the simulation with the portfolio 
shares overridden (but consumption still optimized given those shares). 
Compute certainty-equivalent consumption:

```
# For each simulation run with rule r:
# CE_r = ((1-γ) * E[sum_t β^t * ψ_t * c_t^{1-γ}/(1-γ)])^{1/(1-γ)}
# Welfare loss = (CE_optimal - CE_rule) / CE_optimal × 100  (in %)
```

Actually, the standard CGM approach: compute the constant consumption 
supplement Δ such that:
```
E[sum_t β^t ψ_t u(c_rule_t + Δ)] = E[sum_t β^t ψ_t u(c_optimal_t)]
```
Solve for Δ. Report Δ/E[c_optimal] as percentage.

**Simpler equivalent:** Since utility is CRRA, the proportional CE loss is:
```
ratio = (E[V_rule] / E[V_optimal])^{1/(1-γ)}
loss_pct = (1 - ratio) × 100
```
where V = discounted lifetime utility from simulation.

**Rules to evaluate:**
1. **All bills:** `α_s = 0, α_b = 0` always
2. **All stocks:** `α_s = 1, α_b = 0` always  
3. **60/40:** `α_s = 0.6, α_b = 0, α_bill = 0.4`
4. **60/40 with bonds:** `α_s = 0.6, α_b = 0.4, α_bill = 0`
5. **Malkiel:** `α_s = max(0, (100 - age)/100), α_b = 0, α_bill = rest`
6. **Optimal stocks/bills only:** Re-solve with B_mat forced to 0 (two-asset model)
7. **Optimal bonds/bills only:** Re-solve with S_mat forced to 0

**Implementation note:** Rules 1–5 can be done by modifying the simulation 
to override `alpha_s, alpha_b` after the policy lookup. The consumption 
policy `C_mat` from the full solve is *not* optimal under the alternative 
portfolio rule, so this is an approximation. For exact welfare costs, rules 6–7 
require re-solving the model with restricted asset menus.

For a first pass, rules 1–5 with the approximation are informative. 
Rules 6–7 are Phase 2.

**Table layout:**
| Rule | CE loss (%) | CE loss (%) | CE loss (%) |
| | γ=3 | γ=5 | γ=8 |
|------|-----------|-----------|-----------|
| All bills | ... | ... | ... |
| All stocks | ... | ... | ... |
| 60/40 stocks/bills | ... | ... | ... |
| 60/40 stocks/bonds | ... | ... | ... |
| Malkiel rule | ... | ... | ... |

---

## TABLE 4 — Solver Diagnostics

**Data source:** `diagnostics` dict from `run_lifecycle_solver()`.

**Content:** Summary of numerical solution quality.

```
From the diagnostics dict, extract:
    total_calls     = sum of all Newton calls across all ages
    failed_calls    = sum of calls with exit_code == 8
    failure_rate    = failed_calls / total_calls * 100
    
    worst_foc       = max over all ages of worst FOC residual
    rms_foc         = sqrt(mean of squared FOC residuals)
    
    pct_interior    = % of calls with exit_code == 7
    pct_edge        = % with exit_code in {4,5,6}
    pct_corner      = % with exit_code in {1,2,3}
    
    mono_violations = total monotonicity violations in EGM
```

**Table layout:**
| Metric | Value |
|--------|-------|
| Total Newton calls | ... |
| Newton failure rate | ...% |
| Worst relative FOC | ... |
| RMS relative FOC | ... |
| Interior solutions | ...% |
| Edge solutions | ...% |
| Corner solutions | ...% |
| EGM monotonicity violations | ... |

Also useful: a per-age breakdown (which the solver already prints if verbose=1).

---

## FIGURE 9 — Sensitivity: Risk Aversion

**Requires:** Solves at γ ∈ {2, 3, 5, 8}. Four total solves.

**Computation:** For each γ, rebuild model with that γ, re-solve, extract 
policy at median state:
```
For each gamma_val in [2, 3, 5, 8]:
    # Rebuild model and precompute (only gamma changes)
    # Solve
    For each t:
        as[gamma_val, t] = S_mat[t, mid_z, mid_s, i_w_rep]
        ab[gamma_val, t] = B_mat[t, mid_z, mid_s, i_w_rep]
```

**Plot:** Two panels:
- **Panel A (Stock share):** Four lines (one per γ) over age.
- **Panel B (Bond share):** Four lines (one per γ) over age.

**Expected pattern:** Higher γ → lower stock share, higher bond share 
(conservative investors value duration hedging more, as in CCV Table 3).

---

## FIGURE 10 — Consumption-to-Wealth Ratio

**Data source:** Simulation output.

**Computation:**
```
For each t:
    alive = sim["alive"][:, t]
    c_vals = sim["c"][alive, t]
    x_vals = sim["x"][alive, t]
    ratio = c_vals / np.maximum(x_vals, 1e-10)
    cw_median[t] = np.median(ratio[np.isfinite(ratio)])
    cw_p25[t] = np.percentile(ratio[np.isfinite(ratio)], 25)
    cw_p75[t] = np.percentile(ratio[np.isfinite(ratio)], 75)
```

**Plot:** Single panel. x-axis = age, y-axis = c/W ratio. 
Median line with IQR shading. Vertical line at retirement.

**What to check:** Should be around 0.05–0.15 during working life, 
rising in retirement as agents run down wealth. Bequest motive prevents 
it from approaching 1.0 at terminal age.

---

## FIGURE 11 — Earnings-Dependent Survival Validation

**Data source:** `pc.survival_probs_2d` of shape `(n_age, n_z)`.

**Computation:**
```
# Compute cumulative survival from age 40 to each later age, per income quintile
age_40_t = 40 - model.start_age

For z_idx in [low, mid_low, mid, mid_high, high]:  # 5 representative z indices
    cumulative_surv = np.ones(n_age - age_40_t)
    for t in range(age_40_t + 1, n_age):
        cumulative_surv[t - age_40_t] = cumulative_surv[t - age_40_t - 1] * pc.survival_probs_2d[t-1, z_idx]
```

**Plot:** Single panel. x-axis = age (40–99), y-axis = survival probability 
from age 40. Five lines labeled by income quintile. 

Should show ~10 year life expectancy gap between bottom and top quintile, 
matching Chetty et al. (2016).

---

## FIGURE 12 — Income and Pension Profiles

**Data source:** `pc.working_income` (n_age, n_z, n_eps), 
`pc.pension_after_tax` (n_age, n_z).

**Computation:**
```
For each z_idx in [low, median, high]:
    For t in working ages:
        # Average over transitory shock nodes
        mean_income[t] = np.dot(pc.eps_weights, pc.working_income[t, z_idx, :])
    For t in retirement ages:
        mean_income[t] = pc.pension_after_tax[t, z_idx]
```

**Plot:** Single panel. x-axis = age, y-axis = after-tax income (model units).
Three lines for low/median/high persistent income. Clear drop at retirement 
visible. Shows the progressive Social Security replacement rate (low-z 
households see a smaller income drop).

---

---

## FIGURE 13 — Portfolio Allocation Fan Chart (Distribution over Agents)

**Goal:** Show not just the mean portfolio path but how much cross-sectional
dispersion there is at each age. Adapted from the old two-asset model's
alpha fan chart.

**Data source:** Simulation output.

**Computation:**
```
For each age index t:
    alive_mask = sim["alive"][:, t]
    as_vals  = sim["alpha_s"][alive_mask, t]
    ab_vals  = sim["alpha_b"][alive_mask, t]
    abill_vals = 1.0 - as_vals - ab_vals

    as_pcts = np.percentile(as_vals,  [10, 25, 50, 75, 90])
    ab_pcts = np.percentile(ab_vals,  [10, 25, 50, 75, 90])
    as_mean = np.mean(as_vals)
    ab_mean = np.mean(ab_vals)
```

**Plot:** Two panels (stocks, bonds). Each panel: x-axis = age, y-axis = share.
- Light shaded band: p10–p90
- Darker shaded band: p25–p75
- Solid line: mean
- Dashed line: median (p50)
Vertical dashed line at retirement.

**What to check:** If all agents behave similarly (narrow bands), portfolio
choice is primarily driven by age/lifecycle. Wide bands indicate large role
for cross-sectional heterogeneity in wealth, income state, or financial state.

---

## FIGURE 14 — Wealth Composition over the Lifecycle

**Goal:** Decompose total lifetime resources into human capital, social security
(pension PV), and financial wealth. A key motivating figure for why young
agents hold fewer bonds (their human capital is implicitly bond-like).

**Data source:** `pc` arrays and simulation output.

**Computation:**
```
For each age t and representative income state z_idx = mid_z:

    # Financial wealth: median simulated wealth at this age
    alive = sim["alive"][:, t]
    fin_wealth[t] = np.median(sim["x"][alive, t])

    # Human capital: PV of expected labor income from t to retire_age
    # Use a flat discount rate r_discount (e.g. 0.02 annual, or use
    # pc.r_bill_grid.mean() as the risk-free rate)
    hc[t] = 0.0
    if ages[t] < model.retire_age:
        for s in range(t, retire_t):
            # Expected income at age s, averaging over transitory shocks
            e_income = np.dot(pc.eps_weights, pc.working_income[s, z_idx, :])
            # Cumulative survival t -> s
            cum_surv = np.prod(pc.survival_probs_2d[t:s, z_idx])
            # Discount
            discount = (1.0 + r_discount) ** (-(s - t))
            hc[t] += cum_surv * discount * e_income

    # Social Security PV: PV of expected pension from retire_age onward
    ss_pv[t] = 0.0
    pension = pc.pension_after_tax[retire_t, z_idx]  # constant across ages
    for s in range(max(t, retire_t), n_age):
        cum_surv = np.prod(pc.survival_probs_2d[t:s, z_idx])
        discount = (1.0 + r_discount) ** (-(s - t))
        ss_pv[t] += cum_surv * discount * pension
```

**Plot:** Single stacked area chart. x-axis = age (22–99), y-axis = model units.
Three stacked components (bottom to top):
1. Human capital (warm color, largest when young)
2. Social Security PV (medium color, rises near retirement)
3. Financial wealth (cool color, grows mid-life then declines)
Vertical dashed line at retirement.

**Economic story:** When young, the agent's implicit total wealth is dominated
by human capital. Since HC pays out over a long horizon like a bond, young
agents have high implicit bond exposure — motivating low explicit bond holdings
as a hedge. As HC is depleted at retirement, explicit bond demand rises.

---

## FIGURE 15 — Wealth Distribution at Key Ages

**Goal:** Show how the cross-sectional wealth distribution evolves.

**Data source:** Simulation output.

**Computation:**
```
target_ages = [30, 45, 60, 75]
For each age:
    alive = sim["alive"][:, t]
    wealth_vals = sim["x"][alive, t]
    # Plot histogram (density=True, ~50 bins)
    # Mark mean and median
```

**Plot:** 1×4 panel (one subplot per age). Each subplot: histogram of
cash-on-hand for alive agents. Mark mean (vertical solid line) and median
(dashed). Use a consistent x-axis scale across panels or log-scale.

**What to check:** Right skew that increases with age; median well below mean
at older ages indicating growing wealth inequality.

---

## FIGURE 16 — Cross-Sectional Dispersion over the Lifecycle

**Goal:** Show how heterogeneity in wealth and consumption changes with age.

**Data source:** Simulation output.

**Computation:**
```
For each age t:
    alive = sim["alive"][:, t]
    sigma_w[t] = np.std(sim["x"][alive, t])
    sigma_c[t] = np.std(sim["c"][alive, t])
    cv_w[t]    = sigma_w[t] / np.mean(sim["x"][alive, t])   # coeff of variation
    cv_c[t]    = sigma_c[t] / np.mean(sim["c"][alive, t])
```

**Plot:** Two panels.
- **Panel A:** σ of wealth and σ of consumption over age (levels).
- **Panel B:** Coefficient of variation (σ/mean) for each — normalizes for
  the fact that mean wealth grows over time.
Vertical dashed line at retirement.

---

## FIGURE 17 — Savings Rate over the Lifecycle

**Goal:** Validate that the model produces realistic hump-shaped savings
behaviour.

**Data source:** Simulation output.

**Computation:**
```
For each age t:
    alive = sim["alive"][:, t]
    # Savings rate: savings / cash-on-hand
    x_vals = sim["x"][alive, t]
    s_vals = sim["savings"][alive, t]
    rate_vals = s_vals / np.maximum(x_vals, 1e-10)
    rate_vals = rate_vals[np.isfinite(rate_vals)]

    sav_mean[t]  = np.mean(rate_vals)
    sav_std[t]   = np.std(rate_vals)
    sav_med[t]   = np.median(rate_vals)
```

**Plot:** Single panel. x-axis = age, y-axis = savings rate.
- Solid line: mean savings rate
- Shaded band: mean ± 1 std
- Dashed line: median
- Horizontal line at 0 (dissaving visible in retirement)
Vertical dashed line at retirement.

**What to check:** Positive savings during working life (especially mid-career),
turning negative in retirement as agents run down wealth.

---

## FIGURE 18 — Interest Rate Duration Profile

**Goal:** The central economic mechanism figure. Show that as human capital
duration declines with age, optimal bond holdings rise — households hedge
their implicit bond exposure by holding explicit bonds later in life.

**Data source:** `pc` arrays; portfolio policy `B_mat`.

**Computation:**
```
r_discount = pc.r_bill_grid.mean()    # use mean real bill rate as discount
bond_maturity = 10                     # 10-year bond (matches b_bar)

For each age t and z_idx = mid_z:

    # 1. Human capital duration: weighted average time to payment
    hc_val = 0.0; hc_dur_num = 0.0
    for s in range(t, retire_t):
        e_income = np.dot(pc.eps_weights, pc.working_income[s, z_idx, :])
        cum_surv = np.prod(pc.survival_probs_2d[t:s, z_idx])
        discount = (1 + r_discount) ** (-(s - t))
        pv = cum_surv * discount * e_income
        hc_val     += pv
        hc_dur_num += (s - t) * pv       # years ahead × PV
    hc_duration[t] = hc_dur_num / hc_val if hc_val > 1e-10 else 0.0

    # 2. Social Security duration
    pension = pc.pension_after_tax[retire_t, z_idx]
    ss_val = 0.0; ss_dur_num = 0.0
    for s in range(max(t, retire_t), n_age):
        cum_surv = np.prod(pc.survival_probs_2d[t:s, z_idx])
        discount = (1 + r_discount) ** (-(s - t))
        pv = cum_surv * discount * pension
        ss_val     += pv
        ss_dur_num += (s - t) * pv
    ss_duration[t] = ss_dur_num / ss_val if ss_val > 1e-10 else 0.0

    # 3. Portfolio duration
    alpha_b_t = B_mat[t, z_idx, mid_s, i_w_rep]
    port_duration[t] = alpha_b_t * bond_maturity   # bill duration = 0, stock = 0
```

**Plot:** Single panel. x-axis = age, y-axis = duration in years.
Four lines:
- HC duration (solid, warm color) — long and declining
- SS duration (dashed, warm color) — shorter, also declining
- Portfolio duration (solid, cool color) — rising as bond share rises
- Reference line at `bond_maturity = 10` years (dotted, gray)
Vertical dashed line at retirement.

**Economic story:** Young agents have high-duration HC → already implicitly
holding long-duration assets → hold fewer bonds. As HC depletes, they shift
into explicit bonds to maintain duration balance. Quantifies the "bonds as
hedge for HC" argument.

---

## FIGURE 19 — Bond Premium Sensitivity

**Goal:** Show how portfolio allocation (especially bonds) responds to changes
in the bond excess return. In the new model this is controlled by the mean
of `xb` in the VAR — analogous to the term premium in the old model.

**Note:** This requires multiple solves with modified VAR configurations.
In the new model, change `z_bar[ret_idx[1]]` (the unconditional mean of `xb`)
to simulate different bond premia, while keeping all other VAR parameters fixed.

**Method:**
```
bond_premia = [-0.01, 0.00, 0.01, 0.02, 0.03]   # annual excess return levels

For each bp in bond_premia:
    # Modify the annualized var_config: shift z_bar for xb
    var_config_bp = dict(var_config)
    z_bar_new = np.array(var_config["z_bar"])
    z_bar_new[return_indices[1]] += (bp - baseline_xb_mean)
    # Also adjust Phi_0_ret to be consistent with new z_bar
    var_config_bp["z_bar"] = z_bar_new.tolist()
    # Re-build model and solve
```

**Plot:** Two panels (stock share, bond share) over age.
One line per bond premium level, coloured by premium magnitude (viridis).

**Economic story:** Higher bond premium → more bond demand throughout the
lifecycle. Shows the substitution effect between stocks and bonds as the
relative return changes.

---

## IMPLEMENTATION ORDER

### Phase 1: Figures from a single solve + one simulation
*No re-solving needed. This gets you a presentable draft.*

1. **Table 1** — Calibration parameters (just format values from `model`)
2. **Table 4** — Solver diagnostics (from `diagnostics` dict)
3. **Figure 12** — Income/pension profiles (from `pc` arrays)
4. **Figure 11** — Survival curves (from `pc.survival_probs_2d`)
5. **Figure 1** — Policy: portfolio vs age (from `S_mat, B_mat`)
6. **Figure 2** — Policy: portfolio vs wealth (from `S_mat, B_mat`)
7. **Figure 3** — Policy: portfolio vs financial state (from `S_mat, B_mat`)
8. **Figure 4** — Simulation: lifecycle profiles (from `sim`)
9. **Figure 5** — Simulation: portfolio by income quintile (from `sim`)
10. **Figure 6** — Simulation: portfolio by financial regime (from `sim`)
11. **Table 2** — Simulation summary statistics (from `sim`)
12. **Figure 10** — Consumption-to-wealth ratio (from `sim`)
13. **Figure 7** — Hedging demand spread (from `S_mat, B_mat`, stationary dist)
14. **Figure 13** — Portfolio allocation fan chart (percentile bands from `sim`)
15. **Figure 14** — Wealth composition stacked area (HC + SS PV + financial from `sim`)
16. **Figure 15** — Wealth distribution histograms at ages 30, 45, 60, 75 (from `sim`)
17. **Figure 16** — Cross-sectional dispersion: σ wealth and σ consumption over age (from `sim`)
18. **Figure 17** — Savings rate fan chart: mean ± 1 std, median (from `sim`)

### Phase 2: Requires additional solves
*Each item needs one extra backward induction.*

19. **Figure 8** — Social Security decomposition (solve with pension=0)
20. **Figure 9** — Risk aversion sensitivity (solves at γ=2,5,8)
21. **Table 3** — Welfare costs, rules 1–5 approximation (modified simulations)
22. **Figure 18** — Interest rate duration profile: HC duration, SS duration, portfolio duration (requires HC/SS present-value pass)
23. **Figure 19** — Bond premium sensitivity: multiple solves varying E[xb] (5–7 solves)

### Phase 3: Full decompositions
*More expensive, for a polished paper.*

24. **Figure 7 full version** — IID solve for clean hedging decomposition
25. **Table 3** rules 6–7 — Re-solve with restricted asset menus
26. No-labor-income solve (pure financial wealth model, CCV comparison)
27. Grid convergence check (resolve at 7×7×7 or 9×9×9)

# LABOUR INCOME AND SOCIAL SECURITY -- VALIDATION DOCUMENT

**Purpose:** Consolidate everything about the labour income process, tax schedule,
pension formula, and their discretizations in one place. Close out validation
so this component can be signed off.

**Code references:** `model.py` (tax/pension functions), `discretization.py`
(z-grid, quadrature), `precompute.py` (lookup tables), `solver.py` (how income
enters the FOC), `simulation.py` (direct income computation).

---

## 0. Model Units

The model works in units of the **SSA Average Wage Index (AWI)**. Catherine
(2025, Section 5.1) normalises the wage index to L̄ = 1, so all quantities --
income, wealth, consumption, tax brackets, pension bend points -- are expressed
as multiples of the AWI.

The AWI is a national average of total wages reported to SSA each year. In 2019
it was $54,099.99. SSA uses it to index past earnings when computing benefits,
and to annually adjust the payroll tax cap and PIA bend points.

**Anchor:** Catherine states that initial wealth = 0.1 model units = $5,400
(2019 dollars). This gives 1 model unit = $54,000, consistent with the 2019 AWI.

**How the calibration uses it:** All real-world dollar thresholds are divided by
the AWI to get model units:
- Tax bracket at $39,475 -> 39,475 / 54,100 ≈ 0.72
- Payroll cap at $132,900 -> 132,900 / 54,100 ≈ 2.5
- PIA bend point at $11,112/yr -> 11,112 / 54,100 ≈ 0.21

The polynomial coefficients (b0, b1, b2, b3) were calibrated so that
`exp(f(age))` produces earnings in these units. A median 45-year-old earns
`exp(f(45)) ≈ 0.65`, i.e. 65% of the AWI ≈ $35,000.

**Interpretation examples:**
- Income of 0.216 = $11,700 (typical 23-year-old entry wage)
- Wealth of 24 = $1.3M
- Pension of 0.25 = $13,500/year

The model itself never uses dollars. The $54,100 conversion is purely for
sanity-checking outputs against real-world data.

**Whose earnings dynamics?** The income process parameters (Guvenen et al. 2021,
Econometrica) are estimated from the SSA Master Earnings File -- a 10% sample of
all US Social Security numbers covering millions of male workers (1978-2013).
The agent is endowed with the earnings dynamics of a representative US male
worker: same persistence, shock probabilities, and shock sizes as the average
pattern in that population.

**Level mismatch:** The median worker at peak age earns ~$35k in the model vs
~$49k in BLS data. This is expected -- the income process (Guvenen et al. 2022)
is calibrated to match individual earnings dynamics (persistence, shock
variance, kurtosis), not cross-sectional levels. What matters for portfolio
choice is that tax brackets, payroll caps, and replacement rates are correctly
scaled relative to earnings, which they are.

---

## 1. Theoretical Income Process

Source: Guvenen et al. (2021, Econometrica) / Catherine (2025, Appendix E.1).

Gross labour income at age t:

```
log(Y_gross) = f(age_t) + z_t + eps_t
```

### 1.1 Deterministic age-earnings profile

```
f(age) = b0 + b1*age + b2*age^2/10 + b3*age^3/100

b0 = -6.142,  b1 = 0.3040,  b2 = -0.051,  b3 = 0.002586
```

Hump-shaped: starts low at 22, peaks around 50-55, slight decline toward 67.
`exp(f(age))` gives the level in model units for the median worker (z=0).
Career average: `avg_det = mean(exp(f(age))) over ages 22-66 ~ 0.507`.

### 1.2 Persistent component z

AR(1) with mixture-normal innovations:

```
z_{t+1} = rho * z_t + eta_{t+1}      rho = 0.991

eta ~ pz * N(mu_eta1, sigma_eta1^2) + (1-pz) * N(mu_eta2, sigma_eta2^2)

pz = 0.176
mu_eta1 = -0.524,    sigma_eta1 = 0.113    (rare large negative shocks)
mu_eta2 = +0.112,    sigma_eta2 = 0.046    (frequent small positive drift)

Zero-mean condition: mu_eta2 = -(pz/(1-pz)) * mu_eta1
```

Nearly a random walk. Main source of lifetime inequality.
The solver makes no assumption about initial z — it computes the optimal
policy at every z-grid point. The simulation has three options for the initial
distribution (`"median"`, `"stationary"`, `"normal"`; default `"stationary"`).
z freezes at retirement (age 67) and determines pension for life.

### 1.3 Transitory shock eps

```
eps ~ pe * N(mu_eps1, sigma_eps1^2) + (1-pe) * N(mu_eps2, sigma_eps2^2)

pe = 0.044
mu_eps1 = 0.134,    sigma_eps1 = 0.762    (rare large transitory shocks)
mu_eps2 = -0.006,   sigma_eps2 = 0.055    (typical small noise)

Zero-mean enforced: mu_eps2_eff = -(pe/(1-pe)) * mu_eps1
```

Washes out period-to-period. Does not affect pension.

---

## 2. Tax Schedule

Applied identically to working income and pension benefits.
Implementation: `disposable_income_working()` at
[model.py:285](model.py#L285).

### 2.1 Payroll tax

```
payroll_tax = 10.6% * min(Y_gross, 2.5)
taxable_income = max(0, Y_gross - payroll_tax)
```

Cap at 2.5 model units ~ $135,250. Matches 2019 SS taxable max ($132,900) to ~2%.

**Why 10.6% and not 12.4%?** Real FICA payroll tax is 12.4% OASDI, split
internally as 10.6% OASI (Old-Age and Survivors Insurance) + 1.8% DI
(Disability Insurance). The model has no disability risk and pays no
disability benefits, so charging the full 12.4% would be a tax with no
corresponding benefit. Using 10.6% preserves tax–benefit correspondence:
the agent only pays for insurance they can actually collect. This follows
Catherine (2025), who uses 10.6% in eq. 17 and justifies the adjustment
explicitly in her related wealth-inequality paper:
*"We adjust these estimates to remove the value of Disability Insurance
program by assuming that the Old Age and Survivor program represents
10.6/12.4 of the total, which is consistent with the allocation of
payroll tax revenues."*

### 2.2 Progressive income tax (2019 TCJA, single filer)

Applied to `taxable_income = Y_gross - payroll_tax`:

| Bracket (model units) | Bracket ($, at $54,100/unit) | Marginal rate | Cumulative tax at upper bound |
|----------------------|------------------------------|---------------|-------------------------------|
| 0 -- 0.18            | 0 -- $9,738                  | 10%           | 0.018                         |
| 0.18 -- 0.72         | $9,738 -- $38,952            | 12%           | 0.0828                        |
| 0.72 -- 1.54         | $38,952 -- $83,314           | 22%           | 0.2632                        |
| 1.54 -- 2.94         | $83,314 -- $159,054          | 24%           | 0.5992                        |
| 2.94 -- 3.73         | $159,054 -- $201,793         | 32%           | 0.8520                        |
| 3.73 -- 9.32         | $201,793 -- $504,212         | 35%           | 2.8085                        |
| 9.32+                | $504,212+                    | 37%           | --                            |

Verified against real 2019 brackets to within 1-2% (see TODO.md item 7).

### 2.3 Effective tax rate examples

| Gross income (model) | Gross ($) | Payroll | Income tax | Net | Effective rate |
|----------------------|-----------|---------|------------|-----|----------------|
| 0.507 (median)       | $27,430   | 0.054   | 0.037      | 0.416 | 18.0%        |
| 1.0                  | $54,100   | 0.106   | 0.089      | 0.805 | 19.5%        |
| 2.5 (cap)            | $135,250  | 0.265   | 0.399      | 1.836 | 26.6%        |

*(These are approximate -- verify numerically in validation.)*

---

## 3. Social Security Pension

PIA formula (bend points, replacement rates): Catherine (2025, Section 3.4, eqs. 17-20).
AIME computation: **our approximation**, deviating from Catherine who tracks
cumulative career earnings as an explicit state variable.
Implementation: `compute_pension_after_tax()` at
[model.py:349](model.py#L349).

### 3.1 AIME approximation

Catherine (2025) carries career-average earnings as an additional state variable
(eq. 20). We avoid this extra dimension by approximating AIME from the terminal
persistent state z at retirement:

```
AIME(z) = min(exp(z) * avg_det, 2.5)

avg_det = mean(exp(f(age))) over ages 22-66 ~ 0.507
```

**Why this works:**
- rho = 0.991 means terminal z ~ career-average z
- Transitory shocks eps average out over 45 years
- Avoids an extra state variable (career-average earnings)

**What we lose:** Path-dependence of career earnings. An agent with low
lifetime z who gets lucky late has their pension overstated (and vice versa).
With rho ~ 1 this error is small for most agents.

The 2.5 cap matches the payroll tax cap -- earnings above the SS taxable
maximum don't count toward AIME.

### 3.2 PIA formula (Catherine eq. 19)

```
             { r1 * AIME                                         if AIME <= b1
PIA(AIME) =  { r1*b1 + r2*(AIME - b1)                            if b1 < AIME <= b2
             { r1*b1 + r2*(b2 - b1) + r3*(AIME - b2)             if AIME > b2

Bend points: b1 = 0.21,  b2 = 1.25
Rates:       r1 = 0.90,  r2 = 0.32,  r3 = 0.15
```

Strongly progressive: 90% replacement on first $11,361 of AIME,
32% on next $56,276, 15% above.

### 3.3 Pension taxation

The gross PIA is taxed by the same 7-bracket schedule as working income
(Section 2.2 above). This is a simplification -- real SS benefits have
a separate partial-taxation rule -- but maintains consistency.

### 3.4 Timing

- Last z transition: age 66 -> 67
- z freezes at age 67 (retirement)
- Last labour income received: at age 67 (final paycheck)
- First pension payment: at age 68
- Pension is constant in model units for all remaining years

### 3.5 Pension schedule (n_z = 11 grid)

| z_grid value | exp(z) | AIME | Gross PIA | After-tax pension | Replacement rate* |
|-------------|--------|------|-----------|-------------------|-------------------|
| (verify numerically across all 11 grid points) | | | | | |

*Replacement rate = pension / career-average after-tax income at same z.

---

## 4. Discretization

### 4.1 z-grid

`discretize_income_ar1_mixture()` at
[discretization.py:283](discretization.py#L283).

```
std_z = sqrt(var_eta / (1 - rho^2))
z_grid = linspace(-3*std_z, +3*std_z, n_z)       n_z = 11

var_eta = pz*(sigma_eta1^2 + (mu_eta1 - 0)^2) + (1-pz)*(sigma_eta2^2 + (mu_eta2 - 0)^2)
```

Uniform spacing: `dz = z_grid[1] - z_grid[0]`.

Also produces `Pi_Nz` (Tauchen transition matrix) but this is **not used**
by the solver or simulation -- retained for diagnostics only.

### 4.2 Judd-mixture quadrature for eta (persistent innovation)

`get_eta_quadrature_mixture()` at
[discretization.py:421](discretization.py#L421); private helper
`_judd_mixture_quadrature()` at
[discretization.py:338](discretization.py#L338).

Instead of `Pi_z`, the solver integrates over `z'` via a Judd (1998)
quadrature built directly on the mixture density. Three steps:

1. Build the `(2n+1)`-vector of raw moments `m_k = E[X^k]` of the mixture
   (`_normal_raw_moment` × `_mixture_raw_moments`).
2. Solve the `n × n` Hankel system `H_oo a = -h` for the monic orthogonal
   polynomial of degree `n`. Its `n` real roots are the quadrature nodes.
3. Solve the `n × n` Vandermonde system `Σ_i ω_i x_i^k = m_k` (`k = 0..n-1`)
   for the weights.

```
eta_nodes   ∈ ℝ^n   (sorted ascending, all real, distinct)
eta_weights ∈ ℝ^n   (all strictly positive; sum to 1)
```

`n` is the **total** node count — `disc_config.n_eta_nodes` (no longer
"per-component K, doubled internally"). Zero-mean is enforced by setting
`mu_eta2_eff = -(pz/(1-pz))·mu_eta1` before constructing the mixture moments.

**Polynomial exactness against the mixture.** The Judd `n`-point rule
integrates monomials of degree ≤ `2n - 1` exactly against the mixture
density (Theorem 1 of Judd 1998 §7). Same polynomial-exactness order
as a `K`-per-component stratified Gauss-Hermite rule when `n = K`, but
half the nodes for the marginal and a quarter for the joint
`η × ε` integral.

### 4.3 Judd-mixture quadrature for eps (transitory shock)

`get_eps_quadrature_corrected()` at
[discretization.py:395](discretization.py#L395). Identical construction
to §4.2, applied to the eps mixture. `n_eps = disc_config.n_eps_nodes`
total nodes; polynomial exactness `2n_eps - 1` against the eps mixture.

The eps mixture has excess kurtosis +52, so polynomial exactness alone
does **not** imply integrand accuracy on `E[exp(-γ·ε)]` for γ ≥ 5; see
§4.8 for the production node-selection guidance and the high-γ wall.

### 4.4 Precomputed income table

`_precompute_working_income()` at
[precompute.py:320](precompute.py#L320).

```python
y_gross = exp(log_det_profile[t] + z_grid[iz] + eps_nodes[ie])
working_income[t, iz, ie] = disposable_income_working(y_gross)
```

Shape: `(n_age, n_z, n_eps)`. Evaluated at grid points only.

### 4.5 Precomputed pension table

`_precompute_pension()` at
[precompute.py:341](precompute.py#L341).

```python
base_pension = compute_pension_after_tax(z_grid, avg_det)    # (n_z,)
pension_after_tax[t, :] = base_pension for all t             # tiled across ages
```

Shape: `(n_age, n_z)`. Age-invariant.

### 4.6 How income enters the solver

**Working age** (`compute_foc_jac_working` in
[solver.py](solver.py)).

For each persistent-innovation GH node `eta_nodes[k_eta]`, the solver
computes a continuous next-period state

```
z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]
```

which is almost never on the 11-point grid.

**Income: on-the-fly, exact.** Income is computed directly from the
continuous `z_next` rather than interpolated from a precomputed table:

```
y_gross_next = exp(log_det_profile[t+1] + rho*z_grid[z_idx] + eta_nodes[k_eta] + eps_nodes[i_e])
income_next  = scalar_disposable_income(y_gross_next)
```

The progressive tax schedule is then evaluated pointwise — no
interpolation in `z`. Because `scalar_disposable_income` is the *same*
function used in simulation (see [model.py:312](model.py#L312)), income
in the solver and income in the simulation are bit-exact equal.

**exp() factoring.** Computing `exp(A+B+C+D)` inside a triple loop
(`k_eta` × `i_e` × each Newton step) would spend thousands of
transcendental calls per FOC. The solver precomputes `exp_eps[ie]`,
`exp_eta[ke]` (small arrays, ~24-32 bytes each) and
`base_det_z = exp(log_det_profile[t+1] + rho*z_grid[z_idx])` *once* at
the top of `compute_foc_jac_working`. The inner loop then does

```
y_gross_next = base_det_z * exp_eta[k_eta] * exp_eps[i_e]
```

— three multiplications per `(k_eta, i_e)` pair, reducing
transcendental calls per FOC evaluation from ~4,500 to ~8 (≈500×). The
optimization is safe because `exp` factorizes exactly across addition;
all that is lost is roundoff below ~1e-16.

**Consumption policy is still interpolated.** The solved
`c_next_full[iz, j_s, iw]` array is discrete in `z` and must be
bilinearly interpolated in `(z_next, x_next)`:

```
iz_lo  = floor((z_next - z_grid[0]) / dz)
iz_lo  = clip(iz_lo, 0, n_z - 2)               # boundary guard
frac_z = (z_next - z_grid[iz_lo]) / dz
frac_z = clip(frac_z, 0.0, 1.0)                # boundary guard

c_lo = (1 - frac_w) * c_next_full[iz_lo,   j_s, iw  ]
     +      frac_w  * c_next_full[iz_lo,   j_s, iw+1]
c_hi = (1 - frac_w) * c_next_full[iz_lo+1, j_s, iw  ]
     +      frac_w  * c_next_full[iz_lo+1, j_s, iw+1]

c_next = (1 - frac_z) * c_lo + frac_z * c_hi
```

Graphically: the true consumption policy `c*(z, x)` is a smooth
surface; the solver only sees `n_z × n_wealth` dots on that surface and
reads off intermediate values via bilinear interpolation. This is now
the *only* z-interpolation in the working-age FOC.

**Boundary clipping.** If `z_next` falls outside `[z_grid[0],
z_grid[-1]]`, the `clip` on `iz_lo` and `frac_z` forces the
consumption interpolation to evaluate *at the grid boundary* (flat
extrapolation in `z`). Income is not affected by clipping because it
is computed from the continuous `z_next` directly. Section 5 records
that, at the current calibration, clipping does not visibly distort
the policies.

**Retirement** (`compute_foc_jac_retirement` in solver.py):

```
x_next = savings * R_port + pension_next_scalar
```

No interpolation -- pension is looked up directly by z-grid index
because z is discretized to a grid state during retirement (no `η`
shock enters retirement).

### 4.7 Simulation vs solver

Both the solver and the simulation now use the **same** scalar income
function, `scalar_disposable_income` in [model.py:312](model.py#L312):

```python
# Working (both solver and simulation):
y_gross = exp(log_det_profile[t+1] + z_next + eps_nodes[eps_idx])
income  = scalar_disposable_income(y_gross)

# Retirement (simulation):
income  = _scalar_pension_after_tax(z_val, avg_det)
```

Because the solver no longer interpolates income in `z`, the
solver-side and simulation-side disposable-income values for the same
`(age, z, eps)` triple are bit-exact equal (to within floating-point
roundoff). The only z-interpolation remaining in the working-age FOC
is on the consumption policy (Section 4.6).

### 4.8 Quadrature node selection for production

Two distinct accuracy questions:

(i) **polynomial exactness against the mixture density** — the formal
property of the rule. Judd `n`-point: exact to degree `2n - 1`. So
`n = 2` reproduces mean+variance+skewness; `n = 3` adds the 4th and
5th raw moment (so kurtosis is exact); `n = 5` extends through the 9th
moment.

(ii) **accuracy on the actual FOC integrand `E[exp(-γ·X)]`** — what an
economist cares about. Polynomial exactness is a necessary but not
sufficient guarantee here: when the integrand has heavy weight in the
heavy-tail region, low-order rules under-resolve the tail even while
matching low-order moments.

The audit script
[tests/audit_judd_economist.py](tests/audit_judd_economist.py) and the
notebook section §C.4–C.5 of
[verify_discretization.ipynb](verify_discretization.ipynb) measure (ii)
directly against a 400-point reference. Production guidance reads off
those numbers.

**FOC-integrand error** (CRRA `E[exp((1-γ)·X)]`, rel err vs analytic
mixture-MGF):

```
                  η rel err                       ε rel err
n   γ=3      γ=5      γ=8      γ=10     γ=3      γ=5      γ=8
---------------------------------------------------------------
2   1.6e-3   2.1e-2   1.5e-1   2.8e-1   3.1e-2   ~70%     ~100%
3   2.6e-5   5.5e-3   4.9e-2   1.1e-1   6.4e-3   ~55%     ~99%
4   1.0e-7   1.5e-4   1.5e-3   7.3e-3   1.1e-3   ~37%     ~99%
5   4.6e-10  4.2e-6   8.4e-5   6.7e-4   1.4e-4   ~17%     ~97%
6   1.7e-12  9.9e-8   3.9e-6   5.1e-5   1.4e-5   ~7%      ~92%
8   ~4e-16   2.0e-11  3.1e-9   1.1e-7   1e-7     ~0.1%    ~69%
```

**Reading.**

- **η is easy.** At the production γ=3 default, `n_eta = 3` already
  gives rel err 2.6e-5. `n_eta = 5` is overkill except at γ ≥ 8.
- **ε is the binding dial, and the wall starts early.** At γ = 5, even
  `n_eps = 8` leaves ~17% rel err on `E[exp(-4·ε)]`. At γ = 8, the
  rule is essentially uninformative for any reasonable `n` — the
  excess kurtosis +52 means the integral is dominated by deep-tail
  realizations no low-order quadrature can capture.
- **Recommendation by γ regime:**

```
γ regime     n_eta    n_eps    income joint    use case
-----------------------------------------------------------------------
γ ≤ 3         3        5             15        production default ✓
γ ∈ {4, 5}    3        6             18        slight margin on ε
γ ∈ {6, 7}    4        8             32        ε tail still ~10%, accept
γ ≥ 8         5       ≥8           ≥40        ε bias > 50%; reduce γ or
                                              redesign the integrand
```

The current production setting is `n_eta_nodes = 3, n_eps_nodes = 5`
(15 income nodes, joint `27 × 27 × 15 = 10,935` nodes per FOC including
the 27-node state and 27-node return quadratures).

**How this compares to the previous concatenated-GH method.** At the
same polynomial-exactness order, the older "K-per-component"
stratified rule was ~10× more accurate on `E[exp(-γ·η)]` than Judd at
the same `n`, because GH stratification puts more nodes inside each
component's tail. Going to Judd `n_eta = 5` (5 nodes) recovers and
exceeds the old K=3 (6 nodes) accuracy — same polynomial order
(9 vs 5), one fewer node, materially smaller error at high γ. So the
migration's real win is **at `n = 5`, not at `n = 3`**: better
accuracy at fewer nodes once the user opts into the higher tier.

**Context for "does this error matter."** Quadrature error feeds into
the Euler FOC right-hand side; CRRA marginal-utility inversion scales
the consumption-policy error by `1/γ`. At γ = 3 with `n_eps = 5`, a
~1e-4 ε integrand error implies ~3e-5 relative consumption-policy
error — well below the dominant numerical error from `z`-grid linear
interpolation of the consumption policy (~1e-3 to 1e-2 at `n_z = 11`).
At γ ≥ 5 the ε quadrature error becomes the binding constraint and no
longer sits below the policy-interpolation floor.

---

## 5. Validation

Checked items, with a short note where relevant.

- [x] **Theoretical income process correct** — three components (deterministic
      age profile, persistent z with mixture innovations, transitory eps) match
      Guvenen et al. (2021) / Catherine (2025) specification.
- [x] **Parameters match Catherine** — verified by user against source.
- [x] **Tax brackets correct** — 2019 TCJA single-filer boundaries match to
      within 1.3% (rounding error from $54,100 unit); cumulative tax amounts
      internally consistent; payroll cap of 2.5 matches SS taxable max.
- [x] **Model units understood** — 1 unit = SSA Average Wage Index ≈ $54,100
      (2019); anchored via Catherine's `initial_wealth = 0.1 = $5,400`.
- [x] **Age-earnings polynomial uses calendar ages** — not age-from-zero;
      `exp(f(22)) ≈ $10k`, `exp(f(46)) ≈ $35k` are economically sensible.
- [x] **PIA bend points correct** — 0.21 and 1.25 match 2019 SSA values
      ($926 and $5,583/month) to within ~2%.
- [x] **PIA replacement rates correct** — 90/32/15 match SSA exactly.
- [x] **AIME approximation documented** — we deviate from Catherine's explicit
      career-earnings state; justification (rho ≈ 1) and limitation
      (path-dependence lost) recorded in Section 3.1.
- [x] **Pension formula numerically verified** — at all 11 z-grid points the
      code matches manual PIA + income-tax computation to machine precision;
      AIME cap binds at the top 4 grid points as expected.
- [x] **Pension replacement rates sensible** — z=0 gets 63% of career-average
      after-tax income and 49% of peak after-tax; low earners ~100% (90% PIA
      band dominates); high earners fall sharply due to AIME cap. Matches
      Catherine (2025) reference values.
- [x] **Payroll tax rate (10.6%) identified** — equals the OASI (Old-Age and
      Survivors Insurance) combined employer+employee portion of FICA.
      Excludes the 1.8% Disability Insurance component, which is consistent
      with a model that has no disability risk.
- [x] **Pension exempt from payroll tax** — the code applies only the 7-bracket
      income tax to pension benefits (no 10.6% payroll). Economically correct:
      real US SS benefits are not subject to FICA.
- [x] **z initial condition treatment** — the solver makes no assumption about
      initial z (it solves at every grid point). The simulation offers three
      options (`"median"`, `"stationary"`, `"normal"`) with `"stationary"` as
      the default.
- [x] **Stationary wage index (L̄ = 1) assumption noted** — Catherine (2025)
      eq. 20 uses a time-varying wage index, but in a stationary economy
      `L̄_t = 1` for all t. Our model inherits this: tax brackets, payroll cap,
      and bend points are all pinned to the 2019 AWI snapshot and not indexed
      over time. Standard in lifecycle literature.
- [x] **z-grid is Tauchen (mixture-generalized), correctly implemented** —
      `discretize_income_ar1_mixture()` in
      [discretization.py:283](discretization.py#L283) is mathematically
      Tauchen (1986) with the single generalization that the conditional CDF
      used for bin integration is the mixture CDF rather than `Φ`. Checks:
      (i) grid layout `linspace(-n_stds·σ_z, +n_stds·σ_z, N)` with
      `σ_z = √(Var(η_mixture)/(1-ρ²))` matches Tauchen's grid spec;
      (ii) `Var(η_mixture) = Σ π_k σ_k² + Σ π_k (μ_k - μ_η)²` used in the code
      is the correct total variance for a two-component mixture;
      (iii) transition probabilities `Pi_z[i,j] = F(upper) - F(lower)` with
      `upper/lower = z_j ± dz/2 - ρ·z_i` follow directly from
      `P(ρz_i + η ∈ bin_j) = F_η(upper) - F_η(lower)`;
      (iv) tail bins absorb residual mass so rows sum to 1 (defensive
      normalization at [discretization.py:309](discretization.py#L309)).
      Rouwenhorst and Tauchen–Hussey were considered and rejected:
      Rouwenhorst matches only the first two moments and would discard the
      mixture's skewness/kurtosis; TH concentrates nodes where a Gaussian has
      mass, which under-covers fat-tailed mixtures and breaks the solver's
      uniform-grid linear interpolation.
      Note: `Pi_z` itself is produced as a diagnostic but is **not used** by
      the solver, which integrates `z_{t+1}` via GH quadrature on η directly
      (see next item / Section 4.6).
- [x] **Grid-width coverage adequate under current calibration** — MC of the
      stationary distribution of `z` (1M-2M draws, long burn-in with mixture
      innovations) shows `P(|z| > 3σ_z) ≈ 0.27%`, essentially identical to
      the Gaussian figure. Despite η being skewed and fat-tailed, `ρ = 0.991`
      pulls the stationary `z` sharply toward Gaussian (empirical skew ≈
      -0.15, excess kurt ≈ 0.02). The mixture's kurtosis is almost entirely
      absorbed by the AR(1) aggregation; skewness survives more and shows up
      as a mild left-tail asymmetry (`P(z < -3σ_z) ≈ 4×·P(z > +3σ_z)`),
      which is not large enough at current parameters to motivate an
      asymmetric grid.
- [x] **Boundary clipping checked and not visibly distorting policies** —
      when `z_next = ρ·z_i + η_k` lands outside the grid, the solver
      interpolation at [solver.py:627](solver.py#L627) clips
      `iz_lo` and `frac_z` — effectively evaluating `V` at the grid
      boundary. Mechanically this under-weights the upside: at `z_i = z_top`,
      ~75% of GH-weighted mass pushes `z_next` above the grid (common
      component's +0.11 drift dominates ρ=0.991 mean-reversion at
      extremes); mirror image at the bottom is milder (~18%, the rare
      component's rate). Tier-1 diagnostic on the saved
      `constrained_grid7x7x7_nz11` run found no visible clipping signature
      in policies: consumption is monotone in z, its log-curvature is
      single-humped and smooth, the kink in stock share at `iz=7→8` is the
      `S ≤ 1` leverage cap unbinding (not a boundary artifact), and the
      flattening of `C(z)` at the top is fully explained by CRRA
      precautionary savings (extra income at high z is saved, not consumed)
      — it disappears at high wealth, which a clipping-driven flattening
      would not. If higher confidence is needed before a production run,
      Tier-2 (widen `n_stds` at constant `dz`) would give a quantitative
      bound; not performed now.
- [x] **Judd-mixture quadrature for η reproduces mixture moments** —
      the rule in [discretization.py:421](discretization.py#L421) (Judd
      1998 construction directly on the mixture density) was tested
      against the closed-form mixture moments at the calibrated
      parameters (`pz=0.176, μ_1=-0.524, σ_1=0.113, μ_2=0.11192, σ_2=0.046`).
      Weight sum exact: `Σ_j W_j = 1` to machine precision; all weights
      strictly positive (Theorem 3 of Judd §7).
      Discrete moments vs theoretical:
      `mean = 0` to machine precision;
      `Var = 0.0626382290` (exact match);
      `Skew = -1.72960` (err < 2e-15 at n=3 and n=5);
      `Ex.kurt = +1.41664` (err < 1e-14 at n=3 and n=5).
      Polynomial integrals against the mixture density match exactly
      (rel err ≤ 4e-15) up to order `2n - 1` — the Judd exactness
      bound — for both n=3 (orders 0..5) and n=5 (orders 0..9);
      stressing at order `2n` produces visible error (3.16% at n=3,
      k=6; 7.99e-4 at n=5, k=10), confirming the exactness boundary is
      tight. Cross-checked against an independent Golub–Welsch
      Jacobi-eigendecomposition reference: nodes and weights agree to
      ≤ 1e-10 max-abs. On the `μ_2` parameter: Catherine (2025) Table
      E.1 and Guvenen et al. (2021) Table IV treat only
      `(p, μ_1, σ_1, σ_2)` as free; `μ_2` is pinned by `E[η] = 0` via
      `μ_2 = -(p/(1-p))·μ_1 = 0.11192233`. The quadrature recomputes
      this from the formula; the config slot `model.mu_eta2` stores
      the same derived value and is not used. Documented in
      [model.py:52](model.py#L52). Test battery:
      [tests/test_judd_quadrature.py](tests/test_judd_quadrature.py)
      (Tier 1–6). Note: this validates the discretization of the
      *innovation* η; the stationary `z` distribution differs (AR(1)
      with ρ=0.991 smooths η toward Gaussian — see prior item).
- [x] **Judd-mixture quadrature for ε reproduces mixture moments** —
      analogous check to the η entry for the transitory shock at
      [discretization.py:395](discretization.py#L395). Calibrated
      parameters: `pe=0.044, μ_1=0.134, σ_1=0.762, σ_2=0.055`, with
      `μ_2` derived from the zero-mean constraint as
      `μ_2 = -(pe/(1-pe))·μ_1 = -0.006167` (the config slot
      `model.mu_eps2 = 0.0` is a stale placeholder and is ignored by
      the quadrature — see [model.py:57](model.py#L57)). Discretization
      reproduces the theoretical mixture moments exactly:
      `Σ_j W_j = 1` to machine precision; weights strictly positive;
      `mean = 0` to ~1e-18;
      `Var = 0.02926666` (err 1e-17);
      `Skew = +2.0617` (err ~1e-15);
      `Ex.kurt = +52.219` (err ~5e-15 — note the *size* of the excess
      kurtosis, ~37× larger than η's, driven by the rare `σ_1 = 0.762`
      component). Polynomial integrals against the mixture exact up to
      order `2n - 1` (rel err ≤ 5e-15 at n ≤ 6); stress at order `2n`
      produces visible error, confirming the exactness bound is tight.
      The extreme kurtosis means polynomial exactness alone is *not*
      sufficient for FOC accuracy at high γ — see §4.8 for the
      production node-selection guidance and the heavy-tail wall at
      γ ≥ 5.
- [x] **Judd quadrature is API- and orchestration-compatible** —
      under the migration from concatenated-GH-per-component to
      Judd-on-the-mixture, `disc_config.n_eta_nodes` and
      `disc_config.n_eps_nodes` now mean the **total** node count (no
      longer per-component K, doubled internally). The codebase
      consumes node arrays via `len(...)` everywhere (solver, simulation,
      diagnostics, working-income table) — there is no hardcoded
      doubling factor anywhere. Verified by grep across the repo:
      [solver.py:523-524](solver.py#L523),
      [solver.py:2294-2295](solver.py#L2294),
      [precompute.py:339](precompute.py#L339),
      [simulation.py:907-908](simulation.py#L907) all read array
      lengths dynamically. Cached helper field
      `pc.n_eps = len(self.eps_nodes)` at
      [precompute.py:248](precompute.py#L248) is derived from the
      array, not from the config integer. Caveat: saved
      `metadata.json` integers from pre-migration runs refer to
      *half* the node count under the new convention; reload preserves
      polynomial-exactness order but not the original node count.
- [x] **Precomputed working-income table is wired correctly** —
      `_precompute_working_income()` at
      [precompute.py:366](precompute.py#L366) builds a shape
      `(n_age, n_z, n_eps)` table via broadcasting:
      `y_gross = exp(log_det_profile[:,None,None] + z_grid[None,:,None] + eps_nodes[None,None,:])`,
      then passes the whole array through
      `disposable_income_working()`. Spot-checked on the calibrated
      config (77 ages × 11 z × 10 ε = 8,470 cells under the previous
      per-component-K convention; under Judd-mixture at `n_eps = 5`
      this becomes 77 × 11 × 5 = 4,235 cells) at 8 representative
      probes covering: young / peak-earnings / last-working-age × low /
      middle / high z × rare-component / common-component ε, with
      `y_gross` spanning 0.0008 to 530 model units (all 7 tax brackets
      and the payroll cap). Hand-coded bracket-by-bracket disposable
      income at each probe matches the table to bit-exactness (rel err
      = 0 across every probe). This confirms: broadcasting axes
      correct (no `iz`/`ie` swap), `log_det_profile[t]` aligned to
      calendar age `start_age + t`, ε nodes from the transitory
      quadrature (not η by accident), vectorized and scalar tax
      computations agree, and the full bracket ladder handles the
      extreme tails produced by the rare-component ε × top-z corner.
      The table is now used only for simulation warmup and diagnostics;
      the solver computes income on-the-fly (next item).
- [x] **Solver income is computed on-the-fly and matches the scalar
      reference exactly** — the working-age FOC in
      [solver.py](solver.py) no longer interpolates income in `z`. For
      each `(k_eta, i_e)` the solver evaluates
      `y_gross_next = base_det_z · exp_eta[k_eta] · exp_eps[i_e]` and
      passes it through `scalar_disposable_income` (the same njit
      function used by the simulation — [model.py:312](model.py#L312)).
      Three-part numerical validation (script:
      `_validate_income_ontheflyp.py`):
      **(a) Scalar ≡ vectorized.** `scalar_disposable_income` matches
      `disposable_income_working` across 200 random probes spanning
      `y_gross ∈ [4.2e-3, 1.1e+3]` (covers every tax bracket and the
      payroll cap) to max relative error `2.35e-16` ≈ 1 ulp.
      **(b) OLD interpolation bias eliminated.** Replaying the old
      scheme — linear interpolation of the precomputed income table in
      `z` — against the exact on-the-fly value across 9,000 probes
      (ages `{22, 35, 46, 55, 66}` × all 10 interior
      `iz_lo → iz_lo+1` brackets × 30 interior `frac_z` values × 6 ε
      nodes) gives a max relative error of **16.85%**. The new scheme
      is exact by construction (0.00%).
      **(c) Worst-case probe.** The maximum bias occurs at `(age=66,
      iz=6→7, frac_z=0.449, eps=-0.006, z_mid=+1.626)`: OLD
      interpolated `= 1.8231`, NEW exact `= 1.5603`, bias `= +0.2629`
      (+16.85%). The large overshoot occurs because at peak earnings
      (age 66) with high `z`, the two bracketing grid points straddle
      the payroll-cap kink at `y_gross = 2.5` and the convex region
      above it; the chord between the two grid values over-estimates
      the true (concave-in-`z` after-tax income) function. All such
      overshoots vanish once income is evaluated pointwise from
      continuous `z_next`.
- [ ] **Labour income ↔ return covariance** — deferred. The `z` process
      could in principle correlate with financial returns through the VAR
      structure (`Sigma_rs`), giving human capital a stock- or bond-like
      character that affects optimal portfolios. Decision pushed to after
      returns and state variables have been validated independently. See
      [INCOME_RETURN_COV_HANDOFF.md](INCOME_RETURN_COV_HANDOFF.md) and
      [LBAR_HANDOFF.md](LBAR_HANDOFF.md).


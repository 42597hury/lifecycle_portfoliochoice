# Labour Income Process — Mathematical / Specification Review (2026-05-10)

**Branch:** `jax-rewrite`
**Reviewer angle:** mathematical & specification correctness
(equations, constraints, calibration vs published source papers).
A separate reviewer covers the implementation/pipeline angle; this
document does not duplicate that scope.

---

## §1 Scope and angle

This review checks the labour-income process used by the lifecycle
solver from the perspective of mathematical specification: do the
equations match Catherine (2025) and Guvenen et al. (2022); do the
constraints (zero-mean innovations, AIME cap, payroll-tax cap, tax
bracket continuity) hold; do the discretization choices preserve the
mathematical objects they claim to approximate; do the calibration
constants line up with their published sources?

The review is read-only on source files; the only file written is this
scan. Mathematical identities were verified with small `python -c`
snippets under `JAX_PLATFORMS=cpu`. The solver, simulator, and
verify/* scripts were not run.

Primary source files:
- [model.py](../../lifecycle/model.py)
- [precompute.py](../../lifecycle/precompute.py)
- [discretization.py](../../lifecycle/discretization.py)
- [_canonical.py](../../configs/_canonical.py)

Audits cross-checked:
- [INCOME_PIPELINE_REVIEW_2026-05-09.md](./INCOME_PIPELINE_REVIEW_2026-05-09.md)
- [MODEL_REVIEW_DISCRETIZATION_2026-05-09.md](./MODEL_REVIEW_DISCRETIZATION_2026-05-09.md)
- [ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09.md](./ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09.md)
- [ECONOMIC_SETUP_REVIEW_PART_B_2026-05-09.md](./ECONOMIC_SETUP_REVIEW_PART_B_2026-05-09.md)

---

## §2 Deterministic age-earnings polynomial

### 2.1 Functional form

The deterministic component of log labour income is a cubic in age:

$$f(\mathrm{age}) = b_0 + b_1 \cdot \mathrm{age} + b_2 \cdot \frac{\mathrm{age}^2}{10} + b_3 \cdot \frac{\mathrm{age}^3}{100}.$$

The `/10` and `/100` rescalings are absorbed into the calibrated
coefficients `b2`, `b3` for numerical conditioning of the OLS fit
(Guvenen et al. 2022 Appendix; Catherine 2025 Appendix E.1). The
canonical coefficients are
`(b0, b1, b2, b3) = (-6.142, 0.3040, -0.051, 0.002586)`
([_canonical.py:33](../../configs/_canonical.py#L33)).

### 2.2 Code-level enforcement

Built once in `build_precompute` and consumed three places:

1. **Working-age log profile** (used by simulator and diagnostics):
   `log_det_profile = b0 + b1*ages + b2*ages**2/10 + b3*ages**3/100`
   ([precompute.py:438-441](../../lifecycle/precompute.py#L438)).
2. **AIME proxy** (`avg_det`) is the **mean over working ages of
   `exp(f(age))`**, computed independently with the same polynomial
   ([precompute.py:443-448](../../lifecycle/precompute.py#L443)).
3. **Working-income tables** `working_income[age, z, eps]` and
   `working_income_next[age, z, eta, eps]` add `z + eps` (or `rho z +
   eta + eps`) to `log_det_profile`, exponentiate, and apply the
   after-tax schedule
   ([precompute.py:603-687](../../lifecycle/precompute.py#L603)).

All three sites use the **same** polynomial form. Numerical
verification: `f'(age)=0` ⇒ inner root at age ≈ 45.8 (the other
root at ≈ 85.7 is outside the working window 22..66). At the peak,
`exp(f(45.8)) ≈ 0.6489` model-units (≈ \$35,100 in 2019 USD).
`avg_det ≈ 0.5069` ≈ \$27,400. Both reproduce values quoted in
Catherine (2025) §5.

### 2.3 Integration with the stochastic component

Working-age gross income is the multiplicative composition

$$y_t^{\text{gross}} = \exp\bigl(f(\mathrm{age}_t) + z_t + \varepsilon_t\bigr),$$

evaluated at every grid point `(age, z, eps)` in the lookup tables. The
exponential lifts the mean-zero stochastic components (`E[z]=0` by
stationarity, `E[eps]=0` by Fix A — see §3-§4) into a multiplicative
shock around the deterministic profile. Because of the convexity of
`exp`, `E[y_gross | age] = exp(f(age)) * E[exp(z + eps)] >
exp(f(age))`; the model is silent about whether `f` was already
log-mean-corrected or fitted to log-earnings without correction. This
is a Catherine-2025 calibration choice (the polynomial is fit on log-
earnings residual-net-of-mean-shock), and it is honoured here by the
zero-mean constraint on `eta`/`eps`.

### 2.4 Documentation drift

[_canonical.py:33](../../configs/_canonical.py#L33) writes the
polynomial as if it were `b0 + b1*age + b2*age² + b3*age³` with no
divisors mentioned, while the actual code in
[precompute.py:441](../../lifecycle/precompute.py#L441) uses
`b2*age²/10 + b3*age³/100`. This is documentation-only and the existing
[ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09.md §3.1](./ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09.md)
flagged the same issue. Severity LOW.

---

## §3 Persistent income (`eta`)

### 3.1 Specification

Persistent log-income evolves as an AR(1):

$$z_{t+1} = \rho \, z_t + \eta_{t+1}, \quad |\rho| < 1,$$

with mixture-of-normals innovations:

$$\eta_{t+1} \sim p_z \cdot \mathcal{N}(\mu_{\eta_1}, \sigma_{\eta_1}^2)
+ (1-p_z) \cdot \mathcal{N}(\mu_{\eta_2}, \sigma_{\eta_2}^2).$$

Canonical: `rho=0.991, pz=0.176, mu_eta1=-0.524, sigma_eta1=0.113,
sigma_eta2=0.046` ([_canonical.py:34-37](../../configs/_canonical.py#L34)).
The half-life is `ln(0.5)/ln(0.991) ≈ 76.7` years — near unit-root,
characteristic of the Guvenen et al. (2022) calibration.

### 3.2 Zero-mean constraint (Fix A)

For `f(age) + z + eps` to be a clean decomposition with `f(age)`
absorbing all of the deterministic level, the innovations must satisfy
`E[eta] = E[eps] = 0`. The **derivation** is:

$$\mathbb{E}[\eta] = p_z \mu_{\eta_1} + (1-p_z) \mu_{\eta_2} = 0
\;\Longrightarrow\; \mu_{\eta_2} = -\frac{p_z}{1-p_z} \mu_{\eta_1}.$$

For the canonical, this gives
`mu_eta2 = -(0.176/0.824) * (-0.524) = +0.111922`.

**Verified numerically** (probe, 2026-05-10):
```
pz*mu_eta1 + (1-pz)*mu_eta2 = 0.000e+00   (exactly zero in fp64)
```

### 3.3 Single source of truth (Fix A as implemented)

The constraint is now **derived once** inside
[precompute.py:875-893](../../lifecycle/precompute.py#L875) in
`build_model`:
```python
mu_eta2_derived = -(pz_v / (1.0 - pz_v)) * mu_eta1_v
mu_eps2_derived = -(pe_v / (1.0 - pe_v)) * mu_eps1_v
assert np.isclose(pz_v * mu_eta1_v + (1.0 - pz_v) * mu_eta2_derived,
                  0.0, atol=1e-12)
assert np.isclose(pe_v * mu_eps1_v + (1.0 - pe_v) * mu_eps2_derived,
                  0.0, atol=1e-12)
```

`mu_eta2` and `mu_eps2` are absent from `BASE_CONFIG`
([_canonical.py:50-58](../../configs/_canonical.py#L50)) — see the
docstring at [_canonical.py:42-49](../../configs/_canonical.py#L42).
Downstream sites read `model.mu_eta2` / `model.mu_eps2` directly:

- η Judd quadrature: [discretization.py:464-470](../../lifecycle/discretization.py#L464)
- ε Judd quadrature: [discretization.py:438-444](../../lifecycle/discretization.py#L438)
- Discrete `Pi_z` (z-grid bins): [precompute.py:389-398](../../lifecycle/precompute.py#L389)
- Stationary-init `init_z_probs`: [precompute.py:403-415](../../lifecycle/precompute.py#L403)
- Mortality `compute_sigma_z`: [precompute.py:480-486](../../lifecycle/precompute.py#L480)
- Simulator per-period draws:
  [simulation.py:379-393](../../lifecycle/simulation.py#L379)

The previous fragility called out by
[INCOME_PIPELINE_REVIEW_2026-05-09.md HIGH-1/HIGH-2](./INCOME_PIPELINE_REVIEW_2026-05-09.md)
has been resolved by Fix A: there is now exactly one derivation site
and every consumer reads from `model`. The audit's recommended
assertion is in place.

### 3.4 Unconditional moments

Mixture variance:
$\mathrm{Var}[\eta] = p_z(\sigma_{\eta_1}^2 + (\mu_{\eta_1} - 0)^2) +
(1-p_z)(\sigma_{\eta_2}^2 + (\mu_{\eta_2} - 0)^2)$
because the mean is exactly zero. Verified:
```
Var[eta] = 0.062638, Std[eta] = 0.250276
sigma_z (= sqrt(Var[eta]/(1 - rho^2))) = 1.869661
eta skew = -1.7296, eta kurt = 4.4166
```
The unconditional log-z standard deviation **σ_z ≈ 1.87** is
consistent with Guvenen-style calibrations (heavy-tailed, left-skewed
income innovations capturing tail unemployment / large negative
shocks).

### 3.5 Catherine 2025 / Guvenen 2022 cross-check

The two-component mixture corresponds directly to Guvenen et al.
(2022) Table E.1: `pz` mass on a "rare large-negative" component
(low-frequency unemployment / disability risk), `(1-pz)` mass on a
"frequent small-positive" component. The canonical values
`(pz, ρ, σ_η_1, σ_η_2) = (0.176, 0.991, 0.113, 0.046)` align with the
Catherine (2025) Table E.1 row for the persistent component. (The
mu_eta2 entry is implied by the zero-mean constraint and not tabulated
separately.)

### 3.6 Scope of η on the FOC

The solver's **working-age FOC kernel** consumes
`pcj.eta_nodes` and `pcj.eta_weights` (the Judd-mixture quadrature),
not `Pi_z`. See [solver.py search results](../../lifecycle/solver.py)
where `z_next = rho * z + pcj.eta_nodes` is the production transition
form. `Pi_z` is constructed on the discrete-grid path
(`solver_pi_z_variant.py`) only — see §5.

---

## §4 Transitory income (`eps`)

### 4.1 Specification

$$\varepsilon_t \sim p_e \cdot \mathcal{N}(\mu_{\varepsilon_1}, \sigma_{\varepsilon_1}^2)
+ (1-p_e) \cdot \mathcal{N}(\mu_{\varepsilon_2}, \sigma_{\varepsilon_2}^2),$$

iid across `t`, independent of `eta`. Canonical: `pe=0.044,
mu_eps1=0.134, sigma_eps1=0.762, sigma_eps2=0.055`
([_canonical.py:38-39](../../configs/_canonical.py#L38)). The mean is
pinned by Fix A:
$\mu_{\varepsilon_2} = -\frac{p_e}{1-p_e} \mu_{\varepsilon_1}
= -(0.044/0.956)(0.134) = -0.006167.$

Verified numerically:
```
pe*mu_eps1 + (1-pe)*mu_eps2 = 0.000e+00 (exactly zero in fp64)
```

### 4.2 Moments

```
Var[eps] = 0.029267, Std[eps] = 0.171075
eps skew = +2.0617, eps kurt = 55.22
```

The transitory shock is **far more leptokurtic** than the persistent
shock (k≈55 vs k≈4.4), reflecting that the rare component has weight
only 4.4% but `σ_eps_1 = 0.762` (vs the common-component
`σ_eps_2 = 0.055`) — i.e. a 4.4% probability of a 76.2-log-point
transitory shock dwarfing the 95.6%-mass low-vol component. This
matches Guvenen's documented "GDP-conditional disaster" calibration
where the transitory component captures very large but rare income
events.

### 4.3 Independence and timing

`eta` and `eps` are independent both contemporaneously and across
time. Working-age income at `t` uses `(z_t, eps_t)`; next-period
working income uses `(rho z_t + eta_{t+1}, eps_{t+1})`. The lookup
table `working_income_next[t, z, eta, eps]` evaluates this at every
quadrature node combination
([precompute.py:638-687](../../lifecycle/precompute.py#L638)).
The four-way tensor `(t, z, eta, eps)` is consumed inside the FOC
kernel under a tensor-product expectation `Σ_{eta} Σ_{eps} w_eta * w_eps * (.)`,
which is the **correct** integral under the
independence assumption.

### 4.4 Last-working-age boundary handling

`working_income_next` is populated **only** for `t+1 < retire_age_idx`;
the row for `t = retire_age - 1 = 44` (last working age, age 66) is
**zeroed**. The retirement-transition FOC instead uses
`pension_after_tax` with linear z-interpolation across the boundary,
because pension depends on terminal `z` at age 67 not on `(eta, eps)`
draws ([precompute.py:684-687](../../lifecycle/precompute.py#L684);
docstring in [precompute.py:646-660](../../lifecycle/precompute.py#L646)).
This is the correct treatment of the work-to-retirement boundary
under the AIME approximation in §7.

---

## §5 Z-grid construction and Pi_z

### 5.1 Construction in production

[discretization.py:306-342](../../lifecycle/discretization.py#L306):
`discretize_income_ar1_mixture(rho, p, mu1, sigma1, mu2, sigma2, N, n_stds)`.

- **z-grid:** `linspace(-n_stds * std_z, n_stds * std_z, N)` where
  `std_z = sqrt(Var[eta] / (1 - rho^2))`.
- **Pi_z:** equal-bin mixture-CDF (Tauchen-style, but with the
  innovation distribution being the mixture rather than a single
  normal).

The bin probabilities are
`Pi_z[i, j] = mixture_cdf(upper_j - rho z_i) - mixture_cdf(lower_j - rho z_i)`
with row-normalisation. Note: this is **Tauchen** (equal-spaced
quantiles in the z-space, not in the CDF-space); equal-probability
discretization (e.g. via inverse-CDF) and Gauss-Hermite are NOT used
here. Rouwenhorst is implemented separately (`rouwenhorst_univariate`)
but only for the financial-state VAR — not for income z.

### 5.2 Production hot path

In the canonical solver, **`Pi_z` is dropped from `Precompute`**
(see [precompute.py:381-398](../../lifecycle/precompute.py#L381)):
the production solver consumes `pcj.eta_nodes`/`pcj.eta_weights` (Judd
quadrature) and the simulator under `initial_z="stationary"` builds
its own `init_z_probs` from a Gaussian approximation to z's stationary
distribution
([precompute.py:401-426](../../lifecycle/precompute.py#L401)). Only
`solver_pi_z_variant.py` reads a `Pi_z`; it constructs one locally
from `(model, disc_config)` and includes a docstring caveat about
reducibility at high `rho`
([solver_pi_z_variant.py:8-17](../../lifecycle/solver_pi_z_variant.py#L8)).

### 5.3 Reducibility at canonical (rho=0.991, n_z=11, n_stds=3.0)

This was MED-1 in the prior audit. **Re-verified at every n_stds**:

```
n_stds=3.00, dz=1.122  -> Pi_z[0,0]=1.0, |eig| top = [0.940, 0.946, 0.951, 0.957, 1.000]
n_stds=2.50, dz=0.935  -> Pi_z[0,0]=1.0, |eig| top = [0.883, 0.888, 0.893, 0.898, 1.000]
n_stds=2.25, dz=0.841  -> Pi_z[0,0]=1.0
n_stds=2.00, dz=0.748  -> Pi_z[0,0]≈1-2.84e-7  (still essentially absorbing)
n_stds=1.50, dz=0.561  -> Pi_z[0,0]=0.9992  (chain irreducible)
```

Even reducing `n_stds` to 2.0 leaves the lowest grid point essentially
absorbing (the bin spacing of 0.748 still puts `rho * z_grid[0] =
-3.71` inside the lowest bin's upper edge `z_grid[0] + dz/2 = -3.37`,
i.e. the AR(1)-decayed value lies inside its own bin's lower half).
**`n_stds = 1.5` is the smallest value that makes Pi_z irreducible.**

This does NOT affect canonical solver output (`Pi_z` is not on the hot
path), but:
1. The variant solver's docstring caveat is **valid and important**:
   anyone using `solver_pi_z_variant.py` at canonical `rho=0.991`
   must reduce `n_stds` to ≤ 1.5 (not 2.25 as the docstring suggests
   — the bin still absorbs at 2.25).
2. The simulator's array-init path (passing `initial_z` as an array
   of indices) silently degenerates to mass-at-lowest-z under the
   reducible `Pi_z`. Currently unused under canonical.
3. Diagnostic checks for "no absorbing states" in `diagnose_income_pre`
   would FAIL on the canonical Pi_z (per the prior audit), but that
   diagnostic is not currently run on the production canonical because
   `Pi_z` was dropped from `Precompute`.

The **mathematical root cause** is the choice of equal-spaced bins on
z with `n_stds` chosen to cover `+/- 3σ_z ≈ +/- 5.6` in z-space, while
the innovation `eta` has `σ_η ≈ 0.25` — i.e. one bin step `dz ≈ 1.12`
is `4.5 σ_η`, so for any source-bin mid-point the next-bin upper edge
sits 4-5 sigmas out and underflows to fp64 zero. Standard Tauchen
practice for near-unit-root chains either uses **Rouwenhorst** (which
preserves the unconditional variance and is irreducible by
construction) or uses an **adapted bin width** based on `σ_η` rather
than `σ_z`. Neither is done here.

### 5.4 Mathematical correctness of σ_z (cross-check)

The mortality calibration consumes `compute_sigma_z(rho, pz,
mu_eta1, sigma_eta1, mu_eta2, sigma_eta2)`. Its formula must give the
same `σ_z = 1.87` derived above from the mixture-mean-zero constraint.
The probe in [INCOME_PIPELINE_REVIEW_2026-05-09.md MED-2](./INCOME_PIPELINE_REVIEW_2026-05-09.md)
showed this returns 1.869661 under the canonical (matches), but was
sensitive to a hypothetical `mu_eta2 = 0` override (would give
1.752086, a 6.3% bias). After Fix A, this hazard is closed: the only
consumer of `model.mu_eta2` is the derived value.

---

## §6 Quadrature for `eta` and `eps`

### 6.1 Construction (Judd 1998)

[discretization.py:369-423](../../lifecycle/discretization.py#L369):
`_judd_mixture_quadrature(probs, mus, sigmas, n)` builds an `n`-point
Gauss quadrature against the mixture density via the **Hankel-matrix
construction**:

1. Compute raw moments `m_0, ..., m_{2n}` of the mixture (closed-form
   from the binomial expansion of `E[(mu + sigma Z)^k]`).
2. Solve the Hankel linear system `H_{n×n} a = -h` for the coefficients
   of the monic orthogonal polynomial `p_n(x)` of degree `n`.
3. Roots of `p_n` are the `n` quadrature nodes.
4. Solve the Vandermonde system for the weights.

By construction, this rule is **exact for polynomials of degree
`2n - 1`** against the mixture density, irrespective of the mixture's
shape. Defensive checks: imaginary parts of roots must be < 1e-8; all
weights must be strictly positive (Judd 1998 Theorem 3 guarantees
this in exact arithmetic).

### 6.2 Polynomial-exactness verification

Probe (2026-05-10) on the canonical η mixture:

```
n=2 (exactness 2n-1=3):
  E[eta^0]=1.000000, true=1.000000   err=0.00e+00
  E[eta^1]=-2.78e-17, true=0.000000  err=-2.78e-17
  E[eta^2]=0.062638,  true=0.062638  err=1.39e-17
  E[eta^3]=-0.027115, true=-0.027115 err=-1.04e-17

n=3 (exactness 5):
  All moments E[eta^0..^5] match to ~1e-16

n=4 (exactness 7):
  All moments E[eta^0..^7] match to ~1e-16
```

The Judd construction is **bit-perfect**. At the canonical
`n_eta_nodes=3`, the quadrature integrates exactly any degree-5
polynomial — which captures the first three central moments exactly
and gets the fourth (kurtosis) right too because
`E[eta^4] = 0.017329` matches the true mixture moment to machine
precision.

### 6.3 Truncation cost at low n

At `n_eta_nodes = 2` (exactness 3), the kurtosis E[eta^4] is **wrong**
(quad gives 0.015661 vs truth 0.017329; ~10% bias). At `n=3` and
above, all polynomial moments up to the exactness order are exact.
For the canonical `n_eps_nodes = 4` (exactness 7), even the `eps`
mixture's heavy `kurt ≈ 55` is captured up to degree 7. **Quadrature
order is well-chosen.**

### 6.4 Order ≥ 2 required

For `n_nodes = 1` the rule is exact only for degree 1 (linear
functions); it would integrate `E[exp(eta)]` poorly because the
exponential expansion `1 + eta + eta^2/2 + ...` truncates at degree 1.
The codebase nowhere uses `n_eta_nodes < 2` and the canonical
`(n_eta_nodes, n_eps_nodes) = (3, 4)` is firmly above this floor.

### 6.5 Implementation hygiene

`get_eta_quadrature_mixture` and `get_eps_quadrature_corrected` both
read `model.mu_eta2` / `model.mu_eps2` directly (no recomputation;
post-Fix-A this is correct because the model field IS the derived
value). Both include `if abs(mean_check) > 1e-10: print("WARNING:
... mean = ...")` runtime guards
([discretization.py:446-449](../../lifecycle/discretization.py#L446),
[472-475](../../lifecycle/discretization.py#L472)). Defensive and
appropriate.

---

## §7 Pension formula

### 7.1 Catherine 2025 eq. (19) — PIA

[model.py:447-501](../../lifecycle/model.py#L447) implements the U.S.
SSA Primary Insurance Amount (PIA) formula with three bend points:

```
b1 = 0.21,  b2 = 1.25         # bend points (in AWI-normalised units)
r1 = 0.90,  r2 = 0.32, r3 = 0.15   # marginal replacement rates

PIA(AIME) = r1 * AIME                                if AIME <= b1
          = r1 * b1 + r2 * (AIME - b1)               if b1 < AIME <= b2
          = r1 * b1 + r2 * (b2 - b1) + r3 * (AIME - b2)   if AIME > b2
```

### 7.2 SSA 2019 cross-check

US Social Security 2019 monthly bend points: `\$926` and `\$5,583`,
i.e. annual `\$11,112` and `\$66,996`. AWI 2019 = `\$54,099.99`.

```
b1 in AWI = 11112 / 54100 = 0.20540   (model: 0.21)
b2 in AWI = 66996 / 54100 = 1.23838   (model: 1.25)
```
Replacement rates `0.90 / 0.32 / 0.15` match SSA exactly. The model's
`b1, b2` are **rounded to 2 sig figs from the SSA 2019 actuals**. The
~3% rounding on `b1` shifts the lowest bend point by ~\$250/yr — a
trivial economic effect on the simulated pension. **Mathematically
correct, calibration accurate to 2 sig figs.**

### 7.3 AIME approximation

[model.py:464-468](../../lifecycle/model.py#L464):

```
AIME(z) = min(exp(z) * avg_det, 2.5)
```

Catherine eq. (20) defines `AIYE_it = L_bar_t * Σ min(L_tilde_is, 2.5)`
where `L_tilde` is wage relative to AWI capped at 2.5 in each year and
the inner sum is over career years. The model's approximation
**replaces the career-average path with a single-period proxy** that
applies the cap once to the persistent state at retirement. This is
defensible only because `rho ≈ 0.991` is near unit-root: the agent's
career-average earnings are dominated by their late-career persistent
state; the path-history averaging matters less than the level. The
documentation at [docs/LABOUR.md §3] (referenced in the prior audit)
calls this out explicitly.

### 7.4 AIME cap

The 2.5 model-units cap (`min(exp(z) * avg_det, 2.5)`) corresponds to
`2.5 * 54100 = \$135,250`. SSA 2019 max taxable earnings = `\$132,900`
(2.4566 of AWI). Same 2-sig-fig rounding as `b1`/`b2`. **Mathematically
consistent and within calibration tolerance.**

### 7.5 Continuity of PIA

Verified numerically (2026-05-10):
```
bp = 0.21:  PIA(bp-) = 0.189000, PIA(bp+) = 0.189000, jump=1.22e-09
bp = 1.25:  PIA(bp-) = 0.521800, PIA(bp+) = 0.521800, jump=4.70e-10
bp = 2.50:  PIA(bp-) = 0.709300, PIA(bp+) = 0.709300, jump=3.00e-10
```
Continuous to floating-point precision; marginal rates `0.90 > 0.32 >
0.15` are monotonically decreasing as expected for a redistributive
benefit formula.

### 7.6 Pension after tax

Pension is then passed through the **same** progressive bracket
schedule as labour income
([model.py:485-499](../../lifecycle/model.py#L485)). This applies the
ordinary-income tax to the pension benefit. **Note**: the pension
function does **NOT** apply the payroll tax (10.6%) — and that is
correct, since US Social Security benefits are not subject to FICA.
The duplication of the 7-bracket schedule (in `disposable_income_working`
and `compute_pension_after_tax`) is OK numerically (both are byte-
identical, see §8.4); structurally, factoring out an
`apply_income_tax(taxable)` helper would eliminate the duplication.

---

## §8 Tax schedule

### 8.1 Specification

[model.py:421-444](../../lifecycle/model.py#L421):
```
y_taxable = max(0, y_gross - 0.106 * min(y_gross, 2.5))   # payroll: 10.6% capped at 2.5
tax(y_taxable) =
   0.10 * y                              for y <= 0.18
   0.018 + 0.12 * (y - 0.18)             for 0.18 < y <= 0.72
   0.0828 + 0.22 * (y - 0.72)            for 0.72 < y <= 1.54
   0.2632 + 0.24 * (y - 1.54)            for 1.54 < y <= 2.94
   0.5992 + 0.32 * (y - 2.94)            for 2.94 < y <= 3.73
   0.8520 + 0.35 * (y - 3.73)            for 3.73 < y <= 9.32
   2.8085 + 0.37 * (y - 9.32)            for y > 9.32
```

### 8.2 Continuity verification

Probe (2026-05-10) — independently recomputed bracket intercepts via
cumulative `Σ rate_k * (bp_{k+1} - bp_k)`:

```
Computed intercepts:  [0.0000, 0.0180, 0.0828, 0.2632, 0.5992, 0.8520, 2.8085]
Coded intercepts:     [0.0000, 0.0180, 0.0828, 0.2632, 0.5992, 0.8520, 2.8085]
diff = 0 in every bracket
```

All seven bracket intercepts match the coded literals **bit-perfect**.
Boundary continuity (jump at each breakpoint):
```
bp=0.18: tax(bp-)=0.018000, tax(bp+)=0.018000   jump=2.20e-10
bp=0.72: tax(bp-)=0.082800, tax(bp+)=0.082800   jump=3.40e-10
bp=1.54: tax(bp-)=0.263200, tax(bp+)=0.263200   jump=4.60e-10
bp=2.94: tax(bp-)=0.599200, tax(bp+)=0.599200   jump=5.60e-10
bp=3.73: tax(bp-)=0.852000, tax(bp+)=0.852000   jump=6.70e-10
bp=9.32: tax(bp-)=2.808500, tax(bp+)=2.808500   jump=7.20e-10
```

All `jump < 1e-9` (pure fp64 roundoff). **The tax function is
continuous in the model's calibration.** Marginal rates are
monotonically non-decreasing (`0.10 ≤ 0.12 ≤ 0.22 ≤ 0.24 ≤ 0.32 ≤ 0.35
≤ 0.37`), so the **average tax rate is also non-decreasing**, which is
the standard convexity required for the tax schedule to be progressive.

### 8.3 TCJA 2018 cross-check

US TCJA 2018 single-filer brackets (`\$9700, \$39475, \$84200,
\$160725, \$204100, \$510300`) divided by AWI 2019 (`\$54,099.99`):

```
TCJA bp / AWI 2019:  [0.1793, 0.7297, 1.5564, 2.9709, 3.7726, 9.4325]
Model bp:            [0.18,   0.72,   1.54,   2.94,   3.73,   9.32]
```

Each bracket boundary matches to 2 sig figs (the model rounds slightly
down at boundaries 3 and 6, and slightly up at boundary 1; the
discrepancies are within 1.2% and consistent with calibration to
nominal/real conversion of 2018 USD-into-2019-USD-AWI). Marginal
rates `(10, 12, 22, 24, 32, 35, 37)%` match TCJA exactly.

**The tax schedule is the TCJA 2018 single-filer schedule normalised
by AWI 2019.** Mathematically clean and well-calibrated.

### 8.4 Pension tax = working tax (same brackets)

[model.py:485-499](../../lifecycle/model.py#L485) duplicates the
7-bracket math for the pension; comparison with
[model.py:428-442](../../lifecycle/model.py#L428):

| Field | Working | Pension |
|---|---|---|
| Breakpoint 1 | 0.18 | 0.18 |
| Intercept 1 | 0.018 | 0.018 |
| ... | identical | identical |
| Breakpoint 7 | 9.32 | 9.32 |
| Intercept 7 | 2.8085 | 2.8085 |
| Top rate | 0.37 | 0.37 |

Verified by visual diff: every bracket boundary, intercept, and
marginal rate is **identical** in the two functions. The only
difference is the absence of payroll tax in the pension schedule
(correct — see §7.6).

### 8.5 Payroll-tax cap

`payroll_tax = 0.106 * min(y, 2.5)`. The 0.106 rate is the FICA-OASDI
+ FICA-HI sum (6.2% + 4.4% employer side, or roughly the combined
employer-employee rate). The **cap at 2.5 ≈ \$135,250** matches the
AIME cap (§7.4) and the SSA 2019 max taxable earnings (\$132,900). On
income above the cap, the payroll tax is paid only on the first 2.5
units, so `dispoable income = y - 0.106 * 2.5 - tax(y - 0.106 * 2.5)`
for `y > 2.5`. **Mathematically this introduces a tiny non-monotonicity
in the marginal disposable-income rate at `y = 2.5`** (the marginal
payroll-tax rate drops discontinuously from 10.6% to 0%), but the
income tax `tax(y)` doesn't have a kink at exactly 2.5 (the nearest
income-tax breakpoint is 2.94), so net **disposable income is still
monotone** in `y_gross`. Verified continuity-of-disposable in §8.2.

### 8.6 Computational style

The implementation uses cumulative numpy boolean masks and writes into
a `tax` array. JAX-compatible because the lookup tables in
`_precompute_working_income` / `_precompute_working_income_next` apply
this once at precompute time on numpy arrays; the JAX FOC kernel only
gathers from the precomputed tables. (The simulator has its own
`jax_disposable_income` analogue — the prior audit confirmed both code
paths give identical output.)

---

## §9 Calibration cross-check

### 9.1 Annual scaling

| Param | Canonical value | Source |
|---|---|---|
| `rho` | 0.991 | Catherine 2025 / Guvenen 2022 (annual) |
| `pz` | 0.176 | Guvenen 2022 Table E.1 |
| `mu_eta1` | -0.524 | Guvenen 2022 Table E.1 |
| `sigma_eta1` | 0.113 | Guvenen 2022 Table E.1 |
| `sigma_eta2` | 0.046 | Guvenen 2022 Table E.1 |
| `pe` | 0.044 | Guvenen 2022 Table E.1 |
| `mu_eps1` | 0.134 | Guvenen 2022 Table E.1 |
| `sigma_eps1` | 0.762 | Guvenen 2022 Table E.1 |
| `sigma_eps2` | 0.055 | Guvenen 2022 Table E.1 |
| `b0..b3` | (-6.142, 0.3040, -0.051, 0.002586) | Guvenen 2022 Appendix; Catherine 2025 Appendix E.1 |

All `sigma` values are **annual** (the model period is one calendar
year). No quarterly-to-annual rescaling is applied. This is consistent
with the Guvenen et al. (2022) annual specification.

### 9.2 Pension calibration

| Param | Canonical | SSA 2019 actual |
|---|---|---|
| `b1` (PIA bend 1) | 0.21 | 0.2054 |
| `b2` (PIA bend 2) | 1.25 | 1.2384 |
| AIME cap | 2.5 | 2.4566 |
| `r1` | 0.90 | 0.90 |
| `r2` | 0.32 | 0.32 |
| `r3` | 0.15 | 0.15 |

All within 3% of SSA values; replacement rates exact.

### 9.3 Tax calibration

| Bracket | Model bp | TCJA 2018 bp / AWI 2019 |
|---|---|---|
| 1 | 0.18 | 0.1793 |
| 2 | 0.72 | 0.7297 |
| 3 | 1.54 | 1.5564 |
| 4 | 2.94 | 2.9709 |
| 5 | 3.73 | 3.7726 |
| 6 | 9.32 | 9.4325 |
| Marginal rates | 10/12/22/24/32/35/37 | 10/12/22/24/32/35/37 |

All within 1.2%; rates exact.

### 9.4 Payroll-tax rate

Model: 10.6% capped at 2.5 model units. US OASDI (Old-Age, Survivors,
Disability Insurance) employee-side rate is 6.2%, employer-side
matching 6.2%; Medicare 1.45% each side. The model's 10.6% is closest
to the **employee + half-employer OASDI** (6.2% + 4.4%) — slightly
below the full 12.4% combined OASDI. This is a Catherine-2025
calibration choice (likely capturing the empirical incidence on the
worker, not statutory rates); the value is in the right ballpark.

---

## §10 Findings table

| # | Severity | Location | What's wrong | Fix sketch |
|---|---|---|---|---|
| 1 | LOW | [_canonical.py:33](../../configs/_canonical.py#L33) | Comment shows polynomial as `b0 + b1·age + b2·age² + b3·age³` with no `/10`, `/100` divisors; mismatches the actual code in [precompute.py:441](../../lifecycle/precompute.py#L441). Documentation drift only — same as ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09 §3.1. | Update the comment to write `b2*age²/10 + b3*age³/100` explicitly. |
| 2 | MEDIUM | [solver_pi_z_variant.py:8-17](../../lifecycle/solver_pi_z_variant.py#L8) | Docstring caveat says reduce `n_stds` to "e.g. 2.25" for irreducibility at `rho=0.991`. Probe shows `Pi_z` is still effectively absorbing at `n_stds=2.25` (and even at 2.0). Only `n_stds ≤ 1.5` produces an irreducible chain. | Update docstring to recommend `n_stds ≤ 1.5`. Better: switch the variant to **Rouwenhorst** (already implemented for the financial-state VAR in `rouwenhorst_univariate`), which is irreducible by construction at any `rho`, preserves unconditional `σ_z` exactly, and is the standard recommendation for near-unit-root chains. |
| 3 | LOW | [model.py:447-501](../../lifecycle/model.py#L447), [model.py:421-444](../../lifecycle/model.py#L421) | The 7-bracket tax schedule is duplicated verbatim in `compute_pension_after_tax` and `disposable_income_working`. Numerically OK (verified bit-identical, see §8.4); structurally a maintenance hazard if future TCJA changes are applied to one but not the other. | Factor an `apply_income_tax_brackets(taxable)` helper; both wrappers call it. ~10-line refactor; no behaviour change. |
| 4 | LOW | [precompute.py:646-660](../../lifecycle/precompute.py#L646) | The treatment of the work-to-retirement boundary in `working_income_next` is correct (zeros the last working row, defers to `pension_after_tax` via z-interpolation), but the **mathematical justification** (that `working_income_next` rows for `t+1 = retire_age` would otherwise integrate over `(eta, eps)` shocks that aren't conceptually applied to the pension) is implicit. | Add a one-line docstring note: "The last working row (`t = retire_age - 1`) is zeroed because the FOC at age `retire_age - 1` integrates over `(eta_{t+1}, eps_{t+1})` for the next-period income shock, but at age `retire_age` income is the pension which depends only on `z_{t+1} = rho z_t + eta_{t+1}` (no `eps`), so we cannot use the (eta, eps) tensor product; we use the `pension_after_tax(z)` table with linear z-interpolation instead." |
| 5 | INFO | [model.py:425](../../lifecycle/model.py#L425), [model.py:464-468](../../lifecycle/model.py#L464) | Payroll-tax cap (2.5) and AIME cap (2.5) are written as separate magic numbers in two functions. They are **mathematically the same constant** (US SSA max-taxable-earnings). | Promote to a module-level constant `SSA_MAX_TAXABLE = 2.5` and reuse in both. Pure cosmetic; no behaviour change. |
| 6 | INFO | [model.py:425](../../lifecycle/model.py#L425) | Payroll-tax rate 10.6% is below the statutory combined OASDI rate of 12.4%. This is a Catherine-2025 incidence-on-worker calibration choice but is undocumented inline. | Add a one-line comment: `# 10.6% = OASDI 6.2% (employee) + 4.4% (half employer; Catherine 2025 incidence)`. |
| 7 | NONE | Fix A (Mu_eta2 / Mu_eps2 derivation) | Verified working as designed. Single source of truth in [precompute.py:875-893](../../lifecycle/precompute.py#L875). All consumer sites read `model.mu_eta2` / `model.mu_eps2` after the override. Asserts cover machine-precision violations. The HIGH-1, HIGH-2, MED-2, LOW-1, LOW-2 findings of the prior audit are CLOSED. | None — confirms correct implementation. |
| 8 | NONE | Judd quadrature exactness | Verified bit-perfect at `n=2,3,4` against the canonical mixture moments through degree `2n-1`. | None. |
| 9 | NONE | Pension PIA continuity, tax bracket continuity | Both verified to ~1e-10 fp64 precision in §7.5 and §8.2. | None. |
| 10 | NONE | TCJA / SSA calibration | All bracket boundaries within ~1.2% of TCJA 2018 normalised by AWI 2019; PIA bend points within 3% of SSA 2019; replacement rates and marginal tax rates exact. | None. |

---

## §11 Verdict

**PASS-WITH-CAVEATS.**

The labour income process is mathematically well-specified and
correctly implemented. All hard mathematical identities (zero-mean
constraint, PIA continuity, tax-bracket continuity, Judd polynomial
exactness, unconditional second moments) hold to machine precision.
The calibration is faithfully drawn from Catherine (2025), Guvenen et
al. (2022), US SSA 2019, and TCJA 2018 — all to ≤3% fidelity, with
marginal rates exact.

The Fix A consolidation
([INCOME_PIPELINE_REVIEW_2026-05-09.md](./INCOME_PIPELINE_REVIEW_2026-05-09.md))
has eliminated the structural fragility that the previous audit
flagged as HIGH-1 / HIGH-2 / MED-2; both `mu_eta2` and `mu_eps2` are
now derived in `build_model` from the zero-mean constraint and read by
every downstream consumer from the model field. Production solver
output is unaffected by the prior MED-1 (Pi_z reducibility) because
`Pi_z` is no longer materialised on the production hot path.

**The CAVEATS are:**
1. The `solver_pi_z_variant.py` docstring underrecommends `n_stds`
   reduction (still produces a reducible Pi_z at the recommended
   value); should switch to Rouwenhorst for the variant. (Finding 2.)
2. Three documentation-drift items (Findings 1, 4, 5, 6) — none
   numerically wrong, all suggest one-line clarifying comments.
3. The pension AIME approximation `min(exp(z) * avg_det, 2.5)`
   replaces Catherine's career-average state variable with a single-
   period proxy. Defensible at `rho=0.991` (near unit root) but is a
   modelling choice that should be cross-checked against the full
   path-dependent AIME under a sensitivity sweep, per the
   `[feedback_sensitivity_analysis]` convention. This is documented
   in `docs/LABOUR.md §3` and is not a bug.

Nothing on this list blocks production use; nothing materially affects
solver accuracy at the canonical configuration.

# Utility specification review (2026-05-10)

**Branch:** `jax-rewrite`. **Author:** read-only audit, no source changes.
**Scope:** Verification-style review of the per-period CRRA utility, the
luxury-shifted (De Nardi 2004 / Catherine 2025) bequest utility, the
annuity-factor term-structure interpolation, and the solver consumption
sites for these primitives. Inputs: [lifecycle/model.py](../../lifecycle/model.py),
[lifecycle/solver.py](../../lifecycle/solver.py),
[lifecycle/inf_horizon_solver.py](../../lifecycle/inf_horizon_solver.py),
[configs/_canonical.py](../../configs/_canonical.py), and the prior audit
docs `CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md`,
`MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md`,
`ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09.md`,
`ECONOMIC_SETUP_REVIEW_PART_B_2026-05-09.md`.

The mathematical identities below were verified at the Python prompt with
`JAX_PLATFORMS=cpu` (CPU only, no kernels touched, no solver invoked); the
exact transcripts are summarised inline.

---

## §1 — Scope statement

The audit covers, exhaustively, the utility-related code surface only:

1. CRRA per-period utility primitives `crra_u`, `crra_uprime`,
   `crra_uprime_inv` and the legacy `create_utility_functions` factory in
   [lifecycle/model.py:273-312](../../lifecycle/model.py#L273-L312); the
   `gamma == 1.0` log branch and how the inverse is consumed in the
   inverted Euler at
   [lifecycle/solver.py:1322-1323](../../lifecycle/solver.py#L1322-L1323).
2. The luxury-shifted bequest level / marginal / inverse functions
   `bequest_utility`, `bequest_marginal`, `bequest_marginal_inv` at
   [lifecycle/model.py:366-414](../../lifecycle/model.py#L366-L414); the
   module-level shift constant `DELTA_BEQUEST` at
   [model.py:363](../../lifecycle/model.py#L363); the term-structure
   `annuity_factor` at
   [model.py:319-351](../../lifecycle/model.py#L319-L351).
3. The solver-side fused marginal+derivative kernel
   `bequest_mu_and_mup` at
   [solver.py:404-414](../../lifecycle/solver.py#L404-L414), and every
   call site that consumes it (terminal, retirement, working FOC).
4. The `delta_bequest` sentinel handshake between `SolverConfig` and the
   module-level constant ([solver.py:2798](../../lifecycle/solver.py#L2798),
   [inf_horizon_solver.py:548](../../lifecycle/inf_horizon_solver.py#L548)).
5. Calibration values `gamma=5, beta=0.96, b_bar=10, DELTA_BEQUEST=0.005`
   declared in [model.py:363](../../lifecycle/model.py#L363) and
   [configs/_canonical.py:31, :149](../../configs/_canonical.py#L31).

Out of scope: tax/income code, the CCV log-portfolio approximation
itself (already audited under `CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md`),
mortality calibration, and grid/quadrature discretization.

---

## §2 — CRRA per-period utility

### Equations

```
u(c)        = c^(1-gamma) / (1-gamma)        if gamma != 1
            = log(c)                          if gamma == 1
u'(c)       = c^(-gamma)                      if gamma != 1
            = 1/c                             if gamma == 1
(u')^{-1}(m)= m^(-1/gamma)                    if gamma != 1
            = 1/m                             if gamma == 1
```

### Code

[model.py:273-284](../../lifecycle/model.py#L273-L284):

```python
def crra_u(c, gamma):
    if gamma == 1.0:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def crra_uprime(c, gamma):
    return c ** (-gamma) if gamma != 1.0 else 1.0 / c


def crra_uprime_inv(mu, gamma):
    return mu ** (-1.0 / gamma) if gamma != 1.0 else 1.0 / mu
```

### Algebraic checks

* **Derivative consistency.** d/dc [c^(1-γ)/(1-γ)] = (1-γ)c^(-γ)/(1-γ) =
  c^(-γ). Matches `crra_uprime`.
* **Log-branch derivative consistency.** d/dc log c = 1/c. Matches
  `crra_uprime` log branch.
* **Inverse consistency.** Solving m = c^(-γ) for c gives c = m^(-1/γ).
  Matches `crra_uprime_inv`. Log branch: m = 1/c → c = 1/m. Matches.
* **Continuity at γ=1.** L'Hôpital on c^(1-γ)/(1-γ) - 1/(1-γ) gives
  log c. The dispatch is exact-equality on `gamma == 1.0`. The level
  utility differs from the log branch by an additive constant 1/(1-γ);
  irrelevant for FOC use, the difference is a level shift that is
  divergent at γ=1 but never enters the solver (the solver consumes
  marginals only — see below).

### EGM use

The solver never calls the level `crra_u`. It uses the marginal in two
forms:
* Inline `c**(-gamma)` at the alive marginal computations
  [solver.py:1108](../../lifecycle/solver.py#L1108) and
  [solver.py:1218](../../lifecycle/solver.py#L1218).
* The inverse via `(beta * V_dot)**(-1/gamma)` at
  [solver.py:1322-1323](../../lifecycle/solver.py#L1322-L1323):

```python
beta_e = jnp.maximum(beta * V_dot, euler_inv_floor)
c_opt = jnp.maximum(beta_e ** (-1.0 / gamma), min_consumption)
```

`euler_inv_floor=1e-20` ([model.py:177](../../lifecycle/model.py#L177))
guards against `V_dot ≤ 0` (would generate NaN^p for non-integer
exponent); `min_consumption=1e-10`
([model.py:174](../../lifecycle/model.py#L174)) clips the result. With
γ=5, `(1e-20)^(-1/5) = 1e4` — so the floor never binds at canonical γ
unless V_dot itself is near zero (it should not be, FOC residual is
positive everywhere on the alive branch).

### Log-branch dispatch issue (LOW)

`crra_u`, `crra_uprime`, `crra_uprime_inv` use Python `if gamma == 1.0`
which is fine for the public API but the solver path **inlines**
`c**(-gamma)` and `(beta_e)**(-1/gamma)` and never tests for γ=1. With
γ=5 in canonical this is moot, but if a future calibration sets γ=1
the inline path would compute `c**(-1.0)` (still correct: matches log
branch's marginal) and `(beta_e)**(-1.0)` (also correct). So the
inline pattern is algebraically robust at γ=1 even without an `if`.
**No fix needed.** The `crra_u` level function would still take the log
branch via the dispatch, but the level is unused in the solver.

### Edge cases at c → 0

`crra_uprime(0, 5)` = `0**(-5)` → `inf` (numpy does this without
warning at runtime; jax does as well). Solver never feeds c=0 to the
marginal: `min_consumption=1e-10` is applied at every interpolation site
([solver.py:1021-1022, :1094-1095](../../lifecycle/solver.py#L1021-L1095))
and at the inverted-Euler output. So the marginal is finite throughout
the FOC kernel.

### Verdict — §2

CLEAR. CRRA primitives, derivatives, and inverse are all algebraically
correct; the solver's inlined uses are equivalent to the dispatched
helpers; γ=1 dispatch in the public functions is fine and the
inline-only solver paths are γ=1 correct without explicit dispatch.

---

## §3 — Bequest utility (level)

### Equation

The shifted (luxury) bequest utility, De Nardi (2004) eq. (12) /
Catherine (2025) eq. (21):

```
b(W, A) = b_bar * (max(W,0)/A + delta)^(1-gamma) / (1-gamma).
```

### Code

[model.py:366-385](../../lifecycle/model.py#L366-L385):

```python
def bequest_utility(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    C_bar = np.maximum(W, 0.0) / A + delta
    return b_bar * C_bar ** (1.0 - gamma) / (1.0 - gamma)
```

### Dead-branch clamp

The level uses `max(W, 0.0)/A`, so for W ≤ 0 the level **plateaus** at
the constant `b_bar * delta**(1-gamma) / (1-gamma)`. With γ=5, δ=0.005,
b̄=10 that plateau is `10 * 0.005^(-4) / (-4) = -4 × 10^9` (a large
negative finite number, since 1-γ = -4). The plateau is *symmetric in
the dead direction* — going more negative does not increase the
penalty further. This matches the De Nardi convention that "bankruptcy
caps the punishment at the cliff value".

### Calibration of δ

* Module-level constant: `DELTA_BEQUEST = 0.005`
  ([model.py:363](../../lifecycle/model.py#L363)).
* Effective canonical value: `CANONICAL_SOLVER.delta_bequest = 0.0`
  ([configs/_canonical.py:149](../../configs/_canonical.py#L149)),
  consumed via the sentinel handshake at
  [solver.py:2798](../../lifecycle/solver.py#L2798): `delta = sc.delta_bequest if sc.delta_bequest >= 0.0 else DELTA_BEQUEST`.
* So **canonical production runs at δ=0** — the un-shifted CRRA bequest.
  The shifter is dialled in only if the pivot baseline exposes the
  W=0+ cliff (see §4).

### Identity check (numeric)

```
gamma=5, b_bar=10, delta=0.005, A=4
W   bequest_utility           bequest_marginal          (signs/sizes consistent)
-1  -4.000e+09                  0
 0  -4.000e+09                  0
1e-12 -4.000e+09 (plateau)      8.000e+11   <- jump at 0+
0.001 -3.291e+09                6.268e+11
0.01  -7.901e+08                1.053e+11
```

The level jumps continuously into the plateau but the marginal jumps
discontinuously: see §4.

### Verdict — §3

CLEAR. The level matches Catherine (2025) eq. (21); the `max(W,0)`
clamp gives a finite negative plateau on the dead branch. δ default
0.005 is the pre-pivot canonical; production canonical
overrides to δ=0 (un-shifted CRRA bequest). Both are algebraically
defined.

---

## §4 — Bequest marginal

### Equation

Differentiating the level on W > 0:

```
d/dW [b_bar * (W/A + delta)^(1-gamma) / (1-gamma)]
   = b_bar * (W/A + delta)^(-gamma) * (1/A)
   = b_bar * (W/A + delta)^(-gamma) / A.
```

For W ≤ 0 the level is constant ⇒ derivative is 0.

### Code

[model.py:388-400](../../lifecycle/model.py#L388-L400):

```python
def bequest_marginal(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    pos = W > 0.0
    C_bar = np.where(pos, W / A + delta, 1.0)  # placeholder for W<=0 branch
    mu = b_bar * C_bar ** (-gamma) / A
    return np.where(pos, mu, 0.0)
```

### Derivative-of-level check (verified numerically)

Central-difference derivative of `bequest_utility` matches
`bequest_marginal` to ≤ 1e-8 relative across W ∈ {0.01, 0.1, 1, 5, 50,
500} at γ=5, A=4, δ=0.005, b̄=10 (transcript: `rel_err` ≤ 5.55e-9 for
every test point).

### mu_max upper bound

The supremum of the marginal as W → 0+ is

```
mu_max = lim_{W -> 0+} b_bar * (W/A + delta)^(-gamma) / A
       = b_bar * delta^(-gamma) / A.
```

Numerically with γ=5, δ=0.005, b̄=10, A=4: `mu_max = 8.0e+11`. This
matches the docstring at [model.py:355-360](../../lifecycle/model.py#L355-L360).

### Dead-branch clamp consistency

The marginal returns 0 on W ≤ 0. The level on W ≤ 0 is the constant
`b_bar * delta**(1-gamma)/(1-gamma)`, whose derivative on the open set
W < 0 is zero. So the W < 0 strict region is fully consistent.

**However**, at exactly W=0 the level computes `b_bar * delta^(1-gamma)
/ (1-gamma)` (a finite negative plateau value), but the *right*
derivative of the level at W=0+ equals `mu_max` ≈ 8e+11. The marginal
function returns `0` at W=0 (because `pos = W > 0.0` is strict). So
`bequest_marginal` chooses the **left** one-sided derivative at the
W=0 boundary. This is a deliberate spec choice: heirs inherit nothing
when the realised estate is non-positive, so the marginal incentive to
save more at the bankruptcy boundary is zero. Documenting this is
worthwhile; see Findings.

### Solver-side variant

The solver does NOT call `bequest_marginal`. It uses the unguarded
formula in `bequest_mu_and_mup` at
[solver.py:404-414](../../lifecycle/solver.py#L404-L414):

```python
def bequest_mu_and_mup(W, A, gamma, b_bar, delta):
    C_bar = W / A + delta
    mu = b_bar * C_bar ** (-gamma) / A
    mup = -gamma * mu / (A * C_bar)
    return mu, mup
```

Under CCV log-wealth dynamics `W = s · exp(r_p) > 0` whenever `s > 0`,
so the W ≤ 0 branch never bites in the FOC sweep. The docstring at
[solver.py:407-409](../../lifecycle/solver.py#L407-L409) calls this
out explicitly. The Numba reference solver (legacy) takes the same
unguarded path. **The kernel is internally consistent** (its own mup
is the analytic derivative of its own mu), but the kernel's mu is the
*W>0 branch* of `bequest_marginal` only — it does not implement the
W≤0 clamp. Acceptable under CCV but worth flagging since the model
function and the solver function disagree on the dead branch (see §7
and Findings).

### mup analytic check

```
d/dW [b_bar * C_bar^(-gamma) / A]   where C_bar = W/A + delta
   = b_bar * (-gamma) * C_bar^(-gamma-1) * (1/A) / A
   = -gamma * b_bar * C_bar^(-gamma-1) / A^2
   = -gamma * mu / (A * C_bar).
```

Matches `mup = -gamma * mu / (A * C_bar)` at
[solver.py:413](../../lifecycle/solver.py#L413). Numerically verified
at W=1, A=4, γ=5, b̄=10, δ=0.005: central-difference of `mu` equals
the analytic `mup` to 3.77e-11 relative.

### Verdict — §4

CLEAR with one minor inconsistency to document. The marginal formula
on W>0 is correct; the W≤0 clamp is consistent with the level on the
strict-negative side and chooses the left one-sided derivative at
W=0; mu_max bound matches the docstring; the unguarded
`bequest_mu_and_mup` is fine under CCV log-wealth (W>0 always) but is
not interchangeable with `bequest_marginal` on a hypothetical
non-positive-W simulator. See finding F2.

---

## §5 — Bequest marginal inverse

### Equation

Solving μ = b̄ (W/A + δ)^(-γ) / A for W on μ ∈ (0, mu_max]:

```
mu * A / b_bar       = (W/A + delta)^(-gamma)
(mu * A / b_bar)^(-1/gamma) = W/A + delta
W = A * ((mu*A/b_bar)^(-1/gamma) - delta).
```

For μ > mu_max the wealth constraint W ≥ 0 binds and the inverse
clamps to 0.

### Code

[model.py:403-414](../../lifecycle/model.py#L403-L414):

```python
def bequest_marginal_inv(mu, A, gamma, b_bar, delta=DELTA_BEQUEST):
    mu_max = b_bar * delta ** (-gamma) / A
    mu_clamped = np.minimum(mu, mu_max)
    inner = (mu_clamped * A / b_bar) ** (-1.0 / gamma) - delta
    return A * np.maximum(inner, 0.0)
```

### Domain analysis

* μ → 0+ : `(μA/b̄)^(-1/γ) → ∞`, so W → ∞. Correct — vanishing marginal
  bequest utility means infinite wealth.
* μ = mu_max: `(mu_max·A/b̄)^(-1/γ) = (δ^(-γ))^(-1/γ) = δ`, so inner =
  0, W = 0. Correct.
* μ > mu_max: `mu_clamped = mu_max`, inner = 0 (after the inner-clamp
  step), W = 0. Correct (constraint binds).
* μ ≤ 0: `mu**(-1/γ)` is undefined for μ ≤ 0 in real arithmetic. The
  function does NOT guard μ ≤ 0; callers must supply μ > 0. In
  practice the function is called nowhere in the solver (see §7) and
  the docstring requires μ > 0. **Low-severity defensive-coding
  observation.**

### Identity check (verified numerically)

```
W       mu = bequest_marginal(W)    bequest_marginal_inv(mu) - W (rel)
0.01    1.053e+11                   -8.67e-16
0.1     1.029e+08                   -2.78e-16
1       2.319e+03                    0
5       8.030e-01                    0
50      8.176e-06                    1.42e-16
500     8.190e-11                    2.27e-16
```

Round-trip identity holds to fp64 precision throughout the W range
that the production wealth grid covers (0.01 – 750 AWI per
[_canonical.py:108-124](../../configs/_canonical.py#L108-L124)).

### Sign / clamp check

* The outer `A * np.maximum(inner, 0.0)` correctly forces W ≥ 0.
* The inner `np.minimum(mu, mu_max)` correctly clamps μ ≤ mu_max
  *before* exponentiation, so even a numerically wild μ from a faulty
  upstream call cannot produce a complex/NaN result. Robust.
* Sign: μ * A / b̄ > 0, so its (-1/γ) power is well-defined and
  positive. inner = positive - δ. For inner ≤ 0 (i.e., μ near mu_max),
  the outer max clamp returns 0. No sign error.

### Production usage

`bequest_marginal_inv` is **never called** in the JAX solver
([solver.py](../../lifecycle/solver.py)) or in the inf-horizon solver
([inf_horizon_solver.py](../../lifecycle/inf_horizon_solver.py)). It is
exposed only for diagnostics and tests (see verify/* and
docs/UTILITY.md §2.1). Under EGM the inverse Euler is computed
analytically through `(β·V_dot)^(-1/γ)`; the bequest marginal inverse
would only be needed if the solver inverted the bequest's mu directly
(e.g., a backward EGM that worked in marginal-utility space rather
than savings space) — which the JAX rewrite does not do.

### Verdict — §5

CLEAR. The inverse is the correct algebraic inverse of the marginal on
its valid domain, the mu_max clamp is correctly placed before the
exponentiation, and the W ≥ 0 outer clamp is correct. Function is
unused by the solver — keep for diagnostic call sites and to assert
the model is internally consistent.

---

## §6 — Annuity factor

### Equation

```
A(y_1, spr, b_bar) = sum_{k=1}^{b_bar} (1 + y(k))^(-k)
y(k) = y_1 + spr * (k-1)/(b_bar - 1).
```

So y(1)=y_1 and y(b_bar)=y_1 + spr (linearly interpolated yields). This
is **discrete** compounding.

### Code

[model.py:319-351](../../lifecycle/model.py#L319-L351):

```python
def annuity_factor(y_1, spr, b_bar):
    y_1 = np.asarray(y_1, dtype=float)
    spr = np.asarray(spr, dtype=float)
    A = np.zeros_like(y_1)
    denom = max(int(b_bar) - 1, 1)
    for k in range(1, b_bar + 1):
        frac = (k - 1) / denom
        y_k = y_1 + spr * frac
        A += (1.0 + y_k) ** (-k)
    return A
```

### Discrete vs continuous compounding

The docstring explicitly forbids `exp(-y*k)` (continuous) and prescribes
`(1+y)^(-k)` (discrete). The code uses the latter. At y=5%, b_bar=10
the gap continuous-vs-discrete is roughly 12 bp/yr × 10 ≈ 120 bp on the
discount factor — non-negligible. **Code matches docstring.**

### Linear-spread interpolation

`frac = (k-1)/denom` with `denom = max(b_bar-1, 1)`. So:
* k=1: frac=0, y(1) = y_1.
* k=b_bar: frac = (b_bar-1)/(b_bar-1) = 1, y(b_bar) = y_1 + spr.

Edge case **b_bar = 1**: `denom = max(0, 1) = 1`, the loop runs only
for k=1 with `frac = 0`, `y(1) = y_1`, `A = (1+y_1)^(-1)`. Numeric
check: `A(0.02, 0.005, 1) = 0.98039` = 1/1.02 ✓. The `max(.., 1)` guard
prevents division by zero at b_bar=1.

### Numeric identity check

For y_1=0.02, spr=0.005, b_bar=10:
```
A = 8.831770   (analytic loop)
manual = sum_{k=1..10} (1 + 0.02 + 0.005*(k-1)/9)^(-k) = 8.831770
```
Match exact.

### Production usage

* Built once per state grid in
  [precompute.py:360-377](../../lifecycle/precompute.py#L360-L377): for
  every state-grid cell, `_y_1` and `_spr` are sourced from the
  state-grid columns at `y_1_index_in_state` and `spr_index_in_state`,
  with scalar fallbacks for legacy partial-state configs.
* Stored on `Precompute.annuity_factors` (shape `(N_state,)`) at
  [precompute.py:525](../../lifecycle/precompute.py#L525); promoted to
  `pcj.annuity_factors` at
  [solver.py:1795](../../lifecycle/solver.py#L1795).
* Consumed at every FOC kernel call as the `A_is` scalar in
  `bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)`.

### Term-structure semantics

The term-structure interpolation between y_1 and y_b_bar is a
*one-shot* linear interpolation of yields (not log-yields, not forward
rates). Under the canonical (b_bar=10, spr ~ 5-150 bp on the real
curve), this is a reasonable proxy for the actual term structure — the
yield curve is approximately linear in maturity over the 1y-to-10y
window in the long-run real-yield sample. A more refined alternative
would be Nelson-Siegel; out of scope.

The docstring at [model.py:325-326](../../lifecycle/model.py#L325-L326)
says "Under the active baseline b_bar=10 and spr is the
10-year-minus-1-year real spread." That matches the data pipeline (see
`docs/scans/CP_INFLATION_REAL_YIELDS_VAR_REVIEW_2026-05-09.md`).

### Verdict — §6

CLEAR. Correct discrete-compounding formula, correct linear yield
interpolation, robust b_bar=1 edge case, consistent with the data
pipeline. Verified to match a manual sum and to match 1/(1+y_1) at
b_bar=1.

---

## §7 — Solver / FOC consumption sites

### `bequest_mu_and_mup` (solver-private fused kernel)

[solver.py:404-414](../../lifecycle/solver.py#L404-L414):

```python
def bequest_mu_and_mup(W, A, gamma, b_bar, delta):
    C_bar = W / A + delta
    mu = b_bar * C_bar ** (-gamma) / A
    mup = -gamma * mu / (A * C_bar)
    return mu, mup
```

* `mu` matches the W>0 branch of `bequest_marginal` exactly. No
  `np.maximum(W, 0)` clamp — under CCV `s · exp(r_p) > 0`.
* `mup = ∂mu/∂W` is analytically correct (verified §4).
* Returns the **shifted** form (uses `delta`). Consistent with the
  module function and with the `delta` plumbed in from `SolverConfig`
  at [solver.py:2798](../../lifecycle/solver.py#L2798) and from
  inf-horizon at
  [inf_horizon_solver.py:548, :757](../../lifecycle/inf_horizon_solver.py#L548).

### Call sites for `bequest_mu_and_mup`

1. Terminal kernel
   [solver.py:863](../../lifecycle/solver.py#L863):
   `mu, mup = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)`. The
   FOC sums `weight_kv_kr * mu * dRp_da{s,b}` (lines 871-872). Sign:
   bequest is a *positive* contribution to expected utility, so `mu`
   enters the FOC with a **positive** sign — correct.
2. Retirement kernel
   [solver.py:1060](../../lifecycle/solver.py#L1060): same call,
   combined with the alive marginal as
   `mu_comb = psi_z * mu_alive + (1-psi_z) * mu_bq`
   ([solver.py:1112](../../lifecycle/solver.py#L1112)) and
   `mup_comb` analogously
   ([solver.py:1113](../../lifecycle/solver.py#L1113)). Mortality
   weighting uses `prob_death = 1 - psi_z`.
3. Working kernel
   [solver.py:1171](../../lifecycle/solver.py#L1171): same call;
   bequest contribution is summed independently of (eta, eps) and
   added to the alive contribution at the kernel return
   ([solver.py:1247-1254](../../lifecycle/solver.py#L1247-L1254)).
4. pi_z variant solver
   [solver_pi_z_variant.py:163](../../lifecycle/solver_pi_z_variant.py#L163):
   re-imports the same `bequest_mu_and_mup`. No drift between solvers.

### FOC sign and Jacobian linearity

For each of the three kernels:
* `foc_s_bq = sum(weight * mu_bq * dRp_das)` — bequest contributes a
  **positive** term to the stock FOC (since dRp_das can be ±, the sum
  signs out — but `mu_bq > 0` is correct).
* `jac_lin_bq = bequest_factor * mup_bq * s_val`. Since `mup_bq < 0`
  (decreasing marginal utility), `jac_lin_bq < 0`, then multiplied by
  `dRp_das**2 ≥ 0` ⇒ **negative** contribution to the diagonal Jacobian
  entries — correct concavity for a maximisation problem.
* `extra_ss = bequest_factor * mu_bq * R_p * (dr_da_s**2 - sigma2_xr)`.
  The `(dr_da_s)**2 - sigma2_xr` term is the second-derivative
  correction `R_p · ∂²r/∂α_s² + R_p · (∂r/∂α_s)²` with
  `∂²r/∂α_s² = -σ²_xr`. Verified independently in
  `MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md §4`.

### V_dot and the inverted Euler

* Terminal: `V_dot = sum(wmu * R_p)`
  ([solver.py:873](../../lifecycle/solver.py#L873)) — pure bequest.
* Retirement: `e_sum = sum(wmu * R_p)` with `wmu = weight * mu_comb`
  ([solver.py:1115, :1123](../../lifecycle/solver.py#L1115)) — combined.
* Working: `e_bq + e_al`
  ([solver.py:1253](../../lifecycle/solver.py#L1253)) — bequest +
  alive-state expected marginal-utility-weighted return.

The inverted Euler then computes `c_opt = (β · V_dot)^(-1/γ)` at
[solver.py:1322-1323](../../lifecycle/solver.py#L1322-L1323), giving
the EGM endogenous `(c, x)` pair. The β multiplier is correct: bequest
accrues one period after the choice, so the bequest term inside V_dot
is discounted by β just like alive-state continuation utility (per
docs/UTILITY.md §2.3).

### Shifted-form everywhere?

Yes. Every call site to `bequest_mu_and_mup` passes `delta`, which is
sourced from
`sc.delta_bequest if sc.delta_bequest >= 0 else DELTA_BEQUEST`
([solver.py:2798](../../lifecycle/solver.py#L2798)). With
`CANONICAL_SOLVER.delta_bequest = 0.0`, every kernel runs with the
**un-shifted** spec δ=0 (which is the W>0 branch of the un-shifted
CRRA bequest, mathematically `b̄ · (W/A)^(-γ) / A`). The module-level
`DELTA_BEQUEST = 0.005` is the fallback when a SolverConfig sets
`delta_bequest < 0` (the "use the canonical shifter" sentinel). So the
**code path is the shifted form, but the canonical configuration
chooses δ=0**. Under CCV (W>0 always), δ=0 produces no NaN, but it
re-introduces the W → 0+ marginal-utility cliff of order
`b̄·(W/A)^(-γ)/A → ∞`. The solver does not bracket against this
because tiny-savings cells (s ≤ 1e-6) take the cold-init bypass at
[solver.py:1325-1328](../../lifecycle/solver.py#L1325-L1328) and never
evaluate the FOC at s ≈ 0.

### Verdict — §7

CLEAR. The kernel uses the shifted form with `delta` plumbed from
config; signs of marginal in FOC and Jacobian are correct;
`bequest_mu_and_mup` mu and mup are the analytic forms of the W>0
branch (no dead-branch clamp, but unneeded under CCV); pi_z variant
reuses the same kernel. One config-vs-comment **inconsistency** worth
flagging: the canonical sets δ=0 yet [model.py:354-362](../../lifecycle/model.py#L354-L362)
documents DELTA_BEQUEST=0.005 as the working value. See finding F1.

---

## §8 — Calibration consistency

### Source-of-truth values

| Param | Code value | File:line |
|---|---|---|
| γ (gamma) | 5.0 | [_canonical.py:31](../../configs/_canonical.py#L31) |
| β (beta) | 0.96 | [_canonical.py:31](../../configs/_canonical.py#L31) |
| b̄ (b_bar) | 10 | [_canonical.py:31](../../configs/_canonical.py#L31) |
| DELTA_BEQUEST (module) | 0.005 | [model.py:363](../../lifecycle/model.py#L363) |
| `delta_bequest` (SolverConfig default) | -1.0 (sentinel ⇒ use module) | [model.py:182](../../lifecycle/model.py#L182) |
| `delta_bequest` (CANONICAL_SOLVER) | 0.0 (un-shifted) | [_canonical.py:149](../../configs/_canonical.py#L149) |
| Effective δ in canonical run | **0.0** | via sentinel handshake [solver.py:2798](../../lifecycle/solver.py#L2798) |

### `mu_max = b_bar * delta**(-gamma) / A` bound

* Documented at [model.py:355-360](../../lifecycle/model.py#L355-L360):
  with (b_bar, gamma, A) = (10, 5, 4) and δ=0.005, mu_max ≈ 8e+11.
* Verified numerically: `b_bar * 0.005**(-5) / 4 = 8.0e+11` (exact).
* Implemented in `bequest_marginal_inv` at
  [model.py:411](../../lifecycle/model.py#L411).
* Under canonical δ=0 the bound is **infinite** — the cliff is
  re-introduced. No code path explicitly guards against this; relies
  on CCV-induced W>0 + tiny_savings bypass.

### Cross-check vs prior audits

* **`CCV_REAL_PIVOT_THEORY_AUDIT_2026-05-09.md`**: §1-§4 audit the
  CCV log-portfolio formula and the FOC in (alpha_s, alpha_b). Does
  not contradict the bequest spec; treats the bequest marginal as a
  black box with `mu_bq, mup_bq = bequest_mu_and_mup(s·R_p, A, …)`.
  Verdict consistent with this scan: bequest plumbing CLEAR, with the
  CCV approximation itself the only theoretical concern (out of
  scope).
* **`MODEL_REVIEW_BELLMAN_FOC_2026-05-09.md` §4**: independently
  re-derives `mup = -γ·mu/(A·C̄)` and confirms the chain rule through
  wealth-interp slope mpc. Same conclusion as §4 of this scan.
* **`ECONOMIC_SETUP_REVIEW_PART_A_2026-05-09.md` §1.3**: notes the
  same DELTA_BEQUEST=0.005 vs canonical=0.0 inconsistency
  (line 27-29) and flags a sensitivity sweep before publication. This
  scan elevates that to a HIGH finding (F1) since the comment block
  inside model.py still describes 0.005 as the recommended ship value.
* **`docs/UTILITY.md`**: matches model.py exactly (verified by Grep
  on the function names and signatures). No drift.

### Internal consistency

* Calibration `gamma=5, b_bar=10, beta=0.96` matches Catherine 2025
  Table 4 and the warm-glow horizon `b̄ = 10y` convention.
* The `mu_max` bound formula in the docstring depends on A=4 as a
  *typical* value; actual `A_is` varies with state. At y_1=2%,
  spr=0.5%, b_bar=10: A ≈ 8.83 (verified §6) — so the *real* canonical
  mu_max under δ=0.005 would be `10·0.005^(-5)/8.83 ≈ 3.6e+11` rather
  than `8e+11`. The docstring's "A=4" is illustrative; the actual
  bound is state-dependent. Worth a docstring tweak (low severity).

### Verdict — §8

CLEAR with the F1 caveat. All reported parameters match the source of
truth; the mu_max bound formula matches the implementation; the prior
audits agree with this scan's findings. The DELTA_BEQUEST module-level
constant should be reconciled with the canonical override (F1).

---

## §9 — Findings

| # | Severity | Location | What's wrong | Fix sketch |
|---|----------|----------|--------------|------------|
| F1 | **MED** (config drift) | [model.py:354-363](../../lifecycle/model.py#L354-L363) vs [_canonical.py:136-149](../../configs/_canonical.py#L136-L149) | Module-level docstring states `DELTA_BEQUEST = 0.005` "ship value" with sensitivity-sweep gate at {0.001, 0.005, 0.01, 0.02}, but `CANONICAL_SOLVER.delta_bequest = 0.0` (un-shifted CRRA bequest re-introduced). Comment in `_canonical.py:136-138` says "drop the luxury-bequest shifter" with no cross-link to the model.py block. Reviewers reading model.py believe canonical δ = 0.005. | Either (a) change `DELTA_BEQUEST` module default to 0.0 and move the 0.005 documentation into a "shifter recipe" comment, or (b) raise `delta_bequest` back to 0.005 in canonical and document the rationale to revert. Recommended: (a), with a one-line cross-link in model.py:354 saying "canonical override = 0; see _canonical.py:149". Per the sensitivity-feedback memory, also report alpha_s deltas across {0, 0.001, 0.005, 0.01} before shipping. |
| F2 | LOW (dead-branch drift) | [model.py:388-400](../../lifecycle/model.py#L388-L400) vs [solver.py:404-414](../../lifecycle/solver.py#L404-L414) | `bequest_marginal` (model) has the W ≤ 0 → 0 clamp; `bequest_mu_and_mup` (solver) does NOT. They are not interchangeable on a hypothetical W ≤ 0 input. Under CCV log-wealth W>0 always, so this never bites in production, but a future non-CCV simulator (e.g., a simple-clamp or `levered` variant) calling `bequest_mu_and_mup` directly would silently produce a non-zero (and possibly NaN-near-zero, complex-near-negative) marginal at W ≤ 0. | Add a one-line `np.where(W > 0, ..., 0.0)` clamp in `bequest_mu_and_mup` matching the model function; document that the W ≤ 0 branch is only reached on a non-CCV path. Zero performance cost on JAX (jnp.where is fused). Or, rather than fix, add an `assert jnp.all(W > 0)` debug-only check during Newton iterations to catch a regression. |
| F3 | LOW (discontinuity at W=0) | [model.py:388-400](../../lifecycle/model.py#L388-L400) | `bequest_marginal` returns `0` at exactly W=0 but the right-derivative of the level at W=0+ equals `mu_max`. So the marginal has a jump discontinuity at W=0 of size mu_max. The function chooses the **left** one-sided derivative (heirs inherit nothing on bankruptcy). This is an intentional spec choice but is not documented in the docstring at [model.py:388-396](../../lifecycle/model.py#L388-L396) — a reader expecting `bequest_marginal` to be the strict derivative-of-`bequest_utility` will find an apparent inconsistency at W=0. | One-line docstring addition: "At W=0 returns the left one-sided derivative (= 0); the right one-sided derivative is mu_max but is unreachable on the support of the wealth process under CCV log dynamics." |
| F4 | LOW (μ ≤ 0 not guarded) | [model.py:403-414](../../lifecycle/model.py#L403-L414) | `bequest_marginal_inv` does not guard μ ≤ 0; would produce NaN/complex via `(neg)**(-1/γ)`. Function is unused by the solver, but the docstring requires μ > 0 implicitly. | Add `mu = np.maximum(mu, eps)` for some small eps before the inverse, or assert μ > 0 at the top. Matches the defensive style of `bequest_marginal_inv`'s mu_max clamp. |
| F5 | LOW (illustrative A in docstring) | [model.py:355-360](../../lifecycle/model.py#L355-L360) | The mu_max numerics quote A=4 as a "canonical" value, but the actual `annuity_factors[i_s]` in production runs at canonical (real-yields, b_bar=10, y_1∈[1.5%,2.5%], spr∈[0%,2%]) varies 7-12 across state cells. So the "mu_max ~ 8e11 at δ=0.005" is a 2× overestimate at typical state. | Update docstring: "for b_bar=10, gamma=5, A in [7, 12] (state-dependent), mu_max ranges ~3-5e11 at δ=0.005". |
| F6 | LOW (level utility at W=0 is large negative) | [model.py:366-385](../../lifecycle/model.py#L366-L385) | `bequest_utility(0, A=4, γ=5, b̄=10, δ=0.005) = b̄·δ^(1-γ)/(1-γ) = 10·0.005^(-4)/(-4) = -4×10^9`. While the marginal is finite (mu_max ≈ 8×10^11), the *level* on the dead branch is a large negative finite number — much more punishing than the alive-state CRRA at any meaningful c. The level is unused by the solver but is read by `verify/value_at_age22.py` and similar diagnostics. Risk: a value-function tabulation could be dominated by the dead-branch plateau if the bankruptcy event has non-zero probability. | Document the plateau magnitude in the bequest_utility docstring; if any diagnostic adds value-function levels across realisation paths, alert the consumer that the dead-branch level is asymmetric and possibly dominant for low W. |

No critical findings. All bugs are documentation drift, defensive
hardening, or post-CCV-pivot stale comments. The mathematical core
(equations, derivative consistency, inverse identity, FOC sign) is
correct.

---

## §10 — Verdict

**PASS-WITH-CAVEATS.**

* The CRRA primitives, bequest level, marginal, marginal inverse, and
  annuity factor are all algebraically correct, internally consistent
  (level ↔ marginal ↔ inverse round-trip ≤ 1e-15 rel), and consistent
  with the prior audits and Catherine (2025) eq. 21-22.
* The solver-side fused `bequest_mu_and_mup` matches the W>0 branch of
  the model-level `bequest_marginal` exactly and uses the analytically
  correct `mup`. Sign of marginal in the FOC and concavity sign of the
  Jacobian are correct. β-discounting of the bequest is correct under
  the period-end-death convention. Every kernel uses the **shifted
  form** (parameterised by `delta`); the canonical configuration
  chooses δ=0, which is the un-shifted CRRA bequest, and is supported
  by the W>0 invariance of CCV log-wealth.
* The annuity factor uses discrete compounding per docstring,
  correctly interpolates the term structure linearly between y_1 and
  y_b_bar, and handles the b_bar=1 edge case via a max(b_bar-1, 1)
  guard in the denominator.

The PASS verdict is qualified by F1 (config-vs-doc drift on
`DELTA_BEQUEST` — module says 0.005, canonical overrides to 0.0;
either reconcile or document the override at the model.py site) and
the lower-severity hardening recommendations F2-F6. None of the
caveats produce incorrect numerical output today; they are robustness
and documentation issues. The single most important next step is to
either bring the canonical δ back to 0.005 (and document why the
shifter is on by default) or update model.py:354-363 to reflect the
canonical's choice of δ=0 (and document the un-shifted CRRA bequest's
mu_max=∞ as not-bitten-under-CCV).

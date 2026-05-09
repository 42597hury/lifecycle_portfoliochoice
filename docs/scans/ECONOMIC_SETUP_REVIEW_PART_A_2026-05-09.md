# Economic Setup Review — Part A (Utility, Lifecycle, Income)

**Date:** 2026-05-09
**Branch:** `jax-rewrite`
**Scope:** Read-only review of the post-pivot 3-axis real-yields model. Focus: utility function, lifecycle structure, income process, constraints, and economic-vs-numerical separation. Asset returns / VAR / portfolio formula are reviewed by a parallel agent (see `ARBITRAGE_PIVOT_REVIEW_2026-05-09.md`).

**Documents read:** `configs/_canonical.py`, `lifecycle/model.py`, `lifecycle/precompute.py`, `lifecycle/solver.py`, `lifecycle/simulation.py`, `lifecycle/mortality.py`, `docs/UTILITY.md`, `docs/LABOUR.md`, `docs/STATE_SPACE.md`, `docs/CONFIG.md`.

---

## 1. Utility function

### 1.1 CRRA `u(c)`

**Found:** `crra_u(c, gamma) = c**(1-gamma)/(1-gamma)` for γ≠1, else `log c` (`lifecycle/model.py:251-254`). γ=5 in canonical (`configs/_canonical.py:31`). `u'`, `u'^{-1}` consistent. The solver inlines `c**(-gamma)` and `(beta * euler)**(-1/gamma)` — algebraically equivalent.

**Verdict:** CLEAR.

### 1.2 Bequest spec — shifted/luxury form

**Found:** `bequest_utility(W, A, gamma, b_bar, delta) = b_bar * (max(W,0)/A + delta)**(1-gamma) / (1-gamma)` (`lifecycle/model.py:347-366`). The "/A" is the **annuity factor** `A(y_1, spr, b_bar) = Σ_{k=1}^{b_bar} (1 + y(k))^{-k}` with linearly-interpolated term structure between `y_1` (1-yr) and `y_20 = y_1 + spr` (`lifecycle/model.py:297-332`). Therefore `W/A` is the **flow-equivalent annuitised consumption** the heir would draw if W were converted into a `b_bar`-year coupon at current real-yields term structure — NOT average wealth, NOT AWI normalisation. `b_bar = 10` is the **bequest horizon in years**, not a Catherine warm-glow weight (docstring `lifecycle/model.py:32`; `docs/UTILITY.md:135-148`).

**Verdict:** CLEAR — but the doc is in `docs/UTILITY.md` only; a reader of `_canonical.py` line 29-30 sees `(W/A + δ)` with no inline cue that A is the term-structure annuity. **Suggested fix:** replace the one-liner comment in `_canonical.py:29-30` with `b̄·(W/A(y_1,spr) + δ)^{1-γ}/(1-γ)` and a link to `docs/UTILITY.md §2.1`.

### 1.3 δ (delta_bequest) — luxury shifter

**Found:** Module-level `DELTA_BEQUEST = 0.005` (`lifecycle/model.py:344`); but `CANONICAL_SOLVER.delta_bequest = 0.0` (`configs/_canonical.py:127`) and the solver uses the SolverConfig field as a sentinel: any value `< 0` falls back to module-level (`lifecycle/solver.py:2518`). So **canonical = 0 (un-shifted bequest)**. Comment block in `_canonical.py:113-115` flags this as the "pivot baseline" with the shifter held in reserve for cliff issues.

**Verdict:** CLEAR — but worth noting at thesis time: the un-shifted spec re-introduces the unbounded `mu_max` at W→0 that motivated `DELTA_BEQUEST = 0.005` in the first place (`lifecycle/model.py:336-343`). Under CCV log-wealth dynamics `s·R_p > 0` always, so the cliff doesn't bite, but δ=0 vs δ=0.005 should be on a sensitivity sweep before publication — see [feedback_sensitivity_analysis] convention.

### 1.4 b_bar = 10

**Found:** Catherine (2025) bequest-horizon convention; not a free parameter swept in this codebase. (`docs/CONFIG.md:21`, `docs/LITERATURE_BENCHMARKS.md:59`.)

**Verdict:** CLEAR.

### 1.5 β = 0.96 — annual

**Found:** `beta = 0.96` (`configs/_canonical.py:31`) at 1-year period length; ages 22-99 step in calendar years.

**Verdict:** CLEAR.

### 1.6 min_consumption floor

**Found:** `min_consumption = 1e-10` (`lifecycle/model.py:173`), applied as a clip on interpolated `c_next` in the FOC inner kernel before CRRA arithmetic (`lifecycle/solver.py:1013`, `:943`). This is a **numerical safety floor**, not an economic minimum; the floor is described in `docs/UTILITY.md` as economically irrelevant ("never binds in practice"). Negative consumption cannot occur because EGM inverts `(beta*euler)^{-1/gamma}` which is always positive.

**Verdict:** CLEAR.

---

## 2. Lifecycle structure

### 2.1 Ages and off-by-one

**Found:** `start_age=22, retire_age=67, terminal_age=99` (`configs/_canonical.py:32`). `ages = arange(22, 100)` → 78 periods. Working ages are `t < retire_age_idx = 45` (`lifecycle/simulation.py:561, :656`), i.e. ages 22..66 (45 working ages). Retirement = ages 67..99 (33 ages). Last labour income is at age 66 (paid out at `t = retire_age_idx - 1 = 44`); first pension paid at age 67 (the `is_pre_retire_boundary` branch at `simulation.py:412` substitutes pension(z_next) for working income at the boundary). This matches `docs/LABOUR.md §3.4`.

**Verdict:** CLEAR — the off-by-one is documented and self-consistent in solver+sim. Worth noting that the `retire_age=67` means `age==67` is **already retired** (does not earn labour income). Some readers might expect "retires at 67 = last working age."

### 2.2 Mortality

**Found:** Earnings-dependent stochastic mortality `m_{it} = min(chi(z_i) * m(age_t), 1)` from Catherine (2025) eq. 35, baseline `m(age)` from SSA 2017 period life table, `chi(z)` calibrated to Chetty et al. (2016) life-expectancy-by-percentile (`lifecycle/mortality.py:1-44, :337-420`). Stored as `survival_probs_2d : (n_age, n_z)`. Used in solver as `psi_z` weighting `mu_alive` vs `mu_bq` (`lifecycle/solver.py:1030-1032, :1093`) and in simulation as a Bernoulli draw (`simulation.py:372-374`). At terminal age 99 the agent solves a pure-bequest problem (psi not used there; `_build_per_age_terminal_kernel` at `solver.py:1624`); the SSA q(99) ≈ 0.344 is encoded in the table for ages < 99, and at age 99 death is certain by the lifecycle horizon, not by survival_probs_2d.

**Verdict:** CLEAR.

### 2.3 Bequest trigger

**Found:** Two channels. (i) **Stochastic at every age**: in working/retirement FOC, `prob_death = 1 - psi_z` weighted on bequest marginal (`solver.py:1030-1031, :1091-1093`). (ii) **Deterministic at terminal_age = 99**: terminal kernel solves bequest-only (no continuation value, `solver.py:776-809`). The shifted-luxury form is intended to make the agent **want** to leave wealth (Catherine 2025; De Nardi 2004). Verified: `mu_bq = b_bar * (W/A + δ)^{-γ} / A > 0` for all W>0, so the agent has positive marginal bequest utility at every wealth level.

**Verdict:** CLEAR.

---

## 3. Income process

### 3.1 Deterministic age profile

**Found:** `f(age) = b0 + b1·age + b2·age²/10 + b3·age³/100` (`lifecycle/precompute.py:385-388, :392-394`). With `(b0,b1,b2,b3) = (-6.142, 0.3040, -0.051, 0.002586)`, peak: `f'(age)=0` ⇒ age ≈ **45.8** (the other root is ≈85.7, outside the working window). `exp(f(45.8)) ≈ 0.65 ≈ $35k` matches `docs/LABOUR.md §0`. **Subtle point:** `_canonical.py:33` writes the polynomial as if-it-were `b3·age³` but `precompute.py:388` divides by `100`; the comment in `docs/CONFIG.md §1.3` reflects the divided form. The `age**2/10` and `age**3/100` rescalings are absorbed into b2, b3 — i.e. b2, b3 already carry the corresponding `1/10`, `1/100` factor. Source: Guvenen et al. (2021) / Catherine (2025) Appendix E.1.

**Verdict:** CLEAR — but a thesis reviewer might flag that the polynomial is written three different ways across the codebase (`_canonical.py:33`, `model.py:42-44`, `precompute.py:388`). **Suggested fix:** unify wording. Either always write the divisors, or never. `_canonical.py:33` currently shows `b0 + b1*age + b2*age² + b3*age³` with no divisors, which is **wrong** as a literal description of `precompute.py:388`.

### 3.2 Persistent shock η — mixture spec

**Found:** `η = pz·N(μ_eta1, σ_eta1²) + (1-pz)·N(μ_eta2, σ_eta2²)` with `pz=0.176, μ_eta1=-0.524, σ_eta1=0.113, σ_eta2=0.046`. `μ_eta2` derived: `-(pz/(1-pz))·μ_eta1 ≈ +0.1119` to enforce `E[η]=0` (`configs/_canonical.py:36`, `model.py:49`). This is the **Guvenen et al. (2021) two-component mixture** (rare-large-negative + frequent-small-positive components), Catherine (2025) Table E.1 calibration.

**Verdict:** CLEAR.

### 3.3 ρ = 0.991 — annual AR(1)

**Found:** `z_{t+1} = ρ·z_t + η_{t+1}`. ρ=0.991 implies persistence half-life ≈ ln(0.5)/ln(0.991) ≈ 77 years — near-unit-root, characteristic of the Guvenen calibration. Stationary `σ_z = sqrt(Var(η)/(1-ρ²))`; mixture variance ≈ 0.0626, ρ=0.991 ⇒ σ_z ≈ 1.86.

**Verdict:** CLEAR.

### 3.4 pz, pe — mixture weights

**Found:** `pz = 0.176` is the weight on η component 1 (rare-large-negative); `pe = 0.044` is the weight on ε component 1 (rare-large transitory). Both are documented as "mixture component 1 probability" in `docs/CONFIG.md:46-65` and `docs/LABOUR.md §1.2-§1.3`. They are **NOT skew/kurt parameters** in a parametric sense — they are mass-fractions of a 2-component Gaussian mixture.

**Verdict:** CLEAR (in docs); UNDOCUMENTED in `configs/_canonical.py` itself. **Suggested fix:** add a one-liner above line 34: `# pz, pe: mixture-weight on the rare/large component (Guvenen 2021)`. Otherwise a reviewer might mistake `pz=0.176` for an autocorrelation or a state-transition probability.

### 3.5 Pension and replacement rate

**Found:** `compute_pension_after_tax(z_grid, avg_det)` (`lifecycle/model.py:428-482`): AIME proxy = `min(exp(z) * avg_det, 2.5)` with `avg_det = mean(exp(f(age))) over working ages ≈ 0.507`; PIA = bend-point formula with `b1=0.21, b2=1.25, r1=0.90, r2=0.32, r3=0.15` matching SSA 2019 values. Pension is constant in z post-retirement, indexed by terminal-z. `docs/LABOUR.md §3` documents this is an **approximation** to Catherine's career-average state variable (path-dependence lost; ρ≈1 rationale). z=0 ⇒ ~63% replacement of career-average after-tax; high-z agents see sharply declining replacement rate due to AIME cap.

**Verdict:** CLEAR.

### 3.6 Income units

**Found:** AWI (Average Wage Index) units; 1 model unit ≈ $54,100 in 2019 dollars (Catherine 2025 §5.1; `docs/LABOUR.md §0`). `wealth_min = 0.13` ≈ $7,000 — chosen to skip the EGM constrained region rather than as an economically meaningful minimum (`_canonical.py:84`, `docs/STATE_SPACE.md §4`).

**Verdict:** CLEAR — but `wealth_min` is a **numerical knob** that looks like an economic floor; see §5.

---

## 4. Constraints

### 4.1 wealth_min = 0.13

**Found:** `wealth_min = 0.13` (`_canonical.py:87`). Per `_canonical.py:83-84`, this "skips the EGM constrained region" — i.e. the agent is **assumed liquidity-unconstrained** (no Lagrangian on the borrowing constraint), and the wealth grid simply doesn't extend below the cell where the constraint would bite. This is a **modelling assumption**, not just numerics: any agent who would have wanted to borrow at low W is being silently excluded from the policy domain.

**Verdict:** UNCLEAR — this is a real economic assumption hidden behind a numerical knob. **Suggested fix:** in `docs/CONFIG.md §3.1`, explicitly state "wealth_min implements the assumption that the agent never wants to dis-save below this level (or, equivalently, that the borrowing constraint is non-binding above wealth_min). At wealth_min = 0.13 (≈ $7k) this is plausible for the canonical calibration but should be cross-checked against the simulated wealth distribution's lower tail."

### 4.2 alpha_min / alpha_max

**Found:** No `alpha_min` / `alpha_max` field in the JAX `SolverConfig` (`lifecycle/model.py:128-229`). The legacy nominal model had a constrained Newton with leverage caps (`docs/CONFIG.md §2.6` documents `±6` cap); the JAX rewrite removed this branch (model.py docstring: "Canonical (unconstrained, JAX, CCV log-wealth) only. The constrained Newton, alpha leverage caps, edge/corner solvers, and the simple_clamp wealth-dynamics branch were removed in the JAX rewrite (handoff 2)."). So **the canonical post-pivot solver has NO leverage cap**. Backtracking line-search with `line_search_max_step = 2.0` (`model.py:165`) is the only step bound.

**Verdict:** UNDOCUMENTED in user-facing docs. `docs/CONFIG.md §2.6` is **stale** (still describes the ±6 cap as the only swept solver knob, and refers to `alpha_min`/`alpha_max` fields that don't exist on the new SolverConfig). **Suggested fix:** mark `docs/CONFIG.md` for a JAX-rewrite refresh; add a one-liner in `_canonical.py` (after the comment block) stating "Unconstrained portfolio; no leverage cap. Convergence relies on backtracking line search." Otherwise a thesis reviewer reading `CONFIG.md` will look for a cap that has been removed.

### 4.3 Other constraints

**Found:** No short-sale or margin constraint enforced in the JAX kernel. `α_bill = 1 - α_s - α_b` is unrestricted (can be negative ⇒ borrowing at the bill rate to fund stock/bond positions). The CCV log-portfolio formula `r_p = log(1 + α_s·(R_s - R_b) + α_b·(R_b - R_b) + R_b)` is taken at face value with arbitrary α (see Part B review for arbitrage implications).

**Verdict:** UNDOCUMENTED — there is no thesis-friendly statement of which constraints are imposed. **Suggested fix:** add a "Constraints" subsection to `docs/CONFIG.md §1` listing: (a) no borrowing constraint above wealth_min; (b) no short-sale constraint; (c) no margin requirement; (d) backtracking line-search bounded step (purely numerical).

---

## 5. Numerical vs economic parameters

**Found mixed/confusing knobs:**

| Knob | Looks economic | Actually... | File:line |
|---|---|---|---|
| `wealth_min = 0.13` | minimum wealth threshold | numerical: skips EGM constrained region (and silently embeds a no-borrowing assumption) | `_canonical.py:87` |
| `min_consumption = 1e-10` | consumption floor | numerical safety clip on interpolated c_next | `model.py:173` |
| `tiny_savings = 1e-6` | savings floor | numerical: below this, hold cold-init alphas | `model.py:168` |
| `delta_bequest = 0.0` | bequest shift parameter | economic but acts as a regulariser; canonical = 0 disables it | `_canonical.py:127`, `model.py:344` |
| `b_bar = 10` | bequest weight | bequest **horizon in years** (not a CGM warm-glow weight) | `model.py:32` |
| `wealth_max = 750.0` | wealth ceiling | numerical: above it the simulator rescales c by x/wealth_max | `_canonical.py:88`, `simulation.py:15-19` |
| `n_z = 11` | grid points | purely numerical | `_canonical.py:93` |

**Verdict:** UNCLEAR. The first three rows are the most likely thesis-reviewer landmines: `wealth_min` and `min_consumption` look like preference parameters but are numerics, while `b_bar` looks like a numerical horizon but is a fundamental preference parameter. **Suggested fix:** add a "Numerical vs Economic" callout to `docs/CONFIG.md` (or to `_canonical.py`'s top docstring) explicitly tagging each field.

---

## TL;DR — Verdicts

| # | Area | Verdict |
|---|---|---|
| 1.1 | CRRA u(c) | CLEAR |
| 1.2 | Bequest spec (W/A meaning) | CLEAR (in `UTILITY.md` only; weak inline) |
| 1.3 | δ = 0 in canonical (un-shifted) | CLEAR; sensitivity sweep recommended |
| 1.4 | b_bar = 10 | CLEAR |
| 1.5 | β = 0.96 annual | CLEAR |
| 1.6 | min_consumption | CLEAR (numerical) |
| 2.1 | Ages / off-by-one (last work age 66) | CLEAR |
| 2.2 | Earnings-dependent mortality | CLEAR |
| 2.3 | Bequest trigger (stochastic + terminal) | CLEAR |
| 3.1 | Age polynomial peak ≈ 46 | CLEAR; **`_canonical.py:33` polynomial expression is misleading (missing /10, /100)** |
| 3.2 | η mixture | CLEAR |
| 3.3 | ρ = 0.991 | CLEAR |
| 3.4 | pz, pe interpretation | UNDOCUMENTED inline (only in `LABOUR.md`) |
| 3.5 | Pension & AIME approximation | CLEAR |
| 3.6 | AWI units | CLEAR |
| 4.1 | wealth_min embeds borrowing-constraint assumption | UNCLEAR |
| 4.2 | alpha_min/alpha_max removed in JAX rewrite | UNDOCUMENTED (`CONFIG.md §2.6` stale) |
| 4.3 | No short-sale / margin constraint | UNDOCUMENTED |
| 5 | Numerical-vs-economic separation | UNCLEAR |

**Top thesis-reviewer flag:** `wealth_min = 0.13` is presented as a numerical grid knob in `_canonical.py` and `STATE_SPACE.md`, but it silently embeds the modelling assumption "the household is never borrowing-constrained above $7k." A thesis reviewer reading the config will not realise the canonical model has effectively dropped the EGM constrained branch — and the legacy `docs/CONFIG.md §2.6` description of an ±6 leverage cap is now **stale** (the JAX rewrite removed that branch entirely). Together these two omissions mean a reviewer asking "what constraints does the household face?" cannot answer from the canonical config + main docs alone.

**No potential bugs surfaced** in this review — every implementation matches its documented spec where docs exist; the gaps are documentation/positioning, not code.

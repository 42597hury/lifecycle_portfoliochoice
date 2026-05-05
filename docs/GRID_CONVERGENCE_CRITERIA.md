# Grid Resolution Convergence Criteria — Literature Review and Operational Thresholds

**Purpose.** Companion to `SOLVER_VALIDATION_PLAN.md`. That document specifies *what* sensitivity sweeps to run; this one specifies *what threshold the result must clear* for you (or a referee) to be confident the discretization is fine enough. Every threshold below is anchored to a published source so that "I picked K=5 because n=3 gave 1.2% policy drift" can be defended as "I picked K=5 because Judd (1998) Ch. 7 and AFV-RR (2006) call <1% Euler-error tolerable for production-grade work."

**Scope.** Aimed at the *discretization-sufficiency* question only. Implementation correctness is assumed and is the subject of Tier 1–3 of the validation plan.

**Notation.** Throughout, "EE" = unit-free Euler-equation error in consumption units; `K` denotes any discretization knob; `K_max` denotes the gold-standard reference setting from §4.9 of the validation plan.

---

## 1. The five literature-anchored criteria

A solver's discretization is judged "fine enough" if it clears five distinct tests. They are partly redundant — passing one usually implies the others — but each catches a failure mode the others can miss, and the literature uses all five as a battery rather than picking one.

### 1.1 Unit-free Euler-equation errors (Judd 1992; primary criterion)

**Source.** Judd (1992, "Projection methods for solving aggregate growth models," *J. Econ. Theory*) introduced the convention. Aruoba, Fernández-Villaverde & Rubio-Ramírez (2006, *J. Econ. Dyn. Control*) made the unit-free version standard in the comparative-methods literature. Den Haan (2010, *J. Econ. Dyn. Control*) catalogues what accuracy is achievable by method class.

**Definition.** At any state `(t, z, s, w)` your solver returns a consumption choice `c*`. The Euler equation residual in consumption units is

```
EE(t, z, s, w) = 1 - [ β · E_t [ R_p(t+1) · u'(c*(t+1)) ] / u'(c*(t)) ]^(-1/γ) / c*(t)
```

This is "what fraction of c* is the agent off by"? It is invariant to the units of c, hence "unit-free." The expectation `E_t` must be computed by an *independent, finer* quadrature than the one used inside the solver — otherwise you are testing the integration rule against itself, which is meaningless. AFV-RR (2006) use a 51-node tensor product for evaluation regardless of how the solver was set up; for your model, "evaluation grid 2× the production K in every dimension" is the standard prescription.

**Reporting.** Always log₁₀|EE|, evaluated at:

1. **Every grid point in policy space.** Report `mean`, `max`, and `99th percentile` per age. AFV-RR (2006), Table 4, reports both max and average; the gap between them is itself diagnostic (a wide gap means the worst error is concentrated near a singularity, usually the borrowing constraint).
2. **Along a stochastic simulation.** This is *the* number a referee will ask for. The simulated states are the only ones the model's *output* actually visits, so an Euler error of 10⁻³ at an unreachable corner of the grid is irrelevant. Den Haan (2010, Table 15) tabulates achievable accuracy *on a stochastic simulation* by method class.

**Thresholds.** The community standard is roughly:

| log₁₀|EE| | Interpretation | Source |
|-----------|----------------|--------|
| > −3 (≥ 0.1%) | Inadequate for any quantitative work | AFV-RR (2006), Den Haan (2010) |
| −3 to −4 (0.01% to 0.1%) | Acceptable for qualitative robustness work; weak for SMM/welfare | Christiano-Fisher (2000), Kollmann et al. (2011) — projection methods at this level |
| −4 to −5 (1 to 10 ppm) | Standard publication threshold for nonlinear DSGE | AFV-RR (2006), Maliar-Maliar (2014) |
| −5 to −6 (0.1 to 1 ppm) | Production-grade EGM with adequate quadrature | Carroll (2006), Barillas-FV (2007) |
| < −7 | Approaches floating-point limits in single precision; rarely needed | Den Haan (2010) |

**Gating rule for your model.** Given that you do welfare-equivalent comparisons (consumption-equivalent costs of imposing M-no-labor's allocation; see `Hypothesis_Specification.md`), you are in welfare territory and need the *publication threshold* of `mean log₁₀|EE| < −4` and `max log₁₀|EE| < −3` along the simulated path. Welfare measures are sensitive to the *level* of the value function, which is the *integral* of consumption errors over time — small per-period errors compound. Cocco, Gomes, Maenhout (2005) report welfare effects at the 0.1–4% consumption-equivalent scale, so per-period EE around 10⁻³ is barely below the welfare-effect magnitude — too close. Aim for ≤ 10⁻⁴.

**Notes on EGM specifically.** EGM has a useful property here: by construction, the FOC is satisfied *exactly* at every endogenous grid point (modulo the Newton tolerance, which you've set to 1e-7). Euler errors arise only from the *interpolation* step that maps endogenous-grid policies back to a regular grid for the next iteration. So EE measures interpolation error in the consumption function more than it measures FOC convergence. This is exactly what you want to measure when sweeping `n_wealth`.

### 1.2 Self-convergence (Cauchy / Richardson criterion)

**Source.** Boyd (2001, *Chebyshev and Fourier Spectral Methods*, §2.13); the dynamic-programming version is the "halve the step / double the nodes and look" practice used routinely in numerical analysis. The Carroll (2006) EGM paper and Barillas-Fernández-Villaverde (2007) both report Euler-error sensitivity tables that are essentially Richardson tables.

**Definition.** Pick a summary statistic `S(K)` (a moment, a policy average, a value function level). Compute it at increasing K. If the underlying problem is being correctly approximated, `S(K)` should form a Cauchy sequence — successive differences should shrink at a rate determined by the approximation order.

For *spectral* methods (Chebyshev, Gauss-Hermite on smooth integrands), convergence is exponential: `|S(K+1) − S(K)| / |S(K) − S(K−1)| → 0` quickly. For *linear interpolation* (your wealth-grid policy lookup), convergence is polynomial: `|S(K+1) − S(K)| ~ K⁻²`. Both signatures are diagnostic — if you see polynomial decay where you expect exponential (e.g., on Gauss-Hermite over a smooth Gaussian integrand), something is wrong (heavy tails, kink in the integrand, …).

**Threshold.** `|S(K_prod) − S(K_max)| / |S(K_max)|` < your target tolerance. The validation plan suggests 0.5% on policy averages and 1pp on simulated portfolio shares — these are loose by literature standards.

| Tolerance | Source / context |
|-----------|------------------|
| 5% | Inadequate. Too coarse for quantitative work. |
| 1–2% | Den Haan (2010): this is the low end of what most published projection-method work achieves on lifecycle moments. |
| 0.5% | Standard in CGM (2005)-style papers — they don't report it but their grids are sized to deliver it. |
| 0.1% | Aim point for a publishable life-cycle paper. Achievable with EGM. |
| 0.01% | Probably wasted compute in a model with this many state variables. |

**Cross-validation issue.** The most common mistake (called out in §4.9 of your validation plan) is to converge each dimension *holding others fixed at the production level*. This can give a false convergence reading because the bottleneck dimension absorbs all the error. Always measure against the *gold-standard* config (everything maxed) rather than the production baseline. This is the Richardson-extrapolation analog of the von Neumann stability test in PDE work.

### 1.3 Den Haan-Marcet (1994) residual orthogonality test

**Source.** Den Haan & Marcet (1994), "Accuracy in Simulations," *Rev. Econ. Stud.* 61(1), 3–17.

**Definition.** Under rational expectations, the Euler residual `u_{t+1} = β R_{t+1} u'(c_{t+1}) − u'(c_t)` should be orthogonal to any function of the time-`t` information set: `E[u_{t+1} · h(x_t)] = 0`. The DM test simulates the model, picks an instrument set `h(x_t)` (typically `{1, x_t, x_{t-1}, …}`), forms the sample analog of the orthogonality condition, and tests via χ² whether it's significantly different from zero.

**Why it matters here.** The DM test is most powerful for *projection / parameterized-expectations* methods, where the candidate solution is a parametric approximation of the policy. For *backward-induction* methods with the expectation computed by quadrature (your case), the test is less powerful because the integration rule is an internal choice, not a parametric ansatz. Den Haan & Marcet (1994) §2 explicitly note: "for the backward solution procedure … one would have to test if the innovations to the … " (i.e., test the residuals' iid property rather than orthogonality).

**Operational form for your model.** For each agent in the simulation, at each working-age year, compute the per-period FOC residual using a high-resolution evaluation quadrature (the same one used for the Euler-error metric in §1.1). Test whether the residual sequence is:
1. Mean-zero (a t-test on the simulation cross-section)
2. Serially uncorrelated within agent (Ljung-Box on the time series)
3. Uncorrelated with lagged states (regression of residual on (lag wealth, lag z, lag s); F-test on slopes)

In a correctly-specified, finely-discretized solve, all three should fail to reject. If (1) fails, you have a *bias* (likely from too few quadrature nodes — discrete distribution doesn't match continuous mean). If (2) or (3) fails, you have *systematic prediction errors* (likely from too coarse a wealth or z grid).

**Threshold.** 5% rejection rate at conventional significance levels. Den Haan & Marcet (1994) report results in terms of upper- and lower-tail rejection percentages; an accurate solution sits near 5% on both tails. Anything > 10% indicates a problem.

### 1.4 Markov chain moment-matching (Kopecky-Suen 2010 criterion)

**Source.** Kopecky & Suen (2010), "Finite State Markov-chain Approximations to Highly Persistent Processes," *Rev. Econ. Dyn.* 13, 701–714. Earlier work: Tauchen (1986), Tauchen-Hussey (1991), Floden (2008).

**Setting.** This applies to your two AR-driven discretizations:
- The persistent income state `z` (currently `n_z = 11`, AR(1) with ρ ≈ 0.97-0.98 typical for labor income)
- The financial state grid `state_grid` (currently 7³, derived from a VAR(1) with all eigenvalues < 1 in modulus but some near unity)

**Result.** For an AR(1) `z_{t+1} = ρ z_t + ε`, the discretized Markov chain should *match* (not approximate) the continuous-process moments:
- Unconditional mean `E[z]` — match exactly
- Unconditional variance `Var[z] = σ²_ε / (1 − ρ²)` — match exactly
- First-order autocorrelation `ρ` — match exactly
- Conditional mean `E[z_{t+1} | z_t]` — match exactly
- Conditional variance — match exactly (within node resolution)

Rouwenhorst's (1995) method matches all five for *any* `n` ≥ 2 (Kopecky-Suen 2010 prove this). Tauchen-Hussey only matches them in a polynomial-exactness sense and degrades for ρ > 0.9. **For ρ in the labor-income range (0.95–0.99), Kopecky-Suen show Rouwenhorst with `n = 5` outperforms Tauchen-Hussey with `n = 25`.**

**Operational test.** Verify that your discretized chain reproduces:
1. `μ_disc - μ_true` < 1e-12 (machine precision; unconditional mean)
2. `(σ²_disc − σ²_true) / σ²_true` < 1e-3 (unconditional variance, relative)
3. `ρ_disc − ρ_true` < 1e-3 (autocorrelation)
4. `(σ²_cond, disc − σ²_cond, true) / σ²_cond, true` < 1% on average across origin states (conditional variance)

These tests are essentially free — you compute them once at precompute time. They detect (a) bugs in the discretization code, (b) failure modes when grids are too coarse or too narrow.

**For your VAR.** The 3-D state VAR is more delicate. Gospodinov-Lkhagvasuren (2014) provide a direct extension of Rouwenhorst to VARs with the same moment-matching property, but most papers (including Catherine 2025) just use a tensor-product cholesky-axis grid plus Gauss-Hermite quadrature for the innovations — what you do. The diagnostic that matters then is whether `Pi_state @ E[s] = E[s]` (stationary distribution recovers the unconditional mean) and `Σ̂_state - Σ_state_target` is small. Your `verify_discretization.ipynb` already does the second of these (§A.5); ensure the first is also reported.

### 1.5 Quadrature: polynomial exactness vs integrand curvature

**Source.** Judd (1998, *Numerical Methods in Economics*, Ch. 7); Stroud & Secrest (1966); Trefethen (2022, "Exactness of quadrature formulas," *SIAM Rev.*).

**Standard fact.** An `n`-point Gauss-Hermite rule integrates polynomials of degree `≤ 2n − 1` *exactly* against the standard normal density. This is the polynomial-exactness order. Whether that's enough for your CRRA integrand depends on how well the integrand is approximated by polynomials of that degree on the support where the Gaussian density has appreciable mass.

**Heuristic for CRRA portfolio choice.** The integrand inside the FOC is

```
g(ε) = R_p(ε)^(1−γ) · 1{c'(W' (ε))}
```

For γ = 6 and a one-standard-deviation move in returns of, say, 17%, `R_p^(1−γ)` swings by a factor of `(1.17 / 0.83)^5 ≈ 11`. A polynomial of degree `2n−1` can approximate that curvature if and only if the function has its Taylor remainder under control on the integration support. Practically (Trefethen 2022, §5):

| Integrand | Recommended `n` |
|-----------|-----------------|
| Smooth (analytic), light tails | n = 3–5 |
| CRRA with γ ≤ 4, light-tailed shocks | n = 3 |
| CRRA with γ ≥ 6, light-tailed shocks | n = 5–7 |
| CRRA with γ ≥ 6, heavy-tailed shocks (kurtosis > 5) | n ≥ 9, or stratified rule |
| CRRA with kurtosis > 50 (your eps shock) | Standard rules fail; use specialized rules or accept bias |

The last row is your `n_eps` problem. Your `verify_discretization.ipynb` §C.5 already documented that `n_eps = 5` leaves > 50% relative error at γ = 8. This is a *known impossibility*: for a mixture with kurtosis 52, no low-order Gauss rule resolves the tail. Mitigation options (in order of cost):

1. **Reduce γ in the headline run** — your `Hypothesis_Specification.md` already plans γ ∈ {3, 5, 7, 10}; flag γ = 10 as having higher quadrature uncertainty.
2. **Use a stratified GH rule** — dedicate a sub-rule to each component of the mixture, then combine. Older code did this; the move to Judd's rule sacrificed it for fewer nodes.
3. **Truncate the eps shock** — economically defensible if extreme tails are less plausible than the mixture suggests.
4. **Document the bias explicitly** — report bias estimates from `n_eps` ∈ {5, 9, 13} and let the referee judge.

**Cross-check via integrand curvature.** Compute `E[exp(−γ·ε)]` (the marginal-utility integrand at the production γ) by your quadrature rule and compare to the analytic value (closed-form for a 2-component normal mixture). If the relative error exceeds 1e-3 at production γ, your quadrature is undersized.

---

## 2. Mapping criteria to your discretization knobs

Each of your nine knobs maps to a specific subset of the criteria above. The table below tells you what to measure and what threshold to clear.

| Knob | Default | Primary criterion | Secondary criterion | Threshold (publication) | Source |
|------|---------|-------------------|---------------------|-------------------------|--------|
| `n_wealth` | 150 | EE on simulated path (§1.1) | Self-convergence of policies (§1.2) | mean log₁₀|EE| < −4 along sim; <0.5% policy drift vs n=300 | Carroll (2006), AFV-RR (2006) |
| `n_savings` | 150 | EE on simulated path | Self-convergence of EGM consumption | Same as `n_wealth` (the two are coupled) | Barillas-FV (2007) |
| `state_grid_sizes` | 7³ | Self-convergence of moments (§1.2) | Stationary-distribution check (§1.4) | <1pp drift in equity share (lifecycle avg) vs 9³ | Cocco-Gomes-Maenhout (2005) implicit |
| `n_z` | 11 | Moment-matching of discretized chain (§1.4) | EE on sim | Conditional variance error <1%; mean <0.5pp drift in retirement wealth vs 13 | Kopecky-Suen (2010) — Rouwenhorst floor |
| `n_eps` | 3 | Polynomial exactness vs integrand (§1.5) | EE on sim | rel error on `E[exp(−γε)]` < 1e-3; or document bias if infeasible | Judd (1998) Ch. 7; your §C.5 |
| `n_eta` | 3 | Polynomial exactness | EE on sim | rel error on `E[exp(−γη)]` < 1e-4 | Judd (1998); LABOUR.md analysis |
| `n_ret_nodes_1d` | (3,3,3) typical | Polynomial exactness vs `R^{1−γ}` | EE on sim | Per-axis error < 1e-3 on the test integrand at production γ; <1pp policy drift | Trefethen (2022); empirical your H1b |
| `n_state_quad_nodes` | 3 | Polynomial exactness on `exp(a·v)` | Self-conv. of `mu_r` | rel err < 1e-6 on test integrands; <0.5pp policy drift to K=5 | Stroud-Secrest (1966) |
| `wealth_max` (boundary) | 200 | Boundary-mass simulation diagnostic | — | <0.5% of agent-years at upper bound | Standard (open item in your TODO) |

Reading this table: most criteria reduce to a small number of headline quantities. If you measure (a) the Euler-error distribution along the simulation, (b) the self-convergence of three or four headline moments vs the gold-standard config, and (c) the moment-matching of your Markov chains against their continuous targets — you have covered the literature's full battery.

---

## 3. Recommended tiered thresholds for your sweep

Use the table below as the *gating criterion* for declaring a sweep run "converged" at production resolution. Each tier is a step up — most papers ship at "publication-grade." "Welfare-grade" is needed if your headline is welfare-equivalent (which yours is).

| Diagnostic | Acceptable | Publication-grade | Welfare-grade |
|------------|------------|-------------------|---------------|
| `mean log₁₀|EE|` along sim, working-age | < −3 | < −4 | < −5 |
| `max log₁₀|EE|` along sim, ex-borrowing-constraint kink | < −2 | < −3 | < −4 |
| `mean log₁₀|EE|` along sim, retirement | < −3.5 | < −4.5 | < −5.5 |
| Self-conv. of avg equity share vs gold std | < 2 pp | < 0.5 pp | < 0.1 pp |
| Self-conv. of avg consumption-to-wealth vs gold std | < 2 % | < 0.5 % | < 0.1 % |
| Self-conv. of median wealth at age 60 vs gold std | < 5 % | < 1 % | < 0.5 % |
| Markov chain `Var(z)_disc / Var(z)_true − 1` | < 5 % | < 1 % | < 0.1 % |
| Markov chain `ρ_disc − ρ_true` | < 0.01 | < 0.005 | < 0.001 |
| Stationary distribution `‖π · Pi − π‖_∞` | < 1e-4 | < 1e-6 | < 1e-9 |
| Boundary-mass at `wealth_grid[-1]` in sim | < 2 % | < 0.5 % | < 0.1 % |
| Boundary-mass at `z_grid` extremes in sim | < 5 % | < 2 % | < 0.5 % |
| Quadrature: rel err on `E[exp(−γ·ε)]` | < 1e-2 | < 1e-3 | < 1e-4 |
| Newton convergence rate (free; from §1.1 of plan) | > 99 % | > 99.9 % | > 99.99 % |
| FOC residual 99th-percentile | < 100 × tol | < 10 × tol | < 5 × tol |
| DM-style residual orthogonality test rejection rate | n/a | < 10 % | < 5 % |

**For your specific paper.** Given the Hypothesis_Specification.md emphasizes welfare-equivalent costs and a duration-substitution mechanism that operates on small-magnitude differences (a few pp on equity/bond shares, a fraction-of-a-percent welfare cost), you should target the **welfare-grade** column for the headline-relevant diagnostics: Euler errors, self-convergence on equity share, and Markov-chain moment matching. Boundary-mass and Newton-rate items can ship at publication-grade.

---

## 4. Concrete protocol for the sweep run

This operationalises §3 into a worked sequence you can execute. It complements §4 of `SOLVER_VALIDATION_PLAN.md` and assumes you already have the diagnostic infrastructure from `diagnostics.py` and `state_distribution_diagnostic.py`.

**Step 0. Compute the gold-standard reference once.**

Per §4.9 of the validation plan: a single solve at `n_wealth = 300, n_savings = 300, state = 9³, n_z = 13, K_state = 5, K_ret = (5, 7, 5), n_eps = 7, n_eta = 5`. Cache the policy arrays and the simulated moments (1 large-N simulation, say `N = 200000`). All sweep results below are evaluated *against this reference*, not against the production baseline.

**Step 1. Compute Euler errors at production config.**

Run the production solve. Then, on a 5000-agent simulation (your existing simulation pipeline), at each working-age year and each agent, evaluate the FOC using a doubled-resolution quadrature (`K_ret`, `K_state`, `n_eps`, `n_eta` all doubled vs production). Record the EE. Tabulate `mean`, `99th percentile`, `max` per age. Plot log₁₀|EE| vs age.

This is *the* number to report. AFV-RR (2006) Table 4 is the visual template.

**Step 2. Sweep each dimension; tabulate against gold standard.**

For each knob `K` and each setting in {coarse, default, fine}:
- Produce policy averages at fixed `(t, i_z, i_s)` for several representative tuples (median z, median s; lowest z, lowest s; highest z, highest s)
- Simulate fixed shocks against the new policy
- Compute mean/median/p25-p75 of: lifetime average c, equity share at 30/45/60, bond share at 30/45/60, wealth at 30/45/60, value function at age 30
- Report all numbers as `(value − gold_value) / |gold_value|`

For `n_wealth`, `n_savings`, `n_z` — also report `mean log₁₀|EE|` per setting. These are the dimensions where EE is most informative.

For `n_eps`, `n_eta`, `n_ret_nodes_1d`, `n_state_quad_nodes` — also report the relative error on the analytic test integrand `E[exp(−γ·ε)]` (or its multivariate equivalent).

**Step 3. Diagonal vs cross-validation.**

Step 2 sweeps each knob with others held at production. Step 3 holds each knob at production and varies *all others* up to gold standard. The cross-table tells you which knob is the bottleneck. This is the test that will catch the §4.9 trap.

**Step 4. Markov-chain diagnostics.**

For the persistent income chain at each `n_z`, report:
- `(Var_disc − Var_true) / Var_true`
- `ρ_disc − ρ_true`
- `||π_disc · Pi − π_disc||_∞`
- Conditional variance error averaged across origin states

Same for the state VAR at each `state_grid_sizes`. Your `verify_discretization.ipynb` §A.5 already does most of this — just promote the numbers from "informational" to "gating."

**Step 5. Boundary diagnostics.**

For the production solve, simulate. Report:
- Fraction of agent-years at `wealth_grid[0]` (excluding the borrowing-constraint corner that's economically valid)
- Fraction at `wealth_grid[-1]` (the dangerous one — extrapolation kicks in)
- Fraction at `z_grid[0]` and `z_grid[-1]`
- Fraction at each face of the `state_grid` cube

Anything > 0.5% at the upper wealth or z-grid extremes means you should extend the grid before declaring convergence. You already flag this as an open item in `RETURNS.md` §6.8.

**Step 6. DM-style residual orthogonality (optional but defensible).**

On the simulated path at production config, regress the Euler residual on `(lag c, lag wealth, z, all three state variables)`. Report F-statistic and p-value. Against `H_0`: solution is accurate, `H_0` should fail to reject 95% of the time.

This is overkill for most papers but distinguishes "bias from too-coarse grid" (residual is correlated with lag wealth) from "bias from too-few quadrature nodes" (residual is mean-non-zero) cleanly.

---

## 5. Notes on specific dimensions

A few dimensions warrant extra discussion because the literature has specific guidance.

### 5.1 Wealth grid (`n_wealth`)

CGM (2005) used 81 cash-on-hand grid points (linearly spaced) in their original paper. With geometric spacing concentrating points near zero (your default), 100–150 typically delivers 10⁻⁴ Euler errors at γ = 5; 200 is needed at γ = 10. The bottleneck is the curvature of `c(W)` near W = 0, which scales as `W^(1−1/γ)` near the constraint and gets sharper as γ rises. Carroll (2006) shows EGM is competitive with much smaller grids than VFI for the same accuracy because the endogenous grid concentrates points where the policy bends.

**For your model:** `n_wealth = 150` should comfortably hit publication-grade at γ ≤ 7. At γ = 10 in your sweep, push to 200–250 and verify.

### 5.2 Persistent income grid (`n_z`)

Cocco, Gomes, Maenhout (2005) used 9 points; Catherine (2021) used 11; Fagereng, Gottlieb, Guiso (2017) used 21. Kopecky-Suen (2010) show that **for ρ > 0.95, no number of Tauchen-Hussey points is reliably enough**, while Rouwenhorst with `n = 5` already matches all five moments. Your model uses Tauchen-Hussey-style discretization; if your ρ is in the 0.97–0.99 range, switch to Rouwenhorst before adding points.

**For your model:** check which method `discretization.py` uses. If Tauchen-Hussey, the marginal benefit of going from 11 to 13 is small *and* you may have systematic bias regardless. Switching to Rouwenhorst is more impactful than adding nodes. (Open task: read the `discretization.py` source to confirm.)

### 5.3 Joint financial state grid (`state_grid_sizes`)

This is your most expensive dimension (cubic scaling) and the one with the weakest literature precedent — most life-cycle papers don't have a 3-D persistent return state. Closest analog: Fugazza-Gomes-Campanale (2015) and the earlier Lynch (2001), who use approximations on 1-D return predictability. A two-dimensional version (Brennan-Xia 2002) used 21×21 grids on a discretized OU process. For 3-D, **the literature shipping standard is roughly 7³** and what you have is consistent with that.

**Convergence-rate expectation:** under tensor-product Gauss-Hermite for a smooth integrand, going from 5³ to 7³ should improve accuracy by ~10× (each axis goes from poly-exactness order 9 to 13). 7³ to 9³ should improve by another factor of 5–10×. If the actual self-convergence rate from your sweep is much slower than this, you have either (a) a non-smooth integrand somewhere (the borrowing constraint? the corner-arbitrage states flagged in `RETURNS.md` §6.12?), or (b) interpolation error dominating (the snap-to-nearest-grid in simulation, §3.5 of `STATE_SPACE.md`).

### 5.4 Return quadrature `n_ret_nodes_1d`

Already analyzed extensively in your `verify_discretization.ipynb` §B.4 and the H1b investigation. The asymmetric `(K_rtb, K_xr, K_xb)` form is the right structure — Cocco-Gomes-Maenhout used a uniform 3 nodes per dim. Your finding that the bond residual axis (xb) was undersized at K=3 and that 5–7 helps is a *finer* analysis than CGM (2005) did. Document the per-axis choice; this is publishable methodological care, not over-engineering.

### 5.5 Income shock quadrature `n_eps`, `n_eta`

The kurtosis-52 transitory shock is genuinely hard. Catherine (2025) uses the same shock structure; her Appendix should document her quadrature choice (worth checking). The fact that no Gauss-type rule resolves the tail at γ ≥ 5 is a recognized issue in the literature on Guvenen et al. (2021)-style processes. Most papers either (a) use sparse-grid Gauss-Hermite with high `n`, (b) use Kotlikoff-style importance sampling with a separate tail quadrature, or (c) just acknowledge bias at high γ.

**For your model:** the cleanest defense is to sweep `n_eps` ∈ {3, 5, 9} and *report the bias estimate at each γ*. A single sentence in the appendix — "at γ = 10, the residual quadrature error in the transitory income shock implies approximately 0.X% bias in the equity share" — is much stronger than ignoring the issue.

---

## 6. Quick reference — checklist for the sweep report

When the sweep is complete, the appendix or referee response should include:

- [ ] Gold-standard config specified, with runtime and memory numbers
- [ ] Per-knob sweep tables: setting × {policy averages, simulated moments, Euler errors, runtime}, all relative to gold standard
- [ ] One headline figure: `log₁₀|EE|` vs age along simulation, at production config
- [ ] Markov-chain moment-match table: `(disc − true)` for mean, var, autocorr at each `n_z` and `state_grid_sizes`
- [ ] Boundary-mass percentages in simulation at production config
- [ ] One sentence justifying the production config: "We ship at K=… because the marginal change to gold standard is below [target] in all reported moments while [coarser] gives [target+]."
- [ ] One paragraph acknowledging the residual quadrature error in `n_eps` at high γ

This is what AFV-RR (2006), Den Haan (2010), and the Maliar surveys provide for their workhorse models. For a life-cycle paper using a published methodology backbone (Catherine 2025), this level of validation is more than sufficient and visibly more thorough than the field median.

---

## 7. References

1. Aruoba, Fernández-Villaverde & Rubio-Ramírez (2006), "Comparing solution methods for dynamic equilibrium economies," *J. Econ. Dyn. Control* 30, 2477–2508.
2. Barillas & Fernández-Villaverde (2007), "A generalization of the endogenous grid method," *J. Econ. Dyn. Control* 31, 2698–2712.
3. Carroll (2006), "The method of endogenous gridpoints for solving dynamic stochastic optimization problems," *Economics Letters* 91, 312–320.
4. Cocco, Gomes & Maenhout (2005), "Consumption and portfolio choice over the life cycle," *Rev. Financial Studies* 18, 491–533.
5. Den Haan & Marcet (1994), "Accuracy in simulations," *Rev. Econ. Studies* 61, 3–17.
6. Den Haan (2010), "Comparison of solutions to the incomplete markets model with aggregate uncertainty," *J. Econ. Dyn. Control* 34, 4–27.
7. Floden (2008), "A note on the accuracy of Markov-chain approximations to highly persistent AR(1) processes," *Economics Letters* 99, 516–520.
8. Gospodinov & Lkhagvasuren (2014), "A moment-matching method for approximating vector autoregressive processes by finite-state Markov chains," *J. Applied Econometrics* 29, 843–859.
9. Judd (1992), "Projection methods for solving aggregate growth models," *J. Econ. Theory* 58, 410–452.
10. Judd (1998), *Numerical Methods in Economics*, MIT Press, Chs. 7 (quadrature) and 12 (DP).
11. Kopecky & Suen (2010), "Finite state Markov-chain approximations to highly persistent processes," *Rev. Econ. Dynamics* 13, 701–714.
12. Maliar & Maliar (2014), "Numerical methods for large-scale dynamic economic models," in *Handbook of Computational Economics*, vol. 3, ch. 7.
13. Rouwenhorst (1995), "Asset pricing implications of equilibrium business cycle models," in *Frontiers of Business Cycle Research*, Princeton.
14. Stroud & Secrest (1966), *Gaussian Quadrature Formulas*, Prentice-Hall.
15. Tauchen (1986), "Finite state Markov-chain approximations to univariate and vector autoregressions," *Economics Letters* 20, 177–181.
16. Tauchen & Hussey (1991), "Quadrature-based methods for obtaining approximate solutions to nonlinear asset pricing models," *Econometrica* 59, 371–396.
17. Trefethen (2022), "Exactness of quadrature formulas," *SIAM Review* 64, 132–170.

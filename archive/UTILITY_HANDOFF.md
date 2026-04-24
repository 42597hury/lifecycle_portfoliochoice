# UTILITY VALIDATION HANDOFF

## Task

Build a `UTILITY.md` file in the same style and depth as `LABOUR.md`. This document should consolidate the complete specification, implementation mapping, and validation status of the three components of the agent's objective function: **CRRA utility**, **mortality/survival risk**, and **bequest utility**. The document is the single source of truth for anyone auditing or extending the utility specification.

## Scope

The utility side of the model has three components that jointly determine the agent's objective:

1. **CRRA consumption utility** — `u(c) = c^{1-γ}/(1-γ)`, its derivatives, and how they enter the Euler equation
2. **Survival risk** — age- and earnings-dependent mortality `m(age, z)` following Catherine (2025, eq. 35) calibrated to Chetty et al. (2016) life-expectancy targets
3. **Bequest utility** — Catherine (2025, eqs. 21-22), where bequeathed wealth is converted to a flow-equivalent annuity consumption stream

These three interact in the Bellman equation as:

```
V_t(W) = max_{c,α}  u(c) + β · E[ ψ_t · V_{t+1}(W') + (1-ψ_t) · b(a·R_p, r_ft) ]
```

where ψ_t = 1 - m_t is the survival probability.

## What We Have Validated So Far

### Bequest utility — VALIDATED

The bequest term follows Catherine (2025) Section 3.6, equations 21-22:

```
b(W, r_f) = b̄ · (W/A)^{1-γ} / (1-γ)

where:
  C̄ = W / A(y_nom)                    flow-equivalent consumption
  A(y) = (1 - (1+y)^{-b̄}) / y        annuity factor (flat yield approx.)
  b̄ = 10                              bequest horizon (years of heir consumption)
```

**Economic interpretation:** The bequest is valued as if the agent lives b̄ = 10 extra years on a fixed consumption stream C̄, where C̄ is the annual coupon from investing the estate in a b̄-year annuity at the current nominal yield.

**Validated items:**

- **Functional forms correct:** `bequest_utility`, `bequest_marginal`, `bequest_marginal_inv` in `model.py` all match Catherine eqs. 21-22 exactly. The marginal `db/dW = b̄ · (W/A)^{-γ} / A` and its inverse `W = A · (μA/b̄)^{-1/γ}` are algebraically verified.
- **Second derivative (Jacobian):** `mup_bequest = -γ · mu_bequest / (w_A · A)` in `solver.py` matches `d²b/dW² = -γ b̄ (W/A)^{-γ-1} / A²`. Used in the Newton portfolio solver's Jacobian.
- **Timing verified against Catherine:** (a) death occurs between t and t+1; (b) bequeathed wealth = a_t · R_p (invested savings, no income); (c) annuity factor uses current-period yield `annuity_factors[i_s]`, not next-period yield — matches Catherine's subscript r_ft; (d) β discounts both alive and dead branches equally; (e) alive branch gets income (W' = aR_p + Y'), dead branch does not.
- **Terminal age special structure:** At age 99, death is certain. Portfolio decouples from consumption due to CRRA homogeneity. Portfolio minimizes E[R_p^{1-γ}] over the simplex (Merton problem). Consumption is a constant fraction of wealth: c* = W · ratio/(ratio+1) where ratio = (β·Ω)^{-1/γ} and Ω = b̄ · A^{γ-1} · E[R_p^{1-γ}]. Diagnostic output confirms c/x = 10.4% at all wealth levels.
- **Bequest hoist optimization valid:** In working-age FOC, bequest contribution is hoisted outside the (k_eta, i_e) income quadrature loops because bequest depends only on invested wealth (j_s, k_r), not on income realization. Valid because sum of income quadrature weights = 1.

**Known approximation — document but do not fix:**

The annuity factor uses a flat term structure: A(y) = Σ(1+y_{10yr})^{-k}, discounting all 10 payments at the 10-year yield. Catherine's paper uses zero-coupon bond prices P_kt for each maturity k (eq. 22), but his model has a zero term premium, making the flat approximation exact. Our model does NOT have zero term premium (the VAR allows a spread between the bill rate and 10-year yield), so a duration-matched discount rate would be more accurate. The annuity's Macaulay duration is ~4.5-5.5 years, suggesting the correct single-rate approximation lies between the bill rate and the 10-year yield. Quantifying the bias and potentially implementing a duration-matched rate from a portfolio of bills and bonds is deferred to returns validation. For now, document the approximation and its direction (current A is too small → C̄ too large → bequest value overstated).

### CRRA utility — NOT YET VALIDATED

The `make_crra_utils(gamma)` function in `model.py` returns `(u, u_prime, u_prime_inv)`. Needs:
- Verify the three functions are algebraically consistent (u_prime_inv(u_prime(c)) = c)
- Verify the γ=1 (log) special case
- Document how u_prime enters the Euler equation and how u_prime_inv is used in the EGM inversion step
- Trace through the EGM line: `c_opt = (beta * euler) ** (-1.0 / gamma)` and confirm it equals `u_prime_inv(beta * euler_sum)`

### Mortality — NOT YET VALIDATED  

The mortality module `mortality.py` implements Catherine (2025, eq. 35). Needs:
- Document the SSA 2017 life table source and gender-averaging
- Document the Chetty et al. (2016) calibration targets (life expectancy at age 40 by income percentile)
- Verify the chi(z_i) root-finding procedure
- Verify that survival_probs_2d has correct shape (n_age, n_z) and is correctly indexed in the solver
- Verify the mapping from z-grid to income percentile via Φ(z/σ_z)
- Check boundary behavior: does m(age, z) ever exceed 1? (Capped by min(..., 1))
- Confirm terminal age has ψ = 0 (certain death at 99) or document what actually happens

## Format Instructions

Follow the structure of `LABOUR.md` exactly:
1. Start with a **Section 0** on units and context
2. Then walk through the **theoretical specification** (equations, parameters, sources)
3. Map each equation to its **code implementation** (file, line, function name)
4. Present **validation items** as a checklist with `[x]` (validated) or `[ ]` (pending)
5. For each validated item, include the specific evidence (numerical test, algebraic derivation, or code trace)
6. For known approximations, explain the economic direction of the bias

## Key Code References

- `model.py`: `make_crra_utils()`, `annuity_factor()`, `bequest_utility()`, `bequest_marginal()`, `bequest_marginal_inv()`
- `mortality.py`: `calibrate_earnings_dependent_mortality()`, SSA life table data, Chetty targets
- `precompute.py`: `annuity_factors` array construction, `survival_probs_2d` construction
- `solver.py`: retirement FOC (`compute_foc_jac_retirement`), working FOC (`compute_foc_jac_working`), terminal solver (`solve_terminal_age`), EGM inversion (`c_opt = (beta * euler) ** (-1.0/gamma)`)
- `DESIGN.md`: Sections 1.1, 1.1b, 4.4, 4.7 for the design-level specification

## Key Paper References

- Catherine (2025): eq. 21-22 (bequest), eq. 35 (mortality), Section 5.1 (calibration), Appendix C.4 (EZ value function)
- Cocco, Gomes & Maenhout (2005): eq. 1 (preference structure with mortality and bequest — the CGM formulation your model nests as a special case)
- Guvenen et al. (2021): income process parameters (already in LABOUR.md but mortality uses the same z-grid)
- Chetty et al. (2016, JAMA): life expectancy by income percentile (mortality calibration targets)

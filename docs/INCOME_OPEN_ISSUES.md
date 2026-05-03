# Income & Social Security — Open Issues Summary

All issues found during the review session of 8 April 2026, ordered by economic relevance.

---

## CRITICAL — Must fix before results can be trusted

### 1. Pi_z transition matrix has no upward transitions (n_z = 11)

**Status:** Diagnosed, not yet fixed. Handoff written for Pi_z diagnostic.

**The problem:** The Tauchen bin-probability method discretizes the persistent income AR(1) onto an 11-point grid with spacing dz = 1.12. The mixture-normal innovation has two components: a narrow positive-drift component (82.4% weight, μ₂ = +0.112, σ₂ = 0.046) and a wider negative-shock component (17.6% weight, μ₁ = −0.524, σ₁ = 0.113). At dz = 1.12, the positive component's entire mass lands inside the current bin every period (half_bin = 0.56 >> μ₂ = 0.112), so the discretization registers it as "stay." Only the negative component is large enough to cross a bin boundary. The resulting transition matrix has P(up) = 0 at every row.

**Economic consequence:** Agents can only get poorer. Starting from z = 0 at age 22, the median agent reaches z = −2.24 by retirement — a factor of 10× too low in income. This distorts saving (too much precautionary saving against an artificially grim income future), pension benefits (median pension 0.048 vs correct 0.284), and portfolio choice (agents facing declining income behave very differently from agents facing stable income).

**The decision to make:** Either increase n_z to ~51 for correct Tauchen transitions (5× solver cost), or switch to Rouwenhorst at n_z = 11 (correct first two moments, loses the mixture's skewness of −1.73 and kurtosis of 4.42). There is no cheap middle ground with Tauchen — any N between 11 and ~47 is strictly worse than Rouwenhorst at 11. We don't know what Catherine uses.

**Impact on every result:** This affects all lifecycle dynamics — income profiles, saving rates, consumption paths, pension adequacy, portfolio allocations, wealth accumulation. It is the single most important issue.

---

## MODERATE — Should fix or explicitly document

### 2. Standard Tauchen edge bins

**Status:** Handoff written (TAUCHEN_FIX_HANDOFF.md).

**The problem:** The first and last bins use finite boundaries instead of extending to ±∞. At N = 11 this makes zero numerical difference (verified: max |diff| = 0 at every entry). At higher N it matters only at the 2–3 outermost rows that agents rarely visit.

**Impact:** Negligible at any practical grid size. Fix is for methodological correctness and alignment with the standard literature, not for numerical results.

### 3. Model unit documentation: $54,100, not $61,000

**Status:** Confirmed from Catherine (2025) Section 5.1. Not yet updated in code/docs.

**The fact:** Catherine states initial wealth = "0.1 × the national wage index, the equivalent of $5,400 in 2019." This pins 1 model unit = SS Wage Index ≈ $54,100 (2019 dollars). The tax brackets match the 2019 TCJA thresholds at this scaling to within 1–2%. Previous documentation used "$61k" which was derived by incorrectly matching to 2023 brackets.

**Impact:** Doesn't affect any computation (all internal calculations are in model units). Affects thesis text, tables, and any real-dollar interpretations.

### 4. working_income array is larger than needed

**Status:** Diagnosed, low priority.

**The fact:** `working_income` has shape `(n_age, n_z, n_eps)` — at the production
setting `(n_age, n_z, n_eps_nodes) = (78, 11, 5)` under the Judd-mixture
quadrature (was `(78, 11, 10)` under the previous concatenated-GH rule
at the same `n_eps_nodes=5`). The array covers all ages 22–99, but only
ages 22–67 (indices 0–45) are working ages. Rows 46–77 contain valid
after-tax income values for hypothetical workers at those ages, but the
solver and simulation never read them — retired agents use
`pension_after_tax` instead.

**Impact:** No numerical impact. Wastes ~42% of the array's memory. Could cause confusion if someone reads the array without understanding the retirement boundary. At higher n_z (e.g., 51), this dead data grows to ~200 KB — still trivial.

**Fix:** Either truncate to working ages only (`n_work = retire_age - start_age + 1`) or document why the full array exists. Low priority.

---

## ACCEPTABLE SIMPLIFICATIONS — Document in thesis

### 5. Pension taxed at 100% of benefit

Real SSA: only 50–85% of benefits are included in taxable income, depending on total income. Our model taxes 100%. At pension levels in this model (≤ 0.71 gross), the overtaxation is ~2–4 percentage points of the benefit. Catherine (2025) appears to do the same.

### 6. No standard deduction

The 2019 standard deduction ($12,200 ≈ 0.225 model units) is not subtracted before applying income tax brackets. Catherine's b₀ = −6.142 was calibrated with this in mind — the effective tax treatment is absorbed into the income level. Documenting this suffices.

### 7. AIME uses terminal z as career-average proxy

True AIME = career average of earnings over 35 highest years. We approximate AIME(z) ≈ exp(z_retire) × avg_det. With ρ = 0.991, the persistent component barely mean-reverts over a career, so terminal z ≈ career-average z for most agents. Tracking full earnings history would require an additional continuous state variable. This is the standard approach in lifecycle models.

### 8. E[exp(ε)] = 1.019 ≠ 1 (Jensen's inequality)

The zero-mean condition E[ε] = 0 means E[exp(ε)] > 1 by Jensen's inequality (~1.9% upward bias in mean gross income). This is standard; Catherine's b₀ calibration absorbs it. Worth one sentence in the thesis.

---

## VERIFIED CORRECT — No action needed

### 9. Income process parameters

All parameters (ρ, pz, μ_η1, σ_η1, σ_η2, pe, μ_ε1, σ_ε1, σ_ε2, b₀–b₃) match Catherine (2025) Table E.1 exactly.

### 10. Transitory shock quadrature

The Judd-mixture quadrature for ε matches all moments through degree
`2n_nodes − 1` to machine precision against the closed-form mixture.
At `n_eps_nodes = 3` (3 total nodes), exactness is order 5 — covers
mean, variance, skewness, and kurtosis (which is the moment of
interest for ε given its excess kurtosis +52). At `n_eps_nodes = 5`
(production), exactness extends through order 9. Polynomial integrals
match the analytic mixture moments to ≤ 5e-15 rel err. Cross-checked
against an independent Golub–Welsch eigendecomposition reference
([tests/test_judd_quadrature.py](tests/test_judd_quadrature.py))
agreeing to ≤ 1e-10. **Caveat for high γ:** polynomial exactness is
necessary but not sufficient for the FOC integrand `E[exp(-γ·ε)]`.
The +52 kurtosis means at γ ≥ 5 the integral is dominated by deep-tail
mass that no low-`n` quadrature can resolve (rel err ~17% at
`n_eps = 5, γ = 5`). See [LABOUR.md](LABOUR.md) §4.8 for production
node-selection guidance.

### 11. Pension formula (post-fix)

AIME scaling by avg_det ≈ 0.507, cap at 2.5, PIA bend points 0.21/1.25, replacement rates 90%/32%/15%, progressive tax on pension — all correctly implemented per Catherine (2025) eqs. (17)–(20). Hand-calculations match precomputed arrays to machine precision. 18 diagnostic tests pass.

### 12. Retirement boundary timing

Last labor paycheck at age 67 (t = 45), first pension at age 68 (t = 46). z transitions one final time at age 67 and freezes. Correctly implemented in both solver and simulation. Index arithmetic verified by the diagnostic retirement boundary trace.

### 13. Initial wealth

Catherine: "0.1 × the national wage index = $5,400 in 2019." Code uses `initial_wealth = 0.1`. Correct.

### 14. σ_z,0 = 0.652 (initial income dispersion)

Used as `initial_z_normal_std = 0.652` in simulation. This is Catherine's combined z + α initial dispersion. Correctly implemented.

### 15. λ = 0.016 (displacement probability)

From Catherine Table E.1. Catherine combines z and α processes, absorbing this parameter into σ_z,0. We do not need to implement λ separately.

### 16. PIA bend points

0.21 × $54,100 = $11,361 vs real 2019 SSA bend point 1 = $11,112/yr. 
1.25 × $54,100 = $67,625 vs real 2019 SSA bend point 2 = $67,008/yr. 
Both match to within 2%.

### 17. Payroll tax / AIME cap consistency

Both sides use 2.5. Payroll cap: 2.5 × $54,100 = $135,250 vs real 2019 SS taxable max = $132,900 (2% match). Verified by diagnostic test.

---

## SUMMARY TABLE

| # | Issue | Severity | Status | Action |
|---|-------|----------|--------|--------|
| 1 | Pi_z no upward transitions | **CRITICAL** | Diagnosed | Choose Rouwenhorst or increase n_z |
| 2 | Tauchen edge bins | Low | Handoff written | Implement standard ±∞ |
| 3 | Model unit docs ($54k not $61k) | Moderate | Confirmed | Update all references |
| 4 | working_income oversized | Low | Diagnosed | Document or truncate |
| 5 | Pension taxed at 100% | Simplification | — | Document in thesis |
| 6 | No standard deduction | Simplification | — | Document in thesis |
| 7 | AIME terminal-z proxy | Simplification | — | Document in thesis |
| 8 | Jensen's E[exp(ε)] ≈ 1.02 | Simplification | — | Document in thesis |
| 9–17 | Various verified items | None | Correct | — |

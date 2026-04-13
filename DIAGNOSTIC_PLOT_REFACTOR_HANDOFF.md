# Diagnostic & Plotting Module Refactor — Coding Agent Handoff

## Context

This is a lifecycle portfolio choice model (Catherine 2025) with three assets, VAR-driven returns, and mixture-normal labour income. The diagnostics need to verify that the calibration, discretization, and simulation are economically correct.

Read all project files before implementing. The current diagnostics live in `diagnostics.py` as one monolithic function. This task splits them into modular, independently callable functions with a consistent output format.

---

## 1. Architecture

Two files, two purposes:

| File | Purpose | Dependencies | Environment |
|------|---------|-------------|-------------|
| `diagnostics.py` | Text-based pass/fail/warn tests | numpy only | Terminal, notebook, CI, log files |
| `plots.py` | Visual inspection figures | matplotlib, numpy | Notebook only |

The diagnostics answer "is it correct?" — fast, always works, scriptable.
The plots answer "does it look right?" — for human review after diagnostics pass.

Diagnostics never import matplotlib. If a test is borderline, the diagnostic can print: "Run plot_labour_socialsec_sim(model, pc, sim) for visual inspection." But it never renders a figure.

---

## 2. Output Format

Every test prints with this structure:

```
  ✓  Higher persistent income → higher after-tax earnings
     Workers with higher z always earn more after taxes and payroll.
     Min gap between adjacent z-states: 1.85e-05
```

Line 1: Status symbol + title (economic meaning, understandable by someone who knows lifecycle models but not this codebase).
Line 2: One-line description of why this property matters.
Line 3: The measured value.

Three status levels:

| Symbol | Status | Meaning |
|:---:|---|---|
| ✓ | pass | Property holds. Results trustworthy for this check. |
| ⚠ | warn | Borderline or known simplification. Model usable but note this. |
| ✗ | fail | Property violated. Results should not be trusted until resolved. |

Use Unicode text symbols (✓ ✗ ⚠), NOT emoji (✅ ❌ ⚠️). Emoji render inconsistently across terminals.

Group tests into logical sub-sections with headers:

```
── Income Process Parameters ──────────────────────────
  ✓  Innovation mean = 0
     ...
  ✓  Eta quadrature captures variance
     ...

── Tax & Pension Formula ──────────────────────────────
  ✓  After-tax income monotone in persistent state
     ...
```

---

## 3. Return Type

Every diagnostic function returns a structured result list for programmatic use:

```python
def diagnose_labour_socialsec(model, pc) -> list[tuple[str, str, str]]:
    """
    Returns list of (status, title, detail) tuples.
    status: 'pass', 'warn', or 'fail'
    title: economic description of the test
    detail: measured values as string
    """
```

This enables aggregation across modules:

```python
results = []
results += diagnose_labour_socialsec(model, pc)
results += diagnose_var(model, pc)
results += diagnose_grids(model, pc)

n_fail = sum(1 for s, _, _ in results if s == 'fail')
print(f"Overall: {n_fail} failures out of {len(results)} tests")
```

---

## 4. Shared Helpers

Put at the top of `diagnostics.py`:

```python
_W = 76  # output width

def _header(title):
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)

def _sub(title):
    print(f"\n  ── {title} " + "─" * max(0, _W - len(title) - 6))

def _test(results_list, status, title, description, detail):
    """Record and print one test result.
    
    status: 'pass', 'warn', or 'fail'
    title: what is being checked (economic meaning)
    description: why it matters (one line, can be empty string)
    detail: measured value (can be empty string)
    """
    sym = {"pass": "✓", "warn": "⚠", "fail": "✗"}[status]
    print(f"  {sym}  {title}")
    if description:
        print(f"     {description}")
    if detail:
        print(f"     {detail}")
    results_list.append((status, title, detail))

def _summary(label, results):
    """Print pass/fail/warn summary for a diagnostic group."""
    n_pass = sum(1 for s, _, _ in results if s == "pass")
    n_warn = sum(1 for s, _, _ in results if s == "warn")
    n_fail = sum(1 for s, _, _ in results if s == "fail")
    total = len(results)
    print()
    print("─" * _W)
    if n_fail == 0 and n_warn == 0:
        print(f"  {label}: ALL {total} TESTS PASSED")
    elif n_fail == 0:
        print(f"  {label}: {total} tests — {n_pass} passed, {n_warn} warnings")
    else:
        print(f"  {label}: {n_fail} FAILED / {total} total")
        for s, title, _ in results:
            if s == "fail":
                print(f"    ✗  {title}")
    print("─" * _W)
    return results
```

---

## 5. Pre-Solve Functions

### 5.1 `diagnose_labour_socialsec(model, pc)`

Source: current Section 2 (lines 96–610) of `print_model_diagnostic_report`, plus eta quadrature and Pi_z blocks.

#### ── Income Process Parameters ──

| # | Test | Pass | Warn | Fail | Economic meaning |
|---|------|------|------|------|-----------------|
| 1 | Innovation mean ≈ 0 | \|E[η]\| < 1e-6 | | otherwise | Good and bad income shocks cancel out on average |
| 2 | Transitory shock mean ≈ 0 | \|E[ε]\| < 1e-10 | | otherwise | Transitory luck doesn't bias income levels |
| 3 | Eta quadrature captures variance | ratio within 5% | | otherwise | Solver correctly weights income uncertainty |
| 4 | Eta quadrature captures skewness | \|error\| < 0.1 | \|error\| < 0.5 | otherwise | Rare large income drops are correctly priced |

Also print (informational, no test): ρ, pz, component means/stds, unconditional σ_z, z-grid range, n_eta nodes, n_eps nodes.

#### ── Tax Schedule ──

| # | Test | Pass | Warn | Fail | Economic meaning |
|---|------|------|------|------|-----------------|
| 5 | Effective tax rate < 50% | max < 50% | | otherwise | Tax schedule doesn't confiscate majority of income |
| 6 | Tax schedule broadly progressive | top rate > bottom rate | | otherwise | Higher earners pay a larger share of income |
| 7 | Tax bracket boundary verified | tax at 10%/12% = 0.018 | | otherwise | Tax code matches 2019 TCJA thresholds |

Also print: effective tax rate at representative income levels.

#### ── Pension Formula (Social Security) ──

| # | Test | Pass | Warn | Fail | Economic meaning |
|---|------|------|------|------|-----------------|
| 8 | After-tax income positive everywhere | min > 0 | | otherwise | No worker or retiree receives negative income |
| 9 | Income rises with persistent state (z) | all adjacent gaps > 0 | | otherwise | Higher-skilled workers always earn more |
| 10 | Income rises with transitory shock (ε) | all adjacent gaps > 0 | | otherwise | Better luck within a year means higher pay |
| 11 | Pension rises with persistent state | gaps ≥ 0 | | otherwise | Higher lifetime earners receive at least as much SS |
| 12 | Pension constant across retirement ages | max variation = 0 | | otherwise | SS benefit doesn't change after claiming |
| 13 | Pension cap at max AIME | pension(z_max) = PIA(2.5) | | otherwise | SS earnings cap correctly implemented (~$135k) |
| 14 | AIME pipeline matches precomputed table | max error < 1e-10 | | otherwise | The AIME → PIA → tax → net pension chain is correct |
| 15 | Payroll tax cap = AIME cap | both = 2.5 | | otherwise | Tax side and benefit side use same earnings limit |

Also print: AIME pipeline trace table at representative z values: z → exp(z) → exp(z)×avg_det → AIME (capped) → PIA → tax → net pension.

#### ── Replacement Rates ──

| # | Test | Pass | Warn | Fail | Economic meaning |
|---|------|------|------|------|-----------------|
| 16 | Median replacement rate in [40%, 80%] | in range | | otherwise | Median worker replaces a realistic share of pre-retirement income |
| 17 | Rates decline from median z upward | monotone decreasing | flat at bottom OK | rising | SS provides proportionally more to lower earners (progressive) |
| 18 | Bottom earners replaced more than top | rate(z_min) > rate(z_max) | | otherwise | Redistributive structure of SS is working |

Also print: replacement rate table across z (vs career average, vs last year, vs peak year).

#### ── Retirement Boundary ──

| # | Test | Pass | Warn | Fail | Economic meaning |
|---|------|------|------|------|-----------------|
| 19 | Income array covers retirement age | index check | | otherwise | No out-of-bounds at work-retire transition |
| 20 | Pension array covers terminal age | index check | | otherwise | Pension available at all retirement ages |

Also print: income/pension sequence at the retirement boundary for median z.

#### ── z-Transition Quality (Pi_z, simulation-only) ──

| # | Test | Pass | Warn | Fail | Economic meaning |
|---|------|------|------|------|-----------------|
| 21 | Pi_z row sums = 1 | deviation < 1e-10 | | otherwise | Transition probabilities are valid |
| 22 | Upward transitions exist from mid state | P(up) > 0 | P(up) = 0 (WARN, solver unaffected) | | Simulation can represent income recovery |
| 23 | No absorbing states in Pi_z | all rows have escape | some absorbing (WARN) | | Simulation agents can leave boundary states |

Print: conditional mean, variance, skewness from Pi_z at mid row. Note prominently that Pi_z is simulation-only — the solver uses eta quadrature.

#### Summary

```
──────────────────────────────────────────────────────────────
  LABOUR INCOME & SOCIAL SECURITY: 21 passed, 2 warnings / 23 total
──────────────────────────────────────────────────────────────
```

---

### 5.2 `diagnose_var(model, pc, periods_per_year=1)`

Source: current Sections 3 + 4 + 5 (lines 164–292) plus VAR-related checks from Section 7.

Sub-sections:
- VAR Parameters (Phi_11, Phi_21, means, M matrix — informational print)
- State Grid Quality
- Return Distribution Quality

Tests:

| # | Test | Economic meaning |
|---|------|-----------------|
| 1 | Phi_11 stationary (max eigenvalue < 1) | Financial states are mean-reverting |
| 2 | Pi_state row sums = 1 | State transitions are valid probabilities |
| 3 | Return quadrature mean ≈ 0 | Return residuals are centered |
| 4 | Return quadrature covariance matches Sigma_r_cond | Return variance correctly captured |
| 5 | Sigma_ss positive definite | State innovation covariance is valid |
| 6 | Sigma_r_cond positive definite | Return residual covariance is valid |
| 7–9 | State grid coverage ≥ 2.5σ (per dimension) | Grid spans enough of each state's distribution |

---

### 5.3 `diagnose_grids(model, pc)`

Source: current Sections 6 + 7 (lines 294–440), non-VAR/non-income parts.

Contents (mostly informational, few tests):
- Dimension table (n_w, n_z, N_state, n_eps, n_eta, n_ret_quad, n_age)
- Memory estimate for policy and value function arrays
- Grid ranges (wealth min/max, savings min/max)
- Survival probabilities in (0, 1]
- Wealth grid strictly positive

Tests:

| # | Test | Economic meaning |
|---|------|-----------------|
| 1 | Survival probs in (0, 1] | Mortality rates are biologically plausible |
| 2 | Wealth grid strictly positive | Cash-on-hand bounded away from zero (log utility safe) |

---

## 6. Post-Simulation Functions

### 6.1 `diagnose_labour_socialsec_sim(model, pc, sim)`

Source: current `print_simulation_income_report` (lines ~860–1175).

Expected `sim` keys: `'income'`, `'alive'`, `'z_idx'`, `'z'`, `'ages'`, `'death_age'`.

#### ── Simulated Income Moments ──

| # | Test | Pass | Economic meaning |
|---|------|------|-----------------|
| 1 | Sim/theory income ratio within 20% at all ages | worst deviation < 20% | Simulated income draws are consistent with precomputed tables |

Compute E[Y_theory] using the **simulated z-distribution** at each age (count how many agents are at each z-grid point, weight the precomputed income table by those fractions). This decouples the income-table check from z-transition quality.

#### ── Realized Replacement Rates ──

| # | Test | Pass | Economic meaning |
|---|------|------|-----------------|
| 2 | Median replacement rate in [20%, 200%] | in range | Retirement income transition is plausible |

Print: percentile table (p10, p25, p50, p75, p90) of pension / last-working-year income.

#### ── Retirement Boundary ──

| # | Test | Pass | Economic meaning |
|---|------|------|-----------------|
| 3 | z frozen after retirement (traced agents) | z constant post-retire | SS benefit locked at retirement earnings level |
| 4 | z frozen globally at retire+1 | all agents checked | No agent has drifting z after retirement |

Trace 3 individual agents (low/mid/high z at retirement) through 2 years before and after.

#### ── Survivor Selection ──

| # | Test | Pass | Economic meaning |
|---|------|------|-----------------|
| 5 | Mean z rises with age among survivors | late-life z > entry z | Richer people live longer (Chetty et al. mortality is working) |

Print: mean z_idx table at ages 22, 30, 50, 67, 75, 85, 95.

---

### 6.2 `diagnose_var_sim(model, pc, sim)` — Stub

```python
def diagnose_var_sim(model, pc, sim):
    """Post-simulation return and state diagnostics. Not yet implemented."""
    _header("POST-SIMULATION VAR DIAGNOSTICS")
    print("  (Not yet implemented)")
    return []
```

---

## 7. Convenience Wrappers

```python
def diagnose_all_pre(model, pc, periods_per_year=1):
    """All pre-solve diagnostics."""
    r = []
    r += diagnose_labour_socialsec(model, pc)
    r += diagnose_var(model, pc, periods_per_year)
    r += diagnose_grids(model, pc)
    n_fail = sum(1 for s, _, _ in r if s == 'fail')
    if n_fail:
        print(f"\n  *** {n_fail} FAILURE(S) across all pre-solve diagnostics ***")
    return r

def diagnose_all_post(model, pc, sim):
    """All post-simulation diagnostics."""
    r = []
    r += diagnose_labour_socialsec_sim(model, pc, sim)
    r += diagnose_var_sim(model, pc, sim)
    n_fail = sum(1 for s, _, _ in r if s == 'fail')
    if n_fail:
        print(f"\n  *** {n_fail} FAILURE(S) across all post-simulation diagnostics ***")
    return r
```

---

## 8. Companion File: `plots.py`

Separate file. Never imported by `diagnostics.py`. Requires matplotlib.

### `plot_labour_socialsec(model, pc)` — 3 figures

**Figure 1 — Income lifecycle profile.** Three lines (low/median/high z) showing E_eps[after-tax income] by age 22–99. Clear drop at retirement to pension level. Shows progressive replacement (low-z has smaller income drop). Data: `pc.working_income` for working ages (average over eps using `pc.eps_weights`), `pc.pension_after_tax` for retirement ages.

**Figure 2 — Replacement rate curve.** x-axis: z-grid values. y-axis: replacement rate (pension / career-average after-tax income). Should decline from ~100% at low z to <10% at high z. Mark where AIME cap binds.

**Figure 3 — AIME pipeline.** x-axis: z. Four lines: exp(z)×avg_det (raw earnings proxy), AIME (capped at 2.5), gross PIA, net pension. Shows where each bend point and cap activates.

### `plot_labour_socialsec_sim(model, pc, sim)` — 4 figures

**Figure 4 — z distribution at key ages.** Histogram of `sim["z"]` (continuous values) at ages 22, 45, 67, 85. Should start narrow (σ=0.652), widen over career, stay centered near 0. This is the key visual confirming the quadrature fix worked — previously would show mass piling up at z_min.

**Figure 5 — Income fan chart.** x-axis: age 22–99. Shaded bands for p10/p25/p50/p75/p90 of `sim["income"]` among alive agents. Shows hump-shaped working income and flat pension.

**Figure 6 — Realized replacement rate distribution.** Histogram of pension / last-working-year income across surviving retirees. Should center around 50–70% with right tail.

**Figure 7 — Survivor selection.** Mean of `sim["z"]` among alive agents by age. Should be flat or slightly rising. Previously fell due to Pi_z drift.

### Future: `plot_var(model, pc)` and `plot_var_sim(model, pc, sim)` — not implemented now, same pattern.

---

## 9. Old Functions — Deprecation

Keep but redirect:

```python
def print_model_diagnostic_report(model, pc, periods_per_year=1):
    """Deprecated. Use diagnose_all_pre() instead."""
    import warnings
    warnings.warn("Use diagnose_all_pre(model, pc) instead", DeprecationWarning)
    return diagnose_all_pre(model, pc, periods_per_year)

def print_simulation_income_report(model, pc, sim):
    """Deprecated. Use diagnose_labour_socialsec_sim() instead."""
    import warnings
    warnings.warn("Use diagnose_labour_socialsec_sim(model, pc, sim)", DeprecationWarning)
    return diagnose_labour_socialsec_sim(model, pc, sim)
```

---

## 10. Implementation Order

1. Shared helpers — 15 min
2. `diagnose_labour_socialsec` — 45 min (biggest: extract, reformat, add descriptions)
3. `diagnose_labour_socialsec_sim` — 15 min (rename + reformat)
4. `diagnose_var` — 30 min (merge sections 3+4+5)
5. `diagnose_grids` — 20 min (merge sections 6+7, add survival)
6. Convenience wrappers + deprecation wrappers — 10 min
7. `plots.py` with 7 figures — 45 min
8. Verify output matches old functions — 10 min

Total: ~3 hours. No new test logic — pure refactoring, reformatting, and a new plots file.

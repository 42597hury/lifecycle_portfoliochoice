# HANDOFF: Fix Off-by-One in `life_expectancy_at_40()` — mortality.py

## Summary

`life_expectancy_at_40()` in `mortality.py` has a two-line ordering bug that inflates the returned life expectancy by exactly 1.0 years. This causes all calibrated χ values to be ~10% too high, making agents face uniformly higher mortality than the Chetty et al. (2016) targets intend.

---

## 1. The Bug

In `life_expectancy_at_40()` (lines ~203–214), the loop accumulates the survival probability **before** applying that period's mortality:

```python
# CURRENT (BUGGY)
survival = 1.0
le_sum = 0.0
for age in range(40, max_age + 1):
    m_base = m_table.get(age, 1.0)
    m_adj = min(chi * m_base, 1.0)
    le_sum += survival              # <-- adds S(40) = 1 on first iteration
    survival *= (1.0 - m_adj)
    if survival < 1e-15:
        break
return 40.0 + le_sum
```

On the first iteration (`age = 40`), `survival = 1.0` is added to `le_sum` before any mortality is applied. This means `le_sum` includes the term `₀p₄₀ = 1` (the trivial probability that a 40-year-old is alive at age 40). The standard actuarial curtate expectation of life at age 40 is:

```
e₄₀ = Σ_{k=1}^{ω−40} ₖp₄₀  =  S(41) + S(42) + ...
```

The code computes `le_sum = 1 + e₄₀` and returns `40 + 1 + e₄₀ = 41 + e₄₀`, but the correct expected age at death is `40 + e₄₀`.

## 2. The Fix

Swap the two lines so that survival is updated **before** accumulation:

```python
# FIXED
survival = 1.0
le_sum = 0.0
for age in range(40, max_age + 1):
    m_base = m_table.get(age, 1.0)
    m_adj = min(chi * m_base, 1.0)
    survival *= (1.0 - m_adj)        # apply mortality first
    le_sum += survival               # then accumulate
    if survival < 1e-15:
        break
return 40.0 + le_sum
```

Now the first term added is `S(41) = 1 − m(40)`, so `le_sum = e₄₀` and the function returns `40 + e₄₀`.

**This is the only code change needed.** No other function in `mortality.py` or elsewhere needs modification — the Brent solver, the χ calibration, and the 2D survival array all flow downstream from this function and will automatically produce corrected values.

## 3. Verification Steps

Before applying the fix, **independently confirm the bug exists** using these tests:

### Test A — Toy example (closed-form)
Construct a 3-age mortality table: `{40: 0.5, 41: 0.5, 42: 1.0}`. By enumeration, the exact expected age at death is:
```
E[death age] = 40×0.5 + 41×(0.5×0.5) + 42×(0.5×0.5×1.0) = 40.75
```
Call `life_expectancy_at_40(chi=1.0, m_table={40:0.5, 41:0.5, 42:1.0}, max_age=42)`. The current code should return **41.75** (off by 1.0). After the fix it should return **40.75**.

### Test B — Full SSA table, direct enumeration
With `chi = 1.0` and the full `SSA_DEATH_PROB_2017` table, compute `E[death age]` by direct enumeration:
```python
S = {}
S[40] = 1.0
for a in range(40, 120):
    S[a+1] = S[a] * (1 - SSA_DEATH_PROB_2017[a])
E_death = sum(a * (S[a] - S.get(a+1, 0)) for a in range(40, 120))
```
This should give ≈ 79.125. The current `life_expectancy_at_40(1.0)` returns ≈ 80.125 (off by 1.0). After the fix it should return ≈ 79.125.

### Test C — Calibration round-trip
After applying the fix, run the full calibration self-test at the bottom of `mortality.py` (`if __name__ == "__main__"` block). Verify that the `err(yr)` column in the printout shows errors at machine precision (< 1e-8 years) for all z-grid points. This confirms the Brent solver is still hitting the Chetty targets exactly.

### Test D — χ direction check
The fixed χ values should be **lower** than the old ones (roughly 9–10% lower at every grid point). Lower χ means lower mortality, which is the correct direction: the old code was over-estimating LE by 1 year, so the solver was pushing χ too high to compensate.

## 4. What NOT to Change

- **Do not change `build_survival_probs_2d()`** — it computes per-period survival `ψ(age, z) = 1 − min(χ·m(age), 1)` which is correct and unrelated to the LE summation convention.
- **Do not change `_solve_chi()`** — the Brent solver and its bracket [0.01, 20.0] are correct.
- **Do not change `calibrate_chi_vector()` or `calibrate_earnings_dependent_mortality()`** — they call `life_expectancy_at_40()` and will automatically pick up the fix.
- **Do not change the solver (`solver.py`) or simulation (`simulation.py`)**  — they consume `survival_probs_2d` which is constructed from χ, and the fix propagates through the existing pipeline.
- **Do not touch the Chetty data or SSA table.**

## 5. Files Involved

| File | Action |
|------|--------|
| `mortality.py` | Swap two lines in `life_expectancy_at_40()` (Section 4, lines ~209–210) |

One file, one function, two lines swapped. Nothing else.

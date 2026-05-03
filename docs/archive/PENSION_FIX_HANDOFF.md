# PENSION CALIBRATION FIX — CODING AGENT HANDOFF

## 1. THE BUG

### 1.1 What the code does now

`compute_pension_after_tax(z_grid)` in **model.py** (line 292) computes Social Security pension benefits. It sets `career_rank = exp(z)` and feeds that directly into the SSA PIA (Primary Insurance Amount) three-bracket progressive formula with bend points at 0.21 and 1.25, replacement rates 90%/32%/15%. There is no earnings cap.

`_precompute_pension(self)` in **precompute.py** (line 363) calls `compute_pension_after_tax(self.z_grid)` and tiles the result across all ages to produce the `pension_after_tax` array of shape `(n_age, n_z)`.

### 1.2 What the code should do (per Catherine 2025, Section 3.4, equations 17–20)

In Catherine's paper, the PIA formula (eq. 19) takes **AIYE** (Average Indexed Yearly Earnings) as input — not `exp(z)`. AIYE is the career average of total income (including the deterministic age-earnings profile), capped at the SSA taxable maximum of 2.5 × L̄ (the SS wage index, which equals 1.0 in model units). The formula is:

```
AIYE_it = L̄_t × Σ_{s=t0}^{t} min{ L̃_is, 2.5 }     (Catherine eq. 20)
```

where `L̃_is = L_is / L̄_s` is income relative to the wage index.

In our model, total gross income is `exp(f(age) + z + eps)`. Since `z` is the persistent component and `eps` averages to zero, the career-average income for a worker at persistent state `z` is approximately:

```
AIME(z) ≈ exp(z) × avg_det
```

where `avg_det = mean(exp(f(age)))` over working ages 22–66.

### 1.3 Why this matters quantitatively

The deterministic profile `f(age) = b0 + b1×age + b2×age²/10 + b3×age³/100` peaks at age 46 with `exp(f(46)) = 0.649`. The career average is:

```
avg_det = mean(exp(f(age))) for age ∈ [22, 66] = 0.5069
```

So the median worker (z=0) has AIME ≈ 0.507, not 1.0. The current code overestimates AIME by a factor of ~2, which pushes it deep into the second PIA bracket and produces:

| Metric | Current (broken) | Correct |
|--------|-----------------|---------|
| AIME at z=0 | 1.000 | 0.507 |
| Pension (after-tax) at z=0 | 0.392 | 0.254 |
| Replacement rate at z=0 | **117%** | **63%** |
| Pension at z_max (=5.609) | **26.636** | **0.628** |
| Pension range | [0.003, 26.636] | [0.002, 0.628] |

The z_max pension of 26.6 model units (≈ $1.6M/year) is absurd — it arises because there is no earnings cap. With the cap at 2.5, the maximum pension is capped at the PIA(2.5) ≈ 0.71 pre-tax.

### 1.4 Economic consequences

The over-generous pension acts as a massive implicit bond holding for retirees, crowding out explicit bond demand and distorting the lifecycle portfolio allocation. Fixing this will:
- Lower retirement income → more precautionary saving during working life
- Shrink the implicit "Social Security wealth" → higher explicit bond demand (agents need more duration exposure to compensate)
- Produce more realistic wealth accumulation profiles
- Strengthen the model's story about interest-rate hedging demand

---

## 2. THE FIX (Approach A — Catherine-faithful)

### 2.1 Overview

Two functions change. No other files need modification. The `pension_after_tax` array retains its shape `(n_age, n_z)` and all downstream consumers (solver, simulation, diagnostics) are unaffected.

### 2.2 Changes to `model.py`

#### 2.2.1 Change the signature of `compute_pension_after_tax`

**Old signature:** `compute_pension_after_tax(z_grid)`  
**New signature:** `compute_pension_after_tax(z_grid, avg_det)`

#### 2.2.2 Replace the body (lines 292–321)

Replace the entire function with:

```python
def compute_pension_after_tax(z_grid, avg_det):
    """
    Social Security benefits following Catherine (2025, eq. 19).

    Parameters
    ----------
    z_grid : array, shape (n_z,)
        Persistent income grid (log, mean-zero).
    avg_det : float
        Mean of exp(f(age)) over working ages.  Converts the persistent
        component exp(z) to an AIME proxy:  AIME(z) = exp(z) * avg_det.

    Returns
    -------
    pension_net : array, shape (n_z,)
        After-tax annual pension benefit in model units.
    """
    z = np.asarray(z_grid, dtype=float)

    # AIME: career-average earnings, capped at taxable maximum (2.5 × L̄)
    # Catherine eq. (20): AIYE = L̄_t × Σ min{L̃_is, 2.5}
    aime = np.minimum(np.exp(z) * avg_det, 2.5)

    # PIA formula — Catherine eq. (19)
    # Bend points in wage-index units (L̄ = 1 in model)
    b1, b2 = 0.21, 1.25
    r1, r2, r3 = 0.90, 0.32, 0.15

    pension = np.zeros_like(aime)

    lo = aime <= b1
    pension[lo] = aime[lo] * r1

    mid = (aime > b1) & (aime <= b2)
    pension[mid] = r1 * b1 + r2 * (aime[mid] - b1)

    hi = aime > b2
    pension[hi] = r1 * b1 + r2 * (b2 - b1) + r3 * (aime[hi] - b2)

    # Income tax on pension benefits (same progressive schedule as labor income)
    tax = np.zeros_like(pension)
    m = pension <= 0.18
    tax[m] = pension[m] * 0.10
    m = (pension > 0.18) & (pension <= 0.72)
    tax[m] = 0.018 + (pension[m] - 0.18) * 0.12
    m = (pension > 0.72) & (pension <= 1.54)
    tax[m] = 0.0828 + (pension[m] - 0.72) * 0.22
    m = (pension > 1.54) & (pension <= 2.94)
    tax[m] = 0.2632 + (pension[m] - 1.54) * 0.24
    m = (pension > 2.94) & (pension <= 3.73)
    tax[m] = 0.5992 + (pension[m] - 2.94) * 0.32
    m = (pension > 3.73) & (pension <= 9.32)
    tax[m] = 0.8520 + (pension[m] - 3.73) * 0.35
    m = pension > 9.32
    tax[m] = 2.8085 + (pension[m] - 9.32) * 0.37

    return pension - tax
```

**What changed vs. the old function:**
1. Added `avg_det` parameter.
2. Line `career_rank = np.exp(z)` → `aime = np.minimum(np.exp(z) * avg_det, 2.5)`. This is the two-part fix: (a) multiply by `avg_det` to convert persistent component to career-average income, (b) cap at 2.5 (SSA taxable maximum).
3. Named constants `b1, b2, r1, r2, r3` for clarity.
4. The tax bracket code is unchanged.

### 2.3 Changes to `precompute.py`

#### 2.3.1 Replace `_precompute_pension` (lines 363–371)

Replace the entire method with:

```python
def _precompute_pension(self):
    """
    After-tax pension table: shape (n_age, n_z).

    Computes avg_det = mean(exp(f(age))) over working ages, then
    passes it to compute_pension_after_tax so that AIME is correctly
    scaled by the deterministic lifecycle profile.

    Catherine (2025) eq. (20):
        AIYE_it = L̄_t × Σ min{L̃_is, 2.5}
    Our approximation:
        AIME(z) ≈ min(exp(z) × avg_det, 2.5)
    """
    model = self.model

    # Average deterministic income component over working ages
    working_ages = np.arange(model.start_age, model.retire_age)
    log_det = (model.b0
               + model.b1 * working_ages
               + model.b2 * working_ages**2 / 10.0
               + model.b3 * working_ages**3 / 100.0)
    avg_det = np.mean(np.exp(log_det))

    base_pension = compute_pension_after_tax(self.z_grid, avg_det)

    n_age = len(self.ages)
    n_z = len(self.z_grid)
    out = np.empty((n_age, n_z), dtype=float)
    for t_idx in range(n_age):
        out[t_idx, :] = base_pension
    return out
```

**What changed vs. the old method:**
1. Added computation of `avg_det` from the model's age-earnings coefficients (`b0, b1, b2, b3`).
2. Pass `avg_det` as the second argument to `compute_pension_after_tax`.
3. Everything else (shape, tiling, return) is identical.

---

## 3. FILES AFFECTED AND NOT AFFECTED

### 3.1 Files that MUST change

| File | Location | Change |
|------|----------|--------|
| `model.py` | Lines 292–321 | Replace `compute_pension_after_tax` (new signature + body) |
| `precompute.py` | Lines 363–371 | Replace `_precompute_pension` (add `avg_det` computation) |

### 3.2 Files that MUST NOT change

| File | Reason |
|------|--------|
| `solver.py` | Consumes `pension_after_tax[t, iz]` — shape and semantics unchanged |
| `simulation.py` | Same — reads pension array by index |
| `mortality.py` | Unrelated — uses `z` for mortality scaling, not pension |
| `discretization.py` | Unrelated |
| `var.py` | Unrelated |
| `policy_io.py` | Unrelated |

### 3.3 File that SHOULD be updated (optional but recommended)

| File | Location | Change |
|------|----------|--------|
| `diagnostics.py` | Lines 153–159 | Update replacement rate reporting (see Section 4) |
| `DESIGN.md` | Section 1.4 | Update pension documentation to reflect AIME scaling |

---

## 4. DIAGNOSTIC REPORT UPDATE (OPTIONAL)

The diagnostic report in `diagnostics.py` (lines 148–159) currently prints:

```
Last working year (age 66): Y_net = 0.3348
Pension at z = 0:                   0.3924
Replacement rate:                   117.20%
Pension range (min z, max z):       [0.0030, 26.6364]
```

After the fix, this will automatically show corrected values (~0.254 pension, ~76% replacement vs last-year income, range [0.002, 0.628]) because it reads from `pc.pension_after_tax` which is rebuilt by the changed `_precompute_pension`.

**Recommended enhancement:** Add `avg_det` and AIME reporting. After line 159, add:

```python
# Show AIME scaling
working_ages_diag = np.arange(model.start_age, model.retire_age)
log_det_diag = (model.b0 + model.b1 * working_ages_diag
                + model.b2 * working_ages_diag**2 / 10.0
                + model.b3 * working_ages_diag**3 / 100.0)
avg_det_diag = np.mean(np.exp(log_det_diag))
aime_z0 = np.exp(pc.z_grid[iz0]) * avg_det_diag
print(f"  avg_det (mean exp(f(age))):      {avg_det_diag:.4f}")
print(f"  AIME at z=0:                     {aime_z0:.4f}")
print(f"  SSA taxable cap:                 2.5000")
```

This makes the pension calibration transparent in the diagnostic output.

---

## 5. VERIFICATION CHECKLIST

After applying the changes, run the model build + diagnostics and verify:

### 5.1 Expected diagnostic output (approximate)

```
Last working year (age 66): Y_net = 0.3348      ← UNCHANGED
Pension at z = 0:                   ~0.254       ← was 0.3924
Replacement rate:                   ~76%         ← was 117% (vs last-year income)
Pension range (min z, max z):       [~0.002, ~0.628]  ← was [0.003, 26.636]
avg_det (mean exp(f(age))):         ~0.507
AIME at z=0:                        ~0.507
```

### 5.2 Key sanity checks

1. **Pension at z=0 should be ~0.25 after-tax** (was 0.39). The gross PIA at AIME=0.507 is: `0.9×0.21 + 0.32×(0.507−0.21) = 0.189 + 0.095 = 0.284`. After tax (~10% bracket): ~0.254.

2. **Pension at z_max should be ~0.63 after-tax** (was 26.6). The cap at 2.5 binds for all z ≥ ~1.6. PIA(2.5) = `0.9×0.21 + 0.32×(1.25−0.21) + 0.15×(2.5−1.25) = 0.189 + 0.333 + 0.188 = 0.709`. After tax: ~0.628.

3. **Replacement rate at z=0 vs career-average income should be ~63%**. Career-average after-tax income at z=0 is ~0.402. Pension/career_avg = 0.254/0.402 = 63%.

4. **Replacement rate at z≈1.12 (75th percentile) should be ~43%**. This matches SSA actuarial targets.

5. **All `pension_after_tax` values should be positive** (the diagnostic already checks this at line 404).

6. **No changes to solver behavior should be needed** — the `pension_after_tax` array has the same shape and dtype. The solver reads `pension_1d = pension_after_tax[t, :]` per age, which still works.

### 5.3 Full model re-solve

After the fix, a full backward-induction solve is required because the value function and policy rules will change. Expected qualitative effects:
- **More saving during working life** (lower expected retirement income → precautionary motive)
- **Higher bond demand** (smaller implicit SS wealth → agents need more explicit duration)
- **Lower consumption in early retirement** (lower pension income)
- The solver should converge without issues — the pension level changes are smooth and moderate.

---

## 6. ECONOMIC JUSTIFICATION

### 6.1 Paper reference

Catherine (2025), "Interest Rate Risk and Household Portfolios," Section 3.4 (pp. 22–23), equations (17)–(20). The paper is in the project as `Interest_Rate_Risk_and_Household_Portfolios_99c0f47725df40dc831b1990f35df31a_2.pdf`. The relevant content is on pages 22–23 of the PDF.

### 6.2 The formula chain

```
Payroll tax:   T_it = 0.106 × min{L_it, 2.5 × L̄_t}              (eq. 17)
Indexed earn:  L^indexed_it = min{L_it, 2.5×L̄_t} × L̄_t60/L̄_t   (eq. 18)
PIA formula:   B_it = piecewise(AIYE; b1=0.21, b2=1.25)           (eq. 19)
AIYE:          AIYE_it = L̄_t × Σ min{L̃_is, 2.5}                 (eq. 20)
```

In the model with stationary wage index (L̄ = 1), this simplifies to:
```
AIME(z) ≈ min(exp(z) × avg_det, 2.5)
```
where `avg_det = mean(exp(f(age)))` over ages 22–66 = 0.5069.

### 6.3 Why Approach A (not B or C)

- **Approach A** (use AIME as input): Minimal code change, transparent mapping to Catherine's equations, bend points retain their SSA interpretation. Easy to explain in thesis.
- **Approach B** (calibrate replacement rates directly): More work, less connected to paper.
- **Approach C** (rescale bend points): Mathematically equivalent to A but bend points lose SSA meaning.

### 6.4 Consistency with payroll tax

The `disposable_income_working` function in `model.py` (line 266) already applies the correct payroll tax with the 2.5 cap: `payroll_tax = 0.106 * np.minimum(y, 2.5)`. The pension fix now makes the benefit side consistent with the tax side — both use the same 2.5 cap.

---

## 7. EXACT DIFF SUMMARY

### model.py

```diff
-def compute_pension_after_tax(z_grid):
-    """Social Security benefits with progressive formula and taxes."""
-    z = np.asarray(z_grid, dtype=float)
-    career_rank = np.exp(z)
-    pension = np.zeros_like(career_rank)
-
-    m = career_rank <= 0.21
-    pension[m] = career_rank[m] * 0.90
-    m = (career_rank > 0.21) & (career_rank <= 1.25)
-    pension[m] = 0.189 + (career_rank[m] - 0.21) * 0.32
-    m = career_rank > 1.25
-    pension[m] = 0.5218 + (career_rank[m] - 1.25) * 0.15
-
-    tax = np.zeros_like(pension)
+def compute_pension_after_tax(z_grid, avg_det):
+    """
+    Social Security benefits following Catherine (2025, eq. 19).
+
+    Parameters
+    ----------
+    z_grid : array, shape (n_z,)
+        Persistent income grid (log, mean-zero).
+    avg_det : float
+        Mean of exp(f(age)) over working ages.  Converts the persistent
+        component exp(z) to an AIME proxy:  AIME(z) = exp(z) * avg_det.
+
+    Returns
+    -------
+    pension_net : array, shape (n_z,)
+        After-tax annual pension benefit in model units.
+    """
+    z = np.asarray(z_grid, dtype=float)
+
+    # AIME: career-average earnings, capped at taxable maximum (2.5 × L̄)
+    aime = np.minimum(np.exp(z) * avg_det, 2.5)
+
+    # PIA formula — Catherine eq. (19)
+    b1, b2 = 0.21, 1.25
+    r1, r2, r3 = 0.90, 0.32, 0.15
+
+    pension = np.zeros_like(aime)
+
+    lo = aime <= b1
+    pension[lo] = aime[lo] * r1
+
+    mid = (aime > b1) & (aime <= b2)
+    pension[mid] = r1 * b1 + r2 * (aime[mid] - b1)
+
+    hi = aime > b2
+    pension[hi] = r1 * b1 + r2 * (b2 - b1) + r3 * (aime[hi] - b2)
+
+    # Income tax on pension benefits
+    tax = np.zeros_like(pension)
     m = pension <= 0.18
     tax[m] = pension[m] * 0.10
     # ... (remaining tax bracket code is IDENTICAL — no changes needed)
```

### precompute.py

```diff
     def _precompute_pension(self):
-        """After-tax pension table: [age, z_state]."""
-        base_pension = compute_pension_after_tax(self.z_grid)
+        """
+        After-tax pension table: shape (n_age, n_z).
+        Computes avg_det = mean(exp(f(age))) over working ages,
+        then passes to compute_pension_after_tax for AIME scaling.
+        """
+        model = self.model
+        working_ages = np.arange(model.start_age, model.retire_age)
+        log_det = (model.b0
+                   + model.b1 * working_ages
+                   + model.b2 * working_ages**2 / 10.0
+                   + model.b3 * working_ages**3 / 100.0)
+        avg_det = np.mean(np.exp(log_det))
+
+        base_pension = compute_pension_after_tax(self.z_grid, avg_det)
         n_age = len(self.ages)
         n_z = len(self.z_grid)
         out = np.empty((n_age, n_z), dtype=float)
         for t_idx in range(n_age):
             out[t_idx, :] = base_pension
         return out
```

---

## 8. IMPORT DEPENDENCIES

No new imports are needed. Both `model.py` and `precompute.py` already import `numpy`. The `compute_pension_after_tax` import in `precompute.py` (from model import ...) continues to work since only the signature changed — Python doesn't enforce signature checking at import time.

**However**, verify that `compute_pension_after_tax` is not called from any other location with the old one-argument signature. Search all `.py` files for `compute_pension_after_tax`:

```bash
grep -rn "compute_pension_after_tax" *.py
```

Expected hits:
- `model.py`: function definition (change it)
- `precompute.py`: the only call site (change it)
- `diagnostics.py`: not called directly (reads `pc.pension_after_tax` array)

If any other call site exists, it must also be updated to pass `avg_det`.

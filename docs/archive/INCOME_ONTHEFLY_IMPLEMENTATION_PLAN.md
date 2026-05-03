# IMPLEMENTATION PLAN: On-the-Fly Solver Income

**Consolidates:** `INCOME_ONTHEFLY_DESIGN.md` + `INCOME_ONTHEFLY_REVIEW.md`

**Purpose:** Replace the solver's z-interpolated income lookup (14–17%
systematic error from `cosh(dz/2) − 1` chord overshoot) with exact
on-the-fly computation using the same scalar tax function the simulation
already uses.

**Cost:** +15% ± 10% solver wall-time
**Payoff:** Eliminates income interpolation error; expected to close
the ~17% solver–simulation gap flagged in `ISSUES.md`

---

## Step 1 — `model.py`: Add `scalar_disposable_income`

Move from `simulation.py:301` to `model.py`, adjacent to the vectorized
`disposable_income_working` (line 268). Drop the leading underscore — this
is now a public API consumed by both solver and simulation.

**Insert after `disposable_income_working` (after line 291):**

```python
@njit(fastmath=True)
def scalar_disposable_income(y_gross):
    """After-tax labor income for a single scalar gross income value.

    Identical tax schedule to disposable_income_working() but operates
    on a single float for use inside Numba-compiled solver loops.

    Parameters
    ----------
    y_gross : float
        Gross labor income in model units.

    Returns
    -------
    float
        Disposable (after-tax, after-payroll) income.
    """
    payroll_tax = 0.106 * min(y_gross, 2.5)
    taxable = max(0.0, y_gross - payroll_tax)

    if taxable <= 0.18:
        tax = taxable * 0.10
    elif taxable <= 0.72:
        tax = 0.018 + (taxable - 0.18) * 0.12
    elif taxable <= 1.54:
        tax = 0.0828 + (taxable - 0.72) * 0.22
    elif taxable <= 2.94:
        tax = 0.2632 + (taxable - 1.54) * 0.24
    elif taxable <= 3.73:
        tax = 0.5992 + (taxable - 2.94) * 0.32
    elif taxable <= 9.32:
        tax = 0.8520 + (taxable - 3.73) * 0.35
    else:
        tax = 2.8085 + (taxable - 9.32) * 0.37

    return taxable - tax
```

**Add required import** at top of `model.py` (if not already present):

```python
from numba import njit
```

Check: `model.py` currently has no numba import. Add it alongside the
existing `import numpy as np`.

---

## Step 2 — `simulation.py`: Replace local function with import

**At the import block** (top of file), add:

```python
from model import scalar_disposable_income
```

**Replace the local function** at line 301–326 with a backward-compat alias:

```python
# Backward-compatible alias — canonical version now in model.py
_scalar_disposable_income = scalar_disposable_income
```

This preserves the existing `_scalar_disposable_income` name used
internally in `simulate_lifecycle` (line 558) without changing any
simulation logic. The `@njit` decorator is on the canonical version
in `model.py`; Numba handles the alias correctly.

---

## Step 3 — `solver.py`: Import

**Change line 31:**

```python
# CURRENT
from model import SolverConfig

# NEW
from model import SolverConfig, scalar_disposable_income
```

---

## Step 4 — `solver.py`: Rewrite `compute_foc_jac_working`

This is the only function with economic logic changes. Everything else
in the solver is mechanical parameter renaming.

### 4.1 Signature (line 770–778)

```python
# CURRENT
def compute_foc_jac_working(alpha_s, alpha_b, s_val, z_idx, i_s,
                             wealth_grid, c_next_full, income_next_table,
                             annuity_factor_is,
                             z_grid, rho, eta_nodes, eta_weights, dz,
                             Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                             eps_nodes, eps_weights,
                             gamma, psi, beta, b_bar,
                             min_wealth_inv=1e-10, min_consumption=1e-10,
                             prob_skip=1e-12):

# NEW — income_next_table → log_det_next (same position)
def compute_foc_jac_working(alpha_s, alpha_b, s_val, z_idx, i_s,
                             wealth_grid, c_next_full, log_det_next,
                             annuity_factor_is,
                             z_grid, rho, eta_nodes, eta_weights, dz,
                             Pi_state, Rx_stock_next, Rx_bond_next, ret_weights, R_bill,
                             eps_nodes, eps_weights,
                             gamma, psi, beta, b_bar,
                             min_wealth_inv=1e-10, min_consumption=1e-10,
                             prob_skip=1e-12):
```

### 4.2 Precompute block (insert after line 790, before the `for j_s` loop)

This is the **critical optimisation** identified in the review. By
factoring `exp(a + b + c) = exp(a)·exp(b)·exp(c)`, all `exp()` calls
are lifted out of the hot (η, ε) inner loop.

```python
    # ── Precompute exp(ε) and exp(η): avoids exp() in the hot loop ──
    # exp(f(t+1) + ρ·z_i + η_k + ε_i) = base_det_z · exp_eta[k] · exp_eps[i]
    # Factoring verified: max relative error < 2e-15 (IEEE 754 rounding only).
    exp_eps = np.empty(n_eps)
    for ie in range(n_eps):
        exp_eps[ie] = exp(eps_nodes[ie])

    exp_eta = np.empty(n_eta)
    for ke in range(n_eta):
        exp_eta[ke] = exp(eta_nodes[ke])

    base_det_z = exp(log_det_next + rho * z_grid[z_idx])
```

**Why this is safe in Numba:** Small array allocations inside `@njit`
are well-optimised. The cost is `n_eta + n_eps + 1` exp() calls per
FOC evaluation (~13 at defaults), replacing what would otherwise be
~4,300 exp() calls per FOC without factoring. The arrays are ~48–80
bytes each and fit in L1 cache.

### 4.3 Inner loop replacement (lines 849–855)

```python
                # ── CURRENT (lines 849–855) ──
                for i_e in range(n_eps):
                    weight = p_out_base * eps_weights[i_e]

                    # Interpolate income in z
                    income_next = ((1.0 - frac_z) * income_next_table[iz_lo, i_e]
                                   + frac_z * income_next_table[iz_lo + 1, i_e])
                    x_next = w_inv + income_next

                # ── NEW ──
                det_z_eta = base_det_z * exp_eta[k_eta]

                for i_e in range(n_eps):
                    weight = p_out_base * eps_weights[i_e]

                    # On-the-fly income: 2 multiplies + 7-bracket tax walk
                    y_gross_next = det_z_eta * exp_eps[i_e]
                    income_next  = scalar_disposable_income(y_gross_next)
                    x_next = w_inv + income_next
```

Note: `det_z_eta = base_det_z * exp_eta[k_eta]` is placed **outside**
the `i_e` loop but **inside** the `k_eta` loop (between the `frac_z`
clamping at line 845 and the `for i_e` at line 849). This mirrors how
the z-bracket computation is already hoisted to the `k_eta` level.

**Everything from line 857 onward is untouched** — `find_bracket`,
bilinear `c_next_full` interpolation, CRRA marginal utility, FOC/Jacobian
accumulators.

### 4.4 Mathematical verification

The new code computes:

```
y_gross = exp(log_det_profile[t+1] + rho·z_grid[z_idx] + eta_nodes[k] + eps_nodes[i])
income  = scalar_disposable_income(y_gross)
```

This is the exact gross-to-net income mapping used by:
- Precompute: `working_income[t, iz, ie] = disp_inc(exp(log_det[t] + z_grid[iz] + eps[ie]))` ([precompute.py:384](precompute.py#L384))
- Simulation: `income = _scalar_disposable_income(exp(log_det[t+1] + z_next + eps[ie]))` ([simulation.py:557](simulation.py#L557))

The only difference from the table lookup is that `z_next` is the
continuous value `rho·z_i + η_k`, not a linear interpolation between
grid neighbours. This eliminates the `cosh(dz/2) − 1 ≈ 16%` chord
error.

**Range safety:** The exp() argument spans [−9.2, 7.4] across all
plausible (age, z, η, ε) combinations. IEEE overflow threshold is 709.8.
The scalar income function returns non-negative values at all inputs
(verified numerically across z ∈ [−8, 8]).

**z_next is NOT clamped for income** (nor should it be). The simulation
clamps z for policy lookup; the income function is well-defined at all
z. The solver already uses unclamped z_next implicitly (the bracket
clamping only affects policy interpolation indices, not the z_next value
itself). This means the solver computes more accurate income than the
simulation at tail z values — a strict improvement.

---

## Step 5 — `solver.py`: Rename parameter in portfolio solvers

Pure mechanical find-replace. **No logic changes.**

### 5.1 `solve_portfolio_2d_working` (line 892)

Signature: `income_next_table` → `log_det_next` at line 894.

8 internal call sites to `compute_foc_jac_working` (lines 914, 924, 936,
947, 962, 982, 1005, 1029): replace `income_next_table` with
`log_det_next` in each argument list. Same positional slot.

### 5.2 `solve_portfolio_unconstrained_working` (line 1208)

Signature: `income_next_table` → `log_det_next` at line 1210.

5 internal call sites (lines 1231, 1241, 1254, 1291, 1320): same rename.

**Verification:** All 13 renamed call sites pass `log_det_next` in the
same argument position where `income_next_table` was. Because both are
positional arguments (not keyword), the type change from `float64[:]` to
`float64` is transparent to Numba — the `@njit` function compiles a new
specialisation on first call.

---

## Step 6 — `solver.py`: Rename parameter in step functions

### 6.1 `_solve_working_age_step_jit` (line 1742)

Signature: `income_next_table` → `log_det_next` at line 1744.

Update docstring at line 1757:
```python
# CURRENT
#     income_next_table  : (n_z, n_eps)          after-tax labor income at t+1

# NEW
#     log_det_next       : float                 log deterministic profile f(age_{t+1})
```

2 internal call sites (lines 1807, 1822): rename parameter in calls to
`solve_portfolio_2d_working` / `solve_portfolio_unconstrained_working`.

### 6.2 `solve_working_age_step` (line 1903)

Non-JIT wrapper. Signature: line 1904. Pass-through: line 1914.
Both: `income_next_table` → `log_det_next`.

---

## Step 7 — `solver.py`: Update `run_lifecycle_solver`

### 7.1 Remove unused table reference (line 1993)

```python
# CURRENT
working_income_table = pc.working_income          # (n_age, n_z, n_eps)

# NEW — table no longer consumed by the solver
# working_income_table: retained in pc for simulation/diagnostics, not used here
log_det_profile = pc.log_det_profile              # (n_age,)
```

### 7.2 Working-age call site (line 2073–2079)

```python
# CURRENT
c, a_s, a_b, _di, _df = solve_working_age_step(
    w_grid, s_grid, z_grid, N_state,
    c_next, working_income_table[t + 1, :, :],
    annuity_factors, rho, eta_nodes, eta_weights, dz,
    ...)

# NEW
c, a_s, a_b, _di, _df = solve_working_age_step(
    w_grid, s_grid, z_grid, N_state,
    c_next, log_det_profile[t + 1],
    annuity_factors, rho, eta_nodes, eta_weights, dz,
    ...)
```

**Age index alignment verified:** `log_det_profile[t]` corresponds to
`ages[t]` (confirmed from [precompute.py:202](precompute.py#L202)). The
solver solves at age `t` looking forward to `t+1`, so `log_det_profile[t+1]`
gives the correct deterministic component for the next-period income.
This matches the current `working_income_table[t+1, :, :]`, which was
built from `exp(log_det_profile[t+1] + z + eps)`.

---

## Step 8 — `test_economics.py`: Update imports and E2

### 8.1 Update imports (lines 19–25)

```python
# CURRENT
from model import (
    DiscretizationConfig, disposable_income_working, compute_pension_after_tax,
)
from simulation import (
    simulate_lifecycle, _scalar_disposable_income, _scalar_pension_after_tax,
    fast_interp_1d,
)

# NEW
from model import (
    DiscretizationConfig, disposable_income_working, compute_pension_after_tax,
    scalar_disposable_income,
)
from simulation import (
    simulate_lifecycle, _scalar_pension_after_tax,
    fast_interp_1d,
)
```

Update the warmup call at line 82: `_scalar_disposable_income(1.0)` →
`scalar_disposable_income(1.0)`.

Update all other `_scalar_disposable_income` references in the file
(lines 189, 445) → `scalar_disposable_income`.

### 8.2 Revise E2 test (lines 155–213)

The old test measured interpolation error between the table-based solver
approach and the direct computation. Now both use the same formula, so
the interpolation error is gone. Replace with a **consistency check**
that verifies the scalar function agrees with the vectorized table at
grid points (where the two must be identical):

```python
    print("=" * 70)
    print("E2. SCALAR vs VECTORIZED INCOME CONSISTENCY")
    print("=" * 70)
    # After the on-the-fly refactor, the solver uses scalar_disposable_income
    # at continuous z. Verify it matches the vectorized table at grid points.
    max_diff = 0.0
    for t_test in [0, 10, 20, 30, 44]:
        for iz in range(pc.n_z):
            for ie in range(len(pc.eps_nodes)):
                y_gross = np.exp(pc.log_det_profile[t_test]
                                 + pc.z_grid[iz] + pc.eps_nodes[ie])
                inc_scalar = scalar_disposable_income(y_gross)
                inc_table  = pc.working_income[t_test, iz, ie]
                max_diff = max(max_diff, abs(inc_scalar - inc_table))

    print(f"  Max |scalar - vectorized table| at grid points: {max_diff:.2e}")
    test("Scalar ≡ vectorized at grid points",
         max_diff < 1e-12,
         f"max diff = {max_diff:.2e}")
```

---

## Step 9 — Verification Checklist

After implementation, before committing:

| Check | Command / Method | Expected |
|-------|-----------------|----------|
| Numba cache clear | `find . -name '__pycache__' -exec rm -rf {} +` | Clean recompilation |
| Unit tests pass | `python test_economics.py` | All E1–E8 pass |
| E2 grid-point consistency | Automatic in test suite | max diff < 1e-12 |
| Solver runs without error | Run one age step in notebook | No Numba type errors |
| FOC residuals comparable | Compare post-solve diagnostics | Residuals ≤ baseline |
| Newton iteration counts | Compare `%int` / `Newt%` columns | No degradation |
| Euler equation residuals (E3) | Automatic in test suite | ≤ baseline or better |
| Full solver wall-time | Time the notebook solve cell | +15% ± 10% vs baseline |

---

## Summary of All Edits

| File | Lines changed | Nature |
|------|:---:|--------|
| `model.py` | +16 | Add `scalar_disposable_income` + `njit` import |
| `simulation.py` | −26, +3 | Remove local function, import + alias |
| `solver.py` line 31 | 1 | Add import |
| `solver.py` line 771 | 1 | Signature: `income_next_table` → `log_det_next` |
| `solver.py` lines 790–791 | +12 | Insert exp() precompute block |
| `solver.py` lines 849–855 | 6 → 6 | Replace interpolation with factored on-the-fly |
| `solver.py` lines 894–1320 | 13 | Rename parameter at call sites (mechanical) |
| `solver.py` lines 1210–1320 | 6 | Same rename in unconstrained solver |
| `solver.py` lines 1742–1914 | 6 | Same rename in step functions + docstring |
| `solver.py` lines 1993, 2075 | 2 | Master solver: pass `log_det_profile[t+1]` |
| `test_economics.py` | ~20 | Update imports + revise E2 |
| **Total** | **~85** | 2 lines economic logic + 12 lines optimisation + ~70 mechanical |

---

## What Does NOT Change

- Consumption policy interpolation (still z-grid bilinear — separate refactor)
- Retirement solver (pension, no z-transition)
- Terminal solver
- Simulation engine (already uses scalar function)
- `Precompute.working_income` table (retained for simulation/diagnostics)
- Pension formula
- Bequest hoist optimisation
- z-grid construction, z-bracketing for consumption
- Newton convergence logic
- All function return types and shapes

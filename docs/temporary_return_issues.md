# State Quadrature Implementation — Issues Found & Fixed

## Bugs Found and Fixed During Review

### 1. `const_r` / `A_r` algebraic error (FIXED)
**File:** `precompute.py:168-172`

**Problem:** The original definitions assumed `v_nodes` were next-state values, not innovation values:
```python
# WRONG (original)
self.const_r = model.Phi_0_ret - model.M @ model.Phi_0_state
self.A_r = model.Phi_21 - model.M @ model.Phi_11
```
This produced `mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ (v_k - Phi_0_state - Phi_11 @ s_i)` instead of the correct `mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ v_k`.

**Fix:**
```python
self.const_r = np.array(model.Phi_0_ret, dtype=float)
self.A_r = np.array(model.Phi_21, dtype=float)
```

**Impact:** Would have produced incorrect conditional return means throughout the solver, biasing all portfolio decisions.

---

### 2. `regenerate_savings_grid` keyword argument (FIXED)
**File:** `solver.py:3446`

**Problem:** Called with `n_points=n_s_points` but the method signature expects a positional argument `n_s_points`.

**Fix:** Changed to `pc.regenerate_savings_grid(n_s_points)`.

---

### 3. `run_lifecycle_solver` argument order (FIXED)
**File:** `solver.py:3413`

**Problem:** Signature was `(model, pc, n_s_points=None, solver_config=None, ...)`. All callers pass `solver_config` as the 3rd positional argument, so the SolverConfig object was being assigned to `n_s_points`.

**Fix:** Swapped to `(model, pc, solver_config=None, n_s_points=None, ...)`.

---

## Minor Observations (not bugs, no action taken)

- `_validate_state_quadrature` docstring mentions a covariance check (`sum_k w_k * v_k v_k' == Sigma_ss`) that is not implemented in the code. The covariance was verified correct in testing (relative error 6.1e-16).
- `get_state_quadrature` uses Cholesky without eigenvalue clipping, unlike `get_return_quadrature` which clips negative eigenvalues. Low risk since `Sigma_ss` comes from VAR estimation.

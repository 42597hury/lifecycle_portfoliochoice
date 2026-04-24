# Implementation Guide: Catmull-Rom Cubic Z-Interpolation in `compute_foc_jac_working`

## 1. Problem Statement

The lifecycle solver evaluates next-period marginal utility `u'(c_{t+1})` at continuous
`z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]`, which generally falls between
z-grid points. The current code linearly interpolates `c_next_full` in z between
two bracketing grid values (`solver.py:874-882`).

Because `c*(z) ~ exp(z)` at high wealth and `exp(dz) = 3.07` per grid step (at nz=11),
linear interpolation **systematically overestimates** consumption in the convex region,
underestimating marginal utility. This bias compounds nearly linearly over 45 working
periods of backward induction, producing large discretization error in the final policy.

The fix: replace linear z-interpolation with Catmull-Rom cubic spline interpolation
inside `compute_foc_jac_working`. This is the **only** interpolation site that needs
to change.

## 2. What to Change

### 2.1 Single modification site

**File:** `solver.py`
**Function:** `compute_foc_jac_working` (line 770, `@njit(fastmath=True)`)
**Lines:** 854-883 (z-bracket computation and bilinear interpolation)

No other function performs z-interpolation. Specifically:
- `compute_foc_jac_retirement` (line 427): no z dimension at all, only wealth + financial state
- `solve_portfolio_2d_retirement` (line 502): calls retirement FOC, no z
- Terminal solvers: no z-interpolation
- All working-age portfolio solvers (constrained `solve_portfolio_2d_working` line 907,
  unconstrained `solve_portfolio_unconstrained_working` line 1223, and the scipy fallback)
  all call `compute_foc_jac_working` — changing that one function covers everything.

### 2.2 Current code (lines 854-883)

```python
# Bracket z_next on the uniform z_grid
iz_lo = int((z_next - z_grid[0]) / dz)
iz_lo = max(0, min(iz_lo, n_z - 2))
frac_z = (z_next - z_grid[iz_lo]) / dz
frac_z = max(0.0, min(1.0, frac_z))

# ... (lines 860-872: income computation, wealth bracket search) ...

# CURRENT: Bilinear interpolation (2 z-points x 2 wealth-points = 4 reads)
c_lo = (1.0 - frac_w) * c_next_full[iz_lo, j_s, iw] + frac_w * c_next_full[iz_lo, j_s, iw + 1]
c_hi = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]

mpc_lo = (c_next_full[iz_lo, j_s, iw + 1] - c_next_full[iz_lo, j_s, iw]) * inv_dw
mpc_hi = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw

c_next = (1.0 - frac_z) * c_lo + frac_z * c_hi
c_next = max(c_next, min_consumption)
mpc = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi
mpc = max(0.0, min(1.0, mpc))
```

### 2.3 Replacement code

The Catmull-Rom spline interpolates between points p1 and p2 using two additional
neighbors p0 and p1. For parameter `f` in [0, 1]:

```
c(f) = p1 + 0.5*f*(-p0 + p2) + 0.5*f^2*(2*p0 - 5*p1 + 4*p2 - p3) + 0.5*f^3*(-p0 + 3*p1 - 3*p2 + p3)
```

This is equivalent to the standard Catmull-Rom basis with tau=0.5. At f=0 it returns p1,
at f=1 it returns p2, and it is C1-continuous across intervals.

Replace lines 854-883 with:

```python
# Bracket z_next on the uniform z_grid
iz_lo = int((z_next - z_grid[0]) / dz)
iz_lo = max(0, min(iz_lo, n_z - 2))
frac_z = (z_next - z_grid[iz_lo]) / dz
frac_z = max(0.0, min(1.0, frac_z))

# Determine stencil availability for cubic
use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)

# ... (lines 860-872 are UNCHANGED: income computation, wealth bracket) ...

if use_cubic:
    # Catmull-Rom: 4 z-points x 2 wealth-points = 8 reads for c
    c_zm1 = (1.0 - frac_w) * c_next_full[iz_lo - 1, j_s, iw] + frac_w * c_next_full[iz_lo - 1, j_s, iw + 1]
    c_z0  = (1.0 - frac_w) * c_next_full[iz_lo,     j_s, iw] + frac_w * c_next_full[iz_lo,     j_s, iw + 1]
    c_z1  = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]
    c_z2  = (1.0 - frac_w) * c_next_full[iz_lo + 2, j_s, iw] + frac_w * c_next_full[iz_lo + 2, j_s, iw + 1]

    f = frac_z
    f2 = f * f
    f3 = f2 * f
    c_next = (c_z0
              + 0.5 * f  * (-c_zm1 + c_z1)
              + 0.5 * f2 * (2.0*c_zm1 - 5.0*c_z0 + 4.0*c_z1 - c_z2)
              + 0.5 * f3 * (-c_zm1 + 3.0*c_z0 - 3.0*c_z1 + c_z2))
    c_next = max(c_next, min_consumption)

    # MPC: Catmull-Rom on wealth slopes at the same 4 z-points
    mpc_zm1 = (c_next_full[iz_lo - 1, j_s, iw + 1] - c_next_full[iz_lo - 1, j_s, iw]) * inv_dw
    mpc_z0  = (c_next_full[iz_lo,     j_s, iw + 1] - c_next_full[iz_lo,     j_s, iw]) * inv_dw
    mpc_z1  = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw
    mpc_z2  = (c_next_full[iz_lo + 2, j_s, iw + 1] - c_next_full[iz_lo + 2, j_s, iw]) * inv_dw

    mpc = (mpc_z0
           + 0.5 * f  * (-mpc_zm1 + mpc_z1)
           + 0.5 * f2 * (2.0*mpc_zm1 - 5.0*mpc_z0 + 4.0*mpc_z1 - mpc_z2)
           + 0.5 * f3 * (-mpc_zm1 + 3.0*mpc_z0 - 3.0*mpc_z1 + mpc_z2))
    mpc = max(0.0, min(1.0, mpc))

else:
    # Boundary fallback: linear (identical to current code)
    c_lo = (1.0 - frac_w) * c_next_full[iz_lo, j_s, iw] + frac_w * c_next_full[iz_lo, j_s, iw + 1]
    c_hi = (1.0 - frac_w) * c_next_full[iz_lo + 1, j_s, iw] + frac_w * c_next_full[iz_lo + 1, j_s, iw + 1]

    mpc_lo = (c_next_full[iz_lo, j_s, iw + 1] - c_next_full[iz_lo, j_s, iw]) * inv_dw
    mpc_hi = (c_next_full[iz_lo + 1, j_s, iw + 1] - c_next_full[iz_lo + 1, j_s, iw]) * inv_dw

    c_next = (1.0 - frac_z) * c_lo + frac_z * c_hi
    c_next = max(c_next, min_consumption)
    mpc = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi
    mpc = max(0.0, min(1.0, mpc))
```

**IMPORTANT:** The `use_cubic` check must be placed OUTSIDE the `for i_e` loop (but
inside the `for k_eta` loop), since `iz_lo` does not depend on `i_e`. This avoids
re-evaluating the branch on every transitory shock iteration. The code after the
if/else block (lines 885-898: `mu_alive`, `mup_alive`, accumulations into `euler_sum`,
`foc_s`, `foc_b`, `J_ss`, `J_bb`, `J_sb`) is **completely unchanged**.

### 2.4 Where the `use_cubic` branch and the `else` branch apply

The solver has nz=11 z-grid points. `iz_lo` ranges from 0 to 9 (n_z - 2).
- `use_cubic` requires `iz_lo >= 1` and `iz_lo + 2 < 11`, i.e., `iz_lo` in {1, 2, ..., 8}
- `else` (linear fallback): `iz_lo == 0` or `iz_lo == 9`

From the actual eta quadrature landing zones (rho=0.991, 6 GH nodes):
- iz_lo=0 is hit by 9 of 66 (z_idx, eta) combinations (all from z_idx=0, some from z_idx=1)
- iz_lo=9 is hit by 8 of 66 combinations (some from z_idx=9, all from z_idx=10)
- These boundary states carry 0.35% + 1.44% = 1.79% of stationary z-mass (each side)
- **74.2% of (z_idx, eta) combinations use the cubic path**

## 3. Why MPC Must Use the Same Catmull-Rom Stencil

The `mpc` value computes `dc/dx` (marginal propensity to consume out of wealth), which
feeds into the Jacobian of the portfolio FOC:

```python
mup_alive = -gamma * mu_alive / c_next * mpc      # line 886
wmup = weight * psi * mup_alive                    # line 889
jac = wmup * s_val                                 # line 895
J_ss += jac * Rex_s * Rex_s                        # line 896
J_bb += jac * Rex_b * Rex_b                        # line 897
J_sb += jac * Rex_s * Rex_b                        # line 898
```

The 2D Newton solver (both constrained `solve_portfolio_2d_working` and unconstrained
`solve_portfolio_unconstrained_working`) uses these Jacobian elements to compute the
Newton step direction. If `c_next` is computed with cubic but `mpc` with linear, the
Jacobian is inconsistent with the function evaluation. This won't cause crashes (the
Newton solver has damping, singular-Jacobian fallback, and triangle projection), but
it will degrade convergence rate and may produce slightly wrong portfolio allocations.

The correct approach: apply the same Catmull-Rom polynomial to the wealth slopes at
4 z-points, exactly as shown in section 2.3. This gives 16 array reads total
(8 for c, 8 for mpc) vs the current 8 (4 for c, 4 for mpc). The boundary fallback
keeps the current 8 reads.

## 4. Boundary Handling

At `iz_lo == 0` or `iz_lo == n_z - 2`, the 4-point stencil would require out-of-bounds
array access (`iz_lo - 1 = -1` or `iz_lo + 2 = n_z`). The implementation falls back
to the current linear interpolation at these boundaries.

**Why this is acceptable:**
- These boundaries affect only 3.6% of stationary z-mass
- At iz_lo=0 (very low z), consumption is near the borrowing constraint floor and
  nearly flat in z — linear interpolation error is small
- At iz_lo=9 (very high z), the error is larger but affects only z_idx=9 and z_idx=10
  (1.79% of mass combined)
- Testing showed linear boundary P95 error = 4.8%, quadratic fallback = 2.8%.
  The improvement from quadratic is modest and adds complexity. Linear fallback is
  the pragmatic choice for the first implementation.

**If boundary accuracy becomes a concern later**, a quadratic fallback using 3 points
is straightforward:
- iz_lo=0: use c[0], c[1], c[2] with Lagrange quadratic
- iz_lo=n_z-2: use c[n_z-3], c[n_z-2], c[n_z-1] with Lagrange quadratic

## 5. Performance

Benchmarked on the actual loop structure with Numba `@njit(fastmath=True)`:

| Benchmark | Linear | Cubic | Overhead |
|---|---|---|---|
| Inner loop only (200k calls) | 1.90 us/call | 2.03 us/call | **7%** |
| Full FOC function (500 calls, 343 states, 6 eta, 5 eps) | 0.47 ms/call | 0.55 ms/call | **16%** |

The `c_next_full` array is ~4.5 MB (11 x 343 x 150 x 8 bytes) and fits in L2/L3 cache.
The extra array reads do not cause cache misses. The overhead is dominated by the
additional floating-point arithmetic (cubic polynomial evaluation), not memory access.

The FOC function is called inside a Newton loop (typically 3-8 iterations) which sits
inside the EGM savings loop. End-to-end solver slowdown will be well under 16%.

## 6. Convergence Safety

### 6.1 Baseline

The nz=11, 7x7x7 constrained run has:
- 0 Newton failures out of 43,578,150 calls
- 0 monotonicity violations
- Worst FOC residual: 1.0e-7 (at tolerance threshold)

### 6.2 Why cubic should not cause new failures

**The Newton solver has multiple layers of robustness that are NOT being changed:**

1. **Corner detection** (lines 935-967): evaluates FOC at (0,0), (1,0), (0,1) and
   checks sign conditions. Cubic changes magnitudes slightly, not signs. The economic
   direction of excess return premia is not affected by interpolation method.

2. **Edge solvers** (lines 969-1032): 1D Newton on each edge with `edge_max_iter=8`
   and `edge_accept_factor=10.0`. These use the same `compute_foc_jac_working` — they
   get the cubic improvement too, consistently.

3. **Interior Newton** (lines 1034-1071):
   - Step damping: `step_damp=0.2` caps Newton step length
   - Singular Jacobian fallback: gradient descent if `|det| < 1e-15`
   - Triangle projection: `project_to_triangle` clamps to feasible (alpha_s, alpha_b)
   - Max 20 iterations

4. **Unconstrained Newton** (lines 1275-1341): additionally has backtracking line search
   (`max_backtrack_iter=10`), which will accept the cubic-computed FOC as long as the
   residual decreases — it doesn't care about the interpolation method.

### 6.3 Potential concern: intermediate policies

During backward induction, the policy at period `t+1` may have sharper features than
the final converged policy (especially near retirement transition at age 67). If the
cubic overshoots into negative consumption at an intermediate step:

- `c_next = max(c_next, min_consumption)` clamps it (already present for linear)
- `mpc = max(0.0, min(1.0, mpc))` clamps the derivative

Testing on the converged nz=11 policy showed **0 cases** of cubic producing negative
values across 4+ million probes. Intermediate policies may be rougher, but the clamps
provide a hard safety net.

### 6.4 Recommended convergence monitoring

After implementation, run the solver with **the same configuration** as the baseline
(`constrained_grid7x7x7_nz11`) and compare diagnostics:

```python
import pickle

d_old = pickle.load(open('saved_runs/constrained_grid7x7x7_nz11/diagnostics.pkl', 'rb'))
d_new = pickle.load(open('saved_runs/<new_run>/diagnostics.pkl', 'rb'))

print(f"Newton failures:     old={d_old['total_newton_failures']}  new={d_new['total_newton_failures']}")
print(f"Mono violations:     old={d_old['total_mono_violations']}  new={d_new['total_mono_violations']}")
print(f"Worst FOC residual:  old={d_old['worst_foc_resid']:.2e}  new={d_new['worst_foc_resid']:.2e}")

# Per-age diagnostics
# d['age_diag_int'] has shape (78, 13), column indices are DI_* constants
# DI_NEWTON_FAIL=7, DI_INTERIOR=6, DI_NEG_CONSUMPTION=11, DI_MONO_VIOLATIONS=12
```

**Acceptable results:**
- Newton failures: 0 (same as baseline)
- Mono violations: 0 or very small count
- Worst FOC residual: same order of magnitude (1e-7 to 1e-6)

**If Newton failures increase:** check which ages/states fail. If concentrated at
boundary iz_lo values, consider upgrading the boundary fallback to quadratic. If at
interior iz_lo, the cubic may be overshooting on a rough intermediate policy — add
a diagnostic print in the Newton failure path showing iz_lo and frac_z.

## 7. Testing the Implementation

### 7.1 Unit test: cubic reproduces linear when c is linear in z

If `c_next_full[iz, j_s, iw] = a + b * iz` for all iz, both linear and cubic
interpolation should give identical results (Catmull-Rom is exact for linear functions).

```python
# In a test file
c_linear = np.zeros((n_z, N_state, n_w))
for iz in range(n_z):
    c_linear[iz, :, :] = 1.0 + 0.5 * iz  # linear in z

# Call compute_foc_jac_working with c_linear as c_next_full
# Compare output (foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum) between
# old linear code and new cubic code. Should match to machine precision.
```

### 7.2 Unit test: Catmull-Rom is exact for quadratic polynomials

If `c[iz] = a + b*iz + c*iz^2`, Catmull-Rom should interpolate exactly at any frac_z.
Construct such a policy and verify the interpolated value matches the polynomial
evaluated at z_next. (Note: Catmull-Rom is NOT exact for cubics — the error for a
cubic term `d*iz^3` is `d * f*(1-f)*(1-2f)`, peaking at ~9.6% of d.)

### 7.3 Integration test: run the full solver and compare

Run `run_lifecycle_solver` with the cubic change, using the same config as
`constrained_grid7x7x7_nz11`. Save the new policy and compare:

```python
C_old = np.load('saved_runs/constrained_grid7x7x7_nz11/policy_arrays.npz')['C_mat']
C_new = np.load('saved_runs/<new_run>/policy_arrays.npz')['C_mat']

# The policies SHOULD differ (that's the point). Check:
# 1. New policy is everywhere positive
assert np.all(C_new > 0)

# 2. Differences are concentrated at high wealth, working ages
diff = np.abs(C_new - C_old) / np.maximum(C_old, 1e-10)
for t in [0, 20, 40, 55, 77]:
    print(f"age={t+22}: median diff={np.median(diff[t])*100:.2f}%, max={np.max(diff[t])*100:.2f}%")

# 3. At retirement ages (t >= 45), differences should be ~0 (no z-interp)
assert np.max(diff[50:]) < 1e-6

# 4. Differences should grow backward from retirement (compounding)
# age 65: tiny, age 45: moderate, age 25: largest
```

### 7.4 Second-difference test on the new policy

Rerun the same diagnostic from our investigation on the new policy to verify the
interpolation error has decreased:

```python
C = C_new  # (78, 11, 343, 150)
n_age, n_z, n_state, n_w = C.shape

bands = [(0,25), (25,50), (50,75), (75,100), (100,125), (125,150)]
for blo, bhi in bands:
    errs = []
    for t in range(n_age):
        for iz in range(1, n_z - 1):
            for js in range(n_state):
                for iw in range(blo, min(bhi, n_w)):
                    c_lo = C[t, iz-1, js, iw]
                    c_hi = C[t, iz+1, js, iw]
                    c_mid = C[t, iz, js, iw]
                    if c_mid > 1e-10:
                        err = abs(0.5*(c_lo + c_hi) - c_mid) / c_mid / 4.0
                        errs.append(err)
    arr = np.array(errs)
    print(f"iw [{blo:3d},{bhi:3d}): median={np.median(arr)*100:.2f}%  p95={np.percentile(arr,95)*100:.2f}%")
```

**Note:** The second-difference test measures curvature of the POLICY, not the
interpolation error of the solver. With cubic interpolation, the solver produces
a smoother policy, so the second-difference values should decrease — but they
measure a different thing than before. The policy changing is the desired outcome.

## 8. Numerical Details

### 8.1 Catmull-Rom polynomial coefficients

For stencil points (p0, p1, p2, p3) = c at (iz_lo-1, iz_lo, iz_lo+1, iz_lo+2):

```
c(f) = p1 + 0.5*f*(-p0 + p2)
           + 0.5*f^2*(2*p0 - 5*p1 + 4*p2 - p3)
           + 0.5*f^3*(-p0 + 3*p1 - 3*p2 + p3)
```

Properties:
- c(0) = p1, c(1) = p2 (interpolation)
- c'(0) = (p2 - p0) / 2, c'(1) = (p3 - p1) / 2 (C1 continuity across intervals)
- Exact for polynomials up to degree 2 (linear and quadratic)
- For cubics, error is d * f*(1-f)*(1-2f) where d is the cubic coefficient
  (zero at endpoints and midpoint, max ~9.6% of d)

### 8.2 Horner form for numerical stability

The polynomial should be evaluated in Horner form to minimize floating-point operations
and maximize numerical stability:

```python
f = frac_z
f2 = f * f
f3 = f2 * f
```

This is 3 multiplies + 8 multiply-adds for c_next (same for mpc), vs 2 multiply-adds
for the current linear version. The `fastmath=True` flag on the `@njit` decorator
allows Numba to use fused multiply-add (FMA) instructions if available.

### 8.3 Array memory layout

`c_next_full` has shape `(n_z, N_state, n_w)` — z is the FIRST axis. Accessing
`c_next_full[iz_lo-1, j_s, iw]` through `c_next_full[iz_lo+2, j_s, iw]` reads from
4 different z-slices. In C-contiguous (row-major) memory, these are `N_state * n_w`
elements apart. At n_w=150, N_state=343, that's ~400KB between z-slices — all within
L2 cache (typically 256KB-1MB per core). This is why the benchmark showed only 7-16%
overhead despite 4x more z-slice reads.

## 9. What NOT to Change

- **Lines 851-858** (z_next computation and iz_lo bracket): unchanged. The bracket
  finding is identical — cubic just reads additional neighbors.
- **Lines 860-872** (income computation, wealth bracket search): unchanged.
- **Lines 885-898** (marginal utility, accumulations): unchanged. These use `c_next`
  and `mpc` which are set by the new code, but the formulas are the same.
- **The Newton solvers** (`solve_portfolio_2d_working`, `solve_portfolio_unconstrained_working`):
  unchanged. They call `compute_foc_jac_working` which now internally uses cubic.
- **The EGM interpolation** in `_solve_working_age_step_jit` (lines 1757+): this
  interpolates the endogenous-grid policy to the exogenous wealth grid. It does NOT
  interpolate in z. Leave it alone.
- **Retirement solvers**: no z-interpolation at all. Leave them alone.
- **The function signature** of `compute_foc_jac_working`: unchanged. No new parameters
  needed — `n_z` is already available from `len(z_grid)`.
- **Diagnostics**: the exit code and diagnostic tracking infrastructure is unchanged.
  The diagnostics already track Newton failures, monotonicity violations, and FOC
  residuals, which is exactly what we need to verify the change.

## 10. Reference: Test Results from Investigation

### 10.1 Baseline interpolation error (nz=11, linear, second-difference / 4)

| Wealth band | Median | P95 | Max |
|---|---|---|---|
| iw [0,25) < $62 | 0.13% | 0.74% | 1.06% |
| iw [25,50) $62-700 | 0.12% | 1.26% | 5.02% |
| iw [50,75) $700-8k | 0.18% | 4.71% | 6.60% |
| iw [75,100) $8k-92k | 0.84% | 5.23% | 6.86% |
| iw [100,125) $92k-1M | 1.15% | 5.19% | 6.83% |
| iw [125,150) $1M-10.8M | 0.53% | 5.30% | 7.24% |

### 10.2 Cubic vs linear: leave-one-out at 2*dz (interior z only)

| Wealth band | Linear P95 | Cubic P95 | Reduction |
|---|---|---|---|
| iw [0,25) | 0.558% | 0.620% | -11% |
| iw [25,50) | 1.185% | 0.954% | 20% |
| iw [50,75) | 4.398% | 3.272% | 26% |
| iw [75,100) | 5.793% | 4.258% | 27% |
| iw [100,125) | 5.696% | 4.206% | 26% |
| iw [125,150) | 4.598% | 2.613% | 43% |

### 10.3 Marginal utility head-to-head (gamma=3, interior z, all wealth)

| Method | Median | P95 | P99 | Max |
|---|---|---|---|---|
| Linear | 4.27% | 43.91% | 51.22% | 74.98% |
| Cubic | 3.12% | 34.75% | 45.18% | 64.90% |

Cubic wins 74.4% of probes in head-to-head marginal utility comparison.
Zero overshoot/negative cases across 4+ million probes.

### 10.4 Key finding: error is primarily discretization, not interpolation

The per-period interpolation perturbation to the Euler integrand is ~2.5% (P95).
This compounds nearly linearly over 45 working periods because the bias is systematic
(linear interpolation overestimates c in the convex exp(z) region). Cubic reduces the
per-period perturbation by ~26%, which translates to a ~26% reduction in the compounded
discretization error. The full effect can only be measured by re-solving with the cubic
change — the post-hoc tests on the linear-solved policy underestimate the improvement
because the policy itself will shift toward more accurate precautionary savings.

### 10.5 Why log-linear interpolation was rejected

Log-linear (geometric mean) interpolation was also tested. While it reduces consumption
P95 error by ~48% at high wealth, it **worsens** the worst-case marginal utility error
(from 76% to 109%) because c*(z) has mixed curvature: concave at low wealth
(precautionary savings), convex at high wealth (c ~ exp(z)). Log-linear underestimates
in the concave region, and at gamma=3 this underestimate gets amplified. Cubic handles
both convex and concave regions natively.

## 11. Summary Checklist

- [ ] Replace lines 874-883 in `compute_foc_jac_working` with the cubic/fallback code from section 2.3
- [ ] Place `use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)` outside the `for i_e` loop, inside `for k_eta`
- [ ] Verify Numba compilation succeeds (`@njit(fastmath=True)`)
- [ ] Run unit test: linear c(z) produces identical results with both methods
- [ ] Run unit test: quadratic c(z) is interpolated exactly
- [ ] Run full solver with `constrained_grid7x7x7_nz11` config
- [ ] Compare diagnostics: Newton failures = 0, mono violations = 0
- [ ] Compare policies: differences at retirement ages < 1e-6
- [ ] Compare policies: differences grow backward from retirement
- [ ] Run second-difference diagnostic on new policy

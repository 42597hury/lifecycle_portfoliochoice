# Infinite-Horizon Benchmark Implementation Plan

> **Status:** planning handoff only. No solver code has been changed in this pass.
>
> **Goal:** compute an infinite-horizon portfolio policy benchmark and plot it as a
> dotted horizontal comparison line against the lifecycle policy, evaluated at the
> same point in the state space.

## 1. Target object

The benchmark to solve is:

- infinite horizon
- `CRRA` utility
- same annual restricted VAR as the lifecycle model
- same asset menu: bills, stocks, nominal bonds
- state variables are the only predictors
- no labor income
- no pension
- no mortality

This is **not** an exact Campbell-Chan-Viceira replication because the current
project uses `CRRA`, not Epstein-Zin. It is better described as a
"same opportunity set, CRRA, no-human-capital benchmark." That wording should
carry through into the thesis text and figure caption.

The existing VAR setup already matches the required predictor restriction:

- lagged return variables are excluded from forecasting equations
- only state variables enter as predictors

Code references:

- [`var.py:build_nominal_system1_var_config()`](../var.py#L381)
- [`precompute.py:build_model()`](../precompute.py#L424)

---

## 2. Scope of the benchmark

The benchmark is only needed for a **policy comparison figure**.

That means:

- no need for a full new simulation framework
- no need to rewrite the lifecycle solver
- no need to remove the `z` dimension structurally in version 1

What is needed:

1. solve a stationary policy problem
2. evaluate that policy at the same state-space point used for the lifecycle plot
3. overlay the resulting allocation as a dotted horizontal line

The infinite-horizon policy is stationary in age, but it can still depend on:

- wealth
- financial state

Therefore the comparison must be made at the **same reference point** as the
lifecycle plot:

- same financial state
- same wealth

This is the correct apples-to-apples comparison.

---

## 3. Recommended architecture

### 3.1 Keep the lifecycle code path untouched

Do **not** try to add an "infinite-horizon mode" inside
[`solver.py:run_lifecycle_solver()`](../solver.py#L2500) for the first version.

That path is heavily age-structured:

- terminal condition
- backward induction over ages
- working vs retirement branching
- pension timing
- mortality tables
- partial-solve and checkpoint logic

Using it as the base for an infinite-horizon benchmark would create unnecessary
risk of breaking the lifecycle workflow.

### 3.2 Add a separate stationary benchmark entry point

Recommended new file:

- `inf_horizon_solver.py`

Recommended main entry points:

- `run_infinite_horizon_solver(...)`
- `_run_infinite_horizon_core_jit(...)`

These functions should split responsibilities cleanly:

- `run_infinite_horizon_solver(...)`
  - thin Python wrapper
  - input validation
  - warm-start preparation
  - diagnostics packaging
- `_run_infinite_horizon_core_jit(...)`
  - pure numerical fixed-point loop
  - calls the existing retirement kernel directly
  - returns arrays plus primitive diagnostics

This standalone file placement is preferable because it:

- keeps the benchmark out of the lifecycle solve path
- avoids editing existing module responsibilities in `solver.py`
- makes it easy to import only when the benchmark is actually needed
- reduces the risk that experimental benchmark logic interferes with production
  lifecycle code

---

## 4. Why the retirement operator is the right base

The no-income, no-mortality infinite-horizon problem is very close to the
existing retirement operator.

The reusable core is:

- [`solver.py:compute_foc_jac_retirement_quad()`](../solver.py#L402)
- [`solver.py:solve_retirement_step_quad()`](../solver.py#L2042)

Those routines already solve a one-period portfolio and consumption problem with:

- current wealth
- current financial state
- uncertain next-period financial state
- uncertain asset returns
- continuation consumption policy `c_next`

To turn that into the stationary benchmark:

- set survival to one: `psi_vec = np.ones(n_z)`
- set next-period income to zero: `pension_1d = np.zeros(n_z)`
- set `b_bar = 0` in the stationary wrapper as a defense-in-depth safeguard
- iterate on the continuation policy until convergence

When `psi = 1`, the bequest/death branch vanishes automatically inside the FOC.
Setting `b_bar = 0` as well is not mathematically necessary, but it makes the
benchmark robust to any future refactor that might accidentally route a small
death probability into the inner solver.

### 4.1 Why fixed-point iteration should work in this model

The fixed-point step is not an arbitrary numerical hack. In this benchmark, it
is Coleman policy iteration for a stationary `CRRA` consumption-savings problem
with portfolio choice.

Start from the benchmark restrictions:

- no labor income
- no pension
- no mortality
- `b_bar = 0`
- stationary financial opportunity set

Then next-period wealth is:

`W' = a * R_p(s, s', eps_r; alpha)`

where:

- `a = W - c`
- `alpha = (alpha_stock, alpha_bond)`
- `alpha_bill = 1 - alpha_stock - alpha_bond`
- `R_p = alpha_stock * R_stock + alpha_bond * R_bond + alpha_bill * R_bill`

Because the model has `CRRA` utility and no additive income, the problem is
homogeneous in wealth. The exact stationary policy therefore has the form:

- `c(W, s) = xi(s) * W`
- `alpha_stock(W, s) = alpha_stock(s)`
- `alpha_bond(W, s) = alpha_bond(s)`

So the wealth dimension is redundant in the true solution, and the entire
stationary problem reduces to solving for:

- one consumption share `xi(s)` per financial state
- two portfolio shares per financial state

If `xi_old(s')` is the continuation-rule guess, then:

- `c'(W', s') = xi_old(s') * W'`
- `u'(c') = c'^{-gamma} = xi_old(s')^{-gamma} * a^{-gamma} * R_p^{-gamma}`

The Euler equation for current consumption becomes:

`c^{-gamma} = beta * E[ c'^{-gamma} * R_p ]`

Substituting `c = xi(s) * W` and `a = (1 - xi(s)) * W` gives:

`xi(s)^{-gamma} = beta * (1 - xi(s))^{-gamma} * E[ xi_old(s')^{-gamma} * R_p^{1-gamma} ]`

Hence the implied stationary consumption-share update is:

`xi_new(s) = 1 / (1 + [ beta * E( xi_old(s')^{-gamma} * R_p^{1-gamma} ) ]^{1/gamma})`

This is the clean fixed-point characterization in `xi(s)`. The implementation
below does **not** iterate directly on `xi`. It iterates the full stored policy
arrays `C(z, s, w)`, `S(z, s, w)`, and `B(z, s, w)` through the retirement
kernel. Both approaches share the same fixed point. The advantage of iterating
the full arrays is that homogeneity in wealth and invariance across dummy `z`
slices are checked numerically rather than imposed analytically.

The portfolio first-order conditions are:

- `0 = E[ c'^{-gamma} * (R_stock - R_bill) ]`
- `0 = E[ c'^{-gamma} * (R_bond  - R_bill) ]`

and after substituting for `c'`:

- `0 = E[ xi_old(s')^{-gamma} * R_p^{-gamma} * (R_stock - R_bill) ]`
- `0 = E[ xi_old(s')^{-gamma} * R_p^{-gamma} * (R_bond  - R_bill) ]`

These are the interior portfolio conditions. In the constrained benchmark, some
states will lie on a corner or edge of the simplex
`alpha_stock >= 0`, `alpha_bond >= 0`, `alpha_stock + alpha_bond <= 1`, so the
correct characterization there is KKT rather than the interior FOC. The
existing constrained retirement solver already handles those corner and edge
cases, so the fixed-point iteration still converges to the correct constrained
stationary policy.

This is exactly the system the existing retirement kernel is solving in the
benchmark limit `psi = 1`, `pension = 0`, `b_bar = 0`.

Why the operator is well-defined in this project:

1. The discretized state process has finitely many stored grid states and
   finitely many quadrature nodes.
2. All asset gross returns are strictly positive because they are exponentials
   of finite log-return nodes.
3. With constrained shares, the portfolio simplex is compact:
   - `alpha_stock >= 0`
   - `alpha_bond >= 0`
   - `alpha_stock + alpha_bond <= 1`
4. Therefore, on the discretized benchmark, `R_p` is uniformly bounded above
   and away from zero across:
   - all current states
   - all next-state quadrature nodes
   - all return quadrature nodes
   - all feasible portfolio weights
5. Since `R_p` is bounded and positive, the expectation
   `E[ xi_old(s')^{-gamma} * R_p^{1-gamma} ]` is finite whenever `xi_old` is
   bounded away from zero.

This gives:

- existence of a stationary policy by standard finite-state dynamic-programming
  arguments
- uniqueness of the optimizer at each state because the one-period problem is
  strictly concave in consumption and the feasible portfolio set is compact
- a well-behaved Coleman operator on `xi`

The standard finite-value condition is that the relevant moment
`E[ R_p^{1-gamma} ]` exists, which it does automatically here because the
benchmark uses:

- a finite quadrature rule
- strictly positive gross returns
- a compact constrained portfolio set

A stronger and more useful sufficient diagnostic for stability is the
impatience-style bound:

`beta * max_s E[ R_p(s, .)^{1-gamma} ] < 1`

or, more generally, that this object is comfortably below the region where the
Coleman update becomes nearly singular. With annual `beta < 1`, constrained
portfolios, and finite quadrature support, this benchmark should sit in the
stable region. The recommended numerical confirmation is:

- convergence from a cold start
- convergence from a warm start
- agreement of the final policy
- the homogeneity-in-wealth check described below

A useful analytical sanity check is the single-state special case. If
`xi_old = xi` is constant across states, the fixed point satisfies:

`xi_star = 1 - [ beta * E( R_p^{1-gamma} ) ]^{1/gamma}`

For `gamma = 3`, `beta = 0.96`, and a typical optimized portfolio with
`E(R_p^{1-gamma})` modestly below `1`, this implies a wealth MPC in the low
single-digit percentage range. So after convergence, if the average `C / W`
across interior wealth points is around `2%` to `5%`, that is economically
plausible. If it is `50%` or `0.1%`, something is likely wrong.

So the justification is:

- analytically, the benchmark collapses to a stationary `CRRA` Coleman problem
- numerically, the discretized operator is bounded, positive, and compactly
  controlled by the constrained portfolio set
- practically, the existing retirement kernel already implements exactly the
  required Euler and portfolio equations in this limit

---

## 5. Version-1 modeling simplification: keep a dummy z axis

Do **not** remove the `z` dimension structurally in the first implementation.

Reason:

- several parts of the current architecture assume a `z` axis exists
- the retirement operator already accepts a `z` dimension
- with no income and no mortality, the optimal policy should be identical across
  `z` slices

This gives a low-risk implementation:

- keep `pc.z_grid`
- keep policy shape `(n_z, N_state, n_w)` inside the stationary solve
- impose zero income, unit survival, and `b_bar = 0`
- check after the solve that all `z` slices are identical up to tolerance

This is the fastest trustworthy route.

If a later cleanup is wanted, the `z` axis can be removed after the benchmark is
working and validated.

The cost of this simplification is computational waste: each fixed-point
iteration repeats the same solve `n_z` times. That is acceptable for the first
benchmark figure, but if runtime becomes a bottleneck later, removing the `z`
loop from a forked retirement kernel is the natural version-2 optimization.

---

## 6. Fixed-point algorithm

### 6.1 Stationary policy equation

Let the stationary policy operator be `T`.

The benchmark policy solves:

`policy_star = T(policy_star)`

where `policy` consists of:

- consumption policy `C`
- stock-share policy `S`
- bond-share policy `B`

### 6.2 Initialization

Preferred warm start:

- if a lifecycle policy run already exists, initialize from the retirement-entry
  slice `C_mat[t = retire_age]`

This is usually much closer to the infinite-horizon fixed point than a cold
start, because the retirement-entry lifecycle agent is already solving nearly
the same problem with a long but finite horizon.

Fallback cold start:

- consume-all: `C0[z, s, w] = wealth_grid[w]`

Portfolio shares can be initialized implicitly by the first operator call.

Recommended second initialization for robustness:

- consume-all cold start

Optional additional start:

- terminal-age policy from [`solver.py:solve_terminal_age()`](../solver.py#L1103)

### 6.3 Iteration

At each iteration:

1. feed the current continuation policy `C_old` into the retirement-period
   kernel from inside the JIT outer loop
2. use:
   - `psi_vec = 1`
   - `pension_1d = 0`
   - `b_bar = 0`
3. obtain updated policies `C_new, S_new, B_new`
4. compute convergence metrics
5. either stop or continue

### 6.4 Convergence metric

Use a sup norm over policy differences:

- `max |C_new - C_old|`
- `max |S_new - S_old|`
- `max |B_new - B_old|`

Stop when the largest of these is below the policy tolerance.

Recommended starting tolerance:

- `1e-5` or `1e-6`
- `max_iter = 500` as the safe default, especially for cold starts

For a `CRRA`, zero-income benchmark, a better primary convergence object is the
consumption ratio:

`xi(s, w) = C(s, w) / W`

because the exact stationary solution should satisfy homogeneity in wealth:

- `c(W, s) = xi(s) * W`
- `alpha_stock(W, s) = alpha_stock(s)`
- `alpha_bond(W, s) = alpha_bond(s)`

So the preferred diagnostics are:

- sup norm of `xi_new - xi_old`
- sup norm of share changes

and only secondarily the absolute `C_new - C_old` difference.

Recommended stopping rule:

- stop when `max(xi_err, share_err) < tol`
- keep `policy_err` as a logged secondary diagnostic rather than the primary
  termination criterion

To avoid the bottom-grid kink dominating convergence metrics, evaluate the `xi`
metric on a trimmed wealth range that skips the lowest few wealth points.

### 6.5 Damping

If plain iteration is slow or oscillatory, use under-relaxation:

`policy_next = lambda * T(policy_old) + (1 - lambda) * policy_old`

Recommended initial damping parameter:

- `lambda = 0.5` to `0.8`

Plain iteration should be tried first. Damping should only be added if the
undamped sequence is too slow or unstable.

### 6.6 JIT-first implementation sketch

JIT compatibility is a requirement from the start. So the implementation target
should be a **two-layer design immediately**:

1. a thin Python wrapper
2. a pure numerical `@njit` fixed-point core

The Python wrapper should do only:

- input validation
- warm-start preparation
- cold-start array initialization
- allocation of work buffers and fixed-length history arrays
- first-call user messaging about JIT compile time
- packaging the returned scalars/arrays into a diagnostics dict

The fixed-point loop itself should live in the JIT core from day one.

This is primarily an architecture and correctness requirement, not necessarily a
large runtime speedup. On a big production grid the outer Python loop may be a
tiny fraction of runtime, but implementing the loop in a Numba-safe style from
the beginning avoids writing the logic twice.

#### 6.6.1 Smoke test first: verify the inner retirement kernel is callable

Before building the full stationary solver, compile a minimal smoke test that
calls:

- [`solver.py:_solve_retirement_step_quad_jit()`](../solver.py#L1875)

from inside another `@njit` function.

Purpose:

- verify the inner `parallel=True` kernel can be called from a JIT outer loop
- verify `solver_config` resolves cleanly under the outer JIT
- surface any Numba typing issue early, before implementing the full core

Minimal pattern:

```python
from numba import njit
from solver import _solve_retirement_step_quad_jit


@njit
def _smoke_call_inner_kernel(
    wealth_grid, savings_grid, z_grid, N_state,
    c_next_full, pension_1d, annuity_factors,
    state_grid, grids_0, grids_1, grids_2,
    state_bracket_shift, state_bracket_L_inv,
    v_nodes, v_weights, M_v_nodes, const_r, A_r,
    Phi_0_state, Phi_11,
    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
    gamma, psi_vec, beta, b_bar,
    constrained, solver_config,
    out_c, out_s, out_b,
):
    _solve_retirement_step_quad_jit(
        wealth_grid, savings_grid, z_grid, N_state,
        c_next_full, pension_1d,
        annuity_factors,
        state_grid, grids_0, grids_1, grids_2,
        state_bracket_shift, state_bracket_L_inv,
        v_nodes, v_weights, M_v_nodes, const_r, A_r,
        Phi_0_state, Phi_11,
        exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
        gamma, psi_vec, beta, b_bar,
        constrained, solver_config,
        out_c, out_s, out_b,
    )
    return 0
```

If this fails to compile, the next step is:

- verify `SolverConfig` is Numba-resolvable as passed today
- if not, flatten solver-config fields into explicit scalar arguments for the
  JIT core and the inner-kernel call

#### 6.6.2 Numba-safe coding rules for this solver

Several convenient NumPy/Python idioms should **not** appear inside the JIT
fixed-point core:

- Python lists like `policy_supnorm_history = []`
- dict construction
- f-strings and verbose logging
- `None`-driven control flow
- `wealth_grid[None, None, :]` broadcasting
- `np.ptp(..., axis=...)`
- `np.broadcast_to(...)` inside the core

Use these replacements:

- fixed-length arrays: `np.empty(max_iter)`
- scalar reductions written as loops
- convergence metrics computed in explicit loops
- cold-start initialization performed in the Python wrapper, not in the JIT core

#### 6.6.3 JIT-core architecture

Recommended functions:

- `run_infinite_horizon_solver(...)`
  - Python wrapper
- `_compute_ih_metrics_jit(...)`
  - computes:
    - `policy_err`
    - `xi_err`
    - `share_err`
- `_compute_ih_terminal_diagnostics_jit(...)` or wrapper-side postprocessing
  - computes after convergence:
    - cross-z differences
    - wealth-homogeneity spreads
- `_run_infinite_horizon_core_jit(...)`
  - runs the full fixed-point iteration
  - calls `_solve_retirement_step_quad_jit(...)` directly

The JIT core should call the inner JIT kernel directly, not the Python wrapper
[`solver.py:solve_retirement_step_quad()`](../solver.py#L2042).

#### 6.6.4 Full JIT-core signature

The JIT core must receive all arrays and scalars it needs explicitly, including
the benchmark constants and all output buffers. A complete signature is long but
stable:

```python
@njit
def _run_infinite_horizon_core_jit(
    wealth_grid,
    savings_grid,
    z_grid,
    N_state,
    annuity_factors,
    state_grid,
    grids_0,
    grids_1,
    grids_2,
    state_bracket_shift,
    state_bracket_L_inv,
    v_nodes,
    v_weights,
    M_v_nodes,
    const_r,
    A_r,
    Phi_0_state,
    Phi_11,
    exp_ret_bill,
    exp_ret_stock,
    exp_ret_bond,
    ret_weights,
    gamma,
    beta,
    psi_vec,
    pension_1d,
    b_bar,
    constrained,
    solver_config,
    C_old,
    S_old,
    B_old,
    out_c,
    out_s,
    out_b,
    policy_supnorm_history,
    xi_supnorm_history,
    tol,
    max_iter,
    damping,
    trim_wealth_points,
):
    ...
```

Notes:

- `psi_vec`, `pension_1d`, and `b_bar` are required benchmark inputs
- `out_c`, `out_s`, and `out_b` must be preallocated by the caller
- `C_old`, `S_old`, and `B_old` are persistent state buffers
- if `solver_config` does not compile cleanly, flatten its fields into explicit
  scalar arguments

#### 6.6.5 Buffer-rotation pattern

The JIT core should avoid per-iteration allocation. The clean pattern is:

1. keep persistent buffers:
   - `C_old`, `S_old`, `B_old`
2. keep work buffers:
   - `out_c`, `out_s`, `out_b`
3. call the inner retirement kernel with:
   - `c_next_full=C_old`
   - `policy_c=out_c`
   - `policy_alpha_s=out_s`
   - `policy_alpha_b=out_b`
4. compute convergence metrics using:
   - old buffers versus work buffers
5. update `C_old`, `S_old`, `B_old` **in place**

Reference pattern:

```python
@njit
def _apply_update_in_place(C_old, S_old, B_old, out_c, out_s, out_b, damping):
    n_z, N_state, n_w = C_old.shape
    if damping == 1.0:
        for iz in range(n_z):
            for is_ in range(N_state):
                for iw in range(n_w):
                    C_old[iz, is_, iw] = out_c[iz, is_, iw]
                    S_old[iz, is_, iw] = out_s[iz, is_, iw]
                    B_old[iz, is_, iw] = out_b[iz, is_, iw]
    else:
        one_minus = 1.0 - damping
        for iz in range(n_z):
            for is_ in range(N_state):
                for iw in range(n_w):
                    C_old[iz, is_, iw] = (
                        damping * out_c[iz, is_, iw] + one_minus * C_old[iz, is_, iw]
                    )
                    S_old[iz, is_, iw] = (
                        damping * out_s[iz, is_, iw] + one_minus * S_old[iz, is_, iw]
                    )
                    B_old[iz, is_, iw] = (
                        damping * out_b[iz, is_, iw] + one_minus * B_old[iz, is_, iw]
                    )
```

This avoids creating `C_new`, `S_new`, or `B_new` arrays inside the loop.

#### 6.6.6 Numba-safe convergence and diagnostic metrics

Compute all metrics in explicit loops.

Example helper:

```python
from numba import njit


@njit
def _compute_ih_metrics_jit(
    C_old, out_c, S_old, out_s, B_old, out_b,
    wealth_grid, trim_wealth_points,
):
    n_z, N_state, n_w = C_old.shape

    policy_err = 0.0
    xi_err = 0.0
    share_err = 0.0

    for iz in range(n_z):
        for is_ in range(N_state):
            for iw in range(n_w):
                dc = out_c[iz, is_, iw] - C_old[iz, is_, iw]
                if dc < 0.0:
                    dc = -dc
                if dc > policy_err:
                    policy_err = dc

                ds = out_s[iz, is_, iw] - S_old[iz, is_, iw]
                if ds < 0.0:
                    ds = -ds
                if ds > share_err:
                    share_err = ds
                if ds > policy_err:
                    policy_err = ds

                db = out_b[iz, is_, iw] - B_old[iz, is_, iw]
                if db < 0.0:
                    db = -db
                if db > share_err:
                    share_err = db
                if db > policy_err:
                    policy_err = db

            for iw in range(trim_wealth_points, n_w):
                w = wealth_grid[iw]
                vold = C_old[iz, is_, iw] / w
                vnew = out_c[iz, is_, iw] / w
                dxi = vnew - vold
                if dxi < 0.0:
                    dxi = -dxi
                if dxi > xi_err:
                    xi_err = dxi

    return policy_err, xi_err, share_err
```

For terminal diagnostics after convergence:

- compute cross-z differences either:
  - in a small terminal JIT helper using loops, or
  - in the Python wrapper using simple broadcast-style subtraction
- compute wealth-homogeneity spreads with per-state min/max loops, not
  `np.ptp(..., axis=...)`

#### 6.6.7 JIT-first core sketch

```python
@njit
def _run_infinite_horizon_core_jit(
    wealth_grid,
    savings_grid,
    z_grid,
    N_state,
    annuity_factors,
    state_grid,
    grids_0,
    grids_1,
    grids_2,
    state_bracket_shift,
    state_bracket_L_inv,
    v_nodes,
    v_weights,
    M_v_nodes,
    const_r,
    A_r,
    Phi_0_state,
    Phi_11,
    exp_ret_bill,
    exp_ret_stock,
    exp_ret_bond,
    ret_weights,
    gamma,
    beta,
    psi_vec,
    pension_1d,
    b_bar,
    constrained,
    solver_config,
    C_old,
    S_old,
    B_old,
    out_c,
    out_s,
    out_b,
    policy_supnorm_history,
    xi_supnorm_history,
    tol,
    max_iter,
    damping,
    trim_wealth_points,
):
    total_newton_failures = 0
    converged = False
    n_iter_done = 0

    for it in range(max_iter):
        diag_int, diag_float = _solve_retirement_step_quad_jit(
            wealth_grid, savings_grid, z_grid, N_state,
            C_old, pension_1d,
            annuity_factors,
            state_grid, grids_0, grids_1, grids_2,
            state_bracket_shift, state_bracket_L_inv,
            v_nodes, v_weights, M_v_nodes, const_r, A_r,
            Phi_0_state, Phi_11,
            exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
            gamma, psi_vec, beta, b_bar,
            constrained, solver_config,
            out_c, out_s, out_b,
        )

        total_newton_failures += diag_int[:, DI_NEWTON_FAIL].sum()

        policy_err, xi_err, share_err = _compute_ih_metrics_jit(
            C_old, out_c, S_old, out_s, B_old, out_b,
            wealth_grid, trim_wealth_points,
        )
        policy_supnorm_history[it] = policy_err
        xi_supnorm_history[it] = max(xi_err, share_err)

        _apply_update_in_place(C_old, S_old, B_old, out_c, out_s, out_b, damping)

        n_iter_done = it + 1
        if it > 0 and max(xi_err, share_err) < tol:
            converged = True
            break

    return converged, n_iter_done, total_newton_failures
```

This is the actual implementation target if JIT compatibility is required from
the outset.

#### 6.6.8 What stays in the Python wrapper

The Python wrapper should remain thin. It should do only:

- `solver_config` default handling
- normalization of scalars like `trim_wealth_points`
- default solver settings such as:
  - `tol = 1e-6`
  - `max_iter = 500`
- creation of:
  - `psi_vec = np.ones(n_z)`
  - `pension_1d = np.zeros(n_z)`
  - `b_bar = 0.0`
- warm-start loading from `C_mat[retire_t]`
- preferably also warm-start loading from `S_mat[retire_t]` and
  `B_mat[retire_t]` when available, so iteration 0 is not just measuring share
  distance from zeros
- cold-start initialization of `C_old`, `S_old`, `B_old`
- allocation of:
  - `out_c`, `out_s`, `out_b`
  - `policy_supnorm_history = np.empty(max_iter)`
  - `xi_supnorm_history = np.empty(max_iter)`
- one-time user note that first-call JIT compilation may take tens of seconds
- post-JIT packaging of diagnostics into a dict
- slice history arrays to `[:n_iter_done]` before packaging, because the tail of
  `np.empty(max_iter)` remains uninitialized
- optional wrapper-side terminal diagnostics if those are not returned by a
  dedicated JIT helper

The wrapper should **not** contain any fixed-point iteration logic.

#### 6.6.9 Practical expectations

Two pragmatic notes:

1. First-call compile time may be substantial.
   - With a large signature and a nested parallel kernel, the first JIT compile
     can take tens of seconds.
   - This should be documented in the wrapper's user-facing message so it does
     not look like the solve has hung.

2. JIT-ing the outer loop may not materially speed up the largest thesis grid.
   - The inner retirement kernel is already the dominant cost there.
   - So the JIT-first requirement should be viewed mainly as a robustness and
     architecture choice, not guaranteed wall-time improvement.

3. Confirm the inner-kernel return signature once before building around it.
   - The current plan assumes
     `_solve_retirement_step_quad_jit(...) -> (diag_int, diag_float)`.
   - That matches the current reading of [`solver.py`](../solver.py#L1875), but
     it should be verified once at implementation time in case the kernel is
     refactored later.

---

## 7. Required diagnostics

The benchmark should not be trusted without explicit diagnostics.

The solver should return at least:

- number of fixed-point iterations
- sup-norm history by iteration
- final residual
- whether damping was used
- aggregated Newton failure counts from the inner period solver
- max difference across `z` slices in `C`, `S`, and `B`
- a wealth-homogeneity diagnostic based on `C / W`

Suggested diagnostics dictionary fields:

- `n_iter`
- `converged`
- `policy_supnorm_history`
- `final_policy_supnorm`
- `xi_supnorm_history`
- `final_xi_supnorm`
- `used_damping`
- `damping_lambda`
- `total_newton_failures`
- `max_z_slice_diff_c`
- `max_z_slice_diff_s`
- `max_z_slice_diff_b`
- `stability_proxy`
- `max_xi_spread_across_w`
- `max_share_spread_across_w`

If `z` slices are not numerically identical up to a small floating-point
tolerance, the benchmark should be treated as failed or at least suspicious.

It is also worth recording a cheap impatience-style stability proxy, evaluated
at a reference state after the first or final policy update. This is not a
proof of contraction, but it is a useful early warning if a sensitivity run
pushes `beta * E[R_p^{1-gamma}]` too close to `1`.

---

## 8. Validation plan

This benchmark is only useful if it is numerically trustworthy.

Minimum validation steps:

### 8.1 Small-grid smoke test

Run on a small grid first, for example:

- `state_grid_sizes = (3, 3, 3)` or `(5, 5, 5)`
- small `n_w`, `n_s`

Check:

- convergence occurs
- no NaN or Inf
- no explosion in portfolio weights under the chosen constraint regime

### 8.2 Multiple initial guesses

Solve from at least two distinct initial consumption guesses.

Recommended pair:

- warm start from lifecycle retirement-entry policy
- cold consume-all start

Acceptance criterion:

- final policies match up to tight tolerance

### 8.3 Dummy-z invariance

Verify:

- `C[z0] == C[z1] == ...`
- same for `S` and `B`

up to tight numerical tolerance.

Recommended acceptance criterion:

- `max_z_slice_diff_c < max(1e-6, 10 * solver_config.tol)`
- `max_z_slice_diff_s < max(1e-6, 10 * solver_config.tol)`
- `max_z_slice_diff_b < max(1e-6, 10 * solver_config.tol)`

Bit-exact equality should not be expected because each `z` slice still runs its
own inner Newton solves and can accumulate tiny floating-point differences.

### 8.4 CRRA homogeneity in wealth

With `CRRA` utility and no income, the stationary solution should be homogeneous
in wealth.

Verify, for each state, that:

- `C(W, s) / W` is nearly constant in `W`
- `alpha_stock(W, s)` is nearly constant in `W`
- `alpha_bond(W, s)` is nearly constant in `W`

This is the strongest internal sanity check for the benchmark.

Recommended implementation:

- trim the bottom 5-10 wealth points
- compute max-minus-min spread over the remaining wealth points
- require those spreads to be small

### 8.5 Grid robustness

Compare the benchmark line under modest discretization changes:

- small vs medium wealth grid
- small vs medium state grid

Acceptance criterion:

- the evaluation-point allocation changes only modestly

### 8.6 Constraint robustness

If the lifecycle figure is constrained, the safest first benchmark is also
constrained.

If an unconstrained benchmark is wanted as a sensitivity check:

- verify it is not sitting on a numerical boundary
- verify the result is not driven by extreme leverage at bad states

For the first thesis-use figure, the constrained benchmark is easier to trust.

---

## 9. Figure integration

### 9.1 Comparison object

The lifecycle figure and the infinite-horizon benchmark must be evaluated at the
same point in the state space.

That means:

- same financial state index or same financial-state coordinates
- same wealth index or same wealth level

### 9.2 Recommended workflow

1. identify the reference point already used for the lifecycle policy figure
2. solve the infinite-horizon stationary policy
3. extract:
   - `alpha_stock_inf`
   - `alpha_bond_inf`
4. draw dotted horizontal lines across age on the lifecycle figure

If the lifecycle figure is built directly from policy arrays, the stationary
benchmark should also come directly from policy arrays rather than from
simulation.

### 9.3 Which z slice to use

In version 1, any `z` slice should be valid because the slices should be
identical.

Recommended practice:

- use `z = n_z // 2`
- also assert in code that the maximum cross-z discrepancy is below tolerance

### 9.4 Figure wording

Do **not** label the dotted line "CCV" in the figure.

Recommended wording:

- "stationary CRRA benchmark with the same VAR"
- "infinite-horizon CRRA benchmark"

This avoids implying Epstein-Zin preferences or an exact Campbell-Chan-Viceira
replication.

---

## 10. File plan

Recommended new files:

- `inf_horizon_solver.py`
- `tests/test_inf_horizon_solver.py`

Recommended responsibilities:

- `inf_horizon_solver.py`
  - stationary solver wrapper
  - `_smoke_call_inner_kernel(...)` compile smoke test
  - `_compute_ih_metrics_jit(...)`
  - `_run_infinite_horizon_core_jit(...)` numerical core
  - diagnostics construction
- `tests/test_inf_horizon_solver.py`
  - convergence on a small grid
  - no NaN/Inf
  - dummy-z invariance
  - CRRA homogeneity in wealth
  - same solution from multiple initial guesses

Optional:

- notebook or plotting script update for the dotted-line overlay

---

## 11. Suggested implementation order

1. Write and compile `_smoke_call_inner_kernel(...)` to verify the inner
   retirement kernel is callable from an outer `@njit` function.
   - If this fails, inspect `SolverConfig` first and flatten its fields into
     explicit scalar arguments if needed.
2. Build `_compute_ih_metrics_jit(...)` using only Numba-safe idioms.
3. Build `_run_infinite_horizon_core_jit(...)` with in-place buffer rotation.
4. Build the thin Python wrapper around the JIT core.
5. Run a small-grid smoke test with `psi = 1`, zero income, and `b_bar = 0`.
6. Add convergence diagnostics, dummy-z checks, and the wealth-homogeneity
   check.
7. Add a small automated test file.
8. Run the benchmark on the same discretization used for the lifecycle figure.
9. Add the dotted horizontal overlay to the policy plot.
10. Run a small robustness check before using the figure in the thesis.

---

## 12. Acceptance criteria

The benchmark is ready for thesis use when all of the following hold:

1. The stationary solver converges from multiple initial guesses.
2. Policy arrays contain no NaN or Inf.
3. The benchmark is invariant across dummy `z` slices up to the chosen floating-
   point tolerance.
4. The benchmark passes the CRRA homogeneity-in-wealth check away from the
   bottom of the wealth grid.
5. The evaluation-point allocation is stable to modest grid changes.
6. The dotted line is evaluated at the exact same state-space point as the
   lifecycle figure.
7. The chosen constraint regime is clearly documented in the figure notes.
8. The figure caption calls this a stationary or infinite-horizon `CRRA`
   benchmark, not "CCV."

---

## 13. Practical estimate

For a first trustworthy benchmark:

- stationary solver wrapper
- diagnostics
- one test file
- dotted-line figure overlay

Estimated effort:

- about 1 focused day for a working prototype
- about 2-3 days for a version with proper checks and enough confidence for
  thesis use

---

## 14. Main recommendation

The highest-value path is:

- implement a **separate stationary benchmark solver**
- **reuse the retirement operator**
- **keep the z axis as a dummy dimension**
- validate carefully
- compare at the **same state-space point** as the lifecycle plot

This gives a benchmark that is close to the current model economically, avoids a
large refactor, and is realistic to trust for a dotted-line comparison figure.

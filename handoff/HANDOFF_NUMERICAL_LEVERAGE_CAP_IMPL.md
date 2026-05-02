# HANDOFF — Implement numerical leverage cap in unconstrained Newton solvers

**Status:** open. Investigation already done; this is the implementation handoff.
**Owner-hand-off-to:** coding agent.
**Last updated:** 2026-05-02.

---

## 1. Purpose

Add a configurable box projection `(α_s, α_b) ∈ [α_min, α_max]^2` to the three
unconstrained Newton portfolio solvers in `solver.py`. The cap is a **numerical
guardrail**: the user will tune `α_min, α_max` per sweep so that the cap is
non-binding on the simulator-visited cells of the policy. The investigation
showed that at the current calibration (`γ=5`, `7×7×7` wide-support, log1p),
a cap of `±10` is comfortably non-binding (max simulated `|α|` ≈ 9.25), and the
user will narrow that during sweeps as needed.

The cap exists so that the solver cannot land at extreme leverage at
quadrature-pathological corners that the same-mode FOC happens to validate but
that finer quadrature flags as non-physical. The fix to that pathology is
quadrature refinement; the cap is a stopgap that makes the solver well-behaved
at solver-side probe points where the production quadrature rule has known gaps.

## 2. Hard rules

- **Do not modify the constrained-branch solvers.** They already handle
  bounding via `project_to_triangle` and the simplex constraint is the
  right thing for that branch.
- **Do not modify the FOC kernels** (`compute_foc_jac_*_quad`,
  `compute_terminal_portfolio_foc_jac`).
- **Do not modify `project_to_triangle`.**
- **Do not refactor or rename anything else.** Minimal surgical patch.
- **JIT-friendly throughout.** All changes go inside `@njit` functions and
  must be Numba-compilable, allocation-free, and `fastmath=True`-safe.

## 3. Files and call sites

### 3.1 `model.py` — `SolverConfig`

`SolverConfig` is a `NamedTuple` at [model.py:126](../model.py#L126). Add two
fields (with defaults wide enough to be non-binding for the current bundles):

```python
# --- Numerical leverage cap (unconstrained branch only) ---
alpha_min: float = -10.0   # lower bound on alpha_s and alpha_b in the unconstrained Newton
alpha_max: float = +10.0   # upper bound on alpha_s and alpha_b in the unconstrained Newton
```

Place these together as a clearly labeled block. They apply only to the
unconstrained branch; the constrained branch ignores them and continues to
use simplex projection.

### 3.2 `solver.py` — three unconstrained Newton functions

The three unconstrained Newton solvers are:

| Function | Defined at | Inner loop site |
|---|---|---|
| `solve_portfolio_unconstrained_terminal_njit` | [solver.py:1204](../solver.py#L1204) | line-search around L1264; damped step around L1284 |
| `solve_portfolio_unconstrained_retirement_quad` | [solver.py:1561](../solver.py#L1561) | line-search around L1644; damped step around L1673 |
| `solve_portfolio_unconstrained_working_quad` | [solver.py:1910](../solver.py#L1910) | line-search around L2009; damped step around L2040 |

The constrained-branch solvers `solve_portfolio_2d_retirement_quad` (ends
~L1551) and `solve_portfolio_2d_working_quad` (ends ~L1900) already terminate
each Newton step with `a_s, a_b = project_to_triangle(a_s + step_s, a_b + step_b)`.
Use that pattern as the structural template — but apply a box clamp instead
of a triangle projection, in the **unconstrained** functions only.

### 3.3 Step-driver call sites

Each unconstrained Newton function is called from the step driver
(`_solve_retirement_step_quad_jit`, `_solve_working_step_quad_jit`,
`solve_terminal_age`) at [solver.py:1331](../solver.py#L1331),
[solver.py:2143](../solver.py#L2143), [solver.py:2354](../solver.py#L2354).
Each call site already threads kwargs from `solver_config`. Add two more:
`alpha_min=sc.alpha_min, alpha_max=sc.alpha_max` (or
`alpha_min=solver_config.alpha_min, ...` at the terminal call site).

## 4. Implementation pattern

### 4.1 Function signature additions

Add two trailing keyword arguments to each of the three unconstrained Newton
functions, default-disabled:

```python
@njit(fastmath=True)
def solve_portfolio_unconstrained_retirement_quad(
        s_val, z_idx, i_s,
        ...,                     # existing args unchanged
        use_line_search=True, max_backtrack_iter=10,
        line_search_max_step=2.0,
        alpha_min=-1e30, alpha_max=+1e30):    # NEW
    ...
```

The sentinel defaults (`-1e30, +1e30`) make the cap effectively non-binding
when the function is called without the new kwargs, preserving any internal
direct callers (none currently, but keeps the patch defensively safe). The
production `SolverConfig` defaults of `(-10, +10)` will reach the inner
function via the step-driver call sites.

### 4.2 Branch-free clamp (JIT-friendly)

Use the standard scalar clamp inside each Newton function. Numba compiles
this to two `min/max` instructions; no allocations, no array operations,
fastmath-safe:

```python
# Clamp scalar a to [alpha_min, alpha_max]
if a_s < alpha_min:
    a_s = alpha_min
elif a_s > alpha_max:
    a_s = alpha_max
if a_b < alpha_min:
    a_b = alpha_min
elif a_b > alpha_max:
    a_b = alpha_max
```

You may equivalently write this as `a = max(alpha_min, min(alpha_max, a))`; both
are njit-safe. Prefer whichever matches surrounding style — currently the
solver uses explicit branches and that is fine.

### 4.3 Where to apply the clamp

There are two code paths in each Newton function: the line-search variant
(`use_line_search=True`) and the damped-step variant (`use_line_search=False`).

**Damped-step path (no line search).** Apply the clamp after the step update
and **before** re-evaluating the FOC. This mirrors the constrained pattern
exactly:

```python
slen = (step_s * step_s + step_b * step_b) ** 0.5
if slen > step_damp:
    cap = step_damp / slen
    step_s *= cap
    step_b *= cap
a_s += step_s
a_b += step_b
# NEW: clamp to numerical box
if a_s < alpha_min: a_s = alpha_min
elif a_s > alpha_max: a_s = alpha_max
if a_b < alpha_min: a_b = alpha_min
elif a_b > alpha_max: a_b = alpha_max
fs, fb, Jss, Jbb, Jsb, e_last = compute_foc_jac_*(a_s, a_b, ...)
err = (fs * fs + fb * fb) ** 0.5
```

**Line-search path.** Apply the clamp **inside the backtracking loop** to the
trial point, before the FOC is evaluated. This keeps the err comparison
between feasible-vs-feasible iterates, which is the correct line-search
invariant when the search space is bounded:

```python
for _bt in range(max_backtrack_iter):
    a_s_t = a_s + alpha * step_s
    a_b_t = a_b + alpha * step_b
    # NEW: clamp trial to numerical box
    if a_s_t < alpha_min: a_s_t = alpha_min
    elif a_s_t > alpha_max: a_s_t = alpha_max
    if a_b_t < alpha_min: a_b_t = alpha_min
    elif a_b_t > alpha_max: a_b_t = alpha_max
    fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = compute_foc_jac_*(a_s_t, a_b_t, ...)
    err_t = (fs_t * fs_t + fb_t * fb_t) ** 0.5
    if err_t < err:
        ...
        break
    alpha *= 0.5
```

**Apply both patches in all three functions:**

- `solve_portfolio_unconstrained_terminal_njit` — both paths.
- `solve_portfolio_unconstrained_retirement_quad` — both paths.
- `solve_portfolio_unconstrained_working_quad` — both paths.

### 4.4 Convergence behavior at the boundary

If the unconstrained Newton would naturally converge to a point inside
`[α_min, α_max]^2`, the clamp is a no-op and the iterate sequence is
unchanged. If the true unconstrained optimum lies outside the box, Newton
will press into the boundary, the clamp will hold the iterate at the
boundary, and the FOC residual will not reach `tol * scale`. The function
will exit with `EC_NEWTON_FAIL` after `max_iter` iterations.

Do **not** add explicit "boundary active set" detection or early-exit logic.
The user will tune `(α_min, α_max)` so the cap is non-binding in normal
operation; if the cap binds at any cell, that itself is the diagnostic
signal the user wants. The current solver tracks `total_newton_failures`
in diagnostics; cap-bound cells will appear there.

## 5. Tests

Add a new test file `tests/test_alpha_cap.py`. Required cases:

### 5.1 Non-binding cap is a no-op

With `alpha_min=-100.0, alpha_max=+100.0`, run a small unconstrained solve
and compare against the same solve with `alpha_min=-1e30, alpha_max=+1e30`
(effectively no cap). The two policy arrays should be **bit-identical**
(`np.array_equal`) because the clamp is the identity on every iterate when
no iterate ever reaches the box.

```python
def test_cap_noop_on_nonbinding_bounds():
    # build small problem, gamma=3 (well-conditioned)
    model, pc = _build_small_problem(constrained=False)
    solver_config_open = SolverConfig(alpha_min=-1e30, alpha_max=+1e30)
    solver_config_wide = SolverConfig(alpha_min=-100.0, alpha_max=+100.0)

    C1, S1, B1 = run_lifecycle_solver(model, pc, solver_config=solver_config_open, ...)
    C2, S2, B2 = run_lifecycle_solver(model, pc, solver_config=solver_config_wide, ...)

    assert np.array_equal(S1, S2)
    assert np.array_equal(B1, B2)
    assert np.array_equal(C1, C2)
```

### 5.2 Tight cap clips and reports failure

Pick a small unconstrained problem where the unconstrained optimum is
known to leverage above some α* (e.g., from a γ=2 or γ=3 small grid).
Run with `alpha_min=-0.5, alpha_max=+0.5` (deliberately tight). Verify:

- `np.max(np.abs(S)) <= 0.5 + 1e-12`
- `np.max(np.abs(B)) <= 0.5 + 1e-12`
- `np.min(S) >= -0.5 - 1e-12` (the cap is two-sided, including short positions)
- The diagnostic exit-code count for `EC_NEWTON_FAIL` increases relative to
  the open-cap baseline (cells that wanted to leverage past 0.5 should now
  fail).

### 5.3 Constrained branch is unaffected

Run a constrained-branch solve with `solver_config = SolverConfig()`
(default `alpha_min=-10, alpha_max=+10`). The result must be **bit-identical**
to a constrained-branch solve from before this patch (or equivalently,
identical to a constrained solve with `alpha_min=-1e30, alpha_max=+1e30`).
The test verifies that the new fields don't accidentally change the
constrained code path.

### 5.4 Both retirement and working ages exercised

Use a small problem with `start_age=64, retire_age=66, terminal_age=68` (or
similar) so that one working-age and at least one retirement-age period are
both solved. Apply tests 5.1 and 5.2 to that solve.

### 5.5 Existing tests still pass

After patching, run the existing test suite:

```bash
python -m pytest tests/test_inf_horizon_solver.py tests/test_partial_solve.py -q
```

Both must pass. They exercise the constrained and unconstrained code paths
in their existing form and will catch any regression in the kernel
signatures or step-driver call sites.

## 6. Validation steps

1. **Unit tests pass:** all of §5 must be green.
2. **Smoke solve:** run a small unconstrained solve at the production
   `7×7×7`, `kret(3,7,5)` discretization with `alpha_min=-10, alpha_max=+10`
   over a partial age window (e.g. `from_age=87`, ~13 ages). Compare to the
   pre-patch bundle: policy arrays should agree to within tol on the body of
   the distribution. `total_newton_failures` should remain 0 (the user has
   verified this is a robust property).
3. **Tight-cap smoke solve:** repeat the partial solve with
   `alpha_min=-4, alpha_max=+4`. Report `total_newton_failures` and the
   per-axis maxima of |S|, |B|. The user expects:
   - max |S| ≤ 4.0 + 1e-12
   - max |B| ≤ 4.0 + 1e-12
   - some non-zero `total_newton_failures`, concentrated at the structured
     probe corners flagged in `diagnostics_gridpoint_ee_log1p_pathB_nextfiner.md`.

## 7. Cache invalidation note

Numba caches compiled JIT functions in `__pycache__/*.nbc`. Modifying the
function body or signature of any `@njit` function invalidates its cache;
the first run after the patch will re-compile. This is expected and not a
regression. No manual cache cleanup is needed.

## 8. Investigation backing this design

For context (not required reading to implement, but useful to interpret
results):

- `diagnostics_gridpoint_ee_log1p_pathB_nextfiner.md` — original pathology
  diagnostic; 18.4% next_finer invalidity at structured probes.
- `handoff/HANDOFF_UNCONSTRAINED_LEVERAGE.md` — older γ=3 investigation;
  H4 fix (near-Merton init `(0.85, 0.44)`) was already applied to the
  current bundle, so the cap operates downstream of the major H4 fix.

The implementing agent does not need to read these to do the patch. They are
listed for review.

## 9. Out of scope

- Refining `n_state_quad_nodes` or `n_ret_nodes_1d`. Quadrature refinement
  is the principled fix for the pathology and is planned as a separate
  AWS run; it is not part of this patch.
- Changing the simulator, the FOC kernel, or `project_to_triangle`.
- Adding new diagnostic scripts. The cap-binding count will be visible
  through the existing `total_newton_failures` aggregate.
- Asymmetric per-component caps. The current cap is symmetric in both
  axes (`α_s ∈ [α_min, α_max]`, `α_b ∈ [α_min, α_max]`). If asymmetric
  caps are needed later (e.g., `α_s ∈ [-4, +6]`, `α_b ∈ [-6, +6]`), that
  would be a separate small change.

## 10. Deliverables

- `model.py`: `SolverConfig` gets `alpha_min`, `alpha_max` fields.
- `solver.py`: three unconstrained Newton functions get a clamp inside their
  damped-step path AND inside their line-search backtrack.
- Three step-driver call sites pass `alpha_min, alpha_max` from `solver_config`.
- `tests/test_alpha_cap.py`: tests 5.1 through 5.4.
- All existing tests in `tests/` continue to pass.
- A short status report (in the response, not a new file) confirming each
  validation step in §6.

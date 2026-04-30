# HANDOFF — Work-to-retirement transition bug (age 66 FOC uses wrong income)

> **Read this entire document before writing any code.** The diagnosis is well-evidenced and the fix looks small, but the FOC inner loop is a hot path for the entire solver and carries hidden invariants. Pause and review the plan against the existing code (and ideally raise any concerns back to the user) **before** editing.

## 1. The issue

### 1.1 Symptom

Unconstrained smoke config (`state_grid=(5,5,5)`, `state_n_stds=2.0`, `n_z=9`, `n_eps_nodes=3`, `n_eta_nodes=3`, `n_ret_nodes_1d=(3,5,3)`, `n_state_quad_nodes=2`) produces the following diagnostic at `age 66`:

```
  66  WORK   261.9s  100.0%     66    0.896    0.440   -0.336  0.359   ...     0.9     33
```

- **66 `EC_NEWTON_FAIL`** at age 66 only. Every other working age (22–65) reports zero failures.
- `max_iter = 33`, far below `max_iter_unconstrained = 8000`. **Exits via line-search stagnation, not budget exhaustion.**
- All retirement ages (67–98) and the terminal age (99) converge with zero failures.

### 1.2 Root cause

Age 66 is the only year where the *working-age* solver is called with `t+1 = retire_age = 67`. The working-age FOC at [solver.py:619-641](solver.py#L619-L641) computes next-period gross income as

```python
y_gross_next = det_z_eta * exp_eps[i_e]                    # = exp(f(67) + ρz + η + ε)
income_next  = scalar_disposable_income(y_gross_next)      # working-age tax schedule
x_next       = w_inv + income_next                          # cash-on-hand at age 67
```

`log_det_next = log_det_profile[t+1]` is the deterministic age-earnings polynomial `f(age)` — defined for *every* age, including retirement ages. So at age 66 the FOC is computing the agent's age-67 income as if they will *still be working* at age 67, paying working-age tax on it.

The model's actual retirement income at age 67 is **pension**, computed from frozen z via the AIME → PIA formula in `compute_pension_after_tax` ([model.py:349](model.py#L349)). Pension is **z-only**, capped (because AIME is capped at 2.5), and bounded — for high z it saturates around 0.628 model units.

### 1.3 Mechanism (why this kills Newton)

At high-z + high-η + high-ε quadrature combinations the buggy income explodes (factor ~100×–500× vs pension). Worked example at the top z grid point:

```
z_idx=8  (z = +5.61)
buggy disposable income at age 67  (η=max, ε=max)  =  307.77 model units
buggy disposable income at age 67  (η=mid, ε=max)  =  218.81
buggy disposable income at age 67  (η=min, ε=max)  =  147.17
correct pension at age 67                          =    0.628    (capped, η/ε independent)
```

Wealth grid maximum is **200 model units**, so the buggy income alone overshoots the grid. `find_bracket` clamps to the last segment and the cubic-then-linear interpolation in `_interp_z_wealth` extrapolates with the slope of the final piece. That extrapolation creates a kink in the FOC surface as `(α_s, α_b)` move. Newton finds a Jacobian step direction; line search backtracks 10 halvings; every halving still leaves ‖F‖ above the previous iteration; the solver exits with `EC_NEWTON_FAIL` at iteration ≤ 33.

### 1.4 Why only age 66, and why both modes are affected

- **Why only age 66.** Ages 22–65 use `log_det_profile[t+1]` for `t+1 ∈ [23, 66]` — all working ages, polynomial is correct. Age 66 is the unique year where `t+1` is retirement.
- **Both modes affected; symptom differs.** Both unconstrained and constrained working-age dispatch call the same `compute_foc_jac_working_quad`. Constrained Newton projects to the simplex and won't run into the kink, so it does not raise an error — but it converges to a *wrong* policy, which then propagates backwards through ages 22–65. **Constrained mode is silently buggy.** A successful constrained run with "0 failures" is not evidence of correctness here.

### 1.5 Code references

- Buggy FOC (working age, used at age 66): [solver.py:488-678](solver.py#L488-L678), specifically the inner-loop income computation at lines 636–641.
- Dispatch boundary: [solver.py:2363-2399](solver.py#L2363-L2399). The `if age >= retire_age:` branch picks the retirement solver from age 67 onward; ages 22–66 go to the working solver. There is no special case for `age == retire_age - 1`.
- Correct retirement FOC structure (no eps, no eta, uses pension): [solver.py:340-465](solver.py#L340-L465), specifically `x_next = w_inv + pension_next_scalar` at line 437.
- Pension formula and table: [model.py:349](model.py#L349) (`compute_pension_after_tax`), `pc.pension_after_tax` shape `(n_age, n_z)` precomputed in [precompute.py](precompute.py).

## 2. Goals of the fix

In priority order:

1. **Correctness.** At the work-retirement transition (`t = retire_age - 1`), the working FOC must use `pension(z_next)` as next-period income, **not** `exp(f(t+1) + ρz + η + ε)`. The η integration over the final persistent shock is correct and must be preserved. The ε integration is meaningless at retirement (retirees have no transitory shocks) and should not bias the result.

2. **No regression for working-to-working transitions.** Ages 22–65 must produce **bit-identical** policies before and after the fix. The fix is conditional on `t = retire_age - 1` only; everywhere else the existing code path runs.

3. **JIT-friendly, no measurable slowdown.** The working FOC is a hot inner loop (~12.99M calls per smoke run). Any added branch or argument must compile cleanly under `@njit(fastmath=True)` and not trigger a recompile cascade or pessimistic codegen. Per-call cost should not increase by more than ~1% on working-to-working calls.

4. **Both unconstrained and constrained modes.** The fix lives in the FOC computation, which both modes call. Verify both.

5. **Minimal surface area.** No new files, no API changes outside the FOC and its caller. No effect on saved-run metadata, simulation, diagnostics, or notebooks.

## 3. Dangers — read carefully

> **Pause here.** Each of these is a way the fix can silently corrupt the model. Make sure you understand each one before designing the implementation.

### D1. JIT-unfriendly arguments (silent slowdown / recompile)

The working FOC is `@njit(fastmath=True)`. New arguments must be types numba can specialize on:
- Scalars (`float`, `int`, `bool`) — fine.
- 1D `np.ndarray` of `float64` — fine.
- `Optional[np.ndarray]` (None or array) — **dangerous**. Numba's union-typing can break specialization. Pass a dummy `np.zeros(n_z)` array when the flag is `False` rather than `None`.
- Tuples or lists — avoid.

If you change the signature of an existing `@njit` function, every caller's compiled code becomes stale. Run a smoke compile after the change to confirm.

### D2. Eps loop removal (silent bias)

`eps_weights` sum to 1 by construction (Judd quadrature). If the fix keeps the eps loop but replaces `income_next` with a value that doesn't depend on `i_e`, the weights still sum to 1 — *no bias, just wasted work*. **If the fix instead drops the eps loop entirely, you must re-fold the eps_weights factor (= 1) into the weight expression somewhere, or you'll silently rescale the entire FOC by `1 / sum(eps_weights)`.**

The safest path: **keep the eps loop intact at the transition year, just substitute `income_next`**. Wastes ~3× work in one inner loop at one age. Tiny global cost; structurally hardest to break.

### D3. Z-interpolation of pension (numerical accuracy)

`pension(z)` is a continuous, monotone, but non-smooth function (capped at AIME = 2.5, then PIA bend points). On the discrete `z_grid` the precomputed table is `pc.pension_after_tax[age, iz]`. The working FOC at age 66 has continuous `z_next = ρ z + η`. Two ways to evaluate `pension(z_next)`:

- **(i) Linear interpolation of the precomputed table in z.** Matches how `_interp_z_wealth` already interpolates `c_next_full` in z. Same accuracy character. Cheap. **Recommended.**
- **(ii) Re-call `compute_pension_after_tax` for each `z_next`.** "Exact" but invokes vectorized NumPy machinery from inside an njit loop — likely won't compile, and even if you reimplement the formula in njit, the AIME cap and PIA bend points introduce branch-heavy code in the hot loop.

Pick (i). At the cost of small (sub-percent) interpolation error in pension between grid nodes — already the same character of error as the existing z-interpolation of c_next.

### D4. Forgetting the dispatch (fix without effect)

The `compute_foc_jac_working_quad` is wrapped by the working-age step-solvers (`solve_portfolio_2d_working_quad` and `solve_portfolio_unconstrained_working_quad`), which in turn are called by `solve_working_age_step_quad` from `run_lifecycle_solver`. The new flag/array must be threaded through **every** layer. Missing one layer means the inner FOC sees the new arguments only at one age and uses the dummy at all other ages — or worse, the new arguments never get set and the bug stays.

Trace the call stack manually before editing: there are **at least four** functions to update.

### D5. Constrained mode silent regression

If you only test the unconstrained run ("did the 66 failures go away?"), you may miss a bug in the constrained path. The constrained FOC uses the same `compute_foc_jac_working_quad`. **You must verify the constrained policy at age 66 changes between the buggy and fixed versions** — that's how you confirm the constrained path actually exercises the new code. If constrained policies are byte-identical before and after, the dispatch didn't reach the constrained branch.

### D6. Wealth-grid extrapolation is still possible

The fix bounds `x_next` for the *transition* year. But for years 22–65, the existing code can in principle still push `x_next` off the wealth_grid at extreme z + extreme η + extreme ε combinations, depending on `f(age)`. **This handoff is not a license to expand grid bounds or refactor the wealth-grid logic.** Stay laser-focused on the work-retirement boundary.

### D7. z_next bracketing edge cases

`z_next = ρ z + η` can land outside `z_grid` for extreme combinations. The existing code at [solver.py:626-629](solver.py#L626-L629) already clamps `iz_lo` to `[0, n_z - 2]` and `frac_z` to `[0, 1]`. Reuse this clamping for the pension interpolation; do **not** reinvent it.

### D8. Hidden constants in the existing FOC

Look at the constants used in the existing inner loop (e.g. `prob_skip`, `min_consumption`, `min_wealth_inv`). These act as numerical floors. The new branch must use the same floors so behavior is consistent across the two paths.

## 4. Proposed plan

> **Review this plan against [solver.py:485-678](solver.py#L485-L678) before implementing.** If anything reads as inconsistent with what you find in the code, raise it back to the user instead of patching around it.

### 4.1 Strategy

Runtime branch inside `compute_foc_jac_working_quad`. Two new arguments:

- `use_pension_next: bool` — set at the dispatch level. `True` only at `t = retire_age - 1`.
- `pension_next_by_z: np.ndarray` of shape `(n_z,)` and `dtype=float64`. When `use_pension_next=True`, this is `pc.pension_after_tax[t+1, :]`. When `False`, pass a dummy `np.zeros(n_z, dtype=float64)` (never read). **Use a dummy array, not `None`** — see D1.

Inside the eps inner loop, replace the income computation with a branch:

```python
if use_pension_next:
    # Linear interp of pension_next_by_z at continuous z_next, reusing existing iz_lo / frac_z
    income_next = (1.0 - frac_z) * pension_next_by_z[iz_lo] + frac_z * pension_next_by_z[iz_lo + 1]
else:
    y_gross_next = det_z_eta * exp_eps[i_e]
    income_next  = scalar_disposable_income(y_gross_next)
```

Keep the eps loop in place (D2 — safer). At the transition year `income_next` becomes constant across `i_e`, so the eps_weights inside the loop simply re-sum to 1 and the integration is mathematically clean.

### 4.2 Files / functions to update

Trace the argument plumbing top-down. Likely list (verify against the actual code):

1. **`compute_foc_jac_working_quad`** at [solver.py:488](solver.py#L488).  Add the two new parameters; insert the branch in the eps inner loop.
2. **`solve_portfolio_2d_working_quad`** at [solver.py:1429](solver.py#L1429) (constrained working-age Newton).  Forward the two new parameters through to every internal call to `compute_foc_jac_working_quad`.
3. **`solve_portfolio_unconstrained_working_quad`** at [solver.py:1626](solver.py#L1626).  Same — forward the parameters to every internal `compute_foc_jac_working_quad` call. There are several Newton iterates and corner/edge probes inside; **don't miss any**.
4. **`solve_working_age_step_quad`** (the per-age batch solver). Forward the parameters down from the dispatcher.
5. **`run_lifecycle_solver`** at [solver.py:2387](solver.py#L2387) (working-age dispatch). Set `use_pension_next = (age == retire_age - 1)`. Set `pension_next_by_z = pension_table[t + 1, :]` when true, else the precomputed dummy zeros. Pass into `solve_working_age_step_quad`.

Also: precompute the dummy `np.zeros(n_z)` once at the top of `run_lifecycle_solver` — don't allocate inside the loop.

### 4.3 Approach — staged, with checkpoints

1. **Read everything first.** Open all five functions. Map the call graph. Identify every call site of `compute_foc_jac_working_quad`. There are several inside the unconstrained working-age solver (corner probes, edge probes, main Newton step, line search). All need updating.
2. **Stub the new parameters first.** Add the two parameters to the FOC signature with the branch implemented but *not* used (set `use_pension_next=False` everywhere). Re-run the existing tests and a smoke solve. Confirm bit-identical results — no regression at all ages. **Do not proceed until this passes.**
3. **Wire up the dispatch.** In `run_lifecycle_solver`, set `use_pension_next=True` at the boundary year. Re-run unconstrained smoke. Confirm: zero failures at age 66, ages 22–65 still bit-identical, age 66 policy changes (it should — that's the fix doing its job).
4. **Constrained mode check.** Re-run constrained at the smoke config. Compare age-66 policy before vs after. **Must differ.** If identical, the constrained path didn't reach the new code — go back and check the dispatch.
5. **Profile.** Compare wall-time of working-to-working ages before vs after. Allowable: < 1% slowdown. If higher, investigate (D1).

### 4.4 What not to do

- Do not add a third njit FOC function. The duplication is not worth it for a single transition year.
- Do not pass `Optional[np.ndarray]` (D1).
- Do not drop the eps loop (D2).
- Do not call `compute_pension_after_tax` from inside the FOC (D3).
- Do not modify the retirement solver, terminal solver, simulation, diagnostics, saved-run metadata, or notebooks. The fix is internal to the working-age FOC and its dispatch.
- Do not change `log_det_profile`. It's used in the working-age FOC as the deterministic part of *current-age* income; the bug is only in *next-period* income at the boundary.

## 5. Validation tests

> **All five must pass before declaring done.** If any are skipped, the fix is not validated.

### V1 — No regression at ages 22–65 (regression)

Unconstrained smoke run with the buggy code (current `main`) and with the fix at the same config. **C_mat[0:44] (ages 22–65), S_mat, B_mat must be bit-identical.** Use `np.array_equal` (not `np.allclose`). Any diff at ages 22–65 means a working-to-working dispatch accidentally turned on `use_pension_next`.

### V2 — Age 66 unconstrained: zero failures

Run the unconstrained smoke config (notebook config). Diagnostic must report **0 EC_NEWTON_FAIL at age 66** (and at every other age). `Newton%` should remain ~100%.

### V3 — Pension lookup bit-identical at z_grid points

Standalone test: at every `iz` in `[0, n_z-1]`, the new linear-interpolation expression at `frac_z = 0` must equal `pension_table[t+1, iz]` exactly:

```python
for iz in range(n_z):
    interp = pension_next_by_z[iz]   # frac_z = 0, iz_lo = iz
    assert interp == pc.pension_after_tax[t + 1, iz]
```

Catches off-by-one in indexing.

### V4 — Constrained mode actually changed

Compare constrained smoke runs before vs after at the same config. **C_mat at age 66 (and propagated, ages 22–65) must differ at high-z slices** (`iz ∈ {6, 7, 8}`). If unchanged, the constrained path didn't pick up the fix.

For the same-z slices at low z (`iz ∈ {0, 1, 2}`), the policies should differ only slightly (the pension and the buggy disposable income are similar in magnitude at low z — see D1 in the issue write-up).

### V5 — JIT compile + per-call cost

After the change, time:
- **Solver startup time** (first call triggers JIT compile). Should not increase by more than a few seconds.
- **Per-age wall time** for ages 22–65 (working-to-working, same code path as before). Should not increase by more than ~1% per age. Compare against pre-fix run with the same config and same machine state.

If solver startup balloons or per-age time jumps materially, suspect D1 (numba specialization issue) and inspect the new function signature.

## 6. After the fix

Once V1–V5 pass:

- Add a one-line entry to `contextfiles/RETURNS.md` (or `LABOUR.md`, whichever section is more accurate) noting the work-retirement transition is now correctly handled.
- Consider deleting this handoff file or moving it to `archive/`.
- The unconstrained smoke run should now report `0 FAIL` across every age, and the post-solve diagnostic's "Failures" line should read `0 (0.000%)`.

## 7. If you get stuck

If at any point the validation tests show something unexpected — V1 fails, V2 still has failures, V4 shows no change in constrained, or the JIT cost balloons — **stop and report back to the user with the specific test that failed and what you observed**. Do not patch around it. The bug is structural and the fix needs to be exactly right.

# Model review — Solver internals (2026-05-09)

**Branch:** `jax-rewrite`. **Scope:** numerical correctness of the 2D Newton
solver, line search, fori_loop masking, and convergence handling
(`lifecycle/solver.py`). **Read-only.** Complementary to
`NEWTON_FAILURE_STRUCTURE_2026-05-08`, `COMPUTE_EFFICIENCY_REVIEW_2026-05-08`,
`NEWTON_HISTOGRAM_AUDIT_2026-05-07`, and `HANDOFF_NEWTON_FORI_LOOP_MASK`.

---

## 1. Newton 2D inner loop (`_newton_fori`, `_newton_while`)

**What I found.** Construction is **analytic**, not finite-difference. The FOC
kernels (`terminal_foc_jac_ccv:776`, `retirement_foc_jac_ccv:952`,
`working_foc_jac_ccv:1059`) return `(fs, fb, Jss, Jbb, Jsb, e)` in one pass:
the Jacobian is the closed-form derivative of the FOC residuals w.r.t.
`(α_s, α_b)` built from `dRp/dα_*` (`_ccv_log_return_and_grad:747`) and the
CRRA marginal-utility chain rule. Symmetric (`Jsb` shared on the off-diagonal).

Inversion uses the explicit 2×2 inverse with `det = Jss*Jbb - Jsb²`
(`solver.py:498`, `682`); guarded by `is_singular = |det| < singular_det`
(default `1e-15`). On singular cells the step falls back to a normalized-
residual direction (see RED FLAG below).

Step length cap (`solver.py:512-515`, `696-699`): **per-iter**, not cumulative
— the raw Newton step is rescaled if `||step|| > line_search_max_step` (= 2.0
canonical). Verified: `slen` is recomputed each iter from this iter's
`(step_s, step_b)`, then `cap = min(1, max_step / slen)`.

Convergence test: `err < tol * scale` where `scale = max(|e0|, 1e-30)` and
`e0 = foc_fn(0,0)[5]` (i.e. expected marginal utility at zero portfolio
allocations) — `_egm_scan_cell:1213-1214`. As `NEWTON_FAILURE_STRUCTURE`
established, this scaling becomes ill-defined at the high-savings tail
where `e0 ~ c^{-γ}` collapses to fp64-noise.

`exit_code` (`EC_INTERIOR=1` / `EC_NEWTON_FAIL=2`) is set on the converged
flag, then aggregated to `age_newton_fail[t]` (`solver.py:2695`, `2787`) and
into `total_newton_failures` in diagnostics (`3023`). Wiring is intact post-
pivot; the new real-yields kernels do not change this path.

Cold-start init: backward-age warm start gathers a **single scalar per
(z, i_s) cell** at `w_ref_idx = n_w // 2` (`solver.py:2122`, `2243`),
then `_egm_scan_cell` reuses that scalar for every savings point. Cold across
the savings axis, warm across age and wealth (mid-slice).

**Verdict.** UNCLEAR — see RED FLAG #1 (singular fallback sign) and
UNDOCUMENTED #1 (mid-wealth warm-start choice).

---

## 2. Backtracking line search (`_backtracking_fori`, `solver.py:585`)

**What I found.** Schedule: pure halving `α_{k+1} = α_k * 0.5` from `α=1`
(`solver.py:619`). Acceptance criterion: **simple monotone decrease**
`err_t < err_old` (`626`) — Armijo / Wolfe sufficient-decrease conditions
are **not** used, only `||f||₂` reduction. Simple but cheap.

The "first-improving alpha wins" semantics are correctly preserved under
fori_loop masking via `improved_now = !found AND err_t < err_old`
(`626`) and `found |= improved_now` (`639`); once `found=True`, subsequent
iters' fields are held constant (the mask uses the current best at each
field). The `n_used` counter increments only while `is_active = !found`
(`640`), so it reports the number of halvings evaluated **before** the
accepting one (0 if α=1 worked). Sticky-acceptance state is correct.

`bt_init` (`602-615`): pre-seeded with the α=1 result iff `full_improves`,
else with the old state. Inside the bt body, the *first* halving `α=0.5`
is evaluated (`new_alpha = alpha * 0.5` from initial `alpha=1.0`). No
double-count of the α=1 step; `full_improves` is sticky and `improved_now`
inside the body is gated by `!found`, so if α=1 already won, halvings
keep computing but never overwrite the best-so-far.

**Caveat (compute-only, already in COMPUTE_EFFICIENCY_REVIEW):** under
fori_loop, all `max_backtrack_iter` halvings run unconditionally; this is a
performance issue, not a correctness one.

**Verdict.** CLEAR.

---

## 3. fori_loop architecture & masking

**What I found.** `_newton_fori:678` masks via
`is_active = !(converged OR ls_failed)`. Once a cell is in the converged or
ls_failed set, every output field (a_s, a_b, fs, fb, Jacobians, e, err,
counters, flags) is held constant via `jnp.where(is_active, new_*, *)`
(`722-734`).

Critical question: is the inner Jacobian computation safe on masked cells?
**Yes.** When the cell holds (a_s, a_b) constant, `foc_fn(a_s, a_b)` returns
the same `(fs, fb, Jss, Jbb, Jsb, e)` as at convergence, so `det`,
`is_singular`, and the next-step direction are recomputed from valid state.
The mask zeroes the *update*, not the read. No garbage propagation.

Determinism gotcha: the n_iter_used counter and n_backtrack counter increment
only on active cells (`731-732`), so the histogram correctly reports
"iters until convergence" rather than "iters total". This was the fix in
`NEWTON_HISTOGRAM_AUDIT_2026-05-07`.

**Verdict.** CLEAR.

---

## 4. Determinism & reproducibility

**What I found.** No PRNGKey, no `jax.random` calls anywhere in
`lifecycle/solver.py` (verified via grep). Initial guesses are scalars from
`SolverConfig` or gathers from previous-age policy arrays — fully
deterministic given the model + precompute + config.

GPU FMA / reduction-order non-determinism is the standard JAX caveat — XLA
can reorder reductions across runs on the same GPU type. This affects
bit-identity but not algorithmic correctness, and is out of solver scope.

f32 gather layering (`gather_precision="f32"`,
`_cast_for_gather:376`, `_interp_c_and_mpc_at_cell:898-941`): the cast-cast
roundtrip is single-direction within the kernel — gather + bracket + interp
in fp32, cast back to fp64 *before* `jnp.maximum(c, min_consumption)` and
`jnp.clip(mpc, 0, 1)`, before all CRRA / FOC arithmetic. No ambiguous
fp32-floored quantities flow into Newton state. Documented sub-1e-5
relative drift vs f64 is consistent with this layering.

**Verdict.** CLEAR.

---

## 5. f32 gather path & wealth-grid bracketing

**What I found.** `_interp_c_and_mpc_at_cell:907` and
`retirement_foc_jac_ccv` `per_kv_kr:995` both bracket via
`jnp.searchsorted(wealth_grid_g, x, side="right") - 1` on the **fp32-cast**
wealth grid. The fp32 spacing safety check
(`validate_wealth_grid:135-140`, `wealth_grid.py`) gates against
`min_rel_diff32 ≤ 8 * eps_f32` and `n_nonpositive_diff32 > 0`.

**RED FLAG #2.** That validator is invoked **only** in `precompute.py:257`
when `wealth_grid_path is not None` (custom file-loaded grid). The canonical
log1p path (`legacy_log1p_wealth_grid`) at `precompute.py:247-251` skips
the f32-spacing gate entirely. With the canonical
`(n_wealth=180, wealth_min=0.05, wealth_max=750.0)` and `gather_precision=f32`
(both on by default in `_canonical.py`), there is no runtime check that
the log1p grid remains strictly increasing under fp32 cast or that
`min_rel_diff32` exceeds tolerance. At the dense low-w end of a log1p grid,
the spacing `Δw / w ≈ const`, and the absolute `Δw` near `w=0.05` is small
but the *relative* spacing should be safe; still, this is an **uncovered
invariant** the design relied on per `FP32_NEWTON_PROBE_2026-05-07`.
Quick fix: invoke `validate_wealth_grid` on the log1p grid too in
`precompute.py` before returning it.

`x == y` fp32-comparisons: none in the hot loop (verified by reading
`_interp_c_and_mpc_at_cell` and `per_kv_kr`). The only equality-style
comparison is `slen > 0.0` in the step-cap (`solver.py:513`, `697`), which
is in fp64 always.

**Verdict.** RED FLAG (uncovered fp32-spacing invariant on the canonical
log1p grid; minor likely benign).

---

## 6. Edge cases

**Tiny savings.** `_egm_scan_cell:1227-1230`: when `s_val ≤ tiny_savings`
(`1e-6`), the cell is overridden post-Newton: `c=min_consumption`,
`α_s=init_a_s`, `α_b=init_a_b`. Note: the Newton **still runs** at
`s_val=0`-ish — the result is just discarded. This is wasteful but not
buggy. The α fallback uses the cell's **warm-start scalar** (or the cold
init), which is sensible.

**Singular Jacobian fallback (RED FLAG #1).** Lines 502-503 (`_newton_while`)
and 686-687 (`_newton_fori`):
```
step_s_grad = grad_step_size * fs / (err + grad_denom_eps)
step_b_grad = grad_step_size * fb / (err + grad_denom_eps)
```
This is `+η · f / ||f||` — pointing in the **direction of f**, not against
it. For a scalar root-finding analogue (f > 0, f' > 0), descent toward zero
requires a negative step; this gives a positive one. For sum-of-squares
F = ½||f||², gradient descent is `-η · J^T f`, not `+η · f / ||f||`.
The line search will reject steps that don't reduce `||f||`, so the bug
manifests as a wasted iteration rather than divergence — but on cells
where the Jacobian is genuinely singular and the local geometry favors
this direction, the solver makes **anti-progress** before the line search
backs off. Mitigated by:
- `singular_det = 1e-15` is very strict; activation is rare.
- The line search caps damage to ≤ `err_old`.
- Empirical failure rates do not show a singular-driven mode (per
  `NEWTON_FAILURE_STRUCTURE` §6: dominant cause is tol-unreachable).

But the math is wrong; either negate (`-grad_step_size * fs / ...`) or
better, use a Levenberg-Marquardt damping `(J + λI) s = -f` with `λ` set
when det is small.

**NaN propagation.** Each cell solves independently via `vmap`; a NaN in
one cell does not leak across the batch axis. Within a cell, NaN-producing
ops (e.g. negative `c` ⇒ `c**(-γ)` complex) are guarded:
`min_consumption=1e-10` floor in interp helpers (`solver.py:943`, `1013`),
`euler_inv_floor=1e-20` floor on `β·V_dot` before inversion
(`_egm_scan_cell:1224`).

**`min_consumption` placement.** Applied to `c_next` inside the interp
helper (`solver.py:943`, `1013`) — *after* the multilinear gather but
*before* the CRRA `c**(-γ)` (which happens in the FOC kernel via
`mu_alive = c_at_xn ** (-gamma)`). Correct placement: floors c before any
exponentiation that would amplify near-zero inputs.

**Newton iter count.** `n_used` increments only when `is_active`
(`solver.py:731`). Converged cells stop incrementing immediately. Verified
correct.

**Verdict.** RED FLAG (singular fallback sign).

---

## TL;DR

| Area | Verdict |
|---|---|
| 1. Newton 2D | UNCLEAR (warm-start scalar choice undocumented; convergence test inherits known tol-scale issue) |
| 2. Line search | CLEAR |
| 3. fori_loop masking | CLEAR |
| 4. Determinism | CLEAR |
| 5. f32 gather path | RED FLAG (canonical log1p grid skips fp32-spacing validator) |
| 6. Edge cases | RED FLAG (singular-Jacobian fallback uses wrong-sign gradient direction) |

**Single most important RED FLAG:** the singular-Jacobian fallback
(`solver.py:502-503` / `686-687`) builds the gradient step as
`+grad_step_size · f / ||f||`, which moves **with** the residual rather
than against it. For sum-of-squares minimization the descent direction is
`-J^T f`, and at minimum the sign should be negated. The line search
rejects net-divergent steps so the bug is contained to wasted iterations
and is rarely activated (`singular_det=1e-15`) — but the math is wrong.

**Reproduction path:** construct a 2×2 FOC fixture with
|det(J)| < 1e-15 (e.g. by setting `Jss=1, Jbb=1, Jsb=1` plus a non-zero
residual `(fs, fb)=(1, 1)`); the gradient step proposes
`(0.05/√2, 0.05/√2)` instead of the correct descent direction
`(−0.05/√2, −0.05/√2)`. Line search rejects, halves α, eventually
gives up at `max_backtrack_iter` and flags the cell `EC_NEWTON_FAIL`
without ever attempting descent. Patch: negate both `step_s_grad` and
`step_b_grad`. Optional better fix: Levenberg-Marquardt damping.

**Secondary RED FLAG:** the canonical log1p wealth grid is not validated
against fp32 spacing tolerances (`precompute.py:247-251` skips the
`validate_wealth_grid` call that the custom-grid branch runs at
`precompute.py:257`). With `gather_precision="f32"` the default in
`_canonical.py`, this invariant is uncovered by code; the design assumed
the log1p grid is "obviously safe" but no runtime gate enforces it.

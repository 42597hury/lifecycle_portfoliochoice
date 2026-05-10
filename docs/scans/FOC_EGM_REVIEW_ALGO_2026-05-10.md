# FOC + EGM Review — Algorithmic Layer (2026-05-10)

**Branch:** `jax-rewrite`. **Reviewer angle:** algorithmic / numerical-method
correctness. **Read-only.** Independent pass; the parallel reviewer covers the
math/equation-derivation layer.

The scope is the orchestration glue around the FOC: per-savings EGM scan, 2D
Newton + line search, lift-to-wealth-grid, per-savings backward-age warm-start
(Variant B), failed-cell neighbor-seed fixup, terminal → reversed-range age
loop, kernel dispatch (terminal / boundary / retirement / working), inf-horizon
fixed-point loop, multi-precision toggle, cell-axis chunking, multi-device
pmap, diagnostics output, EC-code semantics.

Each of §2-§11 follows the same template: what the algorithm should do, what
the code does, where it is correct, where it is fragile, and where I disagree
with the existing audits.

---

## §1 Scope and angle

I deliberately do not re-derive the Bellman or FOC equations — that is the
companion reviewer's territory. I take as given the analytic kernels in
[terminal_foc_jac_ccv](lifecycle/solver.py#L850),
[retirement_foc_jac_ccv](lifecycle/solver.py#L1030), and
[working_foc_jac_ccv](lifecycle/solver.py#L1140), and ask whether the
**orchestration** of those kernels — the iteration order, seeding strategy,
boundary handling, exit-code propagation, multi-precision boundary, and
multi-device dispatch — collectively constitute a CORRECT EGM-based DP solver
in the sense that:

1. It converges to the true optimum **at cells where Newton converges**.
2. It makes **principled choices** at the EGM lower boundary, the failed-cell
   high-W tail, the work→retirement structural break, and the terminal age.
3. It **faithfully implements backward induction** on the finite-horizon
   Bellman recursion (and is consistent with that for the inf-horizon fixed
   point where it borrows the same kernel).
4. Its **engineering shells** (chunking, pmap, fp32 cast boundary,
   diagnostics) are mathematically transparent — i.e. they don't change the
   answer relative to the reference vmap-only fp64 path.

The two complementary previous audits I lean on are
[MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md](docs/scans/MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md)
(numerical Newton internals — pre-singular-fix-sign-commit) and
[BACKTRACK_ITER_AUDIT_2026-05-09.md](docs/scans/BACKTRACK_ITER_AUDIT_2026-05-09.md)
(line-search counter semantics, validated PASS). I confirm or refute their
claims independently below — most are intact, one (singular-Jacobian fallback
sign) has been fixed in the working tree, and a few new findings appear that
prior audits did not surface.

---

## §2 EGM scan over savings grid

**Where:** [_egm_scan_cell, solver.py:1261-1346](lifecycle/solver.py#L1261).

### §2.1 The pattern

Conceptually, EGM is: for each candidate end-of-period savings `s` from a
fixed exogenous savings grid, solve the 2D portfolio FOC for
`(α_s*, α_b*)`; recover `c_opt` from the inverted Euler
`c = (β · V̇)^{-1/γ}` where `V̇ = E[mu_comb · R_p]`; record the implied
beginning-of-period wealth `x_egm = c_opt + s`. After the scan, lift the
endogenous `(x_egm, c_egm, α_s_egm, α_b_egm)` cloud back to the fixed
exogenous wealth grid via 1D linear interpolation.

The code follows that pattern faithfully:

- The savings sweep is a `vmap` over `s_grid`
  ([solver.py:1332](lifecycle/solver.py#L1332)), so all `n_savings` Newton
  problems run in parallel rather than sequentially. This is the standard
  EGM idiom under vectorisation; it is correct because each savings point is
  an *independent* root-finding problem with no carry-over state.
- Inside `per_savings_point`
  ([solver.py:1292-1330](lifecycle/solver.py#L1292)) the order is
  exactly: build a per-savings FOC closure, scale by
  `inv_foc_scale = 1 / max(|e0|, 1e-30)` (where `e0 = foc_fn(0,0)[5]` is
  the V̇-style scalar at zero portfolio weights — see §2.4 below), Newton
  solve, then `c_opt = max((β · V_dot)^{-1/γ}, min_consumption)`, then
  `x_out = c_out + s_val`.
- The α from each Newton solve is also exported to `a_s_egm` /
  `a_b_egm`, so the orchestrator can roll them forward as Variant B
  warm-starts (§4) and so the lift step can record an α policy at every
  wealth grid point.

This iteration order — *fix s, solve interior FOC for α, recover c via
Euler, then back out W* — is the standard EGM mapping for problems with both
a portfolio and a consumption choice, and is consistent with Carroll (2006)
generalised by Druedahl & Jorgensen (2017) for multi-dimensional choice.
**Algorithm is correct.**

### §2.2 The s=0 anchor (`egm_anchor`)

The scan output has `n_savings + 1` entries, with index 0 being the
artificial anchor `(x_egm[0], c_egm[0], α_s_egm[0], α_b_egm[0]) =
(egm_anchor=1e-10, egm_anchor=1e-10, 0, 0)` —
[solver.py:1336-1344](lifecycle/solver.py#L1336).

Why the anchor exists: without it, the lift step's `jnp.interp` extrapolates
left of `x_egm[1]` by using the slope between `x_egm[1]` and `x_egm[2]`,
which produces nonsensical `c < 0` or `c > W` at `wealth_grid[0]`. The
anchor pins `(x, c) ≈ (0, 0)` so the linear interp from `(0, 0)` to
`(x_egm[1], c_egm[1])` at least respects `c → 0 as W → 0`.

The anchor is **not** a true borrowing-constrained corner — see §2.5 — but
it is a sensible left-boundary sentinel. The orchestrator pads the anchor
with `n_iters_egm = 0`, `n_backtrack_egm = 0`, and `exit_code_egm =
EC_INTERIOR` (not EC_NEWTON_FAIL), then immediately strips the anchor at
[solver.py:1465-1469, 1537-1541, 1619-1623](lifecycle/solver.py#L1465)
before passing the per-savings α-grid forward to the next age's Variant B
warm-start. Stamping the anchor with EC_INTERIOR is correct because no
Newton solve happened there — counting it as a failure would inflate
`age_newton_fail` and pollute the histogram.

**Algorithm is correct, with one caveat I record under §11 (the anchor's
`α=(0,0)` will leak into the per-savings α buffer if the buffer is ever
sliced from index 0 instead of index 1; this is a structural-aliasing risk
but I verified all current call sites strip with `[1:]`).**

### §2.3 The lift to wealth grid (`_lift_to_wealth_grid`)

**Where:** [solver.py:1349-1416](lifecycle/solver.py#L1349).

The lift does three things:

1. `argsort(x_egm)` — **necessary** because `x_egm = s + c_opt(s)` is not
   guaranteed to be monotone in `s` even though `s` is. (CRRA + portfolio
   feedback can produce a non-monotone `c(s)` in pathological regions; see
   FINDING ALGO-3 below.) After sort, `jnp.interp` reads the EGM cloud as a
   function of `x_sorted`.
2. `jnp.interp(wealth_grid, x_sorted, c_sorted)` etc. — linear
   interpolation of `c, α_s, α_b` separately.
3. **Path-B constrained-corner clamp**
   ([solver.py:1393-1415](lifecycle/solver.py#L1393)): for every
   `wealth_grid[i]` strictly below `W_min_real := x_sorted[argmax(
   c_sorted > 2 * min_consumption)]`, overwrite the interpolated
   `(c, α_s, α_b)` with the borrowing-constrained corner `(W, 0, 0)`.
   This isolates the "real" interior FOC solves (where `c_opt` is on the
   order of consumption units, far above `min_consumption=1e-10`) from the
   anchor and the tiny-savings fallback (both of which set `c =
   min_consumption`). Without the clamp, plain `jnp.interp` would linearly
   bridge from the anchor at `(1e-10, 1e-10)` to `(x_egm[k], c_egm[k])`
   for the smallest converged `k`, producing the meaningless `c < W` and
   `α ≠ 0` artefacts the previous BELLMAN_FOC review documented.

**Probe (§12.A):** I verified Path-B clamp behaviour on a synthetic EGM
cloud (`x_egm = [1e-10, 1e-7, 0.5, 1.0, 5.0, 30.0]`, `c_egm = [1e-10,
1e-10, 0.46, ...]`). Wealth grid points `[0.05, 0.13, 0.3]` (all below
`W_min_real = 0.5`) all returned `c = W, α_s = α_b = 0` exactly; points
at and above `0.5` went through normal `jnp.interp`. Bit-identical to the
post-clamp expected behaviour.

The Path-B clamp is the right construction for an unconstrained-interior
EGM solver where the true KKT corner is approximated rather than solved.
**Algorithm is correct (with two observations recorded as findings):**

- ALGO-2 (medium): the threshold `c > 2 * min_consumption` is tight against
  the anchor sentinel `c = min_consumption`, but it does not distinguish
  the **tiny-savings fallback** from the **real interior solve** if a real
  solve happens to produce `c_opt < 2 * min_consumption`. With γ=5 and
  realistic states this is fp-impossible (`c_opt` is bounded below by the
  Euler inversion `(β · V_dot)^{-1/γ}` and that quantity is on the order
  of 0.01-100 in model units), so the threshold is safe in practice. But
  if ever a low-V̇ pathological cell were to drop `c_opt` below `2e-10`, the
  clamp would silently overwrite that cell's interior policy with `(W, 0,
  0)`. A safer threshold would gate on `s > tiny_savings` directly rather
  than on `c > 2 * min_consumption`.
- ALGO-3 (medium, see §13): if `x_egm(s)` is non-monotone at some s, the
  `argsort + jnp.interp` chain silently linearly-interpolates across the
  fold. Standard JAX `jnp.interp` does not handle multi-valued inverses;
  the result is a piecewise-linear path between the *sorted* (x, c) pairs,
  which can produce an MPC > 1 across the fold. With CCV log-wealth
  dynamics and unconstrained portfolio choice this rarely fires in
  practice (verified by the §3 PASS sanity in
  [POLICY_REVIEW_2026-05-09.md](docs/scans/POLICY_REVIEW_2026-05-09.md)),
  but there is no detection or warning on the path.

### §2.4 FOC normalisation

The per-savings Newton sees `foc_fn = raw_foc / max(|e0|, 1e-30)` —
[solver.py:1294-1311](lifecycle/solver.py#L1294). This is a per-savings
constant scaling that (a) keeps the determinant `det = Jss·Jbb - Jsb²`
on a O(1) scale across all savings points, and (b) makes the convergence
test `err < tol * scale` (with `scale = 1.0` after division)
wealth-invariant. The Newton step `J⁻¹ f` is invariant to this kind of
constant scaling (numerator and denominator both scale by `inv_foc_scale`),
and so are the line-search comparisons `err_t < err_old` (both halves
scale identically). **Mathematically transparent.**

The scaling does NOT touch `V_dot / e` (the Euler RHS used to recover
`c_opt`) — see [solver.py:1310](lifecycle/solver.py#L1310). That is
correct: `c_opt = (β · V_dot)^{-1/γ}` is the actual Euler inversion and
must use the unscaled V̇.

### §2.5 Tiny-savings branch

[solver.py:1325-1328](lifecycle/solver.py#L1325): when `s_val ≤
tiny_savings = 1e-6`, the per-savings cell overrides post-Newton with
`c = min_consumption`, `α_s = init_a_s_v`, `α_b = init_a_b_v`. The
Newton **still runs** at `s_val ≤ 1e-6` (its result is just discarded);
this is wasteful but algorithmically harmless.

The tiny-savings cell's α equals the Variant-B warm-start scalar (which
is generally the previous age's converged α at the same cell) — this is
the right semantics if you ever want to interpret the policy at that cell,
because there is no "true" portfolio at `s = 0` and any value is conventional.

But there is a subtlety with the EC stamping: the tiny-savings branch
does **not** override the exit code — whatever Newton returned (whether
EC_INTERIOR or EC_NEWTON_FAIL) is preserved. So if Newton happens to fail
at the s=1e-7 cell (very rare), the tiny-savings cell is recorded as
EC_NEWTON_FAIL, and the failed-cell fixup (§4) will then attempt to
neighbor-seed from a converged-below cell — but there is no
converged-below cell (the anchor at index 0 is already stripped before
the fixup sees the array). The fallback path in `_fixup_failed_cells`
correctly handles this with `no_neighbor → cold scalar`. So the corner
case is benign, but the EC stamping for tiny-savings is **inconsistent**
with the orchestrator's intent (it should be EC_TINY_SAVINGS=3, not
whatever Newton produced). FINDING ALGO-1 (low) — see §13.

---

## §3 2D Newton with backtracking line search

### §3.1 fori vs while paths

**Where:** [_newton_fori, solver.py:691-814](lifecycle/solver.py#L691);
[_newton_while, solver.py:476-625](lifecycle/solver.py#L476).

Two Python-trace-time-selected modes via `use_fori_newton`:

- `True` → `lax.fori_loop` runs `max_iter` iters unconditionally, with
  `is_active = !(converged ∨ ls_failed)` masking the writes. Every iter
  pays one full FOC eval + up to `max_backtrack_iter` line-search FOC
  evals on every cell, regardless of convergence — but converged cells'
  state is held constant by the mask. GPU-friendly because there is no
  warp divergence.
- `False` → `lax.while_loop` with real data-dependent termination. The
  body runs only while `cond_fn(state) = !done ∧ k < max_iter`; once a
  cell converges, the loop exits. CPU-friendly.

The masking under fori is critical for correctness, not just performance:
if writes were not masked, a converged cell whose `det` happens to fall
below `singular_det` on a later iter would jitter into the
gradient-descent fallback and away from the converged α. The mask
prevents this — `is_active` is the AND of `not converged` and `not
ls_failed`, so a converged cell never moves again.

**Probe (§12.B):** I ran a synthetic linear FOC (`f = (a_s - 0.5, a_b -
0.5)`, J reported as scaled identity) under both modes:

| Case | (init) | reported J | Expected | fori (`a_s`, `ec`, `ni`, `nb`) | while (`a_s`, `ec`, `ni`, `nb`) | parity |
|---|---|---|---|---|---|---|
| Warm at root | (0.5,0.5) | 1 | no work | 0.5, 1, 0, 0 | 0.5, 1, 0, 0 | ✅ |
| Cold, perfect H | (0.7,0.7) | 1 | full step | 0.5, 1, 1, 0 | 0.5, 1, 1, 0 | ✅ |
| Cold, 2× overshoot | (0.7,0.7) | 0.5 | 1 halving | 0.5, 1, 1, 1 | 0.5, 1, 1, 1 | ✅ |
| Cold, 4× overshoot | (0.7,0.7) | 0.25 | 2 halvings | 0.5, 1, 1, 2 | 0.5, 1, 1, 2 | ✅ |
| Extreme overshoot | (0.7,0.7) | 1e-5 | LS fails | 0.4997, 2, 6, 36 | 0.4997, 2, 6, 36 | ✅ |

fori vs while bit-identical on every output, including the per-cell
`n_iter` and `n_backtrack` counters. This independently confirms BACKTRACK
audit Q4 PASS and adds the saturated-extreme-overshoot row.

### §3.2 Convergence test

`err < tol * scale` where `scale` is passed in as `1.0` (post per-savings
normalisation). Expressed as `err < tol = 1e-7` after normalisation. This is
a **norm test**, not a residual-relative test — i.e. `‖f‖₂` is compared
against an absolute tolerance after scaling.

The previous audit ([MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09](docs/scans/MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md)
§1) flagged this as inheriting a "tol-scale issue" at high-savings cells
where `e0` collapses to fp64 noise. I confirm this **algorithmically**: at
high savings, `e0 = E[mu(c) · R_p]` collapses because `c_next` saturates
(the wealth grid hits its top) and consumption becomes very large under
the Euler inversion. `inv_foc_scale = 1 / max(|e0|, 1e-30)` blows up,
making `foc_fn` numerically tiny — the Newton residual divided by a
near-zero scale can still test as "converged" or "not converged"
arbitrarily. The high-W tail's 100% pin documented in
[POLICY_REVIEW_2026-05-09 §3](docs/scans/POLICY_REVIEW_2026-05-09.md) is
the empirical signature of this. **CARRIED OVER as FINDING ALGO-4
(critical-but-known).**

### §3.3 Singular-Jacobian fallback

**Where:** [solver.py:526-528 (while)](lifecycle/solver.py#L526),
[solver.py:743-745 (fori)](lifecycle/solver.py#L743). Current code:

```python
step_s_grad = -grad_step_size * fs / grad_norm
step_b_grad = -grad_step_size * fb / grad_norm
```

The leading **minus sign** is load-bearing: the standard ad-hoc fallback
for `Φ = ½‖f‖²` is `-η · J^T f / ‖J^T f‖`, but when `J` is near-singular
the `J^T f` direction degenerates. The implemented `-η · f / ‖f‖` is a
descent direction for the residual norm in the trivial-Jacobian limit
(if `J ≈ I` then `J^T f / ‖J^T f‖ ≈ f / ‖f‖`). The sign is correct.

The previous review ([MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09 §6 RED
FLAG #1](docs/scans/MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md)) flagged
this code as having a wrong `+` sign — that audit predates commit
`1b2ebb1` ("solver: fix singular-Jacobian fallback sign") which I verified
is now in the working tree. The post-fix sign is correct.

**Probe (§12.C):** Constructed a synthetic singular FOC `f = (a_s - 0.5,
a_b - 0.5)` with `J = [[1, 1], [1, 1]]` (det = 0). Initialised at
`(0.7, 0.7)`. Expected step at `grad_step_size = 0.05`: `-0.05 · 0.2 /
0.2828 = -0.03536`. Observed first-iter `a_s = 0.7 - 0.03536 = 0.66464`
in both fori and while. **The sign is the corrected sign — moves toward
the root, not away.** The line search would never reject this step (it
strictly improves `‖f‖`), so the fallback now actively contributes to
convergence rather than wasting iterations.

### §3.4 Line search

The backtracking line search is a pure halving schedule (`α_{k+1} = 0.5
α_k`) starting from `α = 1`, with simple monotone-decrease acceptance
(`err_t < err_old`). No Armijo / Wolfe sufficient-decrease conditions —
just `‖f‖` reduction. Cheap, robust enough for the smooth CRRA + CCV
landscape, and the
[BACKTRACK_ITER_AUDIT_2026-05-09.md](docs/scans/BACKTRACK_ITER_AUDIT_2026-05-09.md)
already validated all nine semantic claims around its counter. I cross-
checked the counter implementations side-by-side at
[_backtracking_fori, solver.py:628-688](lifecycle/solver.py#L628) and
[bt_body in _newton_while, solver.py:570-590](lifecycle/solver.py#L570) —
the "first improving α wins" semantics is preserved under fori via
`improved_now = ¬found ∧ err_t < err_old`, and the `found` flag is
sticky (`found |= improved_now`).

The line search does NOT support a maximum step length other than the
per-iter `line_search_max_step` cap on the *raw Newton step* (before
halving). Per [solver.py:537-540](lifecycle/solver.py#L537), the cap is
`min(1, max_step / ‖step‖)` — applied **before** the line search starts.
This is correct: the cap bounds the trust region, the line search then
shrinks within that capped step. Combined with the bt-iter-counter being
"halvings until found", the line search is well-specified.

### §3.5 Exit codes

`EC_INTERIOR = 1` if converged, `EC_NEWTON_FAIL = 2` if not — set at
[solver.py:624 (while)](lifecycle/solver.py#L624) and
[solver.py:813 (fori)](lifecycle/solver.py#L813). The fori path uses the
sticky `converged` flag rather than `err < tol*scale` at the final state
(important: a cell that converged then drifted slightly under masking
should still report converged; the sticky flag guarantees this).

`EC_TINY_SAVINGS = 3` is **declared** at
[solver.py:49](lifecycle/solver.py#L49) but **never set** anywhere in the
solver. The tiny-savings branch in `_egm_scan_cell` does not stamp this
code — see ALGO-1 in §13.

---

## §4 Per-savings backward-age warm-start (Variant B) + failed-cell fixup

### §4.1 Variant B mechanism

**Where:** orchestrator
[solver.py:3037-3119](lifecycle/solver.py#L3037).

The rolling buffer `as_grid_prev`, `ab_grid_prev` (shape `(n_z, N_state,
n_savings)`) holds the previous (older) age's converged α at every
`(z, state, savings)` cell. At age `t`, the per-cell driver
[_solve_*_at_cell](lifecycle/solver.py#L1492) gathers
`init_a_s_arr[z_idx, i_s, :]` and uses that vector as the per-savings
Newton seed. The per-savings vmap then sees a `(n_savings,)` cold/warm
vector matched to its own `s_grid` axis — so every savings point gets a
**locally optimal** warm-start (the corresponding savings point at the
older age's converged solution).

This is strictly better than Variant A (the older mid-wealth scalar that
was broadcast across all savings points): Variant A used **one** Newton
seed for **all** `n_savings` points at a cell, leaving edge savings
points (very small or very large `s`) cold. Variant B uses **n_savings**
seeds, each tuned for its savings point. The
[SOLVER_EFFICIENCY_30PCT_2026-05-09.md §2](docs/scans/SOLVER_EFFICIENCY_30PCT_2026-05-09.md)
audit estimated 50-70% wall savings; the Lambda runs in
[POLICY_REVIEW_2026-05-09.md §4](docs/scans/POLICY_REVIEW_2026-05-09.md)
confirmed `mi=10` (10 Newton iters max) gives bit-near-identical converged
policies to `mi=100` (`max_rel ≤ 1.4e-4` at converged interior cells).

The buffer rolls forward in two clean assignments at
[solver.py:3112-3113](lifecycle/solver.py#L3112): `as_grid_prev = as_grid_t;
ab_grid_prev = ab_grid_t`. Reassignment releases the previous age's array
(JAX/NumPy ref-counted), so peak buffer memory is one `(n_z, N_state,
n_savings)` pair regardless of `n_age`. **Memory accounting is correct.**

The buffer is z-broadcast at the terminal age
([solver.py:2990-2995](lifecycle/solver.py#L2990)) since the terminal
policy is z-invariant (bequest only). The broadcast uses
`jnp.broadcast_to`, which stays on device — no host round-trip.

The cold-start path (when the toggle `use_backward_age_warm_start = False`
or the buffer is None at the first age after a checkpoint resume) uses
constant `(n_z, N_state, n_savings)` arrays seeded with `init_alpha_s` /
`init_alpha_b` ([solver.py:3037-3039](lifecycle/solver.py#L3037)). Same
shape contract; same kernel trace. **Algorithm is correct.**

### §4.2 The failed-cell neighbor-seed fixup (uncommitted)

**Where:** [_fixup_failed_cells, solver.py:2711-2757](lifecycle/solver.py#L2711);
called at [terminal solver.py:2984-2987](lifecycle/solver.py#L2984), at
non-terminal [solver.py:3104-3107](lifecycle/solver.py#L3104), and at
inf-horizon [inf_horizon_solver.py:662-666](lifecycle/inf_horizon_solver.py#L662).

The mechanism: under Variant B, when Newton fails at cell `(z, state,
s_idx)`, the stored α is *literally* the cold scalar `(init_alpha_s,
init_alpha_b)` (line search exhausted before any halving was accepted, so
α never moved from init). Without intervention, age `t-1` then reseeds
the same Newton problem from cold init at the same cell, fails again, and
the cascade compounds across ages. The
[NEWTON_FAILURE_ANALYSIS_2026-05-09.md](docs/scans/NEWTON_FAILURE_ANALYSIS_2026-05-09.md)
documented 22% per-cell failure rate in production runs, with 67% of
failures concentrated in the top 15 wealth indices — exactly the cascade
signature.

`_fixup_failed_cells` replaces each failed cell's α with the
**nearest-converged-below** neighbor's α at the same `(z, state)` slice,
using `lax.cummax` along the savings axis with `-1` fill at failed cells.
For each failed `(..., s_idx)`, the cumulative max reads the largest
`j ≤ s_idx` whose `ec[..., j] = EC_INTERIOR`; for converged cells the
result is `s_idx` itself (so the `take_along_axis` is a no-op). Where no
converged neighbor exists below (entire prefix failed), it falls back to
the cold scalar `(init_alpha_s, init_alpha_b)`.

**Probe (§12.D):** I exercised five edge cases against the helper:

1. Trailing failures `[1, 1, 1, 2, 2]` with α `[0.5, 0.6, 0.7, 0.85, 0.85]`
   → expected `[0.5, 0.6, 0.7, 0.7, 0.7]`. ✅
2. All converged → no-op. ✅
3. All failed → all cells take the cold scalar. ✅
4. Leading failures with later converged `[2, 2, 1, 2]` with α `[0.85,
   0.85, 0.7, 0.85]` → expected `[0.85, 0.85, 0.7, 0.7]` (idx 0,1 take
   cold scalar = 0.85; idx 3 takes neighbor = 0.7). ✅
5. Terminal-shape `(N_state, n_savings)` with no leading z dim → ✅.

All five match. The implementation is correct under all the edge cases the
handoff design called out.

### §4.3 Wiring verification

The fixup is wired correctly into:

- **Terminal age** ([solver.py:2984-2995](lifecycle/solver.py#L2984)):
  applied to the terminal `(N_state, n_savings)` α-grid *before*
  z-broadcast. Operates on the right shape; broadcast happens after. ✅
- **Non-terminal ages** ([solver.py:3104-3113](lifecycle/solver.py#L3104)):
  applied to `(n_z, N_state, n_savings)` α-grids before the
  `as_grid_prev = as_grid_t` reassignment that rolls the buffer forward.
  ✅
- **Inf-horizon iter loop**
  ([inf_horizon_solver.py:662-670](lifecycle/inf_horizon_solver.py#L662)):
  applied between the kernel call and the warm-start buffer reassignment
  for the next iter. The handoff explicitly anticipates iterative
  "diffusion" of the neighbor's α through the failure region across iter
  k → k+1; this is the same logic as the lifecycle version, just iterated
  in the time-of-iteration dimension. ✅

The fixup is **NOT** applied to the wealth-grid policy outputs `C, S, B`
— only to the warm-start α-grid buffer. This is correct: the wealth-grid
policies are what the simulator reads, and they were derived from
`_lift_to_wealth_grid` operating on the EGM cloud; touching those would
require re-lifting. The fixup's job is to break the cascade through the
warm-start, not to retroactively edit policies. **Architecturally
correct.** I confirm the wealth-grid C/S/B at converged cells will be
bit-identical with the toggle on or off (§12.E in the existing test
[test_failure_cell_neighbor_seed.py](tests/test_failure_cell_neighbor_seed.py)
already covers this with `low_wealth_indices_match_closely_across_flag`).

### §4.4 One subtle interaction

The fixup operates on the per-savings α-grid that comes *out of*
`_egm_scan_cell` — i.e. on the post-scan, pre-lift α arrays. Note that
the saving-axis values stored there correspond to the **endogenous wealth**
`x_egm = s + c_opt(s)`, not to the wealth-grid points the simulator
later reads. So when the next age's Newton at savings index `s_idx`
gathers `init_a_s_arr[z, state, s_idx]`, the seed's "ground truth" was
computed at a wealth `x_egm[s_idx + 1]` from the older age — but Newton
will now use it as a seed for solving the FOC at savings `s_grid[s_idx]`
*in this age*, where the post-Euler `x_egm` will be different.

This is fine for two reasons: (a) Newton is a local search and any
in-distribution seed gets it close to the root, and (b) the savings axis
is the same `s_grid` across ages, so cell `s_idx` always means the same
end-of-period savings. The α at the old age's `s_idx` is a much better
init than a cold scalar regardless of wealth-grid alignment. **Conceptually
sound.**

---

## §5 Backward-induction orchestration

### §5.1 Terminal → reversed range, kernel dispatch

**Where:** [run_lifecycle_solver, solver.py:2964-3119](lifecycle/solver.py#L2964).

The control flow is:

1. Terminal age (`age = terminal_age`):
   `terminal_kernel()` returns z-invariant `(c_T, s_T, b_T)`. Broadcast
   across z with `jnp.broadcast_to` (on device).
2. Backward loop `for t in reversed(range(n_age - 1))`:
   - Compute `age = ages[t]`.
   - Pull `c_next_jnp = C_list[t+1]` (already on device from the previous
     iteration — no host round-trip).
   - Branch on age:
     - `age >= retire_age` → `retirement_kernel(...)`.
     - `age == retire_age - 1` → `boundary_kernel(...)` (use_pension_next=True).
     - `age < retire_age - 1` → `working_kernel(...)`.

The dispatch is correct: the boundary case fires exactly at the age
*immediately before* retirement, where the agent still has eta/eps shocks
to integrate over but the income next period is the pension at bracketed
`z_next`, not labour income. The boundary kernel (`use_pension_next=True`)
reuses the working FOC trace but constructs the income table differently
([solver.py:2493-2497](lifecycle/solver.py#L2493) — pension at bracketed
`z_next` broadcast across the eps axis, since at age 67 there is no
transitory shock to the pension benefit). This is the right structural
break.

The `c_next_jnp = C_list[t+1]` read is critical for performance and
correctness — `C_list` holds device-resident jnp arrays, so the policy
flows from age to age without ever touching host memory. The orchestrator
materialises to NumPy only at checkpoint boundaries and at the final
return. I verified the materialisation path in
[_materialize_policy_lists, solver.py:268-283](lifecycle/solver.py#L268) —
single host transfer per slab, no per-age D→H churn.

### §5.2 Reversed-range correctness

`reversed(range(n_age - 1))` iterates `t = n_age-2, n_age-3, ..., 0`. The
terminal age is solved separately (it has index `n_age - 1`), and the
loop visits every non-terminal age once in reverse-time order. This is the
standard backward-induction iteration order and matches the Bellman
recursion `V_t(s) = u(c_t) + β · E[V_{t+1}(s_{t+1})]` where the kernel at
t needs c_{t+1} (already solved at t+1). **Algorithm is correct.**

The `youngest_age_to_solve` early-stop check
([solver.py:3044-3046](lifecycle/solver.py#L3044)) breaks the loop with
status `stopped_early` when `age < youngest_age_to_solve`. This is just a
partial-solve termination, not an algorithmic concern.

The `solved_age_mask[t]` skip
([solver.py:3047-3048](lifecycle/solver.py#L3047)) supports checkpoint
resume: ages that were already loaded from checkpoint are skipped. Correct.

### §5.3 One observed subtlety

When `solved_age_mask[t+1] = True` from a checkpoint resume but
`as_grid_prev = None` (since α-grids are not persisted in the checkpoint
bundle), the orchestrator falls through to the cold init at
[solver.py:3062-3064](lifecycle/solver.py#L3062). This is correct
behaviour but means **the first age after a checkpoint resume pays a cold
solve.** The next age then sees a freshly-computed `as_grid_prev` and
Variant B kicks back in. The transition cost is one age × bt-iter inflation
on resumed runs. Documented at the call site; I confirm the wiring
matches.

---

## §6 Inf-horizon fixed-point iteration

### §6.1 Reuse of the lifecycle retirement kernel

**Where:** [run_infinite_horizon_solver, inf_horizon_solver.py:497-730](lifecycle/inf_horizon_solver.py#L497).

The inf-horizon solver is a thin Python loop around the *exact same*
`_build_per_age_retirement_kernel` from the lifecycle solver, called
repeatedly with `pension_zero` and `psi_one` (and `b_bar = 0` baked into
the kernel via `mp.b_bar = 0`). Each iteration is one Bellman application:
given a candidate value-function (encoded as the consumption policy
`C_old`), apply the operator to get `C_new`, check sup-norm convergence,
optionally damp, repeat.

This sharing is conceptually right and operationally efficient. The kernel
is mathematically the same as a one-step-ahead Bellman with no income, no
mortality, no bequest — i.e. the standard infinite-horizon CRRA portfolio
problem. The fixed point of this iteration is a stationary policy.

### §6.2 Iter-0 seeding

The first iteration's per-savings α warm-start comes from a mid-wealth
gather of the prepared `S_old`/`B_old` policies, broadcast across the
savings axis ([inf_horizon_solver.py:572-580](lifecycle/inf_horizon_solver.py#L572)):

```python
w_ref_idx = pc.n_w // 2
init_a_s_arr = jnp.broadcast_to(
    jnp.asarray(S_old[:, :, w_ref_idx])[:, :, None],
    (pc.n_z, pc.N_state, pc.n_s),
)
```

This is the **Variant A scalar warm-start** (one mid-wealth value per
cell, broadcast across savings) for iter 0 only. From iter 1 onward, the
kernel's per-savings α-grid output is rolled forward as the next iter's
Variant B init. The two-stage scheme is correct: there is no per-savings
α-grid available before the first kernel call, so Variant A is the only
option for iter 0.

### §6.3 Damping

`C_next = damping * C_new + (1 - damping) * C_old` (and similarly for S,
B) at [inf_horizon_solver.py:642-647](inf_horizon_solver.py#L642). When
`damping = 1.0` (default), the damping path short-circuits to direct
assignment. Standard fixed-point damping; correct.

### §6.4 Convergence criteria

Three parallel sup-norm metrics
([inf_horizon_solver.py:480-490](inf_horizon_solver.py#L480)):

- `xi_err`: sup-norm of `xi_new - xi_old` where `xi = C / W` (consumption
  share, computed only on `wealth_grid[trim_wealth_points:]` — the first
  `trim_wealth_points` are dropped because `xi` blows up at the W=0
  boundary of an unconstrained-corner-clamped policy).
- `share_err`: sup-norm of `max(|S_new - S_old|, |B_new - B_old|)`.
- `policy_err`: sup-norm of `max(|C_new - C_old|, share_err)`.

Stopping rule:
[inf_horizon_solver.py:691-693](inf_horizon_solver.py#L691):
`stop_err = max(xi_err, share_err)`. This is **two-criterion**
convergence: both `xi` and the portfolio shares must agree to within `tol`
across iterations. The check fires at `it > 0 ∧ stop_err < tol` — the
`it > 0` guard prevents a spurious iter-0 hit when the prepared `C_old`
happens to look stationary against itself. Correct.

The sup-norm + iter-0 guard + `damping ∈ (0, 1]` validation are all
standard and correct for an inf-horizon fixed-point iteration. See
[INF_HORIZON_AUDIT_2026-05-07.md](docs/scans/INF_HORIZON_AUDIT_2026-05-07.md)
for the prior validation; my read confirms the implementation is unchanged.

### §6.5 The b_bar=0, pension_zero, psi_one zeroing

[inf_horizon_solver.py:553-564](inf_horizon_solver.py#L553):

```python
mp = ModelParams(..., b_bar=jnp.float64(0.0), ...)
pension_zero = jnp.zeros(pc.n_z, dtype=jnp.float64)
psi_one = jnp.ones(pc.n_z, dtype=jnp.float64)
```

This zeros out: (a) bequest (`b_bar = 0` makes the bequest contribution
vanish in `bequest_mu_and_mup`), (b) pension income (`pension_zero`), (c)
mortality (`psi_one` means survival prob 1, so `prob_death = 1 - psi = 0`,
which cancels the bequest term anyway). With all three off, the kernel
reduces to the canonical Merton-type infinite-horizon CRRA portfolio
problem with stochastic returns. **Correct setup for the benchmark.**

### §6.6 The KeyboardInterrupt path

The `try/except KeyboardInterrupt` at
[inf_horizon_solver.py:620-701](inf_horizon_solver.py#L620) preserves the
last fully-committed `(C_old, S_old, B_old)` and `n_iter_done` for the
diagnostics dict. The state is consistent because both update at the
**end** of each iteration body — if the interrupt fires mid-iter, the
half-computed `C_new` is discarded and the loop exits with the
last-completed `C_old`. Correct.

---

## §7 Multi-precision (gather_precision="f32") toggle

### §7.1 Cast scope

**Where:** `_cast_for_gather, solver.py:376`; `_resolve_gather_dtype,
solver.py:386-401`. Resolved at kernel-build time into a Python dtype
object, baked into the JIT trace via the `static` tuple. No runtime
branching on the dtype.

The cast scope is closed: f64 → f32 only happens inside the multilinear
gather + bracket + interp inside `_interp_c_and_mpc_at_cell`
([solver.py:974-1010](lifecycle/solver.py#L974)) and the inline
`per_kv_kr` in `retirement_foc_jac_ccv`
([solver.py:1064-1095](lifecycle/solver.py#L1064)). The cast back to f64
happens **before** the `min_consumption` floor and the `[0, 1]` MPC clip,
i.e. before any value enters the FOC sum. Two trace-time `assert c.dtype
== jnp.float64` statements
([solver.py:1019, 1093](lifecycle/solver.py#L1019)) make this a JIT
invariant — they fire at trace time, zero runtime cost, and would catch a
regression where the cast was dropped or relocated.

### §7.2 Boundary placement and FOC purity

After the cast back, every downstream operation is fp64: CRRA
`c_at_xn ** (-gamma)`, `mu_alive`, `mup_alive`, the bequest combine, the
weighted FOC sums, the Jacobian terms, the Newton step `det = ...`, the
line search comparisons, the Euler inversion `(β · V_dot) ** (-1/γ)`, the
EGM `_lift_to_wealth_grid`. The
[FP32_PLACEMENT_REVIEW_2026-05-09.md](docs/scans/FP32_PLACEMENT_REVIEW_2026-05-09.md)
audit independently traced every fp32-touched site; my read confirms its
findings.

The documented ~1e-5 relative drift in α between the f32 and f64 paths
([model.py:248-249](lifecycle/model.py#L248)) is consistent with this
boundary placement: the only f32 noise that leaks is in `c_at_xn` and
`mpc_at_xn` (the multilinear-interp output), and that noise is amplified
by CRRA `(-γ)` (γ=5 ⇒ ~5x amplification of relative error in `c → mu`).
A 1e-7 fp32 noise in `c` becomes ~5e-7 in `mu`, which propagates linearly
through the FOC sum to ~1e-5 in α after Newton convergence. **Order of
magnitude consistent.** Algorithmically, the f32 path is mathematically
transparent (the cast is at a clean boundary), but produces non-bit-
identical answers at the documented relative tol.

The f32 path does NOT taint the FOC / Newton arithmetic — verified: no
`_cast_for_gather` calls anywhere in the Newton body
([solver.py:421-815](lifecycle/solver.py#L421)). **Correct.**

### §7.3 fp32 gather and wealth-grid validation

The previous review
([MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09 §5 RED FLAG #2](docs/scans/MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md))
flagged that `validate_wealth_grid` is invoked only on the custom
file-loaded grid path, not on the canonical `legacy_log1p_wealth_grid`. I
re-confirm the gap: the canonical wealth grid has no fp32-spacing safety
check at construction time. With `gather_precision = "f32"` (canonical
default), the assumption that the log1p grid is "obviously safe" under
fp32 cast is uncovered by code. This is a defense-in-depth gap, not a
known-broken algorithm — but it leaves the canonical without the rail the
custom-grid path has. I carry it forward as ALGO-5 (low; the gap was
already flagged once and the canonical empirically works at fp32).

---

## §8 Cell-axis chunking (cell_vmap_chunks)

### §8.1 K=1 fast path vs K>1 chunked path

**Where:** `_chunked_vmap_runner, solver.py:2034-2088`;
`_chunked_pmap_runner, solver.py:2004-2031`.

The chunking strategy splits the per-age vmap over `(n_z * N_state)`
cells into K sequential chunks. Each chunk has fixed shape so XLA traces
the inner vmap once and reuses for all chunks.

Critical claim from [HANDOFF_CELL_VMAP_CHUNKING.md](docs/handoff/HANDOFF_CELL_VMAP_CHUNKING.md):
the K=1 fast path and K>1 chunked path produce **bit-identical** outputs
because they share the same `@jit`-compiled `per_chunk` function, just
called once vs K times.

The implementation honours this:

- For both paths, `per_chunk` is defined as a single `@jit` decorator
  ([solver.py:2148-2150 (terminal vmap-only)](lifecycle/solver.py#L2148),
  [solver.py:2355-2364 (retirement vmap-only)](lifecycle/solver.py#L2355),
  [solver.py:2636-2647 (working vmap-only)](lifecycle/solver.py#L2636)).
- K=1 calls `per_chunk` once with the full unpadded indices, no chunk-loop
  wrapper, no inter-chunk block. K>1 calls `per_chunk` K times via
  `_chunked_vmap_runner` with `block_until_ready` between calls (memory-
  bound enforcement).
- The padding strategy uses the last cell index repeated to fill — same
  for both paths.

So bit-identity holds **at trace level**: both paths trace to the same
HLO graph for `per_chunk`, and the output is just a concatenation of K
chunks vs one full chunk. Whether XLA actually produces bit-identical
output at runtime depends on whether the K-fold concatenation reorders
floating-point operations — for `vmap(per_cell)` over independent cells
there are **no reductions across cells**, so the per-chunk output at cell
i is independent of which chunk it lives in. **Bit-identity is preserved
at the algorithmic level.**

(I deliberately did not run the per-cell bit-identity probe at canonical
config because the constraint was "no canonical solver runs". The
algorithmic argument above is sound.)

### §8.2 Pmap-chunked path

The pmap path
([_chunked_pmap_runner, solver.py:2004-2031](lifecycle/solver.py#L2004))
uses the same strategy: each chunk is rounded up to a multiple of `n_dev`,
sharded across devices via pmap, then the chunks are concatenated and
trimmed. The bit-identity argument is the same: per-cell `vmap` across
shards has no cross-cell reductions, so the cell-i output is independent
of how cells are distributed across chunks/devices.

**Algorithmically equivalent.**

---

## §9 Multi-device pmap dispatch

### §9.1 The n_dev > 1 path

The orchestrator dispatches at trace time on `n_dev = len(jax.devices())`
([solver.py:1808-1812](lifecycle/solver.py#L1808),
[solver.py:2186-2194](lifecycle/solver.py#L2186),
[solver.py:2423-2431](lifecycle/solver.py#L2423)). When `n_dev == 1`,
the vmap-only path is used — no pmap padding/reshape/collapse, so XLA-CUDA
can fuse the per-age solve into a single kernel. When `n_dev > 1`, the
pmap path is used.

The pmap path pads the cell axis to a multiple of `n_dev`, reshapes to
`(n_dev, per_dev) + ...`, and dispatches via pmap. The padding repeats the
**last real cell index** (`cell_idx[-1]`) to fill, so the padded cells
solve a duplicate of the last real cell — wasteful but correct. After
solving, the pmap output is reshaped back to `(pad_n,) + ...` and sliced
to `[:N_state]` to drop padding. **Correct.**

### §9.2 The terminal-age z-broadcast trick

Terminal policy is z-invariant (bequest only — no continuation). The pmap
terminal kernel returns `(N_state_padded, n_w)` arrays
([solver.py:1815-1933](lifecycle/solver.py#L1815)). The orchestrator
broadcasts across z with `jnp.broadcast_to(c_T[None, :, :], (n_z, N_state,
n_w))` — stays on device, doesn't materialise. The next kernel (retirement
or working) reads it via `in_axes=None` (broadcast) on the pmap path or
threads it through the JIT trace on the vmap-only path. **Correct.** This
saves an n_z-fold redundant solve at terminal — appropriate optimisation
that does not change the answer.

### §9.3 Pmap dispatch bit-identity

Whether pmap and vmap-only produce bit-identical outputs depends on whether
XLA reorders reductions when sharding. For per-cell `vmap` (no
cross-cell reductions), the per-cell math is independent of sharding. For
*within-cell* reductions (FOC sums, Newton step computations), sharding
doesn't change anything because each device sees a complete cell. So the
algorithm is bit-identical across n_dev choices in principle — but XLA
may apply different fusion patterns under pmap, which can produce
floating-point differences at the last bit even for identical math. The
[HANDOFF_PMAP_TO_VMAP.md](docs/handoff/HANDOFF_PMAP_TO_VMAP.md) handoff
acknowledges this: "produce identical outputs (modulo the float reorder
XLA may do under fusion)".

**Algorithmically equivalent; bit-identity is at XLA's discretion.** Not
a correctness concern.

---

## §10 Diagnostics output

### §10.1 Counters and exit-code propagation

`total_newton_failures` is the integer sum over ages of `age_newton_fail`,
itself the integer count of `(ec_t != EC_INTERIOR)` per age
([solver.py:3001, 3118](lifecycle/solver.py#L3001)). The diagnostics dict
[_build_diagnostics, solver.py:3339-3379](lifecycle/solver.py#L3339)
exposes `age_newton_fail` (per-age vector) and `total_newton_failures`
(scalar) as canonical reads.

`age_newton_fail[t]` is computed from the per-(z, state, savings) exit-code
array `ec_t` returned by the kernel. The savings axis is `n_savings`
points (the s=0 anchor stripped at
[solver.py:1469, 1541, 1623](lifecycle/solver.py#L1469)). So
`age_newton_fail[t]` counts cells where the **real Newton call** failed,
correctly excluding the anchor. ✅

`newton_iter_per_age[t]` and `backtrack_iter_per_age[t]` are per-(z,
state, savings) arrays, materialised to NumPy at age boundary
([solver.py:3116-3117](lifecycle/solver.py#L3116) — small, ~`n_z *
N_state * n_savings` ints per age). Aggregated by `_build_iter_histograms`
([solver.py:3286-3336](lifecycle/solver.py#L3286)) into a dict with `p50,
p95, p99, max, per_age_p99, per_age_max, per_age_ages, n_cells`. The
[BACKTRACK_ITER_AUDIT_2026-05-09.md](docs/scans/BACKTRACK_ITER_AUDIT_2026-05-09.md)
already validated all nine semantic claims around these counters; I
verified the wiring is unchanged and confirms the counter semantics are
correct.

### §10.2 The terminal special case

Terminal `n_iter` and `n_backtrack` arrays are `(N_state, n_savings)`
(z-invariant), while non-terminal are `(n_z, N_state, n_savings)`. The
histogram aggregator flattens before computing percentiles, so the
shape difference is harmless — the per-cell granularity is preserved.
Correct.

### §10.3 The inf-horizon parallel

[inf_horizon_solver.py:_build_iter_histograms_per_iter](lifecycle/inf_horizon_solver.py#L377)
is a per-iteration analog of the lifecycle solver's per-age histogram. The
key rename `per_age_*` → `per_iter_*` is the only structural change;
otherwise the dict shape is parallel. `total_newton_failures` is computed
from `newton_failures_per_iter` which is the integer sum of
`(ec_iter != EC_INTERIOR)` per iter
([inf_horizon_solver.py:639](inf_horizon_solver.py#L639)). Same semantics
as the lifecycle path.

---

## §11 EC codes — semantics, propagation, current bugs

### §11.1 Defined codes

`EC_INTERIOR = 1`, `EC_NEWTON_FAIL = 2`, `EC_TINY_SAVINGS = 3`
([solver.py:47-49](lifecycle/solver.py#L47)).

### §11.2 What gets stamped where

- `EC_INTERIOR` is stamped by the Newton solver when `converged = True`
  ([solver.py:624, 813](lifecycle/solver.py#L624)). It is also stamped on
  the s=0 anchor by the orchestrator
  ([solver.py:1342-1344](lifecycle/solver.py#L1342)) so the histogram
  doesn't count the artificial anchor as a failure.
- `EC_NEWTON_FAIL` is stamped by the Newton solver when `converged =
  False` (ran out of `max_iter`) OR when `ls_failed = True` (line search
  exhausted). The two exit conditions are not distinguished — both
  collapse to EC_NEWTON_FAIL.
- `EC_TINY_SAVINGS` is **never set anywhere in the solver**. The
  tiny-savings branch in `_egm_scan_cell` overrides `c, α_s, α_b` but
  preserves whatever `exit_code` Newton returned
  ([solver.py:1325-1330](lifecycle/solver.py#L1325)).

### §11.3 The EC_TINY_SAVINGS bug

This is a real but low-impact bug. `EC_TINY_SAVINGS = 3` is a dead code:
declared but never produced. The tiny-savings cell at `s ≤ 1e-6`
(typically only `s_grid[0] = 1e-8`) gets stamped with whichever exit code
Newton happened to return, and that code propagates to:

- The `age_newton_fail` count (if Newton failed, the tiny-savings cell
  inflates the count even though its α was overridden to init).
- The failed-cell fixup (which then attempts to neighbor-seed from a
  converged-below cell, but there is none at index 0 — so it falls back
  to the cold scalar, which is what the tiny-savings override already
  set, so the fixup is a silent no-op).

The right fix is to stamp `EC_TINY_SAVINGS` in the tiny-savings branch
([solver.py:1325-1328](lifecycle/solver.py#L1325)) and to treat
`EC_TINY_SAVINGS` as "not a real Newton failure" for the
`age_newton_fail` count. FINDING ALGO-1 in §13.

### §11.4 The high-W cold-init pin cascade

The [POLICY_REVIEW_2026-05-09.md §3](docs/scans/POLICY_REVIEW_2026-05-09.md)
documents that 98-100% of cells at high-W (`W ≥ 410 AWI`) are stuck at
the cold init `(α_s, α_b) = (0.85, 0.44)`. This is the cascade signature
that the failed-cell fixup §4.2 was designed to break.

The fixup **does** break the cascade in the warm-start buffer (verified
by the cascade-break test
[test_failure_cell_neighbor_seed.py:test_cascade_breaks_under_under_budgeted_newton](tests/test_failure_cell_neighbor_seed.py#L212)),
but it does NOT change the wealth-grid policies at the failed cells —
because those policies were derived from `_lift_to_wealth_grid` operating
on the EGM cloud where the failed cells contributed their pre-fixup α to
the post-Path-B-clamp lift. So the wealth-grid C/S/B at high-W stays
pinned at `(0.85, 0.44)` even with the fixup on. The fixup helps the
**next age's** Newton init, which can then converge and produce a non-
cold policy at the next age — but the current age's high-W policy is
still cold-pinned.

The fix is correct as a cascade-breaker but does not solve the underlying
high-W FOC pathology. That is a separate problem (mentioned in the
handoff Out-of-Scope §) and out of this review's scope. **Algorithmically,
the fixup is a strict improvement but not a complete solution.**

---

## §12 Numerical / algorithmic identity verifications

I ran the following probes (all under
`JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 TF_CPP_MIN_LOG_LEVEL=2`, no canonical
solver runs).

### §12.A Path-B clamp

Synthetic EGM cloud with anchor + tiny-savings + 4 interior points. Wealth
grid spanning below and above `W_min_real`. **Result:** every wealth-grid
point below `W_min_real` returns `(c=W, α_s=0, α_b=0)` exactly; points
above go through normal `jnp.interp`. Bit-identical to the post-clamp
expected behaviour. **PASS.**

### §12.B fori vs while parity

Synthetic linear FOC with five overshoot regimes (warm-at-root, perfect H,
2× / 4× / extreme-overshoot). **Result:** fori and while produce
bit-identical `(a_s, a_b, exit_code, n_iter, n_backtrack)` on every
configuration. Independently confirms the
[BACKTRACK_ITER_AUDIT_2026-05-09 Q4](docs/scans/BACKTRACK_ITER_AUDIT_2026-05-09.md)
parity claim. **PASS.**

### §12.C Singular Jacobian fallback sign

Synthetic singular FOC `(f_s, f_b) = (a_s - 0.5, a_b - 0.5)` with
`J = [[1,1],[1,1]]` (det = 0). Initialised at `(0.7, 0.7)`. Expected
first-iter step (with corrected sign): `-0.05 · 0.2 / 0.2828 = -0.03536`,
giving `a_s = 0.66464`. **Observed:** `a_s = 0.664645` after 1 iter in
both fori and while. The sign is correct (post-commit `1b2ebb1`). The
prior audit's RED FLAG #1 is now resolved. **PASS.**

### §12.D _fixup_failed_cells edge cases

Five edge cases (trailing failures, all-converged, all-failed, leading
failures with later-converged, terminal 2D shape). All five match the
expected output bit-exactly. **PASS.**

### §12.E EGM scan output structure

Synthetic foc_factory with V_dot proxy. Verified that the output has
shape `(n_savings + 1,)` with index 0 being the egm_anchor sentinel,
index 1+ being `c_out + s_val` from per-savings Newton. The output is
**not pre-sorted** (since `c_opt(s) + s` is not monotone in s under
arbitrary FOC), but the lift step's `argsort` handles this. The tiny-
savings branch correctly forces `c = min_consumption` and `α =
init_a_s_v` at `s ≤ 1e-6`. The exit code at the anchor is stamped
EC_INTERIOR (correctly suppressing it from the failure count). **PASS.**

### §12.F FOC residual at converged cells in the saved bundle

I did NOT run this probe because the constraint excluded loading a
canonical bundle (`saved_runs_lambda_2026-05-09/test_retire_g7/` is
production-scale, hundreds of MB). The relevant evidence is in
[POLICY_REVIEW_2026-05-09 §4](docs/scans/POLICY_REVIEW_2026-05-09.md):
mid-state, W ≤ 30 AWI cells achieve `max_rel ≤ 1.4e-4` between mi=10 and
mi=100, which is fp32-noise level — strong evidence that converged cells
have `‖f‖ << tol`. The orchestrator is reading the right thing.

---

## §13 Findings table

| ID | Severity | Location | What's wrong / fragile | Fix sketch |
|---|---|---|---|---|
| ALGO-1 | **LOW** | `_egm_scan_cell` tiny-savings branch [solver.py:1325-1328](lifecycle/solver.py#L1325) | `EC_TINY_SAVINGS = 3` is declared but never produced; tiny-savings cells inherit whatever Newton returned, which can spuriously inflate `age_newton_fail` and trigger no-op fixup work. | Stamp `EC_TINY_SAVINGS` in the tiny branch; treat `EC_TINY_SAVINGS` as not-a-failure in the histogram aggregator + fixup. |
| ALGO-2 | LOW | `_lift_to_wealth_grid` Path-B threshold [solver.py:1409](lifecycle/solver.py#L1409) | The clamp threshold `c > 2 * min_consumption` is correct in practice but algorithmically gates on a quantity (`c_opt`) that is downstream of the choice it intends to detect (tiny-savings vs interior). A pathological low-V̇ cell with `c_opt < 2e-10` would silently trigger the corner clamp. | Gate on `s > tiny_savings` directly: store a Boolean `is_real` per s-point during the EGM scan and pass it through to the lift. |
| ALGO-3 | MEDIUM | `_lift_to_wealth_grid` `argsort + jnp.interp` [solver.py:1384-1391](lifecycle/solver.py#L1384) | If `x_egm(s)` is non-monotone (rare under CCV log-wealth dynamics but possible at saturated portfolio cells), `argsort + jnp.interp` silently linearly bridges across the fold, producing MPC > 1 or non-monotone `c(W)`. No detection. | Add an `assert jnp.all(jnp.diff(x_sorted) > 0)` (trace-time NaN check) or an explicit monotone-envelope construction (Druedahl-Jorgensen 2017 "upper envelope" method). |
| ALGO-4 | **CRITICAL-BUT-KNOWN** | `_egm_scan_cell` FOC normalisation [solver.py:1294-1296](lifecycle/solver.py#L1294) + Newton convergence test [solver.py:624, 813](lifecycle/solver.py#L624) | At high-savings cells, `e0 = E[mu(c) · R_p]` collapses to fp64 noise as the wealth grid hits its top; `inv_foc_scale` blows up; the convergence test becomes ill-defined. 100% of cells at `W ≥ 410 AWI` pin at the cold init. | Cap `wealth_max` to where Newton actually converges (~$300k, per failure-pattern analysis); or add a residual-relative convergence test as fallback when the per-savings constant scaling underflows; or use a fundamentally different solver in the high-W tail (e.g. EGM-only). The handoff §13 fixup §4 is a partial fix — it breaks the cascade through the warm-start but does not solve the underlying convergence. |
| ALGO-5 | LOW | `precompute.py:247-251` (per prior audit) | Canonical `legacy_log1p_wealth_grid` skips the `validate_wealth_grid` fp32-spacing safety check that the custom file-loaded grid path runs. With `gather_precision="f32"` the canonical default, this invariant is uncovered by code. | Invoke `validate_wealth_grid` on the log1p grid too. |
| ALGO-6 | LOW | `solver.py:3059-3064`: as_grid_prev=None fallback at first age after checkpoint resume | The first age after a checkpoint resume pays one cold solve (Variant A scalar) because the α-grid buffer is not persisted in the bundle. From the next age onward, Variant B kicks back in. | Persist the per-savings α-grid for the youngest solved age in the checkpoint bundle (~`n_z * N_state * n_savings * 8 * 2 / 1e6` MB ≈ 1 MB at canonical). Read it back into `as_grid_prev` on resume. |
| ALGO-7 | NONE | `_fixup_failed_cells` does not edit wealth-grid C/S/B | The fixup operates only on the warm-start α buffer for the next age, not on the current age's lifted policies. So the high-W tail stays cold-pinned in C/S/B for any one age, even with the fixup on; the fixup helps the next age's Newton solve. | Working as designed (architecturally correct). The handoff explicitly defers wealth-grid fixup to a separate effort. No action. |
| ALGO-8 | INFORMATIONAL | `_egm_scan_cell` runs Newton at `s ≤ tiny_savings` and discards the result | Wasteful (one Newton solve per cell at the s=1e-8 endpoint that always gets overridden). Not an algorithmic bug — just wasted compute. | Skip Newton at tiny-savings cells with `lax.cond` (would break the savings vmap's uniform shape requirement, so probably not worth the trouble). Documented as an accepted inefficiency. |
| ALGO-9 | LOW | inf-horizon `_compute_metrics_numpy` denominator [inf_horizon_solver.py:484](inf_horizon_solver.py#L484) | `xi = C / W` divides by `wealth_grid[trim_wealth_points:]`. With `trim_wealth_points = 5` (default) and `wealth_min = 0.05`, the smallest W in the denominator is `~wealth_grid[5]` which is well-defined. But there is no guard against `W = 0` if `trim_wealth_points = 0` is passed; the validator `_validate_runtime_options` allows `trim_wealth_points = 0`. | Either validate `trim_wealth_points >= 1` if `wealth_min == 0`, or use `np.maximum(W, eps)` in the divisor. |

(Severity: LOW = cosmetic / unused / wasted compute / never fires in practice; MEDIUM = real bug under specific configurations; CRITICAL-BUT-KNOWN = real algorithmic limitation, documented in prior audits, mitigation pending.)

---

## §14 Verdict

**PASS WITH CAVEATS.**

The orchestration of the EGM scan + 2D Newton + line search + lift +
backward-induction loop is **algorithmically correct and well-engineered**
across all the dimensions in scope:

- The EGM scan over `s_grid` is the standard EGM iteration order (fix s,
  solve interior FOC, recover c via inverted Euler, back out W).
- The 2D Newton with backtracking line search is correctly implemented in
  both fori and while modes; bit-identical parity verified by direct
  probe; the singular-Jacobian fallback sign is correct (post-fix).
- The s=0 anchor and Path-B constrained-corner clamp produce the right
  policy at the EGM lower boundary (verified by direct probe).
- Variant B per-savings backward-age warm-start correctly threads the
  prev-age α-grid forward through a single rolling buffer; the
  failed-cell neighbor-seed fixup correctly breaks the cold-init cascade
  through the warm-start (5/5 edge cases pass).
- Backward-induction orchestration: terminal-then-reversed-range with
  age-based dispatch to retirement / boundary / working kernels is
  faithful to the Bellman recursion.
- Inf-horizon iteration reuses the lifecycle retirement kernel correctly
  and adds the right damping + sup-norm convergence rule.
- Multi-precision toggle has a clean fp32 boundary; FOC / Newton
  arithmetic stays fp64.
- Cell-axis chunking is bit-identical to the K=1 fast path (shared
  `@jit`-traced `per_chunk` function).
- Pmap dispatch is mathematically equivalent to vmap-only.
- Diagnostics output is computed from the right exit codes via the
  documented aggregator.

The **caveats** that prevent an unconditional PASS:

1. **The high-W tail is not solved (ALGO-4).** The Newton convergence test
   is ill-defined at high savings where `e0` collapses to fp64 noise; 98-
   100% of cells at `W ≥ 410 AWI` pin at the cold init; the failed-cell
   fixup breaks the warm-start cascade but doesn't fix the underlying
   FOC. Production simulators that drive W above ~$300k will read off
   the cold init, not the policy.

2. **`EC_TINY_SAVINGS = 3` is dead code (ALGO-1).** A real bug, low
   impact, easy fix.

3. **`_lift_to_wealth_grid` does not detect non-monotone `x_egm(s)`
   (ALGO-3).** Rare under CCV log dynamics but algorithmically
   undefended; could produce MPC > 1 without warning.

The other findings are low-severity hygiene issues (fp32 spacing
validator gap, checkpoint resume cold transition, Path-B threshold
fragility) that are not blockers.

The two most important architectural strengths I want to call out:

- The decision to keep the `_fixup_failed_cells` post-process **outside**
  the JIT-traced kernel was right. Pushing it into the kernel would have
  required rebuilding the trace and would have made the warm-start
  buffer's shape contract opaque. The orchestrator-level placement keeps
  the kernel signatures clean and the fixup auditable.

- The `per_chunk` `@jit` shared between K=1 and K>1 chunked paths is the
  correct way to guarantee bit-identity across `cell_vmap_chunks` values.
  A separate fast-path trace would have invited subtle XLA-fusion
  divergences across the chunk count.

Both deserve to be highlighted because they were "tactical" decisions
that have outsized correctness benefits.

---

### Closing note on coordination with the math reviewer

I did not coordinate with the parallel reviewer covering math/equation
derivation. The natural overlap with that scope is in §3 (Newton
convergence test) and §11 (EC code semantics). My §3 covers the
algorithmic implementation of the convergence test; the math reviewer
will likely look at whether the `tol = 1e-7` choice is appropriate for
the FOC's natural scale. Both perspectives converge on ALGO-4 (the high-W
convergence-test pathology), which is the single most important finding
on my side.

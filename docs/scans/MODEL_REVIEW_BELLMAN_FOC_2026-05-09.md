# Model review — Bellman / FOC mathematical structure (2026-05-09)

**Branch:** `jax-rewrite`. **Scope:** Read-only review of the Bellman
recursion, value-function decomposition, FOC derivations, EGM mapping,
and constrained-branch handling for the post-pivot 3-axis real-yields
canonical. Complements `MODEL_REVIEW_SOLVER_INTERNALS_2026-05-09.md`
(numerical solver), `ECONOMIC_SETUP_REVIEW_PART_A/B_2026-05-09.md`
(primitives), and the parallel discretization / coherence scans.

---

## 1. Bellman equation structure

**What I found.** The recursion is correctly decomposed by age dispatch in
`run_lifecycle_solver` (`solver.py:2755-2777`):

- **Terminal (age 99)** — `terminal_kernel` (`solver.py:2680`) solves
  `V_T(state, W) = max_{c, α_s, α_b} u(c) + β · E[b(s · R_p, A_is)]`
  via EGM (`_solve_terminal_at_i_s` → `terminal_foc_jac_ccv:776`). The
  agent saves `s = W − c`, invests one period, and dies at T+1; heirs
  receive `s · R_p`. β scales the bequest because the bequest accrues
  one period after the choice (`docs/UTILITY.md §2.3-2.4`). z is
  inert at terminal — policy is broadcast across z (`solver.py:2687`).
- **Retirement (67 ≤ age ≤ 98)** — `retirement_kernel`. z is frozen
  (no eta/eps integration). Income next is `pension_table_jnp[t+1, z]`
  (`solver.py:2756`). FOC at `retirement_foc_jac_ccv:952` integrates
  bequest+alive contributions over `(k_v, k_r)` only.
- **Working (22 ≤ age ≤ 65)** — `working_kernel`. z transitions via
  `z' = ρ·z + η`; income next is `working_income_next[t+1, z, η, ε]`.
  FOC integrates over `(k_v, k_r, k_eta, k_eps)`.
- **Boundary (age 66)** — `boundary_kernel` (`solver.py:2762-2769`).
  Working FOC trace, but `income_table[k_eta, i_e] = pension(z')` —
  flat across ε since at age 67 the agent collects pension (no eps
  shock). Linear-z bracket `(iz_lo, frac_z)` consistently feeds both
  the c_next gather and the pension interpolation, so the agent's age-66
  saving anticipates the correct age-67 retirement income at the
  realized z'.

**Verdict: CLEAR.** Bellman dispatch matches the lifecycle; terminal
β-discounted bequest is correct under the period-end-death convention;
the work→retirement transition handles the income-shock structural
change correctly via the dedicated `boundary_kernel` trace.

---

## 2. Continuation value E_t[V_{t+1}] and Euler equation

**What I found.** The continuation enters via the inverted Euler:
`c_t = (β · V_dot)^{-1/γ}` at `_egm_scan_cell:1224-1225`, where

```
V_dot = E[ψ·u'(c_{t+1})·R_p + (1−ψ)·b'(W_T+1)·R_p]
     = sum over (k_v, k_r [, k_eta, k_eps]) of weight × mu_comb × R_p
```

(`working_foc_jac_ccv:1149` for alive, `:1098` for bequest; recombined
in `_egm_scan_cell` as `e_bq + e_al`). `mu_alive = c_at_xn^{-γ}` is
the CRRA marginal utility (`solver.py:1027, 1137`).

The expectation order is `weight_kv_kr ⊗ eta_weights ⊗ eps_weights`
(`solver.py:1140-1144`), respecting the conditional independence
`(v^s ⊥ ε ⊥ η) | s_t` and the standard CGM 2005 / Catherine 2025
joint-quadrature factorisation. State innovation `v_{t+1}` enters the
return mean via `M_v_nodes` (`solver.py:832`), the residual
`ε_{ret,t+1}` enters via `ret_nodes` (`solver.py:835-836, 843-844`).
Bill leg is deterministic given `s_t` (`solver.py:840-842`) — the
real-yields pivot reviewed in `REAL_YIELDS_PIVOT_REVIEW_2026-05-08.md`.

**Verdict: CLEAR.**

---

## 3. EGM mapping and constrained-branch handling

**What I found.** EGM scan in `_egm_scan_cell:1180`:
1. For each `s ∈ s_grid`, solve `(α_s*, α_b*)` from the 2D portfolio FOC.
2. Recover `c_opt = (β · V_dot)^{-1/γ}` from the inverted Euler.
3. Implied wealth `x_egm = c_opt + s`. Store `(x_egm, c_egm, α*)`.
4. `_lift_to_wealth_grid:1249` does `jnp.interp(wealth_grid, x_sorted, ...)`.

Two mechanisms suggest "constrained branch" but neither solves the
true constrained corner `(c=W, s=0, α* irrelevant)`:

- **`tiny_savings = 1e-6` branch** (`_egm_scan_cell:1227-1230`): when
  `s ≤ 1e-6`, output `c = min_consumption = 1e-10`, `α = init_α`
  (the cold init `(0.85, 0.44)`). With `s_grid[0] = 1e-8`, only the
  smallest savings point trips this branch.
- **`egm_anchor = 1e-10` prepend** (`_egm_scan_cell:1236-1238`): an
  artificial `(x=1e-10, c=1e-10, α=0)` point is concatenated to the
  EGM arrays before sorting + linear interp.

For wealth `W` between the anchor `(1e-10, 1e-10)` and the smallest
solved `x_egm[1]` (typically ≈ 0.4–0.6 AWI at γ=5 in working ages),
`jnp.interp` returns a piecewise-linear bridge: `c(W) ≈ W ·
c_egm[1]/x_egm[1]` and `α(W) ≈ W · α_egm[1]/x_egm[1]`. This
**approximates** `c=W, α=0` in the limit, but the slope `c/W` at
small `W` is `c_egm[1]/x_egm[1]` (typically ≈ 0.93), NOT 1.0 — so the
agent does NOT fully consume at `W = wealth_min = 0.05`.

The portfolio policy for low `W` is even further off: it linearly
interpolates from `α=0` (at the anchor) to `α* ≈ 0.85` at `x_egm[1]`,
giving values like `α(W=0.05) ≈ 0.05/0.5 · 0.85 ≈ 0.085` — an
artefact of the linear-interp-from-anchor trick, not an economically
meaningful constrained policy.

**Verdict: RED FLAG (UNDOCUMENTED).** Commit `5a0e25c` lowered
`wealth_min` 0.13 → 0.05 with the message *"exposes the constrained
branch to the solver. The kink region is now part of the solve, not
elided."* This is **misleading**: there is no constrained-corner
branch in the JAX rewrite. The kink at `W=0` is approximated by an
artificial `egm_anchor` + linear interp from the smallest unconstrained
EGM point. The solver is unaware of the kink. `docs/STATE_SPACE.md:219`
still describes `wealth_min` as "skip the EGM constrained region",
which is now stale in both directions: it neither *skips* nor *solves*
the constrained branch — it linearly approximates it from a tiny
anchor. Reviewers cross-checking the canonical's constraint set against
CGM 2005 / Catherine 2025 (which solve `c=W` exactly at the kink) will
not find that branch in this codebase. **Reproduction:** at any
`(z, i_s)` near retirement, simulate an agent with `W < 0.5` and read
the policy: `c/W ≈ 0.93` and `α_s ≈ 0.10`, not `c=W` and `α=0`.

---

## 4. Portfolio + consumption FOC

**What I found.** Standard CGM 2005 / Catherine 2025 structure:

```
∂L/∂c    : u'(c) = β · E[mu_comb · R_p]                      (Euler)
∂L/∂α_s  : β · s · E[mu_comb · ∂R_p/∂α_s] = 0                (stock FOC)
∂L/∂α_b  : β · s · E[mu_comb · ∂R_p/∂α_b] = 0                (bond FOC)
```

with `mu_comb = ψ_z · u'(c_{t+1}) + (1-ψ_z) · b'(s·R_p)`
(`solver.py:1031`). The 2D Newton (`newton_2d_with_line_search:421`)
solves the two portfolio FOCs given `s`; consumption is then recovered
analytically via the Euler at `_egm_scan_cell:1225` (`c_opt =
(β·V_dot)^{-1/γ}`).

Mortality enters as `prob_death = 1 − ψ_z` weighting the bequest
marginal utility against alive consumption marginal utility — Catherine
2025 eq. 35 standard. ψ depends on current `z` (the period-t mortality
rate is determined by current persistent income), not z_next ✓.

The Jacobian terms `dRp_das = R_p · dr_da_s`, `dRp_dab = R_p · dr_da_b`
where `dr_da_s = log_x_s + σ²_xr·(0.5 − α_s) − α_b·σ_xrxb`
(`_ccv_log_return_and_grad:767`) — analytically correct derivative of
the CCV log-return formula. `extra_ss = wmu·R_p·(dr_da_s² − σ²_xr)`
captures the second-order term `R_p·(dr/dα_s)² + R_p·d²r/dα_s²` with
`d²r/dα_s² = -σ²_xr` (correct).

`s_val · ...` enters Jacobian via `mu' · dW/dα_s = mu' · s · dR_p/dα_s`
(at line 801, 1044, 1099, 1151). `mup = -γ · mu / (A · C̄)` for
bequest (`bequest_mu_and_mup:413`) and `mup_alive = -γ·mu_alive·mpc/c`
(`solver.py:1028, 1138`) — chain rule through wealth-interp slope mpc.

**Verdict: CLEAR.** The FOC math, gradient, and Jacobian are all
analytically correct and consistent with the CGM 2005 / Catherine 2025
spec. (Already independently flagged "CORRECT" in
`REAL_YIELDS_PIVOT_REVIEW_2026-05-08.md §2`; this scan re-derives the
gradient and Hessian extras and finds no error.)

---

## 5. CCV log-portfolio derivation

**What I found.** Formula (`_ccv_log_return_and_grad:755-765`):

```
r_p = log_R_bill + α_s·log_x_s + α_b·log_x_b
       + 0.5·(α_s·σ²_xr + α_b·σ²_xb)
       − 0.5·(α_s²·σ²_xr + 2·α_s·α_b·σ_xrxb + α_b²·σ²_xb)
```

This is the 2-asset Campbell-Viceira approximation (CCV w8566 eq. 10).
The Jensen-correction `+0.5·α·σ²` converts `E[α·log(R/R_bill)]` to
`α·log(E[R/R_bill])` under log-normal returns; the `−0.5·α'Σα` term is
the Itô vol-drag from log-aggregating a portfolio. Cross-term
`−α_s·α_b·σ_xrxb` is the standard quadratic-form expansion. Variance
scalars sourced from `Sigma_rr` (unconditional return covariance)
indexed by `xr_pos / xb_pos` — matches CCV Table 2 vols and Markowitz
benchmarks per `precompute.py:332-342`. (Reviewed CORRECT in the
real-yields pivot scan.)

**Verdict: CLEAR.**

---

## 6. Constrained branch (re-prosecution at the bottom of the wealth grid)

**What I found.** No explicit Lagrangian or KKT branch in the canonical
JAX kernel:
- No `alpha_min`/`alpha_max` field on `SolverConfig`
  (`model.py:128-229`).
- No simplex projection, no `c_constrained = W` branch.
- The 2D Newton solves only the interior FOC.
- The "constrained" policy values at `W < x_egm[1]` are produced by
  `jnp.interp` linearly bridging from the artificial `egm_anchor`
  (1e-10, 1e-10, α=0) — see §3.

Implication: the canonical model **assumes the household is
unconstrained interior**, with the small-W kink approximated rather
than solved. With CCV log dynamics `R_p > 0` always, so `s · R_p > 0`
whenever `s > 0` — i.e., the agent never goes bankrupt. This rules
out the bankruptcy boundary, but does NOT rule out the borrowing
constraint at `s ≥ 0`. Whether the unconstrained interior solution
satisfies `s* ≥ 0` for all `(z, state, W)` near `wealth_min = 0.05`
is not verified anywhere in the codebase. If the unconstrained `s*`
ever wants to be negative (i.e., the agent wants to borrow), the EGM
returns `s = s_grid[0] = 1e-8` plus `c_opt` from interior Euler, which
is not the same as the proper `s = 0, c = W` corner.

The pivot review (`ECONOMIC_SETUP_REVIEW_PART_A §4.1`) flagged this
gap before the wealth_min change; the change moved `wealth_min` lower
but did not add a constrained branch. The EGM linear-interp trick
(§3) is the only constrained-region "handling" in the canonical.

**Verdict: UNDOCUMENTED.** The "no constrained branch" assumption is
implicit and is not stated in `docs/CONFIG.md` or
`docs/STATE_SPACE.md`. A reader of `_canonical.py:83-88` would
plausibly conclude the constrained branch is now solved by the
canonical (per the comment "the EGM constrained branch is solved, NOT
skipped"). It is not.

---

## TL;DR

| # | Area | Verdict |
|---|---|---|
| 1 | Bellman dispatch (terminal / retirement / working / boundary) | CLEAR |
| 2 | Continuation E[V_{t+1}] and Euler equation | CLEAR |
| 3 | EGM mapping and the egm_anchor approximation | **RED FLAG** |
| 4 | Portfolio + consumption FOC derivation | CLEAR |
| 5 | CCV log-portfolio formula | CLEAR |
| 6 | Constrained branch (no explicit handling) | UNDOCUMENTED |

**Single most important RED FLAG.** The canonical's "constrained
branch" is not solved — it is approximated by linear interpolation
from an artificial `(W=1e-10, c=1e-10, α=0)` anchor to the smallest
EGM-solved point. Commit `5a0e25c`'s `wealth_min: 0.13 → 0.05` with
the rationale *"exposes the constrained branch to the solver"* is
**misleading**: lowering `wealth_min` only widens the wealth grid into
a region where the policy is dominated by the linear-interp-from-anchor
artefact, not by a true constrained corner. At `W = 0.05`, the
canonical produces `c/W ≈ 0.93` and `α_s ≈ 0.085` (interpolated
fractions), not the standard liquidity-constrained `c = W, α = 0`.
Reviewers comparing against CGM 2005 / Catherine 2025 (where the kink
is solved exactly) will not find that branch in this codebase. Either
add a constrained-corner branch to the EGM scan, or document the
linear-interp approximation explicitly in `docs/UTILITY.md §2.3` and
`docs/STATE_SPACE.md §4`.

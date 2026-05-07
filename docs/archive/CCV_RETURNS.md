# CCV LOG-WEALTH DYNAMICS — THEORY AND IMPLEMENTATION

**Status:** CCV is the **default** return spec across `SolverConfig` and
`simulate_lifecycle` as of May 2026 (production bundle
`system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire`).
The legacy `simple_clamp` path remains in the code as a regression
backstop and as a tool for re-running historical bundles, but is no
longer the default and no production figure or table is generated under
it. To use simple+clamp, callers must opt in explicitly via
`SolverConfig._replace(wealth_dynamics_spec="simple_clamp")` and pass the
matching kwarg to `simulate_lifecycle`.

**Authoritative config:** [configs/run_canonical_ccv.py](../configs/run_canonical_ccv.py)
inherits from [configs/_canonical.py](../configs/_canonical.py) and flips
exactly three fields: `wealth_dynamics_spec="ccv_log"`, `wealth_min=0.01`,
`youngest_age_to_solve=67`.

**Source references in this document use** `lifecycle/<file>.py:<line>` for
package modules and `scripts/<path>.py:<line>` for command-line entry points.

---

## 0. One-paragraph summary

The legacy spec defines per-period gross portfolio return as the simple
linear combination
`R_p = α_s·R_s + α_b·R_b + (1−α_s−α_b)·R_bill` and clamps next-period wealth
to `max(s·R_p, 0)` so that leveraged shocks driving `R_p ≤ 0` cannot send
the agent into negative wealth. The clamp introduces a discontinuity in the
P-FOC integrand at the bankruptcy boundary `{sR_p = 0}`, which prevents
quadrature from delivering publication-grade Euler residuals in leveraged
cells. **Under the CCV spec we instead form the log portfolio return
`r_p = r_bill + α_s·xr + α_b·xb + (½)α'·σ²_x − (½)α'·Σ_xx·α` and set
`R_p = exp(r_p)`.** The bankruptcy boundary disappears because `exp(·) > 0`
strictly, the indicator branches drop out of the FOC, and the wealth
transition becomes `x_{t+1} = s·R_p + π` unconditionally.

---

## 1. Theoretical background

### 1.1 The Campbell–Viceira loglinear portfolio return

For a portfolio with risky log shares `α = (α_s, α_b)` and bill share
`α_bill = 1 − α_s − α_b`, Campbell, Chacko, Rodriguez and Viceira (NBER
w8566, equation 10; reprinted in Campbell & Viceira 2002) derive the
discrete-time log-return approximation

```
r_p,t+1 = r_bill,t+1
        + α_s · xr,t+1 + α_b · xb,t+1                          (1)
        + (½) [α_s · σ²_xr + α_b · σ²_xb]                       — Jensen
        − (½) [α_s²·σ²_xr + 2 α_s α_b·σ_xrxb + α_b²·σ²_xb]      — Itô / vol-drag
```

where

| symbol | meaning |
|---|---|
| `r_bill` | realised log real bill return (= rtb) |
| `xr`, `xb` | realised log excess returns over `r_bill` |
| `σ²_xr`, `σ²_xb` | **unconditional** variance of `xr`, `xb` innovations (= diagonal of `Sigma_rr`, the x×x sub-block of `Sigma_v`); see CCV_RETURN_IMPLEMENT.md §3.1.d |
| `σ_xrxb` | **unconditional** covariance of `xr`, `xb` innovations (= `Sigma_rr[xr, xb]`) |

**May-2026 Sigma_rr patch.** Pre-patch this document and the precompute path
sourced these scalars from `Sigma_r_cond` (the inner-quadrature Cholesky
residual `Sigma_rr - M·Sigma_ss·M'`), which is wrong: eq. (10) is a
path-by-path identity over the **full** VAR innovation, so the variance
correction must use the unconditional `Sigma_rr` block. The two differ
materially on this calibration (Sigma_rr is ~3-30x larger than Sigma_r_cond
along the xr diagonal). The patch is locked at `precompute.py:303-314` and
guarded by `tests/test_sigma_rr_sourcing.py`. CCV_RETURN_IMPLEMENT.md §3.1.d
is the authoritative theory derivation.

**May-2026 partition change.** This document predates the rtb-as-state
migration in which `rtb` moved from the return block to the state block.
References to "the return block" in this archived doc include `rtb`
implicitly; the current code has only `(xr, xb)` in the return block and
`(y_1, spr, dp, rtb)` in the state block. CCV_RETURN_IMPLEMENT.md §2.2 and
the var.py code mapping table reflect the current partition.

The Jensen lift `+(½)α·σ²_x` arises from `E[exp(X)] = exp(E[X] + ½Var[X])`
applied to each individual asset. The vol-drag `−(½)α'Σ_xx α` is the Itô
correction that makes the *portfolio* a true convex combination in arithmetic
expectation — it penalises risk concentration in proportion to the squared
position size. At the corner allocations `α = e_s` (full stock) or `α = e_b`
(full bond) the two corrections cancel exactly:
`r_p = r_bill + xr` (or `r_bill + xb`), reproducing the single-asset log
return. At zero allocation `α = 0` both corrections vanish:
`r_p = r_bill`. This is the canonical sanity check.

The approximation is a 2nd-order Taylor expansion of `log E[R_p^simple]`
around the conditional mean of `(xr, xb)`. The truncation is `O(|α|³·σ⁴)`,
which at the production calibration (`σ²_xr ≈ 0.025`, `σ²_xb ≈ 0.005`) is
sub-percent at `|α| ≤ 4` and grows mildly thereafter.

### 1.2 Why the simple+clamp spec breaks Euler residuals

Under simple returns `R_p = α'·R + R_bill·(1−1·α)` is a linear function of
`(xr, xb, r_bill)` and does *not* preserve positivity. With unconstrained
`α ∈ [−6, 6]²` and Lobatto tail nodes at ±7σ, the realised `R_p` can dip
below zero on a non-trivial measure of the quadrature cloud. The legacy
spec handles this with `x_{t+1} = max(s·R_p, 0) + π`, which is C⁰ but not
C¹ at `{sR_p = 0}`. The Euler integrand
`E[μ_alive(c_{t+1}) · R_p · ∂α(·)]` then has a jump discontinuity along the
bankruptcy boundary. Quadrature converges at first order in node count, not
the spectral rate Gauss–Hermite advertises. Mean log₁₀|EE| stalls around
−2.5 in unconstrained leveraged cells regardless of how many nodes you add.

The bankruptcy event itself is mostly a discrete-time artefact: a
continuous-time agent with the same wealth dynamics never crosses `W=0` at
positive savings because Brownian motion has finite quadratic variation per
unit time. Replacing the simple combination by the loglinear `R_p =
exp(r_p)` removes the artefact directly. See
`docs/handoff/HANDOFF_THEORY_PORTFOLIO_FOC_PATHOLOGY.md` for the original
diagnosis and `docs/handoff/HANDOFF_THEORY_REVIEW_CVC.md` for the
theoretical sign-off.

### 1.3 The portfolio FOC under CCV is the gradient of V

The portfolio decision under simple+clamp is governed by the asset-pricing
moment condition `FOC_j = E[μ_comb · (R_j − R_bill)] = 0` because
`∂R_p^simple/∂α_j = R_j − R_bill` is independent of α. Under CCV that
identity breaks: `R_p = exp(r_p)` depends on α through *both* `r_p` and the
shape of the variance-quadratic, so

```
∂R_p/∂α_j = R_p · ∂r_p/∂α_j                                    (2a)
∂r_p/∂α_s = log_x_s + σ²_xr·(½ − α_s) − α_b·σ_xrxb              (2b)
∂r_p/∂α_b = log_x_b + σ²_xb·(½ − α_b) − α_s·σ_xrxb              (2c)
```

The `½` in (2b–c) comes from differentiating the *combined* Jensen + Itô
quadratic `(½)α·σ² − (½)α²·σ² = (½)α(1−α)σ²` whose derivative is
`(½)(1−2α)σ² = σ²·(½ − α)`. (An earlier draft of the implementation handoff
had `(1 − α)` here from a Jensen-only term — that was a derivation slip,
caught and corrected before merge. The published handoff document still
shows the wrong form in §3.4; the code is correct.)

Because `R_j − R_bill` is itself constant in α, the FOC under CCV is

```
FOC_j(α) = E[μ_comb(c_{t+1}, sR_p) · R_p · ∂r_p/∂α_j]           (3)
```

This is *the gradient of value* with respect to α, not the
asset-pricing moment condition. It is mathematically what the simple+clamp
FOC reduces to once you remember `(R_j − R_bill) = ∂R_p^simple/∂α_j`.

The Newton Jacobian is the *Hessian* of V. By Schwarz's theorem it is
symmetric and has both an outer-product term (from the `μ'_comb · s ·
∂R_p/∂α` chain rule applied twice) and a Hessian-of-`r_p` term:

```
∂²V/∂α_j∂α_k = E[μ'_comb · s · ∂R_p/∂α_j · ∂R_p/∂α_k
              + μ_comb · R_p · (∂r_p/∂α_j·∂r_p/∂α_k − Σ_jk)]    (4)
```

The `−Σ_jk` correction comes from `∂²r_p/∂α_j∂α_k = −Σ_jk` (the variance-
quadratic). Under simple returns the analogue is the standard
`E[μ' · s · Rex_j · Rex_k]` outer product — there is no second-order term
because `r_p` is linear in α under that spec.

Both terms are visible in the kernel:
[lifecycle/solver.py:1015-1018](../lifecycle/solver.py#L1015-L1018) (retirement),
[lifecycle/solver.py:1239](../lifecycle/solver.py#L1239) (working-age),
[lifecycle/solver.py:1804](../lifecycle/solver.py#L1804) (terminal-shifted).

### 1.4 The consumption Euler is form-invariant

The consumption Euler

```
c_t^(−γ) = β · E[ψ · μ_alive(c_{t+1}) · R_p + (1−ψ) · μ_bequest(sR_p) · R_p]
        = β · E[μ_comb · R_p]
        = β · euler_sum
```

is *structurally unchanged* — only the formula for `R_p` differs. The
solver's EGM step inverts `c = (β · euler_sum)^(−1/γ)` exactly as before;
`euler_sum` is computed inside the FOC kernel and passed back. No EGM-side
edit is needed beyond making sure the kernel uses CCV `R_p`.

### 1.5 Range of validity and what changes economically

The CCV approximation is exact for log-normally distributed
`(r_bill, xr, xb)`. For our discretised VAR — Gauss-Hermite quadrature with
Lobatto tail nodes at ±7σ — the residual `O(|α|³σ⁴)` truncation is below
0.5% on the full unconstrained envelope `α ∈ [−6, +6]²`, and below 0.05% on
the constrained baseline `α ∈ [0, 1]²`. Within this envelope the spec
change is benign — the converged optimal stock and bond shares stay within
±3 pp / ±2 pp of simple+clamp at constrained baseline (the
pre-commitment gate of [scripts/_run_cvc_precommitment.py](../scripts/_run_cvc_precommitment.py)).

What *does* change in the unconstrained leveraged regime is exactly what
we want to change: the spurious "fear of bankruptcy" at extreme α gives
way to a smooth log-vol-drag that scales with `α'·Σ_xx·α` rather than
with `1{sR_p ≤ 0}`. This makes the Euler residual diagnostic informative
again.

---

## 2. Implementation surface

CCV is selected at runtime by setting
`SolverConfig.wealth_dynamics_spec = "ccv_log"`. The default of
`"simple_clamp"` keeps every legacy bundle and test in its existing
behaviour. Internally each FOC kernel and the simulator branch on a
boolean `use_ccv` derived once at solver entry; precomputed constants
(`σ²_xr`, `σ²_xb`, `σ_xrxb`) are threaded into every njit hot loop so the
branch test costs nothing.

### 2.1 Configuration plumbing

| File | Purpose |
|---|---|
| [lifecycle/model.py:195](../lifecycle/model.py#L195) | `SolverConfig.wealth_dynamics_spec: str = "ccv_log"` (the user-facing knob) |
| [lifecycle/precompute.py:230-237](../lifecycle/precompute.py#L230-L237) | `Precompute.sigma2_xr`, `sigma2_xb`, `sigma_xrxb` populated from `model.Sigma_r_cond[1:,1:]` |
| [lifecycle/solver.py:2105](../lifecycle/solver.py#L2105) | `use_ccv = (solver_config.wealth_dynamics_spec == "ccv_log")` |
| [lifecycle/solver.py:3849](../lifecycle/solver.py#L3849) | `run_lifecycle_solver` unpacks the same flag for the inner driver loop |
| [lifecycle/policy_io.py:145-167](../lifecycle/policy_io.py#L145-L167) | `save_policy_bundle` writes `metadata["wealth_dynamics_spec"]` so downstream consumers can detect CCV bundles without re-loading the config |
| [lifecycle/inf_horizon_solver.py:558](../lifecycle/inf_horizon_solver.py#L558) | Stability proxy `β·E[R_p^{1−γ}]` branches on the same flag |

`Sigma_r_cond` is the *conditional* covariance of `(rtb, xr, xb)` after
projecting out state innovations — `Sigma_rr − M·Sigma_sr` in `var.py`.
The CCV formula uses the conditional one, *not* the unconditional
`Sigma_rr`. The two differ by ~30–50% in the production VAR because state
predictability explains a meaningful share of return variance; using the
unconditional matrix would mis-state the Itô vol-drag by the same factor.

### 2.2 The five FOC kernels

Every closed-form FOC + Jacobian site in the solver carries both branches
behind `if use_ccv`. The CCV branch uses formulas (1)–(4) above; the
simple+clamp branch is the unchanged legacy path.

| Kernel | Lifecycle role | Location |
|---|---|---|
| Retirement FOC | inner integrand of the ages 67–98 backward sweep | [lifecycle/solver.py:840, 930-1040](../lifecycle/solver.py#L840) |
| Working-age FOC | inner integrand of the ages 25–66 sweep, separate alive/bequest paths | [lifecycle/solver.py:1051, 1195-1320](../lifecycle/solver.py#L1051) |
| Shifted-bequest terminal | bequest motive at age 99 with annuity factor | [lifecycle/solver.py:1720-1810](../lifecycle/solver.py#L1720) |
| Unshifted CRRA terminal | dead code in production (canonical uses shifted bequest); kept for sweeps | [lifecycle/solver.py:1418-1482](../lifecycle/solver.py#L1418) |
| Inf-horizon stability proxy | one-period contraction bound, used by the optional inf-horizon solver | [lifecycle/inf_horizon_solver.py:544-605](../lifecycle/inf_horizon_solver.py#L544) |

The shifted-bequest terminal kernel
[lifecycle/solver.py:1729-1754](../lifecycle/solver.py#L1729-L1754) carries
the canonical docstring describing both branches side-by-side; that is the
one to read first if you are auditing the math.

### 2.3 Simulator parity

The simulator at [lifecycle/simulation.py:774-788](../lifecycle/simulation.py#L774-L788)
branches on the same flag, threaded in as a kwarg
`simulate_lifecycle(..., wealth_dynamics_spec=...)`. Consistency between
solver and simulator is mission-critical: if they disagree on `R_p` at any
quadrature node, every Euler-residual diagnostic becomes meaningless. The
test [tests/test_cvc_solver_sim_consistency.py](../tests/test_cvc_solver_sim_consistency.py)
locks this in by hand-computing both at the same `(state, α, shock)` and
asserting `|R_p_solver − R_p_sim| < 1e-10` on a parametrised grid of
inputs.

### 2.4 Bundle tagging and the refuse-to-mix invariant

`save_policy_bundle` writes the spec into `metadata.wealth_dynamics_spec`
at the top level. Downstream consumers read it explicitly; there is no
auto-detection through magic. Concretely:

- [lifecycle/policy_io.py:145-167](../lifecycle/policy_io.py#L145-L167) extracts the
  spec from `run_config.solver_config.wealth_dynamics_spec` and falls
  back to `"simple_clamp"` for legacy bundles that pre-date the field.
- [scripts/diagnostics/_diag_euler_errors.py:497-504](../scripts/diagnostics/_diag_euler_errors.py#L497-L504)
  reads it into `EulerBundleContext.use_ccv` so the Euler-residual reporter
  evaluates against the right `R_p`.
- [scripts/diagnostics/_diag_euler_errors.py:1445](../scripts/diagnostics/_diag_euler_errors.py#L1445)
  re-passes it into `simulate_lifecycle` when the diagnostic generates its
  own simulation paths.
- [tests/test_cvc_diagnostic_consistency.py:128-147](../tests/test_cvc_diagnostic_consistency.py#L128-L147)
  asserts the round-trip: a bundle saved with `ccv_log` re-loads with
  `meta["wealth_dynamics_spec"] == "ccv_log"`. A bundle saved with no
  config defaults to `"simple_clamp"`.

`simulate_lifecycle`'s default `wealth_dynamics_spec="ccv_log"` matches the
production solver default. Callers working with legacy simple+clamp bundles
must explicitly pass `wealth_dynamics_spec="simple_clamp"`. There is no
auto-detection in the simulator API itself; diagnostics that handle bundles
read the flag from metadata explicitly
(see [tests/test_cvc_diagnostic_consistency.py:164-176](../tests/test_cvc_diagnostic_consistency.py#L164-L176)).

### 2.5 The Euler-error diagnostic

`_diag_euler_errors.py` is the canonical Euler-residual reporter. It has
its own copy of the CCV `R_p` formula in `_compute_euler_sum_*_continuous`
(lines 725–800 retirement, 870–945 working-age) — the diagnostic does not
call into the solver kernel; it re-derives `R_p` from `Sigma_r_cond` and
the state-conditional means. This means there are two implementations of
the same formula in the repository (solver + diagnostic), and they must
stay in sync. [tests/test_cvc_diagnostic_consistency.py:35-86](../tests/test_cvc_diagnostic_consistency.py#L35-L86)
checks them at five representative `(α, shock)` points to a tolerance of
`1e-10` (extracted via `V_dot = mu_b · R_p` from the solver kernel and
recomputed by hand on the diagnostic side).

The same five-arg signature pattern propagates through the four other
helper-using diagnostics — `_diag_invalid_cells`, `_diag_gridpoint_ee`,
`_diag_split_rule_sanity`, `_diag_simpath_worst_cells` — each calling the
shared `_evaluate_age_errors(...)` helper, which received five new
positional arguments (`ret_nodes, sigma2_xr, sigma2_xb, sigma_xrxb,
use_ccv`) when CCV landed.

### 2.6 Tests

| File | Coverage |
|---|---|
| [tests/test_cvc_kernels.py](../tests/test_cvc_kernels.py) | Standalone formula checks: corner-allocation sanity (1.1 above), Jensen at α=0.5, Itô vol-drag at α=(0,3), correctness of (2b–c) gradient by finite-difference at random points |
| [tests/test_cvc_kernels_lifecycle.py](../tests/test_cvc_kernels_lifecycle.py) | The same FD-vs-analytic Jacobian check applied to each of the five kernels above |
| [tests/test_cvc_solver_sim_consistency.py](../tests/test_cvc_solver_sim_consistency.py) | Solver-vs-simulator `R_p` parity to 1e-10 |
| [tests/test_cvc_diagnostic_consistency.py](../tests/test_cvc_diagnostic_consistency.py) | Diagnostic-vs-solver `R_p` parity + bundle metadata round-trip + simulator default-safety |
| [scripts/_verify_ccv_pipeline.py](../scripts/_verify_ccv_pipeline.py) | End-to-end smoke at `(3,3,3)` state, `n_z=3`, `n_w=20`: solve → save → load → run `_diag_euler_errors` from CLI. The test that catches signature regressions before AWS does. |
| [scripts/_run_cvc_precommitment.py](../scripts/_run_cvc_precommitment.py) | The pre-commitment side-by-side: solve under both specs, report max age-by-age difference in α_s, α_b. Gate is ±3 pp stock, ±2 pp bond at constrained baseline |

---

## 3. Default-flip status (May 2026)

CCV is now the default at both decision points:

1. **`SolverConfig.wealth_dynamics_spec`** at
   [lifecycle/model.py:195](../lifecycle/model.py#L195) — default flipped
   from `"simple_clamp"` to `"ccv_log"`. Every config that does not
   override the field automatically picks up CCV.
2. **`simulate_lifecycle(..., wealth_dynamics_spec=...)`** at
   [lifecycle/simulation.py:946](../lifecycle/simulation.py#L946) —
   default flipped from `"simple_clamp"` to `"ccv_log"`.
3. **`policy_io.py` legacy fallback** at
   [lifecycle/policy_io.py:149](../lifecycle/policy_io.py#L149) — kept at
   `"simple_clamp"` deliberately. Untagged legacy bundles continue to be
   interpreted as simple+clamp, preserving the meaning of historical
   bundles on disk. Only newly-saved bundles carry an explicit
   `metadata["wealth_dynamics_spec"]` tag.

To run simple+clamp now, callers must opt in explicitly:

```python
# Solver:
sc = CANONICAL_SOLVER._replace(wealth_dynamics_spec="simple_clamp")

# Simulator:
sim = simulate_lifecycle(..., wealth_dynamics_spec="simple_clamp")
```

This is exactly what [scripts/_run_cvc_precommitment.py](../scripts/_run_cvc_precommitment.py)
does for the side-by-side regression check.

### 3.2 What "remove the bankruptcy clamp" should mean

There are *three things* often grouped under "the bankruptcy clamp" and
they have different fates.

**(a) The literal `max(s·R_p, 0)` clamp.** This appears at
- [lifecycle/simulation.py:788](../lifecycle/simulation.py#L788) (the
  simulator's else-branch),
- inside the simple+clamp branches of every FOC kernel (the
  `if sR_p > 0` indicator).

These are **intrinsic to the simple+clamp spec** — they do not exist
under CCV because `exp() > 0` strictly. They are dead code only if you
also remove the `simple_clamp` branch. **My recommendation: keep them
gated behind `if not use_ccv`.** The cost is a few branches in code that
never executes in production; the value is that a future user investigating
"how did the simple-return spec fail?" can run the legacy branch as a
control. Removing it would also force a chained removal of every
config still pointing at simple+clamp (sweep cells, hypothesis tests,
historical comparison plots).

**(b) `min_return_power=1e-15` floor in `compute_terminal_portfolio_foc_jac`.**
[lifecycle/solver.py:1422, 1470-1471, 1499](../lifecycle/solver.py#L1422)
This is the "soft" version of the clamp — it appears only in the unshifted
CRRA terminal kernel, which is *not used by any production config* (the
canonical config uses shifted bequest). It also has not been wired up with
a `use_ccv` branch. Three options:
1. **Wire `use_ccv` into `compute_terminal_portfolio_foc_jac` for parity.**
   Mechanical extension of the same pattern used in
   `compute_terminal_foc_jac_shifted`. Probably half a day's work and
   one round of FD-Jacobian tests.
2. **Delete the unshifted kernel entirely** if you are committed to the
   shifted bequest specification. This removes the dead branch and the
   `min_return_power` floor along with it. Cleanest but irreversible.
3. **Leave it.** It is dead code in production; the floor is harmless.

I'd lean to option 2 (delete) if you have already decided shifted bequest
is the production utility. If you are not 100% sure, option 1 is the safe
bet.

**(c) The defensive `np.maximum(W, 0.0)` inside `bequest_utility` and
`bequest_marginal`** (`lifecycle/model.py` near the bequest helpers).
Under CCV this is a no-op because `W = sR_p > 0` always. **Keep it.** It
costs nothing in the hot loop and serves as a guard against interpolation
overshoots in rare edge cases (e.g., if a user supplies an off-grid
extrapolation routine that produces transient negative values). The
docstring at the relevant solver helper [lifecycle/solver.py:271-288](../lifecycle/solver.py#L271-L288)
already notes the role-shift.

### 3.3 Things I would *not* couple to this change

The handoff document calls these out as separate decisions; I agree.

- **Tightening the leverage cap from ±6 to ±4.** The CCV truncation grows
  with `|α|³`, so a tighter cap reduces approximation error. The theory
  review recommends ±4 at γ=5. Do this in a follow-up PR with its own
  experiment, not as part of "make CCV default."
- **Recalibrating β, γ, b̄, ρ.** No evidence the parameters need to move;
  the constrained-baseline pre-commitment gate is well within tolerance.
- **Removing the simple+clamp branch from solver and simulator.** Keep
  it. The cost is low and the option value is real.
- **Re-solving the historical bundle library.** Only do this if you need
  a side-by-side simple+clamp/CCV comparison in the paper. Otherwise
  leave the old bundles tagged and dormant.

### 3.4 Hot-loop optimisation that landed with the flip

A side-effect of the flip audit was discovering three dead `exp()` calls
per quadrature node in the CCV branch of each FOC kernel. The CCV branch
was computing `R_bill = exp(log_R_bill)`, `R_s = exp(log_R_bill +
log_x_s)`, `R_b = exp(log_R_bill + log_x_b)` — but `R_bill`, `R_s`, `R_b`,
`Rex_s`, `Rex_b` are *only* used by the simple+clamp asset-pricing FOC. The
CCV gradient FOC uses `dr_p/dα`, never the gross returns. Those three
exponentials are dead under CCV.

The dead block was moved into the simple+clamp `else` branch in three
kernels:

- Retirement FOC: [lifecycle/solver.py:930-960](../lifecycle/solver.py#L930-L960)
- Working-age FOC: [lifecycle/solver.py:1195-1225](../lifecycle/solver.py#L1195-L1225)
- Terminal-shifted: [lifecycle/solver.py:1771-1815](../lifecycle/solver.py#L1771-L1815)
  (skips two dead multiplications rather than three exps)

Per FOC kernel call this saves `3 × n_state_quad × n_ret_quad`
exponentials (roughly 105 per call at canonical 7×5 quadrature). With
three Newton iterations per EGM step, ~150 000 EGM steps per age, and 78
ages, that's a few hundred million `exp()` calls eliminated per full
solve. Walltime improvement is implementation-dependent (numba compiles
`exp` to a SIMD-vectorisable branch and the inner loop is memory-bound on
the wealth-grid interpolation, not transcendental-bound) but it can never
hurt and the change is clean.

Verified by all 40 CCV regression tests passing post-edit
(`tests/test_cvc_*.py`).

---

## 4. Open follow-ups (not part of "make CCV default")

These remain on the road map regardless of the default flip.

- **Tighten leverage cap to ±4 at γ=5** (theory review §6). Separate
  experiment, separate PR.
- **Wire `use_ccv` through `compute_terminal_portfolio_foc_jac`** *or*
  delete the unshifted CRRA terminal kernel. See §3.2(b) above.
- **Repurpose the obsolete diagnostics.** `_diag_arbitrage_quadsweep`,
  `_diag_quadrature_cloud`, `_diag_invalid_cells` are written against the
  bankruptcy boundary. Under CCV their detection criteria don't fire.
  Either delete or convert each to test the CCV truncation
  `O(|α|³·σ⁴)` instead of `R_p ≤ 0`. The implementation handoff §5.8 has
  the per-file disposition.
- **Wealth-grid lower edge audit.** The canonical CCV config sets
  `wealth_min = 0.01` (down from 0.13 under simple+clamp) to handle the
  fact that under CCV `s·R_p` can drop to ~0.3·s at extreme leverage
  + tail shock. The current value passes the `wealth_grid[0]/wealth_grid[1]
  ≤ 0.1` check; revisit if you change the leverage cap or quadrature
  envelope.

---

## Appendix A — formula reference card

```
σ²_xr  = pc.sigma2_xr   = Sigma_r_cond[1, 1]
σ²_xb  = pc.sigma2_xb   = Sigma_r_cond[2, 2]
σ_xrxb = pc.sigma_xrxb  = Sigma_r_cond[1, 2]

# At quadrature node (k_v, k_r):
log_R_bill = mu_r_bill   + ret_nodes[k_r, 0]
log_x_s    = mu_r_stock  + ret_nodes[k_r, 1]
log_x_b    = mu_r_bond   + ret_nodes[k_r, 2]

# CCV log portfolio return (formula 1)
r_p = (log_R_bill
       + α_s · log_x_s + α_b · log_x_b
       + 0.5 · (α_s · σ²_xr + α_b · σ²_xb)
       − 0.5 · (α_s² · σ²_xr + 2·α_s·α_b·σ_xrxb + α_b² · σ²_xb))

R_p = exp(r_p)

# Gradient of r_p (formulas 2b, 2c — note the ½, not 1)
dr_p/dα_s = log_x_s + σ²_xr · (½ − α_s) − α_b · σ_xrxb
dr_p/dα_b = log_x_b + σ²_xb · (½ − α_b) − α_s · σ_xrxb

dR_p/dα_s = R_p · dr_p/dα_s
dR_p/dα_b = R_p · dr_p/dα_b

# Wealth transition (unconditional under CCV — no clamp)
sR_p   = s · R_p          # > 0 strictly, exp() > 0
x_next = sR_p + π          # π = pension or labour income next period

# Hessian-of-V Jacobian (formula 4)
∂²V/∂α_j∂α_k = E[μ'_comb · s · ∂R_p/∂α_j · ∂R_p/∂α_k
              + μ_comb  · R_p · (∂r_p/∂α_j · ∂r_p/∂α_k − Σ_jk)]

# Consumption Euler (form-invariant)
c_t^(−γ) = β · E[μ_comb(c_{t+1}, sR_p) · R_p]
```

End of document.

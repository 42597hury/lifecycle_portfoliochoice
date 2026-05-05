# HANDOFF — Verify CCV log-wealth implementation is theoretically faithful

**Status:** open. Verification, not implementation. The CCV branch landed
in May 2026 and was made the production default; this handoff exists
because the codebase has now committed to CCV economically (paper figures
will be generated under it) and we need an independent theoretical-numerical
audit before any results are written up.

**Mission — read this twice.** Your job is to verify that the CCV return
implementation in this repo *theoretically corresponds* to the
Campbell–Viceira–Chen log-portfolio-return derivation, AND that the
numerical behaviour of the code matches what the theory predicts at the
calibrated VAR. **This is NOT a code review** — assume the code runs
without bugs and the test suite passes (40/40 CCV regression tests
green). Your job is to verify that the *math embedded in the code* is the
*math of the published paper*, and that the *numerical output* of the
code is consistent with that math given the calibrated state.

**Stress this hard:** every claim about what the code "does" should be
defended either by re-derivation from CCV's paper or by a numerical
experiment you ran. Do not take the existing tests as authority — they
encode the same formulas you are auditing. If the formulas are wrong,
the tests are wrong in the same direction. You must come at the math
fresh.

**Scope:** verify (1) every term of CCV w8566 eq.10 maps onto a specific
line of the code with the right sign and coefficient, (2) the conditional
covariance is the right one (not the unconditional), (3) the simulated
distribution of `R_p^CCV` matches the theoretical distribution implied by
the calibrated VAR, (4) the gradient-of-V FOC and symmetric Hessian
construction are correct, (5) the approximation envelope `O(|α|³ σ⁴)`
behaves as advertised across the policy support. Do not re-implement, do
not refactor, do not propose code edits unless you find a bug.

---

## 1. Sources to read in this order

1. **CCV (Campbell, Chacko, Rodriguez, Viceira) NBER w8566**, especially
   §3.1 ("The portfolio choice problem") and Appendix A
   ("Derivation of Equation (10)"). This is the published derivation. Every
   formula in our code must trace back to a line of this derivation.
2. **Campbell & Viceira (2002) book chapter 3** if available — it gives
   the same derivation with more discrete-time exposition.
3. **`docs/CCV_RETURNS.md`** — our internal documentation. Use it as a
   map, NOT as authority. If the doc and the paper disagree, the paper is
   right.
4. **`docs/handoff/IMPLEMENTATION_HANDOFF_CVC_RETURNS.md`** — the
   implementation handoff. Note its §3.4 derivation has a known error
   in the gradient (`(1 − α)` instead of `(½ − α)`); the code is correct,
   the handoff text is wrong. Do not be misled — re-derive yourself.
5. **`docs/handoff/HANDOFF_THEORY_REVIEW_CVC.md`** if it exists, and
   **`docs/handoff/HANDOFF_THEORY_PORTFOLIO_FOC_PATHOLOGY.md`** for the
   diagnosis that motivated the spec switch.
6. **The code:**
   - `lifecycle/solver.py:826-1033` (retirement FOC kernel)
   - `lifecycle/solver.py:1036-1330` (working-age FOC kernel)
   - `lifecycle/solver.py:1720-1827` (terminal-shifted FOC kernel)
   - `lifecycle/precompute.py:230-237` (`Sigma_r_cond` → `pc.sigma2_xr`,
     `sigma2_xb`, `sigma_xrxb`)
   - `lifecycle/simulation.py:760-790` (the simulator's CCV branch)
   - `lifecycle/inf_horizon_solver.py:544-605` (stability proxy)
   - `lifecycle/var.py:60-100` (where `Sigma_r_cond` is constructed from
     the VAR fit)
7. **Existing CCV tests** — read `tests/test_cvc_kernels.py`,
   `tests/test_cvc_solver_sim_consistency.py`,
   `tests/test_cvc_diagnostic_consistency.py` for the FD-Jacobian and
   solver-vs-simulator parity patterns. Reuse their fixtures where
   helpful.

---

## 2. Variable conventions in this codebase (so you read the code right)

The VAR has six variables partitioned as follows (see `lifecycle/var.py`
and `docs/RETURNS.md`):

| index | name | role | meaning |
|-------|------|------|---------|
| 0 | `y_1` | state | nominal 1-yr Treasury yield |
| 1 | `spr` | state | 20yr − 1yr yield spread |
| 2 | `cy` | state | log earnings yield (`−log(CAPE)`) |
| 3 | `rtb` | return | real bill log return = `log(1+y_1) − π` |
| 4 | `xr` | return | excess nominal stock log return = `log(R_s) − log(R_bill)` |
| 5 | `xb` | return | excess nominal bond log return = `log(R_b) − log(R_bill)` |

Within the return block (indices 3, 4, 5), the within-block indices are
`0, 1, 2`. The conditional covariance after projecting out state
innovations is `Sigma_r_cond` (3×3). CCV's `Σ_x` for equation (10) is
the bottom-right 2×2 block of this matrix:

```
σ²_xr  = Sigma_r_cond[1, 1]
σ²_xb  = Sigma_r_cond[2, 2]
σ_xrxb = Sigma_r_cond[1, 2]
```

These are exposed on `Precompute` as `pc.sigma2_xr`, `pc.sigma2_xb`,
`pc.sigma_xrxb`.

The realised log returns at a quadrature node `(k_v, k_r)` inside the
solver kernel are:

```python
log_R_bill = mu_r_bill + ret_nodes[k_r, 0]    # = log(R_bill,t+1)
log_x_s    = mu_r_stock + ret_nodes[k_r, 1]   # = xr,t+1
log_x_b    = mu_r_bond + ret_nodes[k_r, 2]    # = xb,t+1
```

where `mu_r_*` is the state-conditional mean (from the VAR's regression
of returns on state) and `ret_nodes[k_r, :]` is a Gauss–Hermite
quadrature draw from `N(0, Sigma_r_cond)`. So `log_R_bill, log_x_s,
log_x_b` are the realised values of `log(R_bill), xr, xb` at that node.

---

## 3. Verification tasks

For each task below, produce: (a) the theoretical statement, (b) the
code-line that allegedly implements it, (c) a numerical experiment that
either confirms or refutes the correspondence. Cap your report at one
page per task.

### Task A — Term-by-term map of CCV eq.10 onto the code

Re-derive CCV w8566 eq.10 from scratch. Write out the formula in the
paper's notation, then write out the code's formula in the code's notation,
then map every term. The expected formula in the paper is

```
r_p,t+1 = r_1,t+1 + α'·x_{t+1} + (½)·α'·σ²_x − (½)·α'·Σ_xx·α
```

(or the equivalent `(½)·α'·(σ²_x − Σ_xx·α)` factored form), where
`r_1,t+1` is the realised log riskless (bill) return, `x_{t+1}` is the
realised excess log return vector, `σ²_x = diag(Σ_xx)`, and `Σ_xx` is
the conditional covariance of `x`.

The code's formula is at e.g. `lifecycle/solver.py:935-940`:

```python
r_p = (log_R_bill
       + alpha_s * log_x_s + alpha_b * log_x_b
       + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
       - 0.5 * (alpha_s * alpha_s * sigma2_xr
                + 2.0 * alpha_s * alpha_b * sigma_xrxb
                + alpha_b * alpha_b * sigma2_xb))
```

**Check every coefficient.** Wrong sign, factor-of-2 missing, or an `α²`
where it should be `α(1−α)` will silently corrupt the policy and you will
see no test failure.

**Specific things to verify:**

A1. The Jensen term `+(½)·α'·σ²_x` — is the `½` correct? The published
derivation uses a 2nd-order Taylor expansion of `log E_t[1 + α'(R_x −
R_bill·1) + R_bill]` around the mean; the `½` arises from the
second-derivative term. Confirm by re-deriving.

A2. The Itô / vol-drag term `−(½)·α'·Σ_xx·α` — is the sign correct?
This is the curvature of `log` applied to the stochastic part of the
portfolio return.

A3. The bill term enters *as* the realised log bill return, not the
expected log bill return. Is that what the code does? (Confirm by
checking that `mu_r_bill + ret_nodes[k_r, 0]` is the realised, not the
mean.)

A4. The bill is *stochastic* in this model (because `rtb = log(1+y_1) − π`
has uncertain inflation). CCV's derivation allows the bill to be
stochastic — confirm that none of the algebra requires the bill diffusion
to be zero. Cross-check Appendix A.

**Failure mode to flag:** anywhere the code uses `Sigma_rr` (3×3
unconditional) instead of `Sigma_r_cond` (3×3 conditional) the Itô
correction is wrong by 30–50% (because state predictability explains a
meaningful share of return variance). Trace `pc.sigma2_xr` etc. back
through `lifecycle/precompute.py:230-237` and `lifecycle/var.py` to make
sure the matrix entering the formula is the conditional one.

### Task B — Numerical match: CCV r_p vs Monte-Carlo simple-return r_p

The CCV approximation claims that, *in expectation under the conditional
return distribution*, the log of the simple portfolio return is well
approximated by `r_p^CCV`. Verify this by Monte Carlo at a representative
state.

**Procedure:**

1. Build a model and `Precompute` from `configs/run_canonical_ccv.py`
   (the production CCV config).
2. Pick a state `s_*` at the unconditional mean (or the median of the
   stationary distribution — your call, document it).
3. Compute the state-conditional return mean
   `μ_r_*  = const_r + A_r · s_*`
4. For a grid of `α` values spanning the policy support — at minimum
   `α ∈ {(0,0), (0.5, 0), (0, 0.5), (1, 0), (0, 1), (1, 1), (2, −1),
   (−1, 2), (3, 0), (0, 3), (3, 3), (−2, 0), (0, −2)}`
   plus a half-dozen of your own choosing inside the `[−6, 6]²` box —
   do the following:
   - Draw `N = 1,000,000` realisations of `(rtb, xr, xb)` from
     `N(μ_r_*, Sigma_r_cond)`.
   - Form `R_s, R_b, R_bill` from those draws and the simple combination
     `R_p^simple_n = α_s·R_s_n + α_b·R_b_n + (1−α_s−α_b)·R_bill_n`.
   - Form `r_p^CCV` analytically using the code's formula at the *mean*
     of the draws (i.e., evaluate at the conditional mean state, no
     stochastic node).
   - Compute `MC_log_E_R = log(mean_n R_p^simple_n)` (the empirical log
     expected gross return).
   - Compute `r_p^CCV` evaluated at the *mean* shock (`xr = 0`,
     `xb = 0`) plus the conditional means: this gives `E_t[r_p]^CCV`,
     the model's prediction for the conditional expected log return.
   - Then `log E_t[R_p]^CCV = E_t[r_p] + (½)·Var_t[r_p]` (lognormal
     identity); compute `Var_t[r_p]` from `Σ_xx` and `α`.
   - Compare `log E_t[R_p]^CCV` to `MC_log_E_R`.

**Expected behaviour.** The CCV truncation is `O(|α|³ σ⁴)`; you should
see:
- At `α = (0, 0)`: zero error to floating-point precision.
- At constrained allocations (`α ∈ [0, 1]²`): error below 5 bps.
- At `|α| = 4`: error roughly 10–50 bps depending on direction.
- At `|α| = 6`: error up to a few percent in the direction that loads
  on the bond's variance.

If you see *systematic* error in one direction (e.g., always
underestimating `log E[R_p]`), that is a signal that the implementation
sign is wrong. If errors blow up below `|α| = 2`, escalate.

**What to plot / report:**
- Table of `(α, MC log E R_p, CCV log E R_p, error_bps)` across the grid.
- Scaling test: log–log plot of `|error|` vs `|α|`, should have slope ≈ 3
  in the leveraged regime.

### Task C — Sanity limits at corner allocations

Confirm the CCV formula reduces to the right thing at corner α:

C1. `α = (1, 0)` → `r_p = r_bill + xr` exactly (Jensen and Itô cancel).

C2. `α = (0, 1)` → `r_p = r_bill + xb` exactly.

C3. `α = (0, 0)` → `r_p = r_bill` exactly.

C4. `α = (½, 0)` → Jensen lift is `+0.125·σ²_xr`. At production
calibration this is roughly `+0.125 · 0.025 ≈ 31 bps`. Verify the code
reproduces the lift to floating-point precision.

C5. `α = (0, 3)` → vol-drag is `−3·σ²_xb`. At production calibration
this is roughly `−3 · 0.005 = −150 bps`. Verify.

These are exact identities and should hold to 1e-12 or better. If any
fails, the formula has the wrong coefficient.

### Task D — Σ_xx is the *conditional* covariance, not the unconditional

This is the single most likely source of a silent bug. The CCV paper uses
the conditional covariance of `(xr, xb)` given the state. Our code reads
from `Sigma_r_cond` which is constructed at `lifecycle/var.py` as
`Sigma_rr − M·Sigma_sr·M.T` (a Gauss-Markov projection, the "residual
covariance"). The *unconditional* covariance is `Sigma_rr` (the raw
return covariance estimated from the data).

D1. Print the two matrices side-by-side at the calibrated VAR. Compute
the percentage difference for each entry of the bottom-right 2×2 block.
The answer should be in the 30–50% range (state predictability is real
in this VAR).

D2. Trace the population path of `pc.sigma2_xr` etc. to confirm it goes
through `Sigma_r_cond[1:, 1:]`, NOT `Sigma_rr[1:, 1:]`. Cite the line
numbers.

D3. Construct a counterfactual: what would the optimal stock share at
γ = 5, constrained, age 67, mean-state look like under both choices?
Solve the model both ways (one quick way: hand-edit `pc.sigma2_*` to the
unconditional values and re-solve a thin slice). The expected difference
in `α_s` is on the order of 5–10 percentage points. If it's much smaller
than that, your `Sigma_rr` and `Sigma_r_cond` are too close — investigate.

### Task E — Approximation envelope: where does CCV start to fail?

This task answers "for what α does the 2nd-order CCV expansion become
inaccurate?" The MC comparison in Task B gives you the data; in this task
you stress the implications.

E1. From the simulated lifecycle (`scripts/run_solve.py` output for the
production bundle, or run a fresh smoke), report the empirical
distribution of `(α_s, α_b)` at each retirement age. What is the maximum
`|α|` the agent ever optimally chooses?

E2. From Task B, what is the CCV approximation error at that maximum
`|α|`?

E3. Is the approximation accurate *over the support of the converged
policy*? If yes, CCV is well-justified for our calibration. If no — if
the agent is choosing α at which the MC error exceeds 1% — flag this
explicitly. The current leverage cap is `±6`; the theory review
recommends tightening to `±4`. Is `±4` enough? Is `±6` too much?

### Task F — FOC = gradient of V (NOT asset-pricing moment condition)

Under simple+clamp, the portfolio FOC was the asset-pricing moment
condition `E[μ_comb · (R_j − R_bill)] = 0`. Under CCV the code uses (see
`lifecycle/solver.py:1006-1018`)

```
foc_j = E[μ_comb · R_p · (∂r_p/∂α_j)]
```

with

```
∂r_p/∂α_s = log_x_s + σ²_xr · (½ − α_s) − α_b · σ_xrxb
∂r_p/∂α_b = log_x_b + σ²_xb · (½ − α_b) − α_s · σ_xrxb
```

**Verify:**

F1. The factor `(½ − α)` is correct. Re-derive it from `(½)α(1−α)σ² →
∂/∂α = (½)(1−2α)σ² = σ²·(½ − α)`. (The implementation handoff has the
wrong derivation; do this yourself from the formula in Task A.)

F2. At a converged policy `(α_s*, α_b*)` from a small CCV solve, compute
`V(s, α_s* + ε, α_b*)` and `V(s, α_s* − ε, α_b*)` for `ε = 1e-5`,
finite-difference to get `∂V/∂α_s`, and confirm it equals the FOC formula
to `1e-6`. Same for `α_b`.

F3. Confirm the FOC is *not* `E[μ_comb · (R_j − R_bill)]` under CCV. At a
random non-zero-FOC point, evaluate both formulas and confirm they
disagree.

### Task G — Hessian symmetry (Schwarz)

The Newton Jacobian is the Hessian of V, hence symmetric:
`∂²V/∂α_s∂α_b = ∂²V/∂α_b∂α_s`. The code at e.g.
`lifecycle/solver.py:1016-1018` constructs:

```
J_ss = jac · dRp_das² + wmu · R_p · (dr_da_s² − σ²_xr)
J_bb = jac · dRp_dab² + wmu · R_p · (dr_da_b² − σ²_xb)
J_sb = jac · dRp_das · dRp_dab + wmu · R_p · (dr_da_s · dr_da_b − σ_xrxb)
```

G1. Re-derive the `−Σ_jk` term. It comes from the second cross-derivative
of `r_p` with respect to α:
`∂²r_p/∂α_j∂α_k = −Σ_jk` (from the variance-quadratic). Confirm.

G2. Verify symmetry numerically: at a random α, finite-difference the FOC
to get `∂FOC_s/∂α_b` and `∂FOC_b/∂α_s` separately. They should agree to
1e-6.

G3. Confirm the analytic `J_sb` formula equals both finite differences.

### Task H — Solver / simulator / diagnostic mutual consistency

The repo has tests for this (see
`tests/test_cvc_solver_sim_consistency.py` and
`tests/test_cvc_diagnostic_consistency.py`). They check that the three
implementations of the CCV `R_p` formula (solver kernel, simulator, EE
diagnostic) all agree to 1e-10 at fixed `(state, α, shock)`.

H1. Re-confirm independently. Pick a random `(state, α, shock)`, evaluate
all three (solver: extract via `compute_terminal_foc_jac_shifted`'s
`V_dot` output; simulator: run a 1-step trajectory; diagnostic:
`_compute_euler_sum_*_continuous`). Report max absolute difference. Should
be ≤ 1e-10.

H2. The risk: if any of the three drifts away (e.g., a future PR adds a
correction term in one place but not the others), Euler residuals become
meaningless. Document the *test as the canary* — if you find any
implementation that does not have a test pinning it to the others,
recommend adding one.

### Task I — VAR fit consistency

This is a sanity check on the *empirical* side, not the theoretical side.
The CCV approximation is exact for log-normally distributed
`(rtb, xr, xb)`; our model assumes Gaussian VAR residuals. Confirm:

I1. Run `lifecycle/var.py`'s estimation routine. Report the residual
distribution diagnostics (skewness, excess kurtosis, Jarque–Bera test) on
the three return columns. Severe non-normality (e.g., excess kurtosis
> 3 on `xr`) means the lognormal assumption underlying CCV is itself
suspect — separate from anything the code does.

I2. Report `Sigma_r_cond` and confirm it is positive-definite (Cholesky
should succeed). Compare to the values used by `pc.sigma2_xr` etc.

I3. Sanity-check the magnitudes against published numbers. Annual `σ_xr`
for US equity should be roughly 16–17% (so `σ²_xr ≈ 0.025–0.03`). Annual
`σ_xb` for AAA-20yr should be roughly 6–8% (so `σ²_xb ≈ 0.004–0.006`).
Anything more than 2× off from these ballparks is a data / construction
bug.

### Task J — Anything else theoretical you find suspect

If during the audit you spot an issue not on this list — a missing
correction term, an inconsistent treatment of inflation between the
solver and the simulator, a place where the bequest formula is fed an
unguarded `R_p` that *could* in principle be zero, an off-by-one in the
quadrature node indexing — flag it with the same theory-vs-code-vs-numerics
template as the rest of the report. The list above is what we know to
ask; you may catch what we did not think to ask.

---

## 4. Output

Produce one markdown file `docs/handoff/REVIEW_CCV_THEORY_TO_CODE.md` (or
adjacent) with:

- **A one-line GO / NO-GO verdict at the top.** "GO" means every formula
  in the code traces to a published derivation, every numerical
  experiment matches theory within stated tolerance, and the conditional
  covariance is the right matrix. "NO-GO" means at least one of the
  above failed; explain.
- **Per-task section** (A–J) with:
  - Theoretical statement (in your notation, not the paper's, so that
    sign conventions are forced into the open).
  - Code lines verified.
  - Numerical experiment description and result.
  - Verdict: pass / fail / inconclusive.
- **A summary table** of all numerical comparisons and their
  tolerance vs. observed deviation.
- **An open-questions list.** If something is not resolved by your tests,
  say what would resolve it (more MC samples? a different state? access
  to CCV's original numerical code?).

---

## 5. Tools available

- **Tests:** `tests/test_cvc_kernels.py` has FD-vs-analytic Jacobian
  patterns at random α; copy and extend.
- **Pre-built fixtures:** `_small_disc_3d` and `_small_retired_base_config`
  in `tests/test_predictability_ablation.py` give you a 1-second solve
  for sanity checks.
- **`scripts/_verify_ccv_pipeline.py`** runs the full pipeline at smoke
  scale in 25 seconds; reuse for end-to-end sanity.
- **`scripts/_run_cvc_precommitment.py`** runs both specs side-by-side at
  a config of your choice; useful for the Task D counterfactual.
- **A production CCV bundle** lives at
  `saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire/`
  if you need realistic policy distributions for Task E.

---

## 6. What is NOT your job

- Don't propose code edits unless you find a bug. Verification first.
- Don't re-tune calibration. If `σ²_xr` is wrong by 10× because of a data
  bug, that goes in Task J, not as a fix.
- Don't extend the spec to higher-order corrections (CCV's eq.10 is the
  2nd-order; eq.11 has higher-order terms in some treatments). The
  question "is the implemented eq.10 correct" is the question; "should
  we use eq.11 instead" is a different paper.
- Don't change the simple+clamp branch. It is the regression backstop.
- Don't change the leverage cap, the bequest spec, or the VAR.

---

## 7. Red flags that warrant immediate escalation

1. **A coefficient sign disagrees** between the code and your derivation.
   This is the catastrophic case — the policy you have shipped to AWS
   would be optimising the wrong objective. Stop everything and write up
   the discrepancy.
2. **`Sigma_rr` is being used where `Sigma_r_cond` should be**. Any place
   in the code path of `pc.sigma2_*` that touches the unconditional
   matrix.
3. **The MC truncation error blows up below `|α| = 2`.** If the CCV
   approximation is unreliable inside the constrained region the entire
   spec choice is suspect.
4. **The FOC formula equals `E[μ_comb · (R_j − R_bill)]` somewhere under
   CCV.** That is the simple+clamp formula and would be a residual bug
   from the cutover.
5. **The Hessian is not symmetric** to the test tolerance. Either Schwarz
   is wrong (impossible) or the analytic Hessian formula is.

---

## 8. Acceptance criteria

You hand back a "GO" only if all of these pass:

- A1–A4: every term of CCV eq.10 maps to the code with right sign and
  coefficient.
- B: MC error scales as `O(|α|³)` with magnitude consistent with the
  truncation bound at production calibration.
- C1–C5: corner-allocation identities hold to 1e-12.
- D1–D3: `Sigma_r_cond` is the matrix used; the unconditional matrix is
  not used; counterfactual difference is non-trivial.
- E: the converged policy lives within the regime where CCV truncation
  is below 1% (or you flag where it doesn't).
- F1–F3: gradient FOC formula is correct, FD-verified, and not the
  asset-pricing moment condition.
- G1–G3: Hessian construction is correct, symmetric, FD-verified.
- H1–H2: solver / simulator / diagnostic agree to 1e-10.
- I1–I3: VAR residuals are roughly Gaussian, `Sigma_r_cond` is PSD, return
  variances are in the published ballpark.

If any test fires inconclusively, prefer "NO-GO with caveat" to "GO".

End of handoff.

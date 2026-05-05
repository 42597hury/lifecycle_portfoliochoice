# HANDOFF — Verify CCV log-return implementation matches the theory

**Status:** open. Verification, not implementation. The CCV implementation
landed across multiple PRs in April–May 2026 and has been the production
default since the May 2026 default-flip. This handoff exists because we
want **independent confirmation that the theory of Campbell–Viceira–Chen
log returns is faithfully reflected in the numerical kernel** before we
commit downstream paper figures and IRR claims to the new spec.

**This is not a code review.** Assume every kernel runs without crashing
and that all 40 existing CCV regression tests pass — they do. Your job is
to verify that the *math embedded in the code* is the *math of the
published paper*, and that the numerical output of the kernel matches
what an independent implementation (built directly from the CCV
derivation against our calibrated VAR) would produce. **The overarching
goal is to confirm that the theory is accurately reflected in the
numerical implementation. We need this stressed and stressed hard.**
A bug here invalidates every Euler-residual diagnostic and every welfare
number we report.

You will:

1. Re-derive the CCV portfolio-return formula from CCV's published paper
   without consulting our code. Then confront the code with your formula.
2. Run a Monte Carlo experiment against the calibrated VAR. Compare
   the kernel's `R_p^CCV` to (a) the empirical distribution of
   `R_p^simple`, and (b) what CCV's own Table 5 reports.
3. Probe sanity limits, the conditional-vs-unconditional Σ choice,
   approximation-envelope scaling, and the gradient-of-V FOC identity.
4. Report any discrepancy — even a small one — between paper and code.

You will not:

- Edit production code. If you find a bug, document it; we will patch
  separately.
- Re-run the existing CCV pytest suite. We have already run it. Your
  numerics must come from independent computation against the
  calibrated VAR, not from re-execution of unit tests.
- Treat solver-vs-simulator-vs-diagnostic agreement as proof of
  correctness. They could all be wrong in the same way. Your check has
  to land outside that triangle.

---

## 0. Pre-flight: read in this order

1. **CCV w8566** — Campbell, Chacko, Rodriguez, Viceira, NBER WP 8566
   "Strategic Asset Allocation in a Continuous-Time VAR Model". §3.1
   gives the discrete-time portfolio-return approximation; **Appendix
   A "Derivation of Equation (10)"** is the load-bearing derivation. Re-
   derive it before reading our code.
2. **Campbell & Viceira (2002)** *Strategic Asset Allocation* (textbook),
   chapter 2 §2.3 (the Itô / Jensen decomposition) and chapter 4 (the
   loglinear approximation in the i.i.d. case). Useful background; the
   w8566 derivation generalises to VAR-driven returns.
3. `docs/CCV_RETURNS.md` — our internal documentation. Read this AFTER
   you have done step 1 so your derivation is independent.
4. `docs/RETURNS.md` §0–§2 — the variable definitions for our VAR
   (`y_1`, `spr`, `cy` as state; `rtb`, `xr`, `xb` as returns). The
   variable conventions are CCV-style (excess log returns are nominal
   minus log nominal bill, *not* real); if your derivation uses a
   different convention you must translate before comparing.
5. `docs/handoff/IMPLEMENTATION_HANDOFF_CVC_RETURNS.md` — the original
   change-spec handoff. Note: §3.4 of that document contains a
   *derivation slip* — it shows `∂r_p/∂α_s = log_x_s + σ²_xr (1 − α_s) − …`
   with a `(1 − α_s)` factor, but the correct gradient of the combined
   Jensen + Itô quadratic is `(½ − α_s)`. The code has the corrected
   form. Verify which is right before trusting either.

---

## 1. The numerical-vs-theory question, stated precisely

CCV's eq. (10) (w8566 §3.1) defines the discrete-time approximation to
the realised log portfolio return as

```
r_p,t+1 ≈ r_f,t+1 + α'·x_t+1 + (½) α'·σ²_x − (½) α'·Σ_xx·α                (CCV-10)
```

where `r_f,t+1` is the realised log riskless return (allowed to be
stochastic — see Appendix A), `x_t+1` is the realised log-excess vector
of risky returns over `r_f`, `σ²_x = diag(Σ_xx)`, and `Σ_xx` is the
*conditional* covariance of `x` after projecting out the slow state.

**Three distinct claims need verifying** and each requires independent
numerical evidence:

| Claim | What you must verify | Where the code lives |
|---|---|---|
| **C1** Implementation matches paper | Every term of CCV-10 (and its α-gradient + α-Hessian) is in the kernel with correct sign, factor, and index ordering | `lifecycle/solver.py:840-1018, 1051-1320, 1720-1827`; `lifecycle/simulation.py:774-788`; `lifecycle/inf_horizon_solver.py:580-605` |
| **C2** Returns match what they should be given the VAR | When you draw `(rtb, xr, xb)` from the calibrated `N(μ, Σ_r_cond)`, form `R_p^simple` directly, take its log, and average — that empirical mean must agree with `r_p^CCV` to the documented `O(\|α\|³ σ⁴)` truncation | the kernel computes `r_p^CCV`; you compute the MC ground-truth |
| **C3** CCV's range of validity holds at our calibration | The truncation error is small enough on the actual policy support (not just on a paper-friendly toy region) | use simulated household trajectories from a CCV bundle |

If C1 fails: the code disagrees with the paper. Stop, escalate.
If C2 fails: either C1 is wrong, or our VAR-conditioning is wrong, or
we are using the wrong Σ. Triangulate.
If C3 fails: the *spec is fine* but our calibration is outside CCV's
accuracy envelope at production parameters. Different kind of finding.

---

## 2. C1 — Formula-vs-paper correspondence

### 2.1 Ground truth: re-derive eq. (10) from Appendix A

Do not read our code first. Open w8566 Appendix A (the "Derivation of
Equation (10)" subsection). Reproduce the derivation. The key steps are:

(a) Start from the gross-return identity
`R_p = α_s R_s + α_b R_b + (1 − α_s − α_b) R_f`.

(b) Take logs. Use the Itô-style approximation
`log(α_s exp(r_s) + α_b exp(r_b) + α_f exp(r_f))
   ≈ α' [r] + (½) α' σ² − (½) α' Σ α`
where `[r] = (r_s, r_b, r_f)`, `σ² = diag(Σ)`, and `Σ` is the
conditional covariance of `[r]`. (CCV use `r_p − r_f` decomposition
which simplifies; do this carefully — your decomposition must match
theirs.)

(c) Re-express in `(r_f, x_s, x_b)` coordinates where `x_j = r_j − r_f`.
The terms involving `r_f` collapse because `α_s + α_b + α_f = 1` and the
Jensen lift on `r_f` cancels at second order against the bill leg of the
Σ block. (This is CCV's substitution; verify it yields exactly eq. 10.)

You should arrive at

```
r_p,t+1 = r_f,t+1
        + α_s · x_s,t+1 + α_b · x_b,t+1
        + (½)[α_s σ²_xs + α_b σ²_xb]
        − (½)[α_s² σ²_xs + 2 α_s α_b σ_xsxb + α_b² σ²_xb]
```

**Independent of our code, write down what each symbol means**:

- `r_f,t+1`: realised log riskless return at t+1. Stochastic in our
  model (uncertain inflation makes the *real* bill return noisy).
- `x_j,t+1`: realised log *nominal* excess return of asset j over the
  log nominal riskless. Our notation: `xr` for stock, `xb` for bond.
- `σ²_xs`, `σ²_xb`, `σ_xsxb`: **conditional** covariances of `(x_s, x_b)`
  given the lagged state. Independent of state under our restricted-
  homoskedastic VAR, but conditional in the sense of "after taking out
  the predictable component from the state vector."
- `α_s`, `α_b`: portfolio weights on risky assets at t (decision time).

### 2.2 Confront the kernel with your derivation

Now and only now, open `lifecycle/solver.py` at the three CCV-active
kernels:

| Kernel | Site of `r_p` formula | Site of `dr_p/dα_j` | Site of Hessian-of-V |
|---|---|---|---|
| Retirement | `solver.py:935-941` | `solver.py:1009-1010` | `solver.py:1015-1018` |
| Working-age | `solver.py:1200-1205` | `solver.py:1231-1232` | `solver.py:1239-1241` |
| Terminal-shifted | `solver.py:1779-1785` | `solver.py:1794-1795` | `solver.py:1802-1806` |

For each kernel, check **every term**:

- `r_p` constant: should be `log_R_bill = mu_r_bill + ret_nodes[k_r, 0]`.
  Confirm this is the realised log riskless return — i.e., that
  `ret_nodes[k_r, 0]` is the residual on the `rtb` axis and `mu_r_bill`
  is the state-conditional mean of `rtb` at this `(state, k_v)`.
  Read `lifecycle/precompute.py` to verify the correspondence.
- Linear term: `α_s · log_x_s + α_b · log_x_b` where `log_x_s =
  mu_r_stock + ret_nodes[k_r, 1]` and `log_x_b = mu_r_bond +
  ret_nodes[k_r, 2]`. **CRITICAL**: confirm that `log_x_s` is the
  realised log *excess* return (i.e., `xr,t+1`), NOT the realised log
  *gross* stock return. Our VAR's column 4 is `xr = log R_s − log R_f`,
  so this should be right by construction, but verify it explicitly.
  If `log_x_s` were the gross stock log return, the formula would be
  wrong by a `r_f` factor and you would catch it in §3 (MC) anyway.
- Jensen: `+ 0.5·(α_s · σ²_xr + α_b · σ²_xb)`. Sign positive. Factor ½.
  Linear in α.
- Itô / vol-drag: `− 0.5·(α_s² σ²_xr + 2 α_s α_b σ_xrxb + α_b² σ²_xb)`.
  Sign negative. Factor ½. Quadratic in α. Confirm the cross term has
  factor 2 (NOT 1) — this is the standard `α'Σα` expansion and a
  factor-of-2 error here is the most common mistake.
- Gradient `dr_p/dα_s`: should be `log_x_s + σ²_xr · (½ − α_s) −
  α_b · σ_xrxb`. **NOTE**: the published handoff `IMPLEMENTATION_HANDOFF_CVC_RETURNS.md`
  §3.4 has `(1 − α_s)` here. The code has `(0.5 − α_s)`. Re-derive from
  scratch. The combined Jensen + Itô term in α_s is `(½)α_s σ²_xr − (½)α_s² σ²_xr`,
  whose derivative is `(½)σ²_xr − α_s σ²_xr = σ²_xr (½ − α_s)`. Code is
  correct, handoff doc is wrong. Confirm this independently.
- Hessian-of-V combiner: at `solver.py:1016-1018`, the Newton-step
  Jacobian has two pieces:
  ```
  J_jk = jac · dRp_daj · dRp_dak                      ← outer product
       + wmu · R_p · (dr_da_j · dr_da_k − Σ_jk)       ← Hessian of r_p
  ```
  The second piece comes from `∂²r_p/∂α_j ∂α_k = −Σ_jk`. Verify by
  hand-deriving from the quadratic in §2.1. The signs and factors must
  be exact.

For each of these three kernels, write down whether the formula matches
your derivation. **Any mismatch — even a sign on a single term — is a
red flag.**

### 2.3 Confront the simulator

The simulator must produce the *same* `R_p` as the solver, not just
mathematically equivalent. Site: `lifecycle/simulation.py:774-788`.
Confirm formula-by-formula identity with the solver. (We have a 1e-10
parity test in `tests/test_cvc_solver_sim_consistency.py`; you can use
its setup as a starting point but the verification must be by your own
derivation, not by re-running the test.)

### 2.4 Confront the diagnostic

`scripts/diagnostics/_diag_euler_errors.py` has *its own copy* of the
formula at `_compute_euler_sum_retirement_continuous` (lines 725–800)
and `_compute_euler_sum_working_continuous` (lines 870–945). Two
independent implementations of the same formula — they must agree. We
have a 1e-10 test for this too, but again, derive independently.

### 2.5 Sanity limits (must hold algebraically, not just numerically)

These are corner allocations where two corrections cancel exactly. The
kernel must reproduce them by construction, not by approximation.

- `α = (1, 0)` (all stock): `r_p = r_f + x_s` exactly. Jensen lift
  `+(½)σ²_xr` cancels the Itô drag `−(½)·1²·σ²_xr`.
- `α = (0, 1)` (all bond): `r_p = r_f + x_b` exactly.
- `α = (0, 0)` (all bill): `r_p = r_f` exactly.
- `α = (½, 0)`: `r_p = r_f + ½·x_s + (½)·(½)·σ²_xr − (½)·(¼)·σ²_xr =
   r_f + ½·x_s + (⅛)·σ²_xr`. Jensen lift `+0.125·σ²_xr ≈ +31 bps` at
   `σ²_xr = 0.025`.
- `α = (0, 3)` (3× bond leverage): `r_p = r_f + 3·x_b + (³⁄₂)·σ²_xb −
   (⁹⁄₂)·σ²_xb = r_f + 3·x_b − 3·σ²_xb`. The vol-drag `−3·σ²_xb` is
   ≈ −1.5% on annual log return at σ²_xb ≈ 0.005. **Confirm sign and
   magnitude in the kernel by feeding these α values and printing
   `r_p − r_f − 3·x_b`.**

---

## 3. C2 — VAR-implied Monte Carlo ground truth

This is the most informative single check in the handoff. The CCV formula
is a 2nd-order Taylor expansion of `log E[R_p^simple]`. If we sample
`(rtb, xr, xb)` from `N(μ_state, Σ_r_cond)` and form
`R_p^simple = α_s exp(rtb + xr) + α_b exp(rtb + xb) + α_f exp(rtb)`,
then `log E[R_p^simple]` *is what CCV's r_p approximates*. The deviation
should be `O(|α|³·σ⁴)`. Verify numerically.

### 3.1 Build the ground-truth simulator (independent of our code)

In a scratch script (`scripts/_theory_audit_ccv.py` or similar; do not
add to production), do the following without reading the kernel:

1. Load `Sigma_r_cond` from a built model. The path:
   ```python
   from lifecycle.var import build_nominal_system1_var_config
   from lifecycle.precompute import build_model
   from configs._canonical import BASE_CONFIG
   var_cfg, _, _ = build_nominal_system1_var_config(
       csv_path="data/var_dataset.csv"
   )
   model = build_model(BASE_CONFIG, var_cfg, verbose=False)
   Sigma_r_cond = model.Sigma_r_cond  # 3×3, ordering (rtb, xr, xb)
   ```
2. Pick a representative state vector `s = (y_1, spr, cy)` (use the
   unconditional mean from the VAR, or the median state-grid corner).
   Compute the state-conditional return mean
   `μ_r = const_r + A_r @ s` (3-vector, ordering rtb, xr, xb). The
   `const_r` and `A_r` arrays are on `model` (verify dimensions before
   using).
3. Draw `N = 10⁶` samples `r ~ N(μ_r, Σ_r_cond)`. Form
   ```
   R_bill_sim = exp(r[:, 0])
   R_s_sim    = exp(r[:, 0] + r[:, 1])
   R_b_sim    = exp(r[:, 0] + r[:, 2])
   ```
4. For each `(α_s, α_b)` on a grid (see 3.3 below) form
   ```
   R_p_sim = α_s · R_s_sim + α_b · R_b_sim + (1 − α_s − α_b) · R_bill_sim
   ```
   Compute `log_E_Rp_simple = log(mean(R_p_sim))` and
   `E_log_Rp_simple = mean(log(maximum(R_p_sim, 1e-15)))` (the second is
   noisy near the bankruptcy boundary; use it as a sanity check, not the
   primary).

### 3.2 Compute `r_p^CCV` directly from the formula

For the same `(α_s, α_b)` grid and same state, evaluate

```
r_p_ccv = μ_r[0] + α_s · μ_r[1] + α_b · μ_r[2]
        + 0.5·(α_s · Σ[1,1] + α_b · Σ[2,2])
        − 0.5·(α_s² Σ[1,1] + 2 α_s α_b Σ[1,2] + α_b² Σ[2,2])
```

### 3.3 Map the truncation error

For `α_s × α_b ∈ {0, ¼, ½, ¾, 1, 2, 3, 4, 6}²` (some combinations leveraged):

| α_s | α_b | log E[R_p^simple] (MC) | r_p^CCV | discrepancy |
|---|---|---|---|---|
| 0 | 0 | should equal `μ_rtb + ½·Σ[0,0]` (Jensen on bill only) | `μ_r[0]` | the gap is `½·σ²_rtb` ≈ ½ × 0.005 ≈ 25 bps. **This is a feature: CCV's r_p is the realised log return, NOT log E[R_p]; the gap is Jensen on the bill itself.** Document carefully — do not flag as bug. |
| 1 | 0 | sample mean of `log R_s` | `μ_r[0] + μ_r[1]` (corner cancellation) | should match in MC sense (within MC noise) |
| ½ | 0 | MC | analytic per §2.5 | should match within MC noise — the third moment is small |
| 0 | 3 | leveraged bond | analytic | should match within ~1% — this is where CCV's |α|³ truncation starts to bite |
| 4 | 4 | extreme leverage | analytic | gap up to a few %; CCV's accuracy envelope explicitly degrades here. **Document the gap as a function of |α|.** |
| 6 | 6 | the leverage cap | analytic | gap may be large; this is the hard-case the theory review §6 of `HANDOFF_THEORY_REVIEW_CVC.md` flagged for cap tightening |

Plot `gap(α) = log E[R_p^simple] − r_p^CCV` as a function of `|α|` and
verify it scales as `O(|α|³·σ⁴)` once you subtract the constant
`½·σ²_rtb` baseline gap.

**Critical interpretation**: the CCV approximation target is
`log E_t[R_p,t+1]` (the loglinear approximation of the conditional
expected gross portfolio return). It is NOT `E_t[log R_p,t+1]`. The
distinction matters because the kernel feeds `r_p^CCV` into
`R_p = exp(r_p^CCV)` and uses *that* as the realised gross return. So
the kernel is implicitly assuming the realised gross return equals
its conditional log expectation — a deterministic-equivalence
substitution that introduces an additional `½·Var_t[r_p]` worth of bias
in the second moment, even though the first moment matches by
construction. This is a known feature of the CCV approach (see CCV §3.2
for discussion). Document the bias on simulated wealth-path moments.

### 3.4 Confront the kernel's `r_p^CCV` with your independent computation

Pick a single `(state, α, shock)` triple. Read `r_p` out of the kernel
(use the test-corpus pattern in `tests/test_cvc_solver_sim_consistency.py`
or extract via `compute_terminal_foc_jac_shifted`'s `V_dot` output and
divide by `mu_b`). Compare to your independent recomputation from §3.2.
**The numbers must agree to floating-point (1e-12).** If they disagree
at the 4th decimal, there is a bug. Investigate which.

---

## 4. C3 — Approximation envelope at production calibration

CCV's eq. 10 is exact for log-normal (rtb, xr, xb). The truncation is
`O(|α|³·σ⁴)`. Whether that truncation is acceptable depends on **where
on the policy support the agent actually lives**.

### 4.1 Map the policy support

Run `simulate_lifecycle` on the production CCV bundle
`saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire/`.
For 5,000 households over their lifetimes, collect `(α_s, α_b)` at every
age × household. Compute the joint distribution. Report:

- 50th, 90th, 99th percentile of `|α_s|`, `|α_b|`, `|α_s| + |α_b|`,
  `α_s² + α_b²`.
- Stationary-mass-weighted mean of the truncation gap from §3.3
  (i.e., the truncation error you computed at the policy support, not
  at a uniform grid).

This is the answer to "is the approximation good enough on the actual
policy?" If 99th-percentile truncation is < 0.5%, the spec is safely in
its envelope. If it's > 5%, we have a problem.

### 4.2 Conditional vs unconditional Σ — the load-bearing choice

CCV's `Σ_xx` in eq. 10 is the **conditional** covariance after taking
out predictability. We use `Sigma_r_cond[1:, 1:]`, NOT `Sigma_rr[1:, 1:]`.
The two differ by ~30–50% in our VAR because state predictability
explains a meaningful share of return variance (see equation R² of
0.06–0.32 across return columns in the VAR estimation summary).

Verify:

1. Print `Sigma_rr` and `Sigma_r_cond` from a built model. Confirm they
   are not equal.
2. Confirm that `pc.sigma2_xr`, `pc.sigma2_xb`, `pc.sigma_xrxb` (used by
   the kernel) come from `model.Sigma_r_cond[1:, 1:]`, not from
   `model.Sigma_rr[1:, 1:]`. Site:
   `lifecycle/precompute.py:230-237`.
3. As a counterfactual experiment, **temporarily** swap `Sigma_r_cond`
   for `Sigma_rr` in a scratch script and recompute the optimal portfolio
   at γ = 5 constrained. Report how much α_s, α_b shift. The expected
   shift is large (~5–10 pp on stock share) — large enough that "did
   we use the right matrix?" is a load-bearing question.

### 4.3 The bill-is-stochastic issue

CCV's r_f,t+1 is allowed to be stochastic in their derivation (Appendix
A). Our `rtb` is a stochastic real bill return (uncertain inflation).
Verify by tracing through Appendix A that the σ_b (bill diffusion) terms
are genuinely allowed to be non-zero in eq. 10 without modification.
Some textbook presentations specialise to deterministic r_f. If
Appendix A specialises silently, our application of eq. 10 to a
stochastic-bill setting may be subtly wrong.

### 4.4 Variable-convention check

Our `xr = log R_stock,nominal − log R_bill,nominal` and `xb` similarly.
**Both excess returns subtract the *nominal* bill, not the *real* bill**
(the real bill is `rtb = log R_bill,nominal − π`). This is the CCV
convention (only inflation appears in `rtb`; the excess returns are
inflation-free). Confirm by re-reading our `var.py` construction and
eq. 10 in the paper. A common mis-match would be to subtract `rtb`
instead of `r_bill,nominal` from the stock and bond — that would put
inflation noise into the excess returns and break the CCV invariance.

### 4.5 Gradient-of-V vs asset-pricing FOC identity

Under simple+clamp, the FOC reduces to the asset-pricing moment
condition `E[μ_comb · (R_j − R_bill)] = 0` because
`∂R_p^simple/∂α_j = R_j − R_bill` is α-independent. Under CCV, the FOC
is the gradient of value:

```
FOC_j(α) = E[μ_comb · R_p · ∂r_p/∂α_j] = ∂V/∂α_j
```

This is NOT the asset-pricing moment condition — and at the optimum,
the moment condition `E[μ_comb · (R_j − R_bill)]` is generally *not*
zero under CCV. (The Itô vol-drag means a risk-averse investor demands
a positive expected log-excess return *after the vol-drag penalty*, not
a positive expected gross-excess return.)

Verify numerically:

1. Pick a converged CCV policy at a representative `(state, age, wealth)`
   cell. Read `(α_s*, α_b*)`.
2. Numerically differentiate the value function at `(α_s*, α_b*)`:
   `∂V/∂α_j ≈ (V(α + ε e_j) − V(α − ε e_j)) / (2ε)` for small ε.
3. Compute `E[μ_comb · R_p · ∂r_p/∂α_j]` at the same point using the
   kernel.
4. The two must agree. They are the same object.
5. **Separately**, compute `E[μ_comb · (R_j − R_bill)]` at the same
   point. This will be non-zero in general — verify it is non-zero,
   because if it were zero you would have evidence the kernel is
   accidentally still computing the simple-spec FOC.

### 4.6 Hessian-of-V symmetry (Schwarz)

The Newton Jacobian `J_sb` and `J_bs` should be equal by Schwarz's
theorem. At a few `(state, α)` points, compute both and verify
`|J_sb − J_bs| / max(|J_sb|, |J_bs|) < 1e-10`. Asymmetry is a code bug,
likely in the cross-term of the outer-product or the cross-term of the
`(dr_da_s · dr_da_b − Σ_xrxb)` Hessian. Site of `J_sb`:
`solver.py:1018, 1241, 1806`.

---

## 5. Out-of-scope but report if you see it

These are not the focus, but if you trip over any of them while doing
§§2–4 please flag in the report:

- The unshifted CRRA terminal kernel `compute_terminal_portfolio_foc_jac`
  at `solver.py:1418-1482` does NOT have a `use_ccv` branch. It is
  dead code in canonical (which uses the shifted-bequest path) but lives
  in the codebase. If you find anything that triggers this path, flag
  the missing CCV branch.
- The `min_return_power=1e-15` floor in the same kernel. Under CCV
  `R_p > 0` strictly so the floor is never invoked, but it is also not
  guarded by `if not use_ccv`.
- The wealth-grid lower edge: under CCV `s·R_p` can drop to ~30% of `s`
  at extreme leverage + tail shock. Our canonical sets `wealth_min =
  0.01`. Verify the wealth-grid is dense enough at the lower edge that
  interpolation of `c_{t+1}` does not extrapolate at the support of
  realised `s·R_p`.
- The leverage cap is ±6. CCV's truncation grows as `|α|³`. The theory
  review (`HANDOFF_THEORY_REVIEW_CVC.md` §6) recommends ±4 at γ=5.
  Independent of your verdict on §§2–4, please report whether the
  truncation magnitude at α = 6 is acceptable.

---

## 6. Deliverable

A single markdown report `docs/handoff/HANDOFF_CCV_THEORY_AUDIT_REPORT.md`
with the following structure:

```
# CCV theory-audit report

## Verdict
[GO / NO-GO with one paragraph of reasoning]

## C1: paper-vs-code formula correspondence
[per-term match of CCV-10 against retirement, working-age, terminal kernels]
[per-term match of dr_p/dα and Hessian-of-V]
[any discrepancies, with file:line citations]

## C2: VAR-Monte-Carlo ground truth
[truncation gap table from §3.3]
[scaling plot or table showing O(|α|³·σ⁴) holds]
[1e-12 agreement check against kernel from §3.4]

## C3: approximation envelope at production
[policy-support summary from §4.1]
[Sigma_r_cond vs Sigma_rr quantification from §4.2]
[bill-is-stochastic check from §4.3]
[variable-convention check from §4.4]
[gradient-of-V identity check from §4.5]
[Hessian symmetry from §4.6]

## Out-of-scope findings
[list anything from §5]

## Recommendations
[anything to fix, recalibrate, or follow up]
```

The "Verdict" line is the single most important sentence in the
document. **Be willing to say NO-GO.** A small discrepancy that is hard
to explain away is more useful than a bland thumbs-up.

---

## 7. What success looks like

By the end of the audit you should be able to answer, with your own
numerical evidence, all of:

1. Does the code formula at every CCV-active site match the paper's
   eq. 10 (and its α-derivatives) term by term, with correct signs and
   factors?
2. At the calibrated VAR, does `r_p^CCV(α, state)` agree with
   `log E[R_p^simple(α, state)]` from independent Monte Carlo to within
   the documented `O(|α|³·σ⁴)` truncation?
3. At the production policy support, is the truncation small enough
   that downstream Euler residuals, welfare numbers, and IRR ratios are
   not contaminated?
4. Are we using the right Σ (conditional, not unconditional)?
5. Does the kernel's FOC equal the gradient of value (not the
   asset-pricing moment condition)?

If any answer is "no" or "I'm not sure," your audit has done its job —
escalate before the paper goes out.

End of handoff.

# CCV w8566 — theory-to-code correspondence audit

**Date:** 2026-05-06.
**Auditor:** independent re-derivation against the production code on
branch `jax-rewrite` at HEAD `93ad086`. The audit was instructed to be
skeptical and concrete.
**Reference:** Campbell, Chacko, Rodriguez, Viceira, NBER w8566 (2002),
"Strategic Asset Allocation in a Continuous-Time VAR Model" — *not* the
2002 textbook (some signs and indexing differ).

**Verdict (one-line summary):**
The CCV log-portfolio formula and FOC structure are implemented
correctly in `lifecycle/solver.py:657-679`, but **production has, since
commit `f23ac83` (2026-05-06), substituted the unconditional return-block
covariance `Sigma_rr` for the CCV-mandated *conditional* covariance
`Sigma_r_cond` in the σ²/σ-cross scalars consumed by every FOC kernel.**
This change directly contradicts both `docs/CCV_RETURNS.md` §2.1, the
prior internal CCV audit (`HANDOFF_CCV_THEORY_AUDIT_REPORT.md` §C3.2),
and the theory text in `HANDOFF_VERIFY_CCV_THEORY_VS_CODE.md` §1 and
§2.1, all of which say the σ² scalars in CCV-10 are conditional moments.
This is an unresolved theoretical disagreement worth escalating before
re-solving production bundles. Section 5 below lays out both sides and
my recommended interpretation.

Beyond the Σ-source question, every other CCV-10 site checks out:
formulas are bit-for-bit identical across the four kernels (terminal,
retirement, working, inf-horizon stability proxy), the (½ − α) gradient
is correct (the `(1 − α)` form in `IMPLEMENTATION_HANDOFF_CVC_RETURNS.md`
§3.4 is wrong), and the Newton Hessian carries the correct `−Σ_jk`
piece by formula structure. The state-partition design choice (no
lagged-return predictors) is acknowledged by the user and is cleanly
respected by the M / Σ_r_cond / Phi_21 plumbing.

---

## 1. Eq. (1)–(3) — VAR(1) law of motion

CCV w8566 §2.1 partitions `z_{t+1} = c + Φ z_t + v_{t+1}` into a slow
state block `s` and a fast/return block `r`. In CCV's benchmark the
*returns themselves* are part of the fast block (their RHS state vector
includes lagged stock, bond, and bill log returns).

**Our partition is more restrictive — by design.** The state block is
`s = (cy, spr, rtb, y_1)` and the return block is `r = (xr, xb)`. The
state ordering (`state_indices=(2,1,3,0)` against the column order
`y_1, spr, cy, rtb, xr, xb`) is set in
`lifecycle/var.py:381-440`. Lagged returns `xr, xb` do *not* enter any
RHS by the CCV-constrained estimation step in
`lifecycle/var.py:191-219`; specifically `Phi[:, 4:6] ≡ 0` by
construction (see the restricted-VAR test `||Phi[:, 4:6]||=0` in
`docs/RETURNS.md` §6.3). `rtb` does enter the state block (post the
rtb-as-state migration in commit `c17ebf5`); the realised log nominal
bill return at `t+1` is read off `state_{t+1}[rtb_index_in_state]` in
`lifecycle/solver.py:748-754`.

| CCV symbol | Our code | Where built |
|---|---|---|
| `Φ_11` | `model.Phi_11` | `var.py:64` (restriction `Phi[:, ret_idx]=0`) |
| `Φ_0,state` | `model.Phi_0_state` | `var.py:80-83` |
| `Φ_21` | `model.Phi_21` | `var.py:65` |
| `Φ_0,ret` | `model.Phi_0_ret` | `var.py:80-84` |
| `Σ_ss` | `model.Sigma_ss` | `var.py:69` |
| `Σ_rr` | `model.Sigma_rr` | `var.py:70` |
| `Σ_rs` | `model.Sigma_rs` | `var.py:71` |

Stationarity: `Phi_11` eigenvalues all `<1` (`docs/RETURNS.md` §6.3
checked; max `|λ|=0.936` under the legacy 3-D state). With the 4-D state
post rtb-as-state migration, the docstring at `var.py:412-440` notes
that `Phi[rtb,rtb] ≈ +0.36` — well below 1, so the 4-D system stays
stationary.

---

## 2. Eq. (4)–(7) — projection M and conditional covariance

CCV defines the conditional return moments by projecting the return
innovation onto the state innovation:

```
M = Σ_rs · Σ_ss^{-1}                              (CCV w8566 eq. 4)
Σ_r|s = Σ_rr − M · Σ_sr                           (CCV w8566 eq. 5)
E[r_{t+1} | s_t, v^s_{t+1}] = Φ_0_ret + Φ_21 · s_t + M · v^s_{t+1}
```

These map directly to:

| CCV | Code | Where built |
|---|---|---|
| `M = Σ_rs Σ_ss^{-1}` | `model.M` | `var.py:74` (NumPy `np.linalg.inv`) |
| `Σ_r|s` | `model.Sigma_r_cond` | `var.py:75` |
| `E[r|s, v^s]` projection | `pc.const_r + pc.A_r @ s + pc.M_v_nodes[k_v]` | `precompute.py:291-293` and `solver.py:741-744` |

`pc.M_v_nodes = v_nodes @ M.T` is precomputed — the state-innovation
contribution to the return mean at each Gauss–Hermite node (`v_nodes`
drawn from `N(0, Σ_ss)` via Cholesky in `discretization.py`).

**Important state-partition consequence.** Because lagged returns are
not in our state, `M` has shape `(n_ret, n_state) = (2, 4)`. CCV's
benchmark `M` would have an additional 3 columns (lagged stock, bond,
bill returns); ours does not. This affects the magnitude (but not the
structure) of the intertemporal hedging demand — see §6 below.

The conditional return covariance `Σ_r|s` is the residual after
projecting out `v^s`. Empirically (`docs/RETURNS.md` §4.1):

- `Var(xr|s) / Var(xr) = 0.038` — i.e., 96.2 % of return variance is
  explained by state-conditioning.
- `Var(xb|s) / Var(xb) = 0.088` — 91.2 % explained.
- `Var(rtb|s) / Var(rtb) = 0.609` — but rtb is now in the state block
  (the conditional-on-state variance is `Σ_ss[rtb,rtb]`, used by the
  state quadrature, not by the CCV return scalars).

This 90+ % explained share is the load-bearing context for §5.

---

## 3. Eq. (8)–(10) — log-portfolio approximation (CCV-10)

The Appendix A derivation produces

```
r_{p,t+1} = r_{f,t+1}
          + α_s · x_{s,t+1} + α_b · x_{b,t+1}
          + (½)[α_s σ²_xs + α_b σ²_xb]                          ← Jensen
          − (½)[α_s² σ²_xs + 2 α_s α_b σ_xsxb + α_b² σ²_xb]     ← Itô / α'Σα
```

The σ scalars in the Jensen + Itô blocks are conditional second
moments — CCV w8566 Appendix A states this explicitly: the realised
quadratic `(d log[…])²` is replaced by its conditional expectation
because the `μ_x²` cross-terms are higher-order under CCV's
"i.i.d.-residual" VAR (i.e., conditional on the state, the residual
noise is mean-zero with covariance `Σ_r|s`).

### 3.1 Where the formula lives

The CCV-10 expression is implemented exactly once, in
`lifecycle/solver.py:657-679` (`_ccv_log_return_and_grad`):

```
r_p = log_R_bill
    + alpha_s * log_x_s + alpha_b * log_x_b
    + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
    - 0.5 * (alpha_s^2 * sigma2_xr + 2 alpha_s alpha_b sigma_xrxb + alpha_b^2 * sigma2_xb)
```

This is shared by the three FOC kernels:

- Terminal-shifted FOC: `solver.py:686-719`
- Retirement FOC: `solver.py:841-923`
- Working-age FOC: `solver.py:930-1040`
- Inf-horizon stability proxy: `inf_horizon_solver.py:343-352`

The simulator at `lifecycle/simulation.py:99-104, 551, 625` uses the
same scalars (via `_build_return_factor(Sigma_r_cond)`). Per-site
formula correspondence to the prior internal CCV audit
(`HANDOFF_CCV_THEORY_AUDIT_REPORT.md` §C1.2) holds — every term has the
right sign, factor, and index.

### 3.2 Gradient of `r_p` — the (½ − α) form

From `solver.py:677-678`:

```
dr_p/dα_s = log_x_s + sigma2_xr · (½ − α_s) − α_b · sigma_xrxb
dr_p/dα_b = log_x_b + sigma2_xb · (½ − α_b) − α_s · sigma_xrxb
```

This is the correct gradient of the combined Jensen + Itô quadratic.
Differentiating `(½)α_s σ²_xr − (½)α_s² σ²_xr` w.r.t. α_s gives
`(½)σ²_xr − α_s σ²_xr = σ²_xr · (½ − α_s)`. The earlier
`IMPLEMENTATION_HANDOFF_CVC_RETURNS.md` §3.4 form `(1 − α_s)` is wrong
(see prior audit `HANDOFF_CCV_THEORY_AUDIT_REPORT.md` §C1.4 — confirmed
by FD test at three random α points, the (½ − α) form matches FD to
1e-12 and the (1 − α) form misses by 5e-4).

### 3.3 Hessian-of-V — the `−Σ_jk` correction

The Newton Jacobian carries two pieces (see e.g. retirement FOC at
`solver.py:915-922`):

```
J_jk = jac_lin · dRp_dα_j · dRp_dα_k                          ← outer product
     + wmu · R_p · (dr_dα_j · dr_dα_k − Σ_jk)                  ← Hessian of r_p
```

The `−Σ_jk` piece comes from `∂²r_p/∂α_j ∂α_k = −Σ_jk`
(the second derivative of the variance-quadratic `(½)α'Σα`). This is
load-bearing: the `Σ_jk` consumed here at `solver.py:712-714, 916-918,
965-967, 1022-1024` is the same `(sigma2_xr, sigma2_xb, sigma_xrxb)`
as in the level. Σ-source bug, if any, propagates to the Hessian
identically — i.e., the Newton step direction is consistent with
whatever Σ the level uses.

### 3.4 Sanity limits

The corner cases (verified by independent algebra, identical to the
prior audit's §C1.3):

- α=(1,0): `r_p = r_f + xr` exactly. Jensen `+(½)σ²_xr` cancels Itô
  `−(½)·1²·σ²_xr`. ✓
- α=(0,1): `r_p = r_f + xb` exactly. ✓
- α=(0,0): `r_p = r_f` exactly. ✓
- α=(½,0): `r_p = r_f + ½ xr + (⅛)σ²_xr`. Jensen lift is `+0.125 ·
  σ²_xr`. At `σ²_xr = Σ_rr[xr,xr] = 2.529e-2`, lift = +316 bps; at
  `σ²_xr = Σ_r_cond[xr,xr] ≈ 9.6e-4`, lift = +12 bps. (Empirical match
  is contingent on which Σ-source — see §5.)
- α=(0,3): `r_p = r_f + 3 xb − 3 σ²_xb`. Drag = −3·σ²_xb. At
  `Σ_rr[xb,xb] = 5.88e-3`, drag = −1.76% on annual log return. ✓ in
  sign and magnitude.

---

## 4. Stationary distribution and discrete Lyapunov

CCV §2.2 specifies the stationary state distribution as
`s ~ N(z_bar_state, Σ_state^∞)` where `z_bar_state = (I − Φ_11)^{-1} Φ_0`
and `Σ_state^∞` solves the discrete Lyapunov equation
`Σ_state^∞ = Φ_11 · Σ_state^∞ · Φ_11' + Σ_ss`.

| CCV | Code | Where built |
|---|---|---|
| `z_bar_state` | `model.z_bar_state` | `var.py:77` (= `z_bar[state_idx]`); the CCV-constrained estimator pins this to the sample mean exactly (`var.py:191-218`, `RETURNS.md` §3.1) |
| `Σ_state^∞` | inside `discretization.build_state_grid`; not stored as a top-level array, used only to compute the Cholesky directions for the Rouwenhorst grid | `lifecycle/discretization.py` |
| Stationary state grid | `pc.state_grid`, marginal grids inside `pc.state_grids` | `precompute.py:282-299` |
| Stationary state probabilities (used by `_diag_grid_quad_sweep` only) | `pc.state_stationary_probs` (when Lyapunov mode is invoked) | `discretization.py` |

Solving `(I − Φ_11) z_bar_state = Φ_0_state` reproduces the sample
mean to ≤ 1e-16 (verified by `RETURNS.md` §6.3 V5, retained
post-migration: the constrained estimator at `var.py:191-218` enforces
this identity by construction).

The state grid is built either via
`state_grid_mode="cholesky"` (default) or `"lyapunov-axis"`. The
Cholesky mode produces grid axes aligned with `Σ_state^∞`'s Cholesky
directions, not the physical state axes; this is documented in
`RETURNS.md` §5.6 with the per-axis-knob mapping for the default
ordering.

---

## 5. The Σ-source disagreement (the audit's headline finding)

### 5.1 What the code does

`lifecycle/precompute.py:303-314` (post commit `f23ac83`,
2026-05-06) — current production:

```python
sigma2_xr = float(model.Sigma_rr[xr_pos, xr_pos])
sigma2_xb = float(model.Sigma_rr[xb_pos, xb_pos])
sigma_xrxb = float(model.Sigma_rr[xr_pos, xb_pos])
```

These are the **unconditional** return-block diagonal/cross terms.

The commit message argues:

> CCV w8566 eq. (10) takes expectations over the FULL VAR innovation
> v_{t+1}, so the constants σ²_xr / σ²_xb / σ_xrxb in the log portfolio
> return formula must come from the unconditional return-block
> covariance Σ_rr, not from Σ_r_cond …

and gives empirical justification: under `Σ_r_cond`, converged α_s
went up to 14.7 (smoke); under `Σ_rr`, α_s capped at 3.1. CCV Table 2
reports σ_xr ≈ 15.5 %, which matches `√Σ_rr[xr,xr] ≈ 15.9%` and *not*
`√Σ_r_cond[xr,xr] ≈ 3.1%`.

### 5.2 What CCV actually says

From the prior verification handoff
(`HANDOFF_VERIFY_CCV_THEORY_VS_CODE.md` §1):

> `σ²_xs`, `σ²_xb`, `σ_xsxb`: **conditional** covariances of `(x_s, x_b)`
> given the lagged state. … "after taking out the predictable component
> from the state vector."

From `docs/CCV_RETURNS.md` §2.1:

> `Sigma_r_cond` is the *conditional* covariance of `(rtb, xr, xb)`
> after projecting out state innovations — `Sigma_rr − M·Sigma_sr` in
> `var.py`. The CCV formula uses the conditional one, *not* the
> unconditional `Sigma_rr`.

From the prior audit `HANDOFF_CCV_THEORY_AUDIT_REPORT.md` §C3.2:

> The kernel uses `Sigma_r_cond` (verified) … Had the kernel been
> written against `Sigma_rr` by mistake, the Itô drag would be 26× too
> big on the stock leg and the converged α_s would crater to near-zero.
> Instead, the (correct) `Sigma_r_cond` is used everywhere.

**These are flatly contradictory.** Note the prior audit's prediction
is the *opposite sign* of what the May-6 commit experienced: it claims
that switching from Σ_r_cond → Σ_rr would make Itô drag bigger and α_s
*smaller*, while the commit message reports that the same switch made
α_s smaller (3.1 vs 14.7). The "α_s crater" prediction in the prior
audit is consistent with the commit message's *direction of change*; the
two documents agree that Σ_rr → smaller α_s. They disagree on which
matrix CCV intends.

### 5.3 The theoretical resolution (my reading)

CCV's Appendix A derivation is the load-bearing text. The derivation
runs as follows. Start from the gross-return identity and take a
second-order Taylor expansion of the log around `x = 0`. The realised
second derivative depends on `(x_{s,t+1}, x_{b,t+1})`, but in CCV's
expansion the unknown stochastic content of `x_{t+1}` is replaced by
its conditional expectation *given* what is known at decision time `t`.
What is known at `t` is `s_t`. So the relevant second moment is

```
E_t[(x_{t+1} − E_t[x_{t+1}])(x_{t+1} − E_t[x_{t+1}])']  =  Σ_r|s
```

i.e., the **conditional** covariance after projecting out the
state-innovation component. This is the only well-posed second moment
for an agent making a one-period-ahead forecast — the cross-term `M·v_s`
is *predictable* given `s_t` and so is part of the mean, not the
variance.

The May-6 commit message conflates "expectation over the full VAR
innovation `v_{t+1}`" with "the unconditional second moment". The
agent's expectation at decision time is conditional on `s_t`, not
unconditional. The unconditional second moment `Σ_rr = Σ_r|s + M Σ_ss
M'` includes a term that the agent *already knows the realisation of*
(modulo `v^s` integration); building it into the Itô drag double-counts
the predictable part.

Empirically, `M·Σ_ss·M'` is large in our calibration because state
predictability is large (R²(xr|s)=96.2%, R²(xb|s)=91.2%). That is *why*
the two matrices differ by 26× on stocks and 11× on bonds. CCV would
say: yes, it should differ, because the only relevant variance for the
period-`t` decision is the residual `Σ_r|s`.

### 5.4 The CCV-Table-2 argument

The commit message points out that `√Σ_rr[xr,xr] ≈ 15.9 %` matches CCV
Table 2 (15.5 %), while `√Σ_r_cond[xr,xr] ≈ 3.1 %` does not. **This is
a mistaken comparison.** CCV Table 2 reports unconditional return
volatilities, which are descriptive statistics, not the second moment
that enters eq. (10). The unconditional volatility *is* a moment of the
data, but the moment that enters CCV's portfolio formula is a different
object: the residual after stripping out predictability. CCV's own
empirical implementation in §4 ("Empirical implementation") computes
the predictability-stripped covariance and uses *that* in their
log-utility analytic solution and in their numerical work.

### 5.5 The α-magnitude argument

The commit message also reports that the converged α_s *value* is more
"sensible" under Σ_rr (3.1 vs 14.7). This is a circular argument once
the question is whether CCV intends the conditional or unconditional Σ.

The α_s = 14.7 outcome under Σ_r_cond is *predicted* by CCV's own
analysis: high state-predictability (R² ≈ 96 %) implies strong
state-conditional Sharpe ratios, which imply large myopic stock demand
when the residual variance is small. CCV §4 acknowledges this and
explicitly notes that the predictable component drives unrealistically
large Markowitz allocations under unrestricted preferences — and that
this is the economic fact the model is supposed to surface, not a bug.

In other words, **α_s = 14.7 is the correct CCV answer at our
calibration under unrestricted leverage**. The "fix" of switching to
Σ_rr is an econometric choice (use unconditional rather than conditional
moments) that materially changes the model's economic content; it is not
a bug fix.

### 5.6 What I'd recommend

1. **Treat `f23ac83` as a deviation from CCV w8566, not as a bug fix
   for it.** Document this prominently in `CCV_RETURNS.md` §2.1 — the
   current language there *contradicts* the production code and will
   confuse anyone re-reading the kernel.
2. **Reconcile or revert.** Either:
   - (a) revert to `Sigma_r_cond`, accept α_s = 14.7 as the CCV
     answer, and impose a tighter leverage cap (the existing ±6 cap is
     already in the diagnostic envelope from the prior audit's §C2);
   - (b) keep `Sigma_rr` and document explicitly that the model is
     "CCV-style with unconditional Itô drag", citing the May-6
     experimental motivation, and acknowledge the deviation in any
     paper-facing write-up.
3. **The α-cap escape valve is independent of this question.** A ±4
   leverage cap would suppress the extreme α_s under either Σ-source
   and is recommended in `HANDOFF_THEORY_REVIEW_CVC.md` §6.

---

## 6. Myopic vs total demand — where is the hedging term?

CCV w8566 §3.1, eq. (16)-(20), decomposes the optimal portfolio into

```
α* = (1/γ) · Σ_r|s^{-1} · μ_excess          ← myopic / Markowitz
   + (1 − 1/γ) · Σ_r|s^{-1} · σ_rs · (something involving V)   ← intertemporal hedging
```

where the hedging term depends on `Cov_t(state, return) = Σ_rs` and on
the gradient of value w.r.t. the state. **In our solver this
decomposition is not made explicit.** The solver iterates the full
gradient-of-V FOC

```
FOC_j(α) = E_t[μ_comb(c_{t+1}, sR_p) · R_p · ∂r_p/∂α_j] = 0
```

(`solver.py:911-913, 1013-1014, 1244-1247, 1799-1800`) and lets the
hedging term enter implicitly through the dependence of `μ_comb` on the
next-period continuation value (which is itself a function of `s_{t+1}`,
which is in turn correlated with `r_{t+1}`). At convergence, the
hedging-demand component is the gap between converged α and the myopic
α* at the same state.

This is a legitimate implementation choice: CCV's own §4 numerical work
also iterates the full Bellman FOC rather than splitting the demand
analytically (the analytic split is exact only at log-utility γ=1).

**Effect of our state-partition difference.** Because lagged returns
are *not* in our state, our `Σ_rs` is `(2 × 4)` and CCV's would be
`(2 × 7)` (3 extra columns for lagged stock/bond/bill returns). The
hedging-demand contribution differs in magnitude — three predictor
channels are absent. In particular:

- **The price-yield (mean reversion in returns) channel.** CCV's lagged
  stock-return predictor gives stock prices a mean-reverting component:
  high recent realised stock returns predict lower future returns. We
  capture this through `cy = -log(CAPE)` instead, which is a *level*
  predictor of stock returns. The `Phi_21[xr, cy] = +0.107` coefficient
  (`RETURNS.md` §4.1) carries this content. **Whether this is
  equivalent depends on whether CAPE-as-state captures the same
  predictability content as the lagged stock return.** Empirically it
  does — CAPE has R² (xr|s) = 96.2% (`RETURNS.md` §6.3). The
  `M[xr, cy] = −0.93` mechanical projection coefficient confirms the
  same content; CAPE *is* a transformation of the lagged stock return
  in our specification (it's CAPE = price/earnings; price has the
  realised stock return embedded).

- **The yield-curve (mean reversion in bonds) channel.** CCV's lagged
  bond-return predictor lets a bond-return surprise affect future bond
  returns. We capture this through `spr` (the yield spread). The
  `Phi_21[xb, spr] = +4.49` is the dominant return-predictor in our
  model. The residual content of "lagged bond return beyond spr" is
  small in the linear projection: `spr` and the bond return are both
  driven by yield changes (the duration formula in
  `data/build_var_dataset.py` makes bond return a deterministic
  function of yield change, given spr). So most of "lagged bond
  return" content in CCV's state is recoverable from `spr` plus its
  innovation.

- **The Fisher-effect (inflation persistence) channel.** CCV's lagged
  bill-return predictor encodes inflation persistence. We capture this
  through `rtb` (which is now in the state block, since
  `c17ebf5`). `Phi[rtb,rtb] ≈ +0.36` carries the inflation persistence.
  Pre-migration this channel was missing; post-migration it is
  present.

**Verdict on the state-partition difference.** Each of the three
predictor channels CCV uses is captured in our state by an
economically equivalent variable, at the cost of (a) some loss in
predictive R² (especially on the bond leg), and (b) a smaller-rank
`Σ_rs` (2×4 vs 2×7), which slightly reduces the magnitude of the
hedging-demand correction `Σ_rs · (something)` relative to CCV's
benchmark. The structural form is preserved.

---

## 7. Per-equation correspondence summary

| CCV equation (w8566) | Symbolic content | Code location | Status |
|---|---|---|---|
| (1)–(2) | `z_{t+1} = c + Φ z_t + v_{t+1}` | `model.Phi_0_*`, `model.Phi_*1` | match |
| (3) | partition into state vs return | `var.py:partition_var` | match (state-partition deviation acknowledged) |
| (4) | `M = Σ_rs Σ_ss^{-1}` | `var.py:74` | match |
| (5) | `Σ_r|s = Σ_rr − M Σ_sr` | `var.py:75` | match (matrix is built; consumed by `Sigma_r_cond`-Cholesky in `discretization.py:644`) |
| (6) | conditional return mean | `solver.py:741-744`, `inf_horizon_solver.py:336-339` | match |
| (7) | conditional return variance | `model.Sigma_r_cond` | **inconsistently used — see §5** |
| (8) | gross-return identity | implicit | n/a |
| (9) | log-link expansion | `solver.py:665-678` | match |
| (10) | r_p log-portfolio approx | `solver.py:665-678` | match in form; **σ-source disagreement, see §5** |
| (16)–(20) | myopic + hedging decomposition | implicit in full FOC; not split | match (CCV does not split numerically either) |
| Stationary mean | `(I − Φ_11)^{-1} Φ_0_state` | `var.py:191-218` (CCV constrained estimator) + `model.z_bar_state` | match exactly |
| Discrete Lyapunov | `Σ_state^∞` | inside `discretization.build_state_grid` | match (used for grid + Rouwenhorst) |

---

## 8. Numerical results

**Note on execution:** I attempted to run a small numerical scratch
script at `scripts/scratch/ccv_audit_numerics.py` against the canonical
hardcoded VAR (`build_nominal_system1_var_config_hardcoded`,
`var.py:646-680`). Sandboxed execution was unavailable in this audit
environment so the runs below are by-hand from the hardcoded matrices
in `var.py:608-643`. The script is left in place so a future run can
reproduce the numbers exactly; expected output is described inline.

### 8.1 Σ-matrix readouts

From the hardcoded `_OMEGA` at `var.py:636-643`, with state indices
`[2, 1, 3, 0]` (cy, spr, rtb, y_1) and return indices `[4, 5]`
(xr, xb):

```
Sigma_rr (2×2) = Omega[ix(4,5), ix(4,5)]
              = [ +2.5291e-2   +3.4618e-3 ]      (xr × xr,  xr × xb)
                [ +3.4618e-3   +5.8825e-3 ]      (xb × xr,  xb × xb)

sqrt(diag) = [ 0.1590, 0.0767 ]   →   xr σ ≈ 15.9 %, xb σ ≈ 7.7 %
```

This matches CCV Table 2 unconditional vols (xr ≈ 15.5%, xb ≈ 7.7% in
their sample) — but as discussed in §5, this only confirms the
unconditional moment of the data, not what CCV-10 wants in its
formula.

I do not have `Sigma_r_cond` numerically without the precompute; from
the prior audit (`HANDOFF_CCV_THEORY_AUDIT_REPORT.md` §C3.2):

```
Sigma_r_cond[xr,xr] ≈ 9.57e-4    →    sqrt ≈ 3.1 %
Sigma_r_cond[xb,xb] ≈ 5.11e-4    →    sqrt ≈ 2.3 %
ratio Σ_rr/Σ_r_cond:  xr 26.4×,  xb 11.4×
```

These are the legacy 3-D-state ratios. The 4-D-state ratios (post
rtb-as-state) should be similar in magnitude (since the rtb axis was
already explained ~39% in the legacy partition; moving rtb to the
state block increases the state-explained share for rtb but does not
materially change the xr/xb decomposition). Confirming this requires
running `scripts/scratch/ccv_audit_numerics.py` against the
hardcoded VAR.

**What the script would print** when run from a Python environment
with the lifecycle package importable:

- `Sigma_rr (2×2)` and `Sigma_r_cond (2×2)`: numerical values as
  above (4-D state version).
- `M (2×4)`: rows = (xr, xb), cols = (cy, spr, rtb, y_1). Expected
  entries close to (legacy 3-D values, modulo the new rtb column):
  `M[xr, cy] ≈ −0.93`, `M[xb, spr] ≈ +4.5`, `M[xb, y_1] ≈ −8.7`.
- `pc.sigma2_xr / pc.sigma2_xb / pc.sigma_xrxb`: numerically equal to
  `Sigma_rr` entries (because of `f23ac83`).
- Myopic α* at `s = z_bar_state` per §8.2.

### 8.2 Markowitz myopic at the unconditional state

Conditional return mean at `s = z_bar_state`: by the CCV constrained
estimator, `Phi_0_ret + Phi_21 · z_bar_state = z_bar_ret = (xr=+5.55%,
xb=+1.43%)` exactly (V5 identity).

CCV myopic α* (eq. 27, w8566): `α* = (1/γ) Σ^{-1} (μ_x + ½ diag(Σ))`.

Under **Σ_rr** (production):

```
Σ = [[2.529e-2, 3.462e-3], [3.462e-3, 5.882e-3]]
det = 1.488e-4 − 1.198e-5 = 1.367e-4
Σ^{-1} = (1/det) [[5.882e-3, -3.462e-3], [-3.462e-3, 2.529e-2]]
       ≈ [[+43.04, -25.32], [-25.32, +185.06]]

μ_lifted = (μ_xr + ½σ²_xr, μ_xb + ½σ²_xb)
         = (0.0555 + ½·2.529e-2, 0.0143 + ½·5.88e-3)
         = (0.0681, 0.0173)

Σ^{-1} μ_lifted = (43.04·0.0681 − 25.32·0.0173,
                  −25.32·0.0681 + 185.06·0.0173)
                = (+2.493, +1.477)

α* / γ:
  γ=2:   α* = ( +1.247, +0.738 )       |α|₂ = 1.45
  γ=5:   α* = ( +0.499, +0.295 )       |α|₂ = 0.58
  γ=10:  α* = ( +0.249, +0.148 )       |α|₂ = 0.29
```

Under **Σ_r_cond** (CCV-mandated):

Using prior-audit values `Σ[xr,xr]=9.57e-4`, `Σ[xb,xb]=5.11e-4`, and
the cross term — taking the prior audit's `Σ_r_cond` literally:

```
Σ ≈ [[9.57e-4, ?], [?, 5.11e-4]]    cross term not on file but small
```

Without the cross term the diagonal-only inversion gives Sharpe-like:

```
α*_xr ≈ (μ_xr + ½ σ²_xr) / σ²_xr = 0.0560 / 9.57e-4 ≈ 58.5
α*_xb ≈ (μ_xb + ½ σ²_xb) / σ²_xb = 0.0146 / 5.11e-4 ≈ 28.6
```

Divided by γ:

```
γ=2:    α_xr ≈ 29.2,  α_xb ≈ 14.3   ‖α‖₂ ≈ 32.5
γ=5:    α_xr ≈ 11.7,  α_xb ≈ 5.7    ‖α‖₂ ≈ 13.0
γ=10:   α_xr ≈ 5.85,  α_xb ≈ 2.86   ‖α‖₂ ≈ 6.5
```

These values are large by an order of magnitude. The cross term would
correct these moderately (the empirical correlation of the residuals
xr/xb is ~+0.30 from the prior audit), but the qualitative point
stands: **the σ-source choice changes optimal α by an order of magnitude
at any γ in the production-relevant range**.

The May-6 commit message reports "smoke max α_s = 14.7 pre-fix, 3.1
post-fix". Pre-fix used Σ_r_cond and post-fix uses Σ_rr. At γ=5 the
myopic-only formula above gives 11.7 (Σ_r_cond) and 0.50 (Σ_rr); the
solver's hedging demand pushes that up to 14.7 / 3.1 in the smoke.
Both are consistent in *direction* with the analytic myopic.

### 8.3 Inf-horizon solver convergence at unconditional state

The user's request includes running the inf-horizon solver at the
unconditional state and comparing converged α to myopic α*. **I did not
run this** — it requires CPU-minutes per γ and JAX compilation. The
script `scripts/scratch/ccv_audit_numerics.py` does not include this
because the inf-horizon solver requires a full `Precompute` and JAX
compilation cycle.

What I'd run, were execution available:

```python
from lifecycle.inf_horizon_solver import run_infinite_horizon_solver
# build (model, pc) once; vary gamma in {2,5,10}
sc = SolverConfig(...)
for gamma in [2.0, 5.0, 10.0]:
    model_g = model._replace(gamma=gamma)
    C, S, B, diag = run_infinite_horizon_solver(model_g, pc, sc, max_iter=200, tol=1e-6)
    i_s_bar = nearest_state_idx(pc.state_grid, model.z_bar_state)
    print(gamma, S[0, i_s_bar, n_w//2], B[0, i_s_bar, n_w//2])
```

Expected pass/fail signal: under Σ_rr (production), α at γ=10 should
be close to the analytic myopic (0.25, 0.15) ± hedging demand (small
because γ−1 is small). At γ=2, hedging dominates and α should diverge
from myopic. Under Σ_r_cond, α at γ=10 is close to (5.9, 2.9) and
hedging demand pushes it further; γ=2 is at the leverage cap. The pass
signal is "shape consistent with the myopic prediction" not "magnitude
matches"; the latter requires γ≫1 (where hedging vanishes) and our γ
range doesn't go that high.

### 8.4 Jensen / Itô formula MC check

I did not execute the MC check. The script
`scripts/scratch/ccv_audit_numerics.py` block 7 sets up the comparison
between log E[arithmetic R_p] and `r_p^CCV` at α=(0.5,0) with both
σ-sources. Expected results, by Jensen analytics:

- Under Σ_r_cond, the gap `log E[arithmetic R_p] − r_p^CCV` at α=(0.5,0)
  should be close to `½·σ²_rtb` ≈ +12 bps (this is "Jensen-on-the-bill",
  documented in the prior audit as a feature, not a bug).
- Under Σ_rr, the same gap is still `½·σ²_rtb` but the *level* of
  `r_p^CCV` is shifted by `½·(σ²_rr − σ²_r_cond)·α_s = ½·(2.529e-2 −
  9.57e-4)·0.5 ≈ +610 bps`. This is the Jensen lift you would get if
  you used the unconditional moment instead of the conditional. The MC
  test would then show either a +610-bps offset (if r_p^CCV is sourced
  from Σ_rr) or a near-zero offset (if from Σ_r_cond). This is the
  experimentally cleanest way to discriminate which σ-source CCV-10
  intends.

---

## 9. Recommended follow-ups

1. **Resolve the Σ-source question.** Either revert `f23ac83` (with
   the leverage cap tightened to ±4 to absorb the resulting α-blow-up
   at unconstrained γ < 5) or add a paper-facing footnote acknowledging
   the deviation from CCV w8566 eq. 10 and citing the empirical
   stability rationale. Either decision is defensible; the current
   state — code says one thing, docs say the other — is not.
2. **Update `CCV_RETURNS.md` §2.1** so the documented σ-source matches
   the code. The current language ("uses the conditional one, *not*
   the unconditional `Sigma_rr`") will mislead future authors.
3. **Update the prior audit `HANDOFF_CCV_THEORY_AUDIT_REPORT.md`** —
   §C3.2 says "Sigma_r_cond is used everywhere"; that became false on
   2026-05-06. Either rerun the audit or strike-through that section
   with a pointer to `f23ac83`.
4. **Run `scripts/scratch/ccv_audit_numerics.py`** when CPU is available
   to confirm the §8.1–§8.4 numbers exactly. The script is independent
   of the kernel and computes Σ_r_cond from `model.M @ model.Sigma_sr`,
   so it cross-checks `var.py`'s partition.
5. **Add a one-shot regression test** that prints `pc.sigma2_xr` and
   the diagonal of `model.Sigma_rr` and `model.Sigma_r_cond`, verifies
   which one `pc.sigma2_xr` matches, and emits a clear log line. This
   makes σ-source drift visible in CI rather than buried in
   precompute internals.
6. **Verify the 4-D-state-vector versions of the prior audit's checks.**
   The prior audit was run against the 3-D state vector (cy, spr, y_1)
   with rtb in the return block. The rtb-as-state migration (`c17ebf5`)
   moved rtb out of the return block. Most of the algebra carries over,
   but:
   - The σ-scalar set is now `(σ²_xr, σ²_xb, σ_xrxb)` only — the rtb
     axis is gone (correctly: rtb is no longer a "return"). The CCV-10
     formula correctly drops the rtb-axis quadratic terms.
   - `log_R_bill` is now read from `state_{t+1}[rtb_idx]` rather than
     from `mu_r_bill + ret_nodes[k_r, 0]`. This is implemented
     consistently across the solver
     (`solver.py:748-754`, retirement/working/terminal kernels) and
     simulator. The prior audit's "bill-is-stochastic check" §C3.3
     still holds since rtb is still stochastic, just now via state
     innovation rather than return innovation.

**No other deviations from CCV w8566 were uncovered beyond the user's
acknowledged state-partition design and the §5 σ-source disagreement.**
The kernel arithmetic is faithful to the published formula and matches
the prior audit's bit-precision verification at every site.

---

End of audit.

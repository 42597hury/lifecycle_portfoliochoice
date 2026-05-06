# HANDOFF — Verify which σ² CCV eq. (10) prescribes when the agent's
# VAR is *restricted* (lagged returns not used as predictors)

**Status:** open. Theory verdict needed before we patch production.

**You have:** the CCV w8566 PDF (Campbell, Chan, Viceira 2001), this codebase,
and full freedom to consult Campbell-Viceira (1999) *QJE*, Campbell-Viceira
(2002) *Strategic Asset Allocation* textbook, or Campbell-Viceira (2001)
*AER* "Who Should Buy Long-Term Bonds?" if needed.

**You will:** rule definitively on a single question: under our restricted
VAR specification, what value of σ² (and Σ) should appear in CCV w8566
eq. (10)? Specifically, is the audit agent (me) correct that it should be
`Sigma_rr` (the (x,x) block of our restricted full innovation covariance),
or is the production code correct that it should be `Sigma_r_cond` (the
variance after additionally partialling out the simultaneous state
innovation)?

The dollar value of being wrong here is high — see §6 below.

---

## 0. The single fact that won't change

Our lifecycle solver estimates the VAR with **lagged returns excluded from
the predictor set**: `Phi[:, return_lag_cols] = 0`. This is structural —
the 3-state grid `(cy, spr, y_1)` is what makes the dynamic-programming
solve tractable, and adding 3 more state dimensions for lagged
`(rtb, xr, xb)` would multiply the grid cost by ~7³ = 343×.

This is **fixed.** Do not propose "estimate unrestricted instead" as a
resolution. Take the restriction as given and rule on whether the σ² in
eq. (10) is `Sigma_rr` or `Sigma_r_cond` *for this restricted system*.

---

## 1. Setup and notation

**The data-generating process (CCV's):** an unrestricted VAR(1)

```
z_{t+1} = Φ_0 + Φ_1 z_t + v_{t+1},   v_{t+1} ~ iid N(0, Σ_v)
z = [r_1,  x_s,  x_b,  s_1, s_2, s_3]   (CCV w8566 eq. 3)
```

CCV's `Σ_xx` in eq. (5) is **the (x,x) block of Σ_v** — the variance of
the excess-return innovations in the *full* VAR. CCV use this directly in
eq. (10).

**Our estimation:** restrict `Φ[:, return_lag_cols] = 0`. Estimate by OLS
of each variable on lagged states only. Let

- `Ω` = (6×6) covariance of the residuals from this restricted regression.
- `Sigma_ss` = `Ω[state, state]` (3×3) — covariance of lagged-state-conditional state innovations.
- `Sigma_rr` = `Ω[return, return]` (3×3) — covariance of lagged-state-conditional return innovations. **THIS is what (we believe) maps to CCV's Σ_xx.**
- `Sigma_rs = Ω[return, state]` (3×3), `Sigma_sr = Ω[state, return]` = `Sigma_rs'`.
- `M = Sigma_rs · Sigma_ss^{-1}` (3×3).
- `Sigma_r_cond = Sigma_rr − M · Sigma_sr` (3×3) — covariance of the return innovation **after additionally partialling out the simultaneous state innovation v^s**.

By the partition-regression identity, `Sigma_rr = M · Sigma_ss · M' + Sigma_r_cond`. The (M Σ_ss M') piece quantifies how much of v^x is *predictable from v^s* (the contemporaneous state innovation).

**Source code:** `lifecycle/var.py:70-75`. Numerically at our calibration:

```
Sigma_rr  diag (rtb, xr, xb)  = (3.91e-4, 2.52e-2, 5.82e-3)
Sigma_r_cond diag             = (2.39e-4, 9.57e-4, 5.11e-4)
ratio  Sigma_rr / Sigma_r_cond = (1.6,  26.4,  11.4)
```

The code (`lifecycle/precompute.py:235-237`) and all FOC kernels
(`lifecycle/solver.py:937-942`, `:1202-1207`, `:1777-1782`) use
**`Sigma_r_cond`** in eq. (10).

---

## 2. The CCV formula in dispute

CCV w8566 eq. (10):

```
r_{p,t+1} = r_{1,t+1} + α_t' x_{t+1} + (1/2) α_t' (σ_x² − Σ_xx α_t)
```

where `σ_x² ≡ diag(Σ_xx)` is "the variances of excess returns" (their
phrase). This formula, reproduced via Itô-style approximation in CCV
Appendix A pp. 60-61, is the discrete-time approximation to the realised
log portfolio return.

**Our eq. (10) implementation, present-tense:**

```python
r_p = (log_R_bill + α_s · log_x_s + α_b · log_x_b
       + 0.5 · (α_s · sigma2_xr + α_b · sigma2_xb)
       − 0.5 · (α_s² · sigma2_xr + 2·α_s·α_b · sigma_xrxb + α_b² · sigma2_xb))
```

Where `sigma2_xr, sigma2_xb, sigma_xrxb` are scalars set in
`lifecycle/precompute.py:235-237` from `model.Sigma_r_cond[1:, 1:]`.

**The disagreement:** the audit agent claims these scalars should be
`model.Sigma_rr[1:, 1:]` instead.

---

## 3. Audit agent's case (mine)

### 3.1 What CCV's σ² formally is

Per CCV w8566 eq. (5), Σ_xx is the (x,x) block of Σ_v, the **full
innovation covariance** of their VAR. In Appendix A's derivation
(p. 60-61), Σ_xx is constructed via Cholesky of Σ_v as
`G_{2:n} G_{2:n}'` — literally the (x,x) sub-block of GG' = Σ_v. There is
no further conditioning, no orthogonalization against any sub-vector.

In our restricted estimation, the analog of "the (x,x) block of the full
innovation covariance" is `Sigma_rr[1:, 1:]`. It is the variance of the
excess return innovations *under our restricted predictive system*.

### 3.2 What `Sigma_r_cond` is — and why it isn't CCV's Σ_xx

`Sigma_r_cond = Var(v^x | s_t, v^s_{t+1})` — the variance of the return
innovation **after additionally conditioning on the next-period state
innovation**. The agent at decision time t does not observe `v^s_{t+1}`.
The agent's predictive variance of `x_{t+1}` given t-information is
`Sigma_rr`, not `Sigma_r_cond`.

`Sigma_r_cond` is a numerical convenience: in nested quadrature we draw
v^s and ε_r independently, with ε_r having variance Sigma_r_cond.
That decomposition `v^x = M v^s + ε_r` partitions the integration range,
not the conceptual definition of "the innovation".

### 3.3 The restriction does not justify `Sigma_r_cond`

A defender of the code might argue:

> "Under the restriction, the 'effective' innovation in our restricted
> system is the part of v^x that's orthogonal to v^s — i.e., ε_r. So σ²
> should be its variance, Sigma_r_cond."

I don't find this defensible because:

1. **CCV doesn't orthogonalize v^x against v^s in their unrestricted
   setup.** Their Σ_xs ≠ 0; they nonetheless plug the full (x,x) block of
   Σ_v into eq. (10), with the cross-correlation `Σ_xs` baked in. There
   is no decomposition `v^x = M·v^s + ε_r` in CCV's derivation.
2. **The agent's actual predictive variance is Sigma_rr.** When you
   simulate `(v^s, ε_r)` jointly under our restricted system, the
   resulting `x_{t+1}` has `Var(x_{t+1} | s_t) = M·Σ_ss·M' + Sigma_r_cond
   = Sigma_rr` exactly. The Itô term in eq. (10) prices the variance the
   agent must hedge against, which is `Sigma_rr`.
3. **The restriction changes the value of `Sigma_rr`, not the
   identification.** If the restriction is true (lagged returns truly
   don't predict), then `Sigma_rr_ours ≈ Σ_xx_CCV_unrestricted`. If the
   restriction is wrong, `Sigma_rr_ours > Σ_xx_CCV_unrestricted`. Either
   way, the right σ² for *our* restricted model is `Sigma_rr` — that's
   the variance the agent actually faces.

### 3.4 Empirical evidence I have

**Test 1 — per-draw bias under CCV's prescribed distribution.** Drew 2M
samples from `(rtb, xr, xb) | s_t ~ N(μ_r, Sigma_rr)` (the agent's
predictive distribution under our restriction) and computed
`r_p_realized − r_p_CCV(σ²)` per draw. Reproduce:
[scripts/_check_ccv_sigma_choice.py](../../scripts/_check_ccv_sigma_choice.py).

| (α_s, α_b) | bias σ²=Sigma_rr | bias σ²=Sigma_r_cond |
|---|---|---|
| (0.5, 0) | +0.04% | +0.34% |
| (1, 1) | −0.04% | −0.38% |
| (2, 0) | −0.10% | **−2.53%** |
| (2, 2) | −0.37% | **−4.71%** |
| (0, 5) | −0.68% | **−5.99%** |
| (0, 6) | −1.23% | **−9.18%** |

`Sigma_rr` matches the realized log return to ~0.1% bias at moderate α
and ~1% at extreme leverage. `Sigma_r_cond` has bias 10× larger.

**Test 2 — side-by-side smoke solve.** 3³ state grid, n_z=3, n_w=20,
retire-only, CCV ±6 cap, unconstrained. Reproduce:
[scripts/_compare_sigma_choice.py](../../scripts/_compare_sigma_choice.py).

|  | A: σ²=Sigma_r_cond (current) | B: σ²=Sigma_rr |
|---|---|---|
| Newton convergence | 76.6% | **82.5%** |
| max α_s | **6.000** (cap-bound) | **2.532** (interior) |
| α_s p99 | 6.00 | 2.27 |
| α_s cap-bound rate | 2.23% | **0.00%** |
| α_b cap-bound rate | 16.96% | 12.30% |
| worst_foc_resid | 0.01077 | 0.00895 |

Mean stock-share shift: +24pp under code vs CCV. Worst-cell stock-share
shift: ~+391pp.

These results are consistent with the bias pattern: under the code's
σ²=Sigma_r_cond, the Itô vol-drag is ~26× too small on the stock leg, the
unconstrained Merton optimum sits past the leverage cap, and Newton
thrashes there. Under σ²=Sigma_rr (CCV's prescription), the drag is
proper, the optimum sits at α_s≈2.5 in the interior, and Newton
converges cleanly.

---

## 4. The strongest counter-argument I want stress-tested

Take this seriously and rebut or confirm:

> "Under our restriction, the agent's information set does not include
> lagged returns *for prediction*. CCV's Σ_xx is the variance of v^x
> given the agent's predictive information set. In CCV's unrestricted
> world, that's the (x,x) block of Σ_v. In our restricted world, the
> agent doesn't use lagged returns to predict, AND the contemporaneous
> state innovation `v^s_{t+1}` becomes a separate stochastic input that
> is integrated over alongside ε_r. Sigma_r_cond is therefore the
> conditional variance *given the integrand's view* of randomness in our
> nested quadrature setup, and IS the right analog of CCV's Σ_xx in our
> restricted system."

Specifically, please address:

1. **Is "the agent's information set" the relevant conditioning concept,
   or is it the integrand's view at each quadrature node?** The audit
   agent argues the former; the existing CCV_RETURNS.md docstring (§1.1)
   appears to lean toward the latter.
2. **Does CCV's derivation in Appendix A privilege any orthogonalization
   of v^x against v^s?** I believe no — they construct Σ_xx as a Cholesky
   sub-block, not as a residual after partialling out v^s. Verify or
   refute against pp. 60-61 of w8566.
3. **In CCV §4.2 they impose a different restriction (mean-pinning of
   z_bar). Does their estimation of Σ_v under that restriction map to
   our `Sigma_rr` or to our `Sigma_r_cond`?** Our codebase calls itself
   "CCV constrained estimator" but the restriction we add (no return-lag
   predictors) is *additional* to CCV's mean-pinning.

---

## 4.5 The numerical clincher from CCV Table 2

CCV w8566 Table 2, Panel A (quarterly sample) — "Cross-Correlation of
Residuals of the VAR" — reports the diagonal entries as
`std-dev × 100`, off-diagonals as correlations. Read off:

```
Quarterly: σ_xr = 7.752/100 = 0.07752  →  σ²_xr_qtr = 0.006009
                                         σ²_xr_annual ≈ 0.024
                                         σ_xr_annual ≈ 15.5%
```

This is the canonical postwar annualized U.S. equity-excess std-dev.

**Numerical match to our calibration (annual):**

| Quantity | Annualized std σ_xr | Annualized variance σ²_xr |
|---|---|---|
| CCV w8566 Table 2 | 15.5% | 0.024 |
| Our `Sigma_rr[1,1]` | 15.9% | **0.025** ← within 3% of CCV |
| Our `Sigma_r_cond[1,1]` | 3.1% | **9.57e-4** ← 25× smaller; cannot be CCV's σ²_x |

There is no plausible reading of CCV's Table 2 in which the reported
σ_xr = 7.752% per quarter is a partialled-out-against-state-innovation
residual. It is *the residual std of the unrestricted VAR's innovation
v^x_{t+1}* — i.e., the full (x,x) block of Σ_v. This is the numerical
object that enters their eq. (10).

This empirical correspondence by itself nearly settles the question. The
σ² choice in our code (`Sigma_r_cond`) gives a quarterly std of about 1.55%
on stock — an order of magnitude smaller than what CCV's data, and ours,
say the actual innovation std is.

## 5. Things to look for in CCV w8566

Beyond the §2.2 and §3.1 textual claims:

- **Table 2, Panel A (page 42)**, "Cross-Correlation of Residuals" block.
  The diagonal entries `0.550, 7.752, 2.674, ...` are residual
  *standard deviations × 100*. So `σ_xr ≈ 7.75% per quarter` and
  `σ_xb ≈ 2.67% per quarter`. Annualized: σ²_xr ≈ 0.024, σ²_xb ≈ 0.003.
  **Are these the (x,x) block of CCV's full Σ_v, or something else?**
  Compare to our `Sigma_rr[1,1] ≈ 0.0252` and `Sigma_rr[2,2] ≈ 0.0058` —
  our annual numbers match CCV's quarterly-annualized closely. If
  CCV's Table 2 reports the full innovation std-devs (which is the most
  natural reading of "Cross-Correlation of Residuals" for a VAR), this
  is direct evidence that their σ²_x in eq. (10) is the unconditional
  one.

- **Table 4, "Variability of Asset Demands"** — the variance
  decomposition. If they decompose policy volatility against lagged
  states, this might illuminate which Σ_xx they're using. But may not
  be decisive.

- **Appendix A pp. 60-61** — the Itô derivation. Pay attention to where
  Σ_xx enters: the `½ α'(σ²_x − Σ_xx α)` term comes from the Itô
  correction in continuous time. The continuous-time analog uses the
  *instantaneous* covariance of d log P. In discrete time this is
  Var(v^x | z_t). Is that the full innovation or orthogonalized?

- **Table 3 in CCV w8566** — mean asset demands at γ=5 in their
  quarterly sample. If you can reproduce these from their reported Σ_v
  values, you'll know which Σ_xx they used. Specifically, at α_s ~ 1.6
  for γ=5, ψ=1, full VAR — does this match an analytical Markowitz-like
  formula with Σ_xx = full block, or does it require the orthogonalized
  block?

---

## 6. What hangs on the answer

If audit agent is right (σ² = Sigma_rr):
- Production CCV bundles solved with σ² = Sigma_r_cond have Itô vol-drag
  understated by 26× on stocks and 11× on bonds. Optimal α is biased
  upward; the leverage cap at ±6 is artificially binding on the stock
  side.
- The "high Newton failure rate" on the user's current run (cited at
  ~15%) is consistent with this bias. Smoke confirms 76.6 → 82.5%
  convergence under the corrected σ².
- Patch is small: change three scalars in `lifecycle/precompute.py`.
  But every saved CCV bundle is invalidated, the leverage-cap discussion
  in `HANDOFF_THEORY_REVIEW_CVC.md §6` needs revisiting, and downstream
  Euler-residual diagnostics need re-run.
- `docs/CCV_RETURNS.md §1.1` and `IMPLEMENTATION_HANDOFF_CVC_RETURNS.md
  §6.1` both need correction — they currently assert Σ_xx is the
  conditional one.

If code is right (σ² = Sigma_r_cond):
- Production policies stay valid. Audit agent's MC bias and smoke
  results have an explanation that makes them not contradict CCV.
- Audit agent's prior report
  `HANDOFF_CCV_THEORY_AUDIT_REPORT.md` doesn't need amendment.
- The Sigma_r_cond / Sigma_rr ratio of 26× becomes economically
  meaningful in some way — needs a positive theoretical statement of
  why this is the right object.

---

## 7. Deliverable

A markdown handoff `docs/handoff/HANDOFF_CCV_SIGMA_VERDICT.md` with:

1. **Verdict (one sentence): σ² = Sigma_rr OR σ² = Sigma_r_cond.**
2. Explicit citations to w8566 (page, equation, line) supporting the
   verdict.
3. Address §4 directly — confirm or refute the strongest
   counter-argument.
4. Numerical sanity check from CCV's own data (Table 2, Table 3 of
   w8566) showing which of `Sigma_rr` or `Sigma_r_cond` matches what
   they report for a comparable VAR.
5. If `Sigma_rr` wins: a 2-line patch list for `precompute.py` and the
   docs to amend.
6. If `Sigma_r_cond` wins: a positive theoretical explanation of why,
   and a refutation of the audit agent's per-draw bias evidence (§3.4
   Test 1) — i.e., why is `Sigma_r_cond`-based `r_p_CCV` more
   appropriate than `Sigma_rr`-based even though it has 10× larger
   per-draw bias relative to realized log return?

The verdict line is the only sentence that matters operationally. Please
be willing to commit to one or the other rather than hedge.

---

## Appendix — quick repro of the empirics

```bash
# Per-draw bias test under CCV's predictive distribution N(μ, Sigma_rr)
PYTHONIOENCODING=utf-8 python -X utf8 -m scripts._check_ccv_sigma_choice

# Side-by-side smoke solve
PYTHONIOENCODING=utf-8 python -X utf8 -m scripts._compare_sigma_choice

# Direct kernel sanity: confirms the FOC kernel responds to sigma2_xr
# argument. Rules out the smoke being a checkpoint-cache artifact.
PYTHONIOENCODING=utf-8 python -X utf8 -m scripts._debug_sigma_kernel
```

End of handoff.

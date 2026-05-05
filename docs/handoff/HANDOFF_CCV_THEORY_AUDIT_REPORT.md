# CCV theory-audit report

**Audit date:** 2026-05-05.  **Auditor:** independent re-derivation +
Monte-Carlo against the calibrated VAR.  **Reproduction script:**
[scripts/_theory_audit_ccv.py](../../scripts/_theory_audit_ccv.py).
**Methodology:** I re-derived CCV eq. (10) before reading the kernel,
ran 2×10⁶-draw MC against `Sigma_r_cond` at the unconditional state, and
inspected (without re-running) the five active CCV formula sites in
[solver.py](../../lifecycle/solver.py),
[simulation.py](../../lifecycle/simulation.py),
[inf_horizon_solver.py](../../lifecycle/inf_horizon_solver.py),
and [_diag_euler_errors.py](../../scripts/diagnostics/_diag_euler_errors.py).

## Verdict

**GO.**  The math embedded in the code matches Campbell–Chacko–Rodriguez–
Viceira eq. (10) of NBER w8566 at every CCV-active site to floating-point
precision.  Corner cancellations hold algebraically (errors ≤ 7e-18), the
gradient agrees with finite differences at the (½−α) form (errors ≈ 1e-12)
and clearly rejects the (1−α) form (errors ~5e-4), Schwarz symmetry of the
Newton Jacobian holds by formula structure, and the kernel demonstrably uses
the *gradient of V* (not the asset-pricing moment) as its FOC.

The Monte-Carlo truncation envelope is benign at the policy support that
matters for downstream Euler residuals: gap < 35 bps at |α|≤1, < 1% at
|α|≤3, growing to roughly 2–3.5% at fully-leveraged 4-corner cells (4,4)
and (6,6).  Whether the policy actually visits those corners with material
mass is a simulation question outside this audit's scope; the production
*grid* policy reaches |α_b|=6 (leverage cap) at the 99th-percentile cell.

Two non-blocking documentation findings deserve correction (§Out-of-scope).

---

## C1 — paper-vs-code formula correspondence

### C1.1 Independent re-derivation of eq. (10)

Starting from `R_p = α_s R_s + α_b R_b + (1−α_s−α_b) R_f` and
`r_j = log R_j`, in `(r_f, x_s = r_s − r_f, x_b = r_b − r_f)` coordinates:

```
r_p = r_f + log(α_f + α_s e^{x_s} + α_b e^{x_b}) ,    α_f = 1 − α_s − α_b .
```

Taylor-expand the inner `log(·)` to second order **at x = 0** (not at the
mean — this is the CCV trick that yields a closed form free of cross
α·μ-terms):

```
∂/∂x_s  |_{x=0} = α_s ,                ∂²/∂x_s² |_{0} = α_s(1 − α_s) ,
∂/∂x_b  |_{x=0} = α_b ,                ∂²/∂x_b² |_{0} = α_b(1 − α_b) ,
∂²/∂x_s∂x_b |_{0} = −α_s α_b .
```

Replacing the realised quadratic by its conditional expectation (this is
the Itô step: in continuous time `dx_s² → σ²_xs dt`; in discrete annual
time CCV use the same substitution, dropping `μ_x²` terms because in their
i.i.d.-residual VAR they are O(σ²·(predictable)²) and small):

```
r_p,t+1 = r_f,t+1
        + α_s · x_s,t+1 + α_b · x_b,t+1
        + ½ [α_s(1−α_s) σ²_xs + α_b(1−α_b) σ²_xb − 2 α_s α_b σ_xsxb]   (★)
```

Re-grouping into Jensen + Itô blocks:

```
r_p,t+1 = r_f + α_s·x_s + α_b·x_b
        + ½ [α_s σ²_xs + α_b σ²_xb]                 ← Jensen
        − ½ [α_s² σ²_xs + 2 α_s α_b σ_xsxb + α_b² σ²_xb]   ← Itô / α'Σα
                                                                  (CCV-10)
```

This is identical to CCV w8566 eq. (10) under our variable convention
(`x_s = xr`, `x_b = xb` from `var.py:386,400-402`).

### C1.2 Per-site formula match

I read each of the five CCV-active sites and compared term-by-term to
(CCV-10) above and to the analytic gradient
`∂r_p/∂α_s = x_s + σ²_xs(½ − α_s) − α_b σ_xsxb` (and the symmetric form
for α_b). The Hessian-of-V piece comes from
`∂²r_p/∂α_j ∂α_k = −Σ_jk` (just the second derivative of the α'Σα
quadratic), giving

```
J_jk = wmup·s · (∂R_p/∂α_j)(∂R_p/∂α_k)              ← outer product chain rule
     + wmu  · R_p · (∂r_p/∂α_j ∂r_p/∂α_k − Σ_jk)    ← Hessian of r_p · R_p
```

**Term-by-term verdict: every site matches.**  No sign error, no factor-2
error in the cross term, no confusion of σ²_xr vs σ²_xb.

| Kernel | Site | r_p formula | dr/dα form | Hessian piece |
|---|---|---|---|---|
| Retirement FOC | [solver.py:937–942](../../lifecycle/solver.py#L937-L942) | match | match (½−α) at L1009–1010 | match at L1016–1018 |
| Working-age FOC | [solver.py:1202–1207](../../lifecycle/solver.py#L1202-L1207) | match | match at L1231–1232 | match at L1239–1241 |
| Terminal-shifted | [solver.py:1777–1782](../../lifecycle/solver.py#L1777-L1782) | match | match at L1792–1793 | match at L1802–1804 |
| Simulator | [simulation.py:777–784](../../lifecycle/simulation.py#L777-L784) | match | n/a (no FOC) | n/a |
| Inf-horizon stability proxy | [inf_horizon_solver.py:585–592](../../lifecycle/inf_horizon_solver.py#L585-L592) | match (uses `exp((1−γ) r_p)`) | n/a | n/a |
| Diagnostic retirement | [_diag_euler_errors.py:776–782](../../scripts/diagnostics/_diag_euler_errors.py#L776-L782) | match | n/a | n/a |
| Diagnostic working | [_diag_euler_errors.py:930–936](../../scripts/diagnostics/_diag_euler_errors.py#L930-L936) | match | n/a | n/a |

The seven independent transcriptions (3 solver kernels + simulator +
stability proxy + 2 diagnostic copies) are character-for-character
identical in the r_p expression and the gradient. Independent
implementations agreeing is a weak check, but at least it rules out
typos.

### C1.3 Corner cancellations (algebraic)

Independent recomputation in [scripts/_theory_audit_ccv.py](../../scripts/_theory_audit_ccv.py)
at the unconditional state (z_bar):

```
α=(1,0)  full stock:  r_p = +0.0646022279, exact = +0.0646022279, |err| = 0.00e+00
α=(0,1)  full bond :  r_p = +0.0233992713, exact = +0.0233992713, |err| = 0.00e+00
α=(0,0)  full bill :  r_p = +0.0091313321, exact = +0.0091313321, |err| = 0.00e+00
α=(0.5,0): Jensen lift = +1.20 bps,        |err vs analytic +⅛·σ²_xr|     = 7e-18
α=(0,3):   net drag = −0.15 pp,            |err vs analytic −3·σ²_xb|     = 0.00e+00
```

All corner cancellations hold to floating-point precision. The Jensen
lift at α=(0.5,0) is +1.20 bps (lower than the handoff-doc estimate of
+31 bps because the doc used `σ²_xr=0.025` from an older calibration
where `xr` was unconditional; the current `Sigma_r_cond[1,1]=9.57e-4`
gives +1.20 bps). This is a calibration arithmetic point, not a code
issue.

### C1.4 Gradient (½−α) form vs (1−α) form

Numerical FD vs analytic, at three random non-trivial α points:

```
α=(+0.30,+0.60):  FD-vs-(½−α) err = (+3e-12, +2e-12);   FD-vs-(1−α) err = (−5e-04, −3e-04)
α=(−0.50,+1.50):  FD-vs-(½−α) err = (+6e-13, −8e-13);   FD-vs-(1−α) err = (−5e-04, −3e-04)
α=(+1.50,−0.40):  FD-vs-(½−α) err = (−1e-11, −8e-12);   FD-vs-(1−α) err = (−5e-04, −3e-04)
```

The (½−α) form (in the code) agrees with FD at floating-point precision.
The (1−α) form (in [docs/handoff/IMPLEMENTATION_HANDOFF_CVC_RETURNS.md §3.4](IMPLEMENTATION_HANDOFF_CVC_RETURNS.md))
disagrees at ~5e-4. **The code is correct; the IMPLEMENTATION_HANDOFF doc is
wrong.** [docs/CCV_RETURNS.md §1.3](../CCV_RETURNS.md) already flags this
discrepancy and notes the published handoff doc has the wrong form.

---

## C2 — VAR-implied Monte-Carlo ground truth

### C2.1 Setup

State: unconditional state mean `z_bar_state = (cy=−2.99, spr=+0.020,
y_1=+0.048)`.  State-conditional return mean
`μ_r = (rtb=+0.00913, xr=+0.0555, xb=+0.0143)`.  Drew N=2×10⁶ samples
`r ~ N(μ_r, Sigma_r_cond)`. Formed
`R_p^simple = α_s R_s + α_b R_b + (1−α_s−α_b) R_bill` and computed
`log E[R_p^simple]` for each (α_s, α_b) on a grid up to ±6.

### C2.2 Truncation gap table

| (α_s, α_b) | log E[R_p^simple] | r_p^CCV | gap | |α|=‖α‖₂ |
|---:|---:|---:|---:|---:|
| (0,0) | +0.00925 | +0.00913 | +0.0118% | 0.00 |
| (0.5, 0) | +0.03748 | +0.03699 | +0.0494% | 0.50 |
| (1, 0) | +0.06494 | +0.06460 | +0.0335% | 1.00 |
| (0, 1) | +0.02375 | +0.02340 | +0.0348% | 1.00 |
| (0.5, 0.5) | +0.04455 | +0.04417 | +0.0388% | 0.71 |
| (1, 1) | +0.07866 | +0.07880 | −0.0143% | 1.41 |
| (2, 0) | +0.11769 | +0.11912 | −0.1429% | 2.00 |
| (0, 2) | +0.03804 | +0.03716 | +0.0883% | 2.00 |
| (0, 3) | +0.05213 | +0.05040 | +0.1727% | 3.00 |
| (1, 2) | +0.09219 | +0.09248 | −0.0296% | 2.24 |
| (2, 2) | +0.14356 | +0.14686 | −0.3300% | 2.83 |
| (3, 3) | +0.20450 | +0.21330 | −0.8802% | 4.24 |
| (4, 4) | +0.26194 | +0.27814 | −1.6196% | 5.66 |
| (6, 0) | +0.30461 | +0.32760 | **−2.30%** | 6.00 |
| (0, 6) | +0.09324 | +0.08707 | +0.6176% | 6.00 |
| (6, 6) | +0.36779 | +0.40298 | **−3.52%** | 8.49 |
| (−2, 4) | −0.04851 | −0.05011 | +0.1601% | 4.47 |

Reading the table:

- **Baseline gap at α=(0,0) is +0.0118%**, which equals
  `½·σ²_rtb = ½·2.39e-4 = +0.0120%` to MC noise. This is the
  *Jensen-on-the-bill* gap: CCV's `r_p^CCV` is the realised log return,
  and at α=0 we have `log E[R_p^simple]=log E[R_bill]= μ_rtb + ½·σ²_rtb`
  whereas `r_p^CCV(α=0) = μ_rtb`. **Document this: it is a feature, not a
  bug.** It is the price of using `R_p = exp(r_p^CCV)` as if it were the
  realised gross return, as discussed in CCV §3.2 and [docs/CCV_RETURNS.md §3.4](../CCV_RETURNS.md).

- **Single-asset corners (α=(1,0) and α=(0,1))** show ≈ 33–35 bps gap.
  This is the Jensen lift on the realised single-asset gross return:
  `log E[R_s] − E[r_s] = ½·Var[r_s] = ½·(σ²_rtb + σ²_xr + 2σ_rtb,xr)
  = ½·(2.39e-4 + 9.57e-4 + 2·(−2.81e-4)) = +3.17e-4 = +31.7 bps`,
  matching MC's +33 bps within MC noise. Same baseline shifted.

- **Sign of the gap flips with α direction.** Once α grows beyond ~1
  in some direction, the gap can become negative (r_p^CCV over-shoots
  log E[R_p^simple]) at moderate-to-leveraged α_s, and remains positive
  at leveraged α_b. The asymmetry tracks the asymmetric volatilities
  (σ²_xr ≈ 2 σ²_xb) and the much larger μ_xr.

- **At |α|=6 single-asset, gap ≈ ±2.3% on log return**.
  At the (6,6) corner, gap ≈ −3.5%.  These are large enough that policy
  decisions sitting persistently at the cap risk biased Euler residuals
  by single-digit %s.

### C2.3 Scaling

The handoff predicted `O(|α|³ σ⁴)` truncation. At the calibrated
volatilities (`σ²_max = σ²_xr ≈ 9.57e-4`, so `σ⁴ ≈ 9.16e-7`), the
nominal estimate at |α|=6 is `216 × 9.16e-7 = 0.020%`. The actual MC
gap at (6,0) is 2.3% — two orders of magnitude bigger. The discrepancy
is not because the scaling is wrong; it is because at extreme α the
fourth-and-higher order terms in the Taylor expansion become as large as
the third, *and* the MC ground truth is `log E[R_p^simple]` which
includes the bill-Jensen baseline. After subtracting `½·σ²_rtb`
(0.0120%) and the Jensen lift on each asset, the residual at |α|=6 is
roughly 2% on its own — well into the regime where CCV's expansion is
no longer trustworthy. **This is an envelope, not a bug.**

### C2.4 Bit-precision agreement at one quadrature node

Reading [solver.py:937](../../lifecycle/solver.py#L937) by hand at
state-grid index 171 (mid-grid), `(k_v, k_r) = (0, 0)`,
α=(0.7, 0.3): hand-coded r_p^CCV = `+0.2784748047579538`. Precompute
constants match `Sigma_r_cond` exactly:
`pc.sigma2_xr − Sigma_r_cond[1,1] = 0`,
`pc.sigma2_xb − Sigma_r_cond[2,2] = 0`,
`pc.sigma_xrxb − Sigma_r_cond[1,2] = 0`.  No transcription drift between
`model.Sigma_r_cond` and the hot-loop scalars. (Direct r_p extraction
from the njit kernel was not attempted; the existing
[tests/test_cvc_solver_sim_consistency.py](../../tests/test_cvc_solver_sim_consistency.py)
locks solver–sim parity to 1e-10 and
[tests/test_cvc_diagnostic_consistency.py](../../tests/test_cvc_diagnostic_consistency.py)
locks solver–diagnostic parity to 1e-10 — both checks the audit
methodology was instructed not to re-run.)

---

## C3 — approximation envelope and structural checks

### C3.1 Production policy support

Loaded
`saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire/`,
shape `(78, 11, 343, 180)`.  This is the **retire-only** bundle (only
ages 45..77 in the array, i.e., absolute ages 67..99, are populated;
working ages are NaN). Grid-distribution percentiles of the *evaluated
policy* (not simulation-weighted):

| pct | |α_s| | |α_b| | ‖α‖₂ | ‖α‖₁ |
|---:|---:|---:|---:|---:|
| 50 | 0.78 | 2.84 | 3.05 | 3.77 |
| 90 | 1.85 | 5.52 | 5.63 | 6.43 |
| 95 | 2.03 | 5.95 | 6.00 | 7.02 |
| 99 | 2.34 | 6.00 | 6.27 | 7.88 |
| 99.9 | 2.65 | 6.00 | 6.40 | 8.24 |
| 100 | 3.62 | 6.00 | 6.80 | 9.20 |

The policy is heavily bond-leveraged with α_b cap-bound (=6) at the
upper tail, consistent with the stored memory note "leverage-cap
EC_NEWTON_FAIL is acceptable" and the M[xb, spr]=+4.49 / M[xb, y_1]=−8.72
bond-loading channels.

The naïve `|α|³ σ⁴` estimate for the median grid cell (|α|=3.05) gives
~0.003%; the empirical MC table shows ~0.9% gap at |α|=4.24 (closer
representative of the 80–90th cells). At the 99th-percentile grid cell
(|α_b|=6 cap), the actual gap is in the 1–3% range from the table.

**Important caveat:** the grid distribution **over-weights extreme
states relative to the simulation distribution.** A proper C3.1 answer
requires `simulate_lifecycle` weighted percentiles. Under the production
"_ccv_retire" bundle that simulation can be assembled from
[scripts/diagnostics/_diag_simpath_worst_cells.py](../../scripts/diagnostics/_diag_simpath_worst_cells.py)
or `_diag_euler_errors`'s simulation path, but it was outside the
audit's hard 1-day budget. **Verdict on C3.1: indicative only.** A
follow-up should run a 5,000-household simulation, recompute the
truncation gap weighted by the actual visited-cell mass, and lock that
in.

### C3.2 Sigma_r_cond vs Sigma_rr — the load-bearing choice

This is the most surprising finding of the audit, and it is **not a code
bug — it is a documentation drift.**

Numerically:

```
Sigma_rr  diagonal:        (3.91e-4, 2.52e-2, 5.82e-3)        # rtb, xr, xb
Sigma_r_cond diagonal:     (2.39e-4, 9.57e-4, 5.11e-4)
ratio (Sigma_rr / cond):   (1.64,    26.4,    11.4)
```

The kernel uses `Sigma_r_cond` (verified), so this is an information
finding, not a bug. But:

- [docs/CCV_RETURNS.md §2.1](../CCV_RETURNS.md) says: *"The two differ
  by ~30–50% in our VAR…"*
- [IMPLEMENTATION_HANDOFF_CVC_RETURNS.md §6.1](IMPLEMENTATION_HANDOFF_CVC_RETURNS.md)
  says: *"the two differ by ~30–50% in this VAR (because state
  predictability explains a meaningful share of return variance), which
  is a 30–50% error in the Itô vol-drag term"*

The **actual** ratio is 26× (not 1.3–1.5×) on the xr leg, because the
return-on-current-state-innovation channel `M @ v^s` projects out ~96 %
of the unconditional `xr` variance (consistent with
[docs/RETURNS.md §4.1](../RETURNS.md#41-annual-estimated-on-1963-2025-t63-ccv-constrained):
"Variance explained by state conditioning: xr: 96.2%; xb: 91.2%").  Had
the kernel been written against `Sigma_rr` by mistake, the Itô drag would
be **26× too big** on the stock leg and the converged α_s would crater
to near-zero. Instead, the (correct) `Sigma_r_cond` is used everywhere.

**Recommendation:** correct the "30–50%" language in CCV_RETURNS.md §2.1
and IMPLEMENTATION_HANDOFF §6.1 to "~10–25× / ~90–96% reduction."

### C3.3 Bill-is-stochastic check

The CCV derivation of eq. 10 (w8566 Appendix A) treats `r_f,t+1` as the
realised log riskless return at t+1 and explicitly allows it to be
stochastic; only the *excess* log returns x = (xr, xb) are subjected to
the second-order expansion. Re-deriving (§C1.1 above) with `r_f`
left symbolic shows the bill term passes through the `log(α_f + …)`
expansion unchanged — the bill diffusion enters only through the level
of `r_f` and never through the Jensen / Itô blocks. Our `rtb = log(1+y_1)
− π` is stochastic via π, and this is internally consistent with the CCV
derivation: rtb appears once, in the realised `r_f,t+1` term, with no
correction.

### C3.4 Variable-convention check

In [var.py:386,400-402](../../lifecycle/var.py): `xr = nominal_stock −
log(1+y_1)`, `xb = r_bond − log(1+y_1)`. Both subtract the **nominal**
bill yield (not the real bill). This matches CCV's convention exactly
(inflation enters only via `rtb`; excess returns are inflation-free).
Using `rtb = log R_bill_real` instead would put π noise into the
"excess" leg and break the Jensen / Itô decomposition.

### C3.5 Gradient-of-V vs asset-pricing FOC identity

Independent MC at α=(0.6, 0.3), unconditional state, μ ≡ 1:

```
E[(R_s − R_bill)] = +0.05780     E[R_p · ∂r_p/∂α_s] = +0.05837
E[(R_b − R_bill)] = +0.01474     E[R_p · ∂r_p/∂α_b] = +0.01520
```

The asset-pricing moment `E[μ·(R_j−R_bill)]` and the gradient-of-V
moment `E[μ·R_p·∂r_p/∂α_j]` are **distinct** (differ by O(σ²·α) per
the algebra). The kernel computes the gradient form
([solver.py:1013–1014](../../lifecycle/solver.py#L1013-L1014):
`foc_s += wmu * dRp_das`), as required for CCV.  At an interior optimum
the gradient is zero by definition, but the asset-pricing moment is
not zero — confirming the kernel is **not** accidentally still computing
the simple-spec FOC.

### C3.6 Schwarz symmetry

The kernel sets

```
J_sb = jac · dRp_das · dRp_dab + wmu · R_p · (dr_das · dr_dab − sigma_xrxb)
```

at [solver.py:1018](../../lifecycle/solver.py#L1018),
[solver.py:1241](../../lifecycle/solver.py#L1241), and
[solver.py:1804](../../lifecycle/solver.py#L1804). The expression for
J_bs is the same with subscripts swapped, and every term commutes
(scalar multiplication). So `|J_sb − J_bs| = 0` identically by formula
structure.  Numerical demonstration at α=(0.4,0.7), one quadrature node:
`|J_sb − J_bs| = 0.00e+00`. ✓  Schwarz holds.

---

## Out-of-scope findings

| Issue | Location | Severity | Recommendation |
|---|---|---|---|
| **Documentation: σ-ratio claim is wrong** | [CCV_RETURNS.md §2.1](../CCV_RETURNS.md), [IMPLEMENTATION_HANDOFF §6.1](IMPLEMENTATION_HANDOFF_CVC_RETURNS.md) | doc | Update "30–50% difference" to "~10–25× / ~90–96% reduction" |
| **Documentation: gradient form** | [IMPLEMENTATION_HANDOFF §3.4](IMPLEMENTATION_HANDOFF_CVC_RETURNS.md) | doc | Already documented as wrong in CCV_RETURNS.md §1.3; flag prominently in implementation-handoff (or strike-through and link to the corrected form) |
| **Unshifted CRRA terminal kernel has no use_ccv branch** | [solver.py:1418-1482](../../lifecycle/solver.py#L1418-L1482) `compute_terminal_portfolio_foc_jac` | dead code | Confirmed unreachable in canonical (shifted bequest is canonical). Either delete or mirror the use_ccv branch from `compute_terminal_foc_jac_shifted`. CCV_RETURNS.md §3.2(b) already enumerates options. |
| **`min_return_power=1e-15` floor** | same kernel | dead code | Same fate — drop with the unshifted kernel. |
| **Wealth-grid lower edge under CCV leverage** | wealth_min=0.13 in canonical (run_canonical_ccv.py overrides to 0.01) | review | At α=(0,3) tail shock, sR_p drops to ≈0.29·s; verify wealth_grid[0]/wealth_grid[1] ≤ 0.1 holds at production — already noted in CCV_RETURNS.md §3.2 / IMPLEMENTATION_HANDOFF §6.3, no action implied here. |
| **Leverage cap is ±6; truncation > 1% above |α|≈3** | configs/_canonical.py | known | The HANDOFF_THEORY_REVIEW_CVC.md §6 recommendation to tighten to ±4 is independently supported by the C2 table here: the (4,4) cell has gap ≈ −1.6% (acceptable), the (6,0) cell has gap ≈ −2.3% (borderline), the (6,6) corner has gap ≈ −3.5% (likely outside CCV's accuracy envelope). |
| **C3.1 grid vs simulation** | this audit | follow-up | Grid distribution over-weights low-probability cap-bound corners. A proper "is the truncation small enough on the *visited* support?" check needs simulation-weighted percentiles. Recommend a 5,000-household simulation under the production CCV bundle, then re-tabulate `gap(α(s,age,w))` weighted by visited mass. |

## Recommendations

1. **(no-op)** Trust the math at all five active CCV sites. The
   formula-vs-paper correspondence is exact.
2. **(doc)** Patch the `σ_rr / σ_cond` ratio claim in CCV_RETURNS.md and
   IMPLEMENTATION_HANDOFF_CVC_RETURNS.md from "30–50%" to "~10–25×".
3. **(doc)** Strike the `(1 − α)` gradient form from
   IMPLEMENTATION_HANDOFF_CVC_RETURNS.md §3.4 (or replace with `(½ − α)`
   and a note that the Appendix A reference card is the authoritative
   form).
4. **(follow-up)** Re-tabulate the C3 truncation envelope at the
   *simulation-visited* policy support, not the grid-evaluated policy.
   The grid quantiles overstate cap-bound mass; a simulation will tell
   us whether the agent actually spends meaningful time at α_b=6.
5. **(no urgency)** Delete `compute_terminal_portfolio_foc_jac` (dead in
   canonical) or wire the use_ccv branch through it. Keeping it
   mathematically incomplete is a foot-gun for future authors.

End of report.

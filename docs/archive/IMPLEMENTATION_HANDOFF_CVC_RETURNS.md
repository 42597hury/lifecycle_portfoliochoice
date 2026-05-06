# IMPLEMENTATION HANDOFF — Campbell-Viceira log-return wealth dynamics

**Conversion of simple-return + bankruptcy-clamp wealth dynamics to the log-linearised (CCV/Campbell-Viceira-Chen 2003) formulation.**

You are the implementing agent. This document contains everything you need to make the change, plus a list of safety gates you must clear before merging. The change is structurally simple — the FOC integrand is replaced everywhere by a smooth integrand — but it touches many call sites in the kernel, the simulator, the inf-horizon solver, and the diagnostic suite. The risk surface is mostly in inconsistency (solver and simulator silently drifting apart) and in policy bundles on disk that were solved under the old spec.

Read sections 0-4 before touching code. Do not skip the theoretical review in §4.

---

## 0. What this is and what's expected of you

You will:

1. Read the linked theory review (`HANDOFF_THEORY_PORTFOLIO_FOC_PATHOLOGY.md`) and the response document `HANDOFF_THEORY_REVIEW_CVC.md`. They establish *why* this change is needed and bound the regime where it is valid.
2. Independently re-derive the formulas in §3 and confirm or challenge them. Do not just copy. The Itô correction in particular is signed and easy to flip.
3. Run the pre-commitment validation experiment (§7.1) **before any kernel rewrite**. If that experiment fails, stop and escalate.
4. Implement the change file-by-file as specified in §5, in the order specified in §8.
5. Run the validation gates in §7 and report results.

You will not:

- Remove the simple-return code path before validation passes. Keep both available behind a SolverConfig flag for at least one validation cycle.
- Re-tune calibration parameters (β, γ, b̄) as part of this change. Recalibration is a separate task.
- Make this change a kernel performance refactor at the same time. One thing at a time.

---

## 1. Pre-flight: read in this order

1. `HANDOFF_THEORY_PORTFOLIO_FOC_PATHOLOGY.md` — the original diagnosis. Establishes that the discontinuity in the P-FOC integrand at {sR_p = 0} is structural, not a bug.
2. `HANDOFF_THEORY_REVIEW_CVC.md` — the response document (written by the theory reviewer). This is the authority on whether the switch is benign for the lifecycle problem. Quote from the conclusion: *"adopt CVC; the economic content of the bankruptcy event was an artifact of grafting leverage onto a no-leverage calibration; tighten the leverage cap to ±4 to stay inside CVC's accuracy envelope."*
3. `Hypothesis_Specification.md` §3 (calibration table) and §5 (open questions). The headline result depends on ±4 leverage being adequate at γ ∈ {5, 7}, which both reviews confirm.
4. The CCV source: `w8566.pdf` §3.1 and Appendix A "Derivation of Equation (10)". This is the published derivation; refer to it whenever you need to confirm a formula.
5. The current spec in `solver.py:826-911` (retirement FOC kernel), `solver.py:1037-1152` (working-age FOC kernel), `solver.py:1212-1276` (unshifted terminal), `solver.py:1513-1570` (shifted terminal), and `simulation.py:745-770`.

---

## 2. The change in one paragraph

The current spec computes per-period gross portfolio return as a simple linear combination of gross asset returns, `R_p = α_s R_s + α_b R_b + (1-α_s-α_b) R_bill`, then clamps next-period wealth to `max(s·R_p, 0)` to handle the case where leveraged shocks drive `R_p ≤ 0`. The clamp introduces a discontinuity in the P-FOC integrand at the bankruptcy boundary {sR_p = 0}, which prevents quadrature from delivering publication-grade Euler residuals in leveraged cells. **The change is to compute the log portfolio return via the Campbell-Viceira approximation `r_p = r_bill + α_s·xr + α_b·xb + (1/2)α'(σ²_x − Σ_xx α)` and set `R_p = exp(r_p)`. The bankruptcy boundary disappears (R_p > 0 deterministically), the indicator branches are removed from the FOC, and the wealth transition becomes `x_{t+1} = s·R_p + π` unconditionally.**

The economic content lost is only the discrete-time-only "bankruptcy event" that is itself an artifact of the simple-return spec rather than a real economic feature; see §6.4 of `HANDOFF_THEORY_REVIEW_CVC.md`.

---

## 3. The math: side-by-side

### 3.1 Variable conventions

The codebase uses 3 return variables (`rtb, xr, xb`) where:
- `rtb` = real bill log return = `log(R_bill)`
- `xr`  = excess stock log return = `log(R_s) − log(R_bill)`
- `xb`  = excess bond log return = `log(R_b) − log(R_bill)`

Confirm at `var.py:386,400-402`. **This is already the CCV/Campbell-Viceira form.** No change to the VAR or to the return decomposition is needed.

The conditional-on-state covariance of `(rtb, xr, xb)` after projecting out state innovations is `Sigma_r_cond` in `model.py:80`, computed in `var.py:75` as `Sigma_rr - M @ Sigma_sr`. Define:

```
Σ_xx ≡ Sigma_r_cond[1:, 1:]      # 2x2 conditional covariance of (xr, xb)
σ²_xr ≡ Sigma_r_cond[1, 1]       # conditional variance of log excess stock return
σ²_xb ≡ Sigma_r_cond[2, 2]       # conditional variance of log excess bond return
σ_xrxb ≡ Sigma_r_cond[1, 2]      # conditional covariance
```

These are state-INDEPENDENT (the VAR has homoskedastic residuals).

### 3.2 The CCV log portfolio return

For a portfolio with risky shares `α = (α_s, α_b)` and bill share `1 − α_s − α_b`:

```
r_p,t+1 = r_bill,t+1
        + α_s · xr_residual + α_b · xb_residual
        + (1/2) [α_s · σ²_xr + α_b · σ²_xb]                          ← Jensen correction
        − (1/2) [α_s² · σ²_xr + 2·α_s·α_b·σ_xrxb + α_b² · σ²_xb]    ← Itô / vol-drag correction
```

where `xr_residual` and `xb_residual` are the realised (drawn) log excess returns including their conditional means at the current state — i.e., what the kernel currently calls `mu_xr + ret_nodes[k_r, 1]` and `mu_xb + ret_nodes[k_r, 2]`.

Equivalently, factoring:

```
r_p,t+1 = r_bill,t+1 + α_s·xr_residual + α_b·xb_residual
        + (1/2)·α_s·(1 − α_s)·σ²_xr − α_s·α_b·σ_xrxb + (1/2)·α_b·(1 − α_b)·σ²_xb
```

Then:

```
R_p,t+1 = exp(r_p,t+1)
```

`R_p > 0` is now a mathematical consequence of `exp(·) > 0`, not a property to be enforced by clamping.

### 3.3 Side-by-side at a single (k_v, k_r) quadrature node

Current code (`solver.py:867-870, 875-879, 905-911`):

```python
R_bill = exp_mu_bill * exp_ret_bill[k_r]
R_s    = R_bill * exp_mu_s * exp_ret_stock[k_r]
R_b    = R_bill * exp_mu_b * exp_ret_bond[k_r]
R_p    = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

sR_p = s_val * R_p
if sR_p > 0.0:
    x_next = sR_p + pension_next_scalar
    mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(sR_p, ...)
else:
    x_next = pension_next_scalar
    mu_bequest = 0.0
    mup_bequest = 0.0
```

New code:

```python
# Realised log returns at this quadrature node
log_R_bill = mu_r_bill  + ret_nodes[k_r, 0]    # = mu_rtb + eps_rtb
log_x_s    = mu_r_stock + ret_nodes[k_r, 1]    # = mu_xr  + eps_xr  (log excess stock)
log_x_b    = mu_r_bond  + ret_nodes[k_r, 2]    # = mu_xb  + eps_xb  (log excess bond)

# CCV log portfolio return
r_p = (log_R_bill
       + alpha_s * log_x_s + alpha_b * log_x_b
       + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
       - 0.5 * (alpha_s*alpha_s*sigma2_xr
                + 2.0*alpha_s*alpha_b*sigma_xrxb
                + alpha_b*alpha_b*sigma2_xb))
R_p = exp(r_p)

# Realised gross asset returns (still needed for Rex_s, Rex_b)
R_bill = exp(log_R_bill)
R_s    = exp(log_R_bill + log_x_s)
R_b    = exp(log_R_bill + log_x_b)

# Wealth transition is unconditional under CVC
sR_p   = s_val * R_p
x_next = sR_p + pension_next_scalar
mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(sR_p, ...)
```

Note: `Rex_s = R_s − R_bill` and `Rex_b = R_b − R_bill` are unchanged; they are not affected by the spec switch because the FOC moment condition is in `(R_j − R_bill)`, which is the same realised excess-return object as before. Only `R_p` changes.

### 3.4 Newton Jacobian — what changes

The portfolio FOC for asset j ∈ {s, b} is

```
FOC_j(α) = E[μ_comb(c_{t+1}, sR_p) · (R_j − R_bill)]
```

The dependence of the integrand on α now flows through:

1. `c_{t+1}` via `x_{t+1} = sR_p + π`, where `R_p = exp(r_p(α))`.
2. `μ_bequest(sR_p)`, same channel.
3. `(R_j − R_bill)` is **independent of α** under CVC (just like under simple returns).

Under CVC,

```
∂r_p/∂α_s = log_x_s + σ²_xr − α_s σ²_xr − α_b σ_xrxb
          = log_x_s + σ²_xr(1 − α_s) − α_b σ_xrxb

∂r_p/∂α_b = log_x_b + σ²_xb − α_s σ_xrxb − α_b σ²_xb
          = log_x_b + σ²_xb(1 − α_b) − α_s σ_xrxb
```

and `∂R_p/∂α_k = R_p · ∂r_p/∂α_k`. Compare to simple returns where `∂R_p/∂α_k = R_k − R_bill` (a constant in α).

The Jacobian becomes (per-node contribution):

```
∂FOC_j/∂α_k  =  weight · (R_j − R_bill) · {
                  μ'_alive · s · ∂R_p/∂α_k         ← alive branch (same structure as before)
                + μ'_bequest · s · ∂R_p/∂α_k       ← bequest branch (always lives now)
                }
              + 0   ← (R_j − R_bill) is constant in α
```

where `μ'_alive` and `μ'_bequest` are the existing `mup_*` quantities. **Important**: the existing kernel sets `jac = wmup * s_val` and then multiplies by `Rex_k * Rex_l`. That's the simple-return Jacobian. Under CVC you must multiply by `(∂R_p/∂α_k) * (R_j − R_bill)` rather than `Rex_k * Rex_l = (R_k − R_bill)(R_l − R_bill)`. Re-derive carefully.

### 3.5 EGM — the consumption-Euler inversion

The consumption Euler is unchanged in form:

```
c_t^(-γ) = β · E[ψ · μ_alive(c_{t+1}) · R_p + (1−ψ) · μ_bequest(sR_p) · R_p]
         = β · E[μ_comb · R_p]
         = β · euler_sum
```

What changes is that `R_p = exp(r_p^CVC)` instead of `α_s R_s + α_b R_b + (1−α_s−α_b) R_bill`. The kernel already accumulates `euler_sum += wmu * R_p` at each quadrature node (`solver.py:918, 1069, 1143, 1269, 1566`). Update those lines to use the new `R_p` and EGM works unchanged.

---

## 4. SANITY CHECK — please verify the theoretical claim before coding

Do the following three checks before writing any production code. Each should take under 30 minutes.

**Check 4.1 — Sign of the Itô correction.** Pick a portfolio with α_s = 1, α_b = 0 (full stock). The CCV log return reduces to `r_p = r_bill + xr + (1/2)·σ²_xr − (1/2)·σ²_xr = r_bill + xr`. So `E[r_p] = E[r_bill] + E[xr]` and `E[R_p] = E[exp(r_bill + xr)] = E[R_s]` (a tautology because at α = e_s the portfolio is just the stock). Confirm this. If your formula gives anything else, you have a sign error in the Jensen or Itô term.

**Check 4.2 — Jensen at α = (0.5, 0).** With α_s = 0.5, α_b = 0:

```
r_p = r_bill + 0.5·xr + 0.5·(0.5·σ²_xr) − 0.5·(0.25·σ²_xr)
    = r_bill + 0.5·xr + 0.125·σ²_xr
```

So `E[r_p | state] = E[r_bill] + 0.5·E[xr | state] + 0.125·σ²_xr`. The Jensen lift is +0.125·σ²_xr ≈ +0.125·0.025 = 31 bps. Verify your code reproduces this exactly at the unconditional state with α = (0.5, 0).

**Check 4.3 — Itô vol-drag at α = (0, 3).** With α_s = 0, α_b = 3 (high bond leverage):

```
r_p = r_bill + 3·xb + 0.5·(3·σ²_xb) − 0.5·(9·σ²_xb)
    = r_bill + 3·xb + 1.5·σ²_xb − 4.5·σ²_xb
    = r_bill + 3·xb − 3·σ²_xb
```

So at high bond leverage the vol-drag dominates. With σ²_xb ≈ 0.01, the drag is −3% on annual log return. This is a meaningful penalty and is exactly the CCV mechanism that "prevents bankruptcy at leveraged positions" by curving down expected log return at high |α|. Confirm.

**Check 4.4 — Σ_xx is the bottom-right 2×2 block of Sigma_r_cond.** Print `pc.Sigma_r_cond` and confirm row/column ordering matches `(rtb, xr, xb)`. Σ_xx for the CVC formula is `Sigma_r_cond[1:, 1:]`; σ²_xr is `Sigma_r_cond[1, 1]`; σ²_xb is `Sigma_r_cond[2, 2]`; σ_xrxb is `Sigma_r_cond[1, 2]`. Do not confuse with `Sigma_rr` (unconditional) — CCV uses the conditional covariance.

If any of 4.1–4.4 do not hold, stop and re-derive. Do not proceed to implementation.

---

## 5. Pipeline impact map

Tagged by file, with line-level pointers.

### 5.1 `model.py` — minor edits, document semantics

`bequest_utility` (line 285-298) and `bequest_marginal` (line 300-320) currently apply `np.maximum(W, 0.0)` inside the bequest. Under CVC, W = sR_p > 0 always, so the clamp becomes a no-op. **Keep the `np.maximum` for defensive coding** (catches extrapolation overshoots in interpolation) but update the docstring to note that under CVC the clamp is never expected to bind in solver call sites.

`DELTA_BEQUEST` (line 282) — keep at 0.005. Under CVC the δ shift is no longer mission-critical for boundedness because R_p > 0 means sR_p > 0 means b' is always finite. But δ remains useful to bound b' away from infinity at extreme small-sR_p outliers. Document the change in role: δ was *required* under simple+clamp (without it the marginal at the boundary spikes to ∞); under CVC δ is *defensive*. Run the existing δ ∈ {0.001, 0.005, 0.01, 0.02} sensitivity sweep again under CVC to confirm policy stability.

No structural change to `LifecyclePortfolioModel`. `Sigma_r_cond` is already on the model; you do **not** need new fields. The CVC corrections are computed from existing `Sigma_r_cond`.

### 5.2 `solver.py` — major edits, the core of the change

#### 5.2.1 Sites that need the simple-return → CVC swap

There are five FOC kernels with the old `R_p = α_s R_s + α_b R_b + a_bill R_bill` line:

| Function | Lines | Notes |
|---|---|---|
| `foc_quadrature_retirement_njit` (or whatever the retirement FOC kernel is named) | 826–928 | Indicator branches at 876–879 and 905–911. The `if sR_p > 0` dual-branch becomes a single unconditional branch. |
| `foc_quadrature_working_njit` | 935–1152 | Indicator branch at 1059–1080 (lines 1077–1080 set `w_inv = 0.0` on the bankrupt branch — this whole `else` block goes away). The alive branch already used the bankruptcy-clamped `w_inv = 0` to keep continuity in the alive integrand; under CVC we just set `w_inv = sR_p` always. |
| `compute_terminal_portfolio_foc_jac` (unshifted-CRRA terminal) | 1212–1276 | Uses `Rp_mg = max(R_p, min_return_power)**(-gamma)`. Under CVC, R_p > 0 strictly, so the `max(·, min_return_power)` clamp can be removed. **Caution**: keep a defensive numerical floor like `R_p_floor = 1e-15` to catch overflow if r_p underflows to a huge negative; but this should never bind in practice. |
| `compute_terminal_portfolio_foc_jac` (shifted-bequest variant) | 1513–1569 | Indicator branch at 1554. Same single-branch simplification. |
| Any other njit `_foc` you find by `grep -n "R_p = alpha_s" solver.py`. As of writing, there are 4 such sites in this file. |

For each site:

1. Replace the `R_p = α_s·R_s + α_b·R_b + a_bill·R_bill` computation with the CVC log-return formula, then `R_p = exp(r_p)`.
2. Remove the `if sR_p > 0:` indicator and its `else` branch. Set `x_next = sR_p + income_next` (or equivalent) unconditionally.
3. Update the Jacobian. The current `jac * Rex_k * Rex_l` term becomes `mup·s·(∂R_p/∂α_k)·(R_l − R_bill)` summed over assets. **Re-derive carefully — see §3.4.** Do not blindly substitute; the Jacobian quadratic form changes structure.
4. Keep `R_bill`, `R_s`, `R_b`, `Rex_s`, `Rex_b` as before — they're the realised gross returns at the quadrature node, used in the FOC moment condition `(R_j − R_bill)`. These are NOT changed by CVC.

#### 5.2.2 The shifted-bequest helper

`_shifted_bequest_mu_and_mup` at `solver.py:271-288`. The docstring at line 274–276 says "Caller must guard `W > 0`". Under CVC the caller no longer guards, but `W > 0` still holds because of `exp(·) > 0`. Update the docstring to: "Caller no longer needs to guard W > 0 because R_p = exp(r_p^CVC) > 0 strictly under the log-linearised wealth dynamics. The argument is expected to be positive; defensive callers may still wrap in `max(W, 1e-15)` to catch interpolation overshoots." The function body itself does not change.

#### 5.2.3 Newton solvers — `solve_terminal_portfolio_at_s_constrained_njit` etc.

Lines 1577 onwards. These solvers call the FOC/Jacobian functions above. As long as the FOC/Jacobian functions are correctly updated, the Newton iteration is unchanged in structure. **Do verify**: the convergence tolerance and step damping may need to be re-tuned. The CVC FOC has a different curvature — the Itô vol-drag adds α-quadratic terms that increase Newton's quadratic convergence basin. Empirically, you may find Newton converges *faster* under CVC, but if the line search trips, retune.

#### 5.2.4 EGM consumption inversion

Lines 2042 onwards (`solve_portfolio_2d_retirement_quad`) and 2378 onwards (`solve_portfolio_2d_working_quad`). The EGM step inverts `c = (β · euler_sum)^(-1/γ)` where `euler_sum = E[μ_comb · R_p]`. The structure is unchanged; `euler_sum` is computed inside the FOC kernel and passed back. No changes here unless the kernel signature changes (which it shouldn't).

#### 5.2.5 Leverage cap

`Hypothesis_Specification.md` ships at α ∈ [-6, +6]². The theory review recommends tightening the unconstrained cap to ±4. **Do not change the cap as part of this PR.** Make the change behind a config flag and benchmark separately. Mention the recommendation in the PR description so the next pass can act on it.

#### 5.2.6 The `min_return_power` floor

Search for `min_return_power` in solver.py; it's used in the unshifted terminal kernel. Under CVC this floor becomes unnecessary but it's harmless to keep. Either remove for clarity or keep for defensiveness — your call. Document in either case.

### 5.3 `inf_horizon_solver.py`

Two sites to touch:

1. **Main kernel** — the inf-horizon solver iterates `_run_infinite_horizon_core_jit` (line 145), which calls into the same retirement FOC kernel as `solver.py`. Once you fix the retirement kernel in solver.py, the inf-horizon solve picks up the CVC spec automatically. **Confirm by inspection** that no FOC computation is inlined inside `inf_horizon_solver.py`'s core jit function; if there is, mirror the change.

2. **`_compute_stability_proxy`** at line 544–585. This computes a contraction-mapping bound `β · E[R_p^(1−γ)]` using the simple-return formula at line 576. Under CVC this becomes `β · E[exp((1−γ)·r_p)]`. Update accordingly. Note that the `R_p ≤ 0 → return inf` guard at line 577–578 becomes vacuous under CVC; remove or replace with a `r_p > log(1e-15)` guard for defensiveness.

### 5.4 `simulation.py` — must mirror solver

This is the single most dangerous file in the change because solver and simulator must produce identical wealth dynamics. Under simple+clamp, both used `max(s·R_p, 0)`. Under CVC, both must use `s·exp(r_p^CVC)` with no clamp.

Sites to change at `simulation.py:745-770`:

```python
# Current (line 765-767):
alpha_bill_t = 1.0 - alpha_s_t - alpha_b_t
R_port = alpha_s_t * R_stock + alpha_b_t * R_bond + alpha_bill_t * R_bill
estate_t = max(savings_t * R_port, 0.0)
```

becomes

```python
# Realised log returns this period
log_R_bill = mu_rtb + rtb_res
log_x_s    = mu_xr  + xr_res
log_x_b    = mu_xb  + xb_res

# CCV log portfolio return
r_p = (log_R_bill
       + alpha_s_t * log_x_s + alpha_b_t * log_x_b
       + 0.5 * (alpha_s_t * sigma2_xr + alpha_b_t * sigma2_xb)
       - 0.5 * (alpha_s_t**2 * sigma2_xr
                + 2.0 * alpha_s_t * alpha_b_t * sigma_xrxb
                + alpha_b_t**2 * sigma2_xb))
R_port = np.exp(r_p)
estate_t = savings_t * R_port    # no clamp
```

The `sigma2_xr`, `sigma2_xb`, `sigma_xrxb` scalars must be threaded into the simulator the same way they are threaded into the solver kernel. Source them from `pc.Sigma_r_cond` once at simulator-entry and pass into the inner loop. They are constants, not state-dependent.

The `mu_rtb`, `mu_xr`, `mu_xb` quantities at `simulation.py:743-745` already include the M-coupling state-conditional means; they don't change.

This change applies to **both** the `use_mc_returns` branch (line 747-758, Monte Carlo integration over inflation/return shocks) and the discrete-quadrature branch (line 759-763). Both must produce log returns from the same CVC formula.

Subtle: `simulation.py:1347-1351` computes statistics on `sim_R_port`. Under CVC, `R_port` is now lognormal-shaped rather than a noisy linear combination. Distribution diagnostics may report different quantiles. This is expected. Update the interpretation, not the code.

### 5.5 `precompute.py`

You should not need to compute new quantities. `Sigma_r_cond` is already populated at `precompute.py:589`. Two small additions for kernel efficiency:

1. Add three scalars to `Precompute`: `sigma2_xr`, `sigma2_xb`, `sigma_xrxb`, populated at construction from `Sigma_r_cond[1, 1]`, `Sigma_r_cond[2, 2]`, `Sigma_r_cond[1, 2]`. This avoids indexing into Sigma_r_cond inside the inner kernel loop.
2. Optionally precompute the diagonal vector `sigma2_x = (sigma2_xr, sigma2_xb)` for easy passing.

No change to quadrature setup. The shock distribution is unchanged (it's still N(0, Sigma_r_cond) per the VAR); only the integrand changes.

### 5.6 `predictability_ablation.py`

This module sets up alternative VAR specifications (no-predictability, etc.) and dispatches to the same solver/simulation kernels. After the kernel change, this module picks up CVC automatically. **Verify by inspection** that no return computation is inlined here. Grep for `R_p` and `R_port` in this file; as of writing, there are no hits.

### 5.7 `var.py`, `discretization.py`, `quadrature_with_tails.py`, `numerics.py`

No changes expected.

- `var.py` constructs the VAR and produces `Sigma_r_cond`. The CVC formula consumes this; no change to construction.
- `discretization.py` builds quadrature for state innovations and return residuals. Under CVC the *same* shock distribution is integrated over a *different* (smooth) integrand. Quadrature setup is unchanged.
- `quadrature_with_tails.py` provides the Lobatto rule with explicit tail nodes. Unchanged.
- `numerics.py` contains PCHIP and bracketing helpers. Unchanged.

If you find yourself touching these, stop and ask why.

### 5.8 Diagnostics — many become obsolete or change meaning

Diagnostic files are one of two kinds: those that probe the bankruptcy-boundary pathology directly, and those that compute Euler residuals or policy quality metrics that depend on R_p.

**Becomes obsolete (probe a pathology that no longer exists):**

- `_diag_arbitrage_quadsweep.py` — checks for negative-R_p quadrature nodes. Under CVC there are none. Either delete or repurpose to check the truncation magnitude `|α|³ σ⁴` of the CCV approximation.
- `_diag_invalid_cells.py` — defines invalid cells by R_p ≤ 0. Same fate.
- `_diag_quadrature_cloud.py` — implements T-Q1...T-Q7 of the bankruptcy diagnosis. Either delete or repurpose.
- `_diag_simpath_worst_cells.py` — flags worst-case sR_p in simulation paths. Under CVC the criterion changes from `min(sR_p)` to something like `max(|α|·|shock|)` for tail-leverage diagnosis.
- `_diag_per_axis_tail.py`, `_diag_state_tail_node.py`, `_diag_tail_node_position.py` — tail-quadrature analyses tied to the boundary. Repurpose to check that the CCV truncation stays small at tail nodes.
- `_diag_bundle_state_clipping.py` — boundary-clipping check. Obsolete.
- `_diag_cap_vs_merton_overlap.py` — leverage-cap analysis tied to boundary. Repurpose to check |α|³σ⁴ bound at the cap.

**Must be updated to use CVC R_p:**

- `_diag_euler_errors.py` — recomputes Euler residuals on simulation paths. Currently uses simple R_p and `max(sR_p, 0)`. **This is the canonical Euler-residual reporter; it MUST mirror the solver/simulator exactly.** Update the R_p computation and remove the clamp.
- `_diag_split_rule_sanity.py` — uses simple R_p in a sanity check. Update.
- `_diag_gridpoint_ee.py` — gridpoint Euler residuals; check whether it computes R_p directly or pulls from sim_R_port. Update if direct.

**Likely unaffected (return-spec-agnostic):**

- `_diag_consumption_curvature.py` — concavity of c(w). Independent of R_p spec.
- `_diag_policy_convergence.py` — convergence of α(state) over Bellman iterations. Independent.
- `_diag_state_grid_coverage.py` — coverage of the state-quadrature cloud. Independent.
- `_diag_wealth_grid_tightness.py` — wealth-grid edge analysis. Independent (but see §6.5).
- `_diag_grid_quad_sweep.py` — sweep over discretization configs. Independent.

**Specifically relevant to CVC validation:**

- `_diag_quad_mgf.py` — moment-generating-function diagnostic. The CCV approximation IS a 2nd-order MGF expansion of `log E[R_p^simple]` around the conditional mean. This file is the natural place to add a check: at a few representative (state, α) cells, compute `E[R_p^simple]` and `E[R_p^CVC]` over the actual quadrature cloud and compare. The discrepancy should be O(|α|³σ⁴) and small wherever the converged policy actually lives. Extend this diagnostic into a CVC-truncation test.
- `_diag_age66.py` — terminal-age check. Update R_p computation.
- `_diag_merton_hc.py` — Merton-HC overlap. Update R_p computation.

### 5.9 `policy_io.py` and saved bundles

No code changes to `policy_io.py`. But:

**Saved policy bundles on disk are now stale.** Any bundle solved before this change was solved under simple+clamp dynamics; running it through the new simulator (which uses CVC) creates a solver/simulator mismatch and meaningless residuals.

Add a version tag to saved bundles (e.g., a `wealth_dynamics_spec` key with values `"simple_clamp"` or `"ccv_log"`). The simulator should refuse to run a `"simple_clamp"` bundle through the CVC simulator path and vice versa. Make this a hard error, not a warning.

For the regression-test corpus, regenerate all reference bundles under CVC. Document the change in the test fixtures' README so future maintainers know the cutover date.

---

## 6. Important subtleties and gotchas

### 6.1 Σ_xx is the CONDITIONAL covariance, not the unconditional

`Sigma_rr` (line 70 of var.py) is the *unconditional* covariance of (rtb, xr, xb). `Sigma_r_cond` is the *conditional* covariance after projecting out the state innovations. CCV's Σ_xx in equation (10) is the conditional one. **Use `Sigma_r_cond[1:, 1:]`, not `Sigma_rr[1:, 1:]`.** The two differ by ~30-50% in this VAR (because state predictability explains a meaningful share of return variance), which is a 30-50% error in the Itô vol-drag term — large enough to materially shift the converged policy.

### 6.2 The bill is risky in this model — does CVC still apply?

Yes. CCV's r_{1,t+1} is the realised log riskless return in their derivation; it is allowed to be stochastic (as in the nominal-bill-with-inflation case). The Itô / Jensen corrections are over the *excess* log returns x = (xr, xb) only — not over the bill innovation. The bill innovation rides through r_p unchanged. Confirm by re-reading w8566.pdf Appendix A "Derivation of Equation (10)" (the σ_b in their derivation is the bill diffusion, which is allowed to be non-zero).

### 6.3 The wealth-grid lower edge

Under CVC, sR_p is positive but can be very small at extreme-leverage + extreme-shock cells. For example, at α_b = 4 with a -3σ bond shock (xb_residual ≈ -3·σ_xb ≈ -0.3), the bond term is 4 × (-0.3) = -1.2; with vol-drag at α_b = 4, additional -8·σ²_xb ≈ -0.08; so r_p ≈ r_bill - 1.28. R_p ≈ exp(-1.28) · exp(r_bill) ≈ 0.28 · R_bill ≈ 0.29.

So sR_p can plausibly drop to 30% of s. If the wealth grid's lowest point is too high (e.g., wealth_grid[0] = 0.1·permanent_income), interpolation of c_{t+1}(x_{t+1}) at low x_{t+1} extrapolates and produces noisy μ_alive values. This was masked under simple+clamp because the bankrupt branch zeroed out the bequest path entirely. Under CVC the alive branch sees the small-wealth values and feeds them back into the Euler equation.

**Action**: verify `pc.wealth_grid[0]` is at least one decade below `pc.wealth_grid[1]`. If not, add a log-spaced lower tail to the wealth grid. The theory review (§6 of `HANDOFF_THEORY_REVIEW_CVC.md`) flags this explicitly.

### 6.4 Bequest motive at extreme leverage — economic content not lost

The theory review (§2 of `HANDOFF_THEORY_REVIEW_CVC.md`) walks through this. Under simple+clamp, P(bankrupt | leveraged) was non-trivial and delivered b(0) ≈ −4×10⁹ utils — a fictitious, oversized fear-of-bankruptcy that has no real-world counterpart. Under CVC, the agent still leaves a small but positive bequest at extreme states, and the bequest motive still does work. Do not introduce a separate "extreme loss" punishment to "preserve" the bankruptcy fear. The point of the change is to remove that artifact.

### 6.5 Consistency check between solver and simulator

This is the highest-risk inconsistency. After the change:

```python
# In an end-to-end test, at fixed (state, α, shock):
solver_R_p = compute_R_p_inside_FOC_kernel(state, alpha, shock)
sim_R_p    = compute_R_port_inside_simulation(state, alpha, shock)
assert abs(solver_R_p - sim_R_p) < 1e-10
```

This **must** be a regression test. Add it to the test suite. If it ever fires, you have introduced a silent inconsistency — solve and sim use different specs — and any Euler residual diagnostic becomes meaningless.

### 6.6 Numerical stability of exp(r_p)

`r_p` is bounded by O(|α|·|shock|) − vol-drag. At α = (6, 6) with -7σ shocks (the Lobatto tail), r_p could plausibly be around -10 or -15, giving R_p ≈ 1e-5 to 1e-7. This is tiny but well-defined; no overflow. At positive 7σ shocks, r_p could reach +5 to +8, giving R_p ≈ 100-3000. Also fine.

The dangerous direction is overflow at extreme positive r_p with wealth scaling. Compute `s·R_p` and check it doesn't exceed `wealth_grid[-1]` by more than a factor of 2. If it does, the Euler equation interpolates at the wealth grid's upper edge and you get extrapolation noise.

**Action**: add a max-wealth-grid coverage check to the diagnostics. Should already exist in `_diag_wealth_grid_tightness.py`; verify the upper-edge case is covered.

### 6.7 Working-age: alive branch was already continuous

Under simple+clamp, the working-age FOC's *alive* branch was continuous in α and shocks (because the bankruptcy clamp set `w_inv = 0` and labor income kept the agent fed, so c_{t+1}(income_next) is continuous across the boundary). The discontinuity was entirely from the bequest branch. Under CVC, both branches become unconditional. The structural simplification is real but the working-age discontinuity was always smaller than retirement's. Do not be surprised if working-age Euler residuals improve less dramatically than retirement's.

### 6.8 The unshifted-bequest terminal kernel

`compute_terminal_portfolio_foc_jac` at line 1212 uses unshifted CRRA bequest with `R_p^(1-γ)` — the original spec. Under CVC the `max(R_p, min_return_power)` clamp becomes unnecessary but the kernel is otherwise unchanged in form. The Jacobian still picks up the CCV ∂R_p/∂α terms.

---

## 7. Validation plan — must complete before merging

### 7.1 Pre-commitment test (run BEFORE any kernel rewrite)

Per `HANDOFF_THEORY_REVIEW_CVC.md` §9: solve **M-no-labor at γ = 5, constrained (α ∈ [0, 1]²)** under both specs side-by-side.

Under constrained α, sR_p > 0 always under both specs (no leverage means α'·gross-returns is a convex combination of positive quantities), so the bankruptcy clamp never binds. The only difference between the two specs is the Jensen + Itô corrections, which at α ∈ [0, 1]² are small.

**Decision rule**: if the converged optimal stock share differs by more than ±3 percentage points at any age, OR the optimal bond share differs by more than ±2 percentage points at any age, the spec change is shifting the economics in a way that requires recalibration of γ or β. Stop and escalate.

If both differences stay within those bands, CVC is benign for the constrained baseline and you have green light to rebuild the kernel.

### 7.2 Unit tests (during implementation)

For each FOC kernel changed:

1. **Sanity at α = e_s** (full stock): r_p = r_bill + xr exactly; Jensen + Itô cancel.
2. **Sanity at α = e_b** (full bond): r_p = r_bill + xb exactly; Jensen + Itô cancel.
3. **Sanity at α = 0** (full bill): r_p = r_bill; corrections are zero.
4. **Jacobian consistency**: at a few random (α, state) points, finite-difference the FOC numerically and compare to the analytical Jacobian. Should match to 1e-6.
5. **Solver-simulator R_p consistency**: at the same (state, α, shock), solver and sim produce identical R_p to 1e-10. (See §6.5.)

### 7.3 Integration test — Euler residuals

After the full pipeline change, regenerate all bundles in the test corpus and run `_diag_euler_errors.py`. The headline expectation is mean log₁₀|EE| ≤ −4.5 at retirement-age cells, max ≤ −3.5. (Currently mean −2.57, max −0.08.)

If you don't reach −4.5 mean, do not assume the spec change failed. Check:

1. Wealth-grid lower edge (§6.3).
2. Wealth-grid upper edge (§6.6).
3. Newton convergence tolerance.
4. Quadrature-node tail magnitude.

These are mechanical fixes. If after addressing all of them you still can't reach −4 mean, escalate.

### 7.4 Hypothesis-relevant tests

Per `Hypothesis_Specification.md`:

1. **M vs M-no-labor gap** at γ = 5, constrained: should be qualitatively unchanged from any pre-CVC preliminary results. The gap is a comparative; symmetric spec changes don't affect it. If the gap reverses sign or shrinks dramatically, escalate.
2. **Catherine-IRR bond hump** at γ = 5, M, constrained: should still appear. The Itô vol-drag will slightly attenuate the peak height (theory review §4); this is expected.
3. **Figure 5 sanity check** — M-no-labor unconstrained at γ ∈ {5, 10}: under CVC, the equity allocation should land *closer to* CCV Table 5 numbers than it did under simple+clamp. This is the cleanest "CVC is benign" signal because CCV themselves use CVC units.

### 7.5 Regression — saved bundles

Re-solve and re-simulate the entire policy bundle library used in figure generation. Compare to pre-CVC bundles at *constrained* baseline; differences should be within ±3 pp on stock share, ±2 pp on bond share at every age. Larger differences in the *unconstrained* variants are expected and OK (this is precisely where CVC differs).

---

## 8. Order of operations

Do these in order. Do not skip ahead.

1. **Read** §0–§4 and the linked theory documents.
2. **Run §4 sanity checks** on a scratch script. Verify your formulas before touching production code.
3. **Run §7.1 pre-commitment test** under the existing simple+clamp code path, then build a one-off side-by-side that runs the same problem with a hand-coded CVC R_p. This is a 1-day experiment with no kernel rewrite.
4. **If §7.1 passes**, proceed. If it fails, escalate.
5. **Add `Sigma_r_cond` accessors** to `Precompute` (§5.5).
6. **Update one FOC kernel first** — start with the **shifted-bequest terminal kernel** (`solver.py:1513-1569`). It's the smallest, the bankrupt branch is the cleanest indicator example, and it has no EGM coupling. Verify §7.2 unit tests on this kernel.
7. **Update the retirement FOC kernel** (`solver.py:826-928`). Run the retirement portion of the lifecycle solver in isolation. Compare policies at constrained baseline to step 3's hand-coded CVC.
8. **Update the working-age FOC kernel** (`solver.py:935-1152`). Run the full lifecycle solve. Verify §7.4 (hypothesis-relevant tests).
9. **Update the unshifted terminal kernel** (`solver.py:1212-1276`).
10. **Update `simulation.py`** (§5.4). Verify §6.5 solver-sim consistency test.
11. **Update `inf_horizon_solver.py:_compute_stability_proxy`** (§5.3). Run the inf-horizon variant.
12. **Update / repurpose / delete diagnostics** (§5.8). Update `_diag_euler_errors.py` first because it's the canonical reporter.
13. **Tag saved bundles** (§5.9) and refuse to mix specs.
14. **Run §7.3, §7.4, §7.5** validation. Report results in the PR.

If at any step a validation fires, stop and diagnose. Do not "soldier on" past a failed gate.

---

## 9. Out of scope for this change

Do NOT include in this PR:

- Recalibration of β, γ, b̄, ρ, or any other model primitive. The theory review allows that constrained results are preserved without recalibration; if you find otherwise, that's a finding to report, not an action item.
- Tightening the leverage cap from ±6 to ±4. The theory review recommends it; do that in a follow-up PR with its own experiment.
- Removing the simple+clamp code path. Keep it behind a `SolverConfig.wealth_dynamics_spec` flag for at least one validation cycle.
- Performance refactoring of the kernel (numba inlining, SIMD, etc.). The CVC formula is a few extra FLOPs per node; if there's a performance regression, it's your problem, but solve it in a follow-up.
- Changes to the VAR, the labor-income process, the mortality calibration, or the bequest annuity factor.
- Changes to quadrature node placement or counts.

---

## 10. PR description checklist

When you open the PR, the description must include:

- [ ] Confirmation that §4.1–§4.4 sanity checks pass.
- [ ] Result of §7.1 pre-commitment test (M-no-labor γ=5 constrained, both specs side-by-side, age-by-age share comparison).
- [ ] Mean and max log₁₀|EE| on retirement-age simulation paths, before and after.
- [ ] Reference policy bundle: stock share difference (max over ages) at γ ∈ {3, 5, 7, 10}, constrained and unconstrained.
- [ ] Confirmation that solver-simulator R_p match to 1e-10.
- [ ] List of diagnostics deleted, repurposed, or updated.
- [ ] Confirmation that saved bundles are tagged with `wealth_dynamics_spec`.
- [ ] Open follow-up items (leverage cap tightening, simple+clamp removal).

---

## Appendix A — Formula reference card

```
σ²_xr  = pc.Sigma_r_cond[1, 1]
σ²_xb  = pc.Sigma_r_cond[2, 2]
σ_xrxb = pc.Sigma_r_cond[1, 2]

# At quadrature node (k_v, k_r), state-conditional means already computed:
log_R_bill = mu_r_bill   + ret_nodes[k_r, 0]
log_x_s    = mu_r_stock  + ret_nodes[k_r, 1]
log_x_b    = mu_r_bond   + ret_nodes[k_r, 2]

r_p = (log_R_bill
       + alpha_s * log_x_s + alpha_b * log_x_b
       + 0.5 * (alpha_s * σ²_xr + alpha_b * σ²_xb)
       - 0.5 * (alpha_s**2 * σ²_xr
                + 2*alpha_s*alpha_b * σ_xrxb
                + alpha_b**2 * σ²_xb))

R_p   = exp(r_p)

dr_p_dalpha_s = log_x_s + σ²_xr * (1 - alpha_s) - alpha_b * σ_xrxb
dr_p_dalpha_b = log_x_b + σ²_xb * (1 - alpha_b) - alpha_s * σ_xrxb

dR_p_dalpha_s = R_p * dr_p_dalpha_s
dR_p_dalpha_b = R_p * dr_p_dalpha_b

# Gross asset returns (unchanged from current spec; needed for Rex_*)
R_bill = exp(log_R_bill)
R_s    = exp(log_R_bill + log_x_s)
R_b    = exp(log_R_bill + log_x_b)
Rex_s  = R_s - R_bill
Rex_b  = R_b - R_bill

# Wealth transition (unconditional under CVC)
sR_p   = s_val * R_p
x_next = sR_p + income_next   # no max(·, 0) clamp
```

End of handoff.

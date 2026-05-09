# Economic Setup Review — Part B (Returns / VAR / Portfolio / State Dynamics)

**Date:** 2026-05-09
**Branch:** `jax-rewrite`
**Scope:** Read-only review of the post-pivot real-yields lifecycle model. Focus
on asset-return structure, VAR specification, CCV log-portfolio formula, state
dynamics, portfolio constraints, and internal consistency. (Utility / lifecycle
/ income covered by Part A.)

---

## 1. Asset return structure

**What I found.**
- `solver.py:840-842` and `simulation.py:334`: `log_R_bill = state_t[y_1_idx]` —
  deterministic in current state, no v^s shock channel.
- `data/build_var_dataset.py:177` constructs `y_1_real = y_1_nom - E_t[π_{t+1}] -
  λ_n` with `LAMBDA_BP = 0` (build_var_dataset.py:81).
- `data/build_var_dataset.py:188-192`: bond return is the **CLM constant-duration
  real holding-period return** `r_n_real_t = D_n·y_t - (D_n-1)·y_{t+1}` with
  `n=20` (`N_BOND=20`); `xb = r_n_real - y_1_real_lag`.
- `data/build_var_dataset.py:181`: `cape = -log(Shiller CAPE)` — i.e. log
  earnings yield (positive predictor of stock returns), correct sign.
- `xr = log((P+D)/P_lag) - y_1_NOM_lag` (build_var_dataset.py:186): excess over
  the NOMINAL bill, justified by CCV §2.1.5 invariance.

**Verdict: UNDOCUMENTED (inflation-expectation channel) + CLEAR (bond, CAPE).**

The bill is *not really* risk-free in real terms. `y_1_real_t = y_1_nom_t -
E_t[π_{t+1}]` is the *expected* real return; the *realized* real return is
`y_1_nom_t - π_{t+1}` and differs by the inflation surprise. The model imposes
that this surprise equals zero by treating `y_1_t` as the realized one-period
real bill return — i.e. it suppresses inflation-uncertainty risk on the bill
leg. This is the standard Campbell-Viceira / CGM real-rate convention but is
not documented explicitly in either `solver.py` or `build_var_dataset.py` as
an *assumption* (it is implicit in the Fisher decomposition step).

The bond construction is internally consistent with this: `xb = r_n_real -
y_1_real_lag` uses Fisher-deflated yields on both legs, so xb's residual
volatility absorbs both real-yield innovations and bond-term-premium shocks
(but again no inflation-surprise channel). The CLM `D_n·y_t - (D_n-1)·y_{t+1}`
is the correct one-period holding-period log-return approximation for a
constant-duration n=20 bond, NOT just yield-to-maturity. Good.

**Suggested fix.** Add a one-paragraph docstring in `solver.py:9-13` (the
"Real-yields pivot" comment) noting that the real bill is risk-free *only
under the rational-expectations Fisher restriction* `realized π = E_t[π]` and
that inflation-surprise risk is shut down on the bill leg. Reviewer will ask.

## 2. VAR specification

**What I found.**
- `lifecycle/var.py:191-268`: VAR(1) with linear conditional mean. The estimator
  uses the **CCV constrained estimator** (mean-pinning §2.2.μ + lagged-return-
  zeroed §2.2.r per `build_real_full_var_config_hardcoded` docstring 421-423).
- `lifecycle/var.py:418-447`: canonical = `build_real_full_var_config`, columns
  `[cape, spr, y_1, xr, xb]`, state idx (0,1,2), return idx (3,4),
  `estimation="restricted"` (default at line 412).
- `var.py:530-545`: hardcoded fallback documents sample 1920-2011 (T=92), max
  |eig(Φ)| = 0.9525 (stationary), R²: cape=0.80, spr=0.67, y_1=0.62, xr=0.07,
  xb=0.25.
- `precompute.py:517-518`: conditional return mean uses `mu_r[i,j] = (Phi_0_ret
  - M·Phi_0_state) + (Phi_21 - M·Phi_11) @ s_i + M @ s_j` — i.e., conditional
  on BOTH s_t AND s_{t+1} via Cholesky decomposition of `Sigma_rs Sigma_ss⁻¹`.

**Verdict: CLEAR but POTENTIAL DUPLICATION CONCERN.**

Annual frequency is implicit (`year` index in CSV; `chap_26` Shiller annual
data). The VAR is stable (max |eig| = 0.9525), so multi-year IRFs are
well-defined. R²(xr) = 0.07 is on the low end of the equity-predictability
literature (Welch-Goyal: in-sample R² ~0.05-0.10) but reasonable.

**POTENTIAL DUPLICATION:** The repo has TWO real-yields data builders:
`data/build_var_dataset.py` (output: `var_dataset.csv`, columns `cape, spr,
y_1, xr, xb`, headline used by `var.py:DEFAULT_CSV_PATH`) and
`data/build_var_dataset_real.py` (output: `var_dataset_real.csv`, columns
`y_1, spr, dp, xr, xb` — note `dp` not `cape`). The latter has `dp` = log
dividend-price ratio, which used to be the equity predictor in the legacy
nominal pipeline. The two files have similar names and overlapping logic;
a reader could easily import the wrong one. `build_var_dataset_real.py`
appears to be a stale artefact that was superseded by `build_var_dataset.py`
during the pivot.

**Suggested fix.** Delete `data/build_var_dataset_real.py` (and downstream
`var_dataset_real*.csv`) or move them to `data/legacy/`; OR add a top-of-file
warning that they're not the headline.

## 3. CCV log-portfolio formula

**What I found.** `solver.py:747-769` and `simulation.py:356-365` implement
the identical formula:
```
r_p = log_R_bill + α_s·log_x_s + α_b·log_x_b
      + 0.5·(α_s·σ²_xr + α_b·σ²_xb)
      − 0.5·(α_s²·σ²_xr + 2·α_s·α_b·σ_xrxb + α_b²·σ²_xb)
```
Source of σ²_xr, σ²_xb, σ_xrxb: `precompute.py:340-342` reads from
**`model.Sigma_rr` (UNCONDITIONAL return covariance)**, NOT `Sigma_r_cond`.
Patch comment at precompute.py:331-339 explicitly justifies this on CCV
w8566 grounds: the Itô vol drag is in expectation over the FULL VAR
innovation, so the unconditional Σ_rr is correct.

**Verdict: CLEAR.** Signs and cross-term `2·α_s·α_b·σ_xrxb` correct. The
derivation: for log-normal returns, `log E[R_p] = E[r_p] + 0.5·Var(r_p)`,
and the Jensen-correction-plus-vol-drag combination
`+0.5·α·σ² − 0.5·α²·σ²` reduces to `0.5·α(1−α)·σ²` (i.e. the variance
contribution of fraction α of the risky asset). The diagonal solver and
simulator are bit-identical (parity test mentioned `verify/ccv_solver_sim_parity.py`
at `simulation.py:355`).

The choice of `Sigma_rr` vs `Sigma_r_cond` for the vol-drag is the most
delicate design choice in the formula; the inline comment at `precompute.py:
331-339` is excellent documentation.

## 4. State dynamics under VAR

**What I found.**
- `precompute.py:281-298`: `build_state_grid` constructs the state grid using
  `Phi_0_state, Phi_11, Sigma_ss` (state-block stationary distribution) with
  `state_grid_mode="cholesky"` (canonical, `_canonical.py:91`).
- State quadrature (`precompute.py:311-321`): `M_v_nodes = v_nodes @ M.T` — the
  state innovations are routed into return shocks via M = Σ_rs · Σ_ss⁻¹, the
  feedback channel. This means returns are integrated jointly over (v_state,
  ε_residual_return), not just residuals.
- `_validate_state_quadrature` (precompute.py:526-547): asserts that
  Σ_k w_k · μ_r_k = Φ_0_ret + Φ_21·s_i to 1e-10.
- `state_n_stds = (2.0, 2.25, 2.25)` and `n_state_quad_nodes = (3, 3, 5)`
  per `_canonical.py:92,99`. K-bump on y_1 axis preserved (last axis).

**Verdict: UNDOCUMENTED — `state_n_stds` per-axis values.**

The canonical comment at `_canonical.py:64-68` explicitly admits: "PLACEHOLDERS
per the real-yields pivot handoff. Validate before production by re-running
the System I / Full sensitivity sweeps on the new VAR." So the per-axis values
2.0/2.25/2.25 are inherited from the legacy 4-axis nominal model and have NOT
been validated on the new (cape, spr, y_1) state vector. The state grid
truncation directly affects the tails of the wealth-and-policy distribution.

The K-bump (3,3,5) on the y_1 axis is also a heuristic carry-over: the
bond-return predictor M[xb, ·] is concentrated on y_1 (the dominant entry of
M), so the y_1 axis dominates xb's conditional mean — bumping K_state[y_1] to
5 reduces integration error there. This is documented at `_canonical.py:70-72`.

**Suggested fix.** Per the existing TODO at `_canonical.py:64-68`, run the
state-grid axis sensitivity sweep (`scripts/analysis/state_grid_axis_sensitivity.py`
already exists) on the new VAR and update n_stds / K-bump accordingly before
the headline thesis solve. Without this validation a reviewer can flag the
state-grid choice as ad-hoc.

## 5. Constraints on portfolio

**What I found.**
- `solver.py:7`: "Unconstrained portfolio (no simplex projection, no leverage
  caps)." Confirmed by absence of any `alpha_min`/`alpha_max` clipping in the
  Newton kernel (`_ccv_log_return_and_grad`, `terminal_foc_jac_ccv`).
- `simulation.py:776`: `alpha_bill = 1 − alpha_s − alpha_b` — fully
  unconstrained, can be negative (i.e., shorting the bill / margin lending of
  bills to fund risky positions).
- `inf_horizon_solver.py:107` has `_project_simplex_nonnegative` for a
  *separate* infinite-horizon solver branch (`solver_pi_z_variant.py`); the
  finite-horizon canonical does not use it.

**Verdict: UNDOCUMENTED at the assumption level (CLEAR at the code level).**

There are no portfolio constraints — α_s and α_b can take any real value, and
α_bill can be negative (funding leverage by borrowing at the bill rate). This
matches CCV (2003) but is a non-trivial economic assumption: the model's
optimal portfolio can prescribe levered positions of |α| > 1 (which is
typically observed empirically in the JV / CGM literature). There is no FOC
"constrained" branch — the unconstrained Newton interior solution is taken
even at corner.

**Suggested fix.** Add a one-line note in the thesis methodology section that
the model permits unrestricted leverage at the real bill rate (no margin /
short-sale constraint). Reviewer's first question: "do you allow shorting
bills?" — answer must be yes by construction.

## 6. Internal consistency

**What I found.**
- `precompute.py:340-342`: σ²_xr, σ²_xb, σ_xrxb from `Sigma_rr` (matched on
  xr/xb name lookup at `precompute.py:326-327`, robust to permutation).
- `solver.py:780-786,975-976,1083-1084`: same scalars passed into every FOC
  kernel (terminal, working, retired branches).
- `simulation.py:730-732`: same scalars passed into the simulator kernel.
- Predictor coefficients: `precompute.py:319-321` sources `const_r =
  Phi_0_ret`, `A_r = Phi_21` from the partitioned VAR; the same arrays flow
  to both the solver (`solver.py:817 const_r, A_r`) and simulator
  (`simulation.py:340 const_r + A_r @ s_t`). No drift between the two.
- Fallback consistency: `var.py:559-586` hardcoded fallback returns the
  identical structure as the live estimator output.

**Verdict: CLEAR.** Solver, simulator, and Euler-residual paths all read
σ_xrxb / Σ_rr / Φ_21 from the same precompute fields; the partition is
done once (`build_model` in `precompute.py:708-856`) and frozen.

---

## TL;DR

| # | Area                       | Verdict      | Severity |
|---|----------------------------|--------------|----------|
| 1 | Bill return / inflation    | UNDOCUMENTED | medium — reviewer will ask if real bill is truly risk-free |
| 2 | VAR specification          | CLEAR + duplication risk | low — stale `build_var_dataset_real.py` artefact |
| 3 | CCV log-portfolio formula  | CLEAR        | none |
| 4 | State dynamics / quad.     | UNDOCUMENTED | medium — `state_n_stds` placeholders, validation TODO open |
| 5 | Portfolio constraints      | UNDOCUMENTED | low — unconstrained leverage by design, just needs a thesis-text note |
| 6 | Internal consistency       | CLEAR        | none |

**Single most-important reviewer flag:** The real bill is treated as risk-
free, but `y_1_real = y_1_nom − E_t[π]` is the *expected* real return;
realized real return diverges by the inflation surprise. The model implicitly
assumes rational-expectations equality of expected and realized inflation
(the Fisher restriction with λ_n = 0). This is a defensible, standard CCV /
CGM convention but is not stated as an assumption anywhere visible in
`solver.py` or in the canonical config docs — a thesis reviewer will notice
the absent inflation-uncertainty channel on the bill leg and ask why.

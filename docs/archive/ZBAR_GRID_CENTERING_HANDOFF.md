# HANDOFF: z_bar / Grid Centering Problem

## The Problem

The state grid (Rouwenhorst) is centered on `z_bar = (I - Phi)^{-1} @ const`, the VAR's implied stationary mean. This diverges significantly from the sample mean:

| Variable | Sample mean | z_bar | Ratio |
|----------|-------------|-------|-------|
| rtb      | 0.70%       | 0.08% | 9x too low |
| y_nom    | 5.80%       | 4.50% | 22% too low |
| dp       | -3.66       | -3.92 | shifted by 0.27 |

The bill rate is the most affected: the agent's median-state bill return is 0.08% instead of 0.70%. This makes bills systematically unattractive, producing near-zero bill allocation across most of the lifecycle.

## Root Cause

The VAR is estimated on 1962–2025 annual data (T=63). The state variables y_nom and dp have secular trends over this period (y_nom fell from 14% to 4%, dp fell from -3.0 to -4.5). The VAR treats these as stationary AR(1) processes, but the implied stationary mean `(I - Phi)^{-1} @ const` is pulled away from the sample average by these trends.

This is NOT a coding bug. The OLS estimation is correct — mean fitted rtb = 0.685% matches the data. The intercept `const` and coefficients `Phi` are verified. The problem is that `(I - Phi)^{-1} @ const` is a poor estimate of the "typical" state when the data has trending components.

## What Is Confirmed Correct

Everything downstream of the grid placement has been audited:

1. **Data construction** — rtb formula, timing, sample statistics all correct
2. **VAR OLS estimation** — Phi, const, Omega match direct verification
3. **Partition algebra** — Phi_0, Phi_11, Phi_21, M, Sigma_ss, Sigma_r_cond all correct
4. **Gauss-Hermite quadrature** — 19 moment checks pass (innovation mean, covariance, third moments, return formula identity, etc.)
5. **Conditional return formula** — `mu_r_k = Phi_0_ret + Phi_21 @ s_i + M @ v_k` correct at every state
6. **Solver FOC and timing** — R_bill = exp(rtb_current), CCV timing convention, zero Newton failures

The problem is isolated to: **where the grid is placed in the state space**.

## What the Grid Centering Affects

1. **Which states the agent evaluates** — with z_bar as center, the median grid point has rtb = 0.08%. The agent's "normal" bill return is nearly zero.
2. **Grid resolution** — grid points are wasted on unrealistic low-rtb / low-y_nom states that rarely occur in the data.
3. **Unconditional expectations** — if the agent starts at the grid midpoint in simulation, their expected lifetime bill return is 0.08% instead of 0.70%.
4. **Policy functions** — the entire policy surface (C, S, B as functions of state × wealth × income) is evaluated at the wrong state locations.

## Key Code Locations

- **Grid construction:** `discretization.py`, `rouwenhorst_multivariate()` — line 115: `mu_bar = np.linalg.solve(np.eye(k) - Phi, mu)` centers grid on z_bar
- **z_bar computation:** `var.py`, line 215: `z_bar = np.linalg.solve(np.eye(len(columns)) - Phi, const)`
- **Partition that produces Phi_0_state:** `var.py`, `partition_var()` — line 82: `Phi_0_full = (np.eye(n) - Phi_full) @ z_bar`
- **Precompute passes Phi_0_state as mu:** `precompute.py`, line 122: `mu=model.Phi_0_state`
- **Hardcoded z_bar:** `var.py`, line 677: `_Z_BAR` array
- **Model stores z_bar:** `model.py`, `LifecyclePortfolioModel` fields `z_bar_state`, `z_bar_ret`

## VAR Parameters for Reference

```
rtb equation:
  rtb_{t+1} = -0.0802 + 0.432*rtb_t + 0.553*y_nom_t - 0.014*dp_t + eps

Sample means (1962–2025, T=64):
  rtb:   0.006973 (0.70%)
  xr:    0.054920 (5.49%)
  xb:    0.019450 (1.95%)
  y_nom: 0.057772 (5.78%)
  dp:    -3.670204

z_bar from (I-Phi)^{-1} @ const:
  rtb:   0.000792 (0.08%)
  xr:    0.050059 (5.01%)
  xb:    0.021552 (2.16%)
  y_nom: 0.044959 (4.50%)
  dp:    -3.923637

cond(I - Phi) = 242.9  (amplifies small errors)
```

## What Needs to Be Decided

The core question: **what should the grid be centered on?**

### Option 1: Center on sample means

Replace z_bar with sample means for grid construction only. The VAR dynamics (Phi, const, Sigma) stay unchanged. The Rouwenhorst `rouwenhorst_univariate(N, mu_bar, rho, sigma)` takes `mu_bar` as the grid center — just pass sample means instead of `(I - Phi_11)^{-1} @ Phi_0_state`.

Pros: Simple. Grid covers the historically observed region.
Cons: Grid center no longer matches the VAR's stationary distribution. Marginal Rouwenhorst transition probabilities are derived assuming the grid center IS the stationary mean — changing the center without adjusting probabilities breaks the moment-matching property of Rouwenhorst. But we use quadrature for transitions, not Pi_state, so this may not matter.

### Option 2: Demean the VAR

Estimate the VAR in deviations from sample means: `(z_t - z_bar_sample)`. Then:
- Phi is unchanged (same regression slopes)
- const becomes zero by construction
- z_bar = 0 (the demeaned stationary mean)
- Grid is centered at 0 (demeaned), but the model adds z_bar_sample back when computing actual returns

Pros: Clean separation between dynamics (Phi, Sigma) and levels (sample means).
Cons: Need to carry z_bar_sample through the code and add it back everywhere returns and bill rates are computed.

### Option 3: Use CCV convention directly

Campbell, Chan & Viceira (2003) work with demeaned variables and add means back when computing portfolio returns. Check exactly how CCV handle this and follow their convention.

### Option 4: Revisit the VAR specification

The non-stationarity in y_nom and dp may suggest the VAR specification needs adjustment (e.g., include a trend, use first-differenced yields, or shorten the sample to a more stationary period). This is a bigger change.

## Constraints

- The solver, simulation, and all downstream code expect `model.Phi_0_state`, `model.Phi_0_ret`, and `pc.state_grid` in their current form. Any fix must be consistent across all these.
- The quadrature integration uses `Phi_0_state + Phi_11 @ s_i + v_k` to compute next states. If the grid is re-centered, this formula must still give correct next-state values.
- `Phi_0_state` appears in the conditional return formula: `mu_r = Phi_0_ret + Phi_21 @ s_i + M @ v_k`. The `const_r = Phi_0_ret` and `A_r = Phi_21` precomputation assumes v_k are zero-mean innovations — this must remain valid.
- The bill rate is read as `r_bill_grid[i_s] = state_grid[i_s, 0]`. If the grid is re-centered, these values must reflect actual rtb levels, not demeaned values.
- Simulation uses `Phi_0_state + Phi_11 @ s_t + v` for continuous state propagation (simulation.py, line 504-507).

## Validation That Should Have Been Done (and now must be)

After any fix, verify:
1. `mean(state_grid[:, 0])` ≈ sample mean rtb (0.70%)
2. `mean(state_grid[:, 1])` ≈ sample mean y_nom (5.78%)
3. Unconditional variance from Lyapunov equation matches sample covariance
4. At the median grid point, E[rtb_next] ≈ sample mean rtb
5. Long simulation (10k agents, 78 years) produces mean rtb ≈ 0.70%
6. Bill allocation is non-trivial during at least some working-age periods

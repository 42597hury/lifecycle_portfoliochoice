# HANDOFF — Unconstrained solver produces pathological leverage

**Status:** open. Simulator-side ruled out. Convex-hull arbitrage (H1a) ruled out 2026-04-29. **Bundle was re-solved 2026-04-30 with `init_alpha=(0.85, 0.44)`; H1b is no longer present and H4 (multi-optimum FOC) was the dominant cause at the median state.** Median α_s at age 22 dropped from 4.92 → 0.92 (close to Merton). Tail extremes (p99 α_b ≈ 39, max α_b = 82) remain at corner states (extreme cy). Diagnosed as joint K_state=2 + K_xb=3 under-truncation (T10/T11) compounded with H9 (numerical floor sensitivity at corners). Sim wealth pathology in §1 was measured on the pre-re-solve bundle; **re-simulate on the new bundle before chasing further structural fixes**.
**Owner-hand-off-to:** investigation continues at the corner. Quick wins: (a) re-simulate to quantify residual pathology, (b) bump K_state from 2 → 3 or 4, (c) bump K_xb from 3 → 7 or 9 in production. The corner residual may still be H1b on the refined cloud — confirm by re-running T9 after refinement.
**Last updated:** 2026-04-30 evening (added bundle re-solve note, T9-full / T10 / T11 / T12 / T13 findings — see §8).

---

## 1. The problem

The unconstrained CRRA solver at `γ=3` with the partitioned VAR returns produces equity allocations that are an order of magnitude beyond any economically reasonable benchmark, and the resulting simulated wealth distribution is orders of magnitude above empirical referents.

### Quantified evidence

Saved policy bundle: `saved_runs/unconstrained_principal_grid5x5x5_nz9` (5×5×5 state grid, n_z=9, `n_ret_nodes_1d=(3,5,3)`, `n_state_quad_nodes=2`, `state_n_stds=2.0`, `state_grid_mode="principal"`, `gamma=3`, `constrained=False`).

Simulation at 5,000 households, MC-mode return draws, with the patched simulator (state-trilinear, flat wealth extrapolation, Y_67 boundary fix, survival/pension linear-interp):

```
                              alpha_s p25/p50/p75   median x   median x in 2019 $
age 22 (initial leverage)     [4.92, 4.92, 4.92]    0.25       $13,700
age 60 (mid-retirement entry) [—, ~5x, —]           4,455      $241 million
age 99 (terminal)             —                     ~4e7       ~$2 trillion
```

`alpha_s ≈ 4.92` at age 22 means ~5× stock leverage funded by ~4× short bills. By age 60, median household wealth is **$244 million** in 2019 dollars (1 model unit = SSA AWI ≈ $54,100, see `contextfiles/LABOUR.md` §0). For comparison, US Survey of Consumer Finances median household wealth at age 60 is roughly **$200–300 thousand** — a factor of ~1,000× discrepancy.

### Sanity check that isolates the cause to the policy, not the simulator

Same simulator, same calibration, same income process, **same C_mat**, but with portfolio shares forced to a fixed 50/50 stock/bond split, produces:

```
age 40: median x ≈ 1.53 model units = $82,900
age 60: median x ≈ 2.36 model units = $127,800
age 99: median x ≈ 4.46 model units = $241,500
```

Median wealth at age 60 of $128k lands within the calibration-error band of SCF empirics (slightly below because the unconstrained-optimal C_mat under-consumes — using a 50/50-optimal C_mat would consume more aggressively and pull median down further toward SCF). Off-grid wealth share peaks at 1.4% (vs 92% under unconstrained). So the simulator is producing the right wealth distribution when given a reasonable portfolio. **The defect is in the solver's portfolio policy.**

### Closed-form benchmark (Merton-Samuelson)

Using the unconditional excess-return moments from `contextfiles/RETURNS.md` §4.1:

```
σ_xr = 15.89%,  σ_xb = 7.63%,  ρ_xr,xb ≈ 0.29
μ_xr = 5.55%,   μ_xb = 1.43%

α* = (1/γ) Σ_e^{-1} μ_e ≈ [0.67, 0.41]
```

Classical Merton at γ=3 implies `α_stock ≈ 0.67`, `α_bond ≈ 0.41`, `α_bill ≈ −0.08`. The constrained solve lands at roughly this neighbourhood (~0.6 stock, ~0.3 bond, ~0.1 bill at age 22). The unconstrained solve is ~7× the Merton answer. Hedging demand from VAR predictability would normally modify Merton by 10–50%, never 700%.

---

## 2. What's been ruled out

These were checked during the simulator review and are not the cause:

- **Simulator state interpolation.** Patched to trilinear-in-s × linear-in-z × linear-in-x, matching the solver's continuation-value lookup at `solver.py:bracket_state_3d`. Verified bit-exact on saved bundle.
- **Wealth-domain extrapolation.** Was linear, now flat. A 2,000-household run on the unconstrained bundle stays finite (no NaN/Inf). The flat-extrapolation diagnostic correctly fires at >5% off-grid in the unconstrained run; it stays silent under 50/50.
- **Y_67 income timing.** Solver assumed pension at age 67; simulator was passing labor income. Now linearly z-interpolates `pension_after_tax[t+1, :]` at the boundary, bit-exact match to `solver.py:651–654`.
- **Continuous-z survival and retirement pension.** Both now use the same `iz_lo/frac_z` linear interp the policy lookup uses. Bit-exact match to the solver's discrete-z evaluation.
- **Income mixtures.** Continuous mixture draws for η and ε with `mu_eta2_eff`, `mu_eps2_eff` enforcing E[η]=E[ε]=0; matches `discretization.get_eta_quadrature_mixture` / `get_eps_quadrature_corrected` distribution.
- **Initial conditions.** `x_0 = wealth_0 + Y_0` with continuous-mixture Y_0; verified.

The simulator now reports faithfully what the solver's policies produce. **Do not modify `simulation.py` further during this investigation** unless one of the tests below directly motivates it.

---

## 3. Hypotheses (leads), ranked by expected diagnostic yield

### H1. Discrete quadrature constrains the support — two related mechanisms

**H1a (convex-hull arbitrage).** Documented in `contextfiles/RETURNS.md` §6.12. If the convex hull of `{(R_stock^(n) − R_bill^(n), R_bond^(n) − R_bill^(n))}_n` over the `n_state_quad · n_ret_quad` nodes does not contain the origin, a separating direction `(d_s, d_b)` makes `d · X^(n) ≥ 0` at every node — a discrete free lunch. Unconstrained CRRA at γ=3 then has no interior optimum.

> **Investigation 2026-04-29 finding (see §8 below):** convex-hull arbitrage was tested across all 125 states at age 22 and found ZERO arbitrage states (H1a is ruled out at the production config). The pathology persists, so the cause must be H1b or downstream.

**H1b (no-bankruptcy boundary on bounded support).** Distinct from arbitrage. Even when the origin is *inside* the convex hull, the discrete cloud has *bounded support*: there exists a smallest `R_port` over the `n_state_quad · n_ret_quad` nodes at any candidate `(α_s, α_b)`. Unconstrained CRRA at γ>1 levers up until `min_n R_port_n` reaches ~0, beyond which `c_next^{−γ}` at the worst node clamps to the wealth-grid floor and the FOC sign flips discontinuously. The continuous-Gaussian counterpart of this problem has *unbounded* support and a finite interior optimum (bounded by variance × MU); the discrete approximation does not.

> **Investigation 2026-04-29 finding:** at age 22, iz=4, all 125 states, W=200, the saved policy has `min_n R_port_n ∈ [4e-7, 1e-5]` — essentially clamped to zero at the worst quadrature node. A 1% scaling of `(α_s, α_b)` flips the FOC by 13+ orders of magnitude. The plateau leverage *is* this boundary.

The cure for H1b is structurally different from H1a: more nodes don't fix it, they just push the boundary. A proper fix requires either (i) a borrowing-constraint argument that respects natural debt limits (Carroll), (ii) a continuous-distribution moment-matching scheme rather than node-by-node quadrature, or (iii) acceptance that unconstrained policies be interpreted only as upper bounds.

The current production config `n_ret_nodes_1d=(3, 5, 3)` was specifically chosen to eliminate H1a at γ=3 according to a comment in `main.ipynb`. H1b is independent of node count to first order.

### H2. Multi-period EGM machinery amplifies a per-period defect

Even if the per-period quadrature is locally arbitrage-free, the EGM continuation-value interpolation across (z, s, x) can imply a value function whose curvature differs systematically from the true value function — driving the per-period FOC toward leverage that the one-period Merton problem on the same cloud wouldn't choose.

### H3. Spurious VAR predictability through state grid

The VAR is estimated on T=63 annual obs. The OLS R² for `xr` is 5.91% but the cross-equation residual structure (96% of `xr` variance "explained by state conditioning" in `RETURNS.md` §4.1) is conditional on `v^s_{t+1}`. The state innovation isn't observable to the agent at decision time, so this isn't actual predictability — but if the partition_var() bookkeeping is wrong somewhere, the agent might effectively be "seeing" `v^s` as if it were observable, dramatically reducing perceived risk.

### H4. Multi-optimum FOC

The unconstrained 2D Newton from `init_alpha_s=0.1, init_alpha_b=0.4` may converge to a stationary point that is not the global optimum. The default initial guess is far from the saved policy's `(4.92, 0.09)`; the solver got there through line-search-bounded steps over up to 5,000 iterations. There may be other local optima at lower leverage that the default starting point can't reach.

### H5. Calibration-internal feature

The model genuinely wants this leverage at this calibration. Unlikely (Merton + hedging demand can't justify 7× the textbook answer), but worth ruling out cleanly.

### H6. State-innovation quadrature K_state=2 truncates the v^s tail

Production uses `n_state_quad_nodes=2` → 2³ = 8 joint nodes for the 3-D state innovation `v^s ~ N(0, Σ_ss)`. Two-point Gauss-Hermite places nodes at ±1σ in standardized units (with weight 0.5 each). The discrete representation of v^s is therefore truncated at the ±σ box — the worst-case state-innovation contribution `M @ v_k` to the conditional return mean is bounded at ±|M·σ_v|, never the ±2.5σ to ±3σ a real Gaussian agent would face.

This shrinks `min_n R_port_n` and lets the unconstrained Newton lever further before hitting the H1b boundary. Distinct from H1b in mechanism (this is about the *state* quadrature, H1b is about the *return-residual* quadrature) but compounds with it: refining `n_state_quad_nodes` should push the bankruptcy boundary inward.

Note: K_state=2 is exact only to polynomial degree 3 (matches mean and variance). Higher moments of M·v_k are not captured. With M[xb, y_1] = −8.7 and M[xb, spr] = −8.5 (RETURNS.md §4.1), the tails of M·v_k are large in absolute terms — a 1σ y_1 shock alone moves expected bond return by ~14pp.

### H7. Continuation-value interpolation across grid points smooths future flexibility

Three interpolation layers stack between current FOC and future continuation:

1. Trilinear in `(y_1, spr, cy)` over the 5³ state lattice
2. Linear in z over the 9-point z-grid
3. Linear in W over the 150-point geometric wealth grid

Across the 125 states the *saved policy itself* spans `α_s ∈ [−0.1, +12.3]` and `α_b ∈ [−9.2, +18.0]` (from the prior investigation, age 22 W=200). Trilinear blending of these extremes at off-grid intermediate states gives a future-self policy that is a *weighted average of extremes* rather than the optimal-at-intermediate policy.

The agent today therefore overestimates future flexibility: they think "if I land at intermediate state, my future self will mix the extreme policies optimally" when the reality is "my future self will run a different, less extreme optimization at the new state." This biases today's value function upward, which lowers today's effective MU on bad outcomes, which permits more current leverage.

Distinct from H2 in scope (H2 is about EGM machinery generally; H7 is specifically about the smoothing introduced by trilinear state interpolation).

### H8. Conditional Sharpe ratio is artifactually high after state conditioning

`Σ_r_cond` (residual return covariance after conditioning on `v^s`) has stock std 3.10% and bond std 2.26% (RETURNS.md §4.1). With `μ_xr = 5.55%` and `μ_xb = 1.43%`, the conditional Sharpe ratios are ~1.79 and ~0.63. Compare:

- Unconditional Sharpe (full Σ_rr): stock 0.35, bond 0.19 — empirically reasonable.
- Conditional Sharpe (Σ_r_cond): stock 1.79, bond 0.63 — extraordinary, well above any historical estimate.

The agent at decision time integrates over both v^s (8 nodes) and ε_r (45 nodes), so the *integrated* uncertainty equals Σ_rr in expectation. But the integration is pointwise per `(k_v, k_r)` and the FOC gradient at each node uses `c_next^{−γ}` evaluated at that node's specific R_port. If MU is highly nonlinear (which it is at γ=3 with R_port near 0), the agent effectively sees a "best of both worlds": the conditional mean from the state, plus a small residual.

Worth a sanity check: sum the variance contribution of M·Σ_ss·M' and Σ_r_cond and verify it equals Σ_rr. If not, there's a bookkeeping bug in `partition_var()` or the quadrature construction.

### H9. Wealth grid floor at 1e-4 is the implicit "softener" on the H1b boundary

`wealth_grid[0] = 0.0001`. At the H1b boundary, the worst quadrature node delivers `x_next = s_val · R_port_n + Y_next`. For tiny R_port and tiny labor income (rare-component ε at low z), `x_next` clamps to the floor and `c_next` is interpolated at the floor — finite and bounded. So `c_next^{-γ}` at the worst node is bounded by `c_next(W=1e-4)^{-3}` instead of diverging.

The choice of floor implicitly determines how aggressive the bankruptcy boundary feels to the FOC integrator. A lower floor (1e-8) would let MU grow ~10⁹× more at the worst node, shrinking the H1b plateau. A higher floor (1e-2) would soften it and permit more leverage.

This is a *parameter* of the discretization, not the model; the saved policy depends on it in ways that are easy to miss. Distinct from H1b in that H1b is about quadrature support; H9 is about the *interpolation domain* on which c_next is evaluated.

---

## 4. Tests, in priority order

Each test should produce a single, comparable diagnostic number. Where possible, write tests as standalone scripts that load `saved_runs/unconstrained_principal_grid5x5x5_nz9` rather than re-solving (saves ~5 min per iteration). When a fresh solve is required, use a small smoke grid (3×3×3 state, n_z=5, n_w=40, n_state_quad=2, n_ret_nodes_1d=2; ~10s solve).

Each test specifies an **acceptance criterion** that says what the result rules in or out.

### T1. Closed-form Merton at a single state vs solver α

Pick three representative `(t, i_z, i_s)` triples: median state `(t=0, i_z=n_z//2, i_s=N_state//2)`, top-z state `(t=0, i_z=n_z-1, i_s=N_state//2)`, and bottom-z state `(t=0, i_z=0, i_s=N_state//2)`. At each, compute analytically:

```
μ_e_at_s = E[r_excess | s_t=s]   (using Phi_0_ret + Phi_21·s_grid[i_s])
Σ_total_at_s = M·Σ_ss·M' + Σ_r_cond  (constant in s)
α*_merton = (1/γ) · Σ_total[1:, 1:]^{-1} · μ_e[1:]   (excluding bills)
```

Then look up `S_mat[t, i_z, i_s, i_w_median], B_mat[t, i_z, i_s, i_w_median]` from the saved bundle.

**Acceptance:**
- If `|α_solver / α_merton| ≤ 1.5`: model genuinely wants this leverage at this calibration. Move to T5 (γ sensitivity) and T6 (no-predictability) to confirm.
- If `|α_solver / α_merton| > 3`: solver is producing leverage the per-period continuous problem doesn't imply. Move to T2 (discrete-cloud Merton).
- Report at all three states; if behavior differs across states, note where it's worst.

**Effort:** ~1 hour. Self-contained script; uses model.M, model.Sigma_ss, model.Sigma_r_cond, pc.state_grid, model.Phi_0_ret, model.Phi_21.

### T2. Discrete-cloud Merton at the same state

At the same three states, build the joint quadrature cloud the solver actually integrates over:

```
For each (k_v, k_r) in product(n_state_quad, n_ret_quad):
  weight_n = v_weights[k_v] * ret_weights[k_r]
  μ_r_n   = const_r + A_r·s_grid[i_s] + M·v_nodes[k_v] + ret_nodes[k_r]
  R_bill_n  = exp(μ_r_n[0])
  R_stock_n = R_bill_n * exp(μ_r_n[1])
  R_bond_n  = R_bill_n * exp(μ_r_n[2])
```

Then solve the one-period CRRA portfolio problem on this discrete cloud (use `scipy.optimize.minimize` on `−Σ_n weight_n · u(α_s · R_stock_n + α_b · R_bond_n + (1−α_s−α_b) · R_bill_n)`, no constraints).

**Acceptance:**
- If `α_discrete ≈ α_merton`: discrete cloud faithfully represents the continuous distribution. Solver leverage is a multi-period EGM artifact (H2). Move to T7.
- If `α_discrete ≈ α_solver`: per-period discrete cloud already produces the leverage. Solver isn't multi-period-amplifying anything; it's faithfully integrating a misshapen cloud. Move to T3 (localize the cause in the cloud).
- If scipy.optimize fails to converge / diverges to budget: cloud has actual arbitrage. Definitive H1.

**Effort:** ~1–2 hours. Builds on T1 script.

### T3. Convex-hull arbitrage diagnostic

Implement `convex_hull_arb_gap(i_s)` for each grid state, as described in `RETURNS.md` §6.12:

```
For state i_s:
  cloud[n, :] = (R_stock_n - R_bill_n, R_bond_n - R_bill_n)  for n in joint quadrature
  Solve LP: find d ∈ R² with ||d|| = 1 maximizing min_n d · cloud[n, :]
  gap = max_d min_n d · cloud[n, :]
  arbitrage exists iff gap > 0
```

Report:
- Total count of states with `gap > 0`, out of `n_z * N_state * n_age` (≈ 87,750 for production).
- Histogram of gap values (max, p99, median).
- Where (which `(t, z, s)` corners) the worst gaps live.

**Acceptance:**
- Zero arbitrage states: H1 ruled out, look at H2/H4. (Surprising outcome given the symptoms.)
- Some arbitrage states (>0% but localized): H1 is partially the cause. The cure is per-state quadrature refinement or grid pruning — see RETURNS.md §6.12 closing paragraphs.
- Many arbitrage states (>5–10%): H1 is the dominant cause. Need to aggressively raise `n_ret_nodes_1d` (especially `K_xr`, principal axis) and/or widen `state_n_stds`.

**Effort:** ~3–4 hours. The LP is 2D so it can be solved in closed form (find the support direction along which the convex hull is tangent to the origin).

### T4. Constrained-constraint binding pattern

Solve the **constrained** model (`constrained=True`) at the same discretization as the unconstrained bundle. Tabulate, by age, the fraction of `(z, s, x)` states where the simplex constraint binds at `α_s + α_b = 1` (no bills) or at `α_s = 1, α_b = 0` (corner). Use the solver's existing exit codes (`EC_CORNER_*`, `EC_EDGE_*`, `EC_INTERIOR`) — already in `diagnostics`.

**Acceptance:**
- Binds at >50% of states across all young ages: unconstrained interior optimum is uniformly above the simplex. Consistent with H1 (uniform discretization arbitrage) but also possible under H5 (model just wants leverage everywhere).
- Binds only at specific corner states: localized to those corners, which T3 should also flag. Cross-check.
- Doesn't bind much: the constrained policy is already in the interior, the unconstrained should land near the same place. If unconstrained gives 5× leverage and constrained doesn't bind, something is structurally wrong with the unconstrained solver code — escalate.

**Effort:** ~30 min if the constrained solve already exists in saved_runs; 5–10 min fresh solve at smoke-grid otherwise.

### T5. γ-sensitivity sweep

Re-solve the unconstrained model at `γ ∈ {3, 5, 8, 10}` at smoke-grid scale. Extract `α_s(t=0, median z, median s, median x)` from each.

**Acceptance:**
- α_s scales as `~ 3/γ · α_s(γ=3)` (Merton-like): the model is internally consistent and just risk-tolerant. H5 confirmed; the calibration choice of γ=3 is the issue, not the solver.
- α_s stays > 3 at γ=10: not a risk-aversion-driven leverage. Almost certainly H1 or H2.

**Effort:** ~30 min (3 smoke-grid solves). Note `RETURNS.md` §4.8 warns the ε quadrature wall starts at γ ≥ 5; you may need `n_eps_nodes ≥ 6` to keep this clean.

### T6. No-predictability ablation

Re-solve unconstrained with `Φ_21 = 0` (zero out the state→return loadings) and `M = 0` (zero out the conditioning matrix; equivalently, zero `Σ_rs`). Conditional return mean reduces to the unconditional mean `Φ_0_ret`. Re-simulate.

**Acceptance:**
- α_s ≈ 0.7 at age 22 (Merton): the predictability machinery + discrete grid is the amplifier. Cure is in the predictability discretization (more nodes along principal axis of `M·Σ_ss·M'`).
- α_s still ≫ 1: predictability isn't the source. Look harder at the constant-mean discrete return cloud — even with `Φ_21=0` the residual quadrature could have arbitrage if `n_ret_nodes_1d` is too low.

**Effort:** ~30 min (one fresh solve). Easy to wire by passing modified Phi_21 and M into a hand-built LifecyclePortfolioModel — no need to re-estimate the VAR.

### T7. Multi-start unconstrained Newton

Patch `SolverConfig` to allow multiple starting points; re-solve from `(α_s, α_b) ∈ {(0.0, 0.0), (0.5, 0.5), (5.0, 0.0), (-3.0, 0.5)}` at one specific `(t, z, s, x)` and report convergence.

**Acceptance:**
- All four converge to the same `(α_s, α_b)`: solver is finding a unique stationary point. H4 ruled out.
- Different starts converge to different optima: H4 confirmed. The default `(0.1, 0.4)` produces a saddle point or non-global stationary point. Need a better initial guess heuristic or explicit value-function comparison.

**Effort:** ~1 hour. The solver already supports `init_alpha_s, init_alpha_b` overrides via `SolverConfig`.

### T9. min(R_port) at saved policy across (age, z, state, wealth)

Direct test of H1b. For each cell in the saved bundle, compute over the joint quadrature:

```
min_R_port[t, iz, i_s, iw] = min over all (k_v, k_r) of
    α_s_saved · R_stock_n + α_b_saved · R_bond_n + α_bill_saved · R_bill_n
```

Report:
- Fraction of cells where `min_R_port < 1e-3` (boundary cells).
- Distribution of `min_R_port` across cells.
- Map: at what age and wealth bracket does the boundary first start binding?

**Acceptance:**
- High fraction of boundary cells at young ages, low at old ages: H1b is the dominant cause; the saved policy is a no-bankruptcy boundary at young ages and an interior optimum at old ages. The "ramp from 35 to 22" the user observed is actually the transition from interior to boundary as horizon lengthens.
- Boundary cells uniformly across all ages: H1b binds everywhere; problem is structural to the unconstrained discretization.
- No boundary cells: H1b is ruled out — find another lead.

**Effort:** ~30 min. No fresh solve. Direct loop over saved C/S/B with the precomputed `(v_nodes, ret_nodes)` arrays.

> **Pre-existing finding (2026-04-29):** at age 22, iz=4, W=200, all 125 states are within 1e-2 of the boundary (`min_R_port ∈ [4e-7, 1e-5]`). At W=0.005, no states are at the boundary. The full `(t, iz, i_s, iw)` map has not yet been built — *do this first*.

### T10. State-quadrature refinement at the saved policy

Direct test of H6. For one representative cell `(t=0, iz=4, i_s=62, iw=149)`, recompute the working-age FOC at the saved `(α_s, α_b)` using a refined `K_state ∈ {3, 4, 5}` (giving 27, 64, 125 joint state nodes respectively). The saved policy is fixed; only the FOC integrand changes.

Report:
- FOC value (for stock and bond) at saved policy under each K_state.
- `min_n R_port_n` over the refined quadrature at saved policy.
- The implied "what α would zero the FOC" under each refined K_state.

**Acceptance:**
- FOC magnitude grows monotonically with K_state and the implied optimal α drops materially (e.g., from 4.86 → 3.0 at K=5): H6 confirmed; the K_state=2 truncation is doing real work.
- FOC stays similar across K_state values: H6 ruled out; the state quadrature isn't truncating meaningful tails.

**Effort:** ~1 hour. Self-contained Python script using the saved policy + recomputed quadrature.

### T11. Bond-residual refinement at the saved policy

Same idea as T10 but for bond residuals. Current `n_ret_nodes_1d=(3,5,3)` puts only 3 nodes on bond residual. Recompute FOC at saved policy with `(3,5,5)`, `(3,5,7)`, `(3,5,9)`. Focus on the bond-loaded states (i_s ∈ {48, 72, 96, 120} where `α_b ≥ 14`).

Report:
- FOC values and implied optimal α under each refinement.
- `min_n R_port_n` over the refined quadrature at saved bond-loaded policy.

**Acceptance:**
- Optimal α_b drops materially at bond-loaded states with K_xb=7 or 9: bond-residual coarsening is part of the boundary problem at those states. H1b confirmed for bond-loaded corners; need to refine K_xb in production.
- Optimal α_b unchanged: bond residual node count is sufficient; the boundary is set elsewhere.

**Effort:** ~1 hour, builds on T10 script.

### T12. Wealth-floor sensitivity

Direct test of H9. Re-evaluate the *saved* working-age FOC with a different effective `wealth_min` floor: clamp `c_next` interpolation lookup to `max(W', floor)` for `floor ∈ {1e-6, 1e-4, 1e-2, 1e-1}`. The saved policy is fixed; only how MU diverges at the worst quadrature node changes.

Report: implied optimal α for stock and bond at one representative cell, under each floor.

**Acceptance:**
- Optimal α changes by >50% as floor varies over four orders of magnitude: the policy is sensitive to a parameter that should be a numerical convenience, not an economic input. Indicates an open economic boundary that is being implicitly closed by the floor.
- Optimal α changes by <10%: the floor is genuinely numerical convenience and the policy is set by economic forces.

**Effort:** ~30 min, builds on T9 script.

### T13. Conditional Sharpe ratio diagnostic

Direct test of H8. Compute, at every state grid point:

```
μ_e[i_s] = Phi_0_ret + Phi_21 · state_grid[i_s]   (3-vector: rtb, xr, xb)
Σ_total = M · Σ_ss · M' + Σ_r_cond                (constant 3×3, ~Σ_rr)
Σ_resid = Σ_r_cond                                 (constant 3×3)

Sharpe_total[i_s] = μ_e[1:] / sqrt(diag(Σ_total[1:, 1:]))
Sharpe_resid[i_s] = μ_e[1:] / sqrt(diag(Σ_resid[1:, 1:]))
```

Report:
- Distribution of unconditional/conditional Sharpe across 125 states.
- Verify `Σ_total ≈ Σ_rr` (numeric agreement is a sanity check on `partition_var()`).
- Compute the implied Markowitz allocation at each state under both Σ choices.

**Acceptance:**
- `Σ_total` agrees with `Σ_rr` to machine precision: bookkeeping is correct, H8 is just an artifact of how I framed it. The agent's effective return distribution is the unconditional one.
- `Σ_total` differs from `Σ_rr`: bookkeeping bug. Track down the source of the discrepancy.
- Markowitz allocation at residual Σ matches saved policy more closely than at total Σ: agent is effectively optimizing as if state innovation were known, which it shouldn't be. Indicates conditioning bug in solver loop structure.

**Effort:** ~45 min. No solver changes; pure diagnostic on `model` attributes.

### T8. Policy-field sanity plot

Plot `α_s(t, z, s, x)` along four 1D slices, holding the other three constant at median. Look for:
- Monotonicity in x (typically α decreases with wealth in CRRA)
- Smoothness in z (no jumps at z-grid points)
- Smoothness in s (no jumps when crossing state-grid points)
- Lifecycle profile in t (typically α decreases with age)

Saved bundle policy arrays support this directly.

**Acceptance:**
- Smooth and monotone: solver converged to a self-consistent surface; cause is at the per-period level.
- Jumps or non-monotonicity: EGM/interpolation has localized failures. Cross-check those locations against T3's arbitrage states.

**Effort:** ~30 min using `plots.py` infrastructure.

### T14. Continuation-value interpolation accuracy

Direct test of H7. The `C_mat` slice at `(t+1, iz, :, :)` records optimal consumption at the 5³=125 state lattice points. At off-grid states `s_next = Φ_0_state + Φ_11·s_t + v_k`, the solver bilinearly/trilinearly blends. To measure the smoothing error:

1. Pick one off-grid `s_next` (e.g., midpoint between i_s=62 and i_s=63 in the principal-coord lattice).
2. Compute `c_interp` via the standard trilinear blend.
3. Re-solve the *single-age* portfolio problem at this exact `s_next` on a fine 1×1×1 lattice (one-shot solve, ~1 s) and read off the "true" consumption policy at that point.
4. Compare `c_interp` to the true `c_*`.

Repeat at 8–10 representative off-grid states (mostly midpoints, plus a few principal-axis extremes).

Report:
- `(c_interp − c_true) / c_true` distribution.
- Same for the implied future α (use the resolved policy gradient).

**Acceptance:**
- Median error < 5%: trilinear interp is fine; H7 ruled out.
- Median error 5–20%: interp error is non-negligible but unlikely the dominant cause; flag for future grid refinement work.
- Median error > 20%: interp is materially distorting the continuation value. Saw this happen at corner states (i_s=0, 124) in the prior investigation where saved policies span huge ranges; a single-cell resolve at midpoints would confirm.

**Effort:** ~3–4 hours, requires standing up a one-shot single-age solver for arbitrary `s_next`. Lower priority unless T9–T13 don't produce a clear answer.

### T15. Z-grid pension and income interp accuracy

Related to H7 but specific to the z-axis. The η innovation is heavily skewed (skew=−1.7). Tail draws fall outside the uniform z-grid and clamp. At the boundary of the z-grid, the linear interp on `c_next_full` and on `pension_next_by_z` (working-to-retirement transition) is inaccurate.

Test: at iz=0 (lowest z), evaluate η quadrature nodes that push z_next below the grid. Compare clamped-linear `c_next` and `pension_next_by_z` to the analytic value at the actual z_next (recomputed by the formula, not interp). Report worst-case error.

**Acceptance:**
- Worst-case error < 5%: clamping doesn't matter; H7 ruled out for z-axis.
- Worst-case error > 15%: the rare-component η at the lowest z grid points is inaccurately handled. Could affect young agents at low-z states; cross-check at saved policy.

**Effort:** ~2 hours. Not directly relevant to the leverage plateau, but bears on overall accuracy.

---

## 5. Recommended one-day execution order

Updated 2026-04-30 after the prior investigation ruled out H1a (T3) and confirmed H1b at one slice (T9 partial). The starting picture is: **the saved policy at age 22, high W, is on the no-bankruptcy boundary in 100% of state cells tested**. Boundary mechanism is not arbitrage. The remaining question is whether refining quadrature/grids relaxes the boundary toward Merton, or whether the unconstrained problem is ill-posed at this calibration.

If you can spend a day on this:

1. **First 60 min: complete T9** (full `min_R_port` map across the saved bundle). Establishes *where* the boundary binds: by age, wealth, z, and state. Cheap; no fresh solve.
2. **Next 60 min: T13** (Σ_total vs Σ_rr bookkeeping check). Rules out a partition_var() bug that would generate spurious leverage independent of all other channels. Cheap; no fresh solve.
3. **Next 90 min: T10 + T11** (refined K_state, K_xb at saved policy). Tests whether the boundary moves substantially with quadrature refinement. Cheap; no fresh solve.
4. **After lunch: T12** (wealth floor sensitivity at saved policy). Quantifies how sensitive the boundary plateau is to a numerical convenience. Cheap.
5. **Then choose:**
   - If T10/T11 show large sensitivity to quadrature: refine production config and rerun. The bound was numerical, not structural.
   - If T10/T11/T12 show small sensitivity: the bound is structural (continuous problem is ill-posed at γ=3). Run **T5** (γ sweep) to confirm.
   - If T13 reveals a Σ bookkeeping bug: fix that first; rerun everything.
6. **Document findings in §8 of this file.**

T1, T2, T3, T4, T6, T7, T8 remain useful but are now lower priority — T3 was already executed (H1a ruled out), T1/T2 are subsumed by T9 at the saved-policy level, and T5/T6/T4/T7 are confirmatory rather than diagnostic given the H1b finding.

---

## 6. Hard rules for the investigation

- **Do not modify `simulation.py`.** It's the ground-truth simulator now and is not the cause.
- **Do not change the production calibration** (β=0.96, γ=3, b_bar=10) during diagnosis. Only T5 explicitly varies γ; that's a sweep, not a recalibration.
- **Run fresh solves at smoke-grid scale** (3×3×3 × n_z=5 × n_w=40, ~10s) unless a test specifically requires production grid. The saved bundle covers most needs without re-solving.
- **Report numbers, not narrative.** "α_solver=4.92, α_merton=0.67, ratio=7.34" is a finding. "The solver appears to over-leverage" is not.
- **Each test produces a single primary diagnostic number** plus context. Append findings to this document with date, state coordinates, and acceptance result.
- **No fix attempts until ≥3 tests have landed.** Premature fixes against the wrong hypothesis waste solver-iteration budget.

---

## 7. Pointers

| What | Where |
|---|---|
| Solver entry point | `solver.run_lifecycle_solver` |
| Working-age FOC (where `use_pension_next` lives) | `solver.py:651-654` |
| State-quadrature construction | `discretization.py:get_state_quadrature` |
| Return-quadrature (per-dim refinement) | `discretization.py:get_return_quadrature` |
| Conditional return mean assembly | `solver.py:386-422` (state innov) and `solver.py:556-592` (working) |
| Discretization-arbitrage discussion | `contextfiles/RETURNS.md` §6.12 |
| Model-unit definition (1 unit = AWI ≈ $54,100) | `contextfiles/LABOUR.md` §0 |
| Production discretization config | `main.ipynb` cell `31e38655` |
| Saved unconstrained bundle | `saved_runs/unconstrained_principal_grid5x5x5_nz9` |
| Patched simulator (do not modify) | `simulation.py` |
| Reference 50/50 wealth path (sim baseline) | reproduced via fixed `S_mat = B_mat = 0.5` on the unconstrained C_mat; sd['x'] median at age 60 ≈ 2.36 model units = $128k |

---

## 8. Findings log

> Append one entry per test with date, state, numbers, acceptance result. Keep terse.

| date | test | state | result | implication |
|---|---|---|---|---|
| 2026-04-29 | T2 (terminal slice) | t=77, iz=4, i_s∈{0,12,24,36,48,62,84,96,120,124}, iw=149 | Discrete-cloud one-period optimum matches saved C/S/B[t=77] **bit-exactly** at all 10 states (e.g. i_s=62: solver=(0.834, 0.396), discrete=(0.834, 0.396)). Single-period log-Merton at γ=3 gives (0.832, 0.481) at i_s=62 — close but not identical due to bequest annuity. | Terminal solver is correct. The "ramp from 35→22" is therefore not a per-period defect at the bottom of backward induction; it accumulates over backward induction (rules in H2 or H1b on the backward sweep). |
| 2026-04-29 | T3 (full grid) | t=0, iz=4, all 125 i_s, iw=149 | Convex-hull arbitrage gap = 0 / 125. LP at every state finds origin in convex hull of `{(R_s−R_bill, R_b−R_bill)}_n`. | **H1a ruled out** at production config `(K_state=2, K_ret=(3,5,3))`. Pathology persists, so cause is H1b or downstream. |
| 2026-04-29 | T9 (single slice) | t=0, iz=4, iw=149, all 125 i_s | `min_n R_port_n ∈ [4.3e-7, 1.25e-5]` at the saved policy in 125/125 states. `+1%` scaling of `(α_s, α_b)` drives `min R_port < 0` in 125/125 states. | **H1b confirmed** at this slice. Saved policy is glued to the no-bankruptcy boundary across the entire state lattice at age 22, high W. Need full age × wealth map. |
| 2026-04-29 | T9 (single column) | t∈[0,77], iz=4, i_s=62, iw∈{40,60,80,100,130,149} | At W=0.005 (iw=40), all ages give α_s=0.83 (interior, single-period Merton); at W=200 (iw=149), ages 22–40 give α_s=4.86 with `min R_port`~3e-6 (boundary), age 50 gives interior α_s=5.13 with `min R_port`=4e-4, age 67 gives interior α_s=0.99 with `min R_port`=0.78. | Boundary binding is age × wealth dependent. Young-and-rich agents are on the boundary; old-or-poor agents are interior. Transition between ages 50 and 40 at high W. |
| 2026-04-29 | FOC at saved | t=0, iz=4, i_s=62, iw=149 | At saved (α_s=4.856, α_b=0.554): FOC_s = +1.45e+04 (positive). At α_s=4.866: FOC_s = −2.45e+09 (sign-flipped, 13 orders of magnitude jump). | FOC is *not* zero at saved policy — it's a kink, not an interior optimum. Newton converged to the discontinuity, consistent with H1b. |
| 2026-04-30 | bundle re-solve | saved bundle metadata | Bundle was re-solved 2026-04-30 07:04 UTC with `init_alpha_s=0.85, init_alpha_b=0.44` (was `(0.1, 0.4)` per handoff §3 H4). Canonical cell (t=0,iz=4,i_s=62,iw=149) saved policy now `(α_s, α_b) = (+1.018, +0.139)` — was `(4.92, 0.55)` in §1 evidence and §8 prior FOC entry. Median α_s across age 22 (z, s, w) is now +0.92, consistent with Merton (0.67) + hedging demand. | **H4 was the dominant cause at the median state** — Newton from `(0.1, 0.4)` was reaching a non-global stationary point. Changing init to near-Merton fixed it. Tail extremes (p99 α_b ≈ 39, max α_b = 82 at age 22) remain. Sim wealth pathology in handoff §1 was on the OLD bundle; the user should re-simulate on the new bundle to quantify residual pathology. |
| 2026-04-30 | T9 (full map) | all (t∈[0,77], iz∈[0,8], i_s∈[0,124], iw∈[0,149]) = 13.16M cells | min_R_port at saved policy: 0 / 13,162,500 cells with `min R_port < 1e-3`. Range [+1.24e-2, +0.95]. p1=4.05e-2, p10=0.13, p50=0.40, p99=0.89. No age × wealth bracket near the bankruptcy boundary. | **H1b ruled out for the current bundle.** The saved policy is uniformly interior (no node forces ruin). Prior 2026-04-29 finding of `min R_port ~ 3e-6` was on the pre-re-solve bundle and is no longer applicable. |
| 2026-04-30 | T13 (Σ bookkeeping) | full Σ matrices | `‖M·Σ_ss·M' + Σ_r_cond − Σ_rr‖_max = 7.8e-18` (machine precision). Markowitz under Σ_total: median α_s=+0.68, α_b=+0.41 (Merton); tail α_b∈[−9, +10]. Markowitz under Σ_r_cond (wrong, treats v^s known): tail α_b∈[−91, +104]. Saved policy mean‖saved−Mkw_total‖: stock 0.68, bond 3.07; vs. Mkw_cond: 21.0, 32.7. | **No Σ bookkeeping bug.** Solver is approximately optimizing under Σ_total at the median (correct). Saved α_b at corner states is 1–7× Markowitz_total (e.g. i_s=20: saved=+69 vs. Mkw_total=+9.6) — too big to be Markowitz hedging demand; must be a discrete-cloud artifact. |
| 2026-04-30 | T10 (K_state refine) | t=0, iz∈{4,6}, four cells: i_s∈{62, 20, 4, 45}, iw∈{149, 128, 113, 128} | At median (i_s=62): K_state ∈ {2,3,4,5} all give FOC at saved ≈ 1e-6 (zeroed); refined optimum stable around (1.5, 0.28). At corner states (i_s∈{20, 4, 45}): K_state=2 gives FOC ≈ 1e-6 at saved (saved IS the K=2 optimum); K_state=3 gives FOC=O(1e9) at saved, jumping 12 orders of magnitude. Newton on refined cloud doesn't converge cleanly at corners (residuals 1e7–1e10). | **H6 confirmed at corner states.** K_state=2 (8 nodes) only places v^s tail at ±1σ standardized, missing the >1σ mass that would discipline the optimizer. The saved policy at corner states is the K_state=2 optimum, not the continuous-distribution optimum. The refined cloud is itself ill-conditioned at corners (likely bumps into H1b on the refined support), so refinement alone may not fully solve it. |
| 2026-04-30 | T11 (K_xb refine) | same four cells | Median: completely insensitive to K_xb ∈ {3,5,7,9}. At extreme_bond_pos (i_s=20): opt_b drops 91 → 60.5 → 53.7 → 47.1 monotonically as K_xb goes 3 → 5 → 7 → 9. At extreme_bond_2 (i_s=45): drops 28.7 → no clean Newton (saved=42.1). At extreme_stock (i_s=4): opt_s drops |9.8|→|3.5| as K_xb 3→9 with sign instability. | **K_xb=3 is materially under-truncating bond-residual tail at corner states** — bond loading falls by ~50% with K_xb=9. The 3-node bond-residual quadrature only puts samples at the conditional center; corners need ≥7 nodes. Median is unaffected because conditional bond return is centered there. |
| 2026-04-30 | T12 (wealth floor) | same four cells, x_floor ∈ {1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0} | Median: optimum stable at (+1.82, −0.005) across all six floor values; FOC at saved policy unchanged. Corner states: FOC at saved policy unchanged (saved alpha doesn't drive any node below 1e-4); but implied Newton optimum at extreme_bond_pos shifts 91 → 92 → 89 → 85 → 82 → 92 across floors; at extreme_stock, opt_s = -3.18 → 7.6 → 10.6 → 13.4 → 20.7 (variation > 100%). | **H9 confirmed at corner states** (floor sensitivity > 50% across 4 OOM in floor); ruled out at median. The saved policy itself is floor-invariant (its alpha doesn't push any quadrature node into the floor region), but the optimum is highly sensitive to a numerical convenience parameter at corners. |
| | T1 | | | |
| | T4 | | | |
| | T5 | | | |
| | T6 | | | |
| | T7 | | | |
| | T8 | | | |
| | T14 | | | |
| | T15 | | | |

# Persistent Income Quadrature Refactor — Coding Agent Handoff

## 0. CONTEXT

This is a lifecycle portfolio choice model with backward induction (EGM + 2D Newton). The reference paper is Catherine (2025), "Interest Rate Risk and Household Portfolios." The model has three asset classes (stocks, bonds, bills), VAR-driven financial states, and a labour income process with mixture-normal innovations from Guvenen et al. (2022).

The project files are: `model.py`, `precompute.py`, `discretization.py`, `solver.py`, `simulation.py`, `diagnostics.py`, `DESIGN.md`. Read them before proceeding.

Before implementing, verify for yourself that this approach is:
- **Mathematically correct**: the Euler equation expectation is computed identically, just via a different numerical integration method
- **Economically correct**: the income process, pension formula, and lifecycle dynamics are unchanged — only the numerical method for computing E[V(z')] changes
- **Consistent with the literature**: Cocco, Gomes, and Maenhout (2005) Appendix A describes this approach: "the density function for both innovations to the labor income process were also approximated using Gaussian quadrature"

---

## 1. THE PROBLEM

### 1.1 Current implementation

The persistent income component z follows an AR(1) with mixture-normal innovations:

```
z_{t+1} = ρ z_t + η_{t+1}
η ~ pz × N(μ₁, σ₁²) + (1-pz) × N(μ₂, σ₂²)

ρ = 0.991, pz = 0.176
μ₁ = -0.524, σ₁ = 0.113   (17.6% weight: rare large negative shocks)
μ₂ = +0.112, σ₂ = 0.046   (82.4% weight: frequent small positive drift)
E[η] = 0 (zero-mean enforced)
```

Currently, z is discretized onto an 11-point grid spanning ±3σ_z (σ_z = 1.87) using a Tauchen bin-probability method (`discretize_income_ar1_mixture` in `discretization.py`). This produces a transition matrix Pi_z of shape (n_z, n_z) where Pi_z[i,j] = P(z_{t+1} falls in bin j | z_t = grid point i).

The solver's Euler equation inner loop (in `compute_foc_jac_working`, solver.py line 799) iterates over next-period z grid points:

```python
for j_z in range(n_z):
    p_z = Pi_z[z_idx, j_z]          # transition probability
    if p_z < prob_skip: continue
    for i_e in range(n_eps):
        income_next = income_next_table[j_z, i_e]   # exact grid lookup
        c_row = c_next_full[j_z, j_s, :]            # exact grid lookup
        c_next = fast_interp_1d(x_next, wealth_grid, c_row)  # interp in wealth only
```

### 1.2 Why this breaks

The grid spacing is dz = 1.12 (11 points over ±5.6). The innovation standard deviation is σ_η = 0.25, so dz/σ_η = 4.5 — the bins are 4.5× wider than the innovation.

Component 2 (82.4% of innovations) has mean +0.112 and std 0.046. Its typical shock (+0.10 to +0.13) is far too small to cross the half-bin boundary of 0.56. The entire mass of a "normal good year" lands inside the current bin. The discretization registers it as "nothing happened."

Component 1 (17.6% of innovations) has mean -0.524 and std 0.113. Its typical shock (-0.4 to -0.6) is large enough to cross one bin downward.

Result: P(up) = 0 at every row of Pi_z. Agents can only stay or drift down. Over a 45-year career, the median agent drifts from z=0 to z=-2.24, producing median pension of 0.048 instead of the correct 0.284. Every lifecycle result is distorted.

Increasing n_z to ~51 (dz/σ_η ≈ 0.9) fixes the Tauchen method but costs ~16× more compute and ~5× more memory, because n_z enters both the outer loop (states to solve) and the inner loop (transitions to sum over, with wider Pi_z bandwidth at higher N).

### 1.3 Why Rouwenhorst doesn't fully solve it

Rouwenhorst at N=11 gives correct first two moments (mean, variance) by construction. It produces symmetric transitions P(up) = P(down) = 0.022 and correct lifecycle dynamics (median z stays at 0). But it assumes Gaussian innovations and completely misses the mixture's skewness (-1.73) and kurtosis (4.42). The mixture was chosen specifically for these higher moments — they drive precautionary saving behaviour through the risk of rare large negative income shocks.

---

## 2. THE SOLUTION: QUADRATURE OVER INNOVATIONS

### 2.1 Key insight

The current code conflates two distinct requirements:

1. **Policy storage**: enough z-grid points to represent how consumption and portfolio shares vary with z. Policy functions are smooth in z → 11 points is plenty.

2. **Accurate transitions**: correctly computing E[V(z_{t+1}) | z_t]. With Tauchen, z_{t+1} must land on a grid point, so the grid must resolve the innovations. With quadrature, z_{t+1} lands at a continuous value and you interpolate.

The solution decouples these. Keep z as an 11-point state grid for policy storage. Replace the Pi_z transition matrix with direct Gauss-Hermite quadrature over the innovation η, plus linear interpolation of the next-period policy in z.

### 2.2 The mathematics

The Euler equation's z-expectation currently computes:

```
E_z[...] = Σ_{j=0}^{n_z-1}  Pi_z[i, j] × f(z_grid[j])
```

This is replaced by:

```
E_z[...] = Σ_{k=0}^{n_eta-1}  w_k × f(ρ z_grid[i] + η_k)
```

where (η_k, w_k) are Gauss-Hermite quadrature nodes and weights adapted to the mixture-normal distribution, and f(z_next) is evaluated by interpolating the next-period policy function at z_next on the z-grid.

These are mathematically equivalent ways to compute E_z[f(z_{t+1}) | z_t = z_grid[i]] — one integrates a discrete approximation, the other integrates the continuous density directly. The quadrature approach is standard in lifecycle models (see Cocco, Gomes, and Maenhout 2005, Appendix A).

### 2.3 The z-grid stays as a state variable

This is important: z remains a discretized state variable. The solver still loops `for z_i in range(n_z)` and stores policies at `policy[z_i, i_s, i_w]`. The Bellman equation is still solved at each z-grid point. What changes is only how the expectation over next-period z is computed inside the FOC function.

### 2.4 Independence from financial states

The z-innovation η is orthogonal to the financial state transition (s → s') and to the return residuals. In the current solver loop structure:

```python
for j_s in range(N_state):           # financial state transition
    for k_r in range(n_ret_quad):    # return residual quadrature
        # ... compute R_p, bequest terms ...
        for j_z in range(n_z):       # ← THIS CHANGES
            for i_e in range(n_eps): # transitory shock quadrature
```

The z-loop is the innermost income loop and doesn't interact with j_s or k_r except through the portfolio return R_p (already computed before the z-loop). Replacing it with quadrature + interpolation is a local change that doesn't affect the return or state-transition integration.

### 2.5 The transitory shock (eps) already works this way

The eps quadrature (`get_eps_quadrature_corrected` in `discretization.py`) already handles the mixture-normal transitory shock via Gauss-Hermite nodes per component. It works perfectly at n_nodes=5 (10 total). The eta quadrature is constructed identically — same code pattern, different parameters.

### 2.6 The return integration already works this way

The return residuals (xr, xb) are integrated via Gauss-Hermite quadrature (`get_return_quadrature` in `discretization.py`). Returns land at continuous values, weighted by ret_weights. This is the same principle.

---

## 3. WHAT CHANGES IN THE CODE

### 3.1 discretization.py — Add eta quadrature function (~15 lines)

Add `get_eta_quadrature_mixture(model, n_nodes=5)`. This is nearly identical to the existing `get_eps_quadrature_corrected`:

```python
def get_eta_quadrature_mixture(model, n_nodes=5):
    """
    Gauss-Hermite quadrature for the persistent income innovation eta.
    
    Builds nodes and weights for both mixture components separately,
    then concatenates. Same approach as get_eps_quadrature_corrected.
    
    Parameters
    ----------
    model : LifecyclePortfolioModel
    n_nodes : int
        Gauss-Hermite order per mixture component. Total nodes = 2 * n_nodes.
    
    Returns
    -------
    eta_nodes : (2 * n_nodes,) quadrature nodes
    eta_weights : (2 * n_nodes,) weights summing to 1
    """
    nodes, weights = roots_hermite(n_nodes)
    weights = weights / np.sqrt(np.pi)
    nodes = nodes * np.sqrt(2.0)
    
    # Component 1: N(mu_eta1, sigma_eta1^2), weight pz
    e1 = nodes * model.sigma_eta1 + model.mu_eta1
    w1 = weights * model.pz
    
    # Component 2: N(mu_eta2, sigma_eta2^2), weight (1-pz)
    # Use the zero-mean enforced mu_eta2 (same as eps approach)
    mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1
    e2 = nodes * model.sigma_eta2 + mu_eta2_eff
    w2 = weights * (1.0 - model.pz)
    
    eta_nodes = np.concatenate([e1, e2])
    eta_weights = np.concatenate([w1, w2])
    
    # Verify zero-mean
    mean_check = np.sum(eta_nodes * eta_weights)
    if abs(mean_check) > 1e-10:
        print(f"WARNING: eta quadrature mean = {mean_check:.6e} (should be ~0)")
    
    return eta_nodes, eta_weights
```

### 3.2 precompute.py — Store eta quadrature (~5 lines)

In the `Precompute.__init__` method, after the eps quadrature (around line 186), add:

```python
self.eta_nodes, self.eta_weights = get_eta_quadrature_mixture(model, n_nodes=disc_config.n_eta_nodes)
```

Add `n_eta_nodes: int = 5` to `DiscretizationConfig` in `model.py` (yields 10 total nodes, matching eps).

Also precompute and store `dz = z_grid[1] - z_grid[0]` for use in the solver's z-bracket computation.

**Keep building Pi_z** — it's still needed for the simulation's forward transitions (line 432 of simulation.py). Add a comment noting Pi_z is now used only for simulation, not for the solver.

### 3.3 solver.py — The core change (~20 lines logic + ~80 lines signature plumbing)

#### 3.3.1 The FOC function: `compute_foc_jac_working` (line 738)

**Signature change**: Replace `Pi_z` with `z_grid, rho, eta_nodes, eta_weights, dz`.

**Inner loop change** (lines 799–831): Replace the `for j_z in range(n_z)` loop. The new loop:

```python
n_eta = len(eta_nodes)

# -- alive contribution: quadrature over persistent and transitory innovations --
for k_eta in range(n_eta):
    w_eta = eta_weights[k_eta]
    if w_eta < prob_skip:
        continue
    
    # Next-period z (continuous, generally between grid points)
    z_next = rho * z_grid[z_idx] + eta_nodes[k_eta]
    
    # Bracket z_next on the uniform z_grid
    iz_lo = int((z_next - z_grid[0]) / dz)
    iz_lo = max(0, min(iz_lo, n_z - 2))  # clamp to valid bracket
    frac_z = (z_next - z_grid[iz_lo]) / dz
    frac_z = max(0.0, min(1.0, frac_z))  # safety clamp
    
    p_out_base = p_state_ret * w_eta
    
    for i_e in range(n_eps):
        weight = p_out_base * eps_weights[i_e]
        
        # Interpolate income in z (linear between bracket neighbours)
        income_next = ((1.0 - frac_z) * income_next_table[iz_lo, i_e]
                       + frac_z * income_next_table[iz_lo + 1, i_e])
        
        x_next = w_inv + income_next
        
        # Interpolate consumption in z then wealth
        # Two 1D wealth interpolations, then blend in z
        c_lo, mpc_lo = fast_interp_1d_with_slope(
            x_next, wealth_grid, c_next_full[iz_lo, j_s, :])
        c_hi, mpc_hi = fast_interp_1d_with_slope(
            x_next, wealth_grid, c_next_full[iz_lo + 1, j_s, :])
        
        c_next = (1.0 - frac_z) * c_lo + frac_z * c_hi
        c_next = max(c_next, min_consumption)
        mpc = (1.0 - frac_z) * mpc_lo + frac_z * mpc_hi
        mpc = max(0.0, min(1.0, mpc))
        
        # --- Everything below here is IDENTICAL to current code ---
        mu_alive  = c_next ** (-gamma)
        mup_alive = -gamma * mu_alive / c_next * mpc
        
        wmu  = weight * psi * mu_alive
        wmup = weight * psi * mup_alive
        
        euler_sum += wmu * R_p
        foc_s     += wmu * Rex_s
        foc_b     += wmu * Rex_b
        
        jac = wmup * s_val
        J_ss += jac * Rex_s * Rex_s
        J_bb += jac * Rex_b * Rex_b
        J_sb += jac * Rex_s * Rex_b
```

Key points:
- The z-bracket computation (`iz_lo`, `frac_z`) is done once per k_eta, outside the i_e loop
- Income and consumption are both linearly interpolated between the two z-bracket neighbours
- The MPC (marginal propensity to consume) is also blended — this is approximate but sufficient for Newton convergence
- Everything after the interpolation (marginal utility, FOC/Jacobian accumulation) is unchanged

#### 3.3.2 Retirement FOC: `compute_foc_jac_retirement` (line 395)

**No change needed.** The retirement FOC has no z-transition loop — z is frozen at retirement and the agent receives a fixed pension. The inner loop is only over (j_s, k_r).

#### 3.3.3 Newton solver wrappers

These functions all call `compute_foc_jac_working` and just need their parameter lists updated to pass the new arguments through:

- `solve_portfolio_2d_working` (line 839): ~10 calls to the FOC function — update each argument list
- `solve_portfolio_unconstrained_working` (line 1146): same pattern

The logic inside these wrappers (Newton iteration, convergence checks, corner detection) is completely unchanged.

#### 3.3.4 Step functions

- `_solve_working_age_step_jit` (line 1674): Update signature to accept `z_grid, rho, eta_nodes, eta_weights, dz` instead of `Pi_z`. Pass through to the portfolio solver calls.
- `_solve_retirement_step_jit` (line 1489): **No change** — no z-transition in retirement.

#### 3.3.5 Main solver entry point

- `run_lifecycle_solver` (line ~1900): Extract `eta_nodes`, `eta_weights` from `pc`. Compute `dz`. Pass to `_solve_working_age_step_jit`. Currently passes `Pi_z` at line 1907 — replace.

### 3.4 simulation.py — No change (for now)

Keep using Pi_z for the simulation's forward transitions (line 432). The policies are now correctly solved via quadrature, so even though the simulation's z-transitions use the (imperfect) Pi_z, the policy functions being applied are correct.

A better approach (optional, later): draw continuous η from the mixture and snap z_next to the nearest grid point for policy lookup. This is a simulation improvement, not a solver correctness issue.

### 3.5 diagnostics.py — Minor updates (~10 lines)

- Add a check on eta quadrature quality (mean, variance, skewness of nodes × weights)
- Note that Pi_z is now simulation-only, not used by the solver
- The Pi_z transition quality diagnostic (if implemented from the earlier handoff) should note its role has changed

---

## 4. WHAT DOES NOT CHANGE

| Component | Why unchanged |
|-----------|--------------|
| z_grid construction | Still 11 points ±3σ, same grid |
| Policy array shapes | Still (n_age, n_z, N_state, n_w) with n_z=11 |
| Retirement solver | No z-transition in retirement |
| eps quadrature | Already uses the correct approach |
| Return quadrature | Already uses the correct approach |
| Financial state Rouwenhorst | Independent of z, unaffected |
| Pension formula | Reads pension_after_tax[t, z_idx], unchanged |
| Income lookup tables | Still precomputed, now interpolated in z instead of exact lookup |
| Simulation engine | Keeps Pi_z for forward transitions (for now) |
| Bequest computation | Inside the FOC but outside the z-loop, unchanged |

---

## 5. VERIFICATION

### 5.1 Mathematical verification

After implementing, verify at a single grid point that the quadrature and old Tauchen give the same expectation when Tauchen is run at very high N:

```python
# At z=0, compute E[exp(z_{t+1})] both ways:

# Quadrature:
E_quad = sum(w_k * exp(rho * 0 + eta_k) for k, (eta_k, w_k) in enumerate(zip(eta_nodes, eta_weights)))

# Tauchen at N=201 (ground truth):
z_grid_fine, Pi_fine = discretize_income_ar1_mixture(..., N=201)
mid = 100
E_tauchen = sum(Pi_fine[mid, j] * exp(z_grid_fine[j]) for j in range(201))

# These should agree to ~4 decimal places
```

### 5.2 Moment verification

The eta quadrature should match all innovation moments:

```python
# Mean (should be ~0)
mean = sum(eta_nodes * eta_weights)

# Variance (should be 0.0626)
var = sum(eta_nodes**2 * eta_weights) - mean**2

# Skewness (should be -1.73)
e3 = sum(eta_nodes**3 * eta_weights)
skew = (e3 - 3*mean*var - mean**3) / var**1.5
```

### 5.3 Economic verification

Run the Tier 1 diagnostics (`print_model_diagnostic_report`). All 18 tests should still pass — the income tables, pension formula, and tax schedule are unchanged. The Pi_z diagnostic block (if implemented) will still show the old Tauchen transition quality, which is fine — it's now simulation-only.

Run a 45-year simulation and check:
- Median z at retirement should be near 0 (not -2.24)
- Median pension should be ~0.284 (not 0.048)
- Replacement rates should match the Tier 1 grid-based values

### 5.4 Solver convergence

The Newton solver should converge at the same rate as before. The FOC and Jacobian are computed via a different quadrature rule but the same economic equations. If convergence degrades, check:
- The z-bracket clamping (iz_lo must be in [0, n_z-2])
- The frac_z clamping (must be in [0, 1])
- That `min_consumption` and `min_wealth_inv` floors are still applied after the z-interpolation

---

## 6. COST ESTIMATE

| Metric | Current (broken) | Quadrature (10 nodes) | Tauchen N=51 (correct) |
|--------|:---:|:---:|:---:|
| n_z (state grid) | 11 | 11 | 51 |
| Inner loop size | ~19 | ~200 | ~85 |
| Working-age slowdown | 1× | ~8× | ~16× |
| Total slowdown (weighted) | 1× | ~6× | ~16× |
| Policy memory | 5 MB | 5 MB | 23 MB |
| Results correct? | No | Yes | Yes |

The ~6× slowdown vs the current broken implementation is the cost of getting correct results. It's 2–3× cheaper than the Tauchen N=51 alternative.

The inner loop does 2 wealth interpolations per (η_node, ε_node) pair instead of 1, because consumption must be interpolated at both z-bracket neighbors then blended. An optimization (sharing the wealth-bracket binary search across the two z-neighbors) could reduce this to ~1.3× per pair, bringing the total slowdown to ~4×. Implement this optimization only if profiling shows it matters.

---

## 7. FILES TO MODIFY

| File | What changes | Lines affected |
|------|-------------|---------------|
| `discretization.py` | Add `get_eta_quadrature_mixture` | +15 new |
| `model.py` | Add `n_eta_nodes` to DiscretizationConfig | +1 |
| `precompute.py` | Build and store eta quadrature, store dz | +5 |
| `solver.py` | FOC inner loop + signature propagation | ~100 changed |
| `simulation.py` | Nothing (keep Pi_z for now) | 0 |
| `diagnostics.py` | Add eta quality check, annotate Pi_z | ~10 |

Total: ~130 lines. The conceptually hard part (the inner loop replacement) is ~25 lines. The rest is mechanical signature propagation through the Newton wrapper functions.

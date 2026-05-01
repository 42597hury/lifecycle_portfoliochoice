# STATE SPACE — CANONICAL SHAPES, INDEXING, AND PORTFOLIO CONSTRAINTS

**Purpose:** Single reference for every array shape, indexing system, and
dimensional convention in the model. Answers "what shape should this array be?"
and "how do I go from multi-index to flat index?" without reading the code.

**Code references:** `model.py` (LifecyclePortfolioModel, DiscretizationConfig),
`precompute.py` (Precompute class, all grid/array construction),
`solver.py` (policy shapes, trilinear interpolation, FOC loops),
`simulation.py` (policy lookup, continuous-state propagation).

---

## 0. The Four Indexing Dimensions

The agent's state each period has four dimensions. Policy functions are stored
on a 4D grid over these:

```
C_mat[t, i_z, i_s, i_w]     shape: (n_age, n_z, N_state, n_w)
S_mat[t, i_z, i_s, i_w]     shape: (n_age, n_z, N_state, n_w)
B_mat[t, i_z, i_s, i_w]     shape: (n_age, n_z, N_state, n_w)
```

| Axis | Index | Variable | Grid object | Typical size |
|------|-------|----------|-------------|-------------|
| 0 | `t` | Age period | `ages` | n_age = 78 |
| 1 | `i_z` | Persistent income | `z_grid` | n_z = 7 (default) or 11 (production) |
| 2 | `i_s` | Financial state (joint) | `state_grid` | N_state = 125 or 343 |
| 3 | `i_w` | Cash-on-hand | `wealth_grid` | n_w = 150 |

**Outputs:**
- `C_mat[t, i_z, i_s, i_w]` = optimal consumption at this state
- `S_mat[t, i_z, i_s, i_w]` = optimal stock share (α_s) of savings
- `B_mat[t, i_z, i_s, i_w]` = optimal bond share (α_b) of savings
- Bill share is the residual: `α_bill = 1 − α_s − α_b`

**Memory:**
| Config | N_state | Total elements (×3 arrays) | Approx size |
|--------|---------|---------------------------|-------------|
| 5×5×5, n_z=7 | 125 | 78 × 7 × 125 × 150 × 3 = 30.7M | ~235 MB |
| 5×5×5, n_z=11 | 125 | 78 × 11 × 125 × 150 × 3 = 48.3M | ~370 MB |
| 7×7×7, n_z=11 | 343 | 78 × 11 × 343 × 150 × 3 = 132.7M | ~1.0 GB |

---

## 1. Age Dimension (axis 0)

```
t ∈ {0, 1, ..., 77}        (n_age = 78)
ages[t] = start_age + t    (ages[0] = 22, ages[77] = 99)
```

See CONVENTIONS.md Section 0 for the full age ↔ index mapping and retirement
boundary.

---

## 2. Persistent Income Dimension (axis 1)

```
z_grid : (n_z,) float64
```

- Mean-zero log deviations: `z_grid` is centered on 0 by construction
- Uniform spacing: `dz = z_grid[1] − z_grid[0]`
- Covers ±n_stds unconditional standard deviations (default n_stds = 3.0)
- Constructed by `discretize_income_ar1_mixture()` in `discretization.py`

**Solver treatment:** The z dimension is indexed discretely (`i_z`), but
within the FOC, next-period z values `z' = ρ·z[i_z] + η_k` land off-grid.
Consumption is interpolated by **PCHIP** (Fritsch-Carlson, monotonicity-preserving
cubic Hermite) on interior intervals where the 4-point stencil fits
(`iz_lo ≥ 1` and `iz_lo + 2 < n_z`); the first and last z intervals fall back
to linear. Wealth interpolation is linear on `[iw, iw+1]`. The Jacobian's
`mpc` is the analytical wealth derivative of the interpolant -- PCHIP
evaluated at iw and iw+1 then finite-differenced -- which keeps mpc exactly
consistent with c_val even when the PCHIP slope limiter activates
asymmetrically across the two wealth corners.

**Simulation treatment:** z is tracked as a continuous float. Policies are
linearly interpolated between bracketing z-grid points. Income and pension
are computed directly from continuous z (no table interpolation).

**Associated arrays:**
| Array | Shape | Description |
|-------|-------|-------------|
| `z_grid` | (n_z,) | Persistent income grid points |
| `Pi_z` | (n_z, n_z) | Transition matrix (retained; NOT used by solver) |
| `dz` | scalar | Uniform grid spacing |
| `eta_nodes` | (n_eta,) | Judd-mixture quadrature nodes for persistent innovation |
| `eta_weights` | (n_eta,) | Corresponding weights (sum = 1, mean = 0) |
| `eps_nodes` | (n_eps,) | Judd-mixture quadrature nodes for transitory shock |
| `eps_weights` | (n_eps,) | Corresponding weights (sum = 1, mean = 0) |

Quadrature node counts: `n_eta = disc_config.n_eta_nodes` and
`n_eps = disc_config.n_eps_nodes` (total node count — Judd 1998
construction directly on the mixture density; polynomial exactness
`2n − 1` against the mixture).

---

## 3. Financial State Dimension (axis 2)

### 3.1 The 3D State Vector

The financial state `s_t` has three components.  The default ordering
(production, since 2026-04-30) puts cy first so Cholesky `L[:, 0]` is a
pure-cy column, which makes per-axis `state_n_stds[0]` a clean cy knob:

```
s_t = (cy, spr, y_1)

state_names = ('cy', 'spr', 'y_1')
y_1_index_in_state = 2     # was 0 under the legacy (y_1, spr, cy) ordering
spr_index_in_state = 1     # spr stays in the middle column
```

Saved bundles produced before the reorder use the legacy ordering
`('y_1', 'spr', 'cy')` with `y_1_index_in_state = 0`.  Pass
`state_indices=(0, 1, 2)` and `y_1_index_in_state=0` to
`build_nominal_system1_var_config()` to reproduce the legacy ordering.
See `contextfiles/RETURNS.md` §5.6 for why the reorder matters.

### 3.2 Marginal Grids and Cartesian Product

Each state variable is discretized independently using Rouwenhorst:

```
state_grids: list of 3 arrays
  state_grids[0] : (N_0,) — y_1 marginal grid
  state_grids[1] : (N_1,) — spr marginal grid
  state_grids[2] : (N_2,) — cy marginal grid

state_grid_sizes = (N_0, N_1, N_2)    e.g. (5, 5, 5) or (7, 7, 7)
N_state = N_0 × N_1 × N_2             e.g. 125 or 343
```

The joint state grid is the Cartesian product of these marginals:

```
state_grid   : (N_state, 3) float64 — row i = state vector in MODEL ordering
                                       (default: [cy, spr, y_1]; legacy: [y_1, spr, cy])
state_indices: (N_state, 3) int64   — row i = [idx_0, idx_1, idx_2] into marginals
```

### 3.3 Flat Index ↔ Multi-Index

**Row-major ordering (C order):** dimension 0 (y_1) varies slowest, dimension 2
(cy) varies fastest.

```
i_s = i_0 × N_1 × N_2  +  i_1 × N_2  +  i_2

Inverse:
  i_0 = i_s // (N_1 × N_2)
  i_1 = (i_s % (N_1 × N_2)) // N_2
  i_2 = i_s % N_2
```

This ordering is used everywhere: `_build_state_grid()`, solver trilinear
interpolation (`j000 = lo0 * N1 * N2 + lo1 * N2 + lo2`), and simulation
nearest-neighbor snap (`next_state_idx = best_d0 * N1_s * N2_s + ...`).

### 3.4 Trilinear Interpolation in the Solver

When the solver evaluates the expectation over state innovations, the next
state `s' = Φ_0 + Φ_11 @ s_t + v_k` generally falls off the grid. The solver
uses **trilinear interpolation** across the 8 bracketing grid corners:

```python
lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(s_next_0, s_next_1, s_next_2,
                                               grids_0, grids_1, grids_2)
# 8 corner indices:
j000 = lo0*N1*N2 + lo1*N2 + lo2          # (lo0,   lo1,   lo2  )
j001 = lo0*N1*N2 + lo1*N2 + (lo2+1)      # (lo0,   lo1,   lo2+1)
j010 = lo0*N1*N2 + (lo1+1)*N2 + lo2      # (lo0,   lo1+1, lo2  )
j011 = lo0*N1*N2 + (lo1+1)*N2 + (lo2+1)  # (lo0,   lo1+1, lo2+1)
j100 = (lo0+1)*N1*N2 + lo1*N2 + lo2      # (lo0+1, lo1,   lo2  )
j101 = (lo0+1)*N1*N2 + lo1*N2 + (lo2+1)  # (lo0+1, lo1,   lo2+1)
j110 = (lo0+1)*N1*N2 + (lo1+1)*N2 + lo2  # (lo0+1, lo1+1, lo2  )
j111 = (lo0+1)*N1*N2 + (lo1+1)*N2 + (lo2+1) # all +1

# Trilinear weight for corner (a,b,c) where a,b,c ∈ {0,1}:
w_abc = [(1-f0,f0)[a]] × [(1-f1,f1)[b]] × [(1-f2,f2)[c]]
```

At grid boundaries, fractions are clamped to [0, 1] (flat extrapolation in
the state dimension).

### 3.5 Simulation Treatment

The simulation propagates the financial state continuously:
```
s_{t+1} = Φ_0_state + Φ_11 @ s_t + L_ss @ z    (z ~ N(0,I))
```
where `L_ss = cholesky(Σ_ss)`. The continuous s_{t+1} is then snapped to the
**nearest grid point** for policy lookup (nearest-neighbor, not interpolation):
```python
next_state_idx = best_d0 * N1_s * N2_s + best_d1 * N2_s + best_d2
```

### 3.6 State-Indexed Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| `state_grid` | (N_state, 3) | Joint state vectors |
| `state_indices` | (N_state, 3) | Multi-index into marginals |
| `Pi_state` | (N_state, N_state) | Transition matrix (retained; NOT used by solver) |
| `annuity_factors` | (N_state,) | A(y_1, spr, b_bar) per state |
| `mu_r` | (N_state, N_state, 3) | Conditional return means |

---

## 4. Wealth Dimension (axis 3)

```
wealth_grid : (n_w,) float64     geometric spacing
  wealth_min = 1e-4
  wealth_max = 200.0
  n_w = 150 (default)
```

Constructed as `np.geomspace(wealth_min, wealth_max, n_w)`. The geometric
spacing concentrates grid points near zero where marginal utility is steep
and policies change rapidly.

**EGM savings grid:**
```
s_grid : (n_s,) float64     geometric spacing
  savings_min = 1e-8
  n_s = 150 (default, same as n_w)
```

The EGM (endogenous grid method) solves for consumption on the savings grid,
then maps to the wealth grid. The savings grid is separate because it starts
at a smaller minimum (1e-8 vs 1e-4).

**Interpolation:** Policy lookup in the wealth dimension uses **linear
interpolation** with binary search (`fast_interp_1d`). Beyond grid boundaries,
**linear extrapolation** uses the nearest-interval slope. This can cause
instability at the upper boundary if wealth exceeds 200 AWI units ($10.8M) —
a known issue flagged in TODO.md.

---

## 5. Quadrature Dimensions (NOT Grid Axes)

These dimensions appear inside the solver's FOC loops but are NOT stored as
policy axes. They are integrated out during the backward induction.

### 5.1 State Innovation Quadrature

```
v_nodes   : (K_s^n_state, n_state) = (K_s^3, 3)   state innovation nodes
v_weights : (K_s^n_state,)         = (K_s^3,)      tensor-product weights
M_v_nodes : (K_s^3, 3)                              v_nodes @ M.T (precomputed)
```

Default `K_s = n_state_quad_nodes = 3`, giving `3^3 = 27` joint nodes.

These are Gauss-Hermite nodes on `N(0, Σ_ss)`. For each node k_v, the solver
computes:
- Next state: `s' = Φ_0 + Φ_11 @ s_t + v_nodes[k_v]`
- Conditional return mean: `μ_r = base_μ_r + M_v_nodes[k_v]`

### 5.2 Return Residual Quadrature

```
ret_nodes   : (n_ret_quad, n_ret) = (n_ret_quad, 3)   residual return nodes
ret_weights : (n_ret_quad,)                            weights (sum = 1)
```

`n_ret_nodes_1d` in `DiscretizationConfig` accepts:
- a scalar `int K` — uniform Gauss-Hermite order across all `n_ret`
  dimensions, giving `n_ret_quad = K^n_ret` joint nodes (e.g. `K=2 → 8`,
  `K=3 → 27`); this is the legacy default,
- a length-`n_ret` tuple `(K_rtb, K_xr, K_xb)` — per-dimension order,
  giving `n_ret_quad = prod(K_i)` joint nodes (e.g. `(3,9,3) → 81`).

Use the tuple form to refine the dimensions that matter most for the
problem. As of 2026-04-30, `get_return_quadrature` uses a **Cholesky**
transform `r = z @ L^T` (consistent with `get_state_quadrature`), so the
per-axis labels `(K_rtb, K_xr, K_xb)` are honest:

- `K_rtb` refines the rtb axis (`L` lower-triangular: `z_0` is the only
  component that contributes to the rtb component of `r`).
- `K_xr` refines the xr-residual after the rtb correlation has been
  orthogonalized away.
- `K_xb` refines the pure xb residual (`L[2, 2]` direction, after rtb
  and xr have been orthogonalized away).

Refining the highest-residual-variance axis (xr in our calibration) by
setting `K_xr` is the cheapest way to suppress discretization-arbitrage
in the joint excess-return cloud at unconstrained CRRA. The previous
implementation used eigendecomposition, under which the slot labels did
not match the physical asset axes; this was a labelling bug, see the
note in `RETURNS.md` §6.12. Default stays `2`.

These are Gauss-Hermite nodes on `N(0, Σ_r_cond)`. Precomputed exponentials
avoid recomputation in the hot loop:
```
exp_ret_bill  : (n_ret_quad,)    exp(ret_nodes[:, 0])  — rtb residuals
exp_ret_stock : (n_ret_quad,)    exp(ret_nodes[:, 1])  — xr residuals
exp_ret_bond  : (n_ret_quad,)    exp(ret_nodes[:, 2])  — xb residuals
```

### 5.3 Income Quadrature (Working Age Only)

```
eta_nodes   : (n_eta,) float64    persistent innovation Judd-mixture nodes
eta_weights : (n_eta,) float64    weights (sum = 1, mean = 0, all > 0)
eps_nodes   : (n_eps,) float64    transitory shock Judd-mixture nodes
eps_weights : (n_eps,) float64    weights (sum = 1, mean = 0, all > 0)
```

`n_eta = disc_config.n_eta_nodes` (total node count, no longer
per-component K). `n_eps` likewise. Polynomial exactness `2n − 1`
against the mixture density (Judd 1998 §7).

### 5.4 Total FOC Iterations per Newton Evaluation

| Phase | Outer | × Return | × Persistent | × Transitory | Total |
|-------|-------|----------|-------------|-------------|-------|
| Terminal | n_state_quad = 27 | × n_ret_quad = 27 | — | — | **729** |
| Retirement | 27 | × 27 | — | — | **729** |
| Working | 27 | × 27 | × n_eta = 3 | × n_eps = 5 | **10,935** |

Counts above use uniform `K_s=3` and uniform `K_r=3`
(`n_state_quad = 3^3 = 27`, `n_ret_quad = 3^3 = 27`). With per-dimension
return refinement, `n_ret_quad = prod(n_ret_nodes_1d)` instead — for
example `(3,9,3) → 81` triples each row's "× Return" factor.

(Default settings: K_s=3, K_r=3, n_eta_nodes=3, n_eps_nodes=5; total
income nodes 3 × 5 = 15 — for comparison, the previous concatenated-GH
rule at the same `n_eta_nodes=3, n_eps_nodes=5` produced 6 × 10 = 60
income nodes, i.e. the Judd migration is a 4× cost reduction on the
income loop at the same polynomial-exactness order.)

---

## 6. The Three Assets and Portfolio Constraints

### 6.1 Asset Returns (from Quadrature)

All three returns are constructed from the SAME joint draw `(μ_r + residual)`:

```
R_bill  = exp(μ_rtb  + ret_nodes[k_r, 0])
R_stock = R_bill × exp(μ_xr + ret_nodes[k_r, 1])
R_bond  = R_bill × exp(μ_xb + ret_nodes[k_r, 2])
```

**Critical invariant:** All three returns for a single period use the SAME
return quadrature node k_r. They are components of one joint draw from
`N(μ_r, Σ_r_cond)`.

### 6.2 Portfolio Weights

```
α_s = stock share of savings       (S_mat)
α_b = bond share of savings        (B_mat)
α_bill = 1 − α_s − α_b            (residual)
```

**These are shares of SAVINGS** `a = x − c`, NOT of total wealth.

### 6.3 Feasible Region (Constrained Case)

When `model.constrained = True`, the simplex constraints are:

```
α_s ≥ 0                  (no short-selling stocks)
α_b ≥ 0                  (no short-selling bonds)
α_s + α_b ≤ 1            (no borrowing; α_bill ≥ 0)
```

The feasible set is a triangle in (α_s, α_b) space:

```
α_b
 1 ┤╲
   │  ╲
   │    ╲  feasible
   │      ╲
 0 ┼───────╲──── α_s
   0        1
```

**Corners:** (0,0) = all bills, (1,0) = all stocks, (0,1) = all bonds.
**Edges:** α_s + α_b = 1 (no bills), α_s = 0 (bonds + bills), α_b = 0 (stocks + bills).

The Newton solver checks all 3 corners, then all 3 edges, before attempting
the interior solution. See solver.py exit codes `EC_CORNER_*` and `EC_EDGE_*`.

### 6.4 Unconstrained Case

When `model.constrained = False`, α_s and α_b are unrestricted (can be negative
or sum > 1). The solver uses backtracking line search instead of simplex
projection.

### 6.5 Gross Portfolio Return

```
R_port = α_s × R_stock + α_b × R_bond + (1 − α_s − α_b) × R_bill
```

This is a LEVEL return (not log). Estate = savings × R_port.

---

## 7. What Lives Where

### 7.1 On LifecyclePortfolioModel (Immutable Economic Parameters)

| Field | Shape | Description |
|-------|-------|-------------|
| `gamma`, `beta`, `b_bar` | scalars | Preferences |
| `start_age`, `retire_age`, `terminal_age` | scalars | Lifecycle |
| `b0`, `b1`, `b2`, `b3`, `rho`, `pz`, ... | scalars | Income process |
| `n_state`, `n_ret` | scalars | 3, 3 |
| `state_names`, `ret_names` | tuples | ('y_1','spr','cy'), ('rtb','xr','xb') |
| `Phi_0_state`, `Phi_11` | (3,), (3,3) | State dynamics |
| `Phi_0_ret`, `Phi_21` | (3,), (3,3) | Return equations |
| `Sigma_ss`, `Sigma_rr`, `Sigma_rs` | (3,3) each | Innovation covariances |
| `M`, `Sigma_r_cond` | (3,3) each | Conditioning matrix, residual cov |
| `constrained` | bool | Portfolio constraint flag |

### 7.2 On Precompute (Numerical Approximation)

| Field | Shape | Description |
|-------|-------|-------------|
| `wealth_grid` | (n_w,) | Geometric cash-on-hand grid |
| `s_grid` | (n_s,) | Geometric EGM savings grid |
| `ages` | (n_age,) | Integer ages 22..99 |
| `state_grid` | (N_state, 3) | Joint financial state grid |
| `state_grids` | list[3] | Marginal 1D grids |
| `state_indices` | (N_state, 3) | Multi-index into marginals |
| `Pi_state` | (N_state, N_state) | State transition (NOT used by solver) |
| `v_nodes` | (K_s^3, 3) | State innovation quad nodes |
| `v_weights` | (K_s^3,) | State innovation quad weights |
| `M_v_nodes` | (K_s^3, 3) | v_nodes @ M.T |
| `const_r` | (3,) | = Phi_0_ret |
| `A_r` | (3, 3) | = Phi_21 |
| `mu_r` | (N_state, N_state, 3) | Conditional return means |
| `ret_nodes` | (n_ret_quad, 3) | Return residual quad nodes |
| `ret_weights` | (n_ret_quad,) | Return residual quad weights; sum=1 |
| `exp_ret_bill/stock/bond` | (n_ret_quad,) each | Precomputed exp of residuals |
| `n_ret_quad` | scalar int | `prod(n_ret_nodes_1d)`; uniform K → K^3, tuple (K_rtb, K_xr, K_xb) → K_rtb·K_xr·K_xb |
| `n_ret_nodes_1d` | tuple[int] of length n_ret | Always normalized to a tuple by `Precompute`, even when the user passed a scalar |
| `annuity_factors` | (N_state,) | Bequest annuity per state |
| `z_grid` | (n_z,) | Persistent income grid |
| `Pi_z` | (n_z, n_z) | Income transition (NOT used by solver) |
| `eta_nodes/weights` | (n_eta,) each | Persistent innovation quad |
| `eps_nodes/weights` | (n_eps,) each | Transitory shock quad |
| `dz` | scalar | z_grid spacing |
| `log_det_profile` | (n_age,) | f(age) for each period |
| `avg_det` | scalar | mean(exp(f)) over working ages |
| `working_income` | (n_age, n_z, n_eps) | After-tax income (simulation only) |
| `pension_after_tax` | (n_age, n_z) | After-tax pension |
| `survival_probs_2d` | (n_age, n_z) | Survival probabilities |

### 7.3 On Solver Output (Policy Arrays)

| Array | Shape | Description |
|-------|-------|-------------|
| `C_mat` | (n_age, n_z, N_state, n_w) | Optimal consumption |
| `S_mat` | (n_age, n_z, N_state, n_w) | Optimal stock share α_s |
| `B_mat` | (n_age, n_z, N_state, n_w) | Optimal bond share α_b |
| `diagnostics` | dict | Per-age Newton stats, exit code counts |

---

## 8. Dimension Size Reference

Default configurations and their implications:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_wealth` | 150 | Cash-on-hand grid points |
| `n_savings` | 150 | EGM savings grid points |
| `state_grid_sizes` | (5,5,5) | Marginal state grid → N_state=125 |
| `n_z` | 7 (code default; 11 in production) | Persistent income grid points |
| `n_stds` | 3.0 | z-grid covers ±3σ_z |
| `n_eps_nodes` | 3 | Total Judd-mixture nodes for transitory shock (= n_eps; exactness 2n−1) |
| `n_eta_nodes` | 3 | Total Judd-mixture nodes for persistent innovation (= n_eta; exactness 2n−1) |
| `n_ret_nodes_1d` | 2 | int K → uniform K^n_ret nodes (default 2 → 8); also accepts a tuple `(K_rtb, K_xr, K_xb)` for per-dimension refinement (e.g. `(3,9,3) → 81`) |
| `n_state_quad_nodes` | 3 | GH order per state dim → 3^3=27 state nodes |

**Scaling rules:**
- Policy array size scales as `n_age × n_z × N_state × n_w`
- Solver cost per period scales as `n_z × N_state × (n_state_quad × n_ret_quad)`
  for retirement, plus `× n_eta × n_eps` for working age, where
  `n_state_quad = K_s^n_state` and `n_ret_quad = prod(n_ret_nodes_1d)`
- mu_r memory scales as `N_state^2 × 3`

---

## 9. Notation Cross-Reference

| This document | Code variable | DESIGN.md | Catherine (2025) |
|--------------|---------------|-----------|-----------------|
| α_s | `alpha_s`, S_mat | alpha_stock | α^S |
| α_b | `alpha_b`, B_mat | alpha_bond | α^B |
| α_bill | `1 - alpha_s - alpha_b` | alpha_bill | α^F (bills) |
| x_t | `x`, cash-on-hand | W_t | W_t |
| a_t | `savings` | a_t | a_t |
| s_t | `state_grid[i_s]` | (cy, spr, y_1) default; (y_1, spr, cy) legacy | z_t (VAR state) |
| z_t | `z_grid[i_z]` | z_t (income) | z_t (earnings) |
| ψ_t | `survival_probs_2d[t, iz]` | psi_{t,z} | ψ_t |
| R_port | `R_port` | R_port | R_p |
| A(y_1,spr) | `annuity_factors[i_s]` | A(r_f, b_bar) | A_t |

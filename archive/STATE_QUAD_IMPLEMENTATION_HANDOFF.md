# HANDOFF — State Innovation Quadrature: Implementation Guide

**To:** coding agent  
**From:** design & review team  
**Date:** April 2026  
**Status:** Reviewed and approved for implementation. Mathematical correctness verified. Computational feasibility confirmed.

---

## 0. READ THESE FILES FIRST

Before writing any code, read and understand these files in full:

| File | Why |
|------|-----|
| `solver.py` | You will modify the FOC functions and period solvers. Understand the loop nesting, the Newton solver structure, and how `prange` parallelization works. |
| `precompute.py` | You will add new arrays and modify the constructor. Understand the data flow from `Precompute` to the solver. |
| `discretization.py` | You will add `get_state_quadrature()`. Use `get_return_quadrature()` (lines 216–264) as your template — the new function is structurally identical. |
| `model.py` | You will add one field to `DiscretizationConfig`. |
| `simulation.py` | You will add a continuous state transition mode. Understand `simulate_lifecycle_core` (lines 346–560). |
| `DESIGN.md` §2.4 | The conditional return distribution derivation. Verify you understand the M matrix, `Σ_r_cond`, and the `mu_r[i,j]` formula. |

The earlier income quadrature refactor (`QUADRATURE_REFACTOR_HANDOFF.md`) used the same architectural pattern — replacing a discrete transition matrix with GH quadrature + interpolation. Read §2 of that document to see how it was done for the z dimension.

---

## 1. SCOPE

Replace the discrete Markov chain integration over state innovations `v^s ~ N(0, Σ_ss)` with Gauss-Hermite quadrature. The state grid becomes pure interpolation support for the policy function, no longer a discrete state space for a Markov chain.

**What changes:**
- The `for j_s in range(N_state)` loop in the FOC functions becomes `for k_v in range(n_state_quad)`
- Policy lookups go from `c_next_full[iz, j_s, iw]` (direct grid index) to trilinear interpolation across 8 state-grid corners
- Return means `mu_r[i_s, j_s, :]` (looked up from a precomputed array) become `mu_r_k = const_r + A_r @ s_i + M @ v_nodes[k_v]` (computed on the fly)
- Simulation draws continuous `v^s` instead of discrete next states from `Pi_state`

**What does NOT change:**
- The Newton solver structure (corners, edges, interior)
- The EGM inversion step
- The z-interpolation (Catmull-Rom cubic) — unchanged, just wrapped inside a trilinear state interpolation
- The ε and η quadrature for income — unchanged
- The return residual quadrature — unchanged
- `R_bill` and `annuity_factor_is` — both use current state `i_s`, constant in the `k_v` loop
- The policy array shape: `(n_age, n_z, N_state, n_w)` — unchanged
- The backward induction structure in `run_lifecycle_solver` — unchanged

---

## 2. IMPLEMENTATION PLAN (5 PHASES)

### Phase 1: New quadrature function + config (no solver changes yet)
### Phase 2: New precompute arrays
### Phase 3: New FOC functions (retirement first, then working age)
### Phase 4: New period solvers that use the new FOC functions
### Phase 5: Simulation update + debugging tests

Each phase is independently testable. Do NOT proceed to the next phase until the current phase passes its tests.

---

## 3. PHASE 1 — `get_state_quadrature()` + Config

### 3.1 Add field to `DiscretizationConfig` (model.py)

Add one field after `n_ret_nodes_1d`:

```python
# model.py, class DiscretizationConfig, after line 112
n_state_quad_nodes: int = 3         # GH order per state dimension for state innovation quadrature
```

Default K=3 → 27 total nodes. This matches the income quadrature convention where `n_eta_nodes=3` gives 3 nodes per mixture component.

### 3.2 Add `get_state_quadrature()` to discretization.py

Place it after `get_return_quadrature` (after line 264). Copy the structure of `get_return_quadrature` exactly, but use `model.Sigma_ss` instead of `model.Sigma_r_cond` and `model.n_state` instead of `model.n_ret`:

```python
def get_state_quadrature(model, n_nodes=3):
    """State innovation quadrature for N(0, Sigma_ss).

    Constructs K^n_state tensor-product Gauss-Hermite nodes for integrating
    over the state innovation v^s ~ N(0, Sigma_ss). Used by the solver to
    replace the discrete Markov chain Pi_state.

    Parameters
    ----------
    model : LifecyclePortfolioModel
        Supplies `n_state` and `Sigma_ss`.
    n_nodes : int
        Gauss-Hermite order per state dimension. Total nodes = n_nodes ** n_state.

    Returns
    -------
    v_nodes : ndarray, shape (K_total, n_state)
        Innovation quadrature nodes in the original coordinate system.
    v_weights : ndarray, shape (K_total,)
        Tensor-product quadrature weights summing to one.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1 for state quadrature")

    n_state = int(model.n_state)

    nodes_1d, weights_1d = roots_hermite(n_nodes)
    weights_1d = weights_1d / np.sqrt(np.pi)
    nodes_1d = nodes_1d * np.sqrt(2.0)

    # Tensor product in standard-normal space
    grid_1d = np.meshgrid(*([nodes_1d] * n_state), indexing="ij")
    weight_1d = np.meshgrid(*([weights_1d] * n_state), indexing="ij")

    z_nodes = np.stack([g.ravel() for g in grid_1d], axis=1)     # (K_total, n_state)
    v_weights = np.prod(np.stack(weight_1d, axis=0), axis=0).ravel()  # (K_total,)

    # Transform from standard normal to N(0, Sigma_ss) via Cholesky
    Sigma = 0.5 * (np.asarray(model.Sigma_ss, dtype=float)
                    + np.asarray(model.Sigma_ss, dtype=float).T)
    L = np.linalg.cholesky(Sigma)
    v_nodes = z_nodes @ L.T       # (K_total, n_state)

    return v_nodes, v_weights
```

Note: uses Cholesky (like income quadrature) rather than eigendecomposition (like return quadrature). Both are correct for PD matrices; Cholesky is slightly cheaper.

### 3.3 Phase 1 test

```python
def test_state_quadrature_moments():
    """Verify GH quadrature reproduces N(0, Sigma_ss) moments."""
    from discretization import get_state_quadrature
    # Build model (use your standard build_model function)
    model = build_model(...)

    for K in [2, 3, 4]:
        v_nodes, v_weights = get_state_quadrature(model, n_nodes=K)

        # Weights sum to 1
        assert abs(v_weights.sum() - 1.0) < 1e-14, f"K={K}: weights sum = {v_weights.sum()}"

        # Mean = 0
        mean = v_weights @ v_nodes
        assert np.max(np.abs(mean)) < 1e-13, f"K={K}: mean = {mean}"

        # Covariance = Sigma_ss (exact for K >= 2)
        cov = np.zeros((model.n_state, model.n_state))
        for k in range(len(v_weights)):
            cov += v_weights[k] * np.outer(v_nodes[k], v_nodes[k])
        err = np.max(np.abs(cov - model.Sigma_ss))
        assert err < 1e-12, f"K={K}: cov error = {err}"

    print("PASS: state quadrature moments exact at K >= 2")
```

---

## 4. PHASE 2 — Precompute Arrays

### 4.1 Add new arrays to `Precompute.__init__`

Add the following after the `ret_nodes, ret_weights` block (after line 155 in precompute.py). Also add the import of `get_state_quadrature` at the top.

```python
# --- State innovation quadrature ---
from discretization import get_state_quadrature

self.v_nodes, self.v_weights = get_state_quadrature(
    model, n_nodes=disc_config.n_state_quad_nodes
)
# v_nodes:   (n_state_quad, n_state) float64 — innovation nodes in original coords
# v_weights: (n_state_quad,) float64 — tensor-product weights, sum to 1
self.n_state_quad = len(self.v_weights)

# --- Precomputed return formula constants (for on-the-fly mu_r computation) ---
self.const_r = model.Phi_0_ret - model.M @ model.Phi_0_state    # (n_ret,)
self.A_r = model.Phi_21 - model.M @ model.Phi_11                 # (n_ret, n_state)
# mu_r at quadrature node k_v, source state i_s:
#   mu_r_k = const_r + A_r @ s_i + M @ v_nodes[k_v]
# This is algebraically identical to: Phi_0_ret + Phi_21 @ s_i + M @ v^s_k

# Precompute M @ v_nodes for each quadrature node (avoids matmul in hot loop)
self.M_v_nodes = self.v_nodes @ model.M.T     # (n_state_quad, n_ret)
# Usage in solver: mu_r_k = base_mu_r_i + M_v_nodes[k_v, :]
# where base_mu_r_i = const_r + A_r @ s_i  (computed once per i_s)

# Precompute exp of the return-quadrature residual nodes (avoids exp in hot loop)
self.exp_ret_stock = np.exp(self.ret_nodes[:, 0])  # (n_ret_quad,)
self.exp_ret_bond  = np.exp(self.ret_nodes[:, 1])  # (n_ret_quad,)
```

### 4.2 Keep Pi_state and mu_r

**Do NOT remove `Pi_state` or `mu_r`.** They are still needed for:
- The terminal age solver (which has a simpler structure and can keep using Pi_state)
- The `_validate_conditional_returns` diagnostic
- Backward compatibility with the simulation (until Phase 5 is done)

### 4.3 Add a quadrature validation method

Add after `_validate_conditional_returns`:

```python
def _validate_state_quadrature(self):
    """Verify state quadrature reproduces conditional return moments.

    For each source state i, check:
      sum_k w_k * mu_r_k == Phi_0_ret + Phi_21 @ s_i  (unconditional return mean)
      sum_k w_k * v_k @ v_k.T == Sigma_ss             (innovation covariance)
    """
    model = self.model
    max_err_mean = 0.0
    for i in range(self.N_state):
        s_i = self.state_grid[i]
        base_mu_r = self.const_r + self.A_r @ s_i
        # Weighted average of mu_r_k = base_mu_r + M @ v_k
        avg_mu_r = base_mu_r + self.v_weights @ self.M_v_nodes
        target = model.Phi_0_ret + model.Phi_21 @ s_i
        err = np.max(np.abs(avg_mu_r - target))
        max_err_mean = max(max_err_mean, err)

    if self.verbose:
        print(f"  State quadrature return-mean consistency: max err = {max_err_mean:.2e}")

    assert max_err_mean < 1e-10, (
        f"State quadrature return-mean error {max_err_mean:.2e} too large"
    )
```

Call this at the end of `__init__`, after `_validate_conditional_returns`.

### 4.4 Phase 2 test

```python
def test_precompute_quadrature_arrays():
    """Check that precomputed arrays have correct shapes and values."""
    model = build_model(...)
    dc = DiscretizationConfig(state_grid_sizes=(5,5,5), n_state_quad_nodes=3)
    pc = Precompute(model, disc_config=dc)

    assert pc.v_nodes.shape == (27, 3)
    assert pc.v_weights.shape == (27,)
    assert abs(pc.v_weights.sum() - 1.0) < 1e-14
    assert pc.const_r.shape == (2,)
    assert pc.A_r.shape == (2, 3)
    assert pc.M_v_nodes.shape == (27, 2)
    assert pc.exp_ret_stock.shape == (pc.n_ret_quad,)

    # Verify M_v_nodes = v_nodes @ M.T
    check = pc.v_nodes @ model.M.T
    assert np.allclose(pc.M_v_nodes, check)

    # Verify const_r + A_r algebraic identity
    for i in range(pc.N_state):
        s_i = pc.state_grid[i]
        # Method 1: original formula
        mu_r_direct = model.Phi_0_ret + model.Phi_21 @ s_i
        # Method 2: rearranged formula at v=0 (zero innovation = unconditional)
        mu_r_rearranged = pc.const_r + pc.A_r @ s_i + model.M @ np.zeros(3)
        assert np.allclose(mu_r_direct, mu_r_rearranged), f"State {i}: identity fails"

    print("PASS: precompute quadrature arrays correct")
```

---

## 5. PHASE 3 — New FOC Functions

This is the core of the implementation. You will write two new functions that replace `compute_foc_jac_retirement` and `compute_foc_jac_working`.

### 5.1 Key design decisions (read before coding)

**Verified facts from code review — do NOT re-derive these:**

1. `R_bill = exp(r_bill_grid[i_s])` uses **current** state `i_s`, not next state. It is a scalar constant in the `k_v` loop. (solver.py:1639, precompute.py:157–159)

2. `annuity_factor_is = annuity_factors[i_s]` uses **current** state `i_s`. Also constant in `k_v` loop. (solver.py:1640, precompute.py:161–170)

3. The `Rx_stock_next[j_s, k_r] = exp(mu_r[i_s, j_s, 0] + ret_nodes[k_r, 0])` precomputation (solver.py:220–236) is **eliminated**. Instead, compute `exp(mu_r_k + ret_nodes[k_r])` on the fly inside the `k_v` loop. Factor as: `exp_mu_s = exp(mu_r_k[0])`, then `R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]`.

4. The policy array `c_next_full` has shape `(n_z, N_state, n_w)` for retirement and `(n_z, N_state, n_w)` for working age. The `j_s` index is the second dimension.

5. For trilinear interpolation, the 3D state grid is the Cartesian product of `state_grids[0]`, `state_grids[1]`, `state_grids[2]`. The flat index `j_s` maps to multi-index `(i0, i1, i2)` via `state_indices[j_s, :]`. The reverse mapping (multi-index → flat) is: `j_flat = i0 * N1 * N2 + i1 * N2 + i2` where `N1, N2` are the grid sizes of dimensions 1 and 2.

### 5.2 Trilinear interpolation helper

Add this to solver.py before the FOC functions:

```python
@njit(fastmath=True)
def bracket_state_3d(s_next, grids_0, grids_1, grids_2):
    """Bracket s_next in the 3D state grid.

    Returns (lo0, lo1, lo2, f0, f1, f2) where:
      grids_d[lo_d] <= s_next[d] < grids_d[lo_d + 1]
      f_d = (s_next[d] - grids_d[lo_d]) / (grids_d[lo_d+1] - grids_d[lo_d])
    Clamped to valid range.
    """
    # Dimension 0
    n0 = len(grids_0)
    if s_next[0] <= grids_0[0]:
        lo0, f0 = 0, 0.0
    elif s_next[0] >= grids_0[n0 - 1]:
        lo0 = n0 - 2
        f0 = 1.0
    else:
        lo0 = 0
        for ii in range(n0 - 1):
            if grids_0[ii + 1] > s_next[0]:
                lo0 = ii
                break
        dg = grids_0[lo0 + 1] - grids_0[lo0]
        f0 = (s_next[0] - grids_0[lo0]) / dg if dg > 1e-30 else 0.0
        f0 = max(0.0, min(1.0, f0))

    # Dimension 1
    n1 = len(grids_1)
    if s_next[1] <= grids_1[0]:
        lo1, f1 = 0, 0.0
    elif s_next[1] >= grids_1[n1 - 1]:
        lo1 = n1 - 2
        f1 = 1.0
    else:
        lo1 = 0
        for ii in range(n1 - 1):
            if grids_1[ii + 1] > s_next[1]:
                lo1 = ii
                break
        dg = grids_1[lo1 + 1] - grids_1[lo1]
        f1 = (s_next[1] - grids_1[lo1]) / dg if dg > 1e-30 else 0.0
        f1 = max(0.0, min(1.0, f1))

    # Dimension 2
    n2 = len(grids_2)
    if s_next[2] <= grids_2[0]:
        lo2, f2 = 0, 0.0
    elif s_next[2] >= grids_2[n2 - 1]:
        lo2 = n2 - 2
        f2 = 1.0
    else:
        lo2 = 0
        for ii in range(n2 - 1):
            if grids_2[ii + 1] > s_next[2]:
                lo2 = ii
                break
        dg = grids_2[lo2 + 1] - grids_2[lo2]
        f2 = (s_next[2] - grids_2[lo2]) / dg if dg > 1e-30 else 0.0
        f2 = max(0.0, min(1.0, f2))

    return lo0, lo1, lo2, f0, f1, f2
```

### 5.3 New retirement FOC: `compute_foc_jac_retirement_quad`

Write this as a new function (do NOT modify the existing `compute_foc_jac_retirement`). The existing function must remain for the terminal-age solver and for A/B comparison testing.

The function signature adds new parameters and removes `Pi_state`, `Rx_stock_next`, `Rx_bond_next`:

```python
@njit(fastmath=True)
def compute_foc_jac_retirement_quad(
    alpha_s, alpha_b, s_val, z_idx, i_s,
    wealth_grid, c_next_full, pension_next_scalar,
    annuity_factor_is,
    # --- State quadrature arrays (NEW) ---
    v_nodes, v_weights, M_v_nodes,
    base_mu_r_i,          # const_r + A_r @ s_i, precomputed per i_s
    Phi_0_state, Phi_11, state_grid_i,  # for computing s_next
    grids_0, grids_1, grids_2,          # marginal grids for bracketing
    N1, N2,                              # grid sizes dim 1, dim 2 (for flat indexing)
    # --- Return quadrature (unchanged) ---
    exp_ret_stock, exp_ret_bond, ret_weights, R_bill,
    # --- Model parameters ---
    gamma, psi, beta, b_bar,
    M_matrix,  # (2,3) for computing exp(mu_r_k)
    min_wealth_inv=1e-10, min_consumption=1e-10,
    prob_skip=1e-12,
):
```

**Inner loop structure:**

```python
    a_bill = 1.0 - alpha_s - alpha_b
    prob_death = 1.0 - psi
    foc_s = 0.0; foc_b = 0.0
    J_ss = 0.0; J_bb = 0.0; J_sb = 0.0
    euler_sum = 0.0

    n_state_quad = len(v_weights)
    n_ret_quad = len(ret_weights)

    for k_v in range(n_state_quad):
        w_v = v_weights[k_v]
        if w_v < prob_skip:
            continue

        # --- Continuous next state ---
        s_next_0 = Phi_0_state[0] + Phi_11[0,0]*state_grid_i[0] + Phi_11[0,1]*state_grid_i[1] + Phi_11[0,2]*state_grid_i[2] + v_nodes[k_v, 0]
        s_next_1 = Phi_0_state[1] + Phi_11[1,0]*state_grid_i[0] + Phi_11[1,1]*state_grid_i[1] + Phi_11[1,2]*state_grid_i[2] + v_nodes[k_v, 1]
        s_next_2 = Phi_0_state[2] + Phi_11[2,0]*state_grid_i[0] + Phi_11[2,1]*state_grid_i[1] + Phi_11[2,2]*state_grid_i[2] + v_nodes[k_v, 2]

        # --- Bracket s_next in 3D grid ---
        lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
            (s_next_0, s_next_1, s_next_2), grids_0, grids_1, grids_2
        )
        # NOTE: bracket_state_3d needs to accept a tuple or you inline the bracketing.
        # For numba compatibility, you may need to pass s_next as 3 separate floats
        # or as a small array. Adjust as needed.

        # 8 trilinear weights
        w000 = (1-f0)*(1-f1)*(1-f2)
        w001 = (1-f0)*(1-f1)*f2
        w010 = (1-f0)*f1*(1-f2)
        w011 = (1-f0)*f1*f2
        w100 = f0*(1-f1)*(1-f2)
        w101 = f0*(1-f1)*f2
        w110 = f0*f1*(1-f2)
        w111 = f0*f1*f2

        # 8 flat indices into c_next_full's j_s dimension
        j000 = lo0 * N1 * N2 + lo1 * N2 + lo2
        j001 = lo0 * N1 * N2 + lo1 * N2 + (lo2+1)
        j010 = lo0 * N1 * N2 + (lo1+1) * N2 + lo2
        j011 = lo0 * N1 * N2 + (lo1+1) * N2 + (lo2+1)
        j100 = (lo0+1) * N1 * N2 + lo1 * N2 + lo2
        j101 = (lo0+1) * N1 * N2 + lo1 * N2 + (lo2+1)
        j110 = (lo0+1) * N1 * N2 + (lo1+1) * N2 + lo2
        j111 = (lo0+1) * N1 * N2 + (lo1+1) * N2 + (lo2+1)

        # --- Conditional return mean ---
        mu_r_stock = base_mu_r_i[0] + M_v_nodes[k_v, 0]
        mu_r_bond  = base_mu_r_i[1] + M_v_nodes[k_v, 1]
        exp_mu_s = exp(mu_r_stock)
        exp_mu_b = exp(mu_r_bond)

        for k_r in range(n_ret_quad):
            p_ret = ret_weights[k_r]
            weight = w_v * p_ret
            if weight < prob_skip:
                continue

            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill

            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            w_inv = max(s_val * R_p, min_wealth_inv)
            x_next = w_inv + pension_next_scalar

            # --- Trilinear interpolation of c_next and mpc ---
            # For each of 8 corners, do a 1D wealth interpolation
            c000, mpc000 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j000, :])
            c001, mpc001 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j001, :])
            c010, mpc010 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j010, :])
            c011, mpc011 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j011, :])
            c100, mpc100 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j100, :])
            c101, mpc101 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j101, :])
            c110, mpc110 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j110, :])
            c111, mpc111 = fast_interp_1d_with_slope(x_next, wealth_grid, c_next_full[z_idx, j111, :])

            c_next = (w000*c000 + w001*c001 + w010*c010 + w011*c011
                    + w100*c100 + w101*c101 + w110*c110 + w111*c111)
            c_next = max(c_next, min_consumption)

            mpc = (w000*mpc000 + w001*mpc001 + w010*mpc010 + w011*mpc011
                 + w100*mpc100 + w101*mpc101 + w110*mpc110 + w111*mpc111)
            mpc = max(0.0, min(1.0, mpc))

            # --- Marginal utilities (identical to existing code) ---
            mu_alive = c_next ** (-gamma)
            w_A = w_inv / annuity_factor_is
            mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
            mu_comb = psi * mu_alive + prob_death * mu_bequest

            mup_alive = -gamma * mu_alive / c_next * mpc
            mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
            mup_comb = psi * mup_alive + prob_death * mup_bequest

            wmu = weight * mu_comb
            wmup = weight * mup_comb

            euler_sum += wmu * R_p
            foc_s += wmu * Rex_s
            foc_b += wmu * Rex_b

            jac = wmup * s_val
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum
```

### 5.4 New working-age FOC: `compute_foc_jac_working_quad`

Same structure as the retirement FOC, but the income loops (k_eta, i_e) are inside the (k_v, k_r) loop. The key change vs. the existing `compute_foc_jac_working`: the 8-corner trilinear wraps **around** the existing Catmull-Rom z-interpolation.

The innermost block (currently solver.py lines 866–930) becomes:

```python
# For each (k_eta, i_e), the Catmull-Rom z-interpolation at each of
# 8 state corners gives 8 c_next values. Blend with trilinear weights.
for k_eta in range(n_eta):
    ...
    for i_e in range(n_eps):
        ...
        # Trilinear blend of Catmull-Rom interpolated values
        c_next_interp = 0.0
        mpc_interp = 0.0

        # Corner (0,0,0)
        if use_cubic:
            c_val, mpc_val = _catmull_rom_z_wealth(
                c_next_full, j000, iz_lo, frac_z, iw, frac_w, inv_dw, n_z)
        else:
            c_val, mpc_val = _linear_z_wealth(
                c_next_full, j000, iz_lo, frac_z, iw, frac_w, inv_dw)
        c_next_interp += w000 * c_val
        mpc_interp += w000 * mpc_val

        # ... repeat for all 8 corners ...
```

**IMPORTANT:** This looks like a lot of code duplication. To keep it manageable, extract the Catmull-Rom block into a helper function:

```python
@njit(fastmath=True)
def _interp_z_wealth(c_next_full, j_s, iz_lo, frac_z, iw, frac_w, inv_dw, n_z, use_cubic, min_c):
    """Interpolate c_next and mpc at a single state-grid corner j_s.

    This is the existing Catmull-Rom logic from solver.py:877-916,
    extracted into a reusable function.
    """
    if use_cubic:
        # 4-point z stencil x 2-point wealth
        c_zm1 = (1.0-frac_w)*c_next_full[iz_lo-1, j_s, iw] + frac_w*c_next_full[iz_lo-1, j_s, iw+1]
        c_z0  = (1.0-frac_w)*c_next_full[iz_lo,   j_s, iw] + frac_w*c_next_full[iz_lo,   j_s, iw+1]
        c_z1  = (1.0-frac_w)*c_next_full[iz_lo+1, j_s, iw] + frac_w*c_next_full[iz_lo+1, j_s, iw+1]
        c_z2  = (1.0-frac_w)*c_next_full[iz_lo+2, j_s, iw] + frac_w*c_next_full[iz_lo+2, j_s, iw+1]
        f = frac_z; f2 = f*f; f3 = f2*f
        c_val = (c_z0
                 + 0.5*f*(-c_zm1 + c_z1)
                 + 0.5*f2*(2.0*c_zm1 - 5.0*c_z0 + 4.0*c_z1 - c_z2)
                 + 0.5*f3*(-c_zm1 + 3.0*c_z0 - 3.0*c_z1 + c_z2))
        c_val = max(c_val, min_c)

        mpc_zm1 = (c_next_full[iz_lo-1, j_s, iw+1] - c_next_full[iz_lo-1, j_s, iw]) * inv_dw
        mpc_z0  = (c_next_full[iz_lo,   j_s, iw+1] - c_next_full[iz_lo,   j_s, iw]) * inv_dw
        mpc_z1  = (c_next_full[iz_lo+1, j_s, iw+1] - c_next_full[iz_lo+1, j_s, iw]) * inv_dw
        mpc_z2  = (c_next_full[iz_lo+2, j_s, iw+1] - c_next_full[iz_lo+2, j_s, iw]) * inv_dw
        mpc_val = (mpc_z0
                   + 0.5*f*(-mpc_zm1 + mpc_z1)
                   + 0.5*f2*(2.0*mpc_zm1 - 5.0*mpc_z0 + 4.0*mpc_z1 - mpc_z2)
                   + 0.5*f3*(-mpc_zm1 + 3.0*mpc_z0 - 3.0*mpc_z1 + mpc_z2))
        mpc_val = max(0.0, min(1.0, mpc_val))
    else:
        c_lo = (1.0-frac_w)*c_next_full[iz_lo,   j_s, iw] + frac_w*c_next_full[iz_lo,   j_s, iw+1]
        c_hi = (1.0-frac_w)*c_next_full[iz_lo+1, j_s, iw] + frac_w*c_next_full[iz_lo+1, j_s, iw+1]
        c_val = (1.0-frac_z)*c_lo + frac_z*c_hi
        c_val = max(c_val, min_c)

        mpc_lo = (c_next_full[iz_lo,   j_s, iw+1] - c_next_full[iz_lo,   j_s, iw]) * inv_dw
        mpc_hi = (c_next_full[iz_lo+1, j_s, iw+1] - c_next_full[iz_lo+1, j_s, iw]) * inv_dw
        mpc_val = (1.0-frac_z)*mpc_lo + frac_z*mpc_hi
        mpc_val = max(0.0, min(1.0, mpc_val))

    return c_val, mpc_val
```

Then in the working-age FOC, the innermost loop uses this helper for all 8 corners.

### 5.5 Flat-index consistency check

The flat index `j_s` in `c_next_full[:, j_s, :]` maps to multi-index `(i0, i1, i2)` via the Cartesian product ordering. The formula `j = i0 * N1 * N2 + i1 * N2 + i2` MUST match the ordering used by `rouwenhorst_multivariate` (discretization.py:108). Verify this with:

```python
def test_flat_index_matches():
    """Verify flat indexing is consistent with state_indices."""
    pc = Precompute(model, disc_config=DiscretizationConfig(state_grid_sizes=(5,5,5)))
    N0, N1, N2 = 5, 5, 5
    for j_s in range(pc.N_state):
        i0, i1, i2 = pc.state_indices[j_s]
        j_computed = i0 * N1 * N2 + i1 * N2 + i2
        assert j_s == j_computed, f"Mismatch: j_s={j_s}, computed={j_computed}"
    print("PASS: flat index ordering matches state_indices")
```

**Run this test before proceeding.** If it fails, the trilinear interpolation indices will be wrong and the solver will produce garbage.

---

## 6. PHASE 4 — New Period Solvers

### 6.1 New retirement period solver

Write `_solve_retirement_step_quad_jit` mirroring `_solve_retirement_step_jit` (solver.py:1604–1775). The key differences:

1. **No `Rx_stock_next, Rx_bond_next` precomputation.** Remove the `build_gross_return_arrays` call. Instead, pass the quadrature arrays.

2. **Precompute `base_mu_r_i` once per `i_s`:**
   ```python
   for i_s in prange(N_state):
       s_i = state_grid[i_s]
       base_mu_r_i = np.empty(2)
       base_mu_r_i[0] = const_r[0] + A_r[0,0]*s_i[0] + A_r[0,1]*s_i[1] + A_r[0,2]*s_i[2]
       base_mu_r_i[1] = const_r[1] + A_r[1,0]*s_i[0] + A_r[1,1]*s_i[1] + A_r[1,2]*s_i[2]
       ...
   ```

3. **Call `compute_foc_jac_retirement_quad`** instead of `compute_foc_jac_retirement` inside the Newton solver and corner/edge checks.

4. **Pass the marginal grids** (`grids_0, grids_1, grids_2`) and grid sizes (`N1, N2`) to the FOC function.

### 6.2 New working-age period solver

Same pattern: `_solve_working_age_step_quad_jit` mirroring `_solve_working_age_step_jit`.

### 6.3 Routing in `run_lifecycle_solver`

Add a parameter to `run_lifecycle_solver` to select the integration method:

```python
def run_lifecycle_solver(model, pc, n_s_points=None, solver_config=None, verbose=1,
                         use_state_quadrature=True):
```

When `use_state_quadrature=True`, call the new `_quad` period solvers. When `False`, call the existing ones. This allows A/B comparison.

**The terminal age solver remains unchanged** — it uses `Pi_state` and the existing structure because the terminal period has a simpler optimization that doesn't benefit from the quadrature refactor.

### 6.4 Passing arrays through

The new period solvers need these additional arrays from `Precompute`:

```python
# In run_lifecycle_solver, extract once:
v_nodes = pc.v_nodes
v_weights = pc.v_weights
M_v_nodes = pc.M_v_nodes
const_r = pc.const_r
A_r = pc.A_r
state_grid = pc.state_grid
grids_0 = pc.state_grids[0]
grids_1 = pc.state_grids[1]
grids_2 = pc.state_grids[2]
N1 = len(grids_1)
N2 = len(grids_2)
exp_ret_stock = pc.exp_ret_stock
exp_ret_bond = pc.exp_ret_bond
Phi_0_state = model.Phi_0_state
Phi_11 = model.Phi_11
M_matrix = model.M
```

---

## 7. PHASE 5 — Simulation Update

### 7.1 Continuous state transitions

In `simulate_lifecycle_core` (simulation.py:478–545), replace:

```python
# OLD: discrete transition
next_state_idx = draw_discrete(Pi_state[state_idx, :], uniform_draws[i, t, 2])
```

with:

```python
# NEW: continuous transition
# Draw v^s = L_ss @ z, where z ~ N(0, I)
# Use normal_draws columns after the return shocks
v_s_0 = L_ss[0,0]*normal_draws[i,t,n_ret+1] + L_ss[0,1]*normal_draws[i,t,n_ret+2] + L_ss[0,2]*normal_draws[i,t,n_ret+3]
v_s_1 = L_ss[1,0]*normal_draws[i,t,n_ret+1] + L_ss[1,1]*normal_draws[i,t,n_ret+2] + L_ss[1,2]*normal_draws[i,t,n_ret+3]
v_s_2 = L_ss[2,0]*normal_draws[i,t,n_ret+1] + L_ss[2,1]*normal_draws[i,t,n_ret+2] + L_ss[2,2]*normal_draws[i,t,n_ret+3]

# Propagate state
s_current = state_grid[state_idx]
s_next_0 = Phi_0_state[0] + Phi_11[0,0]*s_current[0] + Phi_11[0,1]*s_current[1] + Phi_11[0,2]*s_current[2] + v_s_0
s_next_1 = Phi_0_state[1] + ...
s_next_2 = Phi_0_state[2] + ...

# Find nearest grid point for next period's policy lookup
next_state_idx = find_nearest_state_idx(s_next, state_grid, N1, N2, grids_0, grids_1, grids_2)
```

This requires:
- Expanding `normal_draws` to include 3 extra columns for state innovations (n_ret + 1 + 3 total)
- Passing `L_ss = cholesky(Sigma_ss)`, `Phi_0_state`, `Phi_11`, `state_grid` to the simulation kernel
- A nearest-state-index function (or trilinear interpolation of policies at continuous s_next — more accurate but more complex)

### 7.2 Return realization

Currently `mu_r[state_idx, next_state_idx, :]` is looked up. With continuous states:

```python
# Compute conditional return mean given continuous v^s
mu_xr = const_r[0] + A_r[0,:] @ s_current + M[0,:] @ v_s
mu_xb = const_r[1] + A_r[1,:] @ s_current + M[1,:] @ v_s
```

Then the return residual draw proceeds as before.

### 7.3 Defer if needed

The simulation update is the lowest-priority phase. If time is short, keep the existing Pi_state-based simulation and note the inconsistency. The solver is the critical path.

---

## 8. DEBUGGING TESTS

Run these tests at each phase. They are ordered from cheapest to most expensive.

### Test 1: Quadrature moment exactness (Phase 1)

Already given in §3.3. Verifies weights sum to 1, mean = 0, covariance = Σ_ss.

### Test 2: Return-mean consistency (Phase 2)

Already given in §4.4. Verifies `Σ_k w_k × mu_r_k = Φ_0_ret + Φ_21 @ s_i` at every source state.

### Test 3: Flat index ordering (Phase 3)

Already given in §5.5. Critical correctness check for trilinear indexing.

### Test 4: FOC equivalence at identity (Phase 3)

When `s_i` is at the unconditional mean and `v_nodes[k]` corresponds to a grid point transition (i.e., `s_next` lands exactly on a grid point), the quadrature FOC should give the same value as the Markov FOC with all weight on that grid point. This is a sanity check, not an exact equivalence test.

```python
def test_foc_single_node():
    """Verify FOC at a single node matches the direct computation."""
    # Set up a single quadrature node at v=0 (the mean)
    # This should give mu_r_k = Phi_0_ret + Phi_21 @ s_i (same as marginalizing over Pi)
    # Compare FOC value to the Markov version with Pi_state weight concentrated at E[s']
    ...
```

### Test 5: One-period retirement solve comparison (Phase 4)

Solve a single retirement period using both the old (Markov) and new (quadrature) methods. Compare policies:

```python
def test_retirement_one_period():
    """Solve one retirement period with both methods and compare."""
    # Terminal condition is the same
    c_T, a_s_T, a_b_T, _ = solve_terminal_age(...)

    # Solve t=T-1 with Markov
    c_old, a_s_old, a_b_old, _, _ = solve_retirement_step(
        ..., Pi_state=pc.Pi_state, mu_r=pc.mu_r, ...)

    # Solve t=T-1 with quadrature
    c_new, a_s_new, a_b_new, _, _ = solve_retirement_step_quad(
        ..., v_nodes=pc.v_nodes, v_weights=pc.v_weights, ...)

    # Policies should be SIMILAR but not identical (different integration methods)
    # At interior states, expect < 5% relative difference in consumption
    mask = a_s_old > 0.01  # skip corners
    rel_err_c = np.abs(c_new[mask] - c_old[mask]) / np.maximum(c_old[mask], 1e-6)
    print(f"Consumption: mean rel err = {rel_err_c.mean():.4f}, max = {rel_err_c.max():.4f}")

    # Portfolio shares may differ more (they're sensitive to covariance structure)
    rel_err_as = np.abs(a_s_new[mask] - a_s_old[mask])
    print(f"Stock share: mean abs err = {rel_err_as.mean():.4f}, max = {rel_err_as.max():.4f}")
```

### Test 6: Full lifecycle solve smoke test (Phase 4)

```python
def test_full_solve_smoke():
    """Run a full solve with small grid and check it completes without errors."""
    dc = DiscretizationConfig(
        state_grid_sizes=(5, 5, 5),
        n_state_quad_nodes=3,
        n_z=7,
        n_wealth=50,
        n_savings=50,
    )
    pc = Precompute(model, disc_config=dc)
    C, S, B, diag = run_lifecycle_solver(
        model, pc, solver_config=SolverConfig(),
        use_state_quadrature=True
    )

    # Basic sanity
    assert not np.any(np.isnan(C))
    assert not np.any(np.isnan(S))
    assert not np.any(np.isnan(B))
    assert diag['total_newton_failures'] < 100  # some failures OK at small grid

    # Economic sanity: stock share should decline with age (on average)
    mean_stock_by_age = S[:, S.shape[1]//2, S.shape[2]//2, S.shape[3]//2]
    assert mean_stock_by_age[0] > mean_stock_by_age[-2], "Stock share should decline with age"

    print(f"PASS: full solve completed, {diag['total_newton_failures']} failures")
```

### Test 7: Conditional moment properties P4–P9 (Phase 2)

```python
def test_conditional_moments():
    """Verify P4-P9 from the property list."""
    pc = Precompute(model, disc_config=DiscretizationConfig(n_state_quad_nodes=3))

    # P4: Conditional mean
    for i in range(pc.N_state):
        s_i = pc.state_grid[i]
        # E[s_next | s_i] = Phi_0 + Phi_11 @ s_i
        target = model.Phi_0_state + model.Phi_11 @ s_i
        computed = np.zeros(3)
        for k in range(pc.n_state_quad):
            s_next = model.Phi_0_state + model.Phi_11 @ s_i + pc.v_nodes[k]
            computed += pc.v_weights[k] * s_next
        assert np.allclose(computed, target, atol=1e-12)

    # P5: Conditional covariance
    # Cov[s_next | s_i] = Sigma_ss (same for all i)
    for i in [0, pc.N_state//2, pc.N_state-1]:
        s_i = pc.state_grid[i]
        mean = model.Phi_0_state + model.Phi_11 @ s_i
        cov = np.zeros((3, 3))
        for k in range(pc.n_state_quad):
            s_next = model.Phi_0_state + model.Phi_11 @ s_i + pc.v_nodes[k]
            dev = s_next - mean
            cov += pc.v_weights[k] * np.outer(dev, dev)
        assert np.allclose(cov, model.Sigma_ss, atol=1e-12)

    # P8: State-return innovation covariance
    # E[v^s × (M @ v^s)'] = Sigma_ss @ M' = Sigma_sr'
    Sigma_sr_expected = model.Sigma_ss @ model.M.T
    cov_sr = np.zeros((3, 2))
    for k in range(pc.n_state_quad):
        v = pc.v_nodes[k]
        ret_innov = model.M @ v  # conditional return innovation
        cov_sr += pc.v_weights[k] * np.outer(v, ret_innov)
    # Sigma_sr = Sigma_rs.T, and Sigma_rs = model.Sigma_rs
    assert np.allclose(cov_sr, model.Sigma_rs.T, atol=1e-12)

    print("PASS: conditional moments P4, P5, P8 exact at K >= 2")
```

---

## 9. RISK AREAS

### 9.1 Numba compatibility

All new functions must be `@njit(fastmath=True)` compatible. Common pitfalls:
- Cannot pass Python lists to njit functions — use numpy arrays
- Cannot create numpy arrays with `np.array([a, b, c])` inside njit — use `np.empty(3)` and assign elements
- Cannot use `np.ix_` or fancy indexing inside njit — use explicit loops or scalar indexing
- The `bracket_state_3d` helper takes 3 separate 1D grids — verify numba accepts this

### 9.2 Grid ordering

The flat index formula `j = i0 * N1 * N2 + i1 * N2 + i2` assumes C-order (row-major) layout matching `np.ndindex`. If `rouwenhorst_multivariate` uses a different ordering, the trilinear interpolation will read the wrong grid corners. **Test 3 in §8 catches this.**

### 9.3 Boundary clamping

When `s_next` falls outside the state grid, the bracket function clamps to the boundary. This means:
- `f0 = 0.0` or `f0 = 1.0` at the boundary
- The interpolation degenerates to bilinear (in 2 remaining dimensions)
- `c_next` may be less accurate at grid extremes

Monitor the fraction of quadrature evaluations that hit a boundary. If > 5% at any age, widen the state grid (increase `n_stds` or grid range coverage in `rouwenhorst_univariate`).

### 9.4 Newton convergence

The quadrature FOC is a weighted sum over 27 nodes (K=3) instead of 125 states. The FOC surface may be smoother but slightly different from the Markov version. The Newton solver's convergence should be at least as good — monitor the failure rate.

### 9.5 Memory layout

`c_next_full[iz, j_s, iw]` is accessed with `j_s` varying across 8 corners and `iz` varying across 4 Catmull-Rom points. For best cache performance, ensure `c_next_full` is C-contiguous (which it is, since it's created by `np.empty`). The 8 corners access nearby `j_s` values (differing by 1, N2, or N1*N2), which may span non-adjacent memory if N1*N2 is large. At (7,7,7), N1*N2=49, so the stride between corner 000 and corner 100 is 49*n_w*8 bytes = 49*150*8 ≈ 59 KB — likely spanning L1 cache lines but fitting in L2.

---

## 10. SUMMARY CHECKLIST

- [ ] Phase 1: `get_state_quadrature()` added to discretization.py, moment test passes
- [ ] Phase 1: `n_state_quad_nodes` added to DiscretizationConfig
- [ ] Phase 2: Precompute stores v_nodes, v_weights, M_v_nodes, const_r, A_r, exp_ret_stock/bond
- [ ] Phase 2: `_validate_state_quadrature` passes
- [ ] Phase 2: Flat index ordering test passes
- [ ] Phase 3: `_interp_z_wealth` helper extracted and tested
- [ ] Phase 3: `bracket_state_3d` helper written and tested
- [ ] Phase 3: `compute_foc_jac_retirement_quad` compiles under numba
- [ ] Phase 3: `compute_foc_jac_working_quad` compiles under numba
- [ ] Phase 4: `_solve_retirement_step_quad_jit` compiles and produces finite policies
- [ ] Phase 4: `_solve_working_age_step_quad_jit` compiles and produces finite policies
- [ ] Phase 4: One-period comparison test shows < 5% consumption difference vs Markov
- [ ] Phase 4: Full lifecycle smoke test passes
- [ ] Phase 4: `run_lifecycle_solver(use_state_quadrature=True)` completes without crash
- [ ] Phase 5: Simulation with continuous state transitions (if time permits)
- [ ] All: Conditional moment tests P4, P5, P8 pass
- [ ] All: No NaN/Inf in policy arrays
- [ ] All: Newton failure rate < 0.1%

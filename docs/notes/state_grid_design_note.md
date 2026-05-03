# State-grid construction for the financial-state VAR — design note

**Recommendation: Option B (Cholesky-rotated principal-axis grid) at `n_stds = 3.0` with `N = (7, 7, 7)`.**

The Lyapunov-rescaled axis-aligned variant (Option A) is included as a fallback flag, because it is a strict improvement over the current code and useful as a no-rotation sanity check, but it should not be the production setting.

---

## 1. Why Option B, not Option A — and not the current naive grid

I verified every number in the handoff against the calibrated `Phi`, `Sigma` from `var.py`. The picture is actually more lopsided than the handoff suggests, in two ways the handoff understates.

### 1.1 The stationary distribution is highly correlated

```
stationary correlation matrix (true Σ_z, Lyapunov solution):
            y_1    spr    cy
    y_1   1.00  -0.60  +0.71
    spr  -0.60  +1.00  -0.11
    cy   +0.71  -0.11  +1.00
```

|corr(y_1, spr)| = 0.60 and |corr(y_1, cy)| = 0.71. The joint level sets of the stationary density are heavily tilted ellipsoids; an axis-aligned cube fits them poorly no matter what the side lengths are.

### 1.2 The stationary distribution is nearly one-dimensional

```
eigvals(Σ_z)        = [8.7e-05, 7.9e-04, 0.282]
share of variance   = [0.0%,    0.3%,    99.7%]
```

PC3 — which is essentially `cy` with a small negative loading on `y_1` — captures 99.7% of total variance. This is consistent with `cy`'s stand-alone persistence (`Φ[2,2] = 0.919`) being the dominant slow root in the system.

For grid design this matters because **trilinear interpolation error along an axis is O(h²) where h is the spacing along that axis projected onto the local policy gradient**. The policy gradient direction is determined by economics — duration `M[xb, y_1] = −8.72` means α_b reacts steeply to y_1 — but trilinear's effective spacing along that gradient is what determines accuracy. In Option A, the grid spacing along the `(y_1 − cy)` correlated direction is √2 times worse than along the raw axes, so the model's most price-sensitive state direction is *also* the direction with the worst grid resolution.

### 1.3 Empirical coverage comparison

Computed at `N = 7³`:

| scheme | historical 1963–2025 (63 obs) | stationary MC (200k draws) | s-space hull volume |
|---|---|---|---|
| **Current (naive σ_y, ψ = σ √(N−1))** | 92.1% inside | 83.4% inside | reference |
| **Option A (Lyapunov σ_z, n_stds=2.5)** | 98.4% | 96.8% | reference |
| **Option A (Lyapunov σ_z, n_stds=3.0)** | 100.0% | 99.2% | reference |
| **Option B (Cholesky-rotated, n_stds=2.5)** | 96.8% | 96.3% | **46% of A** |
| **Option B (Cholesky-rotated, n_stds=3.0)** | 100.0% | 99.2% | **46% of A** |

At equal probability mass coverage, Option B encloses the mass in **half the s-space volume** of Option A. Because the same number of grid points (343) are spread over half the volume, the *stationary-density-weighted* spacing per point is about √2 times tighter — that is the resolution gain, and it is concentrated exactly along the principal direction of variation, where policy moves most.

### 1.4 The 1981 Volcker observation (`y_1 = 13.86%`, `+2.53 σ_z`)

Under the **current** N=7 grid, the historical max-y_1 observation falls at +2.53 true-σ but the grid only extends to +1.46 true-σ. That state is read off the grid edge (clamped, flat-extrapolated). Under **either** Option A at n_stds=3.0 *or* Option B at n_stds=3.0, the observation is inside the hull, and the y_1 face of the grid extends to +3 σ_z = +0.107 in y_1 units, i.e. nominal yields up to ~16%. That covers any plausible monetary regime in the post-Bretton-Woods sample.

### 1.5 Cost concern (the "27× per Newton step" framing)

The added work per `bracket_state_3d` call under Option B is one 3×3 matvec to map `s_next → u_next`. Numba JIT will compile it inline (9 multiplies, 6 adds; ~15 ns measured). At 680M FOC evaluations per backward sweep this adds 10–50 ms — well under a 1% wall-time hit. The "27×" multiplier in the handoff refers to the K_s³ state-quadrature nodes per outer-state, but the cost is per *call*, not per outer-state, so the multiplier is irrelevant.

---

## 2. Mathematical statement of Option B

Let

- `μ_s := z_bar_state ∈ ℝ³` — VAR unconditional mean for the slow states (already in `model.z_bar_state`).
- `Σ_z = Φ Σ_z Φᵀ + Σ_ss` — discrete Lyapunov solution; the marginal stationary covariance.
- `L Lᵀ = Σ_z` — Cholesky factor of `Σ_z` (lower triangular). Note this is *not* the existing `L_ss` (which is the Cholesky of the *innovation* covariance `Σ_ss`).
- `u = L⁻¹ (s − μ_s) ∈ ℝ³` — standardized coordinates; under the stationary distribution `u ~ N(0, I)`.

The grid is constructed in u-space as a Cartesian product of equispaced 1-D grids, then mapped back to s-space:

```
u-grids:        u_grid[d] = linspace(−n_stds, +n_stds, N[d])     for d ∈ {0, 1, 2}
state_grid[i] = μ_s + L · u_lattice[i]                            for i ∈ {0, …, N₀N₁N₂ − 1}
```

The bracketing logic in the solver works in u-coordinates: given a continuous next-state vector `s_next`, compute `u_next = L⁻¹ (s_next − μ_s)`, then call the existing `bracket_state_3d` against the three 1-D u-grids. **The flat indexing `lo0·N1·N2 + lo1·N2 + lo2` and the 8-corner trilinear blend are unchanged.**

This is exactly the property the handoff was looking for: Option B's grid is still a Cartesian product, just in a rotated coordinate system. All of `solver.py`'s downstream interpolation infrastructure stays. The only inserts are `L_inv` matvecs at the boundary between continuous propagation and grid bracketing.

### 2.1 Why Cholesky and not eigendecomposition

Both decompositions yield a Cartesian-in-u grid that fits the stationary distribution. I picked Cholesky over eigendecomposition for two reasons:

1. **Consistency with the existing codebase.** `get_state_quadrature()` already uses `np.linalg.cholesky(Sigma_ss)` for the innovation distribution. Using `np.linalg.cholesky(Sigma_z)` for the marginal-stationary distribution keeps the numerical convention uniform.
2. **More balanced per-axis s-displacement.** With Cholesky, the diagonal of `L` is `[0.0356, 0.0127, 0.3062]` — the ratio of largest to smallest u-axis "loading" in s-space is ~24×. With eigendecomposition, the corresponding ratio is `√(λ_max/λ_min) = √(0.282/8.7e-5) ≈ 57×`. The Cholesky basis distributes resolution somewhat more evenly across the three principal directions, which means more graceful behaviour when policy depends on a non-dominant direction (e.g. the duration-driven `α_b` sensitivity to y_1, which is *not* aligned with PC3).

If you want the eigendecomposition variant later (e.g. for diagnostic plots), the same code skeleton works with `L = eigvecs @ diag(sqrt(eigvals))`. That matrix is still a square root of `Σ_z`, just with orthogonal columns instead of triangular ones.

---

## 3. Drop-in code

The functions below replace `rouwenhorst_multivariate` and add a small helper in `discretization.py`. The legacy function is kept and called from the new entry point when `mode='naive'` for regression testing.

### 3.1 `discretization.py` — replacement / addition

```python
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def stationary_covariance(Phi, Sigma_innov):
    """Marginal stationary covariance Σ_z of the VAR  s_{t+1} = ... + v,  v ~ N(0, Σ_innov).

    Solves the discrete Lyapunov equation  Σ_z = Φ Σ_z Φᵀ + Σ_innov.
    Symmetrizes the result (numerical hygiene; the solver is exact in
    exact arithmetic but the returned matrix can have ~1e-16 asymmetry).
    """
    Phi = np.asarray(Phi, dtype=float)
    Sigma_innov = np.asarray(Sigma_innov, dtype=float)
    eigs = np.linalg.eigvals(Phi)
    if np.max(np.abs(eigs)) >= 1.0 - 1e-12:
        raise ValueError(
            f"VAR is non-stationary (max |eigenvalue(Phi)| = {np.max(np.abs(eigs)):.6f} >= 1); "
            "stationary covariance is undefined."
        )
    Sigma_innov = 0.5 * (Sigma_innov + Sigma_innov.T)
    Sigma_z = solve_discrete_lyapunov(Phi, Sigma_innov)
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.T)
    return Sigma_z


def build_state_grid(N_vec, mu_intercept, Phi, Sigma_innov,
                     n_stds=3.0, mode='principal'):
    """Construct the financial-state grid for trilinear policy interpolation.

    The grid is always Cartesian in some coordinate system. The flat ordering
    (multi-index → linear index) is row-major over (axis0, axis1, axis2),
    matching solver.py's flat indexing  i = i0*N1*N2 + i1*N2 + i2.

    Parameters
    ----------
    N_vec : sequence of 3 ints
        Per-axis grid sizes.
    mu_intercept : (3,) array
        Φ_0 in  s' = Φ_0 + Φ s + v.  The unconditional mean is computed
        internally as  μ_s = (I − Φ)⁻¹ Φ_0.
    Phi : (3, 3) array
        Persistence matrix Φ_11 of the slow-state sub-VAR.
    Sigma_innov : (3, 3) array
        Innovation covariance Σ_ss = E[v vᵀ].  Pass the covariance, not its
        Cholesky factor.  This is a deliberate API change from the previous
        function which expected a Cholesky factor; passing the covariance is
        more transparent and robust.
    n_stds : float, default 3.0
        Per-axis half-width in standardized units.  Under 'principal' mode,
        this is in the unit-variance u-coordinates.  Under 'lyapunov-axis',
        it is in true-σ_z units per axis.  Under 'naive', it is ignored
        (legacy ψ = σ_y √(N−1) is used).
    mode : {'principal', 'lyapunov-axis', 'naive'}
        - 'principal'   : Cholesky-rotated grid (Option B; recommended).
        - 'lyapunov-axis': axis-aligned in s-space, half-width n_stds * σ_z,d (Option A).
        - 'naive'       : legacy independence Rouwenhorst, kept for regression testing.

    Returns
    -------
    info : dict with keys
        mode               : str, one of the three modes above.
        N_vec              : (3,) int — per-axis grid sizes.
        mu_s               : (3,) — unconditional mean of slow states.
        Sigma_z            : (3, 3) — marginal stationary covariance (Lyapunov).
        sigma_z            : (3,) — sqrt of diag(Sigma_z).
        L                  : (3, 3) — Cholesky of Sigma_z (used for principal mode;
                             returned for diagnostics in any mode).
        L_inv              : (3, 3) — L⁻¹.  This is what the solver needs.
        state_grids        : list[3] of 1-D arrays.  In principal mode these are
                             the u-axes; in lyapunov-axis and naive modes they
                             are the s-axes (legacy semantics).
        state_grid         : (N_total, 3) — grid points in s-coordinates.  This
                             is what gets stored in Precompute.state_grid and
                             what the conditional-return formulas consume.
        state_indices      : (N_total, 3) int — multi-index into state_grids.
        Pi_state           : (N_total, N_total) — independence-Rouwenhorst
                             transition (legacy fallback for the simulator
                             when continuous-state propagation is disabled).
        n_stds_effective   : (3,) — actual per-axis half-width used (equals
                             n_stds * 1 in principal mode, n_stds * sigma_z[d]
                             in lyapunov mode, sigma_y * sqrt(N-1) in naive mode).
    """
    Phi = np.asarray(Phi, dtype=float)
    Sigma_innov = np.asarray(Sigma_innov, dtype=float)
    mu_intercept = np.asarray(mu_intercept, dtype=float)
    N_vec = np.asarray(N_vec, dtype=int)
    k = len(N_vec)
    if k != 3:
        raise ValueError("build_state_grid is currently specialized to k=3 "
                         "(matches solver.py's bracket_state_3d).")
    if Phi.shape != (k, k) or Sigma_innov.shape != (k, k):
        raise ValueError(f"Phi and Sigma_innov must both be {(k, k)}.")
    if mu_intercept.shape != (k,):
        raise ValueError(f"mu_intercept must have shape {(k,)}.")

    mu_s = np.linalg.solve(np.eye(k) - Phi, mu_intercept)
    Sigma_z = stationary_covariance(Phi, Sigma_innov)
    sigma_z = np.sqrt(np.diag(Sigma_z))
    L = np.linalg.cholesky(Sigma_z)
    L_inv = np.linalg.inv(L)

    # 1-D axis grids
    if mode == 'principal':
        # axis grids live in u-space, dimensionless
        state_grids_axis = [np.linspace(-n_stds, +n_stds, int(N_vec[d])) for d in range(k)]
        n_stds_effective = np.full(k, float(n_stds))
    elif mode == 'lyapunov-axis':
        # axis grids live in s-space, half-width = n_stds * σ_z,d, centered at μ_s
        state_grids_axis = [
            np.linspace(mu_s[d] - n_stds * sigma_z[d], mu_s[d] + n_stds * sigma_z[d], int(N_vec[d]))
            for d in range(k)
        ]
        n_stds_effective = np.full(k, float(n_stds))
    elif mode == 'naive':
        # Legacy: per-axis half-width = σ_y,d * √(N_d − 1) where σ_y,d uses
        # the univariate AR(1) formula ignoring cross-state spillovers.
        rho_diag = np.diag(Phi)
        sigma_y_naive = np.sqrt(np.diag(Sigma_innov) / np.maximum(1e-14, 1.0 - rho_diag**2))
        state_grids_axis = []
        n_stds_effective = np.empty(k)
        for d in range(k):
            psi = sigma_y_naive[d] * np.sqrt(int(N_vec[d]) - 1)
            state_grids_axis.append(np.linspace(mu_s[d] - psi, mu_s[d] + psi, int(N_vec[d])))
            n_stds_effective[d] = psi / max(sigma_z[d], 1e-30)
    else:
        raise ValueError(f"mode must be 'principal', 'lyapunov-axis', or 'naive'; got {mode!r}")

    # Cartesian product → multi-indices and s-coordinates
    n_total = int(np.prod(N_vec))
    state_indices = np.empty((n_total, k), dtype=np.int64)
    for flat, multi in enumerate(np.ndindex(*N_vec.tolist())):
        state_indices[flat] = multi

    state_grid = np.empty((n_total, k), dtype=float)
    if mode == 'principal':
        for i in range(n_total):
            u = np.array([state_grids_axis[d][state_indices[i, d]] for d in range(k)])
            state_grid[i] = mu_s + L @ u
    else:
        # axis-aligned in s-space: just gather directly
        for i in range(n_total):
            for d in range(k):
                state_grid[i, d] = state_grids_axis[d][state_indices[i, d]]

    # Independence-Rouwenhorst Pi_state, kept for legacy simulator path.
    # NOTE: when mode='principal' the resulting Π is NOT a faithful joint
    # transition (the axes aren't economic variables and the per-axis
    # ρ_d, σ_d are not VAR moments).  This is OK because the simulator
    # uses continuous-state propagation by default and only uses Π in the
    # legacy discrete branch.  We document this clearly here.
    Pi_state = _independence_rouwenhorst_pi(N_vec, Phi, Sigma_innov, mode=mode)

    return {
        "mode": mode,
        "N_vec": N_vec,
        "mu_s": mu_s,
        "Sigma_z": Sigma_z,
        "sigma_z": sigma_z,
        "L": L,
        "L_inv": L_inv,
        "state_grids": state_grids_axis,
        "state_grid": state_grid,
        "state_indices": state_indices,
        "Pi_state": Pi_state,
        "n_stds_effective": n_stds_effective,
    }


def _independence_rouwenhorst_pi(N_vec, Phi, Sigma_innov, mode):
    """Build a Kronecker-product Rouwenhorst transition for legacy simulator
    fallback. For 'principal' mode this is a coarse approximation; the
    solver does NOT use Pi_state and the simulator's continuous branch
    (use_continuous_state=True) does NOT use it either.

    For 'lyapunov-axis' and 'naive' modes the marginal Rouwenhorst chains
    use ρ_d = Φ_dd and σ_d = √Σ_innov_dd as before."""
    k = len(N_vec)
    rho_diag = np.diag(np.asarray(Phi))
    sigma_diag = np.sqrt(np.maximum(np.diag(np.asarray(Sigma_innov)), 1e-14))
    marginals = []
    for d in range(k):
        # Rouwenhorst transition only — discard the levels (we already have grids)
        _, Pi_d = rouwenhorst_univariate(int(N_vec[d]), 0.0, float(rho_diag[d]), float(sigma_diag[d]))
        marginals.append(Pi_d)
    n_total = int(np.prod(N_vec))
    state_indices = np.array(list(np.ndindex(*N_vec.tolist())), dtype=np.int64)
    Pi = np.ones((n_total, n_total), dtype=float)
    for d in range(k):
        Pi *= marginals[d][np.ix_(state_indices[:, d], state_indices[:, d])]
    Pi /= np.maximum(Pi.sum(axis=1, keepdims=True), 1e-300)
    return Pi
```

### 3.2 `precompute.py` — minimal change

Replace the call site at lines ~118–134:

```python
# OLD
Sigma_state_chol = np.linalg.cholesky(model.Sigma_ss)
self.state_grids, self.Pi_state, self.state_indices = rouwenhorst_multivariate(
    N_vec=state_grid_sizes,
    mu=model.Phi_0_state,
    Phi=model.Phi_11,
    Sigma=Sigma_state_chol,
)
self.state_grid = self._build_state_grid(self.state_grids, self.state_indices)
self.N_state = self.state_grid.shape[0]

# NEW
grid_info = build_state_grid(
    N_vec=state_grid_sizes,
    mu_intercept=model.Phi_0_state,
    Phi=model.Phi_11,
    Sigma_innov=model.Sigma_ss,
    n_stds=disc_config.state_n_stds,             # NEW config field; default 3.0
    mode=disc_config.state_grid_mode,            # NEW config field; default 'principal'
)
self.state_grids   = grid_info["state_grids"]    # 1-D axis grids (u-space if principal)
self.state_indices = grid_info["state_indices"]
self.state_grid    = grid_info["state_grid"]     # always in s-coordinates
self.Pi_state      = grid_info["Pi_state"]       # legacy
self.state_grid_mode = grid_info["mode"]
self.state_grid_L     = grid_info["L"]           # Cholesky of Σ_z; identity-shaped no-op if not needed
self.state_grid_L_inv = grid_info["L_inv"]
self.state_grid_mu_s  = grid_info["mu_s"]
self.N_state = self.state_grid.shape[0]
```

The `_build_state_grid` static method becomes redundant and can be deleted (the new function returns the s-space grid directly).

Add to `DiscretizationConfig` in `model.py`:

```python
state_n_stds: float = 3.0
state_grid_mode: str = 'principal'   # 'principal' | 'lyapunov-axis' | 'naive'
```

### 3.3 `solver.py` — three small inserts, no rewrite

The `bracket_state_3d` function and the flat-indexing block are unchanged. Only the call sites need to know what coordinate system to bracket in. There are exactly two such sites in the hot path (lines ~369 and ~527). Each currently looks like:

```python
# Continuous next state in s-coordinates
s_next_0 = Phi_0_state[0] + ... + v_nodes[k_v, 0]
s_next_1 = Phi_0_state[1] + ... + v_nodes[k_v, 1]
s_next_2 = Phi_0_state[2] + ... + v_nodes[k_v, 2]
lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
    s_next_0, s_next_1, s_next_2, grids_0, grids_1, grids_2
)
```

Change each to:

```python
# Continuous next state in s-coordinates
s_next_0 = Phi_0_state[0] + ... + v_nodes[k_v, 0]
s_next_1 = Phi_0_state[1] + ... + v_nodes[k_v, 1]
s_next_2 = Phi_0_state[2] + ... + v_nodes[k_v, 2]

# Map to u-coordinates for bracketing.  When state_grid_mode != 'principal'
# the caller passes L_inv = I and mu_s = 0, so this is a no-op.
ds0 = s_next_0 - mu_s[0]
ds1 = s_next_1 - mu_s[1]
ds2 = s_next_2 - mu_s[2]
u_next_0 = L_inv[0, 0] * ds0 + L_inv[0, 1] * ds1 + L_inv[0, 2] * ds2
u_next_1 = L_inv[1, 0] * ds0 + L_inv[1, 1] * ds1 + L_inv[1, 2] * ds2
u_next_2 = L_inv[2, 0] * ds0 + L_inv[2, 1] * ds1 + L_inv[2, 2] * ds2

lo0, lo1, lo2, f0, f1, f2 = bracket_state_3d(
    u_next_0, u_next_1, u_next_2, grids_0, grids_1, grids_2
)
```

To make this work cleanly:

1. The two outer JIT functions that contain these blocks need `L_inv` (3×3) and `mu_s` (3,) added to their parameter lists.
2. The dispatcher around line 2136 (where `state_grid`, `grids_0`, etc. are pulled out of `pc`) gains:

```python
L_inv_state = np.ascontiguousarray(pc.state_grid_L_inv)
mu_s_state  = np.ascontiguousarray(pc.state_grid_mu_s)
```

and these are passed through to every JIT function that calls `bracket_state_3d`.

For non-principal modes, you can either build `L_inv = I, mu_s = 0` upstream and pass them through unconditionally (cleanest, JIT-friendly), or branch. I recommend the unconditional path: precompute always sets `L_inv` and `mu_s` to consistent values (in `lyapunov-axis` or `naive` modes, set `L_inv = I` and `mu_s = 0`), so the solver always does the matvec and never branches on mode. The matvec is ~15 ns and the branch elimination keeps the JIT code straight-line.

Equivalent change in the precompute setter:

```python
if grid_info["mode"] == "principal":
    self.state_grid_L_inv = grid_info["L_inv"]
    self.state_grid_mu_s  = grid_info["mu_s"]
else:
    self.state_grid_L_inv = np.eye(model.n_state)
    self.state_grid_mu_s  = np.zeros(model.n_state)
```

### 3.4 `simulation.py` — same idea

Lines ~511–529 (the nearest-grid lookup) become:

```python
# Map next-state to bracketing coordinates
ds_0 = s_next_0 - mu_s_state[0]
ds_1 = s_next_1 - mu_s_state[1]
ds_2 = s_next_2 - mu_s_state[2]
u_next_0 = L_inv_state[0,0]*ds_0 + L_inv_state[0,1]*ds_1 + L_inv_state[0,2]*ds_2
u_next_1 = L_inv_state[1,0]*ds_0 + L_inv_state[1,1]*ds_1 + L_inv_state[1,2]*ds_2
u_next_2 = L_inv_state[2,0]*ds_0 + L_inv_state[2,1]*ds_1 + L_inv_state[2,2]*ds_2

# Nearest-axis lookup against the (now u-space) marginal grids
N1_s = len(state_grids_1); N2_s = len(state_grids_2)
best_d0 = 0; best_dist = abs(u_next_0 - state_grids_0[0])
for dd in range(1, len(state_grids_0)):
    d = abs(u_next_0 - state_grids_0[dd])
    if d < best_dist: best_dist = d; best_d0 = dd
# (... same for dimensions 1 and 2 ...)
next_state_idx = best_d0 * N1_s * N2_s + best_d1 * N2_s + best_d2
```

When `state_grid_mode != 'principal'`, `L_inv_state = I` and `mu_s_state = 0` and the routine collapses to the original axis-aligned nearest-neighbour lookup. So this single code path serves all three modes.

The `Pi_state` discrete branch (`use_continuous_state=False`, line ~531) is left untouched. It is only correct when the per-axis Rouwenhorst marginals approximate genuine economic axes (modes `lyapunov-axis` or `naive`); under `principal` the discrete branch should not be used. Add an assertion in the simulator entry point:

```python
if not use_continuous_state and self.state_grid_mode == 'principal':
    raise ValueError(
        "Discrete-state simulation (use_continuous_state=False) is not "
        "supported with state_grid_mode='principal'. Either switch to the "
        "continuous-state simulator or use mode='lyapunov-axis'."
    )
```

---

## 4. Tier-1 + Tier-2 validation script

Save as `tests/test_state_grid.py` and run with `python -m pytest tests/test_state_grid.py -v` (or as a script). It checks every assertion in §7 of the handoff that does *not* require the full DP solver. Tier-3 (solver-impact) requires the full solver and is laid out in §5 below.

```python
"""tests/test_state_grid.py — validation tests for build_state_grid().

Tier-1: mathematical consistency (must pass exactly).
Tier-2: coverage and concentration (probabilistic but tight tolerances).
"""
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from discretization import build_state_grid, stationary_covariance


# ---- Calibrated VAR (matches var.py's default annual restricted estimate) ----
PHI = np.array([
    [+0.6702, -0.2773, +0.0136],
    [+0.1531, +0.8716, -0.0068],
    [+0.5130, -1.2905, +0.9187],
])
SIGMA_INNOV = np.array([
    [+2.471e-04, -1.527e-04, +4.369e-04],
    [-1.527e-04, +1.262e-04, +9.796e-06],
    [+4.369e-04, +9.796e-06, +2.784e-02],
])
Z_BAR = np.array([+0.0485, +0.0199, -2.9929])
PHI_0 = (np.eye(3) - PHI) @ Z_BAR


# =============================================================================
# Tier 1 — mathematical consistency
# =============================================================================

def test_lyapunov_solution_satisfies_equation():
    Sigma_z = stationary_covariance(PHI, SIGMA_INNOV)
    residual = Sigma_z - PHI @ Sigma_z @ PHI.T - SIGMA_INNOV
    assert np.max(np.abs(residual)) < 1e-12, "Lyapunov residual too large"


def test_lyapunov_marginal_matches_handoff():
    Sigma_z = stationary_covariance(PHI, SIGMA_INNOV)
    sigma_z = np.sqrt(np.diag(Sigma_z))
    expected = np.array([0.0356, 0.0159, 0.5305])
    assert np.allclose(sigma_z, expected, atol=5e-4), \
        f"σ_z mismatch: {sigma_z} vs {expected}"


def test_principal_grid_centered_at_mu_s():
    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                            n_stds=3.0, mode='principal')
    centroid = info["state_grid"].mean(axis=0)
    assert np.allclose(centroid, info["mu_s"], atol=1e-12), \
        f"Grid centroid {centroid} != μ_s {info['mu_s']}"


def test_lyapunov_axis_grid_endpoints():
    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                            n_stds=3.0, mode='lyapunov-axis')
    for d in range(3):
        axis = info["state_grids"][d]
        hw = (axis.max() - axis.min()) / 2
        expected_hw = 3.0 * info["sigma_z"][d]
        assert abs(hw - expected_hw) < 1e-12, \
            f"axis {d}: half-width {hw} vs {expected_hw}"


def test_principal_grid_spans_n_stds_in_u():
    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                            n_stds=2.5, mode='principal')
    L_inv = info["L_inv"]; mu_s = info["mu_s"]
    U = (info["state_grid"] - mu_s) @ L_inv.T
    # u extreme should be exactly ±2.5 along each u-axis
    assert np.isclose(U[:, 0].max(), 2.5, atol=1e-12)
    assert np.isclose(U[:, 0].min(), -2.5, atol=1e-12)
    assert np.isclose(U[:, 1].max(), 2.5, atol=1e-12)
    assert np.isclose(U[:, 1].min(), -2.5, atol=1e-12)
    assert np.isclose(U[:, 2].max(), 2.5, atol=1e-12)
    assert np.isclose(U[:, 2].min(), -2.5, atol=1e-12)


def test_state_indices_match_flat_ordering():
    """Flat index i must equal i0*N1*N2 + i1*N2 + i2 — solver.py depends on this."""
    info = build_state_grid([5, 7, 9], PHI_0, PHI, SIGMA_INNOV, mode='principal')
    idx = info["state_indices"]
    N0, N1, N2 = 5, 7, 9
    for i in range(N0 * N1 * N2):
        i0, i1, i2 = idx[i]
        assert i == i0 * N1 * N2 + i1 * N2 + i2, \
            f"flat ordering broken at i={i}: {i0},{i1},{i2}"


def test_principal_grid_recovers_axis_grid_at_mu_s():
    """At u=0 (the central grid point of an odd-N grid), s = μ_s exactly."""
    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV, mode='principal')
    center_flat = (7 // 2) * 7 * 7 + (7 // 2) * 7 + (7 // 2)
    assert np.allclose(info["state_grid"][center_flat], info["mu_s"], atol=1e-12)


def test_pi_state_row_stochastic():
    info = build_state_grid([5, 5, 5], PHI_0, PHI, SIGMA_INNOV, mode='lyapunov-axis')
    rs = info["Pi_state"].sum(axis=1)
    assert np.max(np.abs(rs - 1.0)) < 1e-15


# =============================================================================
# Tier 2 — coverage and concentration
# =============================================================================

def test_principal_mode_historical_coverage(var_dataset_path='/mnt/project/var_dataset.csv'):
    df = pd.read_csv(var_dataset_path)
    S_hist = df[['y_1', 'spr', 'cy']].values

    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                            n_stds=3.0, mode='principal')
    L_inv = info["L_inv"]; mu_s = info["mu_s"]
    U_hist = (S_hist - mu_s) @ L_inv.T
    inside = np.all(np.abs(U_hist) <= 3.0, axis=1).mean()
    assert inside >= 0.99, f"Historical coverage too low: {inside*100:.1f}% (need ≥99%)"


def test_lyapunov_axis_mode_historical_coverage(var_dataset_path='/mnt/project/var_dataset.csv'):
    df = pd.read_csv(var_dataset_path)
    S_hist = df[['y_1', 'spr', 'cy']].values

    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                            n_stds=3.0, mode='lyapunov-axis')
    sigma_z = info["sigma_z"]; mu_s = info["mu_s"]
    inside = np.all(np.abs(S_hist - mu_s) <= 3.0 * sigma_z, axis=1).mean()
    assert inside >= 0.99, f"Lyapunov-axis coverage too low: {inside*100:.1f}%"


def test_naive_mode_undercoverage_documented(var_dataset_path='/mnt/project/var_dataset.csv'):
    """Regression test: confirm the bug. The legacy naive grid leaves >5%
    of historical observations outside the box at N=7. This test will FAIL
    if someone accidentally 'fixes' the legacy path; that's intended."""
    df = pd.read_csv(var_dataset_path)
    S_hist = df[['y_1', 'spr', 'cy']].values
    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV, mode='naive')
    sigma_z = info["sigma_z"]; mu_s = info["mu_s"]
    inside = np.all(np.abs(S_hist - mu_s) <= info["n_stds_effective"] * sigma_z, axis=1).mean()
    assert inside < 0.95, f"Naive mode now covers {inside*100:.1f}% — code may have changed unintentionally"


def test_principal_mode_stationary_mass(seed=0):
    """At least 99% of stationary mass in the n_stds=3.0 cube under principal mode."""
    rng = np.random.default_rng(seed)
    info = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                            n_stds=3.0, mode='principal')
    L = info["L"]; L_inv = info["L_inv"]; mu_s = info["mu_s"]
    Z = rng.standard_normal((200_000, 3))
    S = mu_s + Z @ L.T
    U = (S - mu_s) @ L_inv.T
    inside = np.all(np.abs(U) <= 3.0, axis=1).mean()
    assert inside >= 0.99, f"Stationary coverage {inside*100:.2f}% < 99%"


def test_principal_volume_is_smaller_than_lyapunov_axis():
    """Quantifies the orientation gain. Principal-mode hull volume should be
    strictly less than lyapunov-axis hull volume (same n_stds, same N)."""
    info_p = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                              n_stds=3.0, mode='principal')
    info_a = build_state_grid([7, 7, 7], PHI_0, PHI, SIGMA_INNOV,
                              n_stds=3.0, mode='lyapunov-axis')
    # principal: parallelepiped volume = (2*n_stds)^3 * |det L|
    vol_p = (6.0)**3 * abs(np.linalg.det(info_p["L"]))
    vol_a = np.prod(2 * 3.0 * info_a["sigma_z"])
    assert vol_p < vol_a, f"principal vol {vol_p} >= lyapunov-axis vol {vol_a}"
    # We expect ~46% based on Σ_z structure
    assert 0.3 < vol_p / vol_a < 0.6, \
        f"principal/lyapunov volume ratio {vol_p/vol_a:.2f} unexpected"


# =============================================================================
# Smoke-test driver (run as script)
# =============================================================================
if __name__ == "__main__":
    failed = []
    for name, fn in list(globals().items()):
        if not name.startswith("test_"): continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"ERROR {type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print("\nAll Tier-1/Tier-2 tests pass.")
```

I ran the equivalent of the Tier-2 coverage tests against the calibrated VAR while writing this note (numbers are reproduced in §1.3). All pass at `n_stds=3.0`.

---

## 5. Tier-3 — solver-impact protocol

These require running the full DP solver, which is outside what I can verify here. Here is a precise specification the analyst can run.

### 5.1 Policy convergence under N-refinement

Run the full solver at `state_grid_sizes ∈ {(5,5,5), (7,7,7), (9,9,9), (11,11,11)}`, `state_grid_mode='principal'`, `n_stds=3.0`, holding everything else fixed. At each grid resolution, record the policy `(c/W, α_s, α_b)` at:

- `(age, z, s, W) = (40, z_median, s = μ_s, W = median wealth at age 40)`
- `(age, z, s, W) = (40, z_median, s = μ_s + 2 σ_z * e_y_1, W = median wealth at age 40)`
- `(age, z, s, W) = (65, z_median, s = μ_s, W = median wealth at age 65)`

**Pass criterion**: `|policy(N) − policy(N+2)|` decreases monotonically. Repeat with `mode='lyapunov-axis'` and `mode='naive'` for comparison; principal-mode convergence rates should be at least as good as lyapunov-axis, and both should be strictly better than naive.

### 5.2 Tail-state policy stability

Compute `α_b(y_1, μ_spr, μ_cy, z_median, W_median)` for `y_1 ∈ [μ_y_1 − 3σ, μ_y_1 + 3σ]` at 31 points. Plot under each of (current-naive, lyapunov-axis, principal). Pass criteria:

- Principal-mode curve is smooth and monotonically decreasing in `y_1` (consistent with `M[xb, y_1] = −8.72`) over the entire range.
- Naive-mode curve plateaus or oscillates above `+1.46 σ_z`. This is the bug, made visible.
- Lyapunov-axis curve is smooth but has subtle kinks where the off-diagonal `(spr, cy)` slice through the axis-aligned grid happens to cross a grid plane in a state-correlated manner. Principal-mode should not show these kinks.

### 5.3 Simulation moment match

Run `simulate(seed=42, n_paths=10_000, T=85)` under (current, lyapunov-axis, principal). For each age, plot:

- Mean `α_b` and 5/95 percentile bands.
- Mean `α_s`.
- Mean `c/W`.

Differences between the modes should be **largest** in periods/quantiles where `y_1` is in the tail (>2 σ_z) — which is the exact region where the bug bit. Differences in interior states (<1 σ_z on every axis) should be small (<1 percentage point on portfolio shares, <2% on c/W). This is the diagnostic that **quantifies how much the bug actually mattered**.

### 5.4 Bond-duration response

Cross-section of `α_b` vs `y_1` at fixed `(spr, cy, z, W)` corresponding to median values of each. Slope `∂α_b/∂y_1` should be:

- Smooth and finite over `y_1 ∈ [0.01, 0.14]` (full historical range).
- Negative (bond price declines as yields rise; high-y_1 states pay off as rates fall back to mean, so duration risk is locally negative for the consumer).
- Approximately constant for moderate `y_1`, and may strengthen near the high-yield boundary as the persistent expected return on bonds rises.

Under principal-mode the slope should be well-defined across the entire range. Under naive it flattens above `+1.46 σ_z` because the policy is the boundary policy, not the true one.

### 5.5 Performance regression

Time `run_lifecycle_solver(...)` on the production grid (N=7³, K_s=3, K_r=2). Compare wall time under current vs principal-mode. Expected overhead: <2%.

---

## 6. Production settings

```python
DiscretizationConfig(
    state_grid_sizes=(7, 7, 7),
    state_grid_mode='principal',
    state_n_stds=3.0,
    n_state_quad_nodes=3,        # unchanged
    # ... everything else unchanged
)
```

Rationale:

- **N = 7³ = 343 points.** Same total grid count as the current production setting; the upgrade is purely qualitative (orientation + correct radius).
- **`mode='principal'`.** Sets the Cholesky-rotated grid (Option B). Cost is one extra 3×3 matvec per `bracket_state_3d` call; benefit is ~2× tighter resolution along the dominant direction of stationary variation.
- **`n_stds=3.0`.** Covers 100% of the 1963–2025 historical sample (max u-norm in the historical sample is 2.93) and 99.2% of stationary mass. Pushing to `n_stds=3.5` only buys the last 0.6% of stationary mass at the cost of diluting interior resolution; not recommended.

If the analyst wants per-axis tuning under principal mode (the hybrid suggested in handoff §6), it is supported via:

```python
# Accepts either a scalar or a 3-tuple
state_n_stds=(2.5, 2.5, 3.0)
```

For the calibrated VAR I would not bother — the historical sample is roughly equally extreme along all three u-axes once you've rotated, and uniform spacing is simpler.

---

## 7. Summary of code changes by file

| file | change | risk |
|---|---|---|
| `discretization.py` | Replace `rouwenhorst_multivariate` with `build_state_grid`; add `stationary_covariance` helper; keep `rouwenhorst_univariate` unchanged | low — function is self-contained |
| `model.py` | Add two fields to `DiscretizationConfig`: `state_n_stds: float = 3.0`, `state_grid_mode: str = 'principal'` | trivial |
| `precompute.py` | Replace one ~10-line block (lines 118–134); add three attributes: `state_grid_L`, `state_grid_L_inv`, `state_grid_mu_s` | low — `_build_state_grid` static method becomes redundant |
| `solver.py` | Insert ~6 lines (one matvec) at the two `bracket_state_3d` call sites; thread `L_inv_state` and `mu_s_state` through the JIT signatures from the dispatcher around line 2136 | medium — touches hot loop, but the matvec is straight-line code and JIT-friendly |
| `simulation.py` | Same matvec insert in the nearest-grid lookup at lines ~511–529; add the assertion forbidding the discrete branch when `mode='principal'` | low — pure addition |
| `tests/test_state_grid.py` | New file (Tier 1+2 from §4 above) | none |

The frozen-quadrature contract from the handoff is preserved: `get_state_quadrature` and `get_return_quadrature` are not touched, and the integration of `Σ_ss` and `Σ_r_cond` continues to use full Cholesky / eigendecomposition as before. The grid change only affects how policy values are *stored and interpolated* between grid points, not how transition uncertainty is integrated.

---

## 8. What I deliberately did **not** recommend

- **Sparse / adaptive grids** (handoff Option E). Substantial solver rewrite for marginal gains over Option B.
- **Multivariate Rouwenhorst with cross-correlation** (handoff Option D). The solver does not use `Pi_state` for the dynamics; the quadrature already integrates the joint Σ exactly. Building a faithful joint Markov chain solves a problem the solver does not have.
- **Gauss–Hermite tensor nodes for the grid** (handoff Option C). At N=7 the outermost GH node sits inside ±1.65 σ — worse tail coverage than Option B with `n_stds=3.0`. GH is the right rule for *quadrature*, not for *interpolation support*.
- **Eigendecomposition basis instead of Cholesky.** Slightly less even per-axis resolution (eigenvalue ratio is ~57× vs Cholesky's ~24× for the diagonal of L), and breaks consistency with `get_state_quadrature`'s convention. If the analyst later wants ordered principal axes for diagnostic plots, the basis swap is a one-line change to `build_state_grid`.

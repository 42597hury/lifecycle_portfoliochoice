# HANDOFF — Farmer-Toda Discretization: Implementation Guide

**To:** coding agent  
**From:** design + validation review  
**Date:** April 2026  
**Status:** Math validated, prototype tested. Ready for production implementation.

---

## 0. Scope & Prerequisites

### What this document covers

Replace the independence-Rouwenhorst discretization of the 3-variable financial
state VAR(1) with the Farmer-Toda (2017) maximum-entropy moment-matching method.
The old method stays accessible as `method="independent"` for comparison runs.

### What you need to do

1. Modify `discretization.py` — add Farmer-Toda functions, extend `rouwenhorst_multivariate`
2. Modify `model.py` — add two fields to `DiscretizationConfig`
3. Modify `precompute.py` — change the call site (3 lines)
4. Create `test_discretization.py` — comprehensive validation test module
5. Validate that the provided implementation code is correct by running the tests

### Files to read before starting

| File | Lines | What's there |
|------|-------|-------------|
| `discretization.py` | 1–121 | Existing code you'll extend |
| `precompute.py` | 113–170 | Call site + downstream precomputations |
| `model.py` | 92–117 | `DiscretizationConfig` you'll extend |
| `var.py` | 765–788 | Hardcoded annual VAR parameters (test reference) |

### Dependencies

The new code requires `scipy.linalg.solve_discrete_lyapunov`. This is already
available since `scipy` is a project dependency — just add the import.

### Files with ZERO changes

| File | Why it's unaffected |
|------|-------------------|
| `solver.py` | Consumes `Pi_state`, `mu_r`, `annuity_factors` — same arrays, same shapes |
| `simulation.py` | Draws `next_state = draw_discrete(Pi_state[i,:], u)` — same interface |
| `diagnostics.py` | Reads `Pi_state`, `state_grid` — same interface |
| `plots.py` | Does not reference discretization outputs |
| `mortality.py` | Independent subsystem (income process, not financial state) |
| `var.py` | Produces VAR parameters consumed by discretization — no change |

---

## 1. Code Changes by File

### 1.1 `discretization.py` — The Main Change

**Structure of the change:**

```
discretization.py (current):
    imports
    mixture_cdf, mixture_quantile          (keep)
    rouwenhorst_univariate                 (keep)
    rouwenhorst_multivariate               (EXTEND with new branch)
    discretize_income_ar1_mixture          (keep)
    get_eps_quadrature_corrected           (keep)
    get_eta_quadrature_mixture             (keep)
    get_return_quadrature                  (keep)

discretization.py (after):
    imports                                (ADD: from scipy.linalg import solve_discrete_lyapunov)
    mixture_cdf, mixture_quantile          (keep)
    rouwenhorst_univariate                 (keep)
    _farmer_toda_dual_solve                (NEW — internal Newton solver)
    _farmer_toda_mean_only_fallback        (NEW — fallback for infeasible states)
    _farmer_toda_discretize                (NEW — main Farmer-Toda function)
    rouwenhorst_multivariate               (MODIFIED — add elif method=="farmer_toda" + grid_scale param)
    discretize_income_ar1_mixture          (keep)
    ...rest unchanged...
```

**Import to add** (line 16, after existing scipy imports):

```python
from scipy.linalg import solve_discrete_lyapunov
```

**New functions to add** between `rouwenhorst_univariate` (ends line 71) and
`rouwenhorst_multivariate` (starts line 74). The full implementation code is
in §2 below.

**Modification to `rouwenhorst_multivariate`:**

Replace the current signature and body (lines 74–121) with the version in §2
that adds `grid_scale` parameter and the `elif method == "farmer_toda"` branch.
The `method="independent"` branch is kept verbatim.

### 1.2 `model.py` — Two new fields on `DiscretizationConfig`

Add these two fields inside `DiscretizationConfig` (after line 105):

```python
class DiscretizationConfig(NamedTuple):
    # ... existing fields ...

    # Financial state VAR discretization
    state_grid_sizes: tuple = (5, 5, 5)
    state_discretization_method: str = "farmer_toda"   # NEW: "independent" or "farmer_toda"
    state_grid_scale: float = 1.0                      # NEW: grid coverage scaling; ~0.85 for N=5 debug

    # ... rest unchanged ...
```

### 1.3 `precompute.py` — Call site change (lines 118–125)

Replace:

```python
Sigma_state_chol = np.linalg.cholesky(model.Sigma_ss)
self.state_grids, self.Pi_state, self.state_indices = rouwenhorst_multivariate(
    N_vec=state_grid_sizes,
    mu=model.Phi_0_state,
    Phi=model.Phi_11,
    Sigma=Sigma_state_chol,
    method="independent",
)
```

With:

```python
Sigma_state_chol = np.linalg.cholesky(model.Sigma_ss)
self.state_grids, self.Pi_state, self.state_indices = rouwenhorst_multivariate(
    N_vec=state_grid_sizes,
    mu=model.Phi_0_state,
    Phi=model.Phi_11,
    Sigma=Sigma_state_chol,
    method=disc_config.state_discretization_method,
    grid_scale=disc_config.state_grid_scale,
)
```

### 1.4 `precompute.py` — Validation tolerance update

The existing `_validate_conditional_returns` check (called at line 250) uses
tolerances from `disc_config`. With Farmer-Toda, the conditional mean error
drops from ~0.12 to ~1e-9. You should NOT change the default tolerances in
`DiscretizationConfig` yet — the current loose defaults (warn=2e-2, error=1e-1)
will simply pass silently, which is correct. Tightening them is a future
polish step after the full integration is verified.

---

## 2. Full Implementation Code

The following is the complete code to add to `discretization.py`. It has been
prototype-tested against all 343 source states at N=7 and verified to produce
machine-precision moment matching at feasible states.

**Your task:** Copy this code into `discretization.py` at the locations
specified in §1.1, then run the test module (§4) to validate it end-to-end.
If any test fails, investigate and fix — do not assume the code below is
bug-free just because the prototype passed.

### 2.1 Three new internal functions

Insert these between `rouwenhorst_univariate` (line 71) and
`rouwenhorst_multivariate` (line 74):

```python
# =========================================================================
# FARMER-TODA (2017) MAXIMUM-ENTROPY DISCRETIZATION
# =========================================================================

def _farmer_toda_dual_solve(F, target, lam_init=None,
                            max_iter=200, tol=1e-12):
    """
    Newton solver for the Farmer-Toda maximum-entropy dual problem.

    Minimises D(λ) = log Σ_j exp(F[j,:] · λ) − λ · target,
    which is strictly convex. The optimal primal probabilities are
    p_j* = softmax(F @ λ*).

    Parameters
    ----------
    F : ndarray, shape (N, m)
        Feature matrix.  Rows = grid points, columns = moment features.
    target : ndarray, shape (m,)
        Target moments (centered mean = 0, upper-triangle of Σ_ss).
    lam_init : ndarray, shape (m,) or None
        Initial dual variables.  None → zeros (uniform prior).
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on ‖∇D‖_∞.

    Returns
    -------
    p : ndarray, shape (N,)
        Optimal probability vector.
    converged : bool
        True if ‖∇D‖_∞ < tol within max_iter iterations.
    n_iter : int
        Number of Newton iterations taken.
    grad_norm : float
        Final gradient infinity norm.
    lam : ndarray, shape (m,)
        Final dual variables (useful for warm-starting).
    """
    N, m = F.shape
    lam = lam_init.copy() if lam_init is not None else np.zeros(m)

    for it in range(max_iter):
        # --- Log-sum-exp stabilised probability computation ---
        fj = F @ lam                           # (N,)
        fj_max = fj.max()
        exp_fj = np.exp(fj - fj_max)           # stabilised
        Z = exp_fj.sum()
        p = exp_fj / Z                         # (N,) probabilities

        # --- Gradient: E_p[F] − target ---
        Ep_F = p @ F                           # (m,)
        grad = Ep_F - target                   # (m,)
        grad_norm = np.max(np.abs(grad))

        if grad_norm < tol:
            return p, True, it, grad_norm, lam

        # --- Hessian: Cov_p(F) ---
        sqrt_p = np.sqrt(p)
        Fp = F * sqrt_p[:, None]               # (N, m)
        H = Fp.T @ Fp - np.outer(Ep_F, Ep_F)  # (m, m) PSD

        # Regularise if near-singular
        eigvals_H = np.linalg.eigvalsh(H)
        min_eig = eigvals_H.min()
        if min_eig < 1e-12:
            H += max(1e-10, -min_eig + 1e-10) * np.eye(m)

        # --- Newton step ---
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]

        # --- Armijo backtracking line search ---
        obj_old = fj_max + np.log(Z) - lam @ target
        directional = grad @ delta             # descent along -delta
        alpha = 1.0
        for _ in range(40):
            lam_try = lam - alpha * delta
            fj_try = F @ lam_try
            fj_max_try = fj_try.max()
            obj_try = (fj_max_try
                       + np.log(np.exp(fj_try - fj_max_try).sum())
                       - lam_try @ target)
            if obj_try < obj_old - 1e-4 * alpha * directional:
                break
            alpha *= 0.5

        lam = lam - alpha * delta

    # Did not converge — return best probabilities so far
    fj = F @ lam
    fj_max = fj.max()
    exp_fj = np.exp(fj - fj_max)
    p = exp_fj / exp_fj.sum()
    return p, False, max_iter, grad_norm, lam


def _farmer_toda_mean_only_fallback(state_grid, mu_i, N_total, d):
    """
    Fallback: maximum-entropy with mean-matching only (no covariance).

    Always feasible when mu_i is inside the convex hull of state_grid
    (verified for all grid sizes >= 5 with Lyapunov-based widths).
    """
    Y = state_grid - mu_i
    F_reduced = Y                              # (N_total, d) — mean features only
    target_reduced = np.zeros(d)               # centered mean = 0

    p, converged, _, _, _ = _farmer_toda_dual_solve(
        F_reduced, target_reduced, max_iter=200, tol=1e-12
    )
    return p, converged


def _farmer_toda_discretize(N_vec, mu, Phi, Omega, grid_scale=1.0):
    """
    Farmer-Toda (2017) maximum-entropy moment-matching discretisation
    for a Gaussian VAR(1):  z_{t+1} = mu + Phi · z_t + v_t,  v_t ~ N(0, Omega).

    Parameters
    ----------
    N_vec : array-like of int, shape (d,)
        Grid sizes per state dimension.
    mu : ndarray, shape (d,)
        Intercept (Phi_0_state).
    Phi : ndarray, shape (d, d)
        Persistence matrix (Phi_11).
    Omega : ndarray, shape (d, d)
        Innovation covariance (Sigma_ss = Omega, NOT the Cholesky factor).
    grid_scale : float, optional
        Scaling factor for grid half-width.  Default 1.0 gives the standard
        +/-sqrt(N-1) * sigma_uncond coverage.  Use ~0.85 for small grids (N<=5)
        where the standard width makes the grid spacing too wide relative to
        the innovation std dev, causing covariance-matching infeasibility.

    Returns
    -------
    grids : list of d ndarrays
        Marginal grids per dimension.
    Pi_state : ndarray, shape (N_total, N_total)
        Transition matrix with exact conditional moments (C1, C2).
    state_indices : ndarray, shape (N_total, d), dtype int
        Multi-index into marginal grids.
    """
    N_vec = np.asarray(N_vec, dtype=int)
    d = len(N_vec)
    N_total = int(np.prod(N_vec))

    # ---- Step 1: Unconditional moments ----
    mu_bar = np.linalg.solve(np.eye(d) - Phi, mu)         # (d,) stationary mean
    Sigma_z = solve_discrete_lyapunov(Phi, Omega)          # (d,d) stationary cov

    # ---- Step 2: Lyapunov-based grid construction ----
    grids = []
    for i in range(d):
        sigma_uncond_i = np.sqrt(Sigma_z[i, i])
        psi_i = grid_scale * sigma_uncond_i * np.sqrt(N_vec[i] - 1)
        g_i = np.linspace(mu_bar[i] - psi_i, mu_bar[i] + psi_i,
                          int(N_vec[i]))
        grids.append(g_i)

    # ---- Step 3: Cartesian product grid ----
    state_indices = np.zeros((N_total, d), dtype=np.int64)
    for idx, multi_idx in enumerate(np.ndindex(*N_vec.tolist())):
        state_indices[idx, :] = np.array(multi_idx, dtype=np.int64)

    state_grid = np.empty((N_total, d), dtype=float)
    for i in range(N_total):
        for dd in range(d):
            state_grid[i, dd] = grids[dd][state_indices[i, dd]]

    # ---- Step 4: Moment target vector ----
    # Convention A: upper triangle, no doubling of off-diagonals.
    #   Features: [y_0, y_1, y_2, y_0^2, y_0*y_1, y_0*y_2, y_1^2, y_1*y_2, y_2^2]
    #   Target:   [0, 0, 0, Omega[0,0], Omega[0,1], Omega[0,2], Omega[1,1], Omega[1,2], Omega[2,2]]
    m = d + d * (d + 1) // 2                               # 3 + 6 = 9
    target = np.zeros(m)
    col = d
    for a in range(d):
        for b in range(a, d):
            target[col] = Omega[a, b]
            col += 1

    # ---- Step 5: Solve dual for each source state ----
    Pi_state = np.zeros((N_total, N_total), dtype=float)

    # Warm-start: process states nearest to center first
    dists = np.linalg.norm(state_grid - mu_bar, axis=1)
    order = np.argsort(dists)

    lam_warm = None
    n_fallback = 0

    # The Newton solver uses tol=1e-12 for strict convergence.
    # For the fallback decision, we use a practical threshold: states
    # with grad_norm < 1e-8 have excellent moment matching and should
    # NOT trigger the mean-only fallback.
    fallback_threshold = 1e-8

    for rank, i_src in enumerate(order):
        s_i = state_grid[i_src]
        mu_i = mu + Phi @ s_i                              # (d,) conditional mean

        # Centered destinations
        Y = state_grid - mu_i                              # (N_total, d)

        # Feature matrix
        F = np.zeros((N_total, m))
        F[:, 0:d] = Y
        col = d
        for a in range(d):
            for b in range(a, d):
                F[:, col] = Y[:, a] * Y[:, b]
                col += 1

        # Solve with warm start
        p, converged, n_it, grad_norm, lam_out = _farmer_toda_dual_solve(
            F, target, lam_init=lam_warm, max_iter=200, tol=1e-12
        )

        if not converged and grad_norm >= fallback_threshold:
            # Retry from cold start only if truly non-converged
            p2, converged2, _, gn2, lam2 = _farmer_toda_dual_solve(
                F, target, lam_init=None, max_iter=200, tol=1e-12
            )
            if gn2 < grad_norm:
                p, converged, grad_norm, lam_out = p2, converged2, gn2, lam2

        # Decide: effectively converged (grad < threshold) or fallback
        effectively_converged = converged or (grad_norm < fallback_threshold)

        if not effectively_converged:
            # Genuinely infeasible or very poorly converged — use fallback
            p_fb, conv_fb = _farmer_toda_mean_only_fallback(
                state_grid, mu_i, N_total, d
            )
            if conv_fb:
                p = p_fb
            n_fallback += 1
            warnings.warn(
                f"Farmer-Toda: state {i_src} (s={s_i}) did not converge "
                f"(grad={grad_norm:.2e}). Using mean-only fallback.",
                RuntimeWarning, stacklevel=2
            )
        else:
            # Successful — save dual for warm-starting neighbours
            lam_warm = lam_out.copy()

        # Clip tiny negatives from floating-point
        p = np.maximum(p, 0.0)
        p_sum = p.sum()
        if abs(p_sum - 1.0) > 1e-10:
            p /= p_sum

        Pi_state[i_src, :] = p

    if n_fallback > 0:
        warnings.warn(
            f"Farmer-Toda: {n_fallback}/{N_total} states used mean-only "
            f"fallback (covariance not matched at those states).",
            RuntimeWarning, stacklevel=2
        )

    return grids, Pi_state, state_indices
```

### 2.2 Modified `rouwenhorst_multivariate`

Replace the existing function (lines 74–121) with:

```python
def rouwenhorst_multivariate(N_vec, mu, Phi, Sigma, method="independent",
                             grid_scale=1.0):
    """
    Multivariate Markov chain approximation for a Gaussian VAR(1).

    Supports two methods:
      - "independent": Kronecker product of marginal Rouwenhorst chains.
        Fast but sets all cross-correlations to zero (incorrect for
        off-diagonal Phi).
      - "farmer_toda": Maximum-entropy moment-matching (Farmer & Toda 2017).
        Matches conditional mean and covariance exactly. Uses Lyapunov-
        equation unconditional std devs for correct grid width.

    Parameters
    ----------
    N_vec : list of int
        Grid sizes per state variable, e.g. [7, 7, 7].
    mu : ndarray, shape (d,)
        Intercept in z' = mu + Phi z + eps.
    Phi : ndarray, shape (d, d)
        Persistence matrix.
    Sigma : ndarray, shape (d, d)
        Cholesky factor of innovation covariance (Sigma @ Sigma.T = Omega).
    grid_scale : float, optional
        Grid coverage scaling (Farmer-Toda only). Default 1.0 gives standard
        +/-sqrt(N-1) * sigma_uncond. Use ~0.85 for N=5 debugging grids.

    Returns
    -------
    state_grids : list of d 1-D arrays
        Marginal grids per dimension.
    Pi_state : ndarray, shape (N_total, N_total)
        Transition matrix.
    state_indices : ndarray, shape (N_total, d), dtype int
        Multi-index into marginal grids.
    """
    N_vec = np.asarray(N_vec, dtype=int)
    k = len(N_vec)
    if Phi.shape != (k, k):
        raise ValueError(f"Phi must have shape {(k, k)}, got {Phi.shape}")
    if Sigma.shape != (k, k):
        raise ValueError(f"Sigma must have shape {(k, k)}, got {Sigma.shape}")

    if method == "independent":
        # ---- Existing independence Rouwenhorst (unchanged) ----
        mu_bar = np.linalg.solve(np.eye(k) - Phi, mu)
        Omega = Sigma @ Sigma.T

        grids = []
        marginals = []
        for i in range(k):
            rho_i = Phi[i, i]
            sigma_i = np.sqrt(max(1e-14, Omega[i, i]))
            g_i, Pi_i = rouwenhorst_univariate(int(N_vec[i]), mu_bar[i], rho_i, sigma_i)
            grids.append(g_i)
            marginals.append(Pi_i)

        n_total = int(np.prod(N_vec))
        state_indices = np.zeros((n_total, k), dtype=np.int64)
        for idx, multi_idx in enumerate(np.ndindex(*N_vec.tolist())):
            state_indices[idx, :] = np.array(multi_idx, dtype=np.int64)

        Pi_joint = np.ones((n_total, n_total), dtype=float)
        for dim in range(k):
            Pi_dim = marginals[dim]
            from_idx = state_indices[:, dim]
            to_idx = state_indices[:, dim]
            Pi_joint *= Pi_dim[np.ix_(from_idx, to_idx)]

        row_sums = Pi_joint.sum(axis=1, keepdims=True)
        Pi_joint = Pi_joint / np.maximum(row_sums, 1e-300)

        return grids, Pi_joint, state_indices

    elif method == "farmer_toda":
        # ---- Farmer-Toda maximum-entropy moment matching ----
        Omega = Sigma @ Sigma.T                # recover innovation covariance
        return _farmer_toda_discretize(N_vec, mu, Phi, Omega,
                                       grid_scale=grid_scale)

    else:
        raise NotImplementedError(
            f"Unknown method '{method}'. "
            f"Supported: 'independent', 'farmer_toda'.")
```

---

## 3. Numerical Reference Values

These are the VAR parameters used by the model. Use them in tests to construct
the discretization independently of `build_model()`.

```python
import numpy as np

Phi_11 = np.array([
    [ 0.01695,   0.77529,  -0.00264],
    [ 0.00928,   0.86113,   0.00079],
    [-1.19573,   7.57961,   0.80693]
])

Sigma_ss = np.array([
    [ 4.484e-5,   6.511e-7,  -1.424e-5],
    [ 6.511e-7,   7.949e-6,   1.342e-5],
    [-1.424e-5,   1.342e-5,   1.961e-2]
])

z_bar_state = np.array([-8.350e-4, 9.122e-3, -4.148])

# Derived quantities:
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar_state
# = [-0.01884, 0.00455, -0.87099]

Sigma_chol = np.linalg.cholesky(Sigma_ss)

# From Lyapunov equation:
from scipy.linalg import solve_discrete_lyapunov
Sigma_z = solve_discrete_lyapunov(Phi_11, Sigma_ss)
# Unconditional std devs:
#   rtb:   0.00805
#   y_nom: 0.00615
#   dp:    0.30524

# True unconditional correlations:
#   corr(rtb, y_nom) = 0.497
#   corr(rtb, dp)    = 0.246
#   corr(y_nom, dp)  = 0.533
```

### Expected outputs at N=7 (grid_scale=1.0)

| Metric | Expected |
|--------|---------|
| Fallback states | 2–3 out of 343 |
| Max conditional mean error (feasible) | < 1e-8 |
| Max conditional cov error (feasible, rel Frob) | < 1e-6 |
| Stationary mean error | < 1e-8 |
| Stationary cov error (rel Frob) | < 1e-4 |
| corr(rtb, y_nom) | 0.497 ± 0.001 |
| corr(rtb, dp) | 0.246 ± 0.001 |
| corr(y_nom, dp) | 0.533 ± 0.001 |

### Expected outputs at N=5 (grid_scale=0.85)

| Metric | Expected |
|--------|---------|
| Fallback states | 4–6 out of 125 |
| Max conditional mean error (feasible) | < 1e-8 |
| Stationary cov error (rel Frob) | < 0.01 |
| corr(rtb, y_nom) | 0.497 ± 0.005 |
| corr(rtb, dp) | 0.246 ± 0.005 |
| corr(y_nom, dp) | 0.533 ± 0.005 |

---

## 4. Test Specification

Create `test_discretization.py` in the project root. The test module should be
runnable standalone (`python test_discretization.py`) and report pass/fail for
each test.

The tests are organized in three tiers:

- **Tier 1 (standalone):** Test the discretization in isolation using the VAR
  parameters from §3. No need to build the full model.
- **Tier 2 (integration):** Build a full `Precompute` object and verify that
  the downstream arrays (`mu_r`, `annuity_factors`, `r_bill_grid`) are
  consistent with the new discretization.
- **Tier 3 (comparison):** Run both methods and verify the Farmer-Toda method
  is strictly better on all moment-matching metrics.

### Tier 1: Standalone Discretization Tests

Run for BOTH configurations: `(N=7, scale=1.0)` and `(N=5, scale=0.85)`.

**T1. Row stochasticity**

```python
row_sums = Pi_state.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-12), "Row sums != 1"
assert (Pi_state >= -1e-15).all(), "Negative probabilities"
```

Tolerance: machine precision. Must hold by construction.

**T2. Conditional mean (C1)**

For every source state `i`, verify:

```
E_Pi[s' | s_i] = Phi_0_state + Phi_11 @ s_i
```

```python
state_grid = build_flat_grid(grids, state_indices)  # (N_total, d)
for i in range(N_total):
    computed_mean = Pi_state[i, :] @ state_grid
    target_mean = Phi_0_state + Phi_11 @ state_grid[i]
    error = np.max(np.abs(computed_mean - target_mean))
max_error = max over all i
```

Tolerance at N=7: `max_error < 1e-6`.
Tolerance at N=5 (scale=0.85): `max_error < 1e-6`.

**T3. Conditional covariance (C2)**

For every source state `i`, verify:

```
Cov_Pi[s' | s_i] = Sigma_ss
```

```python
for i in range(N_total):
    mu_i = Phi_0_state + Phi_11 @ state_grid[i]
    Y = state_grid - mu_i
    cond_cov = (Pi_state[i, :, None] * Y).T @ Y
    rel_error = norm(cond_cov - Sigma_ss, 'fro') / norm(Sigma_ss, 'fro')
```

Report: number of states with `rel_error < 1e-6` and number with `rel_error > 0.1`.

Tolerance at N=7: at least 340/343 states have `rel_error < 1e-6`. At most 3
states (fallbacks) have `rel_error > 0.1`.

**T4. Stationary mean (U1)**

Compute the stationary distribution `pi*` (left eigenvector of `Pi_state` at
eigenvalue 1) and verify:

```python
eigvals, eigvecs = np.linalg.eig(Pi_state.T)
idx = np.argmin(np.abs(eigvals - 1.0))
pi_star = np.real(eigvecs[:, idx])
pi_star /= pi_star.sum()
stat_mean = pi_star @ state_grid
error = np.max(np.abs(stat_mean - z_bar_state))
```

Tolerance at N=7: `error < 1e-6`.
Tolerance at N=5: `error < 1e-4`.

**T5. Stationary covariance (U2)**

```python
stat_cov = sum(pi_star[j] * np.outer(state_grid[j] - stat_mean,
                                       state_grid[j] - stat_mean)
               for j in range(N_total))
Sigma_z = solve_discrete_lyapunov(Phi_11, Sigma_ss)
rel_error = norm(stat_cov - Sigma_z, 'fro') / norm(Sigma_z, 'fro')
```

Tolerance at N=7: `rel_error < 1e-3`.
Tolerance at N=5: `rel_error < 0.02`.

**T6. Stationary correlations**

Compute the discrete stationary correlation matrix and compare to the
Lyapunov-equation correlation matrix.

```python
D_stat = np.diag(1.0 / np.sqrt(np.diag(stat_cov)))
corr_stat = D_stat @ stat_cov @ D_stat
D_z = np.diag(1.0 / np.sqrt(np.diag(Sigma_z)))
corr_z = D_z @ Sigma_z @ D_z
for a in range(d):
    for b in range(a+1, d):
        assert abs(corr_stat[a,b] - corr_z[a,b]) < tol
```

Tolerance at N=7: `< 0.002` per entry.
Tolerance at N=5: `< 0.01` per entry.

**T7. Fallback count**

```python
# Count warnings emitted during construction
assert n_fallback <= 3   # at N=7
assert n_fallback <= 8   # at N=5, scale=0.85
```

### Tier 2: Integration Tests

These tests verify that the Farmer-Toda discretization integrates correctly
into the `Precompute` object and that downstream arrays are well-formed.

**T8. Precompute builds successfully**

```python
from precompute import Precompute, build_model

model = build_model(...)   # use standard config
disc_config = DiscretizationConfig(
    state_grid_sizes=(7, 7, 7),
    state_discretization_method="farmer_toda",
)
pc = Precompute(model, disc_config=disc_config)
# Should complete without errors
```

**T9. Conditional return consistency (the key integration test)**

This is the existing `_validate_conditional_returns` check, which verifies:

```
sum_j Pi_state[i,j] * mu_r[i,j,:] == Phi_0_ret + Phi_21 @ state_grid[i]
```

for all source states `i`. With the independence method, this had structural
error up to 0.12. With Farmer-Toda, it should be near zero because conditional
means are matched exactly.

```python
N = pc.N_state
errors = np.empty((N, model.n_ret))
for i in range(N):
    target = model.Phi_0_ret + model.Phi_21 @ pc.state_grid[i]
    avg = pc.Pi_state[i, :] @ pc.mu_r[i, :, :]
    errors[i, :] = np.abs(avg - target)
max_error = errors.max()
```

Tolerance: `max_error < 1e-4` (was ~0.12 with independence method).

This is the single most important integration test. It confirms that the
portfolio Euler equation is evaluated at correct expected returns.

**T10. Bill rate grid uses correct state variable**

```python
assert np.allclose(
    pc.r_bill_grid,
    pc.state_grid[:, model.bill_rate_index_in_state]
)
```

**T11. Annuity factors are well-formed**

```python
assert pc.annuity_factors.shape == (pc.N_state,)
assert np.all(pc.annuity_factors > 0)
assert np.all(np.isfinite(pc.annuity_factors))
```

**T12. mu_r shape and finiteness**

```python
assert pc.mu_r.shape == (pc.N_state, pc.N_state, model.n_ret)
assert np.all(np.isfinite(pc.mu_r))
```

### Tier 3: Comparison Tests

Run both methods on the same grid and verify Farmer-Toda dominates.

**T13. Cross-correlations: independence = 0, Farmer-Toda ≈ true**

```python
# Build both
grids_ft, Pi_ft, si_ft = rouwenhorst_multivariate(
    N_vec, Phi_0, Phi_11, Sigma_chol, method="farmer_toda")
grids_ind, Pi_ind, si_ind = rouwenhorst_multivariate(
    N_vec, Phi_0, Phi_11, Sigma_chol, method="independent")

# Compute stationary correlations for each
# ...

# Independence should have near-zero cross-correlations
assert abs(corr_ind[0,1]) < 0.05  # rtb,y_nom: true is 0.50
assert abs(corr_ind[0,2]) < 0.05  # rtb,dp: true is 0.25
assert abs(corr_ind[1,2]) < 0.05  # y_nom,dp: true is 0.53

# Farmer-Toda should match the true values
assert abs(corr_ft[0,1] - 0.497) < 0.01
assert abs(corr_ft[0,2] - 0.246) < 0.01
assert abs(corr_ft[1,2] - 0.533) < 0.01
```

**T14. Conditional mean error: Farmer-Toda orders of magnitude better**

```python
# Independence max conditional mean error
max_err_ind = max(
    np.max(np.abs(Pi_ind[i] @ sg_ind - (Phi_0 + Phi_11 @ sg_ind[i])))
    for i in range(N))

# Farmer-Toda max conditional mean error
max_err_ft = max(
    np.max(np.abs(Pi_ft[i] @ sg_ft - (Phi_0 + Phi_11 @ sg_ft[i])))
    for i in range(N))

assert max_err_ft < max_err_ind * 1e-4   # at least 10,000x improvement
```

---

## 5. How to Run

After implementing the changes:

```bash
# Run the standalone + comparison tests (Tier 1 + Tier 3):
python test_discretization.py

# Run the integration tests (Tier 2) — requires full model build:
python test_discretization.py --integration

# Quick sanity check at N=5 (fast):
python -c "
from discretization import rouwenhorst_multivariate
import numpy as np
Phi_11 = np.array([[0.01695,0.77529,-0.00264],[0.00928,0.86113,0.00079],[-1.19573,7.57961,0.80693]])
Sigma_ss = np.array([[4.484e-5,6.511e-7,-1.424e-5],[6.511e-7,7.949e-6,1.342e-5],[-1.424e-5,1.342e-5,1.961e-2]])
z_bar = np.array([-8.350e-4, 9.122e-3, -4.148])
Phi_0 = (np.eye(3) - Phi_11) @ z_bar
Sigma_chol = np.linalg.cholesky(Sigma_ss)
grids, Pi, si = rouwenhorst_multivariate([5,5,5], Phi_0, Phi_11, Sigma_chol,
                                          method='farmer_toda', grid_scale=0.85)
print(f'Shape: {Pi.shape}, row sums OK: {np.allclose(Pi.sum(1), 1)}')
print(f'Min prob: {Pi.min():.2e}, all >= 0: {(Pi >= 0).all()}')
"
```

---

## 6. Computational Cost

| Grid | Farmer-Toda time | Independence time | Ratio |
|------|------------------:|------------------:|------:|
| 5³ (125) | ~2 s | < 0.01 s | 200× |
| 7³ (343) | ~11 s | < 0.01 s | 1000× |
| 9³ (729) | ~40 s | < 0.01 s | 4000× |

This is a one-time precomputation cost. The backward induction solver
takes minutes to hours. The Farmer-Toda setup cost is negligible
in the overall workflow.

---

## 7. Typical Usage After the Change

```python
from model import DiscretizationConfig
from precompute import Precompute, build_model

# Quick debug run (N=5, scaled grid, ~2s setup):
disc_debug = DiscretizationConfig(
    state_grid_sizes=(5, 5, 5),
    state_grid_scale=0.85,
)

# Production run (N=7, full width, ~11s setup):
disc_prod = DiscretizationConfig(
    state_grid_sizes=(7, 7, 7),
)

# Comparison run against old method:
disc_old = DiscretizationConfig(
    state_grid_sizes=(7, 7, 7),
    state_discretization_method="independent",
)

# All three produce Precompute objects with the same interface:
pc = Precompute(model, disc_config=disc_prod)
# pc.Pi_state, pc.state_grid, pc.mu_r, etc. — same shapes, same downstream code
```

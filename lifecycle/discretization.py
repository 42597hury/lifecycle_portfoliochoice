"""
discretization.py — Discretization routines for continuous processes.

Contains:
  - rouwenhorst_univariate() — Markov chain approximation for AR(1)
  - build_state_grid() — mode-aware financial-state interpolation grid
  - discretize_income_ar1_mixture() — income AR(1) with mixture-normal innovations
  - get_eps_quadrature_corrected() — transitory shock Judd-style mixture quadrature
  - get_eta_quadrature_mixture() — persistent innovation Judd-style mixture quadrature
  - mixture_cdf() — mixture-normal CDF helper

Dependencies: numpy, scipy, numerics (project)
"""

from math import comb

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import eval_legendre, roots_hermite, roots_jacobi
from scipy.stats import norm
import warnings

from lifecycle.numerics import _normal_bin_probs


# =============================================================================
# MIXTURE-NORMAL HELPERS
# =============================================================================

def mixture_cdf(x, p, mu1, sigma1, mu2, sigma2):
    """CDF of two-component normal mixture."""
    return p * norm.cdf(x, loc=mu1, scale=sigma1) + (1.0 - p) * norm.cdf(x, loc=mu2, scale=sigma2)


# =============================================================================
# ROUWENHORST DISCRETIZATION
# =============================================================================

def rouwenhorst_univariate(N, mu, rho, sigma):
    """Rouwenhorst discretization for AR(1): y' = mu + rho*(y-mu) + sigma*eps."""
    if N < 2:
        raise ValueError("Rouwenhorst N must be >= 2")

    p = (1.0 + rho) / 2.0
    q = p
    Pi = np.array([[p, 1.0 - p], [1.0 - q, q]], dtype=float)

    for n in range(3, N + 1):
        Pi_new = np.zeros((n, n), dtype=float)
        Pi_new[:-1, :-1] += p * Pi
        Pi_new[:-1, 1:] += (1.0 - p) * Pi
        Pi_new[1:, :-1] += (1.0 - q) * Pi
        Pi_new[1:, 1:] += q * Pi
        Pi_new[1:-1, :] *= 0.5
        Pi = Pi_new

    sigma_y = sigma / np.sqrt(max(1e-14, 1.0 - rho * rho))
    psi = sigma_y * np.sqrt(N - 1.0)
    y_grid = np.linspace(mu - psi, mu + psi, N)

    return y_grid, Pi



def stationary_covariance(Phi, Sigma_innov):
    """Solve the stationary covariance of s' = mu + Phi s + v, v ~ N(0, Sigma_innov)."""
    Phi = np.asarray(Phi, dtype=float)
    Sigma_innov = np.asarray(Sigma_innov, dtype=float)
    if Phi.shape[0] != Phi.shape[1]:
        raise ValueError("Phi must be square")
    if Sigma_innov.shape != Phi.shape:
        raise ValueError("Sigma_innov must match Phi shape")

    eigs = np.linalg.eigvals(Phi)
    max_abs = float(np.max(np.abs(eigs)))
    if max_abs >= 1.0 - 1e-12:
        raise ValueError(
            f"VAR is non-stationary (max |eigenvalue(Phi)| = {max_abs:.6f} >= 1); "
            "stationary covariance is undefined."
        )

    Sigma_innov = 0.5 * (Sigma_innov + Sigma_innov.T)
    Sigma_z = solve_discrete_lyapunov(Phi, Sigma_innov)
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.T)
    return Sigma_z


def _stationary_probs_from_transition(Pi, tol=1e-12, max_iter=10000):
    """Stationary distribution of a row-stochastic transition matrix via power iteration."""
    Pi = np.asarray(Pi, dtype=float)
    n = Pi.shape[0]
    probs = np.full(n, 1.0 / n, dtype=float)

    for _ in range(max_iter):
        probs_next = probs @ Pi
        if np.max(np.abs(probs_next - probs)) < tol:
            return probs_next
        probs = probs_next

    warnings.warn("State-grid stationary distribution power iteration did not converge.")
    return probs


def _independence_rouwenhorst_pi(N_vec, Phi, Sigma_innov):
    """Kronecker-product Rouwenhorst transition used for legacy/fallback discrete-state paths."""
    N_vec = np.asarray(N_vec, dtype=int)
    Phi = np.asarray(Phi, dtype=float)
    Sigma_innov = np.asarray(Sigma_innov, dtype=float)
    k = len(N_vec)

    rho_diag = np.diag(Phi)
    sigma_diag = np.sqrt(np.maximum(np.diag(Sigma_innov), 1e-14))

    marginals = []
    for d in range(k):
        Nd = int(N_vec[d])
        if Nd == 1:
            Pi_d = np.ones((1, 1), dtype=float)
        else:
            _, Pi_d = rouwenhorst_univariate(Nd, 0.0, float(rho_diag[d]), float(sigma_diag[d]))
        marginals.append(Pi_d)

    n_total = int(np.prod(N_vec))
    state_indices = np.empty((n_total, k), dtype=np.int64)
    for idx, multi_idx in enumerate(np.ndindex(*N_vec.tolist())):
        state_indices[idx, :] = np.asarray(multi_idx, dtype=np.int64)

    Pi_joint = np.ones((n_total, n_total), dtype=float)
    for dim in range(k):
        Pi_dim = marginals[dim]
        from_idx = state_indices[:, dim]
        to_idx = state_indices[:, dim]
        Pi_joint *= Pi_dim[np.ix_(from_idx, to_idx)]

    row_sums = Pi_joint.sum(axis=1, keepdims=True)
    Pi_joint = Pi_joint / np.maximum(row_sums, 1e-300)
    return Pi_joint, state_indices


def _normalize_n_stds(n_stds, k):
    """Broadcast scalar to length-k float array; validate length and positivity.

    Accepts a scalar (int/float) or a length-k sequence. Returns a length-k
    float ndarray. Raises on length mismatch or non-positive entries.
    """
    arr = np.asarray(n_stds, dtype=float)
    if arr.ndim == 0:
        arr = np.full(int(k), float(arr))
    if arr.shape != (int(k),):
        raise ValueError(
            f"n_stds must be scalar or length-{int(k)} sequence; got shape {arr.shape}"
        )
    if np.any(arr <= 0):
        raise ValueError(f"all n_stds entries must be > 0; got {arr.tolist()}")
    return arr


def build_state_grid(N_vec, mu_intercept, Phi, Sigma_innov, n_stds=3.0, mode="principal"):
    """Build the financial-state interpolation grid.

    Parameters
    ----------
    N_vec : sequence of int
        Grid sizes per state dimension.
    mu_intercept : ndarray, shape (k,)
        Intercept in s' = mu_intercept + Phi s + v.
    Phi : ndarray, shape (k, k)
        State persistence matrix.
    Sigma_innov : ndarray, shape (k, k)
        Innovation covariance matrix.
    n_stds : float or sequence of length k
        Half-width of each interpolation axis in standardized units. A scalar
        broadcasts to all axes. A length-k sequence applies a per-axis bound.
        In `principal` mode, axes are Cholesky directions (not physical state
        variables); in `lyapunov-axis` mode, axes correspond to the physical
        state variables.
    mode : {"naive", "lyapunov-axis", "principal"}
        Grid construction mode.

    Returns
    -------
    dict
        Contains economic grid points in `state_grid`, interpolation axes in
        `state_bracket_grids`, transition fallback arrays, and stationary
        covariance diagnostics.
    """
    N_vec = np.asarray(N_vec, dtype=int)
    Phi = np.asarray(Phi, dtype=float)
    Sigma_innov = np.asarray(Sigma_innov, dtype=float)
    mu_intercept = np.asarray(mu_intercept, dtype=float)

    k = len(N_vec)
    if k < 1 or k > 3:
        raise ValueError("build_state_grid currently requires 1, 2, or 3 state dimensions")
    if Phi.shape != (k, k):
        raise ValueError(f"Phi must have shape {(k, k)}")
    if Sigma_innov.shape != (k, k):
        raise ValueError(f"Sigma_innov must have shape {(k, k)}")
    if mu_intercept.shape != (k,):
        raise ValueError(f"mu_intercept must have shape {(k,)}")
    if np.any(N_vec < 1):
        raise ValueError("Each state grid dimension must have at least 1 point")

    if mode not in {"naive", "lyapunov-axis", "principal"}:
        raise ValueError(f"Unknown state grid mode: {mode!r}")

    n_stds_arr = _normalize_n_stds(n_stds, k)

    mu_s = np.linalg.solve(np.eye(k) - Phi, mu_intercept)
    Sigma_z = stationary_covariance(Phi, Sigma_innov)
    sigma_z = np.sqrt(np.maximum(np.diag(Sigma_z), 0.0))
    L = np.linalg.cholesky(Sigma_z)
    L_inv = np.linalg.inv(L)

    Pi_state, state_indices = _independence_rouwenhorst_pi(N_vec, Phi, Sigma_innov)
    n_total = int(np.prod(N_vec))

    if mode == "principal":
        state_bracket_grids = []
        for d in range(k):
            Nd = int(N_vec[d])
            if Nd == 1:
                state_bracket_grids.append(np.array([0.0], dtype=float))
            else:
                state_bracket_grids.append(
                    np.linspace(-n_stds_arr[d], n_stds_arr[d], Nd, dtype=float)
                )
        state_grid = np.empty((n_total, k), dtype=float)
        for i in range(n_total):
            u = np.empty(k, dtype=float)
            for d in range(k):
                u[d] = state_bracket_grids[d][state_indices[i, d]]
            state_grid[i] = mu_s + L @ u

        probs_1d = [_normal_bin_probs(grid) for grid in state_bracket_grids]
        stationary_probs = np.ones(n_total, dtype=float)
        for i in range(n_total):
            p = 1.0
            for d in range(k):
                p *= probs_1d[d][state_indices[i, d]]
            stationary_probs[i] = p
        stationary_probs /= stationary_probs.sum()

        bracket_shift = np.array(mu_s, copy=True)
        bracket_L_inv = np.array(L_inv, copy=True)
    else:
        if mode == "lyapunov-axis":
            state_bracket_grids = []
            for d in range(k):
                Nd = int(N_vec[d])
                if Nd == 1:
                    state_bracket_grids.append(np.array([float(mu_s[d])], dtype=float))
                else:
                    state_bracket_grids.append(
                        np.linspace(mu_s[d] - n_stds_arr[d] * sigma_z[d],
                                    mu_s[d] + n_stds_arr[d] * sigma_z[d],
                                    Nd, dtype=float)
                    )
        else:
            state_bracket_grids = []
            for d in range(k):
                rho_d = float(Phi[d, d])
                sigma_d = float(np.sqrt(max(1e-14, Sigma_innov[d, d])))
                Nd = int(N_vec[d])
                if Nd == 1:
                    grid_d = np.array([float(mu_s[d])], dtype=float)
                else:
                    grid_d, _ = rouwenhorst_univariate(Nd, float(mu_s[d]), rho_d, sigma_d)
                state_bracket_grids.append(grid_d.astype(float))

        state_grid = np.empty((n_total, k), dtype=float)
        for i in range(n_total):
            for d in range(k):
                state_grid[i, d] = state_bracket_grids[d][state_indices[i, d]]

        stationary_probs = _stationary_probs_from_transition(Pi_state)
        bracket_shift = np.zeros(k, dtype=float)
        bracket_L_inv = np.eye(k, dtype=float)

    return {
        "mode": mode,
        "N_vec": np.array(N_vec, copy=True),
        "mu_s": np.array(mu_s, copy=True),
        "Sigma_z": np.array(Sigma_z, copy=True),
        "sigma_z": np.array(sigma_z, copy=True),
        "L": np.array(L, copy=True),
        "L_inv": np.array(L_inv, copy=True),
        "state_bracket_grids": [np.array(g, copy=True) for g in state_bracket_grids],
        "state_grid": state_grid,
        "state_indices": state_indices,
        "Pi_state": Pi_state,
        "stationary_probs": stationary_probs,
        "bracket_shift": bracket_shift,
        "bracket_L_inv": bracket_L_inv,
    }


# =============================================================================
# INCOME DISCRETIZATION
# =============================================================================

def discretize_income_ar1_mixture(rho, p, mu1, sigma1, mu2, sigma2, N, n_stds=3):
    """Discretize persistent income AR(1) with mixture-normal innovations."""
    mu_eta = p * mu1 + (1.0 - p) * mu2
    var_eta = p * (sigma1 ** 2 + (mu1 - mu_eta) ** 2) + (1.0 - p) * (sigma2 ** 2 + (mu2 - mu_eta) ** 2)
    std_z = np.sqrt(var_eta / max(1e-14, 1.0 - rho ** 2))

    z_grid = np.linspace(-n_stds * std_z, n_stds * std_z, N)
    dz = z_grid[1] - z_grid[0]
    half_bin = 0.5 * dz

    Pi_z = np.zeros((N, N), dtype=float)
    for i, z_t in enumerate(z_grid):
        mean_next = rho * z_t
        for j in range(N):
            if j == 0:
                upper = z_grid[j] + half_bin - mean_next
                Pi_z[i, j] = mixture_cdf(upper, p, mu1, sigma1, mu2, sigma2)
            elif j == N - 1:
                lower = z_grid[j] - half_bin - mean_next
                Pi_z[i, j] = 1.0 - mixture_cdf(lower, p, mu1, sigma1, mu2, sigma2)
            else:
                upper = z_grid[j] + half_bin - mean_next
                lower = z_grid[j] - half_bin - mean_next
                Pi_z[i, j] = mixture_cdf(upper, p, mu1, sigma1, mu2, sigma2) - mixture_cdf(lower, p, mu1, sigma1, mu2, sigma2)
        row_sum = Pi_z[i, :].sum()
        if abs(row_sum - 1.0) > 1e-8:
            Pi_z[i, :] /= row_sum

    return z_grid, Pi_z


def _normal_raw_moment(k, mu, sigma):
    """E[X^k] for X ~ N(mu, sigma^2).

    Uses the binomial expansion E[(mu + sigma Z)^k] with E[Z^j] = (j-1)!! for
    even j and zero for odd j (probabilist's notation, double factorial).
    """
    total = 0.0
    for j in range(0, k + 1, 2):
        ez = 1.0
        for m in range(j - 1, 0, -2):
            ez *= m
        total += comb(k, j) * (mu ** (k - j)) * (sigma ** j) * ez
    return total


def _mixture_raw_moments(probs, mus, sigmas, max_order):
    """Raw moments m[0..max_order] of a finite normal mixture."""
    moments = np.zeros(max_order + 1, dtype=float)
    for k in range(max_order + 1):
        moments[k] = sum(p * _normal_raw_moment(k, mu, s)
                         for p, mu, s in zip(probs, mus, sigmas))
    return moments


def _judd_mixture_quadrature(probs, mus, sigmas, n):
    """Build an n-point Gauss quadrature for a normal mixture via Judd (1998).

    Step 1: solve the Hankel-matrix linear system for the monic orthogonal
            polynomial p(x) of degree n.
    Step 2: roots of p give the quadrature nodes.
    Step 3: solve the Vandermonde system for the weights.

    Returns
    -------
    nodes : ndarray, shape (n,)
        Sorted ascending. All real, distinct.
    weights : ndarray, shape (n,)
        All strictly positive; sum to sum(probs) (= 1 for a probability mass).

    Raises
    ------
    RuntimeError
        If the polynomial roots have non-negligible imaginary parts (would
        indicate a numerical-conditioning failure; should not happen for n <= 6
        with the calibrated parameters).
    """
    moments = _mixture_raw_moments(probs, mus, sigmas, max_order=2 * n)

    # Step 1: Hankel matrix H[i,j] = m_{i+j}, monic constraint a_n = 1,
    # solve H_oo a_lower = -h for the n free coefficients.
    H_oo = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            H_oo[i, j] = moments[i + j]
    h = moments[n: 2 * n].astype(float)
    a_lower = np.linalg.solve(H_oo, -h)
    coeffs_ascending = np.concatenate([a_lower, [1.0]])

    # Step 2: roots of p (numpy.roots wants highest-degree-first ordering)
    roots = np.roots(coeffs_ascending[::-1])
    if np.max(np.abs(roots.imag)) > 1e-8:
        raise RuntimeError(
            f"Judd quadrature: polynomial roots have non-negligible imaginary "
            f"parts (max |Im| = {np.max(np.abs(roots.imag)):.3e}). "
            "Numerical conditioning failed."
        )
    nodes = np.sort(roots.real)

    # Step 3: Vandermonde A omega = b where A[k, i] = x_i^k, b = m_k
    A = np.vander(nodes, n, increasing=True).T  # shape (n, n)
    weights = np.linalg.solve(A, moments[:n])

    if np.any(weights <= 0):
        raise RuntimeError(
            f"Judd quadrature: non-positive weights produced "
            f"(min = {weights.min():.3e}). Numerical conditioning failed; "
            "Theorem 3 guarantees positivity, so this is a float64 issue."
        )
    return nodes, weights


def _gauss_lobatto_1d(n, a=-1.0, b=1.0):
    """1-D Gauss–Lobatto quadrature on [a, b] with unit weight.

    Returns n nodes (with `a`, `b` as fixed endpoints) and n weights such
    that the rule is exact for polynomials of degree ≤ 2n − 3 against
    the unit weight on [a, b].

    Construction
    ------------
    On [−1, 1]: endpoints fixed at ±1; the n − 2 interior nodes are roots
    of P′_{n−1}(x), equivalently the roots of the Jacobi polynomial
    P^{(1,1)}_{n−2}(x). Weights use the closed-form Lobatto formula
    `w_i = 2 / (n(n−1) [P_{n−1}(x_i)]²)`. Linear scaling to [a, b]
    multiplies nodes and weights by `(b − a)/2` and shifts nodes by
    `(a + b)/2`.

    Parameters
    ----------
    n : int
        Total node count, n ≥ 2. The n = 2 case is the trapezoidal rule
        on [a, b].
    a, b : float
        Interval endpoints with b > a.

    Returns
    -------
    nodes : ndarray, shape (n,)
        Sorted ascending; nodes[0] = a, nodes[-1] = b.
    weights : ndarray, shape (n,)
        All strictly positive; sum to (b − a).

    Notes
    -----
    Polynomial exactness verified to machine precision for n ∈ {3..6} on
    [−1, 1]: degrees 0..(2n−3) integrated to ≤ 1e-15; degree (2n−2) has
    a non-zero residual (e.g. ≈ 0.27 for n = 3 at degree 4).

    This is a primitive — to integrate against the standard-normal weight
    φ(z) requires either a φ-reweight construction (truncate, multiply
    weights by φ(z_k), renormalise) or a generalised Lobatto rule
    constructed via moment-matching against the truncated Gaussian. The
    correct choice for the lifecycle return / state quadrature is under
    review; see `docs/handoff/HANDOFF_LOBATTO_QUADRATURE_REVIEW.md`.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Gauss–Lobatto requires integer n >= 2")
    if not (b > a):
        raise ValueError(f"require b > a; got a={a}, b={b}")

    if n == 2:
        nodes_std = np.array([-1.0, 1.0], dtype=float)
        weights_std = np.array([1.0, 1.0], dtype=float)
    else:
        interior, _ = roots_jacobi(int(n) - 2, 1.0, 1.0)
        nodes_std = np.concatenate(
            [[-1.0], np.sort(np.asarray(interior, dtype=float)), [1.0]]
        )
        P_nm1 = eval_legendre(int(n) - 1, nodes_std)
        weights_std = 2.0 / (int(n) * (int(n) - 1) * P_nm1 ** 2)

    half_width = 0.5 * (b - a)
    midpoint = 0.5 * (a + b)
    nodes = half_width * nodes_std + midpoint
    weights = half_width * weights_std
    return nodes, weights


def get_eps_quadrature_corrected(model, n_nodes=3):
    """Transitory shock quadrature, Judd (1998) construction.

    n_nodes is the TOTAL node count (no longer per-component K).
    Polynomial-exactness order = 2 * n_nodes - 1 against the mixture density.

    NOTE: model.mu_eps2 is NOT used. Component 2's mean is computed internally
    to enforce E[eps] = 0:  mu_eps2_effective = -(pe/(1-pe)) * mu_eps1.
    Only model.sigma_eps2 is used from the component-2 parameters.
    """
    pe = float(model.pe)
    mu1 = float(model.mu_eps1)
    mu2_eff = -(pe / (1.0 - pe)) * mu1
    eps_nodes, eps_weights = _judd_mixture_quadrature(
        probs=[pe, 1.0 - pe],
        mus=[mu1, mu2_eff],
        sigmas=[float(model.sigma_eps1), float(model.sigma_eps2)],
        n=n_nodes,
    )

    mean_check = float(np.sum(eps_nodes * eps_weights))
    if abs(mean_check) > 1e-10:
        print(f"WARNING: transitory shock mean = {mean_check:.6e} (should be near 0)")
    return eps_nodes, eps_weights


def get_eta_quadrature_mixture(model, n_nodes=3):
    """Persistent income innovation quadrature, Judd (1998) construction.

    n_nodes is the TOTAL node count (no longer per-component K).
    Polynomial-exactness order = 2 * n_nodes - 1 against the mixture density.

    NOTE: model.mu_eta2 is NOT used. Component 2's mean is computed internally
    to enforce E[eta] = 0:  mu_eta2_effective = -(pz/(1-pz)) * mu_eta1.
    Only model.sigma_eta2 is used from the component-2 parameters.
    """
    pz = float(model.pz)
    mu1 = float(model.mu_eta1)
    mu2_eff = -(pz / (1.0 - pz)) * mu1
    eta_nodes, eta_weights = _judd_mixture_quadrature(
        probs=[pz, 1.0 - pz],
        mus=[mu1, mu2_eff],
        sigmas=[float(model.sigma_eta1), float(model.sigma_eta2)],
        n=n_nodes,
    )

    mean_check = float(np.sum(eta_nodes * eta_weights))
    if abs(mean_check) > 1e-10:
        print(f"WARNING: eta quadrature mean = {mean_check:.6e} (should be near 0)")
    return eta_nodes, eta_weights


def _normalize_lobatto_Z(value, n_axes, axis_kind):
    """Normalise a Lobatto-Z spec to a length-`n_axes` tuple of None|float.

    Accepts:
      - None: no Lobatto on any axis (every entry is None).
      - float / int: Lobatto with this Z on every axis.
      - sequence of length n_axes: per-axis (entries None or float).

    Raises ValueError on length mismatch and TypeError on bad scalar types.
    """
    if value is None:
        return (None,) * int(n_axes)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{axis_kind}_lobatto_Z must be None|float|sequence, not bool")
    if isinstance(value, (int, float, np.integer, np.floating)):
        Z = float(value)
        if Z <= 0:
            raise ValueError(f"{axis_kind}_lobatto_Z must be positive; got {Z}")
        return (Z,) * int(n_axes)
    try:
        seq = list(value)
    except TypeError as exc:
        raise TypeError(
            f"{axis_kind}_lobatto_Z must be None|float|sequence, "
            f"got {type(value).__name__}"
        ) from exc
    if len(seq) != int(n_axes):
        raise ValueError(
            f"{axis_kind}_lobatto_Z length {len(seq)} does not match {axis_kind}={int(n_axes)}"
        )
    out = []
    for d, v in enumerate(seq):
        if v is None:
            out.append(None)
            continue
        if isinstance(v, (bool, np.bool_)):
            raise TypeError(f"{axis_kind}_lobatto_Z[{d}] must be None|float, not bool")
        try:
            Zf = float(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{axis_kind}_lobatto_Z[{d}] must be None|float, got {type(v).__name__}"
            ) from exc
        if Zf <= 0:
            raise ValueError(f"{axis_kind}_lobatto_Z[{d}] must be positive; got {Zf}")
        out.append(Zf)
    return tuple(out)


def _build_axis_grid(K_d, Z_d):
    """Build the 1D (z_nodes, weights) grid for a single quadrature axis.

    K_d=1   -> single zero node (degenerate axis).
    Z_d is None -> standard Gauss-Hermite of order K_d on N(0,1), nodes
                   in z = sqrt(2)*x, weights in w/sqrt(pi).
    Z_d float   -> closed-form prescribed-tails rule (Gauss-Hermite-Lobatto)
                   with tail nodes at +/- Z_d. Validity is enforced inside
                   `gauss_hermite_prescribed_tails`.
    """
    if K_d == 1:
        if Z_d is not None:
            raise ValueError("Lobatto requires K >= 3; got K=1 with Z set")
        return np.zeros(1, dtype=float), np.ones(1, dtype=float)
    if Z_d is None:
        z, w = roots_hermite(int(K_d))
        return z * np.sqrt(2.0), w / np.sqrt(np.pi)
    # Lobatto branch -- import here to avoid a top-level cycle
    # (quadrature_with_tails defers its imports of `_normalize_*` likewise).
    from lifecycle.quadrature_with_tails import gauss_hermite_prescribed_tails
    if int(K_d) % 2 == 0 or int(K_d) not in (3, 5, 7):
        raise ValueError(
            f"Lobatto axis requires odd K in {{3,5,7}}; got K={K_d}. "
            f"For K >= 9, use Hermite (set lobatto_Z entry to None)."
        )
    return gauss_hermite_prescribed_tails(int(K_d), float(Z_d))


def _normalize_ret_nodes(value, n_ret):
    """Normalize per-dimension return-quadrature node count to a length-`n_ret` tuple.

    Accepts either a scalar `int` (uniform K across all return dimensions) or a
    sequence of length `n_ret` (one K per dimension, ordered as `model.ret_names`).
    Single source of truth — every consumer of `n_ret_nodes_1d` calls this.
    """
    if isinstance(value, (bool,)):
        raise TypeError("n_ret_nodes_1d must be int or sequence of ints, not bool")
    if isinstance(value, (int, np.integer)):
        return (int(value),) * int(n_ret)
    try:
        t = tuple(int(v) for v in value)
    except TypeError as exc:
        raise TypeError(
            f"n_ret_nodes_1d must be int or sequence of ints, got {type(value).__name__}"
        ) from exc
    if len(t) != int(n_ret):
        raise ValueError(
            f"n_ret_nodes_1d tuple length {len(t)} does not match n_ret={int(n_ret)}"
        )
    return t


def get_return_quadrature(model, n_nodes=1, lobatto_Z=None):
    """Residual return quadrature for N(0, Sigma_r_cond).

    Uses tensor-product Gauss-Hermite in standardized coordinates `z`,
    transformed to `r ~ N(0, Sigma_r_cond)` via the Cholesky factor:
    `r = z @ L^T` where `L L^T = Sigma_r_cond` and `L` is lower-triangular.
    This matches the transform used by `get_state_quadrature`.

    Per-axis K_i interpretation (under input return order `(rtb, xr, xb)`,
    consistent with the field name `n_ret_nodes_1d = (K_rtb, K_xr, K_xb)`):

    - `K[0] = K_rtb`: refines the `z_0` axis. Because `L` is lower-triangular,
      `z_0` is the only component of `z` that contributes to the rtb axis
      of `r`, so this is a clean rtb-axis refinement.
    - `K[1] = K_xr`: refines the `z_1` axis. `z_1` contributes to xr (and
      not rtb), giving the xr residual after the rtb correlation has been
      orthogonalized away.
    - `K[2] = K_xb`: refines the `z_2` axis. `z_2` is the pure xb residual
      after rtb and xr have been orthogonalized away (the `L[2,2]` direction).

    These per-axis labels are honest under Cholesky, in contrast to the
    legacy eigendecomposition transform where `K[i]` referred to the i-th
    smallest-eigenvalue direction (mislabelled in code/docs as physical
    asset axes — see `_check_kret_naming.py` for the empirical mismatch).

    Parameters
    ----------
    model : LifecyclePortfolioModel
        Supplies `n_ret` and `Sigma_r_cond`.
    n_nodes : int OR sequence of ints of length `model.n_ret`
        Gauss-Hermite order per return dimension. A scalar `K` is equivalent to
        the uniform tuple `(K,) * n_ret`. With per-dimension orders
        `(K_1, ..., K_{n_ret})`, the tensor-product rule has
        `prod(K_i)` joint nodes.

    Returns
    -------
    ret_nodes : ndarray, shape (K_eff, n_ret)
        Residual log-return shocks to add to `mu_r[i, j, :]`.
    ret_weights : ndarray, shape (K_eff,)
        Tensor-product quadrature weights summing to one.

    Notes
    -----
    All-ones (`K_i = 1` for every i) is treated as the exact K=1 approximation
    used previously: a single zero residual shock with weight one.

    Sigma_r_cond is required to be positive definite (Cholesky requirement).
    For our calibration this is always satisfied because the residual return
    components are not collinear after partitioning.
    """
    n_ret = int(model.n_ret)
    K_per_dim = _normalize_ret_nodes(n_nodes, n_ret)
    Z_per_dim = _normalize_lobatto_Z(lobatto_Z, n_ret, "ret")
    if any(k < 1 for k in K_per_dim):
        raise ValueError("All entries of n_ret_nodes_1d must be >= 1")

    if all(k == 1 for k in K_per_dim) and all(z is None for z in Z_per_dim):
        return np.zeros((1, n_ret), dtype=float), np.ones(1, dtype=float)

    grids_z, grids_w = [], []
    for K, Z in zip(K_per_dim, Z_per_dim):
        z, w = _build_axis_grid(K, Z)
        grids_z.append(z)
        grids_w.append(w)

    grid_1d = np.meshgrid(*grids_z, indexing="ij")
    weight_1d = np.meshgrid(*grids_w, indexing="ij")

    z_nodes = np.stack([g.ravel() for g in grid_1d], axis=1)
    ret_weights = np.prod(np.stack(weight_1d, axis=0), axis=0).ravel()

    # Symmetrize then Cholesky: lower-triangular L, L L^T = Sigma_r_cond.
    Sigma = 0.5 * (np.asarray(model.Sigma_r_cond, dtype=float)
                    + np.asarray(model.Sigma_r_cond, dtype=float).T)
    L = np.linalg.cholesky(Sigma)
    ret_nodes = z_nodes @ L.T   # (K_total, n_ret)

    return ret_nodes, ret_weights


def _normalize_state_nodes(value, n_state):
    """Normalize n_state_quad_nodes to a length-n_state tuple of ints.

    Accepts a scalar (broadcast to every axis) or a length-`n_state`
    sequence. Mirrors `_normalize_ret_nodes` for the return quadrature.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("n_state_quad_nodes must be int or sequence of ints, got bool")
    try:
        scalar = int(value)
        if scalar != value:
            raise TypeError
        return (scalar,) * int(n_state)
    except TypeError:
        pass
    try:
        t = tuple(int(v) for v in value)
    except TypeError as exc:
        raise TypeError(
            f"n_state_quad_nodes must be int or sequence of ints, "
            f"got {type(value).__name__}"
        ) from exc
    if len(t) != int(n_state):
        raise ValueError(
            f"n_state_quad_nodes tuple length {len(t)} does not match "
            f"n_state={int(n_state)}"
        )
    return t


def get_state_quadrature(model, n_nodes=3, lobatto_Z=None):
    """State innovation quadrature for N(0, Sigma_ss).

    Constructs tensor-product Gauss-Hermite nodes for integrating over the
    state innovation v^s ~ N(0, Sigma_ss). Used by the solver to replace
    the discrete Markov chain Pi_state.

    The Cholesky transform `v = L u` (lower-triangular) makes the per-axis
    K_per_dim[d] interpretable: under Cholesky factorization of Σ_ss with
    state ordering (cy, spr, y_1) (default since 2026-04-30), axis 0 is
    pure cy innovation, axis 1 is mostly spr innovation, axis 2 is
    residual y_1 innovation. Refining K[2] is the targeted lever for
    bond-return integration accuracy at high γ × |α| because
    `M[xb, y_1] = -8.7` is the dominant v-channel for bond returns.

    Parameters
    ----------
    model : LifecyclePortfolioModel
        Supplies `n_state` and `Sigma_ss`.
    n_nodes : int OR sequence of ints of length `model.n_state`
        Gauss-Hermite order per state-innovation dimension. A scalar `K`
        is equivalent to the uniform tuple `(K,) * n_state`. With per-
        dimension orders `(K_0, ..., K_{n_state-1})`, the tensor-product
        rule has `prod(K_d)` joint nodes.

    Returns
    -------
    v_nodes : ndarray, shape (K_total, n_state)
        Innovation quadrature nodes in the original coordinate system.
    v_weights : ndarray, shape (K_total,)
        Tensor-product quadrature weights summing to one.
    """
    n_state = int(model.n_state)
    K_per_dim = _normalize_state_nodes(n_nodes, n_state)
    Z_per_dim = _normalize_lobatto_Z(lobatto_Z, n_state, "state")
    if any(k < 1 for k in K_per_dim):
        raise ValueError("All entries of n_state_quad_nodes must be >= 1")

    grids_z, grids_w = [], []
    for K, Z in zip(K_per_dim, Z_per_dim):
        z, w = _build_axis_grid(K, Z)
        grids_z.append(z)
        grids_w.append(w)

    # Tensor product in standard-normal space (per-axis K)
    grid_1d = np.meshgrid(*grids_z, indexing="ij")
    weight_1d = np.meshgrid(*grids_w, indexing="ij")

    z_nodes = np.stack([g.ravel() for g in grid_1d], axis=1)            # (K_total, n_state)
    v_weights = np.prod(np.stack(weight_1d, axis=0), axis=0).ravel()    # (K_total,)

    # Transform from standard normal to N(0, Sigma_ss) via Cholesky.
    # Lower-triangular L makes axis d in u-space correspond to the d-th
    # variable's residual contribution after the previous variables have
    # been orthogonalized away (Gram-Schmidt order = state ordering).
    Sigma = 0.5 * (np.asarray(model.Sigma_ss, dtype=float)
                    + np.asarray(model.Sigma_ss, dtype=float).T)
    L = np.linalg.cholesky(Sigma)
    v_nodes = z_nodes @ L.T       # (K_total, n_state)

    return v_nodes, v_weights

"""
discretization.py — Discretization routines for continuous processes.

Contains:
  - rouwenhorst_univariate() — Markov chain approximation for AR(1)
  - rouwenhorst_multivariate() — independent Rouwenhorst across dimensions
  - discretize_income_ar1_mixture() — income AR(1) with mixture-normal innovations
  - get_eps_quadrature_corrected() — transitory shock Gauss-Hermite quadrature
  - mixture_cdf(), mixture_quantile() — mixture-normal CDF/quantile helpers

Dependencies: numpy, scipy (no project imports)
"""

import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm


# =============================================================================
# MIXTURE-NORMAL HELPERS
# =============================================================================

def mixture_cdf(x, p, mu1, sigma1, mu2, sigma2):
    """CDF of two-component normal mixture."""
    return p * norm.cdf(x, loc=mu1, scale=sigma1) + (1.0 - p) * norm.cdf(x, loc=mu2, scale=sigma2)


def mixture_quantile(q, p, mu1, sigma1, mu2, sigma2, tol=1e-10, max_iter=100):
    """Quantile of normal mixture via bisection."""
    lower = min(mu1 - 6.0 * sigma1, mu2 - 6.0 * sigma2)
    upper = max(mu1 + 6.0 * sigma1, mu2 + 6.0 * sigma2)

    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        cdf_val = mixture_cdf(mid, p, mu1, sigma1, mu2, sigma2)
        if abs(cdf_val - q) < tol:
            return mid
        if cdf_val < q:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


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


def rouwenhorst_multivariate(N_vec, mu, Phi, Sigma, method="independent"):
    """
    Multivariate Rouwenhorst with independence approximation across dimensions.

    Parameters:
        N_vec: list of grid sizes per state variable
        mu: intercept in z' = mu + Phi z + eps
        Phi: persistence matrix
        Sigma: Cholesky factor of eps covariance
    """
    if method != "independent":
        raise NotImplementedError("Only method='independent' is currently implemented")

    N_vec = np.asarray(N_vec, dtype=int)
    k = len(N_vec)
    if Phi.shape != (k, k):
        raise ValueError(f"Phi must have shape {(k, k)}, got {Phi.shape}")
    if Sigma.shape != (k, k):
        raise ValueError(f"Sigma must have shape {(k, k)}, got {Sigma.shape}")

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
        for j, z_next in enumerate(z_grid):
            upper = z_next + half_bin - mean_next
            lower = z_next - half_bin - mean_next
            Pi_z[i, j] = mixture_cdf(upper, p, mu1, sigma1, mu2, sigma2) - mixture_cdf(lower, p, mu1, sigma1, mu2, sigma2)
        Pi_z[i, :] /= np.maximum(Pi_z[i, :].sum(), 1e-300)

    return z_grid, Pi_z


def get_eps_quadrature_corrected(model, n_nodes=3):
    """Transitory shock quadrature using Gauss-Hermite with zero-mean enforcement.

    NOTE: model.mu_eps2 is NOT used. Component 2's mean is computed internally
    to enforce E[eps] = 0:  mu_eps2_effective = -(pe/(1-pe)) * mu_eps1.
    Only model.sigma_eps2 is used from the component-2 parameters.
    """
    nodes, weights = roots_hermite(n_nodes)
    weights = weights / np.sqrt(np.pi)
    nodes = nodes * np.sqrt(2.0)

    e1 = nodes * model.sigma_eps1 + model.mu_eps1

    mu_eps2_normalized = -(model.pe / (1.0 - model.pe)) * model.mu_eps1
    e2 = nodes * model.sigma_eps2 + mu_eps2_normalized

    w1 = weights * model.pe
    w2 = weights * (1.0 - model.pe)

    eps_nodes = np.concatenate([e1, e2])
    eps_weights = np.concatenate([w1, w2])

    mean_check = np.sum(eps_nodes * eps_weights)
    if abs(mean_check) > 1e-10:
        print(f"WARNING: transitory shock mean = {mean_check:.6e} (should be near 0)")

    return eps_nodes, eps_weights


def get_return_quadrature(model, n_nodes=1):
    """Residual return quadrature for N(0, Sigma_r_cond).

    Parameters
    ----------
    model : LifecyclePortfolioModel
        Supplies `n_ret` and `Sigma_r_cond`.
    n_nodes : int
        Gauss-Hermite order per return dimension. With `n_ret` return dimensions,
        the tensor-product rule has `n_nodes ** n_ret` joint nodes.

    Returns
    -------
    ret_nodes : ndarray, shape (K_eff, n_ret)
        Residual log-return shocks to add to `mu_r[i, j, :]`.
    ret_weights : ndarray, shape (K_eff,)
        Tensor-product quadrature weights summing to one.

    Notes
    -----
    `n_nodes=1` is treated as the exact K=1 approximation used previously:
    a single zero residual shock with weight one.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1 for return quadrature")

    n_ret = int(model.n_ret)
    if n_nodes == 1:
        return np.zeros((1, n_ret), dtype=float), np.ones(1, dtype=float)

    nodes_1d, weights_1d = roots_hermite(n_nodes)
    weights_1d = weights_1d / np.sqrt(np.pi)
    nodes_1d = nodes_1d * np.sqrt(2.0)

    grid_1d = np.meshgrid(*([nodes_1d] * n_ret), indexing="ij")
    weight_1d = np.meshgrid(*([weights_1d] * n_ret), indexing="ij")

    z_nodes = np.stack([g.ravel() for g in grid_1d], axis=1)
    ret_weights = np.prod(np.stack(weight_1d, axis=0), axis=0).ravel()

    Sigma = 0.5 * (np.asarray(model.Sigma_r_cond, dtype=float) + np.asarray(model.Sigma_r_cond, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    if np.any(eigvals < -1e-12):
        raise ValueError("Sigma_r_cond must be positive semidefinite for return quadrature")
    eigvals = np.clip(eigvals, 0.0, None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    ret_nodes = z_nodes @ transform.T

    return ret_nodes, ret_weights

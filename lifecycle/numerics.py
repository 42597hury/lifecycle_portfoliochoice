"""
numerics.py — Shared numerical helpers used by the model-construction layer
(discretization) and the simulator's host-side initial-z sampling.

Contents:
  * `_normal_bin_probs`
      Probability mass of N(mean, std**2) over the bins induced by a sorted
      grid. Used at model setup (`discretization.py`) for state-grid
      transitions, and at simulation time (`simulation.py`) for initial-z
      sampling.

The previous PCHIP utilities (`_pchip_slope_uniform`,
`_pchip_eval_with_basis`) were dropped in handoff 3 alongside the JAX
solver/simulator rewrite — both kernels now use linear interpolation in z,
so the cubic monotone Hermite is no longer needed.
"""

import numpy as np
from scipy.stats import norm


# =============================================================================
# Normal CDF mass over a grid's induced bins
# =============================================================================

def _normal_bin_probs(grid, mean=0.0, std=1.0):
    """Probability mass of N(mean, std**2) over bins induced by a sorted grid.

    Bin edges are the midpoints between neighbouring grid points, with
    (-inf, +inf) at the tails. The resulting probabilities are clipped
    non-negative and renormalised to sum to one.

    Returns a length-`len(grid)` array. Raises ValueError on empty grid,
    non-positive std, or numerically zero total mass.
    """
    grid = np.asarray(grid, dtype=float)
    n = grid.shape[0]
    if n < 1:
        raise ValueError("grid must contain at least one point")
    if std <= 0.0:
        raise ValueError("std must be strictly positive")
    if n == 1:
        return np.ones(1, dtype=float)

    edges = np.empty(n + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])

    probs = np.empty(n, dtype=float)
    for j in range(n):
        probs[j] = norm.cdf(edges[j + 1], loc=mean, scale=std) - norm.cdf(edges[j], loc=mean, scale=std)

    probs = np.clip(probs, 0.0, None)
    mass = probs.sum()
    if mass <= 0.0:
        raise ValueError("Normal bin probabilities sum to zero")
    return probs / mass

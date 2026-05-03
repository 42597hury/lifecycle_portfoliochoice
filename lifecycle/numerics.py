"""
numerics.py — Shared numerical helpers used by the solver, simulator, and
the model-construction layer (discretization).

Contents:
  * `_pchip_slope_uniform`, `_pchip_eval_with_basis`
      Fritsch-Carlson PCHIP slope and Hermite evaluator on a uniform stencil.
      Used in z-direction policy interpolation. Both `solver.py` and
      `simulation.py` need identical versions so the two share one cubic.

  * `_normal_bin_probs`
      Probability mass of N(mean, std**2) over the bins induced by a sorted
      grid. Used at model setup (`discretization.py`) for state-grid
      transitions, and at simulation time (`simulation.py`) for initial-z
      sampling.

Why a separate module: prior to this file, each function existed twice
(once per consumer) and the "mirrors X.Y" docstring comments were the only
indication that the copies had to stay in sync. Bug-fixing one copy and
forgetting the other was the maintainability hazard.

Note on `fast_interp_1d`: there are two functions with this name, one in
`solver.py` and one in `simulation.py`. They are intentionally NOT
deduplicated here — the solver version uses linear extrapolation (needed
for smooth Newton convergence at the wealth-grid edge) while the simulator
uses flat extrapolation (compounded linear extrapolation blows up across
the lifecycle when households drift outside the solved domain). Do not
"fix" one to match the other; see the docstrings on both.
"""

import numpy as np
from numba import njit
from scipy.stats import norm


# =============================================================================
# PCHIP cubic Hermite on a uniform stencil
# =============================================================================

@njit(fastmath=True, inline='always')
def _pchip_slope_uniform(d_left, d_right):
    """Fritsch-Carlson slope at an interior node for a UNIFORM grid.

    The slope is in units of "per unit fractional index" so it pairs with
    a Hermite evaluator on [0, 1]. Uniform spacing is required; do not use
    on non-uniform grids.
    """
    if d_left == 0.0 or d_right == 0.0:
        return 0.0
    if d_left * d_right <= 0.0:
        return 0.0
    return 2.0 * d_left * d_right / (d_left + d_right)


@njit(fastmath=True, inline='always')
def _pchip_eval_with_basis(p0, p1, p2, p3, h00, h10, h01, h11):
    """Evaluate PCHIP cubic Hermite from a 4-point stencil.

    Caller pre-computes the four Hermite basis values (h00, h10, h01, h11)
    at the target fractional position; this routine combines them with
    monotone slopes from the four stencil values (p0, p1, p2, p3).
    """
    d_l = p1 - p0
    d_m = p2 - p1
    d_r = p3 - p2
    m0 = _pchip_slope_uniform(d_l, d_m)
    m1 = _pchip_slope_uniform(d_m, d_r)
    return h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1


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

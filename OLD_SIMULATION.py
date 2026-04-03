"""
LIFECYCLE PORTFOLIO CHOICE MODEL - SIMULATION
==============================================

This cell contains the simulation function for the lifecycle model.
It uses the policy functions from the solver and state/transitions from precompute.

Timing Convention:
    1. Enter period t with state (z_t, r_t) and savings s_{t-1} from last period
    2. Realize portfolio return R_p on savings (using r_{t-1} -> r_t transition)
    3. Realize income Y_t (draw eps_t if working, else pension - no transitory shock)
    4. Compute cash-on-hand: x_t = s_{t-1} * R_p + Y_t
    5. Look up policy: c_t = C(x_t; z_t, r_t), alpha_t = A(x_t; z_t, r_t)
    6. Savings: s_t = x_t - c_t
    7. Draw survival, then draw (z_{t+1}, r_{t+1}) for next period

"""

import numpy as np
from typing import Union, Optional
import warnings

# Try to import numba for performance, fall back to pure Python if unavailable
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
    warnings.warn("Numba not available. Simulation will run slower.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_stationary_distribution(Pi: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    """
    Compute the stationary distribution of a Markov chain.
    
    The stationary distribution pi satisfies: pi = pi @ Pi
    We find it by iterating from a uniform start until convergence.
    
    Parameters
    ----------
    Pi : np.ndarray
        Transition matrix of shape (n, n) where Pi[i,j] = P(state j | state i)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    np.ndarray
        Stationary distribution of shape (n,)
    """
    n = Pi.shape[0]
    pi = np.ones(n) / n  # Start uniform
    
    for _ in range(max_iter):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    
    warnings.warn("Stationary distribution did not converge")
    return pi


def initialize_states(
    n_simulations: int,
    n_states: int,
    Pi: np.ndarray,
    method: Union[str, np.ndarray],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Initialize state indices for simulation.
    
    Parameters
    ----------
    n_simulations : int
        Number of households
    n_states : int
        Number of discrete states
    Pi : np.ndarray
        Transition matrix (used to compute stationary distribution)
    method : str or np.ndarray
        "median" - all start at median state
        "stationary" - draw from stationary distribution
        array - use provided indices directly
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Array of initial state indices, shape (n_simulations,)
    """
    if isinstance(method, str):
        if method == "median":
            return np.full(n_simulations, n_states // 2, dtype=np.int32)
        elif method == "stationary":
            pi = get_stationary_distribution(Pi)
            return rng.choice(n_states, size=n_simulations, p=pi).astype(np.int32)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    else:
        # Assume it's an array
        arr = np.asarray(method, dtype=np.int32)
        if arr.shape[0] != n_simulations:
            raise ValueError(f"Initial state array has length {arr.shape[0]}, expected {n_simulations}")
        return arr


# =============================================================================
# NUMBA-OPTIMIZED CORE SIMULATION
# =============================================================================

@njit
def fast_interp(x: float, x_grid: np.ndarray, y_grid: np.ndarray) -> float:
    """
    Fast linear interpolation with extrapolation at boundaries.
    
    Parameters
    ----------
    x : float
        Point to interpolate at
    x_grid : np.ndarray
        Grid points (must be sorted ascending)
    y_grid : np.ndarray
        Function values at grid points
        
    Returns
    -------
    float
        Interpolated value
    """
    n = len(x_grid)
    
    # Handle boundaries
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]
    
    # Binary search for interval
    left = 0
    right = n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if x_grid[mid] <= x:
            left = mid
        else:
            right = mid
    
    # Linear interpolation
    x0, x1 = x_grid[left], x_grid[right]
    y0, y1 = y_grid[left], y_grid[right]
    weight = (x - x0) / (x1 - x0)
    
    return y0 + weight * (y1 - y0)


@njit
def draw_discrete(probs: np.ndarray, u: float) -> int:
    """
    Draw from a discrete distribution using inverse CDF method.
    
    Parameters
    ----------
    probs : np.ndarray
        Probability vector (must sum to 1)
    u : float
        Uniform random draw in [0, 1)
        
    Returns
    -------
    int
        Index of the drawn state
    """
    cumsum = 0.0
    for i in range(len(probs)):
        cumsum += probs[i]
        if u < cumsum:
            return i
    return len(probs) - 1  # Fallback for numerical precision


@njit(parallel=True)
def simulate_lifecycle_core(
    # Policy functions (n_age, n_z, n_r, n_s)
    C_policy: np.ndarray,
    A_policy: np.ndarray,
    X_grid: np.ndarray,
    # Precomputed values
    Pi_z: np.ndarray,           # (n_z, n_z)
    Pi_r: np.ndarray,           # (n_r, n_r)
    r_grid: np.ndarray,         # (n_r,)
    bond_returns: np.ndarray,   # (n_age, n_r, n_r)
    working_income: np.ndarray, # (n_age, n_z, n_eps)
    pension: np.ndarray,        # (n_age, n_z)
    eps_nodes: np.ndarray,      # (n_eps,)
    eps_weights: np.ndarray,    # (n_eps,)
    survival_probs: np.ndarray, # (n_age,)
    # Dimensions
    n_simulations: int,
    n_ages: int,
    n_z: int,
    n_r: int,
    n_eps: int,
    retire_age_idx: int,        # Index where retirement starts
    # Initial conditions
    initial_z: np.ndarray,      # (n_simulations,)
    initial_r: np.ndarray,      # (n_simulations,)
    initial_wealth: np.ndarray, # (n_simulations,)
    # Random draws (pre-generated for reproducibility)
    uniform_draws: np.ndarray,  # (n_simulations, n_ages, 4) - survival, z, r, eps
) -> tuple:
    """
    Core simulation loop optimized with Numba.
    
    Returns tuple of arrays:
        sim_x, sim_c, sim_s, sim_a, sim_income, sim_z, sim_r, sim_alive
    """
    # Allocate output arrays
    sim_x = np.zeros((n_simulations, n_ages))
    sim_c = np.zeros((n_simulations, n_ages))
    sim_s = np.zeros((n_simulations, n_ages))
    sim_a = np.zeros((n_simulations, n_ages))
    sim_income = np.zeros((n_simulations, n_ages))
    sim_z = np.zeros((n_simulations, n_ages), dtype=np.int32)
    sim_r = np.zeros((n_simulations, n_ages), dtype=np.int32)
    sim_alive = np.ones((n_simulations, n_ages), dtype=np.bool_)
    
    # Parallel loop over households
    for i in prange(n_simulations):
        # Initialize
        z_idx = initial_z[i]
        r_idx = initial_r[i]
        r_idx_prev = r_idx  # For t=0, no previous rate, use current
        s_prev = 0.0        # No savings before t=0
        
        for t in range(n_ages):
            # Check if household is still alive
            if t > 0 and not sim_alive[i, t]:
                # Already dead, skip (arrays remain zero)
                continue
            
            # Store current state indices
            sim_z[i, t] = z_idx
            sim_r[i, t] = r_idx
            
            # -----------------------------------------------------------------
            # Step 1: Realize portfolio return on previous savings
            # -----------------------------------------------------------------
            if t == 0:
                # First period: no portfolio return, use initial wealth
                R_p = 1.0
                wealth_from_savings = initial_wealth[i]
            else:
                # Portfolio return depends on (r_{t-1}, r_t) transition
                R_bond = bond_returns[t-1, r_idx_prev, r_idx]
                R_bill = np.exp(r_grid[r_idx_prev])
                alpha_prev = sim_a[i, t-1]
                R_p = alpha_prev * R_bond + (1.0 - alpha_prev) * R_bill
                wealth_from_savings = s_prev * R_p
            
            # -----------------------------------------------------------------
            # Step 2: Realize income
            # -----------------------------------------------------------------
            if t < retire_age_idx:
                # Working: draw transitory shock
                eps_idx = draw_discrete(eps_weights, uniform_draws[i, t, 3])
                income_t = working_income[t, z_idx, eps_idx]
            else:
                # Retired: pension income, no transitory shock
                income_t = pension[t, z_idx]
            
            sim_income[i, t] = income_t
            
            # -----------------------------------------------------------------
            # Step 3: Cash-on-hand
            # -----------------------------------------------------------------
            x_t = wealth_from_savings + income_t
            
            # Ensure x_t is within grid bounds (with small buffer)
            x_grid_t = X_grid[t, z_idx, r_idx, :]
            x_min = x_grid_t[0]
            x_max = x_grid_t[-1]
            
            if x_t < x_min:
                x_t = x_min * 1.0001  # Slightly above minimum
            if x_t > x_max:
                x_t = x_max * 0.9999  # Slightly below maximum
            
            sim_x[i, t] = x_t
            
            # -----------------------------------------------------------------
            # Step 4: Look up optimal policies
            # -----------------------------------------------------------------
            c_grid_t = C_policy[t, z_idx, r_idx, :]
            a_grid_t = A_policy[t, z_idx, r_idx, :]
            
            c_t = fast_interp(x_t, x_grid_t, c_grid_t)
            a_t = fast_interp(x_t, x_grid_t, a_grid_t)
            
            # Enforce constraints
            c_t = max(c_t, 1e-8)           # Minimum consumption
            c_t = min(c_t, x_t - 1e-8)     # Can't consume more than wealth
            a_t = max(0.0, min(1.0, a_t))  # Portfolio share in [0, 1]
            
            sim_c[i, t] = c_t
            sim_a[i, t] = a_t
            
            # -----------------------------------------------------------------
            # Step 5: Savings
            # -----------------------------------------------------------------
            s_t = x_t - c_t
            sim_s[i, t] = s_t
            
            # -----------------------------------------------------------------
            # Step 6: Transition to next period (if not terminal)
            # -----------------------------------------------------------------
            if t < n_ages - 1:
                # Survival
                if uniform_draws[i, t, 0] > survival_probs[t]:
                    # Household dies
                    sim_alive[i, t+1:] = False
                    continue
                
                # Draw next period states
                z_idx_new = draw_discrete(Pi_z[z_idx, :], uniform_draws[i, t, 1])
                r_idx_new = draw_discrete(Pi_r[r_idx, :], uniform_draws[i, t, 2])
                
                # Update for next iteration
                r_idx_prev = r_idx
                z_idx = z_idx_new
                r_idx = r_idx_new
                s_prev = s_t
    
    return sim_x, sim_c, sim_s, sim_a, sim_income, sim_z, sim_r, sim_alive


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def simulate_lifecycle(
    C_policy: np.ndarray,
    A_policy: np.ndarray,
    X_grid: np.ndarray,
    precompute,
    model,
    n_simulations: int = 10000,
    initial_wealth: Optional[Union[float, np.ndarray]] = None,
    initial_wealth_multiple: float = 2.0,
    initial_z: Union[str, np.ndarray] = "stationary",
    initial_r: Union[str, np.ndarray] = "median",
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Simulate lifecycle paths for multiple households.
    
    Parameters
    ----------
    C_policy : np.ndarray
        Consumption policy function, shape (n_age, n_z, n_r, n_s)
    A_policy : np.ndarray
        Portfolio allocation policy function, shape (n_age, n_z, n_r, n_s)
    X_grid : np.ndarray
        Cash-on-hand grid, shape (n_age, n_z, n_r, n_s)
    precompute : Precompute
        Precomputed model objects containing grids and transitions
    model : BondsBillsModel
        Model parameters
    n_simulations : int
        Number of households to simulate
    initial_wealth : float, np.ndarray, or None
        Initial wealth at t=0. If None, computed as initial_wealth_multiple * E[Y_0 | z_0]
    initial_wealth_multiple : float
        Multiplier for expected income to set initial wealth (used if initial_wealth is None)
    initial_z : str or np.ndarray
        Initial income state. Options:
        - "median": all start at median state
        - "stationary": draw from stationary distribution
        - array: provide indices directly
    initial_r : str or np.ndarray
        Initial interest rate state. Same options as initial_z.
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict
        Dictionary containing simulation results:
        - 'x': cash-on-hand, shape (n_simulations, n_ages)
        - 'c': consumption, shape (n_simulations, n_ages)
        - 's': savings, shape (n_simulations, n_ages)
        - 'a': portfolio share, shape (n_simulations, n_ages)
        - 'income': realized income, shape (n_simulations, n_ages)
        - 'z': income state indices, shape (n_simulations, n_ages)
        - 'r': interest rate state indices, shape (n_simulations, n_ages)
        - 'alive': survival indicator, shape (n_simulations, n_ages)
        - 'ages': age values, shape (n_ages,)
    """
    if verbose:
        print("=" * 60)
        print("SIMULATING LIFECYCLE PATHS")
        print("=" * 60)
    
    # Extract dimensions
    n_ages = precompute.n_age
    n_z = precompute.n_z
    n_r = precompute.n_r
    n_eps = precompute.n_eps
    
    # Retirement age index (first period where age >= retire_age)
    retire_age_idx = model.retire_age - model.start_age
    
    if verbose:
        print(f"  Households: {n_simulations:,}")
        print(f"  Ages: {model.start_age} to {model.terminal_age} ({n_ages} periods)")
        print(f"  Retirement at age {model.retire_age} (index {retire_age_idx})")
        print(f"  State space: {n_z} income × {n_r} interest rate states")
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # -----------------------------------------------------------------
    # Initialize states
    # -----------------------------------------------------------------
    init_z_indices = initialize_states(n_simulations, n_z, precompute.Pi_z, initial_z, rng)
    init_r_indices = initialize_states(n_simulations, n_r, precompute.Pi_r, initial_r, rng)
    
    if verbose:
        z_method = initial_z if isinstance(initial_z, str) else "custom array"
        r_method = initial_r if isinstance(initial_r, str) else "custom array"
        print(f"  Initial z: {z_method}")
        print(f"  Initial r: {r_method}")
    
    # -----------------------------------------------------------------
    # Initialize wealth
    # -----------------------------------------------------------------
    if initial_wealth is None:
        # Compute expected income at t=0 for each household's z state
        # E[Y_0 | z_0] = sum over eps of working_income[0, z_0, eps] * eps_weights[eps]
        init_wealth_arr = np.zeros(n_simulations)
        for i in range(n_simulations):
            z_idx = init_z_indices[i]
            expected_income = np.sum(precompute.working_income[0, z_idx, :] * precompute.eps_weights)
            init_wealth_arr[i] = initial_wealth_multiple * expected_income
        if verbose:
            print(f"  Initial wealth: {initial_wealth_multiple:.1f} × E[Y_0 | z_0]")
            print(f"    Mean initial wealth: {np.mean(init_wealth_arr):.4f}")
    elif np.isscalar(initial_wealth):
        init_wealth_arr = np.full(n_simulations, initial_wealth)
        if verbose:
            print(f"  Initial wealth: {initial_wealth:.4f} (constant)")
    else:
        init_wealth_arr = np.asarray(initial_wealth)
        if len(init_wealth_arr) != n_simulations:
            raise ValueError(f"initial_wealth array has length {len(init_wealth_arr)}, expected {n_simulations}")
        if verbose:
            print(f"  Initial wealth: custom array (mean={np.mean(init_wealth_arr):.4f})")
    
    # -----------------------------------------------------------------
    # Pre-generate all random draws
    # -----------------------------------------------------------------
    # Shape: (n_simulations, n_ages, 4)
    # [:,:,0] - survival draws
    # [:,:,1] - z transition draws
    # [:,:,2] - r transition draws
    # [:,:,3] - transitory shock draws
    if verbose:
        print("  Generating random draws...")
    uniform_draws = rng.uniform(size=(n_simulations, n_ages, 4))
    
    # -----------------------------------------------------------------
    # Run core simulation
    # -----------------------------------------------------------------
    if verbose:
        print("  Running simulation...")
    
    import time
    start_time = time.time()
    
    (sim_x, sim_c, sim_s, sim_a, sim_income, 
     sim_z, sim_r, sim_alive) = simulate_lifecycle_core(
        C_policy=C_policy,
        A_policy=A_policy,
        X_grid=X_grid,
        Pi_z=precompute.Pi_z,
        Pi_r=precompute.Pi_r,
        r_grid=precompute.r_grid,
        bond_returns=precompute.bond_returns,
        working_income=precompute.working_income,
        pension=precompute.pension_after_tax,
        eps_nodes=precompute.eps_nodes,
        eps_weights=precompute.eps_weights,
        survival_probs=model.survival_probs,
        n_simulations=n_simulations,
        n_ages=n_ages,
        n_z=n_z,
        n_r=n_r,
        n_eps=n_eps,
        retire_age_idx=retire_age_idx,
        initial_z=init_z_indices,
        initial_r=init_r_indices,
        initial_wealth=init_wealth_arr,
        uniform_draws=uniform_draws.astype(np.float64)
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        # Compute summary statistics
        n_alive_final = np.sum(sim_alive[:, -1])
        survival_rate = n_alive_final / n_simulations
        mean_wealth_25 = np.mean(sim_x[sim_alive[:, 0], 0])
        mean_wealth_65 = np.mean(sim_x[sim_alive[:, retire_age_idx], retire_age_idx]) if retire_age_idx < n_ages else np.nan
        mean_wealth_final = np.mean(sim_x[sim_alive[:, -1], -1])
        
        print(f"\n  Simulation complete in {elapsed:.2f} seconds")
        print(f"  Survival to age {model.terminal_age}: {survival_rate:.1%}")
        print(f"  Mean wealth at age 25: {mean_wealth_25:.4f}")
        print(f"  Mean wealth at age 65: {mean_wealth_65:.4f}")
        print(f"  Mean wealth at age {model.terminal_age}: {mean_wealth_final:.4f}")
        print("=" * 60)
    
    return {
        'x': sim_x,
        'c': sim_c,
        's': sim_s,
        'a': sim_a,
        'income': sim_income,
        'z': sim_z,
        'r': sim_r,
        'alive': sim_alive,
        'ages': precompute.ages.copy()
    }


# =============================================================================
# TEST 
# =============================================================================

if __name__ == "__main__":
    sim_data = simulate_lifecycle(
      C_policy, A_policy, X_grid, precompute, model,
      n_simulations=500000,
      initial_z='stationary',
      initial_r='median',
      seed=42)
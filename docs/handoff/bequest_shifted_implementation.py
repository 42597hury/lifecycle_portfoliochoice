"""
Recommended implementation pattern for the De Nardi (2004) luxury-bequest shift.

Single-source the spec in model.py; have solver.py FOC kernels and diagnostics
call (or inline-mirror) the same formula. The constant DELTA_BEQUEST sits in
ONE place so sensitivity sweeps just bump it.

Drop-in replacement for the bequest functions in model.py and inlined math
at solver.py:519, 523, 668, 669 (and the matching diagnostic kernels).
"""

import numpy as np
from numba import njit

# =============================================================================
# Calibration constant — keep in one place
# =============================================================================
# Annuity-normalised luxury-bequest shifter.
#
# Bound on marginal bequest utility:  mu_max = b_bar * DELTA_BEQUEST^(-gamma) / A
# For (b_bar, gamma, A) = (10, 5, 4):
#     DELTA = 0.005  ->  mu_max ~ 8e11   (vs. raw spike ~1e30)
#     DELTA = 0.01   ->  mu_max ~ 2.5e10
#     DELTA = 0.02   ->  mu_max ~ 7.8e8
#
# Sensitivity sweep gate: if optimal alpha is stable to ~5% across
# {0.001, 0.005, 0.01, 0.02}, ship at 0.005 (closest to original CRRA).
DELTA_BEQUEST = 0.005


# =============================================================================
# model.py replacements — keep helpers consistent with the FOC kernels
# =============================================================================

def bequest_utility(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """
    Luxury-shifted bequest utility (De Nardi 2004 form):

        b(W, A) = b_bar * (max(W,0)/A + delta)^(1 - gamma) / (1 - gamma)

    Continuous at W = 0.  Asymptotically identical to the pure-CRRA
    spec for W/A >> delta (deviation ~ gamma * delta * A / W).

    The hard bankruptcy clamp survives via max(W, 0): heirs of bankrupt
    households still receive zero estate; debt does not pass through.
    """
    C_bar = np.maximum(W, 0.0) / A + delta
    return b_bar * C_bar ** (1.0 - gamma) / (1.0 - gamma)


def bequest_marginal(W, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """
    Marginal bequest utility:

        db/dW = b_bar * (max(W,0)/A + delta)^(-gamma) / A     for W > 0
              = 0                                              for W <= 0

    BOUNDED above by  b_bar * delta^(-gamma) / A, achieved at W -> 0+.
    """
    pos = W > 0.0
    C_bar = np.where(pos, W / A + delta, 1.0)  # placeholder for W<=0 branch
    mu = b_bar * C_bar ** (-gamma) / A
    return np.where(pos, mu, 0.0)


def bequest_marginal_inv(mu, A, gamma, b_bar, delta=DELTA_BEQUEST):
    """
    Inverse of bequest_marginal.  Now has a domain restriction:
    mu must lie in (0, mu_max] where mu_max = b_bar * delta^(-gamma) / A.
    Above mu_max the inverse is undefined (the agent CANNOT achieve
    marginal-utility values higher than mu_max — that is the whole point
    of the regularisation).

        W = A * [(mu * A / b_bar)^(-1/gamma) - delta]

    Clamps to W >= 0 for safety.  Used in EGM terminal step; if mu_target
    exceeds mu_max, the constraint W = 0 binds.
    """
    mu_max = b_bar * delta ** (-gamma) / A
    mu_clamped = np.minimum(mu, mu_max)
    inner = (mu_clamped * A / b_bar) ** (-1.0 / gamma) - delta
    return A * np.maximum(inner, 0.0)


# =============================================================================
# solver.py FOC-kernel inline math (for numba @njit kernels — cannot call
# the helpers above directly since they use np.where)
# =============================================================================
#
# Replace the four lines at solver.py:518-523 (retirement) and the analogous
# block at 667-669 (working) with this snippet.  The diagnostic kernels in
# scripts/diagnostics/_diag_euler_errors.py get the same treatment.
#
# OLD:
#     w_A = w_inv / annuity_factor_is
#     mu_bequest = b_bar * w_A ** (-gamma) / annuity_factor_is
#     mup_bequest = -gamma * mu_bequest / (w_A * annuity_factor_is)
#
# NEW (inlined; equivalent to bequest_marginal / its derivative w.r.t. W):

@njit(cache=True, inline='always')
def _shifted_bequest_mu_and_mup(W, A, gamma, b_bar, delta):
    """
    Returns (mu_bequest, dmu/dW) under the luxury-shifted spec.

    For numba kernels — pure scalar arithmetic, no np.where.
    Caller is responsible for the W > 0 / W <= 0 branch (typically via
    the existing min_wealth_inv floor, which keeps W strictly positive).
    """
    # With min_wealth_inv = 1e-10, W is always > 0 in the FOC integrand;
    # the shifter handles the regularisation.  If you remove the floor,
    # add an `if W <= 0: return 0.0, 0.0` guard.
    C_bar = W / A + delta                          # shifted "consumption" arg
    mu = b_bar * C_bar ** (-gamma) / A             # bounded by b_bar*delta^-gamma/A
    # d/dW [b_bar * (W/A + delta)^(-gamma) / A]  =  -gamma * mu / (A * C_bar)
    mup = -gamma * mu / (A * C_bar)
    return mu, mup


# Usage at the FOC-kernel call site (mirrors the existing pattern):
#
#     w_inv = max(s_val * R_p, min_wealth_inv)
#     # ... existing consumption-side interpolation ...
#     mu_alive  = c_next ** (-gamma)
#     mu_bequest, mup_bequest = _shifted_bequest_mu_and_mup(
#         w_inv, annuity_factor_is, gamma, b_bar, DELTA_BEQUEST,
#     )
#     mu_comb  = psi * mu_alive + prob_death * mu_bequest
#     mup_alive = -gamma * mu_alive / c_next * mpc
#     mup_comb = psi * mup_alive + prob_death * mup_bequest
#
# Everything downstream (foc_s, foc_b, J_ss, J_bb, J_sb, euler_sum) is unchanged.


# =============================================================================
# Sanity check — verify the bound holds and matches the proposal's numbers
# =============================================================================

if __name__ == "__main__":
    b_bar, gamma = 10, 5
    A = 4.0
    for delta in [0.001, 0.005, 0.01, 0.02]:
        mu_max = b_bar * delta ** (-gamma) / A
        # Compare to the spike contribution at the worst cell:
        # weight = 1.12e-11, R_p_node = 8e-7 (since s*R_p = 7.7e-6, s = 9.67)
        per_node = 1.12e-11 * mu_max * 8e-7
        # Asymptotic agreement with pure CRRA at W/A = 1 and W/A = 10:
        agree_1  = (1.0 + delta) ** (-gamma) / 1.0 ** (-gamma)
        agree_10 = (10.0 + delta) ** (-gamma) / 10.0 ** (-gamma)
        print(f"delta = {delta:6.3f}:  mu_max = {mu_max:9.2e},  "
              f"per-node bequest contrib at worst cell <= {per_node:.2e},  "
              f"CRRA agreement: {(1-agree_1)*100:5.2f}% @ W/A=1, "
              f"{(1-agree_10)*100:6.4f}% @ W/A=10")

    # Continuity check at W = 0:
    W_grid = np.array([-0.1, -1e-6, 0.0, 1e-6, 0.1])
    print("\nContinuity of b(W) at W=0:")
    for w in W_grid:
        b = bequest_utility(np.array([w]), A, gamma, b_bar)[0]
        mu = bequest_marginal(np.array([w]), A, gamma, b_bar)[0]
        print(f"  W = {w:+8.1e}:  b = {b:+12.3e},  mu = {mu:+12.3e}")

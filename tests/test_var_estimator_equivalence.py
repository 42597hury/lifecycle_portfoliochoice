"""Verify OLS = SUR/GLS = constrained-LS for our restricted VAR setup.

The CCV w8566 §2.2.r restriction zeros out the same set of regressors in EVERY
equation (the lagged-x columns of Phi). So all 6 equations have an identical
regressor matrix X. By Zellner's "irrelevance" result, when all SUR equations
share the same regressors, the SUR/GLS estimator collapses to equation-by-
equation OLS.

This test pins that equivalence numerically on our actual estimated VAR. Three
estimators that should agree:

  M1: equation-by-equation OLS with state-only regressors  (current code)
  M2: feasible GLS / SUR using a residual-derived Sigma_hat as weight
  M3: unrestricted full-X OLS with the §2.2.r restriction imposed via a
       projection (equivalent to Lagrange multipliers on R*vec(Phi)=0)

All three must produce identical Phi to machine precision, and the same
Omega up to the dof correction.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CSV = REPO / "data" / "var_dataset.csv"
COLUMNS = ["cape", "spr", "y_1", "xr", "xb"]
STATE_COLS = [0, 1, 2]       # cape, spr, y_1 (state predictors; real-yields pivot)
RET_COLS = [3, 4]            # xr, xb (zeroed by §2.2.r)


def load_data():
    import pandas as pd
    df = pd.read_csv(CSV)
    return df[COLUMNS].to_numpy(dtype=float)


def estimate_M1_ols_reduced_X(data):
    """Equation-by-equation OLS with reduced regressors (current code path)."""
    z_bar = data.mean(axis=0)
    Z = data - z_bar
    Y = Z[1:, :]                          # (T-1, n)
    X = Z[:-1, STATE_COLS]                # (T-1, k_state)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (k_state, n)

    n = data.shape[1]
    Phi = np.zeros((n, n))
    for k, j in enumerate(STATE_COLS):
        Phi[:, j] = coeffs[k, :]

    Y_hat = X @ coeffs
    resid = Y - Y_hat
    dof = Y.shape[0] - X.shape[1]
    Omega = (resid.T @ resid) / dof
    return Phi, Omega, resid, dof


def estimate_M2_fgls_sur(data, n_iter=5):
    """Feasible GLS / SUR with restricted regressors.

    Theory: when all equations share the same regressor matrix X, the GLS
    estimator equals equation-by-equation OLS for ANY positive-definite
    weighting Sigma. Iteration just refines Sigma; Phi is the same after
    every step.
    """
    z_bar = data.mean(axis=0)
    Z = data - z_bar
    Y = Z[1:, :]                          # (T-1, n)
    X = Z[:-1, STATE_COLS]                # (T-1, k_state)
    T, n = Y.shape
    k = X.shape[1]

    # Initialise Sigma with any PD matrix; OLS-residual estimate is fine
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coeffs
    Sigma = (resid.T @ resid) / (T - k)

    # Stack equations: vec(Y) is (n*T, 1), full design (I_n kron X) is (n*T, n*k)
    # GLS: vec(B) = ((I⊗X)' (Σ⁻¹⊗I_T) (I⊗X))⁻¹ (I⊗X)' (Σ⁻¹⊗I_T) vec(Y)
    #             = (Σ⁻¹ ⊗ X'X)⁻¹ (Σ⁻¹ ⊗ X') vec(Y)
    # Implement explicitly (don't use the algebraic shortcut) — this is the
    # actual numerical comparison.
    coeffs_history = []
    for it in range(n_iter):
        Sigma_inv = np.linalg.inv(Sigma)
        # Construct kron design matrix for the GLS normal equations
        # Use the algebraic identity: (Σ⁻¹ ⊗ X'X)⁻¹ (Σ⁻¹ ⊗ X') vec(Y)
        # but build it the long way without simplification
        XtX = X.T @ X
        # Left side: kron(Sigma_inv, XtX). Shape (n*k, n*k).
        L = np.kron(Sigma_inv, XtX)
        # Right side: kron(Sigma_inv, X.T) @ vec(Y). vec(Y) stacks columns of Y.
        RHS = np.kron(Sigma_inv, X.T) @ Y.flatten(order="F")
        beta_vec = np.linalg.solve(L, RHS)
        beta = beta_vec.reshape((k, n), order="F")
        coeffs_history.append(beta.copy())
        resid = Y - X @ beta
        Sigma = (resid.T @ resid) / (T - k)

    Phi = np.zeros((n, n))
    for k_idx, j in enumerate(STATE_COLS):
        Phi[:, j] = beta[k_idx, :]

    Omega = Sigma
    return Phi, Omega, coeffs_history


def estimate_M3_constrained_full(data):
    """Full-X unrestricted OLS, then apply restriction via the constraint
    R*vec(Phi) = 0.

    Method: estimate unrestricted Phi_full by OLS on all 6 lagged columns.
    The §2.2.r restriction sets the columns of Phi corresponding to xr, xb
    to zero. With all equations sharing X, the constrained OLS solution is
    equivalent to dropping those columns from X. This script verifies that
    by computing the constrained estimator via the standard restricted-LS
    formula
        beta_c = beta_u - (X'X)⁻¹ R' [R(X'X)⁻¹R']⁻¹ R beta_u
    and comparing to the reduced-X solution.
    """
    z_bar = data.mean(axis=0)
    Z = data - z_bar
    Y = Z[1:, :]
    X_full = Z[:-1, :]                     # ALL 6 lagged columns
    T, n = Y.shape

    # Unrestricted: full design
    Phi_unrestr_T, _, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)  # (n, n)
    # Phi_unrestr_T is (regressors, equations) so transpose for (n, n) Phi format
    # In our usual convention: Phi[i,j] = coeff of lagged z_j in eq for z_i.
    # lstsq returns coeffs[k, j] = coeff of regressor k in equation j. With
    # full X (regressor k = lagged z_k), this gives Phi[j, k] = coeffs[k, j],
    # i.e. Phi = coeffs.T.
    Phi_unrestr = Phi_unrestr_T.T

    # Apply restriction column-wise: for each return column ret_j, the corresponding
    # column of Phi must be zero. Equivalent to imposing on each equation that
    # the kth regressor's coefficient is zero, where k is the ret-col index.
    XtX = X_full.T @ X_full
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_full.T @ Y                     # (n, n)
    beta_unr = XtX_inv @ XtY               # (n, n) — same as Phi_unrestr_T

    # Build the restriction matrix R. We want beta[k, :] = 0 for k in RET_COLS
    # for ALL equations simultaneously. Equivalently, in the per-equation
    # restricted-LS formula, the same k's must be zeroed in every eq.
    # Equation j: R_j @ beta_j = 0, with R_j = e_k1' stacked for k1 in RET_COLS.
    R = np.zeros((len(RET_COLS), n))
    for r_idx, ret_col in enumerate(RET_COLS):
        R[r_idx, ret_col] = 1.0
    # For each equation independently:
    Phi_constr_T = np.zeros((n, n))
    inv_term = np.linalg.inv(R @ XtX_inv @ R.T)
    for j in range(n):
        beta_u = beta_unr[:, j]                   # (n,)
        adj = XtX_inv @ R.T @ inv_term @ (R @ beta_u)
        Phi_constr_T[:, j] = beta_u - adj
    Phi_constr = Phi_constr_T.T

    # Build full Phi (n, n) from the constrained solution
    Phi = Phi_constr

    # Sanity: the constrained Phi should have exactly zero in the ret cols
    # under our R restriction.
    Y_hat = X_full @ Phi.T
    resid = Y - Y_hat
    dof = T - len(STATE_COLS)              # effective k after the restriction
    Omega = (resid.T @ resid) / dof
    return Phi, Omega, resid, dof


# =============================================================================
# Tests
# =============================================================================

def test_m1_m2_phi_equal():
    """Equation-by-equation OLS Phi equals FGLS/SUR Phi to machine precision."""
    data = load_data()
    Phi_M1, _, _, _ = estimate_M1_ols_reduced_X(data)
    Phi_M2, _, hist = estimate_M2_fgls_sur(data, n_iter=5)
    np.testing.assert_allclose(Phi_M1, Phi_M2, atol=1e-10, rtol=0,
                               err_msg="M1 (OLS) and M2 (FGLS/SUR) Phi differ — "
                                       "Zellner irrelevance violated")
    # Iteration should also leave coefficients invariant (every iter gives the
    # same answer when X is common across equations).
    for i in range(1, len(hist)):
        np.testing.assert_allclose(hist[0], hist[i], atol=1e-10, rtol=0,
                                   err_msg=f"FGLS iteration {i} changed coeffs — "
                                           "common-X invariance violated")


def test_m1_m3_phi_equal():
    """Reduced-X OLS Phi equals constrained-LS-on-full-X Phi to machine precision."""
    data = load_data()
    Phi_M1, _, _, _ = estimate_M1_ols_reduced_X(data)
    Phi_M3, _, _, _ = estimate_M3_constrained_full(data)
    np.testing.assert_allclose(Phi_M1, Phi_M3, atol=1e-10, rtol=0,
                               err_msg="Reduced-X OLS and Lagrange-constrained "
                                       "full-X OLS Phi differ — restriction "
                                       "implementation inconsistency")


def test_m1_m2_omega_equal():
    """OLS and FGLS/SUR residual covariance match (same residuals, same dof)."""
    data = load_data()
    _, Omega_M1, _, _ = estimate_M1_ols_reduced_X(data)
    _, Omega_M2, _ = estimate_M2_fgls_sur(data, n_iter=5)
    np.testing.assert_allclose(Omega_M1, Omega_M2, atol=1e-10, rtol=0,
                               err_msg="M1 and M2 Omega differ")


def test_m1_m3_omega_equal():
    """OLS and constrained-LS Omega match.

    Both use dof = T_eff - k_state. If you instead used T_eff - k_full = T_eff - 6,
    you'd get a different Omega — but since the §2.2.r restriction zeros out 12
    coefficients (2 ret cols × 6 eqs), the *effective* number of estimated
    parameters per equation is k_state = 4, so T_eff - k_state is the right dof.
    """
    data = load_data()
    _, Omega_M1, _, _ = estimate_M1_ols_reduced_X(data)
    _, Omega_M3, _, _ = estimate_M3_constrained_full(data)
    np.testing.assert_allclose(Omega_M1, Omega_M3, atol=1e-10, rtol=0,
                               err_msg="M1 and M3 Omega differ")


def test_restriction_holds_exactly_in_all_three():
    """The §2.2.r zeros must be exact in all three estimators."""
    data = load_data()
    Phi_M1, _, _, _ = estimate_M1_ols_reduced_X(data)
    Phi_M2, _, _ = estimate_M2_fgls_sur(data, n_iter=5)
    Phi_M3, _, _, _ = estimate_M3_constrained_full(data)
    for nm, Phi in [("M1", Phi_M1), ("M2", Phi_M2), ("M3", Phi_M3)]:
        norm = float(np.linalg.norm(Phi[:, RET_COLS]))
        assert norm < 1e-12, f"{nm} did not zero ret cols of Phi (norm={norm:.3e})"


def test_against_production_estimator():
    """The production estimator (lifecycle.var.estimate_var1_from_csv) must
    match these reference implementations bit-for-bit."""
    from lifecycle.var import estimate_var1_from_csv
    var_core, _, _ = estimate_var1_from_csv(
        csv_path=str(CSV), columns=COLUMNS, state_indices=STATE_COLS,
    )
    Phi_prod = var_core["Phi"]
    Omega_prod = var_core["Omega"]

    data = load_data()
    Phi_M1, Omega_M1, _, _ = estimate_M1_ols_reduced_X(data)
    np.testing.assert_allclose(Phi_prod, Phi_M1, atol=1e-10, rtol=0,
                               err_msg="Production estimator disagrees with "
                                       "reference reduced-X OLS")
    np.testing.assert_allclose(Omega_prod, Omega_M1, atol=1e-10, rtol=0,
                               err_msg="Production Omega disagrees with reference")


if __name__ == "__main__":
    # Manual run with a printed report
    data = load_data()
    Phi_M1, Omega_M1, _, dof_M1 = estimate_M1_ols_reduced_X(data)
    Phi_M2, Omega_M2, hist = estimate_M2_fgls_sur(data, n_iter=5)
    Phi_M3, Omega_M3, _, dof_M3 = estimate_M3_constrained_full(data)

    np.set_printoptions(precision=8, linewidth=140, suppress=False)

    print("=" * 78)
    print("OLS = SUR/GLS = Constrained-LS  EQUIVALENCE TEST")
    print("=" * 78)
    print(f"  Sample: T = {data.shape[0]} obs, n_eqs = 6, k_state = 4")
    print(f"  M1 dof = {dof_M1};  M3 dof = {dof_M3}")
    print()
    print(f"  max |Phi_M1 - Phi_M2|  = {np.max(np.abs(Phi_M1 - Phi_M2)):.3e}")
    print(f"  max |Phi_M1 - Phi_M3|  = {np.max(np.abs(Phi_M1 - Phi_M3)):.3e}")
    print(f"  max |Omega_M1 - Omega_M2| = {np.max(np.abs(Omega_M1 - Omega_M2)):.3e}")
    print(f"  max |Omega_M1 - Omega_M3| = {np.max(np.abs(Omega_M1 - Omega_M3)):.3e}")
    print()
    print(f"  ||Phi_M1[:, ret_cols]||  = {np.linalg.norm(Phi_M1[:, RET_COLS]):.3e}")
    print(f"  ||Phi_M2[:, ret_cols]||  = {np.linalg.norm(Phi_M2[:, RET_COLS]):.3e}")
    print(f"  ||Phi_M3[:, ret_cols]||  = {np.linalg.norm(Phi_M3[:, RET_COLS]):.3e}")
    print()
    print(f"  FGLS coefficient stability across iterations:")
    for i in range(1, len(hist)):
        delta = float(np.max(np.abs(hist[0] - hist[i])))
        print(f"    iter 0 vs iter {i}: max delta = {delta:.3e}")
    print()
    print("=" * 78)

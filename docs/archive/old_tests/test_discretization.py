"""
test_discretization.py — Comprehensive tests for Farmer-Toda discretization.

Tier 1: Standalone discretization tests (no model build)
Tier 2: Integration tests (full Precompute build)
Tier 3: Comparison tests (Farmer-Toda vs independence)

Usage:
    python test_discretization.py              # Tier 1 + Tier 3
    python test_discretization.py --integration  # Tier 1 + Tier 2 + Tier 3
"""

import sys
import warnings
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_discrete_lyapunov
from discretization import rouwenhorst_multivariate

# =========================================================================
# Reference VAR parameters (from var.py / handoff §3)
# =========================================================================

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
Phi_0_state = (np.eye(3) - Phi_11) @ z_bar_state
Sigma_chol = np.linalg.cholesky(Sigma_ss)
Sigma_z_true = solve_discrete_lyapunov(Phi_11, Sigma_ss)

d = 3


def build_flat_grid(grids, state_indices):
    N_total = state_indices.shape[0]
    d = len(grids)
    sg = np.empty((N_total, d))
    for i in range(N_total):
        for dd in range(d):
            sg[i, dd] = grids[dd][state_indices[i, dd]]
    return sg


def compute_stationary(Pi):
    eigvals, eigvecs = np.linalg.eig(Pi.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi_star = np.real(eigvecs[:, idx])
    pi_star /= pi_star.sum()
    return pi_star


def run_tier1(N_vec, grid_scale, label):
    """Run Tier 1 tests for a given configuration. Returns (pass_count, fail_count)."""
    N_total = int(np.prod(N_vec))
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Tier 1: Standalone — {label} (N={N_vec}, scale={grid_scale})")
    print(f"{'='*60}")

    # Build discretization and count fallbacks
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grids, Pi, si = rouwenhorst_multivariate(
            N_vec, Phi_0_state, Phi_11, Sigma_chol,
            method="farmer_toda", grid_scale=grid_scale
        )
        n_fallback = sum(1 for wi in w if "mean-only fallback" in str(wi.message)
                         and "did not converge" in str(wi.message))

    sg = build_flat_grid(grids, si)

    # T1: Row stochasticity
    row_sums = Pi.sum(axis=1)
    check("T1 row sums",
          np.allclose(row_sums, 1.0, atol=1e-12),
          f"max deviation = {np.max(np.abs(row_sums - 1.0)):.2e}")
    check("T1 non-negative",
          (Pi >= -1e-15).all(),
          f"min prob = {Pi.min():.2e}")

    # T2: Conditional mean
    max_mean_err = 0.0
    for i in range(N_total):
        computed = Pi[i, :] @ sg
        target = Phi_0_state + Phi_11 @ sg[i]
        err = np.max(np.abs(computed - target))
        max_mean_err = max(max_mean_err, err)
    check("T2 conditional mean",
          max_mean_err < 1e-6,
          f"max error = {max_mean_err:.2e}")

    # T3: Conditional covariance
    n_good_cov = 0
    n_bad_cov = 0
    for i in range(N_total):
        mu_i = Phi_0_state + Phi_11 @ sg[i]
        Y = sg - mu_i
        cond_cov = (Pi[i, :, None] * Y).T @ Y
        rel_err = norm(cond_cov - Sigma_ss, 'fro') / norm(Sigma_ss, 'fro')
        if rel_err < 1e-6:
            n_good_cov += 1
        if rel_err > 0.1:
            n_bad_cov += 1

    if N_total == 343:  # N=7
        check("T3 conditional cov (good)",
              n_good_cov >= 340,
              f"{n_good_cov}/343 states with rel_error < 1e-6")
        check("T3 conditional cov (bad)",
              n_bad_cov <= 3,
              f"{n_bad_cov}/343 states with rel_error > 0.1")
    else:
        check("T3 conditional cov (good)",
              n_good_cov >= N_total - 8,
              f"{n_good_cov}/{N_total} states with rel_error < 1e-6")

    # T4: Stationary mean
    pi_star = compute_stationary(Pi)
    stat_mean = pi_star @ sg
    mean_err = np.max(np.abs(stat_mean - z_bar_state))
    tol_mean = 1e-6 if N_total == 343 else 1e-4
    check("T4 stationary mean",
          mean_err < tol_mean,
          f"max error = {mean_err:.2e}")

    # T5: Stationary covariance
    stat_cov = sum(pi_star[j] * np.outer(sg[j] - stat_mean, sg[j] - stat_mean)
                   for j in range(N_total))
    cov_rel_err = norm(stat_cov - Sigma_z_true, 'fro') / norm(Sigma_z_true, 'fro')
    tol_cov = 1e-3 if N_total == 343 else 0.02
    check("T5 stationary cov",
          cov_rel_err < tol_cov,
          f"rel Frobenius error = {cov_rel_err:.4e}")

    # T6: Stationary correlations
    D_stat = np.diag(1.0 / np.sqrt(np.diag(stat_cov)))
    corr_stat = D_stat @ stat_cov @ D_stat
    D_z = np.diag(1.0 / np.sqrt(np.diag(Sigma_z_true)))
    corr_z = D_z @ Sigma_z_true @ D_z
    tol_corr = 0.002 if N_total == 343 else 0.01
    max_corr_err = 0.0
    for a in range(d):
        for b in range(a+1, d):
            max_corr_err = max(max_corr_err, abs(corr_stat[a,b] - corr_z[a,b]))
    check("T6 stationary correlations",
          max_corr_err < tol_corr,
          f"max corr error = {max_corr_err:.4e}")

    # T7: Fallback count
    tol_fb = 3 if N_total == 343 else 8
    check("T7 fallback count",
          n_fallback <= tol_fb,
          f"{n_fallback} fallbacks (max {tol_fb})")

    return passed, failed, grids, Pi, si


def run_tier2():
    """Tier 2: Integration tests with full Precompute build."""
    print(f"\n{'='*60}")
    print("Tier 2: Integration — full Precompute build")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    from pathlib import Path
    from precompute import Precompute, build_model
    from model import DiscretizationConfig
    from var import build_nominal_system1_var_config

    # T8: Build succeeds
    try:
        # Locate VAR dataset
        _path_candidates = [Path("data/var_dataset.csv"), Path("../data/var_dataset.csv")]
        _VAR_CSV = next((p for p in _path_candidates if p.exists()), None)
        if _VAR_CSV is None:
            check("T8 Precompute builds", False, "var_dataset.csv not found")
            return passed, failed

        var_config, _, _ = build_nominal_system1_var_config(csv_path=str(_VAR_CSV))

        base_config = {
            "beta": 0.96, "gamma": 3.0, "b_bar": 10,
            "start_age": 22, "retire_age": 67, "terminal_age": 99,
            "b0": -6.142, "b1": 0.3040, "b2": -0.051, "b3": 0.002586,
            "rho": 0.991, "pz": 0.176,
            "mu_eta1": -0.524, "sigma_eta1": 0.113,
            "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524), "sigma_eta2": 0.046,
            "pe": 0.044,
            "mu_eps1": 0.134, "sigma_eps1": 0.762,
            "mu_eps2": 0.0, "sigma_eps2": 0.055,
            "constrained": True,
        }

        model = build_model(base_config, var_config, verbose=False)
        disc_config = DiscretizationConfig(
            state_grid_sizes=(7, 7, 7),
            state_discretization_method="farmer_toda",
        )
        pc = Precompute(model, disc_config=disc_config, verbose=False)
        check("T8 Precompute builds", True)
    except Exception as e:
        check("T8 Precompute builds", False, str(e))
        return passed, failed

    # T9: Conditional return consistency
    N = pc.N_state
    errors = np.empty((N, model.n_ret))
    for i in range(N):
        target = model.Phi_0_ret + model.Phi_21 @ pc.state_grid[i]
        avg = pc.Pi_state[i, :] @ pc.mu_r[i, :, :]
        errors[i, :] = np.abs(avg - target)
    max_err = errors.max()
    check("T9 conditional return consistency",
          max_err < 1e-4,
          f"max error = {max_err:.4e}")

    # T10: Bill rate grid
    check("T10 bill rate grid",
          np.allclose(pc.r_bill_grid,
                      pc.state_grid[:, model.bill_rate_index_in_state]))

    # T11: Annuity factors
    ok_shape = pc.annuity_factors.shape == (pc.N_state,)
    ok_pos = np.all(pc.annuity_factors > 0)
    ok_fin = np.all(np.isfinite(pc.annuity_factors))
    check("T11 annuity factors",
          ok_shape and ok_pos and ok_fin,
          f"shape={pc.annuity_factors.shape}, pos={ok_pos}, fin={ok_fin}")

    # T12: mu_r shape and finiteness
    ok_shape = pc.mu_r.shape == (pc.N_state, pc.N_state, model.n_ret)
    ok_fin = np.all(np.isfinite(pc.mu_r))
    check("T12 mu_r shape/finiteness",
          ok_shape and ok_fin,
          f"shape={pc.mu_r.shape}, finite={ok_fin}")

    return passed, failed


def run_tier3():
    """Tier 3: Comparison — Farmer-Toda vs independence."""
    print(f"\n{'='*60}")
    print("Tier 3: Comparison — Farmer-Toda vs Independence")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}  {detail}")
            failed += 1

    N_vec = [7, 7, 7]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grids_ft, Pi_ft, si_ft = rouwenhorst_multivariate(
            N_vec, Phi_0_state, Phi_11, Sigma_chol, method="farmer_toda")
    grids_ind, Pi_ind, si_ind = rouwenhorst_multivariate(
        N_vec, Phi_0_state, Phi_11, Sigma_chol, method="independent")

    sg_ft = build_flat_grid(grids_ft, si_ft)
    sg_ind = build_flat_grid(grids_ind, si_ind)
    N = sg_ft.shape[0]

    # Stationary correlations — independence
    pi_ind = compute_stationary(Pi_ind)
    mean_ind = pi_ind @ sg_ind
    cov_ind = sum(pi_ind[j] * np.outer(sg_ind[j] - mean_ind, sg_ind[j] - mean_ind)
                  for j in range(N))
    D_ind = np.diag(1.0 / np.sqrt(np.diag(cov_ind)))
    corr_ind = D_ind @ cov_ind @ D_ind

    # Stationary correlations — Farmer-Toda
    pi_ft = compute_stationary(Pi_ft)
    mean_ft = pi_ft @ sg_ft
    cov_ft = sum(pi_ft[j] * np.outer(sg_ft[j] - mean_ft, sg_ft[j] - mean_ft)
                 for j in range(N))
    D_ft = np.diag(1.0 / np.sqrt(np.diag(cov_ft)))
    corr_ft = D_ft @ cov_ft @ D_ft

    # T13: Cross-correlations
    check("T13 indep corr(rtb,y_nom) ~ 0",
          abs(corr_ind[0,1]) < 0.05,
          f"got {corr_ind[0,1]:.4f}")
    check("T13 indep corr(rtb,dp) ~ 0",
          abs(corr_ind[0,2]) < 0.05,
          f"got {corr_ind[0,2]:.4f}")
    check("T13 indep corr(y_nom,dp) ~ 0",
          abs(corr_ind[1,2]) < 0.05,
          f"got {corr_ind[1,2]:.4f}")

    check("T13 FT corr(rtb,y_nom) ~ 0.497",
          abs(corr_ft[0,1] - 0.497) < 0.01,
          f"got {corr_ft[0,1]:.4f}")
    check("T13 FT corr(rtb,dp) ~ 0.246",
          abs(corr_ft[0,2] - 0.246) < 0.01,
          f"got {corr_ft[0,2]:.4f}")
    check("T13 FT corr(y_nom,dp) ~ 0.533",
          abs(corr_ft[1,2] - 0.533) < 0.01,
          f"got {corr_ft[1,2]:.4f}")

    # T14: Conditional mean error comparison
    max_err_ind = max(
        np.max(np.abs(Pi_ind[i] @ sg_ind - (Phi_0_state + Phi_11 @ sg_ind[i])))
        for i in range(N))
    max_err_ft = max(
        np.max(np.abs(Pi_ft[i] @ sg_ft - (Phi_0_state + Phi_11 @ sg_ft[i])))
        for i in range(N))
    check("T14 FT cond mean >> better than indep",
          max_err_ft < max_err_ind * 1e-4,
          f"FT={max_err_ft:.2e}, indep={max_err_ind:.2e}, "
          f"ratio={max_err_ft/max(max_err_ind,1e-300):.2e}")

    return passed, failed


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    run_integration = "--integration" in sys.argv

    total_pass = 0
    total_fail = 0

    # Tier 1: N=7
    p, f, *_ = run_tier1([7,7,7], 1.0, "N=7 production")
    total_pass += p; total_fail += f

    # Tier 1: N=5
    p, f, *_ = run_tier1([5,5,5], 0.85, "N=5 debug")
    total_pass += p; total_fail += f

    # Tier 2: Integration (optional)
    if run_integration:
        p, f = run_tier2()
        total_pass += p; total_fail += f

    # Tier 3: Comparison
    p, f = run_tier3()
    total_pass += p; total_fail += f

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    print(f"{'='*60}")
    sys.exit(1 if total_fail > 0 else 0)

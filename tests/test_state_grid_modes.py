"""Validation script for the mode-aware state-grid implementation.

This is intentionally runnable as a plain script so it works even when
`pytest` is not installed in the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discretization import build_state_grid, stationary_covariance
from model import DiscretizationConfig, SolverConfig
from precompute import Precompute, build_model
from simulation import simulate_lifecycle
from solver import run_lifecycle_solver
from var import build_nominal_system1_var_config


N_PASS = 0
N_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global N_PASS, N_FAIL
    if condition:
        N_PASS += 1
        print(f"  PASS  {name}")
    else:
        N_FAIL += 1
        print(f"  FAIL  {name}  -- {detail}")


def reference_base_config() -> dict:
    return {
        "beta": 0.96,
        "gamma": 3.0,
        "b_bar": 10,
        "start_age": 22,
        "retire_age": 67,
        "terminal_age": 99,
        "b0": -6.142,
        "b1": 0.3040,
        "b2": -0.051,
        "b3": 0.002586,
        "rho": 0.991,
        "pz": 0.176,
        "mu_eta1": -0.524,
        "sigma_eta1": 0.113,
        "mu_eta2": -(0.176 / (1.0 - 0.176)) * (-0.524),
        "sigma_eta2": 0.046,
        "pe": 0.044,
        "mu_eps1": 0.134,
        "sigma_eps1": 0.762,
        "mu_eps2": 0.0,
        "sigma_eps2": 0.055,
        "constrained": True,
    }


def build_reference_model():
    var_config, _, data = build_nominal_system1_var_config(csv_path=str(ROOT / "data" / "var_dataset.csv"))
    model = build_model(reference_base_config(), var_config, verbose=False)
    return model, data


def _trilinear_interp_principal(u_point, u_grids, state_grid, coeff):
    N1 = len(u_grids[1])
    N2 = len(u_grids[2])
    lo = []
    frac = []
    for d in range(3):
        g = u_grids[d]
        idx = max(0, min(np.searchsorted(g, u_point[d]) - 1, len(g) - 2))
        lo.append(idx)
        frac.append(max(0.0, min(1.0, (u_point[d] - g[idx]) / (g[idx + 1] - g[idx]))))

    out = 0.0
    for d0 in range(2):
        for d1 in range(2):
            for d2 in range(2):
                w = ((1.0 - frac[0]) if d0 == 0 else frac[0]) * \
                    ((1.0 - frac[1]) if d1 == 0 else frac[1]) * \
                    ((1.0 - frac[2]) if d2 == 0 else frac[2])
                j = (lo[0] + d0) * N1 * N2 + (lo[1] + d1) * N2 + (lo[2] + d2)
                out += w * (coeff[0] + coeff[1:] @ state_grid[j])
    return out


def run_geometry_checks(model, data):
    print("\n" + "=" * 70)
    print("GEOMETRY CHECKS")
    print("=" * 70)

    Phi = np.asarray(model.Phi_11, dtype=float)
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=float)
    mu_s = np.asarray(model.z_bar_state, dtype=float)
    Sigma_z = stationary_covariance(Phi, Sigma_ss)
    sigma_z = np.sqrt(np.diag(Sigma_z))

    principal = build_state_grid(
        N_vec=(7, 7, 7),
        mu_intercept=model.Phi_0_state,
        Phi=model.Phi_11,
        Sigma_innov=model.Sigma_ss,
        n_stds=3.0,
        mode="principal",
    )
    ly_axis = build_state_grid(
        N_vec=(7, 7, 7),
        mu_intercept=model.Phi_0_state,
        Phi=model.Phi_11,
        Sigma_innov=model.Sigma_ss,
        n_stds=3.0,
        mode="lyapunov-axis",
    )

    residual = Sigma_z - Phi @ Sigma_z @ Phi.T - Sigma_ss
    check("Lyapunov residual", np.max(np.abs(residual)) < 1e-12, f"max err = {np.max(np.abs(residual)):.2e}")

    center_flat = 3 * 7 * 7 + 3 * 7 + 3
    check(
        "Principal center point equals mu_s",
        np.allclose(principal["state_grid"][center_flat], principal["mu_s"], atol=1e-12),
        f"center = {principal['state_grid'][center_flat]}, mu = {principal['mu_s']}",
    )

    U = (principal["state_grid"] - principal["mu_s"]) @ principal["L_inv"].T
    check("Principal u-min exact", np.isclose(U.min(), -3.0, atol=1e-12), f"min = {U.min():.6f}")
    check("Principal u-max exact", np.isclose(U.max(), 3.0, atol=1e-12), f"max = {U.max():.6f}")

    flat_ok = True
    N1 = 7
    N2 = 7
    for i, idx in enumerate(principal["state_indices"]):
        if i != idx[0] * N1 * N2 + idx[1] * N2 + idx[2]:
            flat_ok = False
            break
    check("Flat index ordering preserved", flat_ok)

    # Read historical columns in MODEL order so column-by-column comparisons
    # against mu_s, sigma_z, L are well-defined regardless of state_indices.
    state_cols = list(model.state_names)
    hist = data[state_cols].to_numpy()
    inside_principal = np.mean(np.all(np.abs((hist - principal["mu_s"]) @ principal["L_inv"].T) <= 3.0, axis=1))
    inside_axis = np.mean(np.all(np.abs(hist - mu_s) <= 3.0 * sigma_z, axis=1))
    check("Principal historical coverage >= 99%", inside_principal >= 0.99, f"{inside_principal:.3f}")
    check("Lyapunov-axis historical coverage >= 99%", inside_axis >= 0.99, f"{inside_axis:.3f}")

    vol_axis = np.prod(2.0 * 3.0 * sigma_z)
    vol_principal = (6.0 ** 3) * abs(np.linalg.det(principal["L"]))
    ratio = vol_principal / vol_axis
    check("Principal volume ratio in expected range", 0.3 < ratio < 0.6, f"ratio = {ratio:.3f}")

    rng = np.random.default_rng(123)
    coeff = np.array([1.234, 0.3, -0.5, 0.1])
    max_err = 0.0
    for _ in range(200):
        u_point = rng.uniform(-3.0, 3.0, size=3)
        s_point = principal["mu_s"] + principal["L"] @ u_point
        exact = coeff[0] + coeff[1:] @ s_point
        interp = _trilinear_interp_principal(u_point, principal["state_bracket_grids"], principal["state_grid"], coeff)
        max_err = max(max_err, abs(interp - exact))
    check("Principal trilinear exact for linear functions", max_err < 1e-12, f"max err = {max_err:.2e}")

    check(
        "Principal stationary probs sum to one",
        abs(principal["stationary_probs"].sum() - 1.0) < 1e-14,
        f"sum = {principal['stationary_probs'].sum():.16f}",
    )
    check(
        "Lyapunov-axis stationary probs sum to one",
        abs(ly_axis["stationary_probs"].sum() - 1.0) < 1e-12,
        f"sum = {ly_axis['stationary_probs'].sum():.16f}",
    )


def run_precompute_mode_checks(model):
    print("\n" + "=" * 70)
    print("PRECOMPUTE MODE CHECKS")
    print("=" * 70)

    for mode in ("naive", "lyapunov-axis", "principal"):
        disc = DiscretizationConfig(
            n_wealth=20,
            n_savings=20,
            state_grid_sizes=(5, 5, 5),
            state_grid_mode=mode,
            state_n_stds=3.0,
            n_z=5,
            n_eps_nodes=2,
            n_eta_nodes=2,
            n_ret_nodes_1d=2,
            n_state_quad_nodes=2,
        )
        pc = Precompute(model, disc, verbose=False)
        check(f"{mode}: state_grid_mode stored", pc.state_grid_mode == mode)
        check(f"{mode}: stationary probs sum", abs(pc.state_stationary_probs.sum() - 1.0) < 1e-12)
        check(f"{mode}: state_grid finite", np.all(np.isfinite(pc.state_grid)))
        check(f"{mode}: annuity factors finite", np.all(np.isfinite(pc.annuity_factors)))
        if mode == "principal":
            check(f"{mode}: nonzero bracket shift", np.max(np.abs(pc.state_bracket_shift)) > 0.0)
            check(f"{mode}: transformed axes centered at 0", abs(pc.state_bracket_grids[0].mean()) < 1e-14)
        else:
            check(f"{mode}: zero bracket shift", np.max(np.abs(pc.state_bracket_shift)) < 1e-14)
            check(f"{mode}: identity bracket transform", np.allclose(pc.state_bracket_L_inv, np.eye(model.n_state)))


def run_solver_simulation_smoke(model):
    print("\n" + "=" * 70)
    print("PRINCIPAL-MODE SOLVER/SIMULATION SMOKE")
    print("=" * 70)

    disc = DiscretizationConfig(
        n_wealth=20,
        n_savings=20,
        state_grid_sizes=(3, 3, 3),
        state_grid_mode="principal",
        state_n_stds=3.0,
        n_z=5,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=2,
        n_state_quad_nodes=2,
    )
    pc = Precompute(model, disc, verbose=False)

    C, S, B, diag = run_lifecycle_solver(model, pc, solver_config=SolverConfig(), verbose=0)
    check("Principal solve: no NaN in C", not np.any(np.isnan(C)))
    check("Principal solve: no NaN in S", not np.any(np.isnan(S)))
    check("Principal solve: no NaN in B", not np.any(np.isnan(B)))
    check("Principal solve: shares feasible", np.all(S >= -1e-8) and np.all(B >= -1e-8) and np.all(S + B <= 1.0 + 1e-6))
    fail_rate = diag["total_newton_failures"] / max(diag["total_calls"], 1)
    check("Principal solve: low Newton failure rate", fail_rate < 0.02, f"{diag['total_newton_failures']} / {diag['total_calls']} = {fail_rate:.3%}")

    sim = simulate_lifecycle(
        C,
        S,
        B,
        pc,
        model,
        n_simulations=200,
        initial_z="stationary",
        initial_state="stationary",
        seed=42,
        verbose=False,
    )
    check("Principal sim: valid state indices", sim["state_idx"].min() >= 0 and sim["state_idx"].max() < pc.N_state)
    check("Principal sim: estate finite", np.all(np.isfinite(sim["estate"])))


def run_per_axis_n_stds_checks(model):
    print("\n" + "=" * 70)
    print("PER-AXIS n_stds CHECKS")
    print("=" * 70)

    # T-A: scalar broadcast equals tuple-of-same — bit-equivalence regression
    for mode in ("principal", "lyapunov-axis"):
        g_scalar = build_state_grid(
            N_vec=(7, 7, 7),
            mu_intercept=model.Phi_0_state,
            Phi=model.Phi_11,
            Sigma_innov=model.Sigma_ss,
            n_stds=2.0,
            mode=mode,
        )
        g_tuple = build_state_grid(
            N_vec=(7, 7, 7),
            mu_intercept=model.Phi_0_state,
            Phi=model.Phi_11,
            Sigma_innov=model.Sigma_ss,
            n_stds=(2.0, 2.0, 2.0),
            mode=mode,
        )
        check(
            f"{mode}: scalar==tuple state_grid",
            np.allclose(g_scalar["state_grid"], g_tuple["state_grid"], atol=0.0, rtol=0.0),
        )
        check(
            f"{mode}: scalar==tuple stationary_probs",
            np.allclose(g_scalar["stationary_probs"], g_tuple["stationary_probs"], atol=0.0, rtol=0.0),
        )
        same_brackets = all(
            np.allclose(a, b, atol=0.0, rtol=0.0)
            for a, b in zip(g_scalar["state_bracket_grids"], g_tuple["state_bracket_grids"])
        )
        check(f"{mode}: scalar==tuple bracket grids", same_brackets)

    # T-B: per-axis bounds in principal mode (u-coords half-widths)
    g = build_state_grid(
        N_vec=(7, 7, 7),
        mu_intercept=model.Phi_0_state,
        Phi=model.Phi_11,
        Sigma_innov=model.Sigma_ss,
        n_stds=(2.0, 1.0, 1.5),
        mode="principal",
    )
    bracks = g["state_bracket_grids"]
    check("principal per-axis: bracket[0] half-width = 2.0", np.isclose(bracks[0].min(), -2.0) and np.isclose(bracks[0].max(), +2.0))
    check("principal per-axis: bracket[1] half-width = 1.0", np.isclose(bracks[1].min(), -1.0) and np.isclose(bracks[1].max(), +1.0))
    check("principal per-axis: bracket[2] half-width = 1.5", np.isclose(bracks[2].min(), -1.5) and np.isclose(bracks[2].max(), +1.5))

    # T-C: per-axis bounds in lyapunov-axis mode (physical-axis half-widths)
    g_la = build_state_grid(
        N_vec=(7, 7, 7),
        mu_intercept=model.Phi_0_state,
        Phi=model.Phi_11,
        Sigma_innov=model.Sigma_ss,
        n_stds=(2.0, 1.0, 1.5),
        mode="lyapunov-axis",
    )
    sig = g_la["sigma_z"]
    mu = g_la["mu_s"]
    bracks_la = g_la["state_bracket_grids"]
    for d, ns_d in enumerate([2.0, 1.0, 1.5]):
        check(
            f"lyapunov-axis per-axis[{d}]: extent matches",
            np.isclose(bracks_la[d].min(), mu[d] - ns_d * sig[d])
            and np.isclose(bracks_la[d].max(), mu[d] + ns_d * sig[d]),
        )

    # T-D: stationary probabilities still sum to 1 with asymmetric bounds
    g_asym = build_state_grid(
        N_vec=(7, 7, 7),
        mu_intercept=model.Phi_0_state,
        Phi=model.Phi_11,
        Sigma_innov=model.Sigma_ss,
        n_stds=(1.5, 2.5, 1.0),
        mode="principal",
    )
    check(
        "principal asym: stationary probs sum to one",
        abs(g_asym["stationary_probs"].sum() - 1.0) < 1e-14,
        f"sum = {g_asym['stationary_probs'].sum():.16f}",
    )

    # T-E: trilinear interpolation still exact-on-linear under asym bounds
    rng = np.random.default_rng(987)
    coeff = np.array([0.7, 0.2, -0.3, 0.15])
    max_err = 0.0
    for _ in range(200):
        u_point = rng.uniform(low=[-1.5, -2.5, -1.0], high=[1.5, 2.5, 1.0], size=3)
        s_point = g_asym["mu_s"] + g_asym["L"] @ u_point
        exact = coeff[0] + coeff[1:] @ s_point
        interp = _trilinear_interp_principal(u_point, g_asym["state_bracket_grids"], g_asym["state_grid"], coeff)
        max_err = max(max_err, abs(interp - exact))
    check("principal asym: trilinear exact-on-linear", max_err < 1e-12, f"max err = {max_err:.2e}")

    # T-F: validation errors
    for bad in [(2.0, -1.0, 2.0), (2.0, 0.0, 2.0)]:
        try:
            build_state_grid(
                N_vec=(7, 7, 7),
                mu_intercept=model.Phi_0_state,
                Phi=model.Phi_11,
                Sigma_innov=model.Sigma_ss,
                n_stds=bad,
                mode="principal",
            )
            check(f"validation: rejects non-positive n_stds {bad}", False, "did not raise")
        except ValueError:
            check(f"validation: rejects non-positive n_stds {bad}", True)
    for bad in [(2.0, 2.0), (2.0, 2.0, 2.0, 2.0)]:
        try:
            build_state_grid(
                N_vec=(7, 7, 7),
                mu_intercept=model.Phi_0_state,
                Phi=model.Phi_11,
                Sigma_innov=model.Sigma_ss,
                n_stds=bad,
                mode="principal",
            )
            check(f"validation: rejects length-mismatch n_stds (len={len(bad)})", False, "did not raise")
        except ValueError:
            check(f"validation: rejects length-mismatch n_stds (len={len(bad)})", True)

    # T-G: end-to-end Precompute round-trip with per-axis bounds
    disc_axis = DiscretizationConfig(
        n_wealth=20, n_savings=20,
        state_grid_sizes=(5, 5, 5),
        state_grid_mode="principal",
        state_n_stds=(2.0, 1.0, 1.5),
        n_z=5,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=2,
        n_state_quad_nodes=2,
    )
    pc_axis = Precompute(model, disc_axis, verbose=False)
    check(
        "precompute per-axis: stationary probs sum to one",
        abs(pc_axis.state_stationary_probs.sum() - 1.0) < 1e-12,
    )
    check(
        "precompute per-axis: state_grid finite",
        np.all(np.isfinite(pc_axis.state_grid)),
    )
    check(
        "precompute per-axis: bracket[1] tighter than bracket[0]",
        (pc_axis.state_bracket_grids[1].max() - pc_axis.state_bracket_grids[1].min())
        < (pc_axis.state_bracket_grids[0].max() - pc_axis.state_bracket_grids[0].min()),
    )


def run_per_axis_n_state_quad_checks(model):
    print("\n" + "=" * 70)
    print("PER-AXIS n_state_quad_nodes CHECKS")
    print("=" * 70)
    from discretization import get_state_quadrature
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=float)

    # T-A: scalar broadcast equals tuple-of-same — bit-equivalence regression
    for K in (2, 3, 5):
        v_s, w_s = get_state_quadrature(model, n_nodes=K)
        v_t, w_t = get_state_quadrature(model, n_nodes=(K, K, K))
        check(
            f"K={K}: scalar==tuple v_nodes",
            np.allclose(v_s, v_t, atol=0.0, rtol=0.0),
        )
        check(
            f"K={K}: scalar==tuple v_weights",
            np.allclose(w_s, w_t, atol=0.0, rtol=0.0),
        )
        check(
            f"K={K}: joint node count matches K^n_state",
            v_s.shape[0] == K ** 3,
            f"got {v_s.shape[0]}, expected {K ** 3}",
        )

    # T-B: per-axis tuple shape — node count is product of per-axis K
    for K_per_axis in [(2, 3, 5), (2, 2, 5), (5, 2, 2), (1, 5, 5), (3, 3, 1)]:
        v, w = get_state_quadrature(model, n_nodes=K_per_axis)
        expected = int(np.prod(K_per_axis))
        check(
            f"per-axis K={K_per_axis}: total nodes = prod = {expected}",
            v.shape[0] == expected,
            f"got {v.shape[0]}",
        )
        check(
            f"per-axis K={K_per_axis}: weights sum to 1",
            abs(float(w.sum()) - 1.0) < 1e-14,
            f"sum = {float(w.sum()):.16f}",
        )

    # T-C: moment recovery preserved with asymmetric K (zero mean, exact cov to 1e-14)
    for K_per_axis in [(2, 2, 5), (2, 5, 2), (5, 2, 2), (3, 3, 5), (5, 3, 3)]:
        v, w = get_state_quadrature(model, n_nodes=K_per_axis)
        mean_v = w @ v
        cov_emp = (v.T * w) @ v
        err_mean = float(np.max(np.abs(mean_v)))
        err_cov = float(np.max(np.abs(cov_emp - Sigma_ss)))
        check(
            f"per-axis K={K_per_axis}: E[v] == 0",
            err_mean < 1e-15,
            f"|mean| = {err_mean:.2e}",
        )
        check(
            f"per-axis K={K_per_axis}: E[v v^T] == Sigma_ss",
            err_cov < 1e-14,
            f"|cov err| = {err_cov:.2e}",
        )

    # T-D: validation rejects bad inputs
    for bad in [(2, 2), (2, 2, 2, 2), (2, 0, 2), (2, -1, 2)]:
        try:
            get_state_quadrature(model, n_nodes=bad)
            check(f"validation: rejects {bad}", False, "did not raise")
        except (ValueError, TypeError):
            check(f"validation: rejects {bad}", True)


def run_per_axis_n_ret_quad_checks(model):
    print("\n" + "=" * 70)
    print("PER-AXIS n_ret_nodes_1d (Cholesky transform) CHECKS")
    print("=" * 70)
    from discretization import get_return_quadrature
    Sigma_r_cond = np.asarray(model.Sigma_r_cond, dtype=float)
    n_ret = int(model.n_ret)

    # T-A: scalar broadcast equals tuple-of-same.  Note: scalar K=1 is the
    # special "zero residual node" path, exact under both code branches.
    for K in (2, 3, 5):
        r_s, w_s = get_return_quadrature(model, n_nodes=K)
        r_t, w_t = get_return_quadrature(model, n_nodes=(K, K, K))
        check(
            f"K={K}: scalar==tuple ret_nodes",
            np.allclose(r_s, r_t, atol=0.0, rtol=0.0),
        )
        check(
            f"K={K}: scalar==tuple ret_weights",
            np.allclose(w_s, w_t, atol=0.0, rtol=0.0),
        )
        check(
            f"K={K}: joint node count = K^n_ret = {K**n_ret}",
            r_s.shape[0] == K ** n_ret,
            f"got {r_s.shape[0]}",
        )

    # T-B: per-axis tuple shape — node count = product of per-axis K
    for K_per_axis in [(3, 5, 3), (2, 2, 5), (5, 2, 2), (1, 5, 5), (3, 3, 1)]:
        r, w = get_return_quadrature(model, n_nodes=K_per_axis)
        expected = int(np.prod(K_per_axis))
        check(
            f"per-axis K={K_per_axis}: total nodes = prod = {expected}",
            r.shape[0] == expected,
            f"got {r.shape[0]}",
        )
        check(
            f"per-axis K={K_per_axis}: weights sum to 1",
            abs(float(w.sum()) - 1.0) < 1e-14,
            f"sum = {float(w.sum()):.16f}",
        )

    # T-C: moment recovery preserved for asymmetric K (zero mean, exact cov)
    for K_per_axis in [(2, 2, 5), (2, 5, 2), (5, 2, 2), (3, 5, 3), (5, 3, 3)]:
        r, w = get_return_quadrature(model, n_nodes=K_per_axis)
        mean_r = w @ r
        cov_emp = (r.T * w) @ r
        err_mean = float(np.max(np.abs(mean_r)))
        err_cov = float(np.max(np.abs(cov_emp - Sigma_r_cond)))
        check(
            f"per-axis K={K_per_axis}: E[r] == 0",
            err_mean < 1e-15,
            f"|mean| = {err_mean:.2e}",
        )
        check(
            f"per-axis K={K_per_axis}: E[r r^T] == Sigma_r_cond",
            err_cov < 1e-14,
            f"|cov err| = {err_cov:.2e}",
        )

    # T-D: Cholesky-specific structural test.  Under Cholesky with input
    # order (rtb, xr, xb), L is lower-triangular, so:
    #   K=(K_high, 1, 1): only z_0 varies. r[0]=L[0,0]*z_0; r[1]=L[1,0]*z_0;
    #                     r[2]=L[2,0]*z_0.  All three return components vary
    #                     proportionally to z_0 (their values are along
    #                     L[:, 0] = first column of L).
    #   K=(1, K_high, 1): z_0=0, z_2=0; only z_1 varies.  r[0] = 0 (because
    #                     L[0,1] = 0 by triangularity), r[1] and r[2] vary.
    #   K=(1, 1, K_high): z_0=z_1=0; only z_2 varies.  r[0] = 0, r[1] = 0,
    #                     r[2] = L[2,2]*z_2.  Pure xb-axis variance.
    L = np.linalg.cholesky(0.5 * (Sigma_r_cond + Sigma_r_cond.T))

    # K=(1, 1, K_high): pure xb axis varies, r[0]==0 and r[1]==0
    r, _ = get_return_quadrature(model, n_nodes=(1, 1, 5))
    check(
        "Cholesky K=(1,1,5): r[:,0] all zero (rtb axis collapsed)",
        np.allclose(r[:, 0], 0.0, atol=1e-15),
        f"max |r[:,0]| = {float(np.max(np.abs(r[:, 0]))):.2e}",
    )
    check(
        "Cholesky K=(1,1,5): r[:,1] all zero (xr axis collapsed)",
        np.allclose(r[:, 1], 0.0, atol=1e-15),
        f"max |r[:,1]| = {float(np.max(np.abs(r[:, 1]))):.2e}",
    )
    check(
        "Cholesky K=(1,1,5): r[:,2] varies (xb axis active)",
        float(np.std(r[:, 2])) > 1e-6,
        f"std(r[:,2]) = {float(np.std(r[:, 2])):.4e}",
    )

    # K=(1, K_high, 1): r[0] still zero (lower-triangular), r[1] and r[2] vary
    r, _ = get_return_quadrature(model, n_nodes=(1, 5, 1))
    check(
        "Cholesky K=(1,5,1): r[:,0] all zero (rtb axis collapsed)",
        np.allclose(r[:, 0], 0.0, atol=1e-15),
        f"max |r[:,0]| = {float(np.max(np.abs(r[:, 0]))):.2e}",
    )
    check(
        "Cholesky K=(1,5,1): r[:,1] varies (xr axis active)",
        float(np.std(r[:, 1])) > 1e-6,
    )

    # K=(K_high, 1, 1): all three components vary along the L[:, 0] column.
    # The ratio r[:,1]/r[:,0] should equal L[1,0]/L[0,0] for every node.
    r, _ = get_return_quadrature(model, n_nodes=(5, 1, 1))
    nonzero = np.abs(r[:, 0]) > 1e-15
    if nonzero.any():
        ratio_emp = r[nonzero, 1] / r[nonzero, 0]
        ratio_exp = L[1, 0] / L[0, 0]
        check(
            "Cholesky K=(5,1,1): r[:,1]/r[:,0] == L[1,0]/L[0,0] for all nodes",
            np.allclose(ratio_emp, ratio_exp, atol=1e-13),
            f"max dev = {float(np.max(np.abs(ratio_emp - ratio_exp))):.2e}",
        )

    # T-E: validation rejects bad inputs
    for bad in [(2, 2), (2, 2, 2, 2), (2, 0, 2)]:
        try:
            get_return_quadrature(model, n_nodes=bad)
            check(f"validation: rejects {bad}", False, "did not raise")
        except (ValueError, TypeError):
            check(f"validation: rejects {bad}", True)


def main() -> int:
    model, data = build_reference_model()
    run_geometry_checks(model, data)
    run_precompute_mode_checks(model)
    run_solver_simulation_smoke(model)
    run_per_axis_n_stds_checks(model)
    run_per_axis_n_state_quad_checks(model)
    run_per_axis_n_ret_quad_checks(model)

    print("\n" + "=" * 70)
    print(f"RESULTS: {N_PASS} passed, {N_FAIL} failed")
    print("=" * 70)
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

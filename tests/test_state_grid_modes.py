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

    hist = data[["y_1", "spr", "cy"]].to_numpy()
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
    check("Principal sim: valid z bounds", np.all(sim["z"][sim["alive"]] >= pc.z_grid[0] - 1e-10) and np.all(sim["z"][sim["alive"]] <= pc.z_grid[-1] + 1e-10))
    check("Principal sim: estate finite", np.all(np.isfinite(sim["estate"])))


def main() -> int:
    model, data = build_reference_model()
    run_geometry_checks(model, data)
    run_precompute_mode_checks(model)
    run_solver_simulation_smoke(model)

    print("\n" + "=" * 70)
    print(f"RESULTS: {N_PASS} passed, {N_FAIL} failed")
    print("=" * 70)
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

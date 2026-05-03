"""State-grid coverage diagnostic for principal-mode lifecycle simulations.

Formalizes what `state_n_stds` actually controls in principal mode and reports
the simulation-coverage fraction implied by a candidate setting.

Background
----------
In principal mode the financial-state grid is a parallelepiped in s-space:

    state_grid = { mu_s + L u  :  |u_d| <= n_stds[d]  for d = 0, 1, 2 }

where mu_s is the VAR unconditional mean, L L' = Sigma_z is the Cholesky
factor of the marginal stationary covariance (discrete Lyapunov solution),
and u = L^{-1} (s - mu_s) are the bracketing coordinates the solver and
simulator use.

Under the stationary VAR distribution, u ~ N(0, I_3) by construction.  So
n_stds[d] is literally "number of standard deviations of u_d", and u_d are
mutually orthogonal/independent under the model's stationary law.  The
state_n_stds tuple is therefore directly readable: it is per-axis half-width
in unit-variance, mutually independent standardized coordinates.

The question "what fraction of simulated states fall inside the grid" reduces
to a coverage statement on independent unit Gaussians:

    P_stationary(state in grid) = prod_d [ 2 * Phi(n_stds[d]) - 1 ]

The lifecycle simulation propagates the same VAR forward.  After a short
burn-in (~5-10 years) from any centered start, the per-age marginal of u_d
matches the stationary N(0,1).  Pooled across the lifecycle the empirical
coverage tracks the analytical formula very closely (within ~1% for our
calibration).

Coverage selection rule
-----------------------
Given a target joint coverage P_tgt of simulated agent-years, the simplest
choice is uniform n across axes:

    P_axis = P_tgt^(1/3)
    n_stds = Phi^{-1}((1 + P_axis) / 2)

For 99% joint coverage of u ~ N(0,I) this gives n_stds ~= 2.93.  For 99.9%
it gives n_stds ~= 3.59.  Per-axis tuning helps only when the simulated
distribution is materially non-stationary on one axis (e.g. solving only
ages 65-99 from a centered age-65 start under-spreads the slow cy axis).
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, cholesky as _chol
from scipy.stats import norm

from lifecycle.var import build_nominal_system1_var_config, partition_var


def stationary_moments(Phi_11, Sigma_ss, Phi_0):
    mu_s = np.linalg.solve(np.eye(3) - Phi_11, Phi_0)
    Sigma_z = solve_discrete_lyapunov(Phi_11, Sigma_ss)
    Sigma_z = 0.5 * (Sigma_z + Sigma_z.T)
    L = np.linalg.cholesky(Sigma_z)
    L_inv = np.linalg.inv(L)
    return mu_s, Sigma_z, L, L_inv


def simulate_var(Phi_0, Phi_11, Sigma_ss, s0, n_paths, T, rng):
    L_ss = _chol(Sigma_ss, lower=True)
    s = np.broadcast_to(s0, (n_paths, 3)).copy()
    out = np.empty((n_paths, T, 3))
    for t in range(T):
        innov = rng.standard_normal((n_paths, 3)) @ L_ss.T
        s = Phi_0[None, :] + s @ Phi_11.T + innov
        out[:, t] = s
    return out


def joint_inside(U, n_vec):
    n_arr = np.asarray(n_vec, dtype=float)
    return float(np.all(np.abs(U) <= n_arr, axis=-1).mean())


def stationary_coverage_uniform(n):
    return (2.0 * norm.cdf(n) - 1.0) ** 3


def n_stds_for_target_uniform(P_tgt):
    p_axis = P_tgt ** (1.0 / 3.0)
    return float(norm.ppf(0.5 * (1.0 + p_axis)))


def n_stds_for_target_per_axis_empirical(U_flat, P_tgt):
    p_axis = P_tgt ** (1.0 / 3.0)
    return np.array([float(np.quantile(np.abs(U_flat[:, d]), p_axis)) for d in range(3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/var_dataset.csv")
    ap.add_argument("--n_paths", type=int, default=50_000)
    ap.add_argument("--T", type=int, default=78, help="lifecycle length (default 22..99)")
    ap.add_argument("--start", choices=["mu_s", "stationary"], default="mu_s")
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--targets", nargs="*", type=float, default=[0.95, 0.99, 0.995, 0.999])
    args = ap.parse_args()

    var_config, _, _ = build_nominal_system1_var_config(csv_path=args.csv, estimation="restricted")
    parts = partition_var(
        Phi_full=var_config["Phi"], Omega_full=var_config["Omega"],
        z_bar=var_config["z_bar"], state_idx=var_config["state_indices"],
        ret_idx=var_config["return_indices"],
        variable_names=var_config["variable_names"], verbose=False,
    )
    Phi_0, Phi_11, Sigma_ss = parts["Phi_0_state"], parts["Phi_11"], parts["Sigma_ss"]
    state_names = parts["state_names"]
    mu_s, Sigma_z, L, L_inv = stationary_moments(Phi_11, Sigma_ss, Phi_0)
    sigma_z = np.sqrt(np.diag(Sigma_z))

    print("=" * 72)
    print("Principal-mode coverage formalization")
    print("=" * 72)
    print(f"State ordering             : {state_names}")
    print(f"mu_s                       : {mu_s}")
    print(f"sigma_z (marginal stds)    : {sigma_z}")
    print("Cholesky L (col d = u_d direction in s-space):")
    print(L)
    print()
    print("Grid hull definition:")
    print("    s in grid  iff  u := L^{-1}(s - mu_s) satisfies |u_d| <= n_stds[d]")
    print("Under stationary VAR law:  u ~ N(0, I_3), axes independent.")
    print()

    print("Stationary-law analytical coverage (uniform n across axes):")
    print(f"  {'n':>5} {'P_per_axis':>11} {'P_joint':>11}")
    for n in [1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.3, 3.5, 4.0]:
        p_a = 2 * norm.cdf(n) - 1
        p_j = p_a ** 3
        print(f"  {n:>5.2f} {p_a:>11.4f} {p_j:>11.4f}")
    print()

    rng = np.random.default_rng(args.seed)
    if args.start == "mu_s":
        s0 = mu_s
    else:
        s0 = mu_s + rng.standard_normal((args.n_paths, 3)) @ L.T
    S = simulate_var(Phi_0, Phi_11, Sigma_ss, s0, args.n_paths, args.T, rng)
    U = (S - mu_s[None, None, :]) @ L_inv.T
    flat = U.reshape(-1, 3)

    print(f"Empirical simulation (n_paths={args.n_paths}, T={args.T}, start={args.start}):")
    print(f"  per-axis std of u           : {flat.std(axis=0).round(4)}")
    print(f"  per-axis |u| q99            : {np.quantile(np.abs(flat), 0.99, axis=0).round(3)}")
    print(f"  per-axis |u| q99.9          : {np.quantile(np.abs(flat), 0.999, axis=0).round(3)}")
    print()

    print("Coverage of candidate state_n_stds settings (simulation, pooled ages):")
    candidates = [
        (0.6, 1.75, 2.0),
        (1.0, 1.75, 2.0),
        (1.75, 2.0, 2.25),
        (2.0, 2.0, 2.0),
        (2.5, 2.5, 2.5),
        (2.9, 2.9, 2.9),
        (3.0, 3.0, 3.0),
        (3.5, 3.5, 3.5),
    ]
    print(f"  {'n_stds':<26} {'sim_inside':>10} {'theory_iid':>11}")
    for ns in candidates:
        cov_sim = joint_inside(flat, ns)
        cov_th = float(np.prod([2 * norm.cdf(n) - 1 for n in ns]))
        print(f"  {str(ns):<26} {cov_sim*100:>9.2f}% {cov_th*100:>10.2f}%")
    print()

    print("Recommendations (uniform n_stds, both analytical and empirical):")
    print(f"  {'P_target':>9}  {'n_uniform_iid':>14}  {'n_uniform_emp':>14}")
    for tgt in args.targets:
        n_iid = n_stds_for_target_uniform(tgt)
        # bisect for empirical uniform n
        lo, hi = 0.5, 6.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if joint_inside(flat, (mid,) * 3) >= tgt:
                hi = mid
            else:
                lo = mid
        print(f"  {tgt:>9.4f}  {n_iid:>14.3f}  {hi:>14.3f}")
    print()

    print("Recommendations (per-axis n_stds from empirical |u_d| quantiles):")
    print(f"  {'P_target':>9}  {'n0':>7}  {'n1':>7}  {'n2':>7}  {'realized_joint':>16}")
    for tgt in args.targets:
        n_vec = n_stds_for_target_per_axis_empirical(flat, tgt)
        cov = joint_inside(flat, n_vec)
        print(f"  {tgt:>9.4f}  {n_vec[0]:>7.3f}  {n_vec[1]:>7.3f}  {n_vec[2]:>7.3f}  {cov*100:>15.2f}%")


if __name__ == "__main__":
    main()

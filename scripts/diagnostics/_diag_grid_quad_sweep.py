"""
_diag_grid_quad_sweep.py — Sweep state-grid and quadrature configs.

For each (state_grid_sizes, state_n_stds, K_state, K_ret_per_dim) cell, build
a fresh Precompute (no solver run) and measure the structural metrics that
the three "ameliorating" mechanisms predict will move:

  Mechanism 1 — less probability of extreme states occurring
    metric: cumulative stationary mass of states with ‖α_merton‖∞ > τ
    knob expected to move it: tighten n_stds

  Mechanism 2 — smaller Sharpe ratios at extreme states
    metric: max over states of joint Sharpe
    knob expected to move it: tighten n_stds

  Mechanism 3 — less predictability of future returns
    metric: effective R² = Var(E[r|s])_stat / Var(r) for xr and xb
    knob expected to move it: tighten n_stds

Plus two cross-cutting checks:

  Mechanism 4 — interpolation gradient (Source B amplification)
    metric: max ‖α_merton(s) − α_merton(s')‖∞ between adjacent grid states
    knob expected to move it: refine state_grid_sizes

  Sanity — extreme conditional expected return
    metric: max E[xr|s], max E[xb|s] (and the negatives)
    knob expected to move it: tighten n_stds

Run from repo root: python -m scripts.diagnostics._diag_grid_quad_sweep
"""

from __future__ import annotations

import numpy as np

from lifecycle.var import build_nominal_system1_var_config
from lifecycle.precompute import build_model, Precompute
from lifecycle.model import DiscretizationConfig


def base_config():
    return dict(
        beta=0.96, gamma=3.0, b_bar=10,
        start_age=22, retire_age=67, terminal_age=99,
        b0=-6.142, b1=0.3040, b2=-0.051, b3=0.002586,
        rho=0.991, pz=0.176,
        mu_eta1=-0.524, sigma_eta1=0.113,
        mu_eta2=0.0, sigma_eta2=0.046,
        pe=0.044,
        mu_eps1=0.134, sigma_eps1=0.762,
        mu_eps2=0.0, sigma_eps2=0.055,
        constrained=False,
    )


def build_cloud(model, pc, i_s):
    s_i = pc.state_grid[i_s]
    base = np.asarray(pc.const_r) + pc.A_r @ s_i
    M = np.asarray(model.M)
    Mv = pc.v_nodes @ M.T
    log_r = (base[None, None, :] + Mv[:, None, :] + pc.ret_nodes[None, :, :]).reshape(-1, 3)
    w = np.outer(pc.v_weights, pc.ret_weights).ravel()
    R_bill = np.exp(log_r[:, 0])
    R_stock = R_bill * np.exp(log_r[:, 1])
    R_bond = R_bill * np.exp(log_r[:, 2])
    return np.column_stack([R_bill, R_stock, R_bond]), w


def per_state_metrics(model, pc, gamma):
    """Per-state Merton alpha, Sharpe, mu_e."""
    n = pc.N_state
    mu_e = np.zeros((n, 2))
    sharpe_j = np.zeros(n)
    alpha_m = np.zeros((n, 2))
    for i_s in range(n):
        gross, w = build_cloud(model, pc, i_s)
        Xs = gross[:, 1] - gross[:, 0]
        Xb = gross[:, 2] - gross[:, 0]
        m_s = float(np.sum(w * Xs)); m_b = float(np.sum(w * Xb))
        v_s = float(np.sum(w * (Xs - m_s) ** 2))
        v_b = float(np.sum(w * (Xb - m_b) ** 2))
        cov = float(np.sum(w * (Xs - m_s) * (Xb - m_b)))
        Sigma_e = np.array([[v_s, cov], [cov, v_b]])
        m_vec = np.array([m_s, m_b])
        mu_e[i_s] = m_vec
        try:
            inv_S_mu = np.linalg.solve(Sigma_e, m_vec)
            alpha_m[i_s] = (1.0 / gamma) * inv_S_mu
            sharpe_j[i_s] = float(np.sqrt(max(0.0, m_vec @ inv_S_mu)))
        except np.linalg.LinAlgError:
            alpha_m[i_s] = np.nan
            sharpe_j[i_s] = np.nan
    return mu_e, sharpe_j, alpha_m


def adjacency_gradient(state_grid, state_indices, alpha_merton):
    """Max ‖α_merton(s) − α_merton(s')‖∞ between any two adjacent grid states.

    "Adjacent" = differ by exactly 1 in one bracket-grid coordinate. Returns
    max over all such adjacent pairs.
    """
    # state_indices is (N_state, 3), each row is the multi-index.
    # Build a hash from multi-index tuple to flat index.
    n = state_grid.shape[0]
    idx_to_flat = {tuple(state_indices[i].tolist()): i for i in range(n)}

    max_grad = 0.0
    for flat in range(n):
        idx = state_indices[flat]
        for d in range(3):
            for delta in (-1, 1):
                neighbor = tuple(int(x) + (delta if k == d else 0) for k, x in enumerate(idx))
                if neighbor in idx_to_flat:
                    j = idx_to_flat[neighbor]
                    diff = np.max(np.abs(alpha_merton[flat] - alpha_merton[j]))
                    if diff > max_grad:
                        max_grad = float(diff)
    return max_grad


def effective_r2(model, pc):
    """Effective R² weighted by stationary distribution of the state grid.

    Var(E[r|s]) computed empirically from (Phi_21 · s_i) weighted by stat_probs.
    Total Var(r) = Var(E[r|s]) + (M Σ_ss M' + Σ_r_cond) ≈ analytical Σ_rr.
    """
    stat = np.asarray(pc.state_stationary_probs)
    Phi_21 = np.asarray(model.Phi_21)
    s = np.asarray(pc.state_grid)  # (N_state, 3)
    cond_mean = s @ Phi_21.T  # (N_state, 3) — Phi_21 · s for each state, in (rtb, xr, xb) order
    mu_pred = stat @ cond_mean  # (3,)
    diff = cond_mean - mu_pred
    Cov_pred = (diff.T * stat) @ diff  # (3, 3) — explained covariance
    M = np.asarray(model.M)
    Cov_resid = M @ np.asarray(model.Sigma_ss) @ M.T + np.asarray(model.Sigma_r_cond)
    R2 = np.diag(Cov_pred) / (np.diag(Cov_pred) + np.diag(Cov_resid))  # (3,)
    return float(R2[0]), float(R2[1]), float(R2[2])  # rtb, xr, xb


def cell_metrics(model, pc, gamma):
    stat = np.asarray(pc.state_stationary_probs)
    mu_e, sharpe_j, alpha_m = per_state_metrics(model, pc, gamma)
    alpha_inf = np.max(np.abs(alpha_m), axis=1)
    grad = adjacency_gradient(pc.state_grid, pc.state_indices, alpha_m)
    R2_rtb, R2_xr, R2_xb = effective_r2(model, pc)
    return {
        "N_state": int(pc.N_state),
        "max_mu_xr": float(np.nanmax(mu_e[:, 0])),
        "min_mu_xr": float(np.nanmin(mu_e[:, 0])),
        "max_mu_xb": float(np.nanmax(mu_e[:, 1])),
        "min_mu_xb": float(np.nanmin(mu_e[:, 1])),
        "max_sharpe": float(np.nanmax(sharpe_j)),
        "median_sharpe": float(np.nanmedian(sharpe_j)),
        "max_alpha": float(np.nanmax(alpha_inf)),
        "mass_alpha_gt_2": float(stat[alpha_inf > 2].sum()),
        "mass_alpha_gt_5": float(stat[alpha_inf > 5].sum()),
        "mass_alpha_gt_10": float(stat[alpha_inf > 10].sum()),
        "adjacency_grad": grad,
        "R2_xr": R2_xr,
        "R2_xb": R2_xb,
    }


def main():
    var_config, _, _ = build_nominal_system1_var_config(csv_path="data/var_dataset.csv")
    bc = base_config()
    model = build_model(bc, var_config, verbose=False)

    # Sweep cells: (label, grid_sizes, n_stds, K_state, K_ret)
    cells = [
        # === Baseline (production config) ===
        ("baseline    (5,5,5)/2.0/K2/(3,5,3)", (5, 5, 5), 2.0, 2, (3, 5, 3)),

        # === Mechanism 1/2/3: tighten n_stds ===
        ("nstds=1.75  (5,5,5)/1.75/K2/(3,5,3)", (5, 5, 5), 1.75, 2, (3, 5, 3)),
        ("nstds=1.5   (5,5,5)/1.5/K2/(3,5,3)",  (5, 5, 5), 1.5, 2, (3, 5, 3)),
        ("nstds=1.0   (5,5,5)/1.0/K2/(3,5,3)",  (5, 5, 5), 1.0, 2, (3, 5, 3)),

        # === Mechanism 4: refine state grid (more grid points, same coverage) ===
        ("grid=(5,7,7) (5,7,7)/2.0/K2/(3,5,3)", (5, 7, 7), 2.0, 2, (3, 5, 3)),
        ("grid=(7,7,7) (7,7,7)/2.0/K2/(3,5,3)", (7, 7, 7), 2.0, 2, (3, 5, 3)),
        ("grid=(9,9,9) (9,9,9)/2.0/K2/(3,5,3)", (9, 9, 9), 2.0, 2, (3, 5, 3)),

        # === Combined ===
        ("combo       (5,7,7)/1.5/K2/(3,5,3)", (5, 7, 7), 1.5, 2, (3, 5, 3)),
        ("combo+      (7,7,7)/1.5/K2/(3,5,3)", (7, 7, 7), 1.5, 2, (3, 5, 3)),

        # === Sanity: K_state refinement (analytical claim says no effect) ===
        ("K_state=3   (5,5,5)/2.0/K3/(3,5,3)", (5, 5, 5), 2.0, 3, (3, 5, 3)),
        ("K_state=4   (5,5,5)/2.0/K4/(3,5,3)", (5, 5, 5), 2.0, 4, (3, 5, 3)),

        # === Sanity: K_ret refinement (analytical claim says marginal effect) ===
        ("K_ret(3,9,3) (5,5,5)/2.0/K2/(3,9,3)", (5, 5, 5), 2.0, 2, (3, 9, 3)),
        ("K_ret(5,9,5) (5,5,5)/2.0/K2/(5,9,5)", (5, 5, 5), 2.0, 2, (5, 9, 5)),
    ]

    print()
    print(f"{'config':<38}  {'N_s':>4}  {'max_α':>7}  {'p_α>2':>7}  {'p_α>5':>7}  "
          f"{'p_α>10':>7}  {'maxSh':>6}  {'medSh':>6}  {'max_xr':>7}  {'min_xr':>7}  "
          f"{'max_xb':>7}  {'min_xb':>7}  {'grad':>6}  {'R²_xr':>6}  {'R²_xb':>6}")
    print("-" * 153)

    for label, gs, ns, kst, kret in cells:
        try:
            disc = DiscretizationConfig(
                n_wealth=40, n_savings=40,
                state_grid_sizes=gs, state_grid_mode="principal", state_n_stds=ns,
                n_z=5, n_eps_nodes=3, n_eta_nodes=3,
                n_ret_nodes_1d=kret, n_state_quad_nodes=kst,
            )
            pc = Precompute(model, disc, verbose=False)
            m = cell_metrics(model, pc, gamma=bc["gamma"])
            print(
                f"{label:<38}  {m['N_state']:>4}  {m['max_alpha']:>7.2f}  "
                f"{m['mass_alpha_gt_2']*100:>6.2f}%  {m['mass_alpha_gt_5']*100:>6.2f}%  "
                f"{m['mass_alpha_gt_10']*100:>6.2f}%  {m['max_sharpe']:>6.2f}  "
                f"{m['median_sharpe']:>6.2f}  {m['max_mu_xr']:>7.3f}  {m['min_mu_xr']:>7.3f}  "
                f"{m['max_mu_xb']:>7.3f}  {m['min_mu_xb']:>7.3f}  {m['adjacency_grad']:>6.2f}  "
                f"{m['R2_xr']:>6.3f}  {m['R2_xb']:>6.3f}"
            )
        except Exception as e:
            print(f"{label:<38}  ERROR: {type(e).__name__}: {e}")

    print()
    print("Reading guide:")
    print("  N_s     = N_state (joint states)")
    print("  max_α   = max ||α_merton(s)||_∞ across grid (the per-period upper bound)")
    print("  p_α>k   = cumulative stationary prob of states with ||α_merton||∞ > k")
    print("  maxSh / medSh = max / median joint Sharpe across states")
    print("  max_xr  = max E[xr|s] (log conditional excess stock return at most extreme state)")
    print("  min_xr  = min E[xr|s] (most negative)")
    print("  max_xb / min_xb = same for bond")
    print("  grad    = max ||α_merton(s) − α_merton(s')||∞ between adjacent grid states")
    print("  R²_xr   = effective predictability of stock return under stationary state dist")
    print("  R²_xb   = effective predictability of bond return")
    print()
    print("Mechanisms:")
    print("  M1 'less prob of extremes'         → watch p_α>5, p_α>10  (tighten n_stds)")
    print("  M2 'smaller Sharpe at extremes'    → watch maxSh         (tighten n_stds)")
    print("  M3 'less return predictability'    → watch R²_xr, R²_xb  (tighten n_stds)")
    print("  M4 'less interpolation bleed'      → watch grad          (refine grid_sizes)")


if __name__ == "__main__":
    main()

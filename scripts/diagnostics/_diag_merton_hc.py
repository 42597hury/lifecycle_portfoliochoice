"""Back-of-envelope Merton-with-human-capital check.

For each (age, iz, financial_wealth) in retirement:
  HC = expected PV of remaining pension at 3% discount, weighted by survival
  total = HC + W
  alpha_merton = (E[xr] + 0.5 Var[xr]) / (gamma * Var[xr])    # unconditional, no predictability
  alpha_financial = alpha_merton * total / W

We then check what fraction of (age, iz, W) cells have alpha_financial above
various leverage thresholds. This tells us whether 18% invalid cells in Diag B
can plausibly be explained by the unconditional Merton optimum demanding more
leverage than the cap allows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lifecycle"))

from lifecycle.policy_io import load_policy_bundle  # noqa: E402
from lifecycle.precompute import Precompute, build_model  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402

BUNDLE = "saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v3"
DISCOUNT = 0.03  # back-of-envelope


def main():
    C, S, B, _, meta = load_policy_bundle(BUNDLE)
    rc = meta["run_config"]
    bc = dict(rc["base_config"])
    vc = dict(rc["var_config"])
    for k in list(vc.keys()):
        v = vc.get(k)
        if isinstance(v, dict) and "values" in v:
            shape = v.get("shape")
            arr = np.array(v["values"], dtype=float)
            if shape:
                arr = arr.reshape(shape)
            vc[k] = arr

    model = build_model(bc, vc, verbose=False)
    gamma = float(model.gamma)
    print(f"gamma = {gamma}")

    # --- Unconditional return moments from the stationary VAR ---
    Phi = vc["Phi"]
    const = np.asarray(vc["const"], dtype=float)
    Omega = vc["Omega"]
    n = Phi.shape[0]
    var_names = vc["variable_names"]
    print("variables:", var_names)

    # Unconditional mean: y_bar = (I - Phi)^{-1} const
    y_bar = np.linalg.solve(np.eye(n) - Phi, const)
    print("y_bar (unconditional mean):", dict(zip(var_names, y_bar.round(4))))

    # Unconditional cov: solve discrete Lyapunov: Sigma = Phi Sigma Phi' + Omega
    Sigma_y = np.linalg.solve(
        np.eye(n * n) - np.kron(Phi, Phi),
        Omega.reshape(-1),
    ).reshape(n, n)
    sigma_y = np.sqrt(np.diag(Sigma_y))
    print("uncond std:", dict(zip(var_names, sigma_y.round(4))))

    # Pick stock excess return (xr)
    i_xr = var_names.index("xr")
    mu_xr_uncond = float(y_bar[i_xr])  # unconditional mean of log excess stock return
    var_xr_uncond = float(Sigma_y[i_xr, i_xr])  # unconditional variance (across states + residual)
    var_xr_innov = float(Omega[i_xr, i_xr])      # one-step innovation variance (the right one for myopic Merton)
    sd_xr_uncond = np.sqrt(var_xr_uncond)
    sd_xr_innov = np.sqrt(var_xr_innov)
    print(f"\nlog xr (uncond): mean = {mu_xr_uncond:.4f}, sd = {sd_xr_uncond:.4f}, Sharpe = {mu_xr_uncond/sd_xr_uncond:.3f}")
    print(f"log xr (innov):  sd = {sd_xr_innov:.4f}  (one-step innovation variance Omega[xr,xr])")
    print(f"Persistence ratio (uncond_var / innov_var): {var_xr_uncond/var_xr_innov:.3f}")

    # Myopic Merton fraction with a single risky asset (log-normal approximation),
    # using ONE-STEP CONDITIONAL VARIANCE (the right denominator for one-period rebalancing):
    #   alpha = (mu + 0.5 sigma_innov^2) / (gamma sigma_innov^2)
    var_xr = var_xr_innov  # use innovation variance from here on
    alpha_merton_uncond = (mu_xr_uncond + 0.5 * var_xr) / (gamma * var_xr)
    print(f"Unconditional Merton stock share (gamma={gamma}, one-step var): {alpha_merton_uncond:.3f}")

    # Conditional Merton at each financial-state corner: mu_xr(s) = const[xr] + Phi[xr, :] @ s
    # The state vector in the VAR is the full y; the model's state grid is over
    # the predictor columns only. Use precompute to get the conditional mean.
    disc_d = meta["diagnostics_summary"]["disc_config"]
    disc = DiscretizationConfig(
        n_wealth=disc_d["n_wealth"], wealth_min=disc_d["wealth_min"],
        wealth_max=disc_d["wealth_max"], n_savings=disc_d["n_savings"],
        savings_min=disc_d["savings_min"], savings_max=disc_d.get("savings_max"),
        state_grid_sizes=tuple(disc_d["state_grid_sizes"]),
        state_grid_mode=disc_d["state_grid_mode"],
        state_n_stds=tuple(disc_d["state_n_stds"]),
        n_z=disc_d["n_z"], n_stds=disc_d["n_stds"],
        n_eps_nodes=disc_d["n_eps_nodes"], n_eta_nodes=disc_d["n_eta_nodes"],
        n_ret_nodes_1d=tuple(disc_d["n_ret_nodes_1d"]),
        n_state_quad_nodes=tuple(disc_d["n_state_quad_nodes"]),
    )
    pc = Precompute(model, disc, verbose=False)

    # Conditional mean of xr at each state-grid node: const_r + A_r @ s_node
    # A_r = Phi rows for return indices, restricted to state-predictor columns
    # state_predictor_columns is a list of names; map to integer indices in Phi
    state_pred_names = list(vc["state_predictor_columns"])
    state_pred_cols = [var_names.index(n) for n in state_pred_names]
    print(f"state_pred_cols (indices): {state_pred_cols} (names: {state_pred_names})")
    # state_grid is in "y-space" coordinates (the predictor columns)
    state_grid = pc.state_grid  # (N_state, 3) in predictor-column order
    print(f"\nstate_grid shape: {state_grid.shape}")

    # Conditional mean for xr: const[xr] + Phi[xr, state_pred_cols] @ s
    A_r_xr = Phi[i_xr, state_pred_cols]  # (3,)
    cond_mu_xr = const[i_xr] + state_grid @ A_r_xr  # (N_state,)
    cond_alpha = (cond_mu_xr + 0.5 * var_xr) / (gamma * var_xr)
    print(f"\nConditional Merton α at state corners (no HC):")
    print(f"  min: {cond_alpha.min():.3f}, max: {cond_alpha.max():.3f}, mean: {cond_alpha.mean():.3f}")
    print(f"  share corners > 1: {np.mean(cond_alpha > 1):.1%}")
    print(f"  share corners > 3: {np.mean(cond_alpha > 3):.1%}")
    print(f"  share corners > 6: {np.mean(cond_alpha > 6):.1%}")

    # ---- Now add Human Capital (pension PV) ----
    # In retirement, pension is determined by z. For each iz, get pension level
    # from precompute; model.pension_after_tax has shape (n_age, n_z)
    pension = np.asarray(pc.pension_after_tax)        # (n_age, n_z)
    survival = np.asarray(pc.survival_probs_2d)       # (n_age, n_z); P(alive at t+1 | alive at t, z)
    n_age = pension.shape[0]
    n_z = pension.shape[1]
    retire_idx = model.retire_age - model.start_age
    terminal_idx = n_age - 1

    print(f"\nretire_idx={retire_idx}, n_age={n_age}, n_z={n_z}")
    # HC[t, iz] = sum_{s=t}^{T} prod(survival[t..s-1, iz]) * pension[s, iz] / 1.03^(s-t)
    HC = np.zeros((n_age, n_z))
    for iz in range(n_z):
        for t in range(retire_idx, n_age):
            cum_surv = 1.0
            pv = pension[t, iz]
            for s in range(t + 1, n_age):
                cum_surv *= survival[s - 1, iz]
                pv += cum_surv * pension[s, iz] / (1.0 + DISCOUNT) ** (s - t)
            HC[t, iz] = pv

    print(f"\nHC at age 67 by iz: min={HC[retire_idx].min():.2f}, max={HC[retire_idx].max():.2f}")
    print(f"HC at age 80 by iz: min={HC[80 - model.start_age].min():.2f}, max={HC[80 - model.start_age].max():.2f}")
    print(f"HC at age 95 by iz: min={HC[95 - model.start_age].min():.2f}, max={HC[95 - model.start_age].max():.2f}")
    print(f"HC at age 99 by iz: min={HC[99 - model.start_age].min():.2f}, max={HC[99 - model.start_age].max():.2f}")
    print(f"\nFor reference, pension[67, :] = {pension[retire_idx]}")

    # ---- Implied financial-asset stock share over the wealth grid ----
    wealth_grid = np.geomspace(disc.wealth_min, disc.wealth_max, disc.n_wealth)

    # Use UNCONDITIONAL Merton as user requested ("we approximate it this way")
    # alpha_financial = alpha_merton_uncond * (W + HC) / W = alpha_merton_uncond * (1 + HC/W)
    alpha_uncond = alpha_merton_uncond

    # Aggregate: at each retirement age, for each iz, for each W in the grid,
    # compute alpha_financial; report distribution
    rows = []
    for t in range(retire_idx, n_age):
        for iz in range(n_z):
            hc = HC[t, iz]
            for iw, w in enumerate(wealth_grid):
                alpha_fin = alpha_uncond * (1.0 + hc / w)
                rows.append((t + model.start_age, iz, iw, w, hc, alpha_fin))

    arr = np.array(rows, dtype=[
        ("age", "i4"), ("iz", "i4"), ("iw", "i4"),
        ("w", "f8"), ("hc", "f8"), ("alpha_fin", "f8"),
    ])

    print(f"\n=== Implied alpha_financial under UNCONDITIONAL Merton + HC ===")
    print(f"Total cells: {len(arr)}")
    print(f"Median alpha_fin: {np.median(arr['alpha_fin']):.3f}")
    print(f"Mean alpha_fin:   {np.mean(arr['alpha_fin']):.3f}")
    print(f"Q10/Q90:          {np.quantile(arr['alpha_fin'], 0.1):.3f} / {np.quantile(arr['alpha_fin'], 0.9):.3f}")
    print(f"Max alpha_fin:    {np.max(arr['alpha_fin']):.3f}")
    print()
    print(f"Share alpha_fin > 1.0:  {np.mean(arr['alpha_fin'] > 1.0):.1%}")
    print(f"Share alpha_fin > 2.0:  {np.mean(arr['alpha_fin'] > 2.0):.1%}")
    print(f"Share alpha_fin > 3.0:  {np.mean(arr['alpha_fin'] > 3.0):.1%}")
    print(f"Share alpha_fin > 4.0:  {np.mean(arr['alpha_fin'] > 4.0):.1%}")
    print(f"Share alpha_fin > 6.0:  {np.mean(arr['alpha_fin'] > 6.0):.1%}")
    print(f"Share alpha_fin > 10.0: {np.mean(arr['alpha_fin'] > 10.0):.1%}")

    # ---- Per-age breakdown ----
    print(f"\n=== Share alpha_fin > 6 by retirement age ===")
    print(f"{'age':>4} {'>1':>6} {'>3':>6} {'>6':>6} {'>10':>6}  median")
    for t in range(retire_idx, n_age):
        age = t + model.start_age
        sub = arr[arr["age"] == age]
        print(f"{age:>4d} "
              f"{np.mean(sub['alpha_fin'] > 1):>6.1%} "
              f"{np.mean(sub['alpha_fin'] > 3):>6.1%} "
              f"{np.mean(sub['alpha_fin'] > 6):>6.1%} "
              f"{np.mean(sub['alpha_fin'] > 10):>6.1%}  "
              f"{np.median(sub['alpha_fin']):.2f}")

    # ---- Where high alpha_fin lives in (age, iz, w) ----
    print(f"\n=== Worst 10 cells (highest alpha_fin) ===")
    idx = np.argsort(-arr["alpha_fin"])[:10]
    for i in idx:
        r = arr[i]
        print(f"  age={r['age']} iz={r['iz']} W={r['w']:.3f} HC={r['hc']:.2f} "
              f"HC/W={r['hc']/r['w']:.1f}  alpha_fin={r['alpha_fin']:.2f}")

    print(f"\n=== Best 10 cells (lowest alpha_fin, agents who DON'T need leverage) ===")
    idx = np.argsort(arr["alpha_fin"])[:10]
    for i in idx:
        r = arr[i]
        print(f"  age={r['age']} iz={r['iz']} W={r['w']:.2f} HC={r['hc']:.2f} "
              f"HC/W={r['hc']/r['w']:.2f}  alpha_fin={r['alpha_fin']:.2f}")

    # ---- Now combine with conditional state effect ----
    # alpha_fin_state(t, iz, w, j_s) = alpha_cond[j_s] * (1 + HC[t,iz]/w)
    print(f"\n=== Conditional Merton + HC: share > cap (6) over (age, iz, w, state) ===")
    # Random sample to avoid memory blow-up
    n_state = state_grid.shape[0]
    big = np.empty(0, dtype=float)
    for t in range(retire_idx, n_age):
        for iz in range(n_z):
            hc = HC[t, iz]
            # alpha = alpha_cond[js] * (1 + hc/w) for each (js, iw)
            ratio = 1.0 + hc / wealth_grid  # (n_w,)
            alpha_grid = np.outer(cond_alpha, ratio)  # (n_state, n_w)
            big = np.concatenate([big, alpha_grid.flatten()])

    print(f"Total cells (conditional × HC): {len(big)}")
    print(f"Median alpha_fin: {np.median(big):.3f}")
    print(f"Share > 1: {np.mean(big > 1.0):.1%}")
    print(f"Share > 3: {np.mean(big > 3.0):.1%}")
    print(f"Share > 6: {np.mean(big > 6.0):.1%}")
    print(f"Share > 10: {np.mean(big > 10.0):.1%}")
    print(f"Max: {big.max():.2f}")

    # Compare to the cap
    print(f"\n=== Cap-relevant ===")
    print(f"alpha_max in solver = ±6")
    print(f"Share of (cond+HC) cells where Merton-implied α > 6: {np.mean(big > 6.0):.1%}")
    print(f"This is the UPPER BOUND on what fraction of cells *should* be cap-binding")
    print(f"if our conjecture (extreme Sharpe ratios cause cap-binding) is right.")
    print(f"Diag B invalidity is 18% — does that match?")


if __name__ == "__main__":
    main()

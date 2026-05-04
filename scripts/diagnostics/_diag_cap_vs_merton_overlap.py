"""Cross-tabulate cap-bound policy cells vs Merton-HC implied high-leverage cells.

For each retirement grid cell (age, iz, j_s, iw):
  - actual_cap_bound = |alpha_s_policy| >= 5.95 OR |alpha_b_policy| >= 5.95
  - merton_high_lev   = (alpha_merton_uncond * (1 + HC/W)) > 6.0

Report the overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lifecycle"))

from lifecycle.policy_io import load_policy_bundle
from lifecycle.precompute import Precompute, build_model
from lifecycle.model import DiscretizationConfig

BUNDLE = "saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v3"
DISCOUNT = 0.03
CAP = 6.0
CAP_TOL = 0.05  # within 5 pct of cap → cap-bound


def main():
    C, S, B, _, meta = load_policy_bundle(BUNDLE)
    rc = meta["run_config"]
    bc = dict(rc["base_config"])
    vc = dict(rc["var_config"])
    for k in list(vc.keys()):
        v = vc.get(k)
        if isinstance(v, dict) and "values" in v:
            arr = np.array(v["values"], dtype=float)
            if v.get("shape"):
                arr = arr.reshape(v["shape"])
            vc[k] = arr

    model = build_model(bc, vc, verbose=False)
    gamma = float(model.gamma)
    var_names = vc["variable_names"]
    Phi = vc["Phi"]
    const = np.asarray(vc["const"], dtype=float)
    Omega = vc["Omega"]
    n = Phi.shape[0]
    y_bar = np.linalg.solve(np.eye(n) - Phi, const)
    i_xr = var_names.index("xr")
    mu_xr = float(y_bar[i_xr])
    var_xr = float(Omega[i_xr, i_xr])  # one-step innovation variance
    alpha_merton_uncond = (mu_xr + 0.5 * var_xr) / (gamma * var_xr)
    print(f"Unconditional Merton α (one-step var): {alpha_merton_uncond:.3f}")

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

    pension = np.asarray(pc.pension_after_tax)         # (n_age, n_z)
    survival = np.asarray(pc.survival_probs_2d)        # (n_age, n_z)
    n_age = pension.shape[0]
    n_z = pension.shape[1]
    retire_idx = model.retire_age - model.start_age

    HC = np.zeros((n_age, n_z))
    for iz in range(n_z):
        for t in range(retire_idx, n_age):
            cum = 1.0
            pv = pension[t, iz]
            for s in range(t + 1, n_age):
                cum *= survival[s - 1, iz]
                pv += cum * pension[s, iz] / (1.0 + DISCOUNT) ** (s - t)
            HC[t, iz] = pv

    wealth_grid = np.geomspace(disc.wealth_min, disc.wealth_max, disc.n_wealth)
    n_w = len(wealth_grid)
    N_state = pc.state_grid.shape[0]

    print(f"Bundle policy shape (n_age, n_z, N_state, n_w) = {C.shape}")
    print(f"Retirement ages: {retire_idx} to {n_age - 1}  ({n_age - retire_idx} ages)")

    # --- Build masks over retirement-only cells ---
    # actual cap-bound: |alpha_s| >= CAP - CAP_TOL  OR  |alpha_b| >= CAP - CAP_TOL
    S_ret = S[retire_idx:]  # (n_ret_age, n_z, N_state, n_w)
    B_ret = B[retire_idx:]
    n_ret_age = S_ret.shape[0]

    cap_bound_s = np.abs(S_ret) >= (CAP - CAP_TOL)
    cap_bound_b = np.abs(B_ret) >= (CAP - CAP_TOL)
    cap_bound = cap_bound_s | cap_bound_b

    finite = np.isfinite(S_ret) & np.isfinite(B_ret)
    total_valid = int(finite.sum())
    print(f"\nTotal finite retirement cells: {total_valid}")
    print(f"Actual cap-bound (|α_s| or |α_b| ≥ {CAP - CAP_TOL}): "
          f"{int((cap_bound & finite).sum())} ({100*(cap_bound & finite).sum()/total_valid:.2f}%)")
    print(f"  Of which |α_s| at cap: "
          f"{int((cap_bound_s & finite).sum())} ({100*(cap_bound_s & finite).sum()/total_valid:.2f}%)")
    print(f"  Of which |α_b| at cap: "
          f"{int((cap_bound_b & finite).sum())} ({100*(cap_bound_b & finite).sum()/total_valid:.2f}%)")

    # --- Build Merton-HC implied alpha_fin for each cell ---
    # alpha_fin[ret_t, iz, j_s, iw] = alpha_merton_uncond * (1 + HC[ret_t+retire_idx, iz] / W[iw])
    # (uses unconditional Merton; conditional state effect is modest and adds ≤0.3 to α_total)
    HC_ret = HC[retire_idx:]  # (n_ret_age, n_z)
    ratio = 1.0 + HC_ret[..., None] / wealth_grid[None, None, :]  # (n_ret_age, n_z, n_w)
    # Broadcast over j_s axis: same value across all state corners
    alpha_fin = alpha_merton_uncond * ratio[:, :, None, :]  # (n_ret_age, n_z, 1, n_w) → broadcast
    alpha_fin = np.broadcast_to(alpha_fin, S_ret.shape)
    merton_high_lev = alpha_fin > CAP

    print(f"\nMerton+HC implies α_fin > {CAP}: "
          f"{int((merton_high_lev & finite).sum())} ({100*(merton_high_lev & finite).sum()/total_valid:.2f}%)")

    # --- 2x2 confusion matrix ---
    a = (cap_bound & merton_high_lev & finite).sum()    # both
    b = (cap_bound & ~merton_high_lev & finite).sum()    # cap-bound but Merton doesn't predict
    c = (~cap_bound & merton_high_lev & finite).sum()    # Merton predicts but not cap-bound
    d = (~cap_bound & ~merton_high_lev & finite).sum()   # neither
    total = a + b + c + d

    print(f"\n=== 2×2: actual cap-bound vs Merton+HC predicts α > {CAP} ===")
    print(f"{'':30s} {'Merton α>'+str(CAP):>15s} {'Merton α≤'+str(CAP):>15s}")
    print(f"{'cap-bound (actual)':30s} {a:>15d} {b:>15d}")
    print(f"{'NOT cap-bound (actual)':30s} {c:>15d} {d:>15d}")
    print(f"\n  Total cells: {total}")
    print(f"  Conditional on cap-bound: P(Merton>{CAP}) = {a/(a+b):.1%}  ({a}/{a+b})")
    print(f"  Conditional on Merton>{CAP}: P(cap-bound) = {a/(a+c):.1%}  ({a}/{a+c})")
    print(f"  Cap-bound cells NOT explained by HC: {b} ({100*b/(a+b):.1f}% of cap-bound)")
    print(f"  Merton-predicted high-lev cells without cap-bind: {c} ({100*c/(a+c):.1f}% of Merton predicted)")

    # --- For cap-bound but NOT Merton-predicted: where do they live? ---
    not_explained = cap_bound & ~merton_high_lev & finite
    if not_explained.sum() > 0:
        idx = np.argwhere(not_explained)
        print(f"\n=== {not_explained.sum()} cap-bound cells NOT explained by HC: distribution ===")
        # by retirement-age index, by iz, by iw
        ages = retire_idx + idx[:, 0]
        izs = idx[:, 1]
        iws = idx[:, 3]
        print("By age (top 5):")
        ages_uniq, ages_count = np.unique(ages, return_counts=True)
        order = np.argsort(-ages_count)
        for k in order[:5]:
            print(f"  age {model.start_age + ages_uniq[k]}: {ages_count[k]} cells")
        print("By iz (top 5):")
        izs_uniq, izs_count = np.unique(izs, return_counts=True)
        order = np.argsort(-izs_count)
        for k in order[:5]:
            print(f"  iz {izs_uniq[k]} (pension {pension[retire_idx, izs_uniq[k]]:.4f}): {izs_count[k]} cells")
        print("By wealth-node iw (top 10 highest iw with cells):")
        iws_uniq, iws_count = np.unique(iws, return_counts=True)
        # show the highest wealth nodes that have cap-binding (these are the "rich but levered" cells)
        order = np.argsort(-iws_uniq)
        for k in order[:10]:
            iw = iws_uniq[k]
            print(f"  iw={iw} (W={wealth_grid[iw]:.2f}): {iws_count[k]} cells")

    # --- Inversion: Merton-predicted but not cap-bound: where? ---
    pred_not_realized = (~cap_bound) & merton_high_lev & finite
    if pred_not_realized.sum() > 0:
        idx = np.argwhere(pred_not_realized)
        ages = retire_idx + idx[:, 0]
        izs = idx[:, 1]
        iws = idx[:, 3]
        print(f"\n=== {pred_not_realized.sum()} Merton-predicted cells with NO cap-bind: distribution ===")
        print("By iw (lowest 10):")
        iws_uniq, iws_count = np.unique(iws, return_counts=True)
        order = np.argsort(iws_uniq)
        for k in order[:10]:
            iw = iws_uniq[k]
            print(f"  iw={iw} (W={wealth_grid[iw]:.4f}): {iws_count[k]} cells")
        # Show the actual policy at one of these cells
        idx0 = idx[0]
        rt, iz, js, iw = idx0
        print(f"\n  Example cell: ret_t={rt} (age {model.start_age + retire_idx + rt}), iz={iz}, "
              f"js={js}, iw={iw} (W={wealth_grid[iw]:.4f}), HC={HC[retire_idx + rt, iz]:.3f}")
        print(f"    actual alpha_s={S_ret[rt, iz, js, iw]:.3f}, alpha_b={B_ret[rt, iz, js, iw]:.3f}")
        print(f"    Merton-implied alpha_fin={alpha_fin[rt, iz, js, iw]:.3f}")


if __name__ == "__main__":
    main()

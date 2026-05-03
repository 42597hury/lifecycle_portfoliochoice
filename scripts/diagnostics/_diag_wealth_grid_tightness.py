"""
Solver-side diagnostic for whether wealth_max is too tight.

Purpose
-------
Decide among three grid-design options without rerunning the full lifecycle
solver:

  1. Leave the current grid as is
  2. Increase wealth_max only (keeping savings_max = wealth_max)
  3. Decouple savings_max from wealth_max, then choose a wider wealth_max

This diagnostic works on a saved policy bundle. It evaluates the *solver-side*
continuation lookup geometry, not the simulation distribution.

Key ideas
---------
1. If off-grid x_next carries substantial quadrature mass in upper wealth
   nodes, extrapolation is load-bearing and "leave as is" is hard to justify.
2. If savings_max is tied to wealth_max and there exist return realizations
   with gross return > 1, then the *top savings node* is structurally pushed
   off-grid under any coupled widening. In that case, "increase wealth_max
   only" cannot be a structural fix.
3. A decoupled design lets us keep the economically relevant savings range
   fixed while widening the continuation-lookup domain.

Usage
-----
    python -m scripts.diagnostics._diag_wealth_grid_tightness
    python -m scripts.diagnostics._diag_wealth_grid_tightness saved_runs/unconstrained_principal_grid5x5x5_nz9
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from lifecycle.model import DiscretizationConfig, scalar_disposable_income
from lifecycle.precompute import Precompute, build_model


DEFAULT_RUN_DIR = Path("saved_runs/unconstrained_principal_grid5x5x5_nz9")


def _load_saved_run(run_dir: Path):
    meta = json.loads((run_dir / "metadata.json").read_text())
    arrays = np.load(run_dir / "policy_arrays.npz")

    base_config = meta["run_config"]["base_config"]
    disc_dict = meta["run_config"]["discretization_config"]
    var_raw = meta["run_config"]["var_config"]
    var_config = {
        k: (
            np.asarray(v["values"], dtype=v["dtype"])
            if isinstance(v, dict) and v.get("kind") == "ndarray"
            else v
        )
        for k, v in var_raw.items()
    }

    model = build_model(base_config, var_config, verbose=False)
    disc_config = DiscretizationConfig(
        n_wealth=disc_dict["n_wealth"],
        wealth_min=disc_dict["wealth_min"],
        wealth_max=disc_dict["wealth_max"],
        n_savings=disc_dict["n_savings"],
        savings_min=disc_dict["savings_min"],
        savings_max=disc_dict.get("savings_max"),
        state_grid_sizes=tuple(disc_dict["state_grid_sizes"]),
        state_grid_mode=disc_dict["state_grid_mode"],
        state_n_stds=disc_dict["state_n_stds"],
        n_z=disc_dict["n_z"],
        n_stds=disc_dict["n_stds"],
        n_eps_nodes=disc_dict["n_eps_nodes"],
        n_eta_nodes=disc_dict["n_eta_nodes"],
        n_ret_nodes_1d=tuple(disc_dict["n_ret_nodes_1d"]),
        n_state_quad_nodes=disc_dict["n_state_quad_nodes"],
    )
    pc = Precompute(model, disc_config, verbose=False)
    return model, pc, arrays


def _age_index(pc, age: int) -> int:
    ages = list(pc.ages)
    if age not in ages:
        raise ValueError(f"Age {age} not found in saved run ages {ages[0]}-{ages[-1]}")
    return ages.index(age)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    c /= c[-1]
    return float(np.interp(q, c, v))


def _gross_return_envelope(pc) -> tuple[float, float]:
    r_max = -1.0
    r_min = 1e99
    for i_s in range(pc.N_state):
        base_mu = pc.const_r + pc.A_r @ pc.state_grid[i_s]
        for kv in range(pc.v_weights.shape[0]):
            mu_bill = base_mu[0] + pc.M_v_nodes[kv, 0]
            mu_stock = base_mu[1] + pc.M_v_nodes[kv, 1]
            mu_bond = base_mu[2] + pc.M_v_nodes[kv, 2]
            em_bill = math.exp(mu_bill)
            em_stock = math.exp(mu_stock)
            em_bond = math.exp(mu_bond)
            for kr in range(pc.ret_weights.shape[0]):
                r_bill = em_bill * pc.exp_ret_bill[kr]
                r_stock = r_bill * em_stock * pc.exp_ret_stock[kr]
                r_bond = r_bill * em_bond * pc.exp_ret_bond[kr]
                r_max = max(r_max, r_bill, r_stock, r_bond)
                r_min = min(r_min, r_bill, r_stock, r_bond)
    return r_min, r_max


def _offgrid_stats_for_node(model, pc, arrays, age: int, w_idx: int):
    C = arrays["C_mat"]
    S = arrays["S_mat"]
    B = arrays["B_mat"]

    t = _age_index(pc, age)
    wealth_max = float(pc.wealth_grid[-1])
    w = float(pc.wealth_grid[w_idx])

    total_mass = 0.0
    off_mass = 0.0
    any_off = 0
    total_cells = 0
    cell_off_mass_shares = []
    savings_ratios = []
    max_x = -1e99

    for z_idx in range(pc.n_z):
        for i_s in range(pc.N_state):
            total_cells += 1

            c = float(C[t, z_idx, i_s, w_idx])
            a_s = float(S[t, z_idx, i_s, w_idx])
            a_b = float(B[t, z_idx, i_s, w_idx])
            s_val = max(w - c, 0.0)
            savings_ratios.append(s_val / max(w, 1e-15))
            a_bill = 1.0 - a_s - a_b
            base_mu = pc.const_r + pc.A_r @ pc.state_grid[i_s]

            cell_total = 0.0
            cell_off = 0.0
            for kv in range(pc.v_weights.shape[0]):
                mu_bill = base_mu[0] + pc.M_v_nodes[kv, 0]
                mu_stock = base_mu[1] + pc.M_v_nodes[kv, 1]
                mu_bond = base_mu[2] + pc.M_v_nodes[kv, 2]
                em_bill = math.exp(mu_bill)
                em_stock = math.exp(mu_stock)
                em_bond = math.exp(mu_bond)
                for kr in range(pc.ret_weights.shape[0]):
                    r_bill = em_bill * pc.exp_ret_bill[kr]
                    r_stock = r_bill * em_stock * pc.exp_ret_stock[kr]
                    r_bond = r_bill * em_bond * pc.exp_ret_bond[kr]
                    r_p = a_s * r_stock + a_b * r_bond + a_bill * r_bill
                    w_inv = max(s_val * r_p, 1e-10)

                    if age < model.retire_age:
                        for ke in range(pc.eta_weights.shape[0]):
                            z_next = model.rho * pc.z_grid[z_idx] + pc.eta_nodes[ke]
                            if age == model.retire_age - 1:
                                iz_lo = int((z_next - pc.z_grid[0]) / pc.dz)
                                iz_lo = max(0, min(iz_lo, pc.n_z - 2))
                                frac_z = (z_next - pc.z_grid[iz_lo]) / pc.dz
                                frac_z = max(0.0, min(1.0, frac_z))
                                income = (
                                    (1.0 - frac_z) * pc.pension_after_tax[t + 1, iz_lo]
                                    + frac_z * pc.pension_after_tax[t + 1, iz_lo + 1]
                                )
                                p = pc.v_weights[kv] * pc.ret_weights[kr] * pc.eta_weights[ke]
                                x_next = w_inv + income
                                cell_total += p
                                if x_next > wealth_max:
                                    cell_off += p
                            else:
                                det_z_eta = math.exp(
                                    pc.log_det_profile[t + 1]
                                    + model.rho * pc.z_grid[z_idx]
                                    + pc.eta_nodes[ke]
                                )
                                for ie in range(pc.eps_weights.shape[0]):
                                    p = (
                                        pc.v_weights[kv]
                                        * pc.ret_weights[kr]
                                        * pc.eta_weights[ke]
                                        * pc.eps_weights[ie]
                                    )
                                    x_next = w_inv + scalar_disposable_income(
                                        det_z_eta * math.exp(pc.eps_nodes[ie])
                                    )
                                    cell_total += p
                                    if x_next > wealth_max:
                                        cell_off += p
                            max_x = max(max_x, x_next)
                    else:
                        pension = float(pc.pension_after_tax[t + 1, z_idx])
                        p = pc.v_weights[kv] * pc.ret_weights[kr]
                        x_next = w_inv + pension
                        cell_total += p
                        if x_next > wealth_max:
                            cell_off += p
                        max_x = max(max_x, x_next)

            total_mass += cell_total
            off_mass += cell_off
            cell_off_mass_share = cell_off / cell_total if cell_total > 0 else 0.0
            cell_off_mass_shares.append(cell_off_mass_share)
            if cell_off > 0:
                any_off += 1

    return {
        "wealth": w,
        "cell_any_share": any_off / total_cells,
        "weighted_off_mass": off_mass / total_mass,
        "median_cell_off_mass": float(np.median(cell_off_mass_shares)),
        "p90_cell_off_mass": float(np.quantile(cell_off_mass_shares, 0.9)),
        "median_savings_ratio": float(np.median(savings_ratios)),
        "p90_savings_ratio": float(np.quantile(savings_ratios, 0.9)),
        "max_x_next": max_x,
    }


def _solver_side_quantiles(model, pc, arrays, age: int, w_idx: int):
    C = arrays["C_mat"]
    S = arrays["S_mat"]
    B = arrays["B_mat"]
    t = _age_index(pc, age)

    xs = []
    ws = []
    w = float(pc.wealth_grid[w_idx])

    for z_idx in range(pc.n_z):
        for i_s in range(pc.N_state):
            c = float(C[t, z_idx, i_s, w_idx])
            a_s = float(S[t, z_idx, i_s, w_idx])
            a_b = float(B[t, z_idx, i_s, w_idx])
            s_val = max(w - c, 0.0)
            a_bill = 1.0 - a_s - a_b
            base_mu = pc.const_r + pc.A_r @ pc.state_grid[i_s]

            for kv in range(pc.v_weights.shape[0]):
                mu_bill = base_mu[0] + pc.M_v_nodes[kv, 0]
                mu_stock = base_mu[1] + pc.M_v_nodes[kv, 1]
                mu_bond = base_mu[2] + pc.M_v_nodes[kv, 2]
                em_bill = math.exp(mu_bill)
                em_stock = math.exp(mu_stock)
                em_bond = math.exp(mu_bond)
                for kr in range(pc.ret_weights.shape[0]):
                    r_bill = em_bill * pc.exp_ret_bill[kr]
                    r_stock = r_bill * em_stock * pc.exp_ret_stock[kr]
                    r_bond = r_bill * em_bond * pc.exp_ret_bond[kr]
                    r_p = a_s * r_stock + a_b * r_bond + a_bill * r_bill
                    w_inv = max(s_val * r_p, 1e-10)

                    if age == model.retire_age - 1:
                        for ke in range(pc.eta_weights.shape[0]):
                            z_next = model.rho * pc.z_grid[z_idx] + pc.eta_nodes[ke]
                            iz_lo = int((z_next - pc.z_grid[0]) / pc.dz)
                            iz_lo = max(0, min(iz_lo, pc.n_z - 2))
                            frac_z = (z_next - pc.z_grid[iz_lo]) / pc.dz
                            frac_z = max(0.0, min(1.0, frac_z))
                            income = (
                                (1.0 - frac_z) * pc.pension_after_tax[t + 1, iz_lo]
                                + frac_z * pc.pension_after_tax[t + 1, iz_lo + 1]
                            )
                            xs.append(w_inv + income)
                            ws.append(pc.v_weights[kv] * pc.ret_weights[kr] * pc.eta_weights[ke])
                    else:
                        for ke in range(pc.eta_weights.shape[0]):
                            det_z_eta = math.exp(
                                pc.log_det_profile[t + 1]
                                + model.rho * pc.z_grid[z_idx]
                                + pc.eta_nodes[ke]
                            )
                            for ie in range(pc.eps_weights.shape[0]):
                                xs.append(
                                    w_inv
                                    + scalar_disposable_income(
                                        det_z_eta * math.exp(pc.eps_nodes[ie])
                                    )
                                )
                                ws.append(
                                    pc.v_weights[kv]
                                    * pc.ret_weights[kr]
                                    * pc.eta_weights[ke]
                                    * pc.eps_weights[ie]
                                )

    xs = np.asarray(xs, dtype=float)
    ws = np.asarray(ws, dtype=float)
    mean_x = float(np.sum(xs * ws) / np.sum(ws))
    return {
        "mean_x_next": mean_x,
        "q95_x_next": _weighted_quantile(xs, ws, 0.95),
        "q99_x_next": _weighted_quantile(xs, ws, 0.99),
        "max_x_next": float(xs.max()),
    }


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    model, pc, arrays = _load_saved_run(run_dir)

    wealth_max = float(pc.wealth_grid[-1])
    savings_max = float(pc.s_grid[-1])
    r_min, r_max = _gross_return_envelope(pc)

    print("=" * 78)
    print("WEALTH-GRID TIGHTNESS DIAGNOSTIC")
    print("=" * 78)
    print(f"Run dir       : {run_dir}")
    print(f"wealth_max    : {wealth_max:.6f}")
    print(f"savings_max   : {savings_max:.6f}")
    print(f"gross R range : [{r_min:.6f}, {r_max:.6f}]")
    print()

    print("1. Structural Test: can widen-only solve the top-node problem?")
    coupled_impossible = r_max > 1.0
    print(f"   Because savings_max == wealth_max, the top EGM savings node equals wealth_max.")
    print(f"   Since max gross return > 1 is {'TRUE' if coupled_impossible else 'FALSE'},")
    if coupled_impossible:
        print("   a coupled increase in wealth_max cannot eliminate top-node off-grid x_next.")
        print("   The top savings node will still have some x_next > wealth_max after widening.")
    else:
        print("   widen-only might in principle contain the top node.")
    print()

    print("2. Load-Bearing Extrapolation Scan")
    ages_to_scan = [30, 50, 66, 80]
    n_wealth = len(pc.wealth_grid)
    w_indices = [n_wealth // 2, n_wealth - 5, n_wealth - 3, n_wealth - 1]
    scan_rows = []
    for age in ages_to_scan:
        print(f"   Age {age}")
        for w_idx in w_indices:
            row = _offgrid_stats_for_node(model, pc, arrays, age, w_idx)
            scan_rows.append((age, w_idx, row))
            print(
                f"     w={row['wealth']:8.3f}  "
                f"any_off={row['cell_any_share']:6.1%}  "
                f"weighted_off={row['weighted_off_mass']:6.1%}  "
                f"median_cell={row['median_cell_off_mass']:6.1%}  "
                f"p90_cell={row['p90_cell_off_mass']:6.1%}  "
                f"med_sav={row['median_savings_ratio']:6.1%}  "
                f"p90_sav={row['p90_savings_ratio']:6.1%}"
            )
        print()

    print("3. Decoupled wealth_max sizing proxies (solver-side, not simulation-weighted)")
    sizing_cases = [(50, n_wealth - 5), (50, n_wealth - 1), (66, n_wealth - 5), (66, n_wealth - 1)]
    for age, w_idx in sizing_cases:
        q = _solver_side_quantiles(model, pc, arrays, age, w_idx)
        w = float(pc.wealth_grid[w_idx])
        print(
            f"   Age {age:2d}, w={w:8.3f}: "
            f"mean={q['mean_x_next']:.1f}, q95={q['q95_x_next']:.1f}, "
            f"q99={q['q99_x_next']:.1f}, max={q['max_x_next']:.1f}"
        )
    print()

    print("4. Decision")
    top5_working = [
        row["weighted_off_mass"]
        for age, w_idx, row in scan_rows
        if age in (30, 50, 66) and w_idx == n_wealth - 5
    ]
    severe_top5 = max(top5_working) if top5_working else 0.0

    if severe_top5 > 0.10:
        print("   Leave-as-is: REJECT")
        print("   Reason: extrapolation is load-bearing well below the final wealth node.")
    else:
        print("   Leave-as-is: PLAUSIBLE")
        print("   Reason: off-grid mass looks small outside the extreme upper tail.")

    if coupled_impossible:
        print("   Increase wealth_max only: NOT A STRUCTURAL FIX")
        print("   Reason: with savings_max tied to wealth_max, the top savings node remains off-grid by construction.")
    else:
        print("   Increase wealth_max only: STRUCTURALLY POSSIBLE")

    if severe_top5 > 0.10 and coupled_impossible:
        print("   Recommended option: DECOUPLE savings_max and wealth_max")
        print("   Practical interpretation:")
        print("     - keep the economically relevant savings range fixed")
        print("     - widen wealth_max so continuation lookups are mostly in-grid")
        print("     - then re-solve and re-check solver-side off-grid mass")
    elif severe_top5 > 0.10:
        print("   Recommended option: INCREASE wealth_max first, then reassess")
    else:
        print("   Recommended option: LEAVE GRID AS IS for now")


if __name__ == "__main__":
    main()

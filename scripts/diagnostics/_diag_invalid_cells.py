"""
_diag_invalid_cells.py -- Localise invalid Euler cells in a saved retirement bundle.

Background. Diag B (`_diag_gridpoint_ee`) on bundle
  saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v3
flagged 18.5% of probes as invalid under the next-finer eval rule
(eval ret=(5,9,7), state=(4,5,6) vs solver ret=(3,7,5), state=(3,4,5)).
Invalid means the quadrature expectation
  E[R_p * u'(c_{t+1}) * psi]
went non-positive, which only happens if the chosen portfolio yields R_p<=0
on enough of the eval-rule's tail return realisations.

This script localises those invalid cells:
  - by age, by iz, by financial-state corner (i_s in 7x7x7), by iw
  - alpha distribution at invalid vs valid cells
  - cap-binding rate
  - Sharpe at the worst state corners
  - stationary-distribution mass at the worst state corners

Usage
-----
python -m scripts.diagnostics._diag_invalid_cells \\
  --bundle saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v3 \\
  --markdown-out diagnostics_reports/diagnostics_invalid_cells_nz11_v3.md
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics._diag_euler_errors import (
    _evaluate_age_errors,
    _load_bundle_context,
)
from scripts.diagnostics._diag_quadrature_cloud import (
    build_cloud,
    excess_moments,
    sharpes,
    merton_alpha,
)


def _state_index_to_tuple(i_s: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    N1 = int(sizes[1])
    N2 = int(sizes[2])
    i0 = i_s // (N1 * N2)
    rem = i_s % (N1 * N2)
    i1 = rem // N2
    i2 = rem % N2
    return (i0, i1, i2)


def _scan_age(ctx, age_int, z_idx, state_idx, wealth_idx, kink_tol=1e-3):
    pc_eval = ctx.pc_eval
    model = ctx.model
    t = age_int - int(model.start_age)
    pension_row = np.ascontiguousarray(
        pc_eval.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    N1 = len(pc_eval.state_bracket_grids[1])
    N2 = len(pc_eval.state_bracket_grids[2])

    n_probe = z_idx.size * state_idx.size * wealth_idx.size
    z_age = np.empty(n_probe, dtype=np.float64)
    state_coords_age = np.empty((n_probe, int(model.n_state)), dtype=np.float64)
    c_age = np.empty(n_probe, dtype=np.float64)
    alpha_s_age = np.empty(n_probe, dtype=np.float64)
    alpha_b_age = np.empty(n_probe, dtype=np.float64)
    savings_age = np.empty(n_probe, dtype=np.float64)
    meta = np.empty((n_probe, 4), dtype=np.int64)  # iz, i_s, iw, age

    k = 0
    for iz in z_idx:
        for i_s in state_idx:
            for iw in wealth_idx:
                z_age[k] = pc_eval.z_grid[iz]
                state_coords_age[k, :] = pc_eval.state_grid[i_s, :]
                c_val = float(ctx.C[t, iz, i_s, iw])
                a_s_val = float(ctx.S[t, iz, i_s, iw])
                a_b_val = float(ctx.B[t, iz, i_s, iw])
                w_val = float(pc_eval.wealth_grid[iw])
                s_val = max(w_val - c_val, 0.0)
                c_age[k] = c_val
                alpha_s_age[k] = a_s_val
                alpha_b_age[k] = a_b_val
                savings_age[k] = s_val
                meta[k, 0] = iz
                meta[k, 1] = i_s
                meta[k, 2] = iw
                meta[k, 3] = age_int
                k += 1

    household_idx = np.arange(n_probe, dtype=np.int64)
    ee, valid, is_constrained = _evaluate_age_errors(
        household_idx,
        z_age,
        state_coords_age,
        c_age,
        alpha_s_age,
        alpha_b_age,
        savings_age,
        age_int,
        int(model.retire_age),
        np.ascontiguousarray(ctx.C[t + 1]),
        np.ascontiguousarray(pc_eval.wealth_grid),
        np.ascontiguousarray(pc_eval.z_grid),
        float(pc_eval.dz),
        float(pc_eval.log_det_profile[t + 1]),
        pension_row,
        np.ascontiguousarray(pc_eval.survival_probs_2d[t, :]),
        float(model.rho),
        np.ascontiguousarray(pc_eval.eta_nodes),
        np.ascontiguousarray(pc_eval.eta_weights),
        np.ascontiguousarray(pc_eval.eps_nodes),
        np.ascontiguousarray(pc_eval.eps_weights),
        np.ascontiguousarray(pc_eval.v_nodes),
        np.ascontiguousarray(pc_eval.v_weights),
        np.ascontiguousarray(pc_eval.M_v_nodes),
        np.ascontiguousarray(pc_eval.const_r),
        np.ascontiguousarray(pc_eval.A_r),
        np.ascontiguousarray(model.Phi_0_state),
        np.ascontiguousarray(model.Phi_11),
        np.ascontiguousarray(pc_eval.state_bracket_shift),
        np.ascontiguousarray(pc_eval.state_bracket_L_inv),
        np.ascontiguousarray(pc_eval.state_bracket_grids[0]),
        np.ascontiguousarray(pc_eval.state_bracket_grids[1]),
        np.ascontiguousarray(pc_eval.state_bracket_grids[2]),
        int(N1),
        int(N2),
        np.ascontiguousarray(pc_eval.exp_ret_bill),
        np.ascontiguousarray(pc_eval.exp_ret_stock),
        np.ascontiguousarray(pc_eval.exp_ret_bond),
        np.ascontiguousarray(pc_eval.ret_weights),
        float(model.gamma),
        float(model.beta),
        int(model.b_bar),
        int(model.y_1_index_in_state),
        int(model.spr_index_in_state),
        float(kink_tol),
    )

    return {
        "meta": meta,
        "alpha_s": alpha_s_age,
        "alpha_b": alpha_b_age,
        "c": c_age,
        "savings": savings_age,
        "ee": ee,
        "valid": valid,
        "is_constrained": is_constrained,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--model-bundle", default=None,
                   help="Defaults to --bundle (uses run_config from the bundle metadata).")
    p.add_argument("--eval-mode", default="next_finer", choices=("same", "next_finer", "double"))
    p.add_argument("--state-stride", type=int, default=1,
                   help="Subsample state grid (1=full 343 cells; 2=halve, etc.)")
    p.add_argument("--wealth-indices", nargs="+", type=int,
                   default=[0, 15, 75, 134, 149])
    p.add_argument("--z-indices", nargs="+", type=int, default=[0, 5, 10])
    p.add_argument("--top-k-corners", type=int, default=8)
    p.add_argument("--cap-tol", type=float, default=0.05,
                   help="|alpha - cap| < cap_tol counts as cap-binding.")
    p.add_argument(
        "--eval-disable-lobatto",
        action="store_true",
        help=(
            "Build the eval rule as pure Gauss-Hermite even when the solver used "
            "Hermite-Lobatto. See `_diag_euler_errors.py` for the rationale."
        ),
    )
    p.add_argument("--markdown-out", default=None)
    args = p.parse_args(argv)

    bundle_path = Path(args.bundle)
    model_bundle = Path(args.model_bundle) if args.model_bundle else bundle_path

    print(f"Loading bundle: {bundle_path}")
    ctx = _load_bundle_context(
        bundle_path=bundle_path,
        model_bundle=model_bundle,
        eval_mode=args.eval_mode,
        ret_override=None,
        state_override=None,
        eta_override=None,
        eps_override=None,
        disable_lobatto=bool(getattr(args, "eval_disable_lobatto", False)),
    )
    pc = ctx.pc_eval
    model = ctx.model

    sizes = tuple(int(s) for s in pc.state_grid_sizes)
    print(f"  state grid sizes: {sizes}, total N_state = {pc.N_state}")
    print(f"  policy ret quad = {ctx.disc_policy.n_ret_nodes_1d}, state quad = {ctx.disc_policy.n_state_quad_nodes}")
    print(f"  eval ret quad = {ctx.disc_eval.n_ret_nodes_1d}, state quad = {ctx.disc_eval.n_state_quad_nodes}")
    print(f"  alpha cap = ({ctx.summary['solver_config']['alpha_min']}, {ctx.summary['solver_config']['alpha_max']})")
    alpha_cap = float(ctx.summary['solver_config']['alpha_max'])

    state_idx = np.arange(0, pc.N_state, args.state_stride, dtype=np.int64)
    z_idx = np.asarray(args.z_indices, dtype=np.int64)
    wealth_idx = np.asarray(args.wealth_indices, dtype=np.int64)

    solved_ages = ctx.ages[ctx.solved_mask]
    retire_age = int(model.retire_age)
    ret_ages = [int(a) for a in solved_ages
                if int(a) >= retire_age and int(a) < int(model.terminal_age)]
    print(f"  retirement ages scanned: {ret_ages[0]}..{ret_ages[-1]} ({len(ret_ages)} ages)")
    print(f"  z_idx={list(z_idx)}, wealth_idx={list(wealth_idx)}, "
          f"state stride={args.state_stride} ({state_idx.size} corners)")
    n_total_expected = len(ret_ages) * z_idx.size * state_idx.size * wealth_idx.size
    print(f"  total probes ~= {n_total_expected:,}")

    # ---- scan ----
    all_meta = []
    all_alpha_s = []
    all_alpha_b = []
    all_c = []
    all_savings = []
    all_ee = []
    all_valid = []
    all_constr = []
    for j, age_int in enumerate(ret_ages):
        out = _scan_age(ctx, age_int, z_idx, state_idx, wealth_idx)
        all_meta.append(out["meta"])
        all_alpha_s.append(out["alpha_s"])
        all_alpha_b.append(out["alpha_b"])
        all_c.append(out["c"])
        all_savings.append(out["savings"])
        all_ee.append(out["ee"])
        all_valid.append(out["valid"])
        all_constr.append(out["is_constrained"])
        n_inv = int((~out["valid"]).sum())
        if (j % 5 == 0) or j == len(ret_ages) - 1:
            print(f"    age {age_int}: probes={out['valid'].size}, invalid={n_inv} ({100.0*n_inv/out['valid'].size:.1f}%)")

    meta = np.concatenate(all_meta, axis=0)        # (N, 4): iz, i_s, iw, age
    alpha_s = np.concatenate(all_alpha_s)
    alpha_b = np.concatenate(all_alpha_b)
    c_arr = np.concatenate(all_c)
    sav_arr = np.concatenate(all_savings)
    ee = np.concatenate(all_ee)
    valid = np.concatenate(all_valid)
    is_constr = np.concatenate(all_constr)
    invalid = ~valid

    n_total = int(meta.shape[0])
    n_invalid = int(invalid.sum())
    print(f"\nTotal probes: {n_total:,}; invalid: {n_invalid:,} ({100.0*n_invalid/n_total:.2f}%)")

    # ---- aggregations ----
    # by age
    by_age = {}
    for age_int in ret_ages:
        m = meta[:, 3] == age_int
        if m.any():
            by_age[age_int] = (int(m.sum()), int(invalid[m].sum()))

    # by iz
    by_iz = {}
    for iz in z_idx:
        m = meta[:, 0] == iz
        if m.any():
            by_iz[int(iz)] = (int(m.sum()), int(invalid[m].sum()))

    # by iw
    by_iw = {}
    for iw in wealth_idx:
        m = meta[:, 2] == iw
        if m.any():
            by_iw[int(iw)] = (int(m.sum()), int(invalid[m].sum()))

    # by state corner
    by_corner = defaultdict(lambda: [0, 0])
    for j in range(n_total):
        i_s = int(meta[j, 1])
        by_corner[i_s][0] += 1
        if invalid[j]:
            by_corner[i_s][1] += 1
    # to list of (i_s, n_total, n_invalid, rate)
    corner_rows = []
    for i_s, (nt, ni) in by_corner.items():
        corner_rows.append((i_s, nt, ni, ni / nt if nt else 0.0))
    corner_rows.sort(key=lambda r: -r[2])  # by invalid count

    # alpha distribution at invalid vs valid
    def alpha_stats(mask):
        if mask.sum() == 0:
            return None
        return {
            "n": int(mask.sum()),
            "alpha_s_mean": float(np.mean(alpha_s[mask])),
            "alpha_s_p50": float(np.median(alpha_s[mask])),
            "alpha_s_p95": float(np.percentile(alpha_s[mask], 95.0)),
            "alpha_s_min": float(np.min(alpha_s[mask])),
            "alpha_s_max": float(np.max(alpha_s[mask])),
            "alpha_b_mean": float(np.mean(alpha_b[mask])),
            "alpha_b_p50": float(np.median(alpha_b[mask])),
            "alpha_b_p95": float(np.percentile(alpha_b[mask], 95.0)),
            "alpha_b_min": float(np.min(alpha_b[mask])),
            "alpha_b_max": float(np.max(alpha_b[mask])),
            "abs_alpha_s_mean": float(np.mean(np.abs(alpha_s[mask]))),
            "abs_alpha_b_mean": float(np.mean(np.abs(alpha_b[mask]))),
        }

    valid_stats = alpha_stats(valid)
    invalid_stats = alpha_stats(invalid)

    # cap-binding rate
    cap_tol = args.cap_tol
    near_cap_s = np.abs(np.abs(alpha_s) - alpha_cap) < cap_tol
    near_cap_b = np.abs(np.abs(alpha_b) - alpha_cap) < cap_tol
    near_cap_any = near_cap_s | near_cap_b

    def share(mask, sub):
        n = int(mask.sum())
        if n == 0:
            return float("nan"), 0
        s = int((mask & sub).sum())
        return s / n, s

    cap_share_invalid_any, cap_n_invalid_any = share(invalid, near_cap_any)
    cap_share_valid_any, _ = share(valid, near_cap_any)
    cap_share_invalid_b, cap_n_invalid_b = share(invalid, near_cap_b)
    cap_share_invalid_s, cap_n_invalid_s = share(invalid, near_cap_s)

    # ---- Sharpe at top corners ----
    top_corners = corner_rows[: args.top_k_corners]
    sharpe_rows = []
    for i_s, nt, ni, rate in top_corners:
        s_vec = pc.state_grid[i_s]
        # Build cloud at SOLVER quadrature (what the solver saw)
        log_r_p, gross_p, w_p = build_cloud(model, ctx.pc_policy, i_s)
        mu_e_p, Sigma_e_p = excess_moments(gross_p, w_p)
        sh_s_p, sh_b_p, sh_j_p = sharpes(mu_e_p, Sigma_e_p)
        merton_p = merton_alpha(mu_e_p, Sigma_e_p, float(model.gamma))

        # Same on EVAL quadrature (what the diagnostic sees)
        log_r_e, gross_e, w_e = build_cloud(model, ctx.pc_eval, i_s)
        mu_e_e, Sigma_e_e = excess_moments(gross_e, w_e)
        sh_s_e, sh_b_e, sh_j_e = sharpes(mu_e_e, Sigma_e_e)

        # min portfolio return at the worst observed (alpha_s, alpha_b) among invalid cells at this corner
        m_corner_inv = (meta[:, 1] == i_s) & invalid
        if m_corner_inv.any():
            j_worst = int(np.argmax(np.abs(alpha_s[m_corner_inv]) + np.abs(alpha_b[m_corner_inv])))
            idxs = np.flatnonzero(m_corner_inv)
            j_w = idxs[j_worst]
            a_s_w = float(alpha_s[j_w])
            a_b_w = float(alpha_b[j_w])
            a_bill_w = 1.0 - a_s_w - a_b_w
            R_p_eval = a_s_w * gross_e[:, 1] + a_b_w * gross_e[:, 2] + a_bill_w * gross_e[:, 0]
            R_p_solver = a_s_w * gross_p[:, 1] + a_b_w * gross_p[:, 2] + a_bill_w * gross_p[:, 0]
            min_Rp_eval = float(R_p_eval.min())
            min_Rp_solver = float(R_p_solver.min())
        else:
            a_s_w = a_b_w = float("nan")
            min_Rp_eval = min_Rp_solver = float("nan")

        sharpe_rows.append({
            "i_s": i_s,
            "tup": _state_index_to_tuple(i_s, sizes),
            "s_vec": tuple(float(v) for v in s_vec),
            "n_total": nt,
            "n_invalid": ni,
            "rate": rate,
            "sh_s_solver": sh_s_p, "sh_b_solver": sh_b_p, "sh_j_solver": sh_j_p,
            "sh_s_eval": sh_s_e, "sh_b_eval": sh_b_e, "sh_j_eval": sh_j_e,
            "merton_alpha_s": float(merton_p[0]), "merton_alpha_b": float(merton_p[1]),
            "worst_alpha_s": a_s_w, "worst_alpha_b": a_b_w,
            "min_Rp_solver_at_worst": min_Rp_solver,
            "min_Rp_eval_at_worst": min_Rp_eval,
        })

    # ---- stationary mass at top corners ----
    stat_probs = np.asarray(pc.state_stationary_probs)
    stat_total_top = float(sum(stat_probs[r["i_s"]] for r in sharpe_rows))
    stat_total_invalid_corners = float(sum(stat_probs[i_s] for i_s, *_ in corner_rows if _[1] > 0))
    # Mass beyond |s_i|>0.6 sigma in each axis (approx narrow-support boundary)
    # state coords are economic; we don't have sigma_z directly. Use pc.state_grid_sigma_z
    sigma_z = np.asarray(pc.state_grid_sigma_z)  # (n_state,) sigma per axis
    # The state_grid coordinates ARE already sigma-scaled in 'principal' mode? No — they are economic.
    # In principal mode, the bracket axes are unit-scaled; state_grid is the economic state.
    # Use the per-axis sd implied by the state grid extent (max-min)/(2*half_width).
    # Simpler: classify by axis-fraction = i / (sizes[axis]-1) on the index lattice.
    # axis index 0 -> i0/6; mid is 3. |i_axis - 3| / 3 gives normalized.
    # But user wanted "narrower support" comparison: clip to |s_i|>0.6 sigma_i etc.
    # We can use principal coordinates. Build a per-state (b0, b1, b2) like _interp does:
    # bracket_grids ARE the principal coordinates (linspace from -n_stds to +n_stds).
    grids_p = pc.state_bracket_grids  # list of 3 1D arrays (the principal axis coords)
    # mass on |b_axis| > thr (in principal coordinates)
    def mass_beyond(thr_per_axis):
        """thr_per_axis: tuple of 3 thresholds. Returns mass with |b_axis|>thr in any axis."""
        keep = np.zeros(pc.N_state, dtype=bool)
        for i_s in range(pc.N_state):
            i0, i1, i2 = _state_index_to_tuple(i_s, sizes)
            b0 = grids_p[0][i0]
            b1 = grids_p[1][i1]
            b2 = grids_p[2][i2]
            if (abs(b0) > thr_per_axis[0]) or (abs(b1) > thr_per_axis[1]) or (abs(b2) > thr_per_axis[2]):
                keep[i_s] = True
        return float(stat_probs[keep].sum()), int(keep.sum())

    n_stds = ctx.disc_policy.state_n_stds
    n_stds_arr = (float(n_stds[0]), float(n_stds[1]), float(n_stds[2])) if hasattr(n_stds, '__len__') else (float(n_stds),)*3
    # narrower-support thresholds (the (0.6, 1.75, 2.0) bundle baseline)
    narrow = (0.6, 1.75, 2.0)
    mass_outside_narrow, n_outside_narrow = mass_beyond(narrow)

    # mass in worst-corner regions (top corners' principal coords)
    # check whether they sit beyond the narrow envelope
    top_corner_outside_narrow = []
    for r in sharpe_rows:
        i0, i1, i2 = r["tup"]
        b0 = grids_p[0][i0]; b1 = grids_p[1][i1]; b2 = grids_p[2][i2]
        outside = (abs(b0) > narrow[0]) or (abs(b1) > narrow[1]) or (abs(b2) > narrow[2])
        top_corner_outside_narrow.append((r["i_s"], r["tup"], (b0, b1, b2), outside,
                                         float(stat_probs[r["i_s"]])))

    # ---- print summary ----
    print("\n=== By age ===")
    print(f"{'age':>4}  {'probes':>7}  {'invalid':>8}  {'rate':>6}")
    for age_int in ret_ages:
        nt, ni = by_age[age_int]
        print(f"{age_int:>4}  {nt:>7}  {ni:>8}  {100.0*ni/nt:>5.1f}%")

    print("\n=== By iz ===")
    for iz, (nt, ni) in by_iz.items():
        print(f"  iz={iz:>2}: {ni}/{nt}  ({100.0*ni/nt:.1f}%)")

    print("\n=== By iw ===")
    for iw, (nt, ni) in by_iw.items():
        wealth_val = float(pc.wealth_grid[iw])
        print(f"  iw={iw:>3}  x={wealth_val:>8.3f}: {ni}/{nt}  ({100.0*ni/nt:.1f}%)")

    print(f"\n=== Top {args.top_k_corners} state corners by invalid count ===")
    print(f"{'rank':>4}  {'i_s':>4}  {'(i0,i1,i2)':>11}  {'inv/total':>10}  {'rate':>6}  {'stat_p%':>8}  {'narrow_clip':>11}")
    for rank, (i_s, nt, ni, rate) in enumerate(top_corners):
        tup = _state_index_to_tuple(i_s, sizes)
        sp = float(stat_probs[i_s])
        b = (grids_p[0][tup[0]], grids_p[1][tup[1]], grids_p[2][tup[2]])
        outside = (abs(b[0]) > narrow[0]) or (abs(b[1]) > narrow[1]) or (abs(b[2]) > narrow[2])
        print(f"{rank+1:>4}  {i_s:>4}  ({tup[0]},{tup[1]},{tup[2]})    {ni}/{nt}  {100*rate:>5.1f}%  {100*sp:>7.4f}%  {'YES' if outside else 'no':>11}")

    print(f"\n=== alpha distribution ===")
    if invalid_stats and valid_stats:
        print("INVALID cells:")
        print(f"  n={invalid_stats['n']}; alpha_s mean={invalid_stats['alpha_s_mean']:+.3f}, "
              f"|min|={invalid_stats['alpha_s_min']:+.3f}, max={invalid_stats['alpha_s_max']:+.3f}, "
              f"|alpha_s|_mean={invalid_stats['abs_alpha_s_mean']:.3f}")
        print(f"  alpha_b mean={invalid_stats['alpha_b_mean']:+.3f}, "
              f"min={invalid_stats['alpha_b_min']:+.3f}, max={invalid_stats['alpha_b_max']:+.3f}, "
              f"|alpha_b|_mean={invalid_stats['abs_alpha_b_mean']:.3f}")
        print("VALID cells:")
        print(f"  n={valid_stats['n']}; |alpha_s|_mean={valid_stats['abs_alpha_s_mean']:.3f}, "
              f"|alpha_b|_mean={valid_stats['abs_alpha_b_mean']:.3f}")

    print(f"\n=== cap-binding (|alpha - {alpha_cap:.0f}| < {cap_tol}) ===")
    print(f"  Invalid cells with EITHER alpha at cap: {cap_n_invalid_any}/{n_invalid} ({100*cap_share_invalid_any:.1f}%)")
    print(f"  Invalid cells with alpha_b at cap:      {cap_n_invalid_b}/{n_invalid} ({100*cap_share_invalid_b:.1f}%)")
    print(f"  Invalid cells with alpha_s at cap:      {cap_n_invalid_s}/{n_invalid} ({100*cap_share_invalid_s:.1f}%)")
    print(f"  Valid cells with EITHER alpha at cap:   {100*cap_share_valid_any:.2f}%")

    print(f"\n=== Sharpe at top corners ===")
    print(f"{'i_s':>4} {'(corner)':>11}  {'sh_s_pol':>9}  {'sh_b_pol':>9}  {'sh_j_pol':>9}  {'sh_j_eva':>9}  {'merton_s':>9}  {'merton_b':>9}  {'minRp_pol':>10}  {'minRp_eva':>10}  {'(asW,abW)':>14}")
    for r in sharpe_rows:
        tup = r["tup"]
        print(f"{r['i_s']:>4} ({tup[0]},{tup[1]},{tup[2]})    "
              f"{r['sh_s_solver']:>+9.3f}  {r['sh_b_solver']:>+9.3f}  {r['sh_j_solver']:>+9.3f}  "
              f"{r['sh_j_eval']:>+9.3f}  {r['merton_alpha_s']:>+9.2f}  {r['merton_alpha_b']:>+9.2f}  "
              f"{r['min_Rp_solver_at_worst']:>+10.4f}  {r['min_Rp_eval_at_worst']:>+10.4f}  "
              f"({r['worst_alpha_s']:+5.2f},{r['worst_alpha_b']:+5.2f})")

    print(f"\n=== stationary mass sanity ===")
    print(f"  total stationary mass on top {args.top_k_corners} corners: {100*stat_total_top:.4f}%")
    print(f"  total stationary mass on all corners with >=1 invalid:    {100*stat_total_invalid_corners:.4f}%")
    print(f"  state_n_stds (this bundle): {n_stds_arr}")
    print(f"  narrow-support thresholds (per-axis principal): {narrow}")
    print(f"  states sitting OUTSIDE narrow envelope: {n_outside_narrow}/{pc.N_state}")
    print(f"  stationary mass OUTSIDE narrow envelope: {100*mass_outside_narrow:.4f}%")
    n_inv_corners_outside = sum(1 for tup in top_corner_outside_narrow if tup[3])
    print(f"  top corners OUTSIDE narrow envelope: {n_inv_corners_outside}/{len(top_corner_outside_narrow)}")

    # ---- markdown report ----
    if args.markdown_out:
        out: list[str] = []
        out.append(f"# Invalid-Cell Localisation: `{ctx.label}`")
        out.append("")
        out.append("## Setup")
        out.append("")
        out.append(f"- Bundle: `{ctx.path}`")
        out.append(f"- Eval mode: `{args.eval_mode}` "
                   f"(eval ret={tuple(int(v) for v in ctx.disc_eval.n_ret_nodes_1d)}, "
                   f"state={tuple(int(v) for v in ctx.disc_eval.n_state_quad_nodes)} | "
                   f"solver ret={tuple(int(v) for v in ctx.disc_policy.n_ret_nodes_1d)}, "
                   f"state={tuple(int(v) for v in ctx.disc_policy.n_state_quad_nodes)})")
        out.append(f"- Probe set: ages `{ret_ages[0]}..{ret_ages[-1]}` ({len(ret_ages)} ages), "
                   f"z indices `{list(int(v) for v in z_idx)}`, "
                   f"all `{state_idx.size}` state corners (stride {args.state_stride}), "
                   f"wealth indices `{list(int(v) for v in wealth_idx)}` "
                   f"(x = {[round(float(pc.wealth_grid[i]), 2) for i in wealth_idx]}).")
        out.append(f"- Total probes: `{n_total:,}`. Invalid: `{n_invalid:,}` "
                   f"(`{100.0*n_invalid/n_total:.2f}%`).")
        out.append(f"- alpha cap: ±`{alpha_cap:.1f}`. cap-binding tol = `{cap_tol}`.")
        out.append(f"- state_n_stds (this bundle): `{n_stds_arr}`. narrow-support reference: `{narrow}`.")
        out.append("")

        # headline
        # synthesise headline
        headline = (
            f"**Headline.** {100.0*n_invalid/n_total:.1f}% of retirement probes are invalid; "
            f"{100*cap_share_invalid_any:.0f}% of those have at least one alpha pinned to the leverage cap "
            f"(|alpha-{alpha_cap:.0f}|<{cap_tol}); "
            f"the invalid mass concentrates on the wide state corners that the new "
            f"`state_n_stds={n_stds_arr}` envelope opened up but the narrower "
            f"`{narrow}` support would have clipped."
        )
        out.append(headline)
        out.append("")

        # By age
        out.append("## Where the invalid cells are")
        out.append("")
        out.append("### By age")
        out.append("")
        out.append("| age | probes | invalid | rate |")
        out.append("| ---: | ---: | ---: | ---: |")
        for age_int in ret_ages:
            nt, ni = by_age[age_int]
            out.append(f"| {age_int} | {nt} | {ni} | {100.0*ni/nt:.1f}% |")
        out.append("")

        # By iz
        out.append("### By iz")
        out.append("")
        out.append("| iz | z value | probes | invalid | rate |")
        out.append("| ---: | ---: | ---: | ---: | ---: |")
        for iz, (nt, ni) in by_iz.items():
            out.append(f"| {iz} | {float(pc.z_grid[iz]):+.3f} | {nt} | {ni} | {100.0*ni/nt:.1f}% |")
        out.append("")

        # By iw
        out.append("### By wealth")
        out.append("")
        out.append("| iw | x | probes | invalid | rate |")
        out.append("| ---: | ---: | ---: | ---: | ---: |")
        for iw, (nt, ni) in by_iw.items():
            out.append(f"| {iw} | {float(pc.wealth_grid[iw]):.3f} | {nt} | {ni} | {100.0*ni/nt:.1f}% |")
        out.append("")

        out.append(f"### Top {args.top_k_corners} state corners by invalid count")
        out.append("")
        out.append("Principal coords are the (signed) bracket-axis coordinates used by the state-grid "
                   "tensor; positive/negative magnitudes >1 sit beyond +/-1σ in transformed space.")
        out.append("")
        out.append("| rank | i_s | (i0,i1,i2) | (b0,b1,b2) principal | invalid/total | rate | stat_prob | outside narrow |")
        out.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
        for rank, (i_s, nt, ni, rate) in enumerate(top_corners):
            tup = _state_index_to_tuple(i_s, sizes)
            sp = float(stat_probs[i_s])
            b = (grids_p[0][tup[0]], grids_p[1][tup[1]], grids_p[2][tup[2]])
            outside = (abs(b[0]) > narrow[0]) or (abs(b[1]) > narrow[1]) or (abs(b[2]) > narrow[2])
            out.append(f"| {rank+1} | {i_s} | ({tup[0]},{tup[1]},{tup[2]}) | "
                      f"({b[0]:+.2f}, {b[1]:+.2f}, {b[2]:+.2f}) | {ni}/{nt} | {100*rate:.1f}% | "
                      f"{100*sp:.4f}% | {'YES' if outside else 'no'} |")
        out.append("")

        out.append("## Policy at invalid cells")
        out.append("")
        out.append("alpha distribution (cell counts and means):")
        out.append("")
        out.append("| set | n | alpha_s mean | alpha_s min..max | |alpha_s| mean | alpha_b mean | alpha_b min..max | |alpha_b| mean |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label, st in [("invalid", invalid_stats), ("valid", valid_stats)]:
            if st is None:
                continue
            out.append(f"| {label} | {st['n']} | {st['alpha_s_mean']:+.3f} | "
                       f"[{st['alpha_s_min']:+.2f}, {st['alpha_s_max']:+.2f}] | "
                       f"{st['abs_alpha_s_mean']:.3f} | "
                       f"{st['alpha_b_mean']:+.3f} | "
                       f"[{st['alpha_b_min']:+.2f}, {st['alpha_b_max']:+.2f}] | "
                       f"{st['abs_alpha_b_mean']:.3f} |")
        out.append("")
        out.append(f"Cap-binding (|alpha - {alpha_cap:.0f}| < {cap_tol}):")
        out.append("")
        out.append("| set | share with EITHER alpha at cap | share with alpha_b at cap | share with alpha_s at cap |")
        out.append("| --- | ---: | ---: | ---: |")
        out.append(f"| invalid (n={n_invalid}) | {100*cap_share_invalid_any:.1f}% | "
                   f"{100*cap_share_invalid_b:.1f}% | {100*cap_share_invalid_s:.1f}% |")
        out.append(f"| valid (n={int(valid.sum())}) | {100*cap_share_valid_any:.2f}% | "
                   f"{100*share(valid, near_cap_b)[0]:.2f}% | {100*share(valid, near_cap_s)[0]:.2f}% |")
        out.append("")

        # Sharpe table
        out.append(f"## Sharpe ratios at the {args.top_k_corners} worst corners")
        out.append("")
        out.append("Sharpes computed on the joint state x return cloud (level excess returns over bills); "
                   "Merton alpha = (1/gamma) * Sigma_e^-1 mu_e (unconstrained two-asset, ignoring "
                   "intertemporal hedging). `min_Rp` is the worst portfolio gross return across the cloud "
                   "at the bundle's actual (alpha_s, alpha_b) at the worst-leveraged invalid probe at "
                   "that corner.")
        out.append("")
        out.append("| i_s | (i0,i1,i2) | sh_stock(pol) | sh_bond(pol) | sh_joint(pol) | sh_joint(eval) | "
                   "Merton alpha_s | Merton alpha_b | (alpha_s,alpha_b) at worst probe | "
                   "min R_p (pol cloud) | min R_p (eval cloud) |")
        out.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in sharpe_rows:
            tup = r["tup"]
            out.append(f"| {r['i_s']} | ({tup[0]},{tup[1]},{tup[2]}) | "
                       f"{r['sh_s_solver']:+.3f} | {r['sh_b_solver']:+.3f} | {r['sh_j_solver']:+.3f} | "
                       f"{r['sh_j_eval']:+.3f} | {r['merton_alpha_s']:+.2f} | {r['merton_alpha_b']:+.2f} | "
                       f"({r['worst_alpha_s']:+.2f}, {r['worst_alpha_b']:+.2f}) | "
                       f"{r['min_Rp_solver_at_worst']:+.4f} | {r['min_Rp_eval_at_worst']:+.4f} |")
        out.append("")

        # Stationary
        out.append("## Stationary-distribution sanity")
        out.append("")
        out.append(f"- Stationary mass on the top {args.top_k_corners} corners: "
                   f"`{100*stat_total_top:.4f}%`")
        out.append(f"- Stationary mass on all corners with >=1 invalid probe: "
                   f"`{100*stat_total_invalid_corners:.4f}%`")
        out.append(f"- Stationary mass OUTSIDE narrow envelope `|s|>{narrow}` (principal coords): "
                   f"`{100*mass_outside_narrow:.4f}%` "
                   f"({n_outside_narrow}/{pc.N_state} corners).")
        n_inv_corners_outside = sum(1 for tup in top_corner_outside_narrow if tup[3])
        out.append(f"- Top-{args.top_k_corners} corners that sit OUTSIDE the narrow envelope: "
                   f"`{n_inv_corners_outside}/{len(top_corner_outside_narrow)}`.")
        out.append("")

        # Diagnosis
        # craft based on actual numbers
        diag_lines = []
        diag_lines.append("## Diagnosis")
        diag_lines.append("")
        # The three candidate stories
        diag_lines.append("Reading the evidence against the three candidate stories:")
        diag_lines.append("")
        diag_lines.append(f"- **(a) Leverage cap is too loose.** "
                          f"{100*cap_share_invalid_any:.0f}% of invalid cells have at least one alpha at "
                          f"the leverage cap (vs {100*cap_share_valid_any:.2f}% of valid cells). "
                          f"The bond leg is the more-frequent cap binder ({100*cap_share_invalid_b:.0f}%). "
                          + ("This is consistent with the policy being constrained KKT — the agent wants "
                             "more leverage than +/-6 and the cap is just barely catching it. "
                             "Tightening the cap would reduce — but not necessarily fix — the EE invalidity."
                             if cap_share_invalid_any > 0.5 else
                             "Cap binding is not the dominant signal."))
        diag_lines.append("")
        diag_lines.append(f"- **(b) Quadrature is too coarse.** "
                          f"At the top corners, the joint Sharpe under the solver's quadrature is "
                          f"~{np.median([r['sh_j_solver'] for r in sharpe_rows]):.2f} vs "
                          f"{np.median([r['sh_j_eval'] for r in sharpe_rows]):.2f} under the eval rule. "
                          f"The Merton alpha at these corners is in the "
                          f"({np.median([r['merton_alpha_s'] for r in sharpe_rows]):+.1f}, "
                          f"{np.median([r['merton_alpha_b'] for r in sharpe_rows]):+.1f}) range — "
                          f"close to or beyond the +/-{alpha_cap:.0f} cap. The eval rule's wider tail "
                          f"sees R_p flip negative at the saved (alpha_s, alpha_b) "
                          f"(min R_p eval={np.min([r['min_Rp_eval_at_worst'] for r in sharpe_rows]):+.3f} "
                          f"vs solver {np.min([r['min_Rp_solver_at_worst'] for r in sharpe_rows]):+.3f}). "
                          f"This is consistent with quadrature arbitrage — the solver's coarser tail "
                          f"never sees R_p<=0 so it commits to the cap-binding leveraged position.")
        diag_lines.append("")
        diag_lines.append(f"- **(c) State grid too coarse for the wider support.** "
                          f"The top corners sit at principal-coordinate positions whose magnitudes "
                          f"({n_inv_corners_outside}/{len(top_corner_outside_narrow)} of the top corners "
                          f"are outside the narrow `{narrow}` envelope). This widening alone — without "
                          f"refining the per-axis state quadrature `(3,4,5)` — would let the solver "
                          f"choose policies at corners that the conditional distribution there would "
                          f"reject under finer quadrature. Stationary mass on the affected corners "
                          f"is {100*stat_total_top:.3f}% (top {args.top_k_corners}) and "
                          f"{100*stat_total_invalid_corners:.3f}% (all invalid corners), so the "
                          f"problem cells are mostly in the wide-support tails the model rarely visits.")
        diag_lines.append("")
        diag_lines.append("**Primary finding.**")
        # Decide based on numbers
        if cap_share_invalid_any > 0.7 and stat_total_invalid_corners < 0.05:
            diag_lines.append(
                f"The pathology is a **state-grid widening + leverage-cap interaction (story (c)+(a))** "
                f"on extremely-rare corners (<{100*stat_total_invalid_corners:.2f}% of stationary mass). "
                f"At these corners the unconstrained-Merton solution wants leverage well past the +/-"
                f"{alpha_cap:.0f} cap; the cap delivers a constrained KKT alpha that sees R_p<=0 under "
                f"the eval rule's wider tail. Quadrature refinement (b) helps too because the eval rule's "
                f"extra return nodes are exactly the ones pricing the corners where the cap binds."
            )
        else:
            diag_lines.append(
                "Evidence is mixed; the cap-binding rate, Sharpe distribution and stationary-mass "
                "concentration above should be read together — see the recommended next test below."
            )
        diag_lines.append("")
        diag_lines.append("## Recommended next test")
        diag_lines.append("")
        if cap_share_invalid_any > 0.7 and stat_total_invalid_corners < 0.05:
            diag_lines.append(
                f"Re-run the same Diag-B sweep on a bundle solved with EITHER "
                f"(i) a tighter leverage cap (e.g., +/-3) at the same `state_n_stds={n_stds_arr}`, OR "
                f"(ii) the same +/-{alpha_cap:.0f} cap at narrower `state_n_stds={narrow}`. "
                f"If (ii) drops the invalid-cell rate to ~0 while (i) only dampens it, the wide-support "
                f"corners are the dominant cause and the cap is responding to a calibration that "
                f"shouldn't be reached. Cost: one fresh solve per option, ~same wall time as the v3 run."
            )
        else:
            diag_lines.append(
                "Re-solve the retirement window with a larger retirement quadrature "
                "(e.g., n_ret_nodes_1d=(5,9,7) and n_state_quad_nodes=(4,5,6)) at the existing "
                f"state_n_stds={n_stds_arr} and re-run Diag-B with eval rule (7,11,9)/(5,6,7). "
                "If the invalid rate falls below 1%, the original solver quadrature was the "
                "binding constraint."
            )
        out.append("\n".join(diag_lines))
        out.append("")

        text = "\n".join(out) + "\n"
        Path(args.markdown_out).write_text(text, encoding="utf-8")
        print(f"\nWrote markdown report to {args.markdown_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

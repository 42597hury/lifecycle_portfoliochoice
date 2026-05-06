"""Compare ccv_retire (narrow Lobatto) vs ccv_wide_gh (wide GH) per-cell.

Three diagnostic questions:
1. Per-cell EE histogram, side by side. Does the body shift or just the tail?
2. Where do ccv_wide_gh's worst cells sit in physical state space? Are they
   at coords that ccv_retire's grid clipped, or in shared territory?
3. Alpha distribution in the worst-cell tail. Are these cap-bound, leveraged,
   or modest-alpha cells?

Uses each bundle's own simulation (same RNG seed, same initial conditions).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics._diag_euler_errors import (
    _annuity_factors_per_household,
    _evaluate_age_errors,
    _load_bundle_context,
    _pad_eval_state_inputs_to_3d,
    _simulate_bundle_window,
)


def evaluate_bundle(bundle_path: Path, label: str, n_sim: int = 5000, eval_cap: int = 256):
    """Run sim + per-cell EE evaluation for one bundle. Returns list of cell records."""
    print(f'\n=== {label}: {bundle_path.name} ===')

    ctx = _load_bundle_context(
        bundle_path=bundle_path, model_bundle=bundle_path,
        eval_mode='next_finer',
        ret_override=None, state_override=None,
        eta_override=None, eps_override=None,
    )
    pc = ctx.pc_eval
    model = ctx.model
    sizes = tuple(int(s) for s in pc.state_grid_sizes)

    sim = _simulate_bundle_window(
        ctx, n_simulations=n_sim, seed=42, return_draw_mode='monte_carlo',
        initial_x=None, initial_wealth=None, initial_wealth_distribution=None,
        initial_wealth_normal_std=0.0, initial_z='stationary', initial_z_normal_std=0.0,
        initial_state='median',
    )

    z_h = sim['z']; s_h = sim['state_coords']; x_h = sim['x']
    c_h = sim['c']; a_s_h = sim['alpha_s']; a_b_h = sim['alpha_b']
    sav_h = sim['savings']; alive_h = sim['alive']

    n_total, n_age = z_h.shape
    print(f'  Simulated {n_total} agents x {n_age} ages')

    pension_row = np.ascontiguousarray(
        pc.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    eval_pad = _pad_eval_state_inputs_to_3d(pc, model)
    first_t = ctx.first_solved_t

    rows = []
    for t in range(n_age - 1):
        model_t = first_t + t
        age = int(ctx.ages[model_t])
        if age >= int(model.terminal_age) or model_t + 1 >= ctx.C.shape[0]:
            continue
        is_alive = alive_h[:, t]
        idx_alive = np.flatnonzero(is_alive)
        if len(idx_alive) == 0:
            continue
        if len(idx_alive) > eval_cap:
            rng = np.random.default_rng(seed=42 + t)
            idx_alive = rng.choice(idx_alive, size=eval_cap, replace=False)
        n_eval = len(idx_alive)

        z_age = np.ascontiguousarray(z_h[idx_alive, t])
        s_age = np.ascontiguousarray(s_h[idx_alive, t])
        c_age = np.ascontiguousarray(c_h[idx_alive, t])
        a_s = np.ascontiguousarray(a_s_h[idx_alive, t])
        a_b = np.ascontiguousarray(a_b_h[idx_alive, t])
        sav = np.ascontiguousarray(sav_h[idx_alive, t])
        x_age = x_h[idx_alive, t]
        annuity_age = _annuity_factors_per_household(model, s_age)

        ee, valid, is_constr = _evaluate_age_errors(
            np.arange(n_eval, dtype=np.int64),
            z_age, s_age, c_age, a_s, a_b, sav,
            int(age), int(model.retire_age),
            np.ascontiguousarray(ctx.C[model_t + 1]),
            np.ascontiguousarray(pc.wealth_grid),
            np.ascontiguousarray(pc.z_grid),
            float(pc.dz),
            float(pc.log_det_profile[model_t + 1]),
            pension_row,
            np.ascontiguousarray(pc.survival_probs_2d[model_t, :]),
            float(model.rho),
            np.ascontiguousarray(pc.eta_nodes), np.ascontiguousarray(pc.eta_weights),
            np.ascontiguousarray(pc.eps_nodes), np.ascontiguousarray(pc.eps_weights),
            eval_pad["v_nodes"], np.ascontiguousarray(pc.v_weights),
            np.ascontiguousarray(pc.M_v_nodes),
            np.ascontiguousarray(pc.const_r),
            eval_pad["A_r"], eval_pad["Phi_0_state"], eval_pad["Phi_11"],
            eval_pad["state_bracket_shift"], eval_pad["state_bracket_L_inv"],
            eval_pad["state_grids_0"], eval_pad["state_grids_1"], eval_pad["state_grids_2"],
            int(eval_pad["N0"]), int(eval_pad["N1"]), int(eval_pad["N2"]),
            np.ascontiguousarray(pc.exp_ret_bill), np.ascontiguousarray(pc.exp_ret_stock),
            np.ascontiguousarray(pc.exp_ret_bond), np.ascontiguousarray(pc.ret_weights),
            np.ascontiguousarray(pc.ret_nodes),
            float(pc.sigma2_xr), float(pc.sigma2_xb), float(pc.sigma_xrxb),
            bool(ctx.use_ccv),
            float(model.gamma), float(model.beta),
            int(model.b_bar),
            annuity_age,
            float(1e-3),
            delta=ctx.delta_bequest,
        )

        for j in range(n_eval):
            if not bool(valid[j]):
                continue
            ee_val = float(ee[j])
            abs_ee = abs(ee_val)
            log10_abs = float(np.log10(max(abs_ee, 1e-16)))
            rows.append({
                'age': age,
                'z': float(z_age[j]),
                's0': float(s_age[j, 0]),
                's1': float(s_age[j, 1]),
                's2': float(s_age[j, 2]),
                'x': float(x_age[j]),
                'c': float(c_age[j]),
                'sav': float(sav[j]),
                'sav_share': float(sav[j] / max(x_age[j], 1e-12)),
                'alpha_s': float(a_s[j]),
                'alpha_b': float(a_b[j]),
                'is_constrained': bool(is_constr[j]),
                'log10_abs_ee': log10_abs,
                'abs_ee': abs_ee,
            })
    print(f'  Total cells with valid EE: {len(rows)}')
    return rows


def main():
    bundles = {
        'ccv_retire': Path('saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_retire'),
        'ccv_wide_gh': Path('saved_runs/system_iv_full_var_unconstrained_cholesky_grid7x7x7_nz11_ccv_wide_gh'),
    }

    results = {}
    for label, path in bundles.items():
        results[label] = evaluate_bundle(path, label)

    # Q1: Per-cell EE histogram side by side
    print('\n' + '=' * 78)
    print('Q1. Per-cell log10|EE| distribution comparison')
    print('=' * 78)
    print(f'{"percentile":<14} {"ccv_retire":>13} {"ccv_wide_gh":>13} {"delta":>9}')
    for label, rows in results.items():
        eevals = np.array([r['log10_abs_ee'] for r in rows])
    rcr = np.array([r['log10_abs_ee'] for r in results['ccv_retire']])
    rwd = np.array([r['log10_abs_ee'] for r in results['ccv_wide_gh']])
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        v_cr = np.percentile(rcr, q) if q != 100 else rcr.max()
        v_wd = np.percentile(rwd, q) if q != 100 else rwd.max()
        print(f'p{q:<13} {v_cr:>13.4f} {v_wd:>13.4f} {v_wd - v_cr:>+9.4f}')
    print(f'{"mean":<14} {rcr.mean():>13.4f} {rwd.mean():>13.4f} {rwd.mean() - rcr.mean():>+9.4f}')

    # Q2: where do ccv_wide_gh's worst cells sit in state space
    print('\n' + '=' * 78)
    print('Q2. Where do ccv_wide_gh worst cells sit (state coordinates)?')
    print('=' * 78)
    rwd_sorted = sorted(results['ccv_wide_gh'], key=lambda r: -r['log10_abs_ee'])
    print(f'\nTop 20 worst cells from ccv_wide_gh:')
    print(f'{"age":>3} {"z":>8} {"s0":>8} {"s1":>8} {"s2":>8} {"x":>8} {"alpha_s":>7} {"alpha_b":>7} {"sav%":>5} {"|EE|":>8} {"log10":>7}')
    for r in rwd_sorted[:20]:
        print(f'{r["age"]:>3} {r["z"]:>+8.3f} {r["s0"]:>+8.3f} {r["s1"]:>+8.3f} {r["s2"]:>+8.3f} '
              f'{r["x"]:>8.2f} {r["alpha_s"]:>+7.3f} {r["alpha_b"]:>+7.3f} '
              f'{100*r["sav_share"]:>4.1f} {r["abs_ee"]:>8.4f} {r["log10_abs_ee"]:>+7.3f}')

    # For each top-cell, check: is this state coord inside ccv_retire's grid envelope?
    # ccv_retire uses state_n_stds=(2.0, 2.25, 2.25). To check, we need ccv_retire's stationary mu and L_z
    # State point is "in the narrow envelope" iff its standardised u-coords are within the narrow box.
    print('\nWhich worst-cell state coords are INSIDE ccv_retire grid envelope?')
    # Load ccv_retire's pc to get bracket transform
    from scripts.diagnostics._diag_arbitrage_quadsweep import _make_pc
    from scripts.diagnostics._diag_policy_convergence import (_build_disc_config, _build_model_from_bundle, _extract_disc_config)
    from lifecycle.policy_io import load_policy_bundle
    model_cr, _ = _build_model_from_bundle(bundles['ccv_retire'])
    _C, _S, _B, _, meta_cr = load_policy_bundle(bundles['ccv_retire'])
    disc_cr = _build_disc_config(_extract_disc_config(meta_cr))
    pc_cr = _make_pc(model_cr, disc_cr)
    bracket_shift_cr = pc_cr.state_bracket_shift
    bracket_L_inv_cr = pc_cr.state_bracket_L_inv
    n_stds_cr = disc_cr.state_n_stds
    n_inside_narrow = 0
    for r in rwd_sorted[:50]:
        s_econ = np.array([r['s0'], r['s1'], r['s2']])
        u = bracket_L_inv_cr @ (s_econ - bracket_shift_cr)
        in_narrow = all(abs(u[d]) <= n_stds_cr[d] for d in range(3))
        if in_narrow:
            n_inside_narrow += 1
    print(f'  Of top-50 ccv_wide_gh worst cells: {n_inside_narrow}/50 sit INSIDE ccv_retire narrow envelope (2.0,2.25,2.25)')
    print(f'  {50 - n_inside_narrow}/50 sit OUTSIDE narrow envelope (would have been clipped under ccv_retire grid)')

    # Q3: Alpha distribution in the worst-cell tail
    print('\n' + '=' * 78)
    print('Q3. Alpha distribution in worst-cell tail vs body')
    print('=' * 78)
    rwd_arr = np.array([r['log10_abs_ee'] for r in rwd_sorted])
    p90_thresh = np.percentile(rwd_arr, 90)  # since sorted desc, top 10% are worst
    # actually take top 10% by EE worsening (already sorted)
    n_top10 = max(1, len(rwd_sorted) // 10)
    n_bot80 = max(1, int(0.8 * len(rwd_sorted)))  # bottom 80% by EE
    top10 = rwd_sorted[:n_top10]
    body80 = rwd_sorted[-n_bot80:]
    def alpha_summary(name, cells):
        a_s = np.array([r['alpha_s'] for r in cells])
        a_b = np.array([r['alpha_b'] for r in cells])
        x = np.array([r['x'] for r in cells])
        sav = np.array([r['sav_share'] for r in cells])
        constr = sum(1 for r in cells if r['is_constrained']) / len(cells)
        cap_pct = sum(1 for r in cells if abs(abs(r['alpha_s']) - 6) < 0.05 or abs(abs(r['alpha_b']) - 6) < 0.05) / len(cells)
        print(f'  {name:<20} n={len(cells)}')
        print(f'    alpha_s: mean={a_s.mean():+.3f}  range=[{a_s.min():+.2f}, {a_s.max():+.2f}]  |alpha_s|_mean={np.abs(a_s).mean():.3f}')
        print(f'    alpha_b: mean={a_b.mean():+.3f}  range=[{a_b.min():+.2f}, {a_b.max():+.2f}]  |alpha_b|_mean={np.abs(a_b).mean():.3f}')
        print(f'    wealth:  mean={x.mean():.2f}  range=[{x.min():.2f}, {x.max():.2f}]')
        print(f'    sav%:    mean={100*sav.mean():.1f}%  range=[{100*sav.min():.1f}%, {100*sav.max():.1f}%]')
        print(f'    is_constrained: {100*constr:.1f}%   cap-binding: {100*cap_pct:.1f}%')
    print(f'\nccv_wide_gh:')
    alpha_summary('top-10% (worst EE)', top10)
    alpha_summary('bottom-80% (best EE)', body80)


if __name__ == '__main__':
    main()

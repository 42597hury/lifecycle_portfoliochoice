"""Find the worst-EE simulated cells in the v4_lobatto bundle.

Re-runs the simulation, captures per-household per-age EE residuals,
and reports the top-N worst cells with their (age, z, state-coords, wealth, alpha).
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


def _state_index_to_tuple(i_s, sizes):
    N1 = int(sizes[1]); N2 = int(sizes[2])
    return (i_s // (N1 * N2), (i_s // N2) % N1, i_s % N2)


def _nearest_state_idx(state_coords, state_grid):
    """Find the closest gridpoint to a continuous state vector."""
    diffs = state_grid - state_coords[None, :]
    dist2 = np.sum(diffs * diffs, axis=1)
    return int(np.argmin(dist2))


def main():
    bundle_path = Path('saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v4_lobatto')
    print(f'Bundle: {bundle_path.name}\n')

    ctx = _load_bundle_context(
        bundle_path=bundle_path,
        model_bundle=bundle_path,
        eval_mode='next_finer',
        ret_override=None, state_override=None,
        eta_override=None, eps_override=None,
    )
    pc = ctx.pc_eval
    model = ctx.model
    sizes = tuple(int(s) for s in pc.state_grid_sizes)

    # Simulate
    sim = _simulate_bundle_window(
        ctx,
        n_simulations=5000,
        seed=42,
        return_draw_mode='monte_carlo',
        initial_x=None,
        initial_wealth=None,
        initial_wealth_distribution=None,
        initial_wealth_normal_std=0.0,
        initial_z='median',
        initial_z_normal_std=0.0,
        initial_state='median',
    )

    z_history = sim['z']
    state_history = sim['state_coords']
    wealth_history = sim['x']
    consumption_history = sim['c']
    alpha_s_history = sim['alpha_s']
    alpha_b_history = sim['alpha_b']
    savings_history = sim['savings']
    alive_history = sim['alive']

    n_sim, n_age = z_history.shape
    start_age = int(ctx.ages[ctx.first_solved_t])
    print(f'Simulated {n_sim} households across {n_age} ages (starting at age {start_age})')

    # Run EE evaluation on all live households per age (no cap)
    pension_row = np.ascontiguousarray(
        pc.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    eval_pad = _pad_eval_state_inputs_to_3d(pc, model)

    all_rows = []
    first_t = ctx.first_solved_t
    for t in range(n_age - 1):
        model_t = first_t + t
        age = int(ctx.ages[model_t])
        if age >= int(model.terminal_age):
            continue
        if model_t + 1 >= ctx.C.shape[0]:
            continue
        is_alive = alive_history[:, t]
        idx_alive = np.flatnonzero(is_alive)
        if len(idx_alive) == 0:
            continue
        # Restrict to a sample for speed (use up to 512 per age)
        if len(idx_alive) > 512:
            rng = np.random.default_rng(seed=42 + t)
            idx_alive = rng.choice(idx_alive, size=512, replace=False)
        n_eval = len(idx_alive)

        z_age = np.ascontiguousarray(z_history[idx_alive, t])
        s_age = np.ascontiguousarray(state_history[idx_alive, t])
        c_age = np.ascontiguousarray(consumption_history[idx_alive, t])
        a_s = np.ascontiguousarray(alpha_s_history[idx_alive, t])
        a_b = np.ascontiguousarray(alpha_b_history[idx_alive, t])
        sav = np.ascontiguousarray(savings_history[idx_alive, t])
        x_age = wealth_history[idx_alive, t]

        annuity_factors_age = _annuity_factors_per_household(model, s_age)

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
            np.ascontiguousarray(pc.eta_nodes),
            np.ascontiguousarray(pc.eta_weights),
            np.ascontiguousarray(pc.eps_nodes),
            np.ascontiguousarray(pc.eps_weights),
            eval_pad["v_nodes"],
            np.ascontiguousarray(pc.v_weights),
            np.ascontiguousarray(pc.M_v_nodes),
            np.ascontiguousarray(pc.const_r),
            eval_pad["A_r"],
            eval_pad["Phi_0_state"],
            eval_pad["Phi_11"],
            eval_pad["state_bracket_shift"],
            eval_pad["state_bracket_L_inv"],
            eval_pad["state_grids_0"],
            eval_pad["state_grids_1"],
            eval_pad["state_grids_2"],
            int(eval_pad["N0"]),
            int(eval_pad["N1"]),
            int(eval_pad["N2"]),
            np.ascontiguousarray(pc.exp_ret_bill),
            np.ascontiguousarray(pc.exp_ret_stock),
            np.ascontiguousarray(pc.exp_ret_bond),
            np.ascontiguousarray(pc.ret_weights),
            np.ascontiguousarray(pc.ret_nodes),
            float(pc.sigma2_xr),
            float(pc.sigma2_xb),
            float(pc.sigma_xrxb),
            bool(ctx.use_ccv),
            float(model.gamma), float(model.beta),
            int(model.b_bar),
            annuity_factors_age,
            float(1e-3),
            delta=ctx.delta_bequest,
        )

        for j, h_idx in enumerate(idx_alive):
            if not bool(valid[j]):
                continue
            ee_val = float(ee[j])
            abs_ee = abs(ee_val)
            log10_abs = float(np.log10(max(abs_ee, 1e-16)))
            i_s_nearest = _nearest_state_idx(s_age[j], pc.state_grid)
            i_z_nearest = int(np.argmin(np.abs(z_age[j] - pc.z_grid)))
            i_w_nearest = int(np.argmin(np.abs(x_age[j] - pc.wealth_grid)))
            tup = _state_index_to_tuple(i_s_nearest, sizes)
            all_rows.append({
                'age': int(age),
                'household': int(h_idx),
                'z': float(z_age[j]),
                'iz_near': i_z_nearest,
                'state_coords': tuple(round(float(v), 4) for v in s_age[j]),
                'i_s_near': i_s_nearest,
                'state_tuple_near': tup,
                'x': float(x_age[j]),
                'iw_near': i_w_nearest,
                'c': float(c_age[j]),
                'savings': float(sav[j]),
                'sav_share': float(sav[j] / max(x_age[j], 1e-12)),
                'alpha_s': float(a_s[j]),
                'alpha_b': float(a_b[j]),
                'is_constrained': bool(is_constr[j]),
                'ee': ee_val,
                'abs_ee': abs_ee,
                'log10_abs_ee': log10_abs,
            })

    if not all_rows:
        print('No valid cells.')
        return

    print(f'Total evaluated cells: {len(all_rows)}\n')

    # Sort by worst EE
    all_rows.sort(key=lambda r: -r['log10_abs_ee'])

    # Top-20 worst
    print(f'=== Top 20 worst sim-path cells by |EE| ===\n')
    print(f'{"rank":>4} {"age":>3}  {"hh":>5}  {"iz":>3} z={"":>7}  {"i_s_near":>4} ({"":>5})  {"x":>8}  {"alpha_s":>7}  {"alpha_b":>7}  sav%={"":>5} {"|EE|":>9} log10={"":>6}')
    for rank, r in enumerate(all_rows[:20]):
        print(f'{rank+1:>4} {r["age"]:>3}  {r["household"]:>5}  {r["iz_near"]:>3} z={r["z"]:>+6.3f}  '
              f'{r["i_s_near"]:>4} ({r["state_tuple_near"][0]},{r["state_tuple_near"][1]},{r["state_tuple_near"][2]})  '
              f'{r["x"]:>8.2f}  {r["alpha_s"]:>+7.3f}  {r["alpha_b"]:>+7.3f}  '
              f'sav%={100*r["sav_share"]:>4.1f}  {r["abs_ee"]:>9.4f}  log10={r["log10_abs_ee"]:>+6.3f}')

    # Worst cells: aggregate by feature to find patterns
    bad = [r for r in all_rows if r['log10_abs_ee'] > -1.5]   # > 3% rel error
    very_bad = [r for r in all_rows if r['log10_abs_ee'] > -1.2]  # > 6%
    print(f'\nCells with |EE|>3% (log10>-1.5): {len(bad)} ({100*len(bad)/len(all_rows):.1f}% of total)')
    print(f'Cells with |EE|>6% (log10>-1.2): {len(very_bad)} ({100*len(very_bad)/len(all_rows):.1f}%)')

    if bad:
        print(f'\n=== Pattern in worst cells (log10|EE| > -1.5) ===')
        ages_bad = [r['age'] for r in bad]
        iz_bad = [r['iz_near'] for r in bad]
        is_bad = [r['i_s_near'] for r in bad]
        sav_bad = [r['sav_share'] for r in bad]
        as_bad = [r['alpha_s'] for r in bad]
        ab_bad = [r['alpha_b'] for r in bad]
        x_bad = [r['x'] for r in bad]
        constrained_bad = sum(1 for r in bad if r['is_constrained'])

        print(f'  age range: {min(ages_bad)} - {max(ages_bad)} (mean {np.mean(ages_bad):.1f})')
        print(f'  iz range: {min(iz_bad)} - {max(iz_bad)} (mean {np.mean(iz_bad):.1f})')
        print(f'  alpha_s range: {min(as_bad):+.3f} - {max(as_bad):+.3f} (mean {np.mean(as_bad):+.3f})')
        print(f'  alpha_b range: {min(ab_bad):+.3f} - {max(ab_bad):+.3f} (mean {np.mean(ab_bad):+.3f})')
        print(f'  wealth range: {min(x_bad):.2f} - {max(x_bad):.2f} (mean {np.mean(x_bad):.1f})')
        print(f'  sav/x range: {100*min(sav_bad):.2f}% - {100*max(sav_bad):.2f}% (mean {100*np.mean(sav_bad):.1f}%)')
        print(f'  is_constrained: {constrained_bad}/{len(bad)} ({100*constrained_bad/len(bad):.1f}%)')

        # State-corner concentration
        from collections import Counter
        corner_counts = Counter(r['state_tuple_near'] for r in bad)
        top_corners = corner_counts.most_common(8)
        print(f'  Top state corners by bad-cell count:')
        for tup, n in top_corners:
            print(f'    {tup}: {n} cells')


if __name__ == '__main__':
    main()

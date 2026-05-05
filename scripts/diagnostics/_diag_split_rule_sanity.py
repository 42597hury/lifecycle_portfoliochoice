"""
Sanity check: at the worst leveraged retirement-age sim-path cell of v4_lobatto,
compute the Euler integrand (e_sum), the consumption- and bequest- integrals,
and the portfolio FOC two ways:

  Method A — Lobatto (status quo eval rule): K_ret=(3,5,5) Lobatto Z=(None,7,7),
             K_state=(3,5,5) Lobatto Z=(None,7,7).
  Method B — Dense GH reference:               K_ret=(5,5,101), K_state=(7,7,7),
             no Lobatto on either axis.

The kink at s*R_p = 0 sits near +/-7 sigma in the bond residual under leverage.
Method A explicitly samples the kink via Lobatto tail nodes (~1% weight) and
applies polynomial weights across the discontinuity. Method B has no explicit
tail nodes; K=101 GH on the bond axis spaces nodes finely enough (~0.2 sigma
near 7 sigma) to resolve the local crossing; K=7 GH on each state axis
captures the M-coupled return-mean shift without an explicit tail node.

If the integrals differ by an O(1) amount at the bequest piece (where the kink
lives), the Lobatto rule is contaminated. If the re-solved optimal alpha under
method B moves materially vs the bundle's policy alpha, the contamination is
producing a wrong policy.

Usage:
    python -m scripts.diagnostics._diag_split_rule_sanity
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.model import DELTA_BEQUEST
from lifecycle.precompute import Precompute
from scripts.diagnostics._diag_euler_errors import (
    EulerBundleContext,
    _annuity_factor_scalar,
    _annuity_factors_per_household,
    _evaluate_age_errors,
    _interp_continuation_c_mpc,
    _interp_z_row_value,
    _load_bundle_context,
    _pad_eval_state_inputs_to_3d,
    _shifted_bequest_mu,
    _simulate_bundle_window,
)


# ----------------------------------------------------------------------------
# Step 1: locate the worst leveraged retirement-age cell in v4_lobatto
# ----------------------------------------------------------------------------
def find_worst_leveraged_cell(ctx: EulerBundleContext, n_simulations: int = 5000):
    """Replay the simulator, evaluate EE per live cell, return the worst
    retirement-age cell with |alpha_b| >= 1 OR |alpha_bill| >= 1."""
    pc = ctx.pc_eval
    model = ctx.model

    sim = _simulate_bundle_window(
        ctx,
        n_simulations=n_simulations,
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
    pension_row = np.ascontiguousarray(
        pc.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    eval_pad = _pad_eval_state_inputs_to_3d(pc, model)

    best = None  # (log10_abs_ee, payload)
    first_t = ctx.first_solved_t
    for t in range(n_age - 1):
        model_t = first_t + t
        age = int(ctx.ages[model_t])
        if age < int(model.retire_age):
            continue
        if age >= int(model.terminal_age):
            continue
        if model_t + 1 >= ctx.C.shape[0]:
            continue
        is_alive = alive_history[:, t]
        idx_alive = np.flatnonzero(is_alive)
        if len(idx_alive) == 0:
            continue

        z_age = np.ascontiguousarray(z_history[idx_alive, t])
        s_age = np.ascontiguousarray(state_history[idx_alive, t])
        c_age = np.ascontiguousarray(consumption_history[idx_alive, t])
        a_s = np.ascontiguousarray(alpha_s_history[idx_alive, t])
        a_b = np.ascontiguousarray(alpha_b_history[idx_alive, t])
        sav = np.ascontiguousarray(savings_history[idx_alive, t])
        x_age = wealth_history[idx_alive, t]

        annuity_factors_age = _annuity_factors_per_household(model, s_age)
        n_eval = len(idx_alive)

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

        for j in range(n_eval):
            if not bool(valid[j]):
                continue
            if bool(is_constr[j]):
                continue
            alpha_s_j = float(a_s[j])
            alpha_b_j = float(a_b[j])
            alpha_bill_j = 1.0 - alpha_s_j - alpha_b_j
            if abs(alpha_b_j) < 1.0 and abs(alpha_bill_j) < 1.0:
                continue  # only leveraged cells
            ee_val = float(ee[j])
            log10_abs = math.log10(max(abs(ee_val), 1e-16))
            if best is None or log10_abs > best[0]:
                best = (log10_abs, {
                    'age': age,
                    'model_t': model_t,
                    'z_cur': float(z_age[j]),
                    'state_cur': np.array(s_age[j], dtype=float),
                    'c_t': float(c_age[j]),
                    'alpha_s': alpha_s_j,
                    'alpha_b': alpha_b_j,
                    'alpha_bill': alpha_bill_j,
                    's_val': float(sav[j]),
                    'x': float(x_age[j]),
                    'ee': ee_val,
                    'log10_abs_ee': log10_abs,
                    'annuity_factor': float(annuity_factors_age[j]),
                })

    if best is None:
        raise RuntimeError("No leveraged retirement cell found.")
    return best[1]


# ----------------------------------------------------------------------------
# Step 2: kernel that returns (e_sum, e_alive, e_bequest, foc_s, foc_b,
#                              J_ss, J_bb, J_sb) for one cell under one rule.
# ----------------------------------------------------------------------------
@njit(fastmath=True)
def _foc_and_decomp_kernel(
    alpha_s, alpha_b, s_val, z_cur, state_cur,
    c_next_full, wealth_grid, z_grid, dz,
    pension_row, survival_row,
    v_nodes, v_weights, M_v_nodes,
    const_r, A_r, Phi_0_state, Phi_11,
    state_bracket_shift, state_bracket_L_inv,
    grids_0, grids_1, grids_2, N0, N1, N2,
    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
    gamma, b_bar, annuity_factor_cur,
    min_consumption=1e-10, prob_skip=1e-12,
    delta=DELTA_BEQUEST,
):
    a_bill = 1.0 - alpha_s - alpha_b
    psi = _interp_z_row_value(z_cur, z_grid, survival_row)
    prob_death = 1.0 - psi
    pension_next = _interp_z_row_value(z_cur, z_grid, pension_row)

    base_mu_r_0 = const_r[0] + A_r[0, 0] * state_cur[0] + A_r[0, 1] * state_cur[1] + A_r[0, 2] * state_cur[2]
    base_mu_r_1 = const_r[1] + A_r[1, 0] * state_cur[0] + A_r[1, 1] * state_cur[1] + A_r[1, 2] * state_cur[2]
    base_mu_r_2 = const_r[2] + A_r[2, 0] * state_cur[0] + A_r[2, 1] * state_cur[1] + A_r[2, 2] * state_cur[2]

    e_sum = 0.0
    e_alive = 0.0
    e_bequest = 0.0
    foc_s = 0.0
    foc_b = 0.0
    J_ss = 0.0
    J_bb = 0.0
    J_sb = 0.0

    for k_v in range(len(v_weights)):
        w_v = v_weights[k_v]
        if w_v < prob_skip:
            continue

        s_next_0 = Phi_0_state[0] + Phi_11[0, 0] * state_cur[0] + Phi_11[0, 1] * state_cur[1] + Phi_11[0, 2] * state_cur[2] + v_nodes[k_v, 0]
        s_next_1 = Phi_0_state[1] + Phi_11[1, 0] * state_cur[0] + Phi_11[1, 1] * state_cur[1] + Phi_11[1, 2] * state_cur[2] + v_nodes[k_v, 1]
        s_next_2 = Phi_0_state[2] + Phi_11[2, 0] * state_cur[0] + Phi_11[2, 1] * state_cur[1] + Phi_11[2, 2] * state_cur[2] + v_nodes[k_v, 2]

        exp_mu_bill = math.exp(base_mu_r_0 + M_v_nodes[k_v, 0])
        exp_mu_s = math.exp(base_mu_r_1 + M_v_nodes[k_v, 1])
        exp_mu_b = math.exp(base_mu_r_2 + M_v_nodes[k_v, 2])

        for k_r in range(len(ret_weights)):
            weight = w_v * ret_weights[k_r]
            if weight < prob_skip:
                continue

            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
            Rex_s = R_s - R_bill
            Rex_b = R_b - R_bill

            sR_p = s_val * R_p
            if sR_p > 0.0:
                x_next = sR_p + pension_next
            else:
                x_next = pension_next

            c_next, mpc = _interp_continuation_c_mpc(
                c_next_full, z_cur, s_next_0, s_next_1, s_next_2, x_next,
                wealth_grid, z_grid, dz,
                state_bracket_shift, state_bracket_L_inv,
                grids_0, grids_1, grids_2, N0, N1, N2,
                min_consumption,
            )

            mu_alive = c_next ** (-gamma)
            mup_alive = -gamma * mu_alive / c_next * mpc
            if sR_p > 0.0:
                # Shifted bequest (De Nardi 2004) — mirrors solver's
                # _shifted_bequest_mu_and_mup. Inlined here to avoid the import
                # cycle (this file imports from _diag_euler_errors which already
                # has its own copy).
                C_bar = sR_p / annuity_factor_cur + delta
                mu_bequest = b_bar * C_bar ** (-gamma) / annuity_factor_cur
                mup_bequest = -gamma * mu_bequest / (annuity_factor_cur * C_bar)
            else:
                mu_bequest = 0.0
                mup_bequest = 0.0

            mu_comb = psi * mu_alive + prob_death * mu_bequest
            mup_comb = psi * mup_alive + prob_death * mup_bequest

            wmu = weight * mu_comb
            wmup = weight * mup_comb

            e_sum += wmu * R_p
            e_alive += weight * psi * mu_alive * R_p
            e_bequest += weight * prob_death * mu_bequest * R_p

            foc_s += wmu * Rex_s
            foc_b += wmu * Rex_b

            jac = wmup * s_val
            J_ss += jac * Rex_s * Rex_s
            J_bb += jac * Rex_b * Rex_b
            J_sb += jac * Rex_s * Rex_b

    return e_sum, e_alive, e_bequest, foc_s, foc_b, J_ss, J_bb, J_sb


def evaluate_at_cell(cell, ctx, pc, alpha_s, alpha_b, delta=None):
    """Evaluate kernel at one cell + one alpha under the supplied precompute pc.

    `delta` defaults to ctx.delta_bequest (the value the bundle was solved with);
    pass an explicit float to override (e.g. for sensitivity analyses).
    """
    if delta is None:
        delta = ctx.delta_bequest
    model = ctx.model
    pad = _pad_eval_state_inputs_to_3d(pc, model)
    pension_row = np.ascontiguousarray(
        pc.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    survival_row = np.ascontiguousarray(pc.survival_probs_2d[cell['model_t'], :])
    c_next_full = np.ascontiguousarray(ctx.C[cell['model_t'] + 1])

    return _foc_and_decomp_kernel(
        float(alpha_s), float(alpha_b),
        float(cell['s_val']), float(cell['z_cur']),
        np.ascontiguousarray(cell['state_cur']),
        c_next_full,
        np.ascontiguousarray(pc.wealth_grid),
        np.ascontiguousarray(pc.z_grid),
        float(pc.dz),
        pension_row, survival_row,
        np.ascontiguousarray(pc.v_nodes),
        np.ascontiguousarray(pc.v_weights),
        np.ascontiguousarray(pc.M_v_nodes),
        np.ascontiguousarray(pc.const_r),
        pad["A_r"],
        pad["Phi_0_state"],
        pad["Phi_11"],
        pad["state_bracket_shift"],
        pad["state_bracket_L_inv"],
        pad["state_grids_0"],
        pad["state_grids_1"],
        pad["state_grids_2"],
        int(pad["N0"]), int(pad["N1"]), int(pad["N2"]),
        np.ascontiguousarray(pc.exp_ret_bill),
        np.ascontiguousarray(pc.exp_ret_stock),
        np.ascontiguousarray(pc.exp_ret_bond),
        np.ascontiguousarray(pc.ret_weights),
        float(model.gamma), int(model.b_bar),
        float(cell['annuity_factor']),
        delta=float(delta),
    )


# ----------------------------------------------------------------------------
# Step 3: build dense-GH precompute (Method B reference rule)
# ----------------------------------------------------------------------------
def build_dense_pc(ctx, n_ret_dense=(5, 5, 101), n_state_dense=(7, 7, 7)):
    base = ctx.disc_eval
    disc_dense = base._replace(
        n_ret_nodes_1d=tuple(int(k) for k in n_ret_dense),
        n_state_quad_nodes=tuple(int(k) for k in n_state_dense),
        ret_lobatto_Z=None,
        state_lobatto_Z=None,
    )
    return Precompute(ctx.model, disc_dense, verbose=False)


# ----------------------------------------------------------------------------
# Step 4: re-solve optimal alpha with Method B (dense rule), Newton on FOC.
#
# FOC: G(a_s, a_b) = (foc_s, foc_b) = 0
# Jacobian dG/d(a_s, a_b) under fixed s_val. We have analytical
#   dfoc_s/d a_s, dfoc_b/d a_b, dfoc_s/d a_b   (J_ss, J_bb, J_sb)
# from the kernel (with mup terms accumulated). Use damped Newton with
# back-tracking line search on |G|.
# ----------------------------------------------------------------------------
def resolve_alpha_newton(cell, ctx, pc, alpha_s0, alpha_b0,
                         tol=1e-7, max_iter=40, damp=0.6):
    a_s = float(alpha_s0)
    a_b = float(alpha_b0)
    history = []
    for it in range(max_iter):
        out = evaluate_at_cell(cell, ctx, pc, a_s, a_b)
        foc_s, foc_b, J_ss, J_bb, J_sb = out[3], out[4], out[5], out[6], out[7]
        g = np.array([foc_s, foc_b], dtype=float)
        norm_g = float(np.linalg.norm(g))
        history.append((it, a_s, a_b, foc_s, foc_b, norm_g))
        if norm_g < tol:
            break
        J = np.array([[J_ss, J_sb], [J_sb, J_bb]], dtype=float)
        try:
            step = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            break
        # damped step + back-track if residual grows
        scale = 1.0
        for _ in range(8):
            a_s_new = a_s + damp * scale * step[0]
            a_b_new = a_b + damp * scale * step[1]
            out_new = evaluate_at_cell(cell, ctx, pc, a_s_new, a_b_new)
            g_new = np.array([out_new[3], out_new[4]], dtype=float)
            if np.linalg.norm(g_new) < norm_g:
                break
            scale *= 0.5
        a_s = a_s_new
        a_b = a_b_new
    return a_s, a_b, history


# ----------------------------------------------------------------------------
# Step 5: per-node integrand dump — find which Lobatto node spikes the bequest.
# ----------------------------------------------------------------------------
@njit(fastmath=True)
def _per_node_bequest_kernel(
    alpha_s, alpha_b, s_val, z_cur, state_cur,
    pension_row, survival_row, z_grid,
    v_nodes, v_weights, M_v_nodes,
    const_r, A_r, Phi_0_state, Phi_11,
    exp_ret_bill, exp_ret_stock, exp_ret_bond, ret_weights,
    gamma, b_bar, annuity_factor_cur,
    delta=DELTA_BEQUEST,
):
    """Return an array shape (n_state_quad, n_ret_quad, 4) holding
    (weight, sR_p, mu_bequest, weight*mu_bequest*R_p) for every (k_v, k_r) node.
    """
    a_bill = 1.0 - alpha_s - alpha_b
    psi = _interp_z_row_value(z_cur, z_grid, survival_row)
    prob_death = 1.0 - psi

    base_mu_r_0 = const_r[0] + A_r[0, 0] * state_cur[0] + A_r[0, 1] * state_cur[1] + A_r[0, 2] * state_cur[2]
    base_mu_r_1 = const_r[1] + A_r[1, 0] * state_cur[0] + A_r[1, 1] * state_cur[1] + A_r[1, 2] * state_cur[2]
    base_mu_r_2 = const_r[2] + A_r[2, 0] * state_cur[0] + A_r[2, 1] * state_cur[1] + A_r[2, 2] * state_cur[2]

    n_v = len(v_weights)
    n_r = len(ret_weights)
    out = np.zeros((n_v, n_r, 4), dtype=np.float64)

    for k_v in range(n_v):
        w_v = v_weights[k_v]
        exp_mu_bill = math.exp(base_mu_r_0 + M_v_nodes[k_v, 0])
        exp_mu_s = math.exp(base_mu_r_1 + M_v_nodes[k_v, 1])
        exp_mu_b = math.exp(base_mu_r_2 + M_v_nodes[k_v, 2])

        for k_r in range(n_r):
            weight = w_v * ret_weights[k_r]
            R_bill = exp_mu_bill * exp_ret_bill[k_r]
            R_s = R_bill * exp_mu_s * exp_ret_stock[k_r]
            R_b = R_bill * exp_mu_b * exp_ret_bond[k_r]
            R_p = alpha_s * R_s + alpha_b * R_b + a_bill * R_bill
            sR_p = s_val * R_p

            if sR_p > 0.0:
                mu_bequest = _shifted_bequest_mu(
                    sR_p, annuity_factor_cur, gamma, b_bar, delta
                )
            else:
                mu_bequest = 0.0
            contrib = weight * prob_death * mu_bequest * R_p
            out[k_v, k_r, 0] = weight
            out[k_v, k_r, 1] = sR_p
            out[k_v, k_r, 2] = mu_bequest
            out[k_v, k_r, 3] = contrib
    return out


def per_node_bequest(cell, ctx, pc, alpha_s, alpha_b, delta=None):
    if delta is None:
        delta = ctx.delta_bequest
    model = ctx.model
    pad = _pad_eval_state_inputs_to_3d(pc, model)
    pension_row = np.ascontiguousarray(
        pc.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    survival_row = np.ascontiguousarray(pc.survival_probs_2d[cell['model_t'], :])
    return _per_node_bequest_kernel(
        float(alpha_s), float(alpha_b),
        float(cell['s_val']), float(cell['z_cur']),
        np.ascontiguousarray(cell['state_cur']),
        pension_row, survival_row,
        np.ascontiguousarray(pc.z_grid),
        np.ascontiguousarray(pc.v_nodes),
        np.ascontiguousarray(pc.v_weights),
        np.ascontiguousarray(pc.M_v_nodes),
        np.ascontiguousarray(pc.const_r),
        pad["A_r"],
        pad["Phi_0_state"],
        pad["Phi_11"],
        np.ascontiguousarray(pc.exp_ret_bill),
        np.ascontiguousarray(pc.exp_ret_stock),
        np.ascontiguousarray(pc.exp_ret_bond),
        np.ascontiguousarray(pc.ret_weights),
        float(model.gamma), int(model.b_bar),
        float(cell['annuity_factor']),
        delta=float(delta),
    )


def main():
    bundle_path = Path('saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v4_lobatto')
    print(f'Bundle: {bundle_path.name}\n')

    print('Loading bundle (eval_mode="same" -> Lobatto K_ret=(3,5,5)/Z=(None,7,7), K_state=(3,5,5)/Z=(None,7,7))...')
    ctx = _load_bundle_context(
        bundle_path=bundle_path,
        model_bundle=bundle_path,
        eval_mode='same',
        ret_override=None, state_override=None,
        eta_override=None, eps_override=None,
    )
    pc_lobatto = ctx.pc_eval
    print(f'  pc_lobatto: K_ret={pc_lobatto.disc_config.n_ret_nodes_1d} '
          f'lobatto_Z_ret={pc_lobatto.disc_config.ret_lobatto_Z}  '
          f'K_state={pc_lobatto.disc_config.n_state_quad_nodes} '
          f'lobatto_Z_state={pc_lobatto.disc_config.state_lobatto_Z}')
    print(f'  -> {len(pc_lobatto.ret_weights)} return nodes, {len(pc_lobatto.v_weights)} state nodes\n')

    print('Building dense GH reference (Method B): K_ret=(5,5,101), K_state=(7,7,7), no Lobatto...')
    pc_dense = build_dense_pc(ctx, n_ret_dense=(5, 5, 101), n_state_dense=(7, 7, 7))
    print(f'  -> {len(pc_dense.ret_weights)} return nodes, {len(pc_dense.v_weights)} state nodes\n')

    print('Locating worst leveraged retirement cell (this re-runs the simulator)...')
    cell = find_worst_leveraged_cell(ctx, n_simulations=5000)
    print('  worst leveraged retirement cell:')
    for k in ('age', 'z_cur', 'state_cur', 'x', 'c_t', 's_val',
              'alpha_s', 'alpha_b', 'alpha_bill', 'ee', 'log10_abs_ee'):
        v = cell[k]
        if isinstance(v, np.ndarray):
            v = '[' + ', '.join(f'{x:+.4f}' for x in v) + ']'
        elif isinstance(v, float):
            v = f'{v:+.6f}'
        print(f'    {k:>14}: {v}')
    print()

    # ---- Method A: Lobatto (status quo) at policy alpha ----
    print('Evaluating at policy alpha:')
    print(f'  alpha_s={cell["alpha_s"]:+.4f}  alpha_b={cell["alpha_b"]:+.4f}  alpha_bill={cell["alpha_bill"]:+.4f}\n')

    e_sum_A, e_alive_A, e_beq_A, foc_s_A, foc_b_A, J_ss_A, J_bb_A, J_sb_A = evaluate_at_cell(
        cell, ctx, pc_lobatto, cell['alpha_s'], cell['alpha_b']
    )
    e_sum_B, e_alive_B, e_beq_B, foc_s_B, foc_b_B, J_ss_B, J_bb_B, J_sb_B = evaluate_at_cell(
        cell, ctx, pc_dense, cell['alpha_s'], cell['alpha_b']
    )

    gamma = float(ctx.model.gamma)
    beta = float(ctx.model.beta)
    c_t = cell['c_t']

    def euler_residual(e_sum):
        if beta * e_sum <= 0.0:
            return float('nan')
        c_implied = (beta * e_sum) ** (-1.0 / gamma)
        return 1.0 - c_implied / c_t

    print('Integrand decomposition at policy alpha:')
    print(f'  {"":<28}{"Lobatto (A)":>16}  {"Dense GH (B)":>16}  {"diff (A-B)":>16}  rel-diff')
    for label, va, vb in [
        ('e_sum',          e_sum_A,    e_sum_B),
        ('  e_alive',      e_alive_A,  e_alive_B),
        ('  e_bequest',    e_beq_A,    e_beq_B),
        ('foc_s',          foc_s_A,    foc_s_B),
        ('foc_b',          foc_b_A,    foc_b_B),
        ('J_ss',           J_ss_A,     J_ss_B),
        ('J_bb',           J_bb_A,     J_bb_B),
        ('J_sb',           J_sb_A,     J_sb_B),
    ]:
        diff = va - vb
        rel = diff / vb if abs(vb) > 1e-30 else float('nan')
        print(f'  {label:<28}{va:>+16.6e}  {vb:>+16.6e}  {diff:>+16.6e}  {rel:>+8.3%}')
    print()

    print('Implied Euler residual (1 - c_implied/c_t) at policy alpha:')
    ee_A = euler_residual(e_sum_A)
    ee_B = euler_residual(e_sum_B)
    print(f'  Lobatto (A):  EE = {ee_A:+.6f}   log10|EE| = {math.log10(max(abs(ee_A), 1e-16)):+.3f}')
    print(f'  Dense GH (B): EE = {ee_B:+.6f}   log10|EE| = {math.log10(max(abs(ee_B), 1e-16)):+.3f}')
    print(f'  diff: {ee_A - ee_B:+.6f}\n')

    # ---- Per-node bequest spike dump under Lobatto rule ----
    print('Per-node bequest dump under Lobatto rule (top-5 contributions to e_bequest):')
    grid_A = per_node_bequest(cell, ctx, pc_lobatto, cell['alpha_s'], cell['alpha_b'])
    flat_A = grid_A.reshape(-1, 4)
    n_v_A = grid_A.shape[0]
    n_r_A = grid_A.shape[1]
    order = np.argsort(-np.abs(flat_A[:, 3]))[:5]
    print(f'  {"k_v":>3} {"k_r":>3}   {"weight":>11}   {"sR_p":>14}   {"mu_bequest":>14}   {"contrib":>14}')
    for idx in order:
        k_v = idx // n_r_A
        k_r = idx % n_r_A
        w, sRp, mub, contrib = flat_A[idx]
        print(f'  {k_v:>3} {k_r:>3}   {w:>11.4e}   {sRp:>+14.6e}   {mub:>+14.6e}   {contrib:>+14.6e}')
    n_solvent_A = int(np.sum(flat_A[:, 1] > 0))
    n_kink_band_A = int(np.sum((flat_A[:, 1] > 0) & (flat_A[:, 1] < 1.0)))
    print(f'  Lobatto: {n_solvent_A}/{n_v_A * n_r_A} nodes solvent (sR_p > 0); '
          f'{n_kink_band_A} have 0 < sR_p < 1 (near-kink)')
    print()

    print('Per-node bequest dump under Dense GH rule (top-5 contributions to e_bequest):')
    grid_B = per_node_bequest(cell, ctx, pc_dense, cell['alpha_s'], cell['alpha_b'])
    flat_B = grid_B.reshape(-1, 4)
    n_v_B = grid_B.shape[0]
    n_r_B = grid_B.shape[1]
    order_B = np.argsort(-np.abs(flat_B[:, 3]))[:5]
    print(f'  {"k_v":>3} {"k_r":>3}   {"weight":>11}   {"sR_p":>14}   {"mu_bequest":>14}   {"contrib":>14}')
    for idx in order_B:
        k_v = idx // n_r_B
        k_r = idx % n_r_B
        w, sRp, mub, contrib = flat_B[idx]
        print(f'  {k_v:>3} {k_r:>3}   {w:>11.4e}   {sRp:>+14.6e}   {mub:>+14.6e}   {contrib:>+14.6e}')
    n_solvent_B = int(np.sum(flat_B[:, 1] > 0))
    n_kink_band_B = int(np.sum((flat_B[:, 1] > 0) & (flat_B[:, 1] < 1.0)))
    print(f'  Dense GH: {n_solvent_B}/{n_v_B * n_r_B} nodes solvent (sR_p > 0); '
          f'{n_kink_band_B} have 0 < sR_p < 1 (near-kink)')
    print()

    # ---- Re-solve optimal alpha under Method B (Newton with analytical J) ----
    print('Re-solving (alpha_s, alpha_b) under Method B (dense GH) — damped Newton, analytical Jacobian:')
    a_s_star, a_b_star, history = resolve_alpha_newton(
        cell, ctx, pc_dense, cell['alpha_s'], cell['alpha_b'],
        tol=1e-7, max_iter=40, damp=0.6,
    )
    print(f'  iter  alpha_s     alpha_b     foc_s        foc_b        ||g||')
    for it, a_s, a_b, fs, fb, ng in history:
        print(f'   {it:>3}  {a_s:+.6f}  {a_b:+.6f}  {fs:>+11.4e}  {fb:>+11.4e}  {ng:>+11.4e}')
    a_bill_star = 1.0 - a_s_star - a_b_star
    print()
    print(f'  alpha_s*:    {a_s_star:+.4f}  (policy: {cell["alpha_s"]:+.4f}, shift: {a_s_star - cell["alpha_s"]:+.4f})')
    print(f'  alpha_b*:    {a_b_star:+.4f}  (policy: {cell["alpha_b"]:+.4f}, shift: {a_b_star - cell["alpha_b"]:+.4f})')
    print(f'  alpha_bill*: {a_bill_star:+.4f}  (policy: {cell["alpha_bill"]:+.4f}, shift: {a_bill_star - cell["alpha_bill"]:+.4f})')

    # Sanity-check FOC and EE at new alpha under both rules
    out_star_B = evaluate_at_cell(cell, ctx, pc_dense, a_s_star, a_b_star)
    out_star_A = evaluate_at_cell(cell, ctx, pc_lobatto, a_s_star, a_b_star)
    print(f'\n  At alpha*:')
    print(f'    Method B (dense GH): e_sum={out_star_B[0]:+.4e}  EE={euler_residual(out_star_B[0]):+.6f}  foc_s={out_star_B[3]:+.3e}  foc_b={out_star_B[4]:+.3e}')
    print(f'    Method A (Lobatto):  e_sum={out_star_A[0]:+.4e}  EE={euler_residual(out_star_A[0]):+.6f}  foc_s={out_star_A[3]:+.3e}  foc_b={out_star_A[4]:+.3e}')

    lev_policy = max(abs(cell['alpha_b']), abs(cell['alpha_bill']))
    lev_new = max(abs(a_b_star), abs(a_bill_star))
    print(f'\n  Max(|alpha_b|, |alpha_bill|): policy={lev_policy:.3f}  re-solved={lev_new:.3f}  '
          f'change={lev_new - lev_policy:+.3f}')


if __name__ == '__main__':
    main()

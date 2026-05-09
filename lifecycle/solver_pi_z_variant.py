"""Prototype lifecycle solver variant using the persistent-z ``Pi_z`` chain.

This module intentionally leaves :mod:`lifecycle.solver` untouched.  The public
entry point, :func:`run_lifecycle_solver_pi_z`, wraps the production
orchestrator and swaps only the working-age kernel builder.  Retirement and
terminal-age kernels are reused verbatim.

NOTE (2026-05-09): ``pc.Pi_z`` was dropped from the production ``Precompute``
NamedTuple — it was reducible at the canonical ``(rho=0.991, n_z=11,
n_stds=3.0)`` and was off the production hot path (see
``docs/scans/INCOME_PIPELINE_REVIEW_2026-05-09.md`` MED-1). The variant
solver now constructs its own ``Pi_z`` locally from ``(model, disc_config)``
via :func:`discretize_income_ar1_mixture`. The same reducibility caveat that
prompted the drop applies here: callers using this variant at canonical
``rho=0.991`` should reduce ``disc_config.n_stds`` (e.g. to ``2.25``) so the
upper-tail bin probabilities don't underflow.
"""
from __future__ import annotations

import math
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, pmap, vmap

from lifecycle import solver as _base
from lifecycle.discretization import discretize_income_ar1_mixture


PCJaxPi = namedtuple("PCJaxPi", _base.PCJax._fields + ("Pi_z",))
_ORIGINAL_PC_TO_JNP = _base._pc_to_jnp


def _build_pi_z_local(pc):
    """Construct ``Pi_z`` locally for the variant solver.

    Pulled from ``(pc.model, pc.disc_config)`` since ``Pi_z`` was dropped
    from the production ``Precompute`` (see module docstring).
    """
    model = pc.model
    disc = pc.disc_config
    _, Pi_z = discretize_income_ar1_mixture(
        rho=model.rho,
        p=model.pz,
        mu1=model.mu_eta1,
        sigma1=model.sigma_eta1,
        mu2=model.mu_eta2,
        sigma2=model.sigma_eta2,
        N=disc.n_z,
        n_stds=disc.n_stds,
    )
    return Pi_z


def _pc_to_jnp_with_pi_z(pc, delta):
    """Extend the production PCJax bundle with ``Pi_z`` for the variant."""
    pcj = _ORIGINAL_PC_TO_JNP(pc, delta)
    return PCJaxPi(*pcj, jnp.asarray(_build_pi_z_local(pc)))


def _build_discrete_income_next_table(pc):
    """Build ``(age, z_current, z_next, eps)`` income for the Pi_z kernel.

    The production orchestrator passes ``working_income_next[t + 1]`` into the
    working kernel.  To preserve that age-index convention exactly, row ``q``
    here contains the same deterministic-age component as production row ``q``:
    ``pc.working_income[q + 1]`` where that next row exists.
    """
    out = np.zeros((pc.n_age, pc.n_z, pc.n_z, pc.n_eps), dtype=np.float64)
    if pc.n_age <= 1:
        return out
    discrete_next = np.asarray(pc.working_income[1:], dtype=np.float64)
    out[:-1] = np.broadcast_to(
        discrete_next[:, None, :, :],
        (pc.n_age - 1, pc.n_z, pc.n_z, pc.n_eps),
    )
    return out


def _interp_c_and_mpc_at_zgrid_cell(
    c_corners_kv,
    w_corners_kv,
    iz,
    x_next_scalar,
    wealth_grid,
    min_consumption,
    gather_dtype=jnp.float64,
):
    """Multilinear-state x linear-wealth interp at an exact z-grid node."""
    c_corners_g = _base._cast_for_gather(c_corners_kv, gather_dtype)
    w_corners_g = _base._cast_for_gather(w_corners_kv, gather_dtype)
    x_next_g = _base._cast_for_gather(jnp.asarray(x_next_scalar), gather_dtype)
    wealth_grid_g = _base._cast_for_gather(wealth_grid, gather_dtype)

    n_w = wealth_grid_g.shape[0]
    iw = jnp.clip(
        jnp.searchsorted(wealth_grid_g, x_next_g, side="right") - 1,
        0,
        n_w - 2,
    )
    iw_hi = iw + 1
    x0 = wealth_grid_g[iw]
    x1 = wealth_grid_g[iw_hi]
    inv_dw = jnp.asarray(1.0, dtype=gather_dtype) / (x1 - x0)
    fw = (x_next_g - x0) * inv_dw

    c_w0 = c_corners_g[iz, :, iw]
    c_w1 = c_corners_g[iz, :, iw_hi]
    c_per_corner = c_w0 + (c_w1 - c_w0) * fw
    slope_per_corner = (c_w1 - c_w0) * inv_dw

    c_g = jnp.sum(w_corners_g * c_per_corner)
    mpc_g = jnp.sum(w_corners_g * slope_per_corner)

    c = c_g.astype(jnp.float64)
    mpc = mpc_g.astype(jnp.float64)
    c = jnp.maximum(c, min_consumption)
    mpc = jnp.clip(mpc, 0.0, 1.0)
    return c, mpc


def working_foc_jac_ccv_pi_z(
    alpha_s,
    alpha_b,
    s_val,
    psi_z,
    log_R_bill,
    log_x_s,
    log_x_b,
    weight_kv_kr,
    w_corners,
    c_corners_T,
    wealth_grid,
    income_next_by_z,
    z_transition_probs,
    eps_weights,
    A_is,
    sigma2_xr,
    sigma2_xb,
    sigma_xrxb,
    gamma,
    b_bar,
    delta,
    min_consumption,
    gather_dtype=jnp.float64,
):
    """Working-age FOC with Tauchen-style discrete persistent-income states."""
    R_p, dr_da_s, dr_da_b = _base._ccv_log_return_and_grad(
        alpha_s,
        alpha_b,
        log_R_bill,
        log_x_s,
        log_x_b,
        sigma2_xr,
        sigma2_xb,
        sigma_xrxb,
    )
    sR_p = s_val * R_p

    mu_bq, mup_bq = _base.bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)
    prob_death = 1.0 - psi_z

    bequest_factor = weight_kv_kr * prob_death
    dRp_das = R_p * dr_da_s
    dRp_dab = R_p * dr_da_b
    foc_s_bq = jnp.sum(bequest_factor * mu_bq * dRp_das)
    foc_b_bq = jnp.sum(bequest_factor * mu_bq * dRp_dab)
    e_bq = jnp.sum(bequest_factor * mu_bq * R_p)
    jac_lin_bq = bequest_factor * mup_bq * s_val
    extra_ss = bequest_factor * mu_bq * R_p * (dr_da_s * dr_da_s - sigma2_xr)
    extra_bb = bequest_factor * mu_bq * R_p * (dr_da_b * dr_da_b - sigma2_xb)
    extra_sb = bequest_factor * mu_bq * R_p * (dr_da_s * dr_da_b - sigma_xrxb)
    J_ss_bq = jnp.sum(jac_lin_bq * dRp_das * dRp_das + extra_ss)
    J_bb_bq = jnp.sum(jac_lin_bq * dRp_dab * dRp_dab + extra_bb)
    J_sb_bq = jnp.sum(jac_lin_bq * dRp_das * dRp_dab + extra_sb)

    x_next = sR_p[:, :, None, None] + income_next_by_z[None, None, :, :]
    z_indices = jnp.arange(income_next_by_z.shape[0], dtype=jnp.int32)

    def per_kv(c_kv, w_kv, x_kv):
        def at_z_eps(x_scalar, iz):
            return _interp_c_and_mpc_at_zgrid_cell(
                c_kv,
                w_kv,
                iz,
                x_scalar,
                wealth_grid,
                min_consumption,
                gather_dtype=gather_dtype,
            )

        def per_kr(x_kr):
            def per_z(x_row, iz):
                return vmap(at_z_eps, in_axes=(0, None))(x_row, iz)

            return vmap(per_z, in_axes=(0, 0))(x_kr, z_indices)

        return vmap(per_kr)(x_kv)

    c_at_xn, mpc_at_xn = vmap(per_kv)(c_corners_T, w_corners, x_next)

    mu_alive = c_at_xn ** (-gamma)
    mup_alive = -gamma * mu_alive / c_at_xn * mpc_at_xn

    weight_full = (
        weight_kv_kr[:, :, None, None]
        * z_transition_probs[None, None, :, None]
        * eps_weights[None, None, None, :]
    )
    alive_factor = weight_full * psi_z

    foc_s_al = jnp.sum(alive_factor * mu_alive * dRp_das[:, :, None, None])
    foc_b_al = jnp.sum(alive_factor * mu_alive * dRp_dab[:, :, None, None])
    e_al = jnp.sum(alive_factor * mu_alive * R_p[:, :, None, None])

    jac_lin_al = alive_factor * mup_alive * s_val
    dRp_das_b = dRp_das[:, :, None, None]
    dRp_dab_b = dRp_dab[:, :, None, None]
    R_p_b = R_p[:, :, None, None]
    dr_da_s_b = dr_da_s[:, :, None, None]
    dr_da_b_b = dr_da_b[:, :, None, None]

    extra_ss_al = alive_factor * mu_alive * R_p_b * (
        dr_da_s_b * dr_da_s_b - sigma2_xr
    )
    extra_bb_al = alive_factor * mu_alive * R_p_b * (
        dr_da_b_b * dr_da_b_b - sigma2_xb
    )
    extra_sb_al = alive_factor * mu_alive * R_p_b * (
        dr_da_s_b * dr_da_b_b - sigma_xrxb
    )

    J_ss_al = jnp.sum(jac_lin_al * dRp_das_b * dRp_das_b + extra_ss_al)
    J_bb_al = jnp.sum(jac_lin_al * dRp_dab_b * dRp_dab_b + extra_bb_al)
    J_sb_al = jnp.sum(jac_lin_al * dRp_das_b * dRp_dab_b + extra_sb_al)

    return (
        foc_s_bq + foc_s_al,
        foc_b_bq + foc_b_al,
        J_ss_bq + J_ss_al,
        J_bb_bq + J_bb_al,
        J_sb_bq + J_sb_al,
        e_bq + e_al,
    )


def _solve_working_at_cell_pi_z(
    z_idx,
    i_s,
    c_next,
    income_next_by_z,
    z_transition_probs,
    eps_weights,
    psi_per_z,
    log_R_bill_i,
    log_x_s_i,
    log_x_b_i,
    j_corners_i,
    w_corners_i,
    weight_kv_kr,
    A_per_state,
    s_grid,
    wealth_grid,
    init_a_s,
    init_a_b,
    gamma,
    beta,
    b_bar,
    delta,
    sigma2_xr,
    sigma2_xb,
    sigma_xrxb,
    tol,
    max_iter,
    max_backtrack_iter,
    line_search_max_step,
    singular_det,
    grad_step_size,
    grad_denom_eps,
    tiny_savings,
    euler_inv_floor,
    min_consumption,
    egm_anchor,
    use_fori,
    use_line_search,
    gather_dtype,
):
    A_is = A_per_state[i_s]
    psi_z = psi_per_z[z_idx]

    c_corners = c_next[:, j_corners_i, :]
    c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))
    c_corners_T = _base._cast_for_gather(c_corners_T, gather_dtype)

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return working_foc_jac_ccv_pi_z(
                a_s,
                a_b,
                s_val,
                psi_z,
                log_R_bill_i,
                log_x_s_i,
                log_x_b_i,
                weight_kv_kr,
                w_corners_i,
                c_corners_T,
                wealth_grid,
                income_next_by_z,
                z_transition_probs,
                eps_weights,
                A_is,
                sigma2_xr,
                sigma2_xb,
                sigma_xrxb,
                gamma,
                b_bar,
                delta,
                min_consumption,
                gather_dtype=gather_dtype,
            )

        return foc_fn

    (
        x_egm,
        c_egm,
        a_s_egm,
        a_b_egm,
        n_iters_egm,
        n_backtrack_egm,
        exit_code_egm,
    ) = _base._egm_scan_cell(
        foc_factory,
        s_grid,
        init_a_s,
        init_a_b,
        gamma,
        beta,
        tol,
        max_iter,
        max_backtrack_iter,
        line_search_max_step,
        singular_det,
        grad_step_size,
        grad_denom_eps,
        tiny_savings,
        euler_inv_floor,
        min_consumption,
        egm_anchor,
        use_fori,
        use_line_search,
    )
    c_w, a_s_w, a_b_w = _base._lift_to_wealth_grid(
        x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid, min_consumption,
    )
    a_s_grid_per_s = a_s_egm[1:]
    a_b_grid_per_s = a_b_egm[1:]
    n_iters_per_s = n_iters_egm[1:]
    n_backtrack_per_s = n_backtrack_egm[1:]
    exit_code_per_s = exit_code_egm[1:]
    return (c_w, a_s_w, a_b_w,
            a_s_grid_per_s, a_b_grid_per_s,
            n_iters_per_s, n_backtrack_per_s, exit_code_per_s)


def _build_per_age_working_kernel_pi_z(
    pcj,
    mp,
    sc,
    n_dev,
    n_z,
    N_state,
    use_pension_next,
    per_is_tensors,
):
    if n_dev == 1:
        return _build_per_age_working_kernel_pi_z_vmap_only(
            pcj, mp, sc, n_z, N_state, use_pension_next, per_is_tensors
        )
    return _build_per_age_working_kernel_pi_z_pmap(
        pcj, mp, sc, n_dev, n_z, N_state, use_pension_next, per_is_tensors
    )


def _build_per_age_working_kernel_pi_z_pmap(
    pcj,
    mp,
    sc,
    n_dev,
    n_z,
    N_state,
    use_pension_next,
    per_is_tensors,
):
    gather_dtype = _base._resolve_gather_dtype(sc)
    static = (
        sc.tol,
        sc.max_iter,
        sc.max_backtrack_iter,
        sc.line_search_max_step,
        sc.singular_det,
        sc.grad_step_size,
        sc.grad_denom_eps,
        sc.tiny_savings,
        sc.euler_inv_floor,
        sc.min_consumption,
        sc.egm_anchor,
        bool(sc.use_fori_newton),
        bool(sc.use_line_search),
        gather_dtype,
    )

    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    n_chunks = int(sc.cell_vmap_chunks)
    if n_chunks == 1:
        n_cells = n_z * N_state
        pad_n = math.ceil(n_cells / n_dev) * n_dev
        per_dev = pad_n // n_dev

        cell_idx = np.arange(n_cells, dtype=np.int64)
        cell_idx_padded = np.concatenate(
            [cell_idx, np.full(pad_n - n_cells, cell_idx[-1])]
        )
        z_idx_padded = (cell_idx_padded // N_state).astype(np.int64)
        is_idx_padded = (cell_idx_padded % N_state).astype(np.int64)
        z_pm = jnp.asarray(z_idx_padded.reshape(n_dev, per_dev))
        is_pm = jnp.asarray(is_idx_padded.reshape(n_dev, per_dev))
    else:
        z_chunks_pm, is_chunks_pm, n_cells, chunk_size = (
            _base._build_chunked_pmap_index_arrays(n_z, N_state, n_chunks, n_dev)
        )

    @partial(pmap, in_axes=(0, 0, None, None, None, None, None, None))
    def per_dev_solve(
        z_block,
        is_block,
        c_next,
        income_next_table_z,
        pension_next_by_z,
        psi_per_z,
        init_a_s_arr,
        init_a_b_arr,
    ):
        def per_cell(z_idx, i_s):
            z_probs = pcj.Pi_z[z_idx]
            if use_pension_next:
                income_table = (
                    pension_next_by_z[:, None]
                    * jnp.ones_like(pcj.eps_weights)[None, :]
                )
            else:
                income_table = income_next_table_z[z_idx]

            init_a_s_cell = init_a_s_arr[z_idx, i_s, :]   # (n_savings,)
            init_a_b_cell = init_a_b_arr[z_idx, i_s, :]

            return _solve_working_at_cell_pi_z(
                z_idx,
                i_s,
                c_next,
                income_table,
                z_probs,
                pcj.eps_weights,
                psi_per_z,
                log_R_bill_all[i_s],
                log_x_s_all[i_s],
                log_x_b_all[i_s],
                j_corners_all[i_s],
                w_corners_all[i_s],
                pcj.weight_kv_kr,
                pcj.annuity_factors,
                pcj.s_grid,
                pcj.wealth_grid,
                init_a_s_cell,
                init_a_b_cell,
                mp.gamma,
                mp.beta,
                mp.b_bar,
                mp.delta,
                pcj.sigma2_xr,
                pcj.sigma2_xb,
                pcj.sigma_xrxb,
                *static,
            )

        return vmap(per_cell)(z_block, is_block)

    if n_chunks != 1:
        runner = _base._chunked_pmap_runner(
            per_dev_solve, z_chunks_pm, is_chunks_pm, n_cells, chunk_size, n_chunks
        )

    def call(
        c_next_jnp,
        income_next_table,
        pension_next_by_z,
        psi_per_z,
        init_a_s_arr,
        init_a_b_arr,
    ):
        if n_chunks != 1:
            (c_flat, s_flat, b_flat,
             as_grid_flat, ab_grid_flat,
             ni_flat, nb_flat, ec_flat) = runner(
                c_next_jnp,
                income_next_table,
                pension_next_by_z,
                psi_per_z,
                init_a_s_arr,
                init_a_b_arr,
            )
            return (
                jnp.reshape(c_flat, (n_z, N_state, -1)),
                jnp.reshape(s_flat, (n_z, N_state, -1)),
                jnp.reshape(b_flat, (n_z, N_state, -1)),
                jnp.reshape(as_grid_flat, (n_z, N_state, -1)),
                jnp.reshape(ab_grid_flat, (n_z, N_state, -1)),
                jnp.reshape(ni_flat, (n_z, N_state, -1)),
                jnp.reshape(nb_flat, (n_z, N_state, -1)),
                jnp.reshape(ec_flat, (n_z, N_state, -1)),
            )

        (c_pm, as_pm, ab_pm,
         as_grid_pm, ab_grid_pm,
         ni_pm, nb_pm, ec_pm) = per_dev_solve(
            z_pm,
            is_pm,
            c_next_jnp,
            income_next_table,
            pension_next_by_z,
            psi_per_z,
            init_a_s_arr,
            init_a_b_arr,
        )

        def collapse(a):
            flat = jnp.reshape(a, (pad_n,) + a.shape[2:])
            return jnp.reshape(flat[:n_cells], (n_z, N_state) + a.shape[2:])

        return (
            collapse(c_pm),
            collapse(as_pm),
            collapse(ab_pm),
            collapse(as_grid_pm),
            collapse(ab_grid_pm),
            collapse(ni_pm),
            collapse(nb_pm),
            collapse(ec_pm),
        )

    return call


def _build_per_age_working_kernel_pi_z_vmap_only(
    pcj,
    mp,
    sc,
    n_z,
    N_state,
    use_pension_next,
    per_is_tensors,
):
    gather_dtype = _base._resolve_gather_dtype(sc)
    static = (
        sc.tol,
        sc.max_iter,
        sc.max_backtrack_iter,
        sc.line_search_max_step,
        sc.singular_det,
        sc.grad_step_size,
        sc.grad_denom_eps,
        sc.tiny_savings,
        sc.euler_inv_floor,
        sc.min_consumption,
        sc.egm_anchor,
        bool(sc.use_fori_newton),
        bool(sc.use_line_search),
        gather_dtype,
    )

    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    n_chunks = int(sc.cell_vmap_chunks)
    z_idx_padded, is_idx_padded, n_cells, chunk_size = _base._build_chunked_index_arrays(
        n_z, N_state, n_chunks
    )

    def per_cell(
        z_idx,
        i_s,
        c_next,
        income_next_table_z,
        pension_next_by_z,
        psi_per_z,
        init_a_s_arr,
        init_a_b_arr,
    ):
        z_probs = pcj.Pi_z[z_idx]
        if use_pension_next:
            income_table = (
                pension_next_by_z[:, None] * jnp.ones_like(pcj.eps_weights)[None, :]
            )
        else:
            income_table = income_next_table_z[z_idx]

        init_a_s_cell = init_a_s_arr[z_idx, i_s, :]   # (n_savings,)
        init_a_b_cell = init_a_b_arr[z_idx, i_s, :]

        return _solve_working_at_cell_pi_z(
            z_idx,
            i_s,
            c_next,
            income_table,
            z_probs,
            pcj.eps_weights,
            psi_per_z,
            log_R_bill_all[i_s],
            log_x_s_all[i_s],
            log_x_b_all[i_s],
            j_corners_all[i_s],
            w_corners_all[i_s],
            pcj.weight_kv_kr,
            pcj.annuity_factors,
            pcj.s_grid,
            pcj.wealth_grid,
            init_a_s_cell,
            init_a_b_cell,
            mp.gamma,
            mp.beta,
            mp.b_bar,
            mp.delta,
            pcj.sigma2_xr,
            pcj.sigma2_xb,
            pcj.sigma_xrxb,
            *static,
        )

    @jit
    def per_chunk(
        c_next,
        income_next_table_z,
        pension_next_by_z,
        psi_per_z,
        init_a_s_arr,
        init_a_b_arr,
        z_chunk,
        is_chunk,
    ):
        return vmap(
            per_cell, in_axes=(0, 0, None, None, None, None, None, None)
        )(
            z_chunk,
            is_chunk,
            c_next,
            income_next_table_z,
            pension_next_by_z,
            psi_per_z,
            init_a_s_arr,
            init_a_b_arr,
        )

    if n_chunks == 1:
        z_full = z_idx_padded[:n_cells]
        is_full = is_idx_padded[:n_cells]

        def call(
            c_next_jnp,
            income_next_table,
            pension_next_by_z,
            psi_per_z,
            init_a_s_arr,
            init_a_b_arr,
        ):
            (c_flat, s_flat, b_flat,
             as_grid_flat, ab_grid_flat,
             ni_flat, nb_flat, ec_flat) = per_chunk(
                c_next_jnp,
                income_next_table,
                pension_next_by_z,
                psi_per_z,
                init_a_s_arr,
                init_a_b_arr,
                z_full,
                is_full,
            )
            return (
                jnp.reshape(c_flat, (n_z, N_state, -1)),
                jnp.reshape(s_flat, (n_z, N_state, -1)),
                jnp.reshape(b_flat, (n_z, N_state, -1)),
                jnp.reshape(as_grid_flat, (n_z, N_state, -1)),
                jnp.reshape(ab_grid_flat, (n_z, N_state, -1)),
                jnp.reshape(ni_flat, (n_z, N_state, -1)),
                jnp.reshape(nb_flat, (n_z, N_state, -1)),
                jnp.reshape(ec_flat, (n_z, N_state, -1)),
            )

        return call

    runner = _base._chunked_vmap_runner(
        per_chunk, z_idx_padded, is_idx_padded, n_cells, chunk_size, n_chunks
    )

    def call(
        c_next_jnp,
        income_next_table,
        pension_next_by_z,
        psi_per_z,
        init_a_s_arr,
        init_a_b_arr,
    ):
        (c_flat, s_flat, b_flat,
         as_grid_flat, ab_grid_flat,
         ni_flat, nb_flat, ec_flat) = runner(
            c_next_jnp,
            income_next_table,
            pension_next_by_z,
            psi_per_z,
            init_a_s_arr,
            init_a_b_arr,
        )
        return (
            jnp.reshape(c_flat, (n_z, N_state, -1)),
            jnp.reshape(s_flat, (n_z, N_state, -1)),
            jnp.reshape(b_flat, (n_z, N_state, -1)),
            jnp.reshape(as_grid_flat, (n_z, N_state, -1)),
            jnp.reshape(ab_grid_flat, (n_z, N_state, -1)),
            jnp.reshape(ni_flat, (n_z, N_state, -1)),
            jnp.reshape(nb_flat, (n_z, N_state, -1)),
            jnp.reshape(ec_flat, (n_z, N_state, -1)),
        )

    return call


def run_lifecycle_solver_pi_z(
    model,
    pc,
    solver_config=None,
    n_s_points=None,
    verbose=1,
    solve_control=None,
):
    """Run the production lifecycle solver with the Pi_z working kernel."""
    pc_pi = pc._replace(working_income_next=_build_discrete_income_next_table(pc))

    old_pc_to_jnp = _base._pc_to_jnp
    old_working_builder = _base._build_per_age_working_kernel
    try:
        _base._pc_to_jnp = _pc_to_jnp_with_pi_z
        _base._build_per_age_working_kernel = _build_per_age_working_kernel_pi_z
        C, S, B, diagnostics = _base.run_lifecycle_solver(
            model,
            pc_pi,
            solver_config=solver_config,
            n_s_points=n_s_points,
            verbose=verbose,
            solve_control=solve_control,
        )
    finally:
        _base._pc_to_jnp = old_pc_to_jnp
        _base._build_per_age_working_kernel = old_working_builder

    diagnostics = dict(diagnostics)
    diagnostics["solver_variant"] = "pi_z_tauchen"
    diagnostics["income_discretization_method"] = "Pi_z discrete next-state sum"
    return C, S, B, diagnostics

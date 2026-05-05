"""solver.py — JAX-pure lifecycle backward induction (EGM + 2D Newton).

Canonical model only:
  - Finite horizon, three assets (bills, stocks, nominal bonds).
  - CCV log-wealth dynamics (``r_p`` per Campbell-Viceira w8566 eq.10).
  - Shifted-bequest (luxury form, De Nardi 2004; Catherine 2025 normalisation).
  - Unconstrained portfolio (no simplex projection, no leverage caps).
  - Linear interpolation everywhere (no PCHIP).
  - ``n_state == 3`` (3D state grid). Ablations with smaller state vectors
    are deferred to a follow-up handoff.

Public entrypoint:
    run_lifecycle_solver(model, pc, solver_config, n_s_points, verbose, solve_control)
        -> (C_mat, S_mat, B_mat, diagnostics)

Output policy shape: ``(n_age, n_z, N_state, n_w)`` as np.ndarray, matching
the interface the simulator and downstream tooling already expect.

Module-level setup (jax x64 + virtual CPU device count) lives in
``lifecycle/__init__.py`` so it runs before any jax import.
"""

from __future__ import annotations

import math
import time
from collections import namedtuple
from functools import lru_cache, partial
from pathlib import Path
import csv

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit, pmap, vmap

from lifecycle.model import DELTA_BEQUEST, SolveControl, SolverConfig

# =============================================================================
# Exit codes
# =============================================================================
EC_INTERIOR = 1
EC_NEWTON_FAIL = 2
EC_TINY_SAVINGS = 3


# =============================================================================
# SCF wealth-probe helpers (NumPy; used for the per-age progress line)
# =============================================================================

_AWI_2019_USD = 54_099.99
_AWI_2019_KUSD = _AWI_2019_USD / 1_000.0
_SCF_WEALTH_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "scf_net_worth_by_age_2022.csv"
)
_PROGRESS_WEALTH_SOURCES = frozenset({"grid_midpoint", "scf_median", "scf_mean"})


@lru_cache(maxsize=1)
def _load_scf_wealth_age_table():
    """Load SCF wealth-by-age targets. Returns (age_mid, median_kusd, mean_kusd)
    with wealth in thousands of 2022 USD.
    """
    age_mid = []
    med_kusd = []
    mean_kusd = []
    with _SCF_WEALTH_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        )
        for row in reader:
            if row["age_group"].strip().lower() == "all":
                continue
            age_mid.append(float(row["age_midpoint"]))
            med_kusd.append(float(row["median_2022_k2022usd"]))
            mean_kusd.append(float(row["mean_2022_k2022usd"]))
    if len(age_mid) == 0:
        raise ValueError(f"SCF wealth file {_SCF_WEALTH_CSV} is empty")
    return np.array(age_mid), np.array(med_kusd), np.array(mean_kusd)


def _build_progress_wealth_schedule(ages, w_grid, source):
    source = str(source).strip().lower()
    w_grid = np.asarray(w_grid, dtype=float)
    ages = np.asarray(ages, dtype=float)
    if source == "grid_midpoint":
        probe_w = float(w_grid[len(w_grid) // 2])
        return np.full(ages.shape, probe_w, dtype=float), (
            f"grid midpoint (W={probe_w:.3f})"
        )
    age_mid, med_kusd, mean_kusd = _load_scf_wealth_age_table()
    wealth_kusd = med_kusd if source == "scf_median" else mean_kusd
    label = (f"SCF {'median' if source == 'scf_median' else 'mean'} "
             "wealth by age (2022 $, linear age interp, AWI-normalized)")
    wealth_model_units = wealth_kusd / _AWI_2019_KUSD
    schedule = np.interp(ages, age_mid, wealth_model_units)
    schedule = np.clip(schedule, float(w_grid[0]), float(w_grid[-1]))
    return schedule, label


def _interp_progress_policy_at_wealth(policy_by_wealth, w_grid, wealth):
    """Linear interp of one (n_w,) policy slice at scalar ``wealth``."""
    return float(np.interp(wealth, w_grid, policy_by_wealth))


# =============================================================================
# Solve-control / checkpoint helpers
# =============================================================================

def _default_checkpoint_path(_model, disc_config, youngest_age_to_solve):
    grid_sizes = "x".join(str(v) for v in disc_config.state_grid_sizes)
    age_suffix = (
        f"_to_age{int(youngest_age_to_solve)}"
        if youngest_age_to_solve is not None
        else "_partial"
    )
    name = (
        f"jax_{disc_config.state_grid_mode}"
        f"_grid{grid_sizes}_nz{disc_config.n_z}{age_suffix}"
    )
    return str(Path("saved_runs") / "checkpoints" / name)


def _normalize_solve_control(model, pc, solve_control):
    if solve_control is None:
        return SolveControl(), False
    if not isinstance(solve_control, SolveControl):
        try:
            defaults = SolveControl()._asdict()
            solve_control = SolveControl(
                **{
                    field: getattr(solve_control, field, default)
                    for field, default in defaults.items()
                }
            )
        except Exception as exc:
            raise TypeError(
                "solve_control must be a SolveControl instance or None"
            ) from exc

    youngest = solve_control.youngest_age_to_solve
    if youngest is not None:
        youngest = int(youngest)
        if youngest < model.start_age or youngest > model.terminal_age:
            raise ValueError(
                "youngest_age_to_solve must lie within "
                f"[{model.start_age}, {model.terminal_age}], got {youngest}"
            )

    every = solve_control.checkpoint_every_n_ages
    if every is not None:
        every = int(every)
        if every <= 0:
            raise ValueError("checkpoint_every_n_ages must be positive")

    checkpoint_path = solve_control.checkpoint_path
    if checkpoint_path is None and (
        youngest is not None or every is not None or solve_control.save_on_interrupt
    ):
        checkpoint_path = _default_checkpoint_path(model, pc.disc_config, youngest)

    if checkpoint_path is not None:
        checkpoint_path = str(Path(checkpoint_path))

    progress_wealth_source = solve_control.progress_wealth_source
    if progress_wealth_source is None:
        progress_wealth_source = SolveControl().progress_wealth_source
    progress_wealth_source = str(progress_wealth_source).strip().lower()
    if progress_wealth_source not in _PROGRESS_WEALTH_SOURCES:
        raise ValueError(
            "progress_wealth_source must be one of "
            f"{sorted(_PROGRESS_WEALTH_SOURCES)}, got {progress_wealth_source!r}"
        )

    return solve_control._replace(
        youngest_age_to_solve=youngest,
        checkpoint_path=checkpoint_path,
        checkpoint_every_n_ages=every,
        progress_wealth_source=progress_wealth_source,
    ), True


def _prepare_policy_snapshot(C_mat, S_mat, B_mat, solved_age_mask):
    if np.all(solved_age_mask):
        return C_mat, S_mat, B_mat
    unsolved = ~solved_age_mask
    Cs = C_mat.copy(); Ss = S_mat.copy(); Bs = B_mat.copy()
    Cs[unsolved] = np.nan; Ss[unsolved] = np.nan; Bs[unsolved] = np.nan
    return Cs, Ss, Bs


def _mask_unsolved_ages_in_place(C_mat, S_mat, B_mat, solved_age_mask):
    if np.all(solved_age_mask):
        return
    unsolved = ~solved_age_mask
    C_mat[unsolved] = np.nan
    S_mat[unsolved] = np.nan
    B_mat[unsolved] = np.nan


def _save_policy_checkpoint(checkpoint_path, C_mat, S_mat, B_mat, diagnostics):
    from lifecycle.policy_io import save_policy_bundle
    Cs, Ss, Bs = _prepare_policy_snapshot(
        C_mat, S_mat, B_mat, diagnostics["solved_age_mask"]
    )
    return save_policy_bundle(
        checkpoint_path, Cs, Ss, Bs,
        diagnostics=diagnostics, overwrite=True,
    )


# =============================================================================
# Pure JAX helpers
# =============================================================================

def interp_1d_lin_extrap(x, x_grid, y_grid):
    """Linear interp on a sorted grid with linear extrapolation outside."""
    n = x_grid.shape[0]
    iw = jnp.clip(jnp.searchsorted(x_grid, x, side="right") - 1, 0, n - 2)
    x0 = x_grid[iw]
    x1 = x_grid[iw + 1]
    y0 = y_grid[iw]
    y1 = y_grid[iw + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def interp_1d_lin_with_slope(x, x_grid, y_grid):
    """Linear interp returning ``(y, dy/dx)`` from one bracket."""
    n = x_grid.shape[0]
    iw = jnp.clip(jnp.searchsorted(x_grid, x, side="right") - 1, 0, n - 2)
    x0 = x_grid[iw]
    x1 = x_grid[iw + 1]
    y0 = y_grid[iw]
    y1 = y_grid[iw + 1]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0), slope


def bracket_uniform(z, z_lo, dz, n_z):
    """Bracket ``z`` in a uniform grid starting at ``z_lo`` with spacing ``dz``."""
    iz = jnp.clip(((z - z_lo) / dz).astype(jnp.int32), 0, n_z - 2)
    z0 = z_lo + dz * iz
    frac = jnp.clip((z - z0) / dz, 0.0, 1.0)
    return iz, frac


def bracket_state_3d_jax(s, grids_0, grids_1, grids_2, shift, L_inv):
    """Transform ``s`` to bracket coords and bracket on three axis grids.

    Canonical (n_state == 3) only. Each axis grid must have at least 2 points.
    """
    b = L_inv @ (s - shift)

    def _bracket_axis(grid, val):
        n = grid.shape[0]
        lo = jnp.clip(jnp.searchsorted(grid, val, side="right") - 1, 0, n - 2)
        denom = grid[lo + 1] - grid[lo]
        frac = jnp.clip((val - grid[lo]) / denom, 0.0, 1.0)
        return lo, frac

    lo0, f0 = _bracket_axis(grids_0, b[0])
    lo1, f1 = _bracket_axis(grids_1, b[1])
    lo2, f2 = _bracket_axis(grids_2, b[2])
    return jnp.array([lo0, lo1, lo2]), jnp.array([f0, f1, f2])


def bequest_mu_and_mup(W, A, gamma, b_bar, delta):
    """Shifted-bequest marginal utility and its W-derivative.

    Under CCV log-wealth dynamics ``W = s * exp(r_p)`` is strictly positive,
    so we compute through without a bankruptcy clamp. The Numba reference at
    canonical (CCV) settings takes the same unguarded path.
    """
    C_bar = W / A + delta
    mu = b_bar * C_bar ** (-gamma) / A
    mup = -gamma * mu / (A * C_bar)
    return mu, mup


# =============================================================================
# 2D Newton with backtracking line search
# =============================================================================

def newton_2d_with_line_search(
    foc_fn,
    init_a_s,
    init_a_b,
    scale,
    tol,
    max_iter,
    max_backtrack_iter,
    line_search_max_step,
    singular_det,
    grad_step_size,
    grad_denom_eps,
):
    """2D Newton on (alpha_s, alpha_b) with backtracking line search.

    ``foc_fn(a_s, a_b) -> (fs, fb, Jss, Jbb, Jsb, e)``. Returns
    ``(a_s, a_b, e, exit_code, err_norm, n_iter)``.
    """
    fs0, fb0, Jss0, Jbb0, Jsb0, e0 = foc_fn(init_a_s, init_a_b)
    err0 = jnp.sqrt(fs0 * fs0 + fb0 * fb0)

    init_state = (
        init_a_s, init_a_b,
        fs0, fb0, Jss0, Jbb0, Jsb0,
        e0, err0,
        jnp.int32(0),
        err0 < tol * scale,
    )

    def cond_fn(state):
        *_, k, done = state
        return jnp.logical_and(jnp.logical_not(done), k < max_iter)

    def body_fn(state):
        a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, k, _ = state

        det = Jss * Jbb - Jsb * Jsb
        is_singular = jnp.abs(det) < singular_det

        grad_norm = err + grad_denom_eps
        step_s_grad = grad_step_size * fs / grad_norm
        step_b_grad = grad_step_size * fb / grad_norm

        inv_d = 1.0 / jnp.where(is_singular, 1.0, det)
        step_s_newton = -(Jbb * fs - Jsb * fb) * inv_d
        step_b_newton = -(-Jsb * fs + Jss * fb) * inv_d

        step_s = jnp.where(is_singular, step_s_grad, step_s_newton)
        step_b = jnp.where(is_singular, step_b_grad, step_b_newton)

        slen = jnp.sqrt(step_s * step_s + step_b * step_b)
        cap = jnp.minimum(1.0, line_search_max_step / jnp.where(slen > 0.0, slen, 1.0))
        step_s = step_s * cap
        step_b = step_b * cap

        # Try alpha = 1 first.
        a_s_full = a_s + step_s
        a_b_full = a_b + step_b
        fs_f, fb_f, Jss_f, Jbb_f, Jsb_f, e_f = foc_fn(a_s_full, a_b_full)
        err_f = jnp.sqrt(fs_f * fs_f + fb_f * fb_f)
        full_improves = err_f < err

        bt_init = (
            jnp.int32(0),
            jnp.float64(1.0),
            jnp.where(full_improves, a_s_full, a_s),
            jnp.where(full_improves, a_b_full, a_b),
            jnp.where(full_improves, fs_f, fs),
            jnp.where(full_improves, fb_f, fb),
            jnp.where(full_improves, Jss_f, Jss),
            jnp.where(full_improves, Jbb_f, Jbb),
            jnp.where(full_improves, Jsb_f, Jsb),
            jnp.where(full_improves, e_f, e),
            jnp.where(full_improves, err_f, err),
            full_improves,
        )

        def bt_cond(bt_state):
            k_bt, *_, found = bt_state
            return jnp.logical_and(jnp.logical_not(found), k_bt < max_backtrack_iter)

        def bt_body(bt_state):
            (k_bt, alpha_t, _a_s, _a_b, _fs, _fb, _Jss, _Jbb, _Jsb, _e, _err, _found) = bt_state
            new_alpha = alpha_t * 0.5
            a_s_t = a_s + new_alpha * step_s
            a_b_t = a_b + new_alpha * step_b
            fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = foc_fn(a_s_t, a_b_t)
            err_t = jnp.sqrt(fs_t * fs_t + fb_t * fb_t)
            improved = err_t < err
            return (
                k_bt + 1, new_alpha,
                jnp.where(improved, a_s_t, _a_s),
                jnp.where(improved, a_b_t, _a_b),
                jnp.where(improved, fs_t, _fs),
                jnp.where(improved, fb_t, _fb),
                jnp.where(improved, Jss_t, _Jss),
                jnp.where(improved, Jbb_t, _Jbb),
                jnp.where(improved, Jsb_t, _Jsb),
                jnp.where(improved, e_t, _e),
                jnp.where(improved, err_t, _err),
                improved,
            )

        bt_final = lax.while_loop(bt_cond, bt_body, bt_init)
        (_k_bt, _alpha, new_a_s, new_a_b, new_fs, new_fb,
         new_Jss, new_Jbb, new_Jsb, new_e, new_err, found_any) = bt_final

        new_done = jnp.logical_or(
            new_err < tol * scale,
            jnp.logical_not(found_any),
        )
        return (
            new_a_s, new_a_b, new_fs, new_fb,
            new_Jss, new_Jbb, new_Jsb,
            new_e, new_err, k + 1, new_done,
        )

    final = lax.while_loop(cond_fn, body_fn, init_state)
    a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, k, _done = final
    exit_code = jnp.where(err < tol * scale, EC_INTERIOR, EC_NEWTON_FAIL)
    return a_s, a_b, e, exit_code, err / scale, k


# =============================================================================
# CCV log-return arithmetic shared across kernels
# =============================================================================

def _ccv_log_return_and_grad(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                              sigma2_xr, sigma2_xb, sigma_xrxb):
    """Return ``(R_p, dr/da_s, dr/da_b)`` for CCV log-wealth dynamics.

    ``r_p = log_R_bill + a_s*log_x_s + a_b*log_x_b
            + 0.5*(a_s*sigma2_xr + a_b*sigma2_xb)
            - 0.5*(a_s^2*sigma2_xr + 2*a_s*a_b*sigma_xrxb + a_b^2*sigma2_xb)``
    """
    r_p = (
        log_R_bill
        + alpha_s * log_x_s
        + alpha_b * log_x_b
        + 0.5 * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
        - 0.5 * (
            alpha_s * alpha_s * sigma2_xr
            + 2.0 * alpha_s * alpha_b * sigma_xrxb
            + alpha_b * alpha_b * sigma2_xb
        )
    )
    R_p = jnp.exp(r_p)
    dr_da_s = log_x_s + sigma2_xr * (0.5 - alpha_s) - alpha_b * sigma_xrxb
    dr_da_b = log_x_b + sigma2_xb * (0.5 - alpha_b) - alpha_s * sigma_xrxb
    return R_p, dr_da_s, dr_da_b


# =============================================================================
# Terminal-age FOC (CCV)
# =============================================================================

def terminal_foc_jac_ccv(
    alpha_s, alpha_b, s_val, A_is,
    log_R_bill, log_x_s, log_x_b,        # (n_state_quad, n_ret_quad)
    weight_kv_kr,                          # (n_state_quad, n_ret_quad)
    sigma2_xr, sigma2_xb, sigma_xrxb,
    gamma, b_bar, delta,
):
    """Terminal-period FOC + Hessian + ``V_dot = E[mu_bequest * R_p]``."""
    R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad(
        alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
        sigma2_xr, sigma2_xb, sigma_xrxb,
    )
    sR_p = s_val * R_p
    mu, mup = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)

    dRp_das = R_p * dr_da_s
    dRp_dab = R_p * dr_da_b

    wmu = weight_kv_kr * mu
    wmup = weight_kv_kr * mup

    foc_s = jnp.sum(wmu * dRp_das)
    foc_b = jnp.sum(wmu * dRp_dab)
    V_dot = jnp.sum(wmu * R_p)

    jac_lin = wmup * s_val
    extra_ss = wmu * R_p * (dr_da_s * dr_da_s - sigma2_xr)
    extra_bb = wmu * R_p * (dr_da_b * dr_da_b - sigma2_xb)
    extra_sb = wmu * R_p * (dr_da_s * dr_da_b - sigma_xrxb)

    J_ss = jnp.sum(jac_lin * dRp_das * dRp_das + extra_ss)
    J_bb = jnp.sum(jac_lin * dRp_dab * dRp_dab + extra_bb)
    J_sb = jnp.sum(jac_lin * dRp_das * dRp_dab + extra_sb)
    return foc_s, foc_b, J_ss, J_bb, J_sb, V_dot


# =============================================================================
# Per-cell EGM scan (one (z, i_s) cell, sequential over savings grid)
# =============================================================================

def _build_step_log_returns(state_grid_i, M_v_nodes, ret_nodes, const_r, A_r):
    """Per-i_s log-return scenario tensors of shape ``(n_state_quad, n_ret_quad)``.

    Same arithmetic as Numba ``_build_terminal_log_returns`` but for one i_s.
    """
    base_mu_r = const_r + A_r @ state_grid_i              # (n_ret,)
    mu_r_per = base_mu_r[None, :] + M_v_nodes              # (n_state_quad, n_ret)
    mu_bill = mu_r_per[:, 0]
    mu_xs = mu_r_per[:, 1]
    mu_xb = mu_r_per[:, 2]
    res_bill = ret_nodes[:, 0]
    res_xs = ret_nodes[:, 1]
    res_xb = ret_nodes[:, 2]
    log_R_bill = mu_bill[:, None] + res_bill[None, :]
    log_x_s = mu_xs[:, None] + res_xs[None, :]
    log_x_b = mu_xb[:, None] + res_xb[None, :]
    return log_R_bill, log_x_s, log_x_b


def _build_step_state_brackets(state_grid_i, Phi_0_state, Phi_11, v_nodes,
                                grids_0, grids_1, grids_2, shift, L_inv):
    """Per-i_s state bracketing. Returns ``(j_corners, w_corners)`` tensors of
    shape ``(n_state_quad, 8)``.

    Each ``j_corners[k_v, c]`` is a flat index into the joint state grid
    (``i_s in [0, N_state)``) and ``w_corners[k_v, c]`` is the trilinear
    weight of that corner.
    """
    # s_next: (n_state_quad, n_state)
    s_next = Phi_0_state[None, :] + state_grid_i @ Phi_11.T + v_nodes

    def per_kv(s_next_kv):
        lo, frac = bracket_state_3d_jax(s_next_kv, grids_0, grids_1, grids_2, shift, L_inv)
        f0, f1, f2 = frac[0], frac[1], frac[2]
        w = jnp.array([
            (1 - f0) * (1 - f1) * (1 - f2),
            (1 - f0) * (1 - f1) * f2,
            (1 - f0) * f1 * (1 - f2),
            (1 - f0) * f1 * f2,
            f0 * (1 - f1) * (1 - f2),
            f0 * (1 - f1) * f2,
            f0 * f1 * (1 - f2),
            f0 * f1 * f2,
        ])
        # 8 flat indices into the joint state grid (n_state == 3 layout).
        N1 = grids_1.shape[0]
        N2 = grids_2.shape[0]
        lo0, lo1, lo2 = lo[0], lo[1], lo[2]
        hi0, hi1, hi2 = lo0 + 1, lo1 + 1, lo2 + 1
        j = jnp.array([
            lo0 * N1 * N2 + lo1 * N2 + lo2,
            lo0 * N1 * N2 + lo1 * N2 + hi2,
            lo0 * N1 * N2 + hi1 * N2 + lo2,
            lo0 * N1 * N2 + hi1 * N2 + hi2,
            hi0 * N1 * N2 + lo1 * N2 + lo2,
            hi0 * N1 * N2 + lo1 * N2 + hi2,
            hi0 * N1 * N2 + hi1 * N2 + lo2,
            hi0 * N1 * N2 + hi1 * N2 + hi2,
        ])
        return j, w

    j_corners, w_corners = vmap(per_kv)(s_next)
    return j_corners, w_corners


def _interp_c_and_mpc_at_cell(c_next, j_corners_kv, w_corners_kv,
                               iz_lo, frac_z, x_next_scalar, wealth_grid,
                               min_consumption):
    """Trilinear-state × bilinear-z × linear-wealth interp of c_next and
    the marginal propensity to consume out of wealth (slope dc/dx).
    """
    def per_corner(j, w):
        c_lo, slope_lo = interp_1d_lin_with_slope(
            x_next_scalar, wealth_grid, c_next[iz_lo, j, :]
        )
        c_hi, slope_hi = interp_1d_lin_with_slope(
            x_next_scalar, wealth_grid, c_next[iz_lo + 1, j, :]
        )
        c_corner = (1.0 - frac_z) * c_lo + frac_z * c_hi
        slope_corner = (1.0 - frac_z) * slope_lo + frac_z * slope_hi
        return w * c_corner, w * slope_corner

    c_per, slope_per = vmap(per_corner)(j_corners_kv, w_corners_kv)
    c = jnp.maximum(jnp.sum(c_per), min_consumption)
    mpc = jnp.clip(jnp.sum(slope_per), 0.0, 1.0)
    return c, mpc


# -----------------------------------------------------------------------------
# Retirement-age FOC (CCV)
# -----------------------------------------------------------------------------

def retirement_foc_jac_ccv(
    alpha_s, alpha_b, s_val, z_idx, psi_z,
    log_R_bill, log_x_s, log_x_b, weight_kv_kr,   # (n_state_quad, n_ret_quad)
    j_corners, w_corners,                          # (n_state_quad, 8)
    c_next, wealth_grid,                           # (n_z, N_state, n_w), (n_w,)
    pension_next_z,                                # scalar at z_idx
    A_is,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    gamma, b_bar, delta, min_consumption,
):
    """Retirement FOC sums over ``(k_v, k_r)``: bequest + alive contributions.

    ``z`` is frozen at retirement (no eta/eps), so income_next = pension(z)
    is a scalar and c_next is gathered at exactly ``iz = z_idx`` (frac_z = 0).
    """
    R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad(
        alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
        sigma2_xr, sigma2_xb, sigma_xrxb,
    )                                              # (n_state_quad, n_ret_quad)
    sR_p = s_val * R_p
    x_next = sR_p + pension_next_z                 # (n_state_quad, n_ret_quad)

    mu_bq, mup_bq = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)

    # Trilinear-state × linear-wealth at every (k_v, k_r).
    # iz=z_idx exact, frac_z=0 → c_next gathered at one z slice only.
    def per_kv_kr(x_scalar, j_kv, w_kv):
        # 8 corners for this k_v.
        def per_corner(j, w):
            c, slope = interp_1d_lin_with_slope(
                x_scalar, wealth_grid, c_next[z_idx, j, :]
            )
            return w * c, w * slope
        c_per, slope_per = vmap(per_corner)(j_kv, w_kv)
        c = jnp.maximum(jnp.sum(c_per), min_consumption)
        mpc = jnp.clip(jnp.sum(slope_per), 0.0, 1.0)
        return c, mpc

    # Vmap over k_v (j_kv, w_kv vary), then over k_r (x_scalar varies).
    # Shape: (n_state_quad, n_ret_quad)
    c_at_xn, mpc_at_xn = vmap(
        lambda j_kv, w_kv, x_row: vmap(per_kv_kr, in_axes=(0, None, None))(
            x_row, j_kv, w_kv
        ),
        in_axes=(0, 0, 0),
    )(j_corners, w_corners, x_next)

    mu_alive = c_at_xn ** (-gamma)
    mup_alive = -gamma * mu_alive / c_at_xn * mpc_at_xn

    prob_death = 1.0 - psi_z
    mu_comb = psi_z * mu_alive + prob_death * mu_bq
    mup_comb = psi_z * mup_alive + prob_death * mup_bq

    wmu = weight_kv_kr * mu_comb
    wmup = weight_kv_kr * mup_comb

    dRp_das = R_p * dr_da_s
    dRp_dab = R_p * dr_da_b

    foc_s = jnp.sum(wmu * dRp_das)
    foc_b = jnp.sum(wmu * dRp_dab)
    e_sum = jnp.sum(wmu * R_p)

    jac_lin = wmup * s_val
    extra_ss = wmu * R_p * (dr_da_s * dr_da_s - sigma2_xr)
    extra_bb = wmu * R_p * (dr_da_b * dr_da_b - sigma2_xb)
    extra_sb = wmu * R_p * (dr_da_s * dr_da_b - sigma_xrxb)

    J_ss = jnp.sum(jac_lin * dRp_das * dRp_das + extra_ss)
    J_bb = jnp.sum(jac_lin * dRp_dab * dRp_dab + extra_bb)
    J_sb = jnp.sum(jac_lin * dRp_das * dRp_dab + extra_sb)
    return foc_s, foc_b, J_ss, J_bb, J_sb, e_sum


# -----------------------------------------------------------------------------
# Working-age FOC (CCV) — the main 4-axis broadcast
# -----------------------------------------------------------------------------

def working_foc_jac_ccv(
    alpha_s, alpha_b, s_val, z_idx, psi_z,
    log_R_bill, log_x_s, log_x_b, weight_kv_kr,    # (n_state_quad, n_ret_quad)
    j_corners, w_corners,                           # (n_state_quad, 8)
    c_next, wealth_grid,                            # (n_z, N_state, n_w), (n_w,)
    income_next_table,                              # (n_eta, n_eps) at this z_idx, t_idx
    eta_iz_lo, eta_frac_z,                          # (n_eta,) z-bracket per eta
    eta_weights, eps_weights,                       # (n_eta,), (n_eps,)
    A_is,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    gamma, b_bar, delta, min_consumption,
):
    """Working-age FOC. Bequest summed over ``(k_v, k_r)``; alive summed over
    ``(k_v, k_r, k_eta, i_e)``. Caller supplies the income_next gather table.
    """
    R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad(
        alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
        sigma2_xr, sigma2_xb, sigma_xrxb,
    )                                                # (n_state_quad, n_ret_quad)
    sR_p = s_val * R_p

    # ---- Bequest contribution (no eta/eps dependence) ----
    mu_bq, mup_bq = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)
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

    # ---- Alive contribution ----
    # x_next: (n_state_quad, n_ret_quad, n_eta, n_eps)
    x_next = sR_p[:, :, None, None] + income_next_table[None, None, :, :]

    # c_next interp: trilinear-state × bilinear-z × linear-wealth at every
    # (k_v, k_r, k_eta, i_e). Structured as nested vmaps.
    def at_eta_eps(x_scalar, iz, fz, j_kv, w_kv):
        return _interp_c_and_mpc_at_cell(
            c_next, j_kv, w_kv, iz, fz, x_scalar, wealth_grid, min_consumption,
        )

    # Innermost vmap: over i_e (n_eps); x_scalar varies.
    # Next vmap: over k_eta (n_eta); iz, fz vary, x_row varies along eps axis.
    # Next vmap: over k_r (n_ret_quad); x_block varies along (eta, eps).
    # Outer vmap: over k_v (n_state_quad); j_kv, w_kv, x_block vary.
    def per_kv(j_kv, w_kv, x_kv):
        # x_kv: (n_ret_quad, n_eta, n_eps)
        def per_kr(x_kr):
            # x_kr: (n_eta, n_eps)
            def per_keta(x_row, iz, fz):
                # x_row: (n_eps,)
                return vmap(at_eta_eps, in_axes=(0, None, None, None, None))(
                    x_row, iz, fz, j_kv, w_kv
                )
            return vmap(per_keta, in_axes=(0, 0, 0))(x_kr, eta_iz_lo, eta_frac_z)
        return vmap(per_kr)(x_kv)

    c_at_xn, mpc_at_xn = vmap(per_kv)(j_corners, w_corners, x_next)
    # Both: (n_state_quad, n_ret_quad, n_eta, n_eps)

    mu_alive = c_at_xn ** (-gamma)
    mup_alive = -gamma * mu_alive / c_at_xn * mpc_at_xn

    weight_full = (
        weight_kv_kr[:, :, None, None]
        * eta_weights[None, None, :, None]
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

    extra_ss_al = alive_factor * mu_alive * R_p_b * (dr_da_s_b * dr_da_s_b - sigma2_xr)
    extra_bb_al = alive_factor * mu_alive * R_p_b * (dr_da_b_b * dr_da_b_b - sigma2_xb)
    extra_sb_al = alive_factor * mu_alive * R_p_b * (dr_da_s_b * dr_da_b_b - sigma_xrxb)

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


# =============================================================================
# Per-cell EGM scan (one (z_idx, i_s)) — shared across kernels
# =============================================================================

def _egm_scan_cell(
    foc_factory,                # callable: foc_factory(s_val) -> foc_fn closure
    s_grid, init_a_s, init_a_b,
    gamma, beta,
    tol, max_iter, max_backtrack_iter,
    line_search_max_step, singular_det,
    grad_step_size, grad_denom_eps,
    tiny_savings, euler_inv_floor,
    min_consumption, egm_anchor,
):
    """Sweep ``s_grid`` (largest -> smallest) with warm-started Newton.

    Returns (x_egm, c_egm, a_s_egm, a_b_egm) each shape ``(n_savings + 1,)``,
    with the first entry being the egm_anchor at s = 0.
    """
    s_grid_rev = s_grid[::-1]

    def step(carry, s_val):
        warm_a_s, warm_a_b = carry
        foc_fn = foc_factory(s_val)
        _, _, _, _, _, e0 = foc_fn(0.0, 0.0)
        scale = jnp.maximum(jnp.abs(e0), 1e-30)

        a_s_opt, a_b_opt, V_dot, _exit_code, _err, _n_iter = newton_2d_with_line_search(
            foc_fn, warm_a_s, warm_a_b, scale,
            tol, max_iter, max_backtrack_iter,
            line_search_max_step, singular_det,
            grad_step_size, grad_denom_eps,
        )
        beta_e = jnp.maximum(beta * V_dot, euler_inv_floor)
        c_opt = jnp.maximum(beta_e ** (-1.0 / gamma), min_consumption)

        tiny = s_val <= tiny_savings
        c_out = jnp.where(tiny, min_consumption, c_opt)
        a_s_out = jnp.where(tiny, warm_a_s, a_s_opt)
        a_b_out = jnp.where(tiny, warm_a_b, a_b_opt)
        x_out = c_out + s_val
        return (a_s_out, a_b_out), (x_out, c_out, a_s_out, a_b_out)

    init_carry = (init_a_s, init_a_b)
    _, (x_rev, c_rev, a_s_rev, a_b_rev) = lax.scan(step, init_carry, s_grid_rev)

    x_arr = x_rev[::-1]
    c_arr = c_rev[::-1]
    a_s_arr = a_s_rev[::-1]
    a_b_arr = a_b_rev[::-1]

    x_egm = jnp.concatenate([jnp.array([egm_anchor], dtype=x_arr.dtype), x_arr])
    c_egm = jnp.concatenate([jnp.array([egm_anchor], dtype=c_arr.dtype), c_arr])
    a_s_egm = jnp.concatenate([jnp.array([0.0], dtype=a_s_arr.dtype), a_s_arr])
    a_b_egm = jnp.concatenate([jnp.array([0.0], dtype=a_b_arr.dtype), a_b_arr])
    return x_egm, c_egm, a_s_egm, a_b_egm


def _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid):
    """Linear interp from EGM endogenous grid to fixed wealth grid."""
    order = jnp.argsort(x_egm)
    x_sorted = x_egm[order]
    c_sorted = c_egm[order]
    a_s_sorted = a_s_egm[order]
    a_b_sorted = a_b_egm[order]
    c_w = jnp.interp(wealth_grid, x_sorted, c_sorted)
    a_s_w = jnp.interp(wealth_grid, x_sorted, a_s_sorted)
    a_b_w = jnp.interp(wealth_grid, x_sorted, a_b_sorted)
    return c_w, a_s_w, a_b_w


# =============================================================================
# Per-cell solve drivers (one cell, one age)
# =============================================================================

def _solve_terminal_at_i_s(
    log_R_bill_i, log_x_s_i, log_x_b_i, weight_kv_kr,
    A_is, s_grid, wealth_grid,
    init_a_s, init_a_b,
    gamma, beta, b_bar, delta,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    tol, max_iter, max_backtrack_iter,
    line_search_max_step, singular_det,
    grad_step_size, grad_denom_eps,
    tiny_savings, euler_inv_floor,
    min_consumption, egm_anchor,
):
    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return terminal_foc_jac_ccv(
                a_s, a_b, s_val, A_is,
                log_R_bill_i, log_x_s_i, log_x_b_i, weight_kv_kr,
                sigma2_xr, sigma2_xb, sigma_xrxb,
                gamma, b_bar, delta,
            )
        return foc_fn

    x_egm, c_egm, a_s_egm, a_b_egm = _egm_scan_cell(
        foc_factory, s_grid, init_a_s, init_a_b,
        gamma, beta,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
        tiny_savings, euler_inv_floor, min_consumption, egm_anchor,
    )
    return _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)


def _solve_retirement_at_cell(
    z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
    state_grid, M_v_nodes, ret_nodes, const_r, A_r,
    Phi_0_state, Phi_11, v_nodes,
    grids_0, grids_1, grids_2, shift, L_inv,
    weight_kv_kr, A_per_state,
    s_grid, wealth_grid,
    init_a_s, init_a_b,
    gamma, beta, b_bar, delta,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    tol, max_iter, max_backtrack_iter,
    line_search_max_step, singular_det,
    grad_step_size, grad_denom_eps,
    tiny_savings, euler_inv_floor,
    min_consumption, egm_anchor,
):
    state_grid_i = state_grid[i_s]
    log_R_bill, log_x_s, log_x_b = _build_step_log_returns(
        state_grid_i, M_v_nodes, ret_nodes, const_r, A_r
    )
    j_corners, w_corners = _build_step_state_brackets(
        state_grid_i, Phi_0_state, Phi_11, v_nodes,
        grids_0, grids_1, grids_2, shift, L_inv,
    )
    A_is = A_per_state[i_s]
    psi_z = psi_per_z[z_idx]
    pension_next_z = pension_next_by_z[z_idx]

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return retirement_foc_jac_ccv(
                a_s, a_b, s_val, z_idx, psi_z,
                log_R_bill, log_x_s, log_x_b, weight_kv_kr,
                j_corners, w_corners,
                c_next, wealth_grid,
                pension_next_z, A_is,
                sigma2_xr, sigma2_xb, sigma_xrxb,
                gamma, b_bar, delta, min_consumption,
            )
        return foc_fn

    x_egm, c_egm, a_s_egm, a_b_egm = _egm_scan_cell(
        foc_factory, s_grid, init_a_s, init_a_b,
        gamma, beta,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
        tiny_savings, euler_inv_floor, min_consumption, egm_anchor,
    )
    return _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)


def _solve_working_at_cell(
    z_idx, i_s, c_next,
    income_next_at_z,                  # (n_eta, n_eps) at this z_idx
    eta_iz_lo, eta_frac_z,             # (n_eta,) bracket at z_next[k_eta]
    eta_weights, eps_weights,
    psi_per_z,
    state_grid, M_v_nodes, ret_nodes, const_r, A_r,
    Phi_0_state, Phi_11, v_nodes,
    grids_0, grids_1, grids_2, shift, L_inv,
    weight_kv_kr, A_per_state,
    s_grid, wealth_grid,
    init_a_s, init_a_b,
    gamma, beta, b_bar, delta,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    tol, max_iter, max_backtrack_iter,
    line_search_max_step, singular_det,
    grad_step_size, grad_denom_eps,
    tiny_savings, euler_inv_floor,
    min_consumption, egm_anchor,
):
    state_grid_i = state_grid[i_s]
    log_R_bill, log_x_s, log_x_b = _build_step_log_returns(
        state_grid_i, M_v_nodes, ret_nodes, const_r, A_r
    )
    j_corners, w_corners = _build_step_state_brackets(
        state_grid_i, Phi_0_state, Phi_11, v_nodes,
        grids_0, grids_1, grids_2, shift, L_inv,
    )
    A_is = A_per_state[i_s]
    psi_z = psi_per_z[z_idx]

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return working_foc_jac_ccv(
                a_s, a_b, s_val, z_idx, psi_z,
                log_R_bill, log_x_s, log_x_b, weight_kv_kr,
                j_corners, w_corners,
                c_next, wealth_grid,
                income_next_at_z,
                eta_iz_lo, eta_frac_z,
                eta_weights, eps_weights,
                A_is,
                sigma2_xr, sigma2_xb, sigma_xrxb,
                gamma, b_bar, delta, min_consumption,
            )
        return foc_fn

    x_egm, c_egm, a_s_egm, a_b_egm = _egm_scan_cell(
        foc_factory, s_grid, init_a_s, init_a_b,
        gamma, beta,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
        tiny_savings, euler_inv_floor, min_consumption, egm_anchor,
    )
    return _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)


# =============================================================================
# Per-age JIT kernels (vmap over cells; pmap over devices)
# =============================================================================

# A pytree of jnp arrays + static scalars, threaded through the per-age kernels.
# Static scalars are bound by closure at kernel-build time (the orchestrator
# builds the kernel once per solve so the arrays are baked into the trace).

PCJax = namedtuple("PCJax", [
    "wealth_grid", "s_grid", "z_grid", "dz",
    "state_grid", "grids_0", "grids_1", "grids_2",
    "state_bracket_shift", "state_bracket_L_inv",
    "v_nodes", "v_weights", "M_v_nodes",
    "const_r", "A_r", "Phi_0_state", "Phi_11",
    "ret_nodes", "ret_weights",
    "eta_nodes", "eta_weights", "eps_weights",
    "annuity_factors",
    "sigma2_xr", "sigma2_xb", "sigma_xrxb",
    "weight_kv_kr",
])

ModelParams = namedtuple("ModelParams", [
    "gamma", "beta", "b_bar", "delta", "rho",
])


def _pc_to_jnp(pc, delta):
    """Pack the precompute arrays the JAX kernels need into a PCJax pytree."""
    grids = list(pc.state_bracket_grids)
    if len(grids) != 3:
        raise NotImplementedError(
            f"JAX solver supports n_state == 3 only (got {len(grids)}). "
            "Smaller-state ablations land in a follow-up handoff."
        )
    weight_kv_kr = jnp.asarray(pc.v_weights)[:, None] * jnp.asarray(pc.ret_weights)[None, :]
    return PCJax(
        wealth_grid=jnp.asarray(pc.wealth_grid),
        s_grid=jnp.asarray(pc.s_grid),
        z_grid=jnp.asarray(pc.z_grid),
        dz=jnp.asarray(pc.dz),
        state_grid=jnp.asarray(pc.state_grid),
        grids_0=jnp.asarray(grids[0]),
        grids_1=jnp.asarray(grids[1]),
        grids_2=jnp.asarray(grids[2]),
        state_bracket_shift=jnp.asarray(pc.state_bracket_shift),
        state_bracket_L_inv=jnp.asarray(pc.state_bracket_L_inv),
        v_nodes=jnp.asarray(pc.v_nodes),
        v_weights=jnp.asarray(pc.v_weights),
        M_v_nodes=jnp.asarray(pc.M_v_nodes),
        const_r=jnp.asarray(pc.const_r),
        A_r=jnp.asarray(pc.A_r),
        ret_nodes=jnp.asarray(pc.ret_nodes),
        ret_weights=jnp.asarray(pc.ret_weights),
        eta_nodes=jnp.asarray(pc.eta_nodes),
        eta_weights=jnp.asarray(pc.eta_weights),
        eps_weights=jnp.asarray(pc.eps_weights),
        annuity_factors=jnp.asarray(pc.annuity_factors),
        sigma2_xr=jnp.float64(pc.sigma2_xr),
        sigma2_xb=jnp.float64(pc.sigma2_xb),
        sigma_xrxb=jnp.float64(pc.sigma_xrxb),
        Phi_0_state=jnp.asarray(np.asarray(pc.model.Phi_0_state, dtype=np.float64)),
        Phi_11=jnp.asarray(np.asarray(pc.model.Phi_11, dtype=np.float64)),
        weight_kv_kr=weight_kv_kr,
    )


def _build_per_age_terminal_kernel(pcj, mp, sc, n_dev):
    """Build a pmap'd terminal kernel that returns ``(N_state_padded, n_w)``
    arrays. Padding is invisible at the call site — caller slices to
    ``[:N_state]`` after gather.
    """
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor)

    # Pre-build per-i_s log-return tensors (NumPy → jnp) once.
    state_grid_np = np.asarray(pcj.state_grid)
    const_r_np = np.asarray(pcj.const_r)
    A_r_np = np.asarray(pcj.A_r)
    M_v_nodes_np = np.asarray(pcj.M_v_nodes)
    ret_nodes_np = np.asarray(pcj.ret_nodes)
    base_mu_r = const_r_np[None, :] + state_grid_np @ A_r_np.T   # (N_state, n_ret)
    mu_r_per = base_mu_r[:, None, :] + M_v_nodes_np[None, :, :]  # (N_state, n_state_quad, n_ret)
    res_bill = ret_nodes_np[:, 0]
    res_xs = ret_nodes_np[:, 1]
    res_xb = ret_nodes_np[:, 2]
    log_R_bill = mu_r_per[:, :, 0:1] + res_bill[None, None, :]   # (N_state, n_state_quad, n_ret_quad)
    log_x_s = mu_r_per[:, :, 1:2] + res_xs[None, None, :]
    log_x_b = mu_r_per[:, :, 2:3] + res_xb[None, None, :]
    # Squeeze the singleton n_ret slice.
    log_R_bill = log_R_bill.reshape(log_R_bill.shape[0], log_R_bill.shape[1], -1)
    log_x_s = log_x_s.reshape(log_x_s.shape[0], log_x_s.shape[1], -1)
    log_x_b = log_x_b.reshape(log_x_b.shape[0], log_x_b.shape[1], -1)

    N_state = state_grid_np.shape[0]
    pad_n = math.ceil(N_state / n_dev) * n_dev

    def pad0(arr, target_n):
        if arr.shape[0] == target_n:
            return arr
        last = arr[-1:]
        extra = target_n - arr.shape[0]
        return np.concatenate([arr] + [last] * extra, axis=0)

    log_R_bill_p = pad0(log_R_bill, pad_n)
    log_x_s_p = pad0(log_x_s, pad_n)
    log_x_b_p = pad0(log_x_b, pad_n)
    ann_p = pad0(np.asarray(pcj.annuity_factors), pad_n)

    per_dev = pad_n // n_dev

    def reshape_for_pmap(arr):
        return arr.reshape((n_dev, per_dev) + arr.shape[1:])

    log_R_bill_pm = jnp.asarray(reshape_for_pmap(log_R_bill_p))
    log_x_s_pm = jnp.asarray(reshape_for_pmap(log_x_s_p))
    log_x_b_pm = jnp.asarray(reshape_for_pmap(log_x_b_p))
    ann_pm = jnp.asarray(reshape_for_pmap(ann_p))

    @pmap
    def per_dev_solve(log_Rb_block, lxs_block, lxb_block, ann_block):
        def per_i_s(log_Rb, lxs, lxb, A):
            return _solve_terminal_at_i_s(
                log_Rb, lxs, lxb, pcj.weight_kv_kr, A,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s, init_a_b,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )
        return vmap(per_i_s)(log_Rb_block, lxs_block, lxb_block, ann_block)

    def call(_unused_age_idx=None):
        c_pm, as_pm, ab_pm = per_dev_solve(log_R_bill_pm, log_x_s_pm, log_x_b_pm, ann_pm)

        def collapse(a):
            arr = np.asarray(a).reshape((pad_n,) + a.shape[2:])
            return arr[:N_state]

        return collapse(c_pm), collapse(as_pm), collapse(ab_pm)

    return call


def _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state):
    """Build a pmap'd retirement kernel. Returns a callable
    ``call(c_next, pension_next_by_z, psi_per_z) -> (n_z, N_state, n_w)``.
    """
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor)

    n_cells = n_z * N_state
    pad_n = math.ceil(n_cells / n_dev) * n_dev
    per_dev = pad_n // n_dev

    cell_idx = np.arange(n_cells, dtype=np.int64)
    cell_idx_padded = np.concatenate([cell_idx, np.full(pad_n - n_cells, cell_idx[-1])])
    z_idx_padded = (cell_idx_padded // N_state).astype(np.int64)
    is_idx_padded = (cell_idx_padded % N_state).astype(np.int64)
    z_pm = jnp.asarray(z_idx_padded.reshape(n_dev, per_dev))
    is_pm = jnp.asarray(is_idx_padded.reshape(n_dev, per_dev))

    @partial(pmap, in_axes=(0, 0, None, None, None))
    def per_dev_solve(z_block, is_block, c_next, pension_next_by_z, psi_per_z):
        def per_cell(z_idx, i_s):
            return _solve_retirement_at_cell(
                z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
                pcj.state_grid, pcj.M_v_nodes, pcj.ret_nodes, pcj.const_r, pcj.A_r,
                pcj.Phi_0_state, pcj.Phi_11, pcj.v_nodes,
                pcj.grids_0, pcj.grids_1, pcj.grids_2,
                pcj.state_bracket_shift, pcj.state_bracket_L_inv,
                pcj.weight_kv_kr, pcj.annuity_factors,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s, init_a_b,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )
        return vmap(per_cell)(z_block, is_block)

    def call(c_next_jnp, pension_next_by_z, psi_per_z):
        c_pm, as_pm, ab_pm = per_dev_solve(z_pm, is_pm, c_next_jnp, pension_next_by_z, psi_per_z)
        # (n_dev, per_dev, n_w) -> (pad_n, n_w) -> (n_cells, n_w) -> (n_z, N_state, n_w)
        def collapse(a):
            flat = np.asarray(a).reshape((pad_n,) + a.shape[2:])
            return flat[:n_cells].reshape(n_z, N_state, -1)
        return collapse(c_pm), collapse(as_pm), collapse(ab_pm)

    return call


def _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next):
    """Build a pmap'd working-age kernel. The boundary case
    (work -> retirement, ``use_pension_next == True``) is a separate trace
    selected by the orchestrator.
    """
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor)

    n_cells = n_z * N_state
    pad_n = math.ceil(n_cells / n_dev) * n_dev
    per_dev = pad_n // n_dev

    cell_idx = np.arange(n_cells, dtype=np.int64)
    cell_idx_padded = np.concatenate([cell_idx, np.full(pad_n - n_cells, cell_idx[-1])])
    z_idx_padded = (cell_idx_padded // N_state).astype(np.int64)
    is_idx_padded = (cell_idx_padded % N_state).astype(np.int64)
    z_pm = jnp.asarray(z_idx_padded.reshape(n_dev, per_dev))
    is_pm = jnp.asarray(is_idx_padded.reshape(n_dev, per_dev))

    @partial(pmap, in_axes=(0, 0, None, None, None, None))
    def per_dev_solve(
        z_block, is_block, c_next,
        income_next_table_z, pension_next_by_z, psi_per_z,
    ):
        def per_cell(z_idx, i_s):
            # z_next = rho * z + eta_nodes  ->  (n_eta,)
            z_now = pcj.z_grid[z_idx]
            z_next = mp.rho * z_now + pcj.eta_nodes
            iz_lo, frac_z = vmap(bracket_uniform, in_axes=(0, None, None, None))(
                z_next, pcj.z_grid[0], pcj.dz, pcj.z_grid.shape[0]
            )

            if use_pension_next:
                # Work->retirement boundary: income_next is the pension at
                # bracketed z_next; broadcast across i_e.
                pension_at_eta = (
                    (1.0 - frac_z) * pension_next_by_z[iz_lo]
                    + frac_z * pension_next_by_z[iz_lo + 1]
                )
                income_table = pension_at_eta[:, None] * jnp.ones_like(pcj.eps_weights)[None, :]
            else:
                # Working: gather the precomputed table at z_idx (shape (n_eta, n_eps)).
                income_table = income_next_table_z[z_idx]

            return _solve_working_at_cell(
                z_idx, i_s, c_next,
                income_table,
                iz_lo, frac_z,
                pcj.eta_weights, pcj.eps_weights,
                psi_per_z,
                pcj.state_grid, pcj.M_v_nodes, pcj.ret_nodes, pcj.const_r, pcj.A_r,
                pcj.Phi_0_state, pcj.Phi_11, pcj.v_nodes,
                pcj.grids_0, pcj.grids_1, pcj.grids_2,
                pcj.state_bracket_shift, pcj.state_bracket_L_inv,
                pcj.weight_kv_kr, pcj.annuity_factors,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s, init_a_b,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )

        return vmap(per_cell)(z_block, is_block)

    def call(c_next_jnp, income_next_table, pension_next_by_z, psi_per_z):
        c_pm, as_pm, ab_pm = per_dev_solve(
            z_pm, is_pm, c_next_jnp, income_next_table, pension_next_by_z, psi_per_z
        )
        def collapse(a):
            flat = np.asarray(a).reshape((pad_n,) + a.shape[2:])
            return flat[:n_cells].reshape(n_z, N_state, -1)
        return collapse(c_pm), collapse(as_pm), collapse(ab_pm)

    return call


# =============================================================================
# Top-level orchestrator
# =============================================================================

def run_lifecycle_solver(
    model, pc, solver_config=None, n_s_points=None,
    verbose=1, solve_control=None,
):
    """Lifecycle backward induction solver — JAX implementation.

    Parameters
    ----------
    model : LifecyclePortfolioModel
    pc    : Precompute (NamedTuple from build_precompute)
    solver_config : SolverConfig | None
    n_s_points : int | None
        Override savings grid size (uses ``pc.regenerate_savings_grid``).
    verbose : int  -- 0 silent, 1 per-age progress + post-solve summary
    solve_control : SolveControl | None

    Returns
    -------
    C_mat, S_mat, B_mat : np.ndarray, shape (n_age, n_z, N_state, n_w)
    diagnostics : dict
    """
    if solver_config is None:
        solver_config = SolverConfig()
    sc = solver_config

    if sc.wealth_dynamics_spec != "ccv_log":
        raise NotImplementedError(
            "JAX solver implements the canonical ccv_log wealth dynamics only. "
            f"Got wealth_dynamics_spec={sc.wealth_dynamics_spec!r}."
        )
    delta = sc.delta_bequest if sc.delta_bequest >= 0.0 else DELTA_BEQUEST

    solve_control, control_active = _normalize_solve_control(model, pc, solve_control)

    if verbose >= 1:
        print(f"\n{'='*70}")
        print(f"LIFECYCLE PORTFOLIO SOLVER  (JAX, EGM + 2D Newton)")
        print(f"  Devices: {jax.devices()}")
        print(f"  Solver: {solver_config}")
        print(f"  Discretization: {pc.disc_config}")
        if control_active:
            print(f"  Solve control: {solve_control}")
        print(f"{'='*70}")

    # Override savings grid if requested.
    if n_s_points is not None:
        pc = pc._replace(s_grid=jnp.asarray(pc.regenerate_savings_grid(n_s_points)),
                         n_s=int(n_s_points))

    n_age = pc.n_age
    n_z = pc.n_z
    N_state = pc.N_state
    n_w = pc.n_w
    ages = np.asarray(pc.ages)
    retire_age = model.retire_age
    terminal_age = model.terminal_age

    # ---- Pack arrays for the JAX kernels ----
    pcj = _pc_to_jnp(pc, delta)
    mp = ModelParams(
        gamma=jnp.float64(model.gamma),
        beta=jnp.float64(model.beta),
        b_bar=jnp.float64(model.b_bar),
        delta=jnp.float64(delta),
        rho=jnp.float64(model.rho),
    )

    # Build per-age kernels once (per-(z, i_s) pmap layout is fixed for a run).
    n_dev = len(jax.devices())
    terminal_kernel = _build_per_age_terminal_kernel(pcj, mp, sc, n_dev)
    retirement_kernel = _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state)
    working_kernel = _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next=False)
    boundary_kernel = _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next=True)

    # ---- Output arrays ----
    shape = (n_age, n_z, N_state, n_w)
    C_mat = np.zeros(shape)
    S_mat = np.zeros(shape)
    B_mat = np.zeros(shape)
    solved_age_mask = np.zeros(n_age, dtype=bool)

    # ---- Per-age diagnostics (post-hoc only) ----
    age_max_foc = np.zeros(n_age)
    age_newton_fail = np.zeros(n_age, dtype=np.int64)

    checkpoint_path = solve_control.checkpoint_path
    youngest_age_to_solve = solve_control.youngest_age_to_solve
    checkpoint_every_n_ages = solve_control.checkpoint_every_n_ages
    save_on_interrupt = solve_control.save_on_interrupt
    return_partial_on_interrupt = solve_control.return_partial_on_interrupt
    checkpoint_save_count = 0
    last_saved_nonterminal_count = -1
    last_saved_bundle_path = None

    # ---- Optional resume from checkpoint ----
    if checkpoint_path is not None:
        ckpt_dir = Path(checkpoint_path)
        if (ckpt_dir / "policy_arrays.npz").exists():
            from lifecycle.policy_io import load_policy_bundle
            try:
                Cc, Sc, Bc, ckpt_diag, _ = load_policy_bundle(ckpt_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Found checkpoint at {ckpt_dir} but failed to load it: {exc}. "
                    "Delete the checkpoint or fix the bundle before retrying."
                ) from exc
            if Cc.shape != C_mat.shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch at {ckpt_dir}: "
                    f"got {Cc.shape}, expected {C_mat.shape}. "
                    "Different grid/quadrature/system — refuse to resume."
                )
            ckpt_mask = None
            if ckpt_diag is not None:
                ckpt_mask = ckpt_diag.get("solved_age_mask")
            if ckpt_mask is None or len(ckpt_mask) != n_age:
                raise RuntimeError(
                    f"Checkpoint at {ckpt_dir} missing or malformed solved_age_mask."
                )
            ckpt_mask = np.asarray(ckpt_mask, dtype=bool)
            for t in range(n_age):
                if ckpt_mask[t]:
                    C_mat[t] = Cc[t]; S_mat[t] = Sc[t]; B_mat[t] = Bc[t]
                    solved_age_mask[t] = True
            n_resumed = int(np.sum(solved_age_mask))
            if verbose >= 1 and n_resumed > 0:
                resumed_ages = ages[np.flatnonzero(solved_age_mask)]
                print(
                    f"\n  Resumed from checkpoint {ckpt_dir}: "
                    f"{n_resumed}/{n_age} ages already solved "
                    f"(ages {int(resumed_ages.min())}-{int(resumed_ages.max())})"
                )

    # ---- Per-age progress wealth ----
    progress_wealth_source = solve_control.progress_wealth_source
    progress_wealth_by_age = None
    progress_wealth_label = None
    if verbose >= 1:
        try:
            progress_wealth_by_age, progress_wealth_label = _build_progress_wealth_schedule(
                ages=ages, w_grid=pc.wealth_grid, source=progress_wealth_source,
            )
        except Exception as exc:
            progress_wealth_by_age, progress_wealth_label = _build_progress_wealth_schedule(
                ages=ages, w_grid=pc.wealth_grid, source="grid_midpoint",
            )
            print(f"  WARNING: SCF wealth probe unavailable ({exc}); using grid midpoint.")

    i_z_med = n_z // 2
    i_s_med = N_state // 2

    # ---- Terminal age ----
    if not solved_age_mask[-1]:
        if verbose >= 1:
            print(f"\n  Terminal condition (age {terminal_age}) ... ", end="", flush=True)
        c_T, s_T, b_T = terminal_kernel()
        # Broadcast across z (terminal policy is z-invariant — bequest only).
        C_mat[-1] = np.broadcast_to(c_T[None, :, :], (n_z, N_state, n_w)).copy()
        S_mat[-1] = np.broadcast_to(s_T[None, :, :], (n_z, N_state, n_w)).copy()
        B_mat[-1] = np.broadcast_to(b_T[None, :, :], (n_z, N_state, n_w)).copy()
        solved_age_mask[-1] = True
        if verbose >= 1:
            print(f"done  [c range: {c_T.min():.3f}-{c_T.max():.3f}]")
    elif verbose >= 1:
        print(f"\n  Terminal condition (age {terminal_age}) ... loaded from checkpoint")

    # ---- Backward induction header ----
    if verbose >= 1:
        print(f"\n{'='*100}")
        print(f"  Live policy probe: z=z_grid[{i_z_med}], state midpoint, wealth={progress_wealth_label}")
        hdr = (f" {'Age':>3}  {'Phase':<6} {'Time':>6}  "
               f"{'alpha_s':>7}  {'alpha_b':>7}  {'a_bill':>7}  {'W':>6}  {'c/W':>5}")
        print(hdr)
        print(f"{'='*100}")

    t_start = time.time()
    solve_status = "complete"

    pension_table = np.asarray(pc.pension_after_tax)
    pension_dummy_z = np.zeros(n_z, dtype=np.float64)
    survival = np.asarray(pc.survival_probs_2d)
    working_income_next_full = np.asarray(pc.working_income_next)  # (n_age, n_z, n_eta, n_eps)

    try:
        for t in reversed(range(n_age - 1)):
            age = ages[t]
            if youngest_age_to_solve is not None and age < youngest_age_to_solve:
                solve_status = "stopped_early"
                break
            if solved_age_mask[t]:
                continue

            psi_t = jnp.asarray(survival[t, :])
            c_next_jnp = jnp.asarray(C_mat[t + 1])

            if age >= retire_age:
                pension_next = jnp.asarray(pension_table[t + 1, :])
                c_t, s_t, b_t = retirement_kernel(c_next_jnp, pension_next, psi_t)
                label = "RETIRE"
            else:
                use_pen = (age == retire_age - 1)
                if use_pen:
                    pension_next = jnp.asarray(pension_table[t + 1, :])
                    income_table = jnp.zeros((n_z, pc.n_eta, pc.n_eps))   # ignored on this branch
                    c_t, s_t, b_t = boundary_kernel(c_next_jnp, income_table, pension_next, psi_t)
                else:
                    pension_next = jnp.asarray(pension_dummy_z)
                    income_table = jnp.asarray(working_income_next_full[t + 1])  # (n_z, n_eta, n_eps)
                    c_t, s_t, b_t = working_kernel(c_next_jnp, income_table, pension_next, psi_t)
                label = "WORK  "

            C_mat[t] = c_t
            S_mat[t] = s_t
            B_mat[t] = b_t
            solved_age_mask[t] = True

            if verbose >= 1:
                elapsed = time.time() - t_start
                probe_w = float(progress_wealth_by_age[t])
                probe_as = _interp_progress_policy_at_wealth(s_t[i_z_med, i_s_med, :], pc.wealth_grid, probe_w)
                probe_ab = _interp_progress_policy_at_wealth(b_t[i_z_med, i_s_med, :], pc.wealth_grid, probe_w)
                probe_c = _interp_progress_policy_at_wealth(c_t[i_z_med, i_s_med, :], pc.wealth_grid, probe_w)
                probe_bill = 1.0 - probe_as - probe_ab
                c_over_w = probe_c / probe_w if probe_w > 0 else 0.0
                print(
                    f" {age:3d}  {label:<6} {elapsed:6.1f}s  "
                    f"{probe_as:7.3f}  {probe_ab:7.3f}  {probe_bill:7.3f}  "
                    f"{probe_w:6.2f}  {c_over_w:5.3f}",
                    flush=True,
                )

            if checkpoint_every_n_ages is not None and checkpoint_path is not None:
                solved_nonterminal_count = int(np.sum(solved_age_mask[:-1]))
                if solved_nonterminal_count - last_saved_nonterminal_count >= checkpoint_every_n_ages:
                    checkpoint_save_count += 1
                    diag = _build_diagnostics(
                        solved_age_mask=solved_age_mask, ages=ages,
                        age_max_foc=age_max_foc, age_newton_fail=age_newton_fail,
                        solver_config=solver_config, disc_config=pc.disc_config,
                        solve_control=solve_control, solve_status="checkpoint",
                        wall_time_sec=time.time() - t_start,
                        checkpoint_save_count=checkpoint_save_count,
                        checkpoint_path=checkpoint_path,
                    )
                    last_saved_bundle_path = str(_save_policy_checkpoint(
                        checkpoint_path, C_mat, S_mat, B_mat, diag,
                    ))
                    last_saved_nonterminal_count = solved_nonterminal_count
                    if verbose >= 1:
                        print(f"    checkpoint saved -> {last_saved_bundle_path}", flush=True)
    except KeyboardInterrupt:
        solve_status = "interrupted"
        if verbose >= 1:
            print("\n  Solve interrupted. Finalizing partial output...", flush=True)

    total = time.time() - t_start

    solved_nonterminal_count = int(np.sum(solved_age_mask[:-1]))
    final_save_needed = (
        checkpoint_path is not None
        and (
            solve_status == "stopped_early"
            or (solve_status == "interrupted" and save_on_interrupt)
            or (
                solve_status == "complete"
                and checkpoint_every_n_ages is not None
                and solved_nonterminal_count != last_saved_nonterminal_count
            )
        )
    )

    diag_kwargs = dict(
        solved_age_mask=solved_age_mask, ages=ages,
        age_max_foc=age_max_foc, age_newton_fail=age_newton_fail,
        solver_config=solver_config, disc_config=pc.disc_config,
        solve_control=solve_control, solve_status=solve_status,
        wall_time_sec=total,
        checkpoint_save_count=checkpoint_save_count + (1 if final_save_needed else 0),
        checkpoint_path=checkpoint_path,
    )
    diagnostics = _build_diagnostics(**diag_kwargs)

    if final_save_needed:
        last_saved_bundle_path = str(_save_policy_checkpoint(
            checkpoint_path, C_mat, S_mat, B_mat, diagnostics,
        ))
        checkpoint_save_count += 1

    diagnostics["checkpoint_save_count"] = checkpoint_save_count
    diagnostics["last_saved_bundle_path"] = last_saved_bundle_path

    if verbose >= 1:
        print(f"\n{'='*100}")
        n_solved = int(np.sum(solved_age_mask))
        print(f"  DONE in {total / 60:.2f} min  (avg {total / max(n_solved - 1, 1):.2f}s per age)")
        print(f"  Status: {diagnostics['solve_status']}  ({n_solved}/{n_age} ages solved)")
        if last_saved_bundle_path is not None:
            print(f"  Saved bundle: {last_saved_bundle_path}")
        # Lightweight policy sanity check.
        C_eval = C_mat[solved_age_mask]
        S_eval = S_mat[solved_age_mask]
        B_eval = B_mat[solved_age_mask]
        nan_c = int(np.isnan(C_eval).sum())
        nan_s = int(np.isnan(S_eval).sum())
        nan_b = int(np.isnan(B_eval).sum())
        inf_c = int(np.isinf(C_eval).sum())
        inf_s = int(np.isinf(S_eval).sum())
        inf_b = int(np.isinf(B_eval).sum())
        if nan_c + nan_s + nan_b + inf_c + inf_s + inf_b == 0:
            print(f"  Policy sanity: PASS  (no NaN/Inf in solved ages)")
        else:
            print(f"  Policy sanity: NaN(C={nan_c} S={nan_s} B={nan_b}) Inf(C={inf_c} S={inf_s} B={inf_b})")
        print(f"  alpha_s range: [{S_eval.min():.3f}, {S_eval.max():.3f}]")
        print(f"  alpha_b range: [{B_eval.min():.3f}, {B_eval.max():.3f}]")
        print(f"{'='*100}\n")

    if diagnostics["is_partial"]:
        _mask_unsolved_ages_in_place(C_mat, S_mat, B_mat, solved_age_mask)

    if solve_status == "interrupted" and not return_partial_on_interrupt:
        raise KeyboardInterrupt

    return C_mat, S_mat, B_mat, diagnostics


def _build_diagnostics(
    *, solved_age_mask, ages, age_max_foc, age_newton_fail,
    solver_config, disc_config, solve_control,
    solve_status, wall_time_sec, checkpoint_save_count, checkpoint_path,
):
    solved_idx = np.flatnonzero(solved_age_mask)
    solved_ages = ages[solved_idx] if solved_idx.size > 0 else np.array([], dtype=np.int64)
    youngest_solved_age = int(solved_ages.min()) if solved_ages.size > 0 else None
    oldest_solved_age = int(solved_ages.max()) if solved_ages.size > 0 else None
    is_partial = solve_status != "complete" or solved_idx.size != len(ages)

    return {
        "age_max_foc": age_max_foc.copy(),
        "age_newton_fail": age_newton_fail.copy(),
        "total_newton_failures": int(age_newton_fail.sum()),
        "worst_foc_resid": float(age_max_foc.max()) if age_max_foc.size else 0.0,
        "solver_config": solver_config,
        "disc_config": disc_config,
        "solve_control": solve_control,
        "solve_status": solve_status,
        "is_partial": is_partial,
        "solved_age_mask": solved_age_mask.copy(),
        "solved_age_indices": solved_idx.copy(),
        "youngest_solved_age": youngest_solved_age,
        "oldest_solved_age": oldest_solved_age,
        "n_ages_solved": int(solved_idx.size),
        "wall_time_sec": float(wall_time_sec),
        "checkpoint_save_count": int(checkpoint_save_count),
        "checkpoint_path": checkpoint_path,
    }

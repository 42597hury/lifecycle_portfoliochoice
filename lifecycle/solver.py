"""solver.py — JAX-pure lifecycle backward induction (EGM + 2D Newton).

Canonical model only:
  - Finite horizon, three assets (bills, stocks, nominal bonds).
  - CCV log-wealth dynamics (``r_p`` per Campbell-Viceira w8566 eq.10).
  - Shifted-bequest (luxury form, De Nardi 2004; Catherine 2025 normalisation).
  - Unconstrained portfolio (no simplex projection, no leverage caps).
  - Linear interpolation everywhere (no PCHIP).
  - ``n_state in {1, 2, 3, 4}`` post rtb-as-state migration. Realised
    log_R_bill ( = rtb_{t+1}) is read from the next-period state vector at
    ``model.rtb_index_in_state``; the return block is (xr, xb).

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


def _materialize_policy_lists(C_list, S_list, B_list, shape, solved_age_mask):
    """Stack a list[jnp.ndarray | None] into NumPy arrays of ``shape``.

    Unsolved ages become NaN. This is the single materialization point that
    moves device-resident policies back to host memory; all caller sites
    (final return, checkpoint write) go through here.
    """
    C_np = np.full(shape, np.nan, dtype=np.float64)
    S_np = np.full(shape, np.nan, dtype=np.float64)
    B_np = np.full(shape, np.nan, dtype=np.float64)
    for t in range(len(C_list)):
        if solved_age_mask[t] and C_list[t] is not None:
            C_np[t] = np.asarray(C_list[t])
            S_np[t] = np.asarray(S_list[t])
            B_np[t] = np.asarray(B_list[t])
    return C_np, S_np, B_np


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


def _bracket_axis(grid, val):
    """Bracket ``val`` in a sorted axis grid. Each axis must have >= 2 points."""
    n = grid.shape[0]
    lo = jnp.clip(jnp.searchsorted(grid, val, side="right") - 1, 0, n - 2)
    denom = grid[lo + 1] - grid[lo]
    frac = jnp.clip((val - grid[lo]) / denom, 0.0, 1.0)
    return lo, frac


def bracket_state_jax(s, axis_grids, shift, L_inv):
    """Transform ``s`` to bracket coords and bracket along k axes.

    Returns ``(lo, frac)`` arrays of shape ``(k,)`` where k = len(axis_grids).
    Generic over k in {1, 2, 3, 4}.
    """
    b = L_inv @ (s - shift)
    los = []
    fracs = []
    for d, grid in enumerate(axis_grids):
        lo_d, f_d = _bracket_axis(grid, b[d])
        los.append(lo_d)
        fracs.append(f_d)
    return jnp.stack(los), jnp.stack(fracs)


def _corner_offsets(k):
    """All 2^k binary offset vectors as a NumPy array of shape (2^k, k).

    Used to build static per-corner index/weight arithmetic. k is fixed at
    trace-build time (n_state of the current model), so this stays a constant.
    """
    n_corners = 1 << int(k)
    out = np.zeros((n_corners, k), dtype=np.int32)
    for c in range(n_corners):
        for d in range(k):
            out[c, d] = (c >> (k - 1 - d)) & 1
    return out


def _grid_strides(axis_sizes):
    """Row-major flat-index strides for a k-D grid with the given per-axis sizes.

    Last axis stride = 1; others are products of trailing sizes. The simulator
    and solver use the same row-major convention, so stride math here matches
    discretization.build_state_grid's flat-index ordering.
    """
    k = len(axis_sizes)
    strides = np.empty(k, dtype=np.int32)
    s = 1
    for d in range(k - 1, -1, -1):
        strides[d] = s
        s *= int(axis_sizes[d])
    return strides


def _cast_for_gather(arr, gather_dtype):
    """Cast ``arr`` to ``gather_dtype`` if not already that dtype.

    No-op when ``gather_dtype is jnp.float64`` (the default). Used to
    optionally lower the gather/interp boundary to fp32 — see
    ``SolverConfig.gather_precision``.
    """
    return arr if arr.dtype == gather_dtype else arr.astype(gather_dtype)


def _resolve_gather_dtype(sc):
    """Map ``SolverConfig.gather_precision`` string to a JAX dtype.

    Returns a Python dtype object (``jnp.float64``/``jnp.float32``) suitable
    for embedding in the kernel ``static`` tuple — bakes into the JIT trace
    cleanly without forcing per-call recompiles.
    """
    pref = getattr(sc, "gather_precision", "f64")
    pref = str(pref).strip().lower()
    if pref in ("f64", "float64", "fp64"):
        return jnp.float64
    if pref in ("f32", "float32", "fp32"):
        return jnp.float32
    raise ValueError(
        f"SolverConfig.gather_precision must be 'f64' or 'f32', got {pref!r}"
    )


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
    use_fori=True,
):
    """2D Newton on (alpha_s, alpha_b) with backtracking line search.

    ``foc_fn(a_s, a_b) -> (fs, fb, Jss, Jbb, Jsb, e)``. Returns
    ``(a_s, a_b, e, exit_code, err_norm, n_iter, n_backtrack_total)``,
    where ``n_backtrack_total`` is the sum across all Newton iters of the
    halvings the line search evaluated before finding an improving step.

    ``use_fori`` is a Python bool selected at JIT-trace time:
      - True  -> ``lax.fori_loop`` + mask (GPU-friendly, deterministic dispatch,
                 runs ``max_iter`` iters unconditionally).
      - False -> ``lax.while_loop`` with data-dependent exit (CPU-friendly,
                 real early termination).
    """
    if use_fori:
        return _newton_fori(
            foc_fn, init_a_s, init_a_b, scale,
            tol, max_iter, max_backtrack_iter,
            line_search_max_step, singular_det,
            grad_step_size, grad_denom_eps,
        )
    return _newton_while(
        foc_fn, init_a_s, init_a_b, scale,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
    )


def _newton_while(
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
    """While-loop Newton path. Real early termination via lax.while_loop —
    optimal on CPU (sequential vmap) and the historic baseline.
    """
    fs0, fb0, Jss0, Jbb0, Jsb0, e0 = foc_fn(init_a_s, init_a_b)
    err0 = jnp.sqrt(fs0 * fs0 + fb0 * fb0)

    init_state = (
        init_a_s, init_a_b,
        fs0, fb0, Jss0, Jbb0, Jsb0,
        e0, err0,
        jnp.int32(0),
        jnp.int32(0),                  # n_backtrack_total (summed across iters)
        err0 < tol * scale,
    )

    def cond_fn(state):
        *_, k, _n_bt_total, done = state
        return jnp.logical_and(jnp.logical_not(done), k < max_iter)

    def body_fn(state):
        a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, k, n_bt_total, _ = state

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
        (k_bt, _alpha, new_a_s, new_a_b, new_fs, new_fb,
         new_Jss, new_Jbb, new_Jsb, new_e, new_err, found_any) = bt_final

        new_done = jnp.logical_or(
            new_err < tol * scale,
            jnp.logical_not(found_any),
        )
        return (
            new_a_s, new_a_b, new_fs, new_fb,
            new_Jss, new_Jbb, new_Jsb,
            new_e, new_err, k + 1, n_bt_total + k_bt, new_done,
        )

    final = lax.while_loop(cond_fn, body_fn, init_state)
    a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, k, n_bt_total, _done = final
    exit_code = jnp.where(err < tol * scale, EC_INTERIOR, EC_NEWTON_FAIL)
    return a_s, a_b, e, exit_code, err / scale, k, n_bt_total


def _backtracking_fori(
    a_s, a_b, step_s, step_b,
    a_s_full, a_b_full,
    fs_f, fb_f, Jss_f, Jbb_f, Jsb_f, e_f, err_f, full_improves,
    fs_old, fb_old, Jss_old, Jbb_old, Jsb_old, e_old, err_old,
    foc_fn, max_backtrack_iter,
):
    """Mask-based backtracking line search. Runs ``max_backtrack_iter`` halvings
    unconditionally; once the first improving alpha is found, subsequent iters
    keep computing ``foc_fn`` but mask out the result.

    Returns the best-so-far state plus a ``found`` flag (False if no halving
    improved on the current err) and ``n_used`` — the number of halvings
    actually evaluated before finding an improvement (0 if the full step
    succeeded; capped at ``max_backtrack_iter`` if none improved).
    """
    # Best-so-far init: full-step result if it improved, else current state.
    init = (
        jnp.float64(1.0),                     # alpha (current step size)
        jnp.where(full_improves, a_s_full, a_s),
        jnp.where(full_improves, a_b_full, a_b),
        jnp.where(full_improves, fs_f, fs_old),
        jnp.where(full_improves, fb_f, fb_old),
        jnp.where(full_improves, Jss_f, Jss_old),
        jnp.where(full_improves, Jbb_f, Jbb_old),
        jnp.where(full_improves, Jsb_f, Jsb_old),
        jnp.where(full_improves, e_f, e_old),
        jnp.where(full_improves, err_f, err_old),
        full_improves,                         # found
        jnp.int32(0),                          # n_used (halvings evaluated before found)
    )

    def bt_body(_i, bt_state):
        alpha, a_s_b, a_b_b, fs_b, fb_b, Jss_b, Jbb_b, Jsb_b, e_b, err_b, found, n_used = bt_state
        new_alpha = alpha * 0.5
        a_s_t = a_s + new_alpha * step_s
        a_b_t = a_b + new_alpha * step_b
        fs_t, fb_t, Jss_t, Jbb_t, Jsb_t, e_t = foc_fn(a_s_t, a_b_t)
        err_t = jnp.sqrt(fs_t * fs_t + fb_t * fb_t)
        # Replicate while_loop semantics: only adopt the FIRST halving that
        # improves on err_old. Once found, hold state.
        improved_now = jnp.logical_and(jnp.logical_not(found), err_t < err_old)
        is_active = jnp.logical_not(found)
        return (
            jnp.where(found, alpha, new_alpha),
            jnp.where(improved_now, a_s_t, a_s_b),
            jnp.where(improved_now, a_b_t, a_b_b),
            jnp.where(improved_now, fs_t, fs_b),
            jnp.where(improved_now, fb_t, fb_b),
            jnp.where(improved_now, Jss_t, Jss_b),
            jnp.where(improved_now, Jbb_t, Jbb_b),
            jnp.where(improved_now, Jsb_t, Jsb_b),
            jnp.where(improved_now, e_t, e_b),
            jnp.where(improved_now, err_t, err_b),
            jnp.logical_or(found, improved_now),
            n_used + jnp.where(is_active, jnp.int32(1), jnp.int32(0)),
        )

    final = lax.fori_loop(0, max_backtrack_iter, bt_body, init)
    _alpha, a_s_b, a_b_b, fs_b, fb_b, Jss_b, Jbb_b, Jsb_b, e_b, err_b, found, n_used = final
    return a_s_b, a_b_b, fs_b, fb_b, Jss_b, Jbb_b, Jsb_b, e_b, err_b, found, n_used


def _newton_fori(
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
    """fori_loop + mask Newton path. Runs ``max_iter`` iters unconditionally;
    converged or line-search-failed cells run identical math but mask out the
    update. GPU-friendly (no warp divergence, deterministic dispatch).
    """
    fs0, fb0, Jss0, Jbb0, Jsb0, e0 = foc_fn(init_a_s, init_a_b)
    err0 = jnp.sqrt(fs0 * fs0 + fb0 * fb0)

    init_state = (
        init_a_s, init_a_b,
        fs0, fb0, Jss0, Jbb0, Jsb0,
        e0, err0,
        jnp.int32(0),                  # n_iters_used (active iters only)
        jnp.int32(0),                  # n_backtrack_total (sum across active iters)
        err0 < tol * scale,            # converged
        jnp.bool_(False),              # ls_failed
    )

    def fori_body(_i, state):
        a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, n_used, n_bt_total, converged, ls_failed = state
        is_active = jnp.logical_not(jnp.logical_or(converged, ls_failed))

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

        (new_a_s, new_a_b, new_fs, new_fb, new_Jss, new_Jbb, new_Jsb,
         new_e, new_err, found_any, n_bt_iter) = _backtracking_fori(
            a_s, a_b, step_s, step_b,
            a_s_full, a_b_full,
            fs_f, fb_f, Jss_f, Jbb_f, Jsb_f, e_f, err_f, full_improves,
            fs, fb, Jss, Jbb, Jsb, e, err,
            foc_fn, max_backtrack_iter,
        )

        new_converged = new_err < tol * scale
        new_ls_failed = jnp.logical_not(found_any)

        # Mask: hold all fields constant for cells already converged or failed.
        return (
            jnp.where(is_active, new_a_s, a_s),
            jnp.where(is_active, new_a_b, a_b),
            jnp.where(is_active, new_fs, fs),
            jnp.where(is_active, new_fb, fb),
            jnp.where(is_active, new_Jss, Jss),
            jnp.where(is_active, new_Jbb, Jbb),
            jnp.where(is_active, new_Jsb, Jsb),
            jnp.where(is_active, new_e, e),
            jnp.where(is_active, new_err, err),
            n_used + jnp.where(is_active, jnp.int32(1), jnp.int32(0)),
            n_bt_total + jnp.where(is_active, n_bt_iter, jnp.int32(0)),
            jnp.where(is_active, new_converged, converged),
            jnp.where(is_active, new_ls_failed, ls_failed),
        )

    final = lax.fori_loop(0, max_iter, fori_body, init_state)
    a_s, a_b, fs, fb, Jss, Jbb, Jsb, e, err, n_used, n_bt_total, converged, _ls_failed = final
    exit_code = jnp.where(converged, EC_INTERIOR, EC_NEWTON_FAIL)
    return a_s, a_b, e, exit_code, err / scale, n_used, n_bt_total


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

def _build_step_log_returns(state_grid_i, M_v_nodes, ret_nodes,
                             const_r, A_r, s_next, rtb_idx, xr_pos, xb_pos):
    """Per-i_s log-return scenario tensors of shape ``(n_state_quad, n_ret_quad)``.

    Post rtb-as-state migration:
      - ``log_R_bill`` is read from the next-period state vector at
        ``rtb_idx``: shape ``(n_state_quad,)``, broadcast to
        ``(n_state_quad, n_ret_quad)`` via ``[:, None]`` so the FOC kernel
        signatures stay unchanged.
      - ``mu_r`` covers only the return block (xr, xb); ``xr_pos`` and
        ``xb_pos`` index into ``ret_names``.

    ``s_next`` shape: ``(n_state_quad, n_state)`` — passed in (computed once
    per i_s in ``_build_step_state_brackets``).
    """
    base_mu_r = const_r + A_r @ state_grid_i              # (n_ret,)
    mu_r_per = base_mu_r[None, :] + M_v_nodes              # (n_state_quad, n_ret)
    mu_xs = mu_r_per[:, xr_pos]
    mu_xb = mu_r_per[:, xb_pos]
    res_xs = ret_nodes[:, xr_pos]
    res_xb = ret_nodes[:, xb_pos]
    # rtb is in the state vector — read its realisation at each k_v.
    log_R_bill_kv = s_next[:, rtb_idx]                     # (n_state_quad,)
    n_ret_quad = ret_nodes.shape[0]
    log_R_bill = jnp.broadcast_to(log_R_bill_kv[:, None],
                                   (log_R_bill_kv.shape[0], n_ret_quad))
    log_x_s = mu_xs[:, None] + res_xs[None, :]
    log_x_b = mu_xb[:, None] + res_xb[None, :]
    return log_R_bill, log_x_s, log_x_b


def _build_step_state_brackets(state_grid_i, Phi_0_state, Phi_11, v_nodes,
                                axis_grids, axis_sizes, corner_offsets,
                                strides, shift, L_inv):
    """Per-i_s state bracketing. Returns ``(s_next, j_corners, w_corners)``.

    - ``s_next``: (n_state_quad, n_state)
    - ``j_corners``: (n_state_quad, 2^n_state) flat indices into joint state grid
    - ``w_corners``: (n_state_quad, 2^n_state) multilinear weights

    Generic over n_state via the static ``corner_offsets`` table (shape
    (2^k, k)) and the static row-major ``strides`` (shape (k,)) — both are
    NumPy arrays computed once at trace-build time from the precompute layout.
    """
    s_next = Phi_0_state[None, :] + state_grid_i @ Phi_11.T + v_nodes

    corner_offsets_j = jnp.asarray(corner_offsets)
    strides_j = jnp.asarray(strides)

    def per_kv(s_next_kv):
        lo, frac = bracket_state_jax(s_next_kv, axis_grids, shift, L_inv)
        # Multilinear weights: w[c] = prod_d (frac[d] if offset[c,d] else 1-frac[d])
        # corner_offsets shape (2^k, k); broadcast frac to match
        per_axis = jnp.where(corner_offsets_j > 0, frac[None, :], 1.0 - frac[None, :])
        w = jnp.prod(per_axis, axis=1)                    # (2^k,)
        # Flat indices: j[c] = sum_d (lo[d] + offset[c,d]) * stride[d]
        idx_per_axis = lo[None, :] + corner_offsets_j     # (2^k, k)
        j = jnp.sum(idx_per_axis * strides_j[None, :], axis=1).astype(jnp.int32)
        return j, w

    j_corners, w_corners = vmap(per_kv)(s_next)
    return s_next, j_corners, w_corners


def _interp_c_and_mpc_at_cell(c_corners_kv, w_corners_kv,
                               iz_lo, frac_z, x_next_scalar, wealth_grid,
                               min_consumption,
                               gather_dtype=jnp.float64):
    """Multilinear-state × bilinear-z × linear-wealth interp of c_next and
    the marginal propensity to consume out of wealth (slope dc/dx).

    Caller pre-gathers c_corners_kv = c_next[:, j_corners[k_v], :] once per
    cell, so the inner FOC reads from c_corners_kv with scalar (iz, iw)
    indices and a single full slice on the corner axis — this lowers to
    dynamic_slice instead of an advanced gather (verified via jaxpr).

    When ``gather_dtype is jnp.float32`` (mixed-precision mode), the gather
    + bracket + multilinear interp run in fp32; ``c`` and ``mpc`` cast back
    to fp64 BEFORE the ``min_consumption`` floor and the ``[0,1]`` clip so
    callers see fp64 outputs identical in dtype to the all-fp64 path.
    """
    # Cast bracket inputs to the gather precision. Indices iz_lo/iz_hi stay
    # int (untouched by the cast). frac_z is a [0,1]-bounded weight.
    c_corners_g = _cast_for_gather(c_corners_kv, gather_dtype)
    w_corners_g = _cast_for_gather(w_corners_kv, gather_dtype)
    x_next_g = _cast_for_gather(jnp.asarray(x_next_scalar), gather_dtype)
    wealth_grid_g = _cast_for_gather(wealth_grid, gather_dtype)
    frac_z_g = _cast_for_gather(jnp.asarray(frac_z), gather_dtype)

    n_w = wealth_grid_g.shape[0]
    iw = jnp.clip(jnp.searchsorted(wealth_grid_g, x_next_g, side="right") - 1, 0, n_w - 2)
    iw_hi = iw + 1
    iz_hi = iz_lo + 1
    x0 = wealth_grid_g[iw]
    x1 = wealth_grid_g[iw_hi]
    inv_dw = jnp.asarray(1.0, dtype=gather_dtype) / (x1 - x0)
    fw = (x_next_g - x0) * inv_dw

    # c_corners_g: (n_z, n_corners, n_w) where n_corners = 2**n_state. Scalar
    # (iz_*, iw_*) on axes 0 and 2 with a full slice on axis 1 lowers to
    # dynamic_slice — no advanced indexing in the Newton hot loop.
    c_zlo_w0 = c_corners_g[iz_lo, :, iw]
    c_zlo_w1 = c_corners_g[iz_lo, :, iw_hi]
    c_zhi_w0 = c_corners_g[iz_hi, :, iw]
    c_zhi_w1 = c_corners_g[iz_hi, :, iw_hi]

    # Linear in wealth at each (z, state corner). Shape: (n_corners,).
    c_zlo = c_zlo_w0 + (c_zlo_w1 - c_zlo_w0) * fw
    c_zhi = c_zhi_w0 + (c_zhi_w1 - c_zhi_w0) * fw
    slope_zlo = (c_zlo_w1 - c_zlo_w0) * inv_dw
    slope_zhi = (c_zhi_w1 - c_zhi_w0) * inv_dw

    # Linear in z at each state corner. Shape: (n_corners,).
    one = jnp.asarray(1.0, dtype=gather_dtype)
    c_per_corner = (one - frac_z_g) * c_zlo + frac_z_g * c_zhi
    slope_per_corner = (one - frac_z_g) * slope_zlo + frac_z_g * slope_zhi

    # Multilinear in state. Scalars.
    c_g = jnp.sum(w_corners_g * c_per_corner)
    mpc_g = jnp.sum(w_corners_g * slope_per_corner)

    # Cast back to fp64 BEFORE the floor / clip. Downstream FOC arithmetic
    # (CRRA, Newton residual) consumes c / mpc in fp64.
    c = c_g.astype(jnp.float64)
    mpc = mpc_g.astype(jnp.float64)

    c = jnp.maximum(c, min_consumption)
    mpc = jnp.clip(mpc, 0.0, 1.0)
    return c, mpc


# -----------------------------------------------------------------------------
# Retirement-age FOC (CCV)
# -----------------------------------------------------------------------------

def retirement_foc_jac_ccv(
    alpha_s, alpha_b, s_val, psi_z,
    log_R_bill, log_x_s, log_x_b, weight_kv_kr,   # (n_state_quad, n_ret_quad)
    w_corners,                                     # (n_state_quad, n_corners)
    c_corners_at_z, wealth_grid,                   # (n_state_quad, n_corners, n_w), (n_w,)
    pension_next_z,                                # scalar at z_idx
    A_is,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    gamma, b_bar, delta, min_consumption,
    gather_dtype=jnp.float64,
):
    """Retirement FOC sums over ``(k_v, k_r)``: bequest + alive contributions.

    ``z`` is frozen at retirement (no eta/eps), so income_next = pension(z)
    is a scalar and the caller pre-slices c_next at z_idx → c_corners_at_z
    has shape (n_state_quad, n_corners, n_w) where n_corners = 2**n_state,
    and the inner gather is a dynamic_slice.

    When ``gather_dtype is jnp.float32`` the inline ``per_kv_kr`` bilinear
    interp runs in fp32; ``c`` and ``mpc`` cast back to fp64 before the
    ``min_consumption`` floor / ``[0,1]`` clip. The FOC sum (mu_bq + mu_alive
    aggregates over k_v, k_r) stays fp64 throughout.
    """
    R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad(
        alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
        sigma2_xr, sigma2_xb, sigma_xrxb,
    )                                              # (n_state_quad, n_ret_quad)
    sR_p = s_val * R_p
    x_next = sR_p + pension_next_z                 # (n_state_quad, n_ret_quad)

    mu_bq, mup_bq = bequest_mu_and_mup(sR_p, A_is, gamma, b_bar, delta)

    # Cast wealth_grid + corner inputs to gather precision once for the inline
    # interp. fp64 default → no-op.
    wealth_grid_g = _cast_for_gather(wealth_grid, gather_dtype)
    c_corners_at_z_g = _cast_for_gather(c_corners_at_z, gather_dtype)
    w_corners_g = _cast_for_gather(w_corners, gather_dtype)

    n_w = wealth_grid_g.shape[0]
    def per_kv_kr(x_scalar, c_kv, w_kv):
        # c_kv: (n_corners, n_w), w_kv: (n_corners,) where n_corners = 2**n_state.
        # Scalar iw → dynamic_slice on axis 1.
        x_scalar_g = _cast_for_gather(jnp.asarray(x_scalar), gather_dtype)
        iw = jnp.clip(jnp.searchsorted(wealth_grid_g, x_scalar_g, side="right") - 1, 0, n_w - 2)
        iw_hi = iw + 1
        x0 = wealth_grid_g[iw]
        x1 = wealth_grid_g[iw_hi]
        inv_dw = jnp.asarray(1.0, dtype=gather_dtype) / (x1 - x0)
        fw = (x_scalar_g - x0) * inv_dw

        c_w0 = c_kv[:, iw]                  # (n_corners,)
        c_w1 = c_kv[:, iw_hi]
        c_per_corner = c_w0 + (c_w1 - c_w0) * fw
        slope_per_corner = (c_w1 - c_w0) * inv_dw

        c_g = jnp.sum(w_kv * c_per_corner)
        mpc_g = jnp.sum(w_kv * slope_per_corner)

        # Cast back to fp64 BEFORE floor / clip. FOC arithmetic below is fp64.
        c = c_g.astype(jnp.float64)
        mpc = mpc_g.astype(jnp.float64)
        c = jnp.maximum(c, min_consumption)
        mpc = jnp.clip(mpc, 0.0, 1.0)
        return c, mpc

    # Vmap over k_v (c_kv, w_kv vary), then over k_r (x_scalar varies).
    # Shape: (n_state_quad, n_ret_quad). Pass the gather-precision-cast tensors
    # so the inner per_kv_kr receives c_kv / w_kv at gather_dtype.
    c_at_xn, mpc_at_xn = vmap(
        lambda c_kv, w_kv, x_row: vmap(per_kv_kr, in_axes=(0, None, None))(
            x_row, c_kv, w_kv
        ),
        in_axes=(0, 0, 0),
    )(c_corners_at_z_g, w_corners_g, x_next)

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
    alpha_s, alpha_b, s_val, psi_z,
    log_R_bill, log_x_s, log_x_b, weight_kv_kr,    # (n_state_quad, n_ret_quad)
    w_corners,                                      # (n_state_quad, n_corners)
    c_corners_T, wealth_grid,                       # (n_state_quad, n_z, n_corners, n_w), (n_w,)
    income_next_table,                              # (n_eta, n_eps) at this z_idx, t_idx
    eta_iz_lo, eta_frac_z,                          # (n_eta,) z-bracket per eta
    eta_weights, eps_weights,                       # (n_eta,), (n_eps,)
    A_is,
    sigma2_xr, sigma2_xb, sigma_xrxb,
    gamma, b_bar, delta, min_consumption,
    gather_dtype=jnp.float64,
):
    """Working-age FOC. Bequest summed over ``(k_v, k_r)``; alive summed over
    ``(k_v, k_r, k_eta, i_e)``. Caller supplies the income_next gather table
    and pre-gathers c_next at multilinear-state corners → c_corners_T has k_v
    leading so the outer vmap consumes its leading axis. ``n_corners`` =
    ``2**n_state``.

    When ``gather_dtype is jnp.float32`` the c_corners gather + multilinear
    interpolation runs in fp32; ``c_at_xn`` and ``mpc_at_xn`` returned from
    ``_interp_c_and_mpc_at_cell`` are already cast back to fp64, so the FOC
    sum (CRRA, bequest combine, residual, Jacobian) stays fp64 throughout.
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

    # c_next interp: multilinear-state × bilinear-z × linear-wealth at every
    # (k_v, k_r, k_eta, i_e). c_corners_T is pre-gathered at j_corners[k_v] so
    # the inner reads use scalar (iz, iw) → dynamic_slice (no advanced gather).
    # Innermost vmap: over i_e (n_eps); x_scalar varies.
    # Next vmap: over k_eta (n_eta); iz, fz vary, x_row varies along eps axis.
    # Next vmap: over k_r (n_ret_quad); x_block varies along (eta, eps).
    # Outer vmap: over k_v (n_state_quad); c_kv, w_kv, x_block vary.
    def per_kv(c_kv, w_kv, x_kv):
        # c_kv: (n_z, n_corners, n_w); w_kv: (n_corners,); x_kv: (n_ret_quad, n_eta, n_eps)
        def at_eta_eps(x_scalar, iz, fz):
            return _interp_c_and_mpc_at_cell(
                c_kv, w_kv, iz, fz, x_scalar, wealth_grid, min_consumption,
                gather_dtype=gather_dtype,
            )

        def per_kr(x_kr):
            # x_kr: (n_eta, n_eps)
            def per_keta(x_row, iz, fz):
                # x_row: (n_eps,)
                return vmap(at_eta_eps, in_axes=(0, None, None))(x_row, iz, fz)
            return vmap(per_keta, in_axes=(0, 0, 0))(x_kr, eta_iz_lo, eta_frac_z)
        return vmap(per_kr)(x_kv)

    c_at_xn, mpc_at_xn = vmap(per_kv)(c_corners_T, w_corners, x_next)
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
    use_fori,
):
    """Solve the EGM portfolio FOC at every savings point IN PARALLEL.

    Each savings point's Newton starts from the canonical cold init
    ``(init_a_s, init_a_b)`` instead of warming from the previous savings
    point's converged solution. This breaks the sequential dependency that
    forced a ``lax.scan`` loop, letting JAX/XLA vmap the entire savings
    sweep into a single batched kernel call.

    The trade-off: cold-start Newton typically needs more iterations to
    converge (the warm-start was buying ~5-15 iters/cell). The robustness
    cost is that cells where the optimum is far from the cold init may
    fail to converge if ``max_iter`` is too tight — bump max_iter and
    track ``newton_failures`` post-hoc.

    Returns ``(x_egm, c_egm, a_s_egm, a_b_egm, n_iters_egm, n_backtrack_egm)``
    each shape ``(n_savings + 1,)``, with the first entry being the egm_anchor
    at s = 0 (iter counts padded with 0 there).
    """
    def per_savings_point(s_val):
        foc_fn = foc_factory(s_val)
        _, _, _, _, _, e0 = foc_fn(0.0, 0.0)
        scale = jnp.maximum(jnp.abs(e0), 1e-30)

        (a_s_opt, a_b_opt, V_dot, _exit_code, _err,
         n_iter_used, n_bt_total) = newton_2d_with_line_search(
            foc_fn, init_a_s, init_a_b, scale,
            tol, max_iter, max_backtrack_iter,
            line_search_max_step, singular_det,
            grad_step_size, grad_denom_eps,
            use_fori=use_fori,
        )
        beta_e = jnp.maximum(beta * V_dot, euler_inv_floor)
        c_opt = jnp.maximum(beta_e ** (-1.0 / gamma), min_consumption)

        tiny = s_val <= tiny_savings
        c_out = jnp.where(tiny, min_consumption, c_opt)
        a_s_out = jnp.where(tiny, init_a_s, a_s_opt)
        a_b_out = jnp.where(tiny, init_a_b, a_b_opt)
        x_out = c_out + s_val
        return x_out, c_out, a_s_out, a_b_out, n_iter_used, n_bt_total

    x_arr, c_arr, a_s_arr, a_b_arr, ni_arr, nb_arr = vmap(per_savings_point)(s_grid)

    x_egm = jnp.concatenate([jnp.array([egm_anchor], dtype=x_arr.dtype), x_arr])
    c_egm = jnp.concatenate([jnp.array([egm_anchor], dtype=c_arr.dtype), c_arr])
    a_s_egm = jnp.concatenate([jnp.array([0.0], dtype=a_s_arr.dtype), a_s_arr])
    a_b_egm = jnp.concatenate([jnp.array([0.0], dtype=a_b_arr.dtype), a_b_arr])
    n_iters_egm = jnp.concatenate([jnp.array([0], dtype=ni_arr.dtype), ni_arr])
    n_backtrack_egm = jnp.concatenate([jnp.array([0], dtype=nb_arr.dtype), nb_arr])
    return x_egm, c_egm, a_s_egm, a_b_egm, n_iters_egm, n_backtrack_egm


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
    use_fori,
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

    (x_egm, c_egm, a_s_egm, a_b_egm,
     n_iters_egm, n_backtrack_egm) = _egm_scan_cell(
        foc_factory, s_grid, init_a_s, init_a_b,
        gamma, beta,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
        tiny_savings, euler_inv_floor, min_consumption, egm_anchor,
        use_fori,
    )
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)
    # Drop the s=0 anchor (padded with 0 iters / 0 backtracks; no Newton solve
    # happens there) so the histogram reflects only real Newton calls.
    n_iters_per_s = n_iters_egm[1:]
    n_backtrack_per_s = n_backtrack_egm[1:]
    return c_w, a_s_w, a_b_w, n_iters_per_s, n_backtrack_per_s


def _solve_retirement_at_cell(
    z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
    log_R_bill_i, log_x_s_i, log_x_b_i,    # (n_state_quad, n_ret_quad) — pre-gathered for this i_s
    j_corners_i, w_corners_i,               # (n_state_quad, n_corners) — pre-gathered for this i_s
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
    use_fori,
    gather_dtype,
):
    A_is = A_per_state[i_s]
    psi_z = psi_per_z[z_idx]
    pension_next_z = pension_next_by_z[z_idx]

    # Pre-gather c_next at multilinear-state corners for this (z_idx, i_s).
    # Hoisted out of the FOC so the Newton inner loop sees only dynamic_slice
    # reads instead of a 4-axis advanced gather. j_corners_i: (n_state_quad, n_corners).
    # Cast to gather_dtype here lets XLA fuse the gather + dtype convert into a
    # single load (verify in HLO at gate 3). Storage of c_next stays fp64.
    c_corners_at_z = c_next[z_idx, j_corners_i, :]   # (n_state_quad, n_corners, n_w)
    c_corners_at_z = _cast_for_gather(c_corners_at_z, gather_dtype)

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return retirement_foc_jac_ccv(
                a_s, a_b, s_val, psi_z,
                log_R_bill_i, log_x_s_i, log_x_b_i, weight_kv_kr,
                w_corners_i,
                c_corners_at_z, wealth_grid,
                pension_next_z, A_is,
                sigma2_xr, sigma2_xb, sigma_xrxb,
                gamma, b_bar, delta, min_consumption,
                gather_dtype=gather_dtype,
            )
        return foc_fn

    (x_egm, c_egm, a_s_egm, a_b_egm,
     n_iters_egm, n_backtrack_egm) = _egm_scan_cell(
        foc_factory, s_grid, init_a_s, init_a_b,
        gamma, beta,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
        tiny_savings, euler_inv_floor, min_consumption, egm_anchor,
        use_fori,
    )
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)
    # Drop the s=0 anchor (padded with 0 iters / 0 backtracks; no Newton solve
    # happens there) so the histogram reflects only real Newton calls.
    n_iters_per_s = n_iters_egm[1:]
    n_backtrack_per_s = n_backtrack_egm[1:]
    return c_w, a_s_w, a_b_w, n_iters_per_s, n_backtrack_per_s


def _solve_working_at_cell(
    z_idx, i_s, c_next,
    income_next_at_z,                  # (n_eta, n_eps) at this z_idx
    eta_iz_lo, eta_frac_z,             # (n_eta,) bracket at z_next[k_eta]
    eta_weights, eps_weights,
    psi_per_z,
    log_R_bill_i, log_x_s_i, log_x_b_i,    # (n_state_quad, n_ret_quad) — pre-gathered for this i_s
    j_corners_i, w_corners_i,               # (n_state_quad, n_corners) — pre-gathered for this i_s
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
    use_fori,
    gather_dtype,
):
    A_is = A_per_state[i_s]
    psi_z = psi_per_z[z_idx]

    # Pre-gather c_next at multilinear-state corners for this i_s, transposed
    # so k_v leads (matches the outer vmap in working_foc_jac_ccv). The Newton
    # inner loop reads via dynamic_slice instead of a 4-axis advanced gather.
    # Memory: per-cell (n_z, n_state_quad, n_corners, n_w) where n_corners =
    # 2**n_state; under vmap this batches to (n_cells, ...). At n_state=3 the
    # per-cell footprint is ~3MB; at n_state=4 it's ~6MB before fusion — verify
    # peak HBM on the first 4-D GPU run before running canonical.
    # Cast to gather_dtype lets XLA fuse the gather + transpose + dtype convert
    # into a single load chain (verify in HLO at gate 3). c_next storage stays fp64.
    c_corners = c_next[:, j_corners_i, :]                 # (n_z, n_state_quad, n_corners, n_w)
    c_corners_T = jnp.transpose(c_corners, (1, 0, 2, 3))  # (n_state_quad, n_z, n_corners, n_w)
    c_corners_T = _cast_for_gather(c_corners_T, gather_dtype)

    def foc_factory(s_val):
        def foc_fn(a_s, a_b):
            return working_foc_jac_ccv(
                a_s, a_b, s_val, psi_z,
                log_R_bill_i, log_x_s_i, log_x_b_i, weight_kv_kr,
                w_corners_i,
                c_corners_T, wealth_grid,
                income_next_at_z,
                eta_iz_lo, eta_frac_z,
                eta_weights, eps_weights,
                A_is,
                sigma2_xr, sigma2_xb, sigma_xrxb,
                gamma, b_bar, delta, min_consumption,
                gather_dtype=gather_dtype,
            )
        return foc_fn

    (x_egm, c_egm, a_s_egm, a_b_egm,
     n_iters_egm, n_backtrack_egm) = _egm_scan_cell(
        foc_factory, s_grid, init_a_s, init_a_b,
        gamma, beta,
        tol, max_iter, max_backtrack_iter,
        line_search_max_step, singular_det,
        grad_step_size, grad_denom_eps,
        tiny_savings, euler_inv_floor, min_consumption, egm_anchor,
        use_fori,
    )
    c_w, a_s_w, a_b_w = _lift_to_wealth_grid(x_egm, c_egm, a_s_egm, a_b_egm, wealth_grid)
    # Drop the s=0 anchor (padded with 0 iters / 0 backtracks; no Newton solve
    # happens there) so the histogram reflects only real Newton calls.
    n_iters_per_s = n_iters_egm[1:]
    n_backtrack_per_s = n_backtrack_egm[1:]
    return c_w, a_s_w, a_b_w, n_iters_per_s, n_backtrack_per_s


def _all_is_log_returns_numpy(pcj):
    """Build (log_R_bill, log_x_s, log_x_b) for all i_s at once on host (NumPy).

    Used by the terminal-age kernel builders that pad/reshape on host before
    shipping to devices. Mirrors `_build_step_log_returns` but vectorised over
    all source states.

    Post rtb-as-state: log_R_bill[i_s, k_v, :] = state_next[i_s, k_v, rtb_idx]
    broadcast across the k_r axis (shape (N_state, n_state_quad, n_ret_quad)).
    """
    state_grid_np = np.asarray(pcj.state_grid, dtype=float)
    const_r_np = np.asarray(pcj.const_r, dtype=float)
    A_r_np = np.asarray(pcj.A_r, dtype=float)
    M_v_nodes_np = np.asarray(pcj.M_v_nodes, dtype=float)
    ret_nodes_np = np.asarray(pcj.ret_nodes, dtype=float)
    Phi_0_np = np.asarray(pcj.Phi_0_state, dtype=float)
    Phi_11_np = np.asarray(pcj.Phi_11, dtype=float)
    v_nodes_np = np.asarray(pcj.v_nodes, dtype=float)

    base_mu_r = const_r_np[None, :] + state_grid_np @ A_r_np.T   # (N_state, n_ret)
    mu_r_per = base_mu_r[:, None, :] + M_v_nodes_np[None, :, :]  # (N_state, n_state_quad, n_ret)

    # s_next[i_s, k_v, :] = Phi_0_state + Phi_11 @ s_t + v[k_v]
    s_next = (Phi_0_np[None, None, :]
              + state_grid_np[:, None, :] @ Phi_11_np.T[None, :, :]
              + v_nodes_np[None, :, :])
    log_R_bill_kv = s_next[:, :, pcj.rtb_idx]                    # (N_state, n_state_quad)

    n_ret_quad = ret_nodes_np.shape[0]
    log_R_bill = np.broadcast_to(log_R_bill_kv[:, :, None],
                                  log_R_bill_kv.shape + (n_ret_quad,))

    res_xs = ret_nodes_np[:, pcj.xr_pos]                         # (n_ret_quad,)
    res_xb = ret_nodes_np[:, pcj.xb_pos]
    log_x_s = mu_r_per[:, :, pcj.xr_pos:pcj.xr_pos + 1] + res_xs[None, None, :]
    log_x_b = mu_r_per[:, :, pcj.xb_pos:pcj.xb_pos + 1] + res_xb[None, None, :]
    log_x_s = log_x_s.reshape(log_x_s.shape[0], log_x_s.shape[1], -1)
    log_x_b = log_x_b.reshape(log_x_b.shape[0], log_x_b.shape[1], -1)

    # ascontiguousarray so downstream pad0/reshape don't trip on broadcast strides
    return np.ascontiguousarray(log_R_bill), log_x_s, log_x_b


def _precompute_per_is_tensors(pcj):
    """Pre-build per-i_s log-return scenario tensors and multilinear-state corners.

    These depend only on ``pc.state_grid[i_s]`` (independent of z), so computing
    them once per i_s instead of once per (z, i_s) cell saves n_z-fold redundant
    work in the kernel builder.

    Returns:
      log_R_bill_all : (N_state, n_state_quad, n_ret_quad)
      log_x_s_all    : (N_state, n_state_quad, n_ret_quad)
      log_x_b_all    : (N_state, n_state_quad, n_ret_quad)
      j_corners_all  : (N_state, n_state_quad, 2^n_state) int
      w_corners_all  : (N_state, n_state_quad, 2^n_state) float
    """
    def per_is_state_and_returns(s_i):
        s_next, j_corners, w_corners = _build_step_state_brackets(
            s_i, pcj.Phi_0_state, pcj.Phi_11, pcj.v_nodes,
            pcj.axis_grids, pcj.axis_sizes,
            pcj.corner_offsets, pcj.strides,
            pcj.state_bracket_shift, pcj.state_bracket_L_inv,
        )
        log_R_bill, log_x_s, log_x_b = _build_step_log_returns(
            s_i, pcj.M_v_nodes, pcj.ret_nodes,
            pcj.const_r, pcj.A_r, s_next,
            pcj.rtb_idx, pcj.xr_pos, pcj.xb_pos,
        )
        return log_R_bill, log_x_s, log_x_b, j_corners, w_corners

    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = (
        vmap(per_is_state_and_returns)(pcj.state_grid)
    )
    return (log_R_bill_all, log_x_s_all, log_x_b_all,
            j_corners_all, w_corners_all)


# =============================================================================
# Per-age JIT kernels (vmap over cells; pmap over devices iff n_dev > 1)
# =============================================================================
#
# Each builder dispatches at trace time on ``n_dev = len(jax.devices())``:
#   - n_dev > 1 (multi-core CPU with virtual XLA devices, multi-GPU host):
#     pmap shards cells across devices, with an inner vmap per shard. Real
#     parallelism on CPU; suboptimal but correct on multi-GPU.
#   - n_dev == 1 (single-GPU, or CPU with virtual devices disabled): pure
#     vmap over flat cell indices. Drops the pmap padding/reshape/collapse
#     so XLA-CUDA can fuse the per-age solve into a single kernel.
#
# The two paths produce identical outputs (modulo the float reorder XLA may
# do under fusion). See docs/handoff/HANDOFF_PMAP_TO_VMAP.md.

# A pytree of jnp arrays + static scalars, threaded through the per-age kernels.
# Static scalars are bound by closure at kernel-build time (the orchestrator
# builds the kernel once per solve so the arrays are baked into the trace).

PCJax = namedtuple("PCJax", [
    "wealth_grid", "s_grid", "z_grid", "dz",
    "state_grid",
    # axis_grids is a tuple of jnp arrays (length n_state); axis_sizes a tuple
    # of Python ints (static, used to compute corner offsets and strides at
    # trace-build time).
    "axis_grids", "axis_sizes", "corner_offsets", "strides",
    "state_bracket_shift", "state_bracket_L_inv",
    "v_nodes", "v_weights", "M_v_nodes",
    "const_r", "A_r", "Phi_0_state", "Phi_11",
    "ret_nodes", "ret_weights",
    "eta_nodes", "eta_weights", "eps_weights",
    "annuity_factors",
    "sigma2_xr", "sigma2_xb", "sigma_xrxb",
    "weight_kv_kr",
    # rtb-as-state metadata (static ints; used by _build_step_log_returns)
    "rtb_idx", "xr_pos", "xb_pos",
])

ModelParams = namedtuple("ModelParams", [
    "gamma", "beta", "b_bar", "delta", "rho",
])


def _pc_to_jnp(pc, delta):
    """Pack the precompute arrays the JAX kernels need into a PCJax pytree."""
    grids = list(pc.state_bracket_grids)
    n_state = len(grids)
    if n_state < 1 or n_state > 4:
        raise NotImplementedError(
            f"JAX solver supports n_state in {{1, 2, 3, 4}} (got {n_state})."
        )
    axis_grids = tuple(jnp.asarray(g) for g in grids)
    axis_sizes = tuple(int(g.shape[0]) for g in grids)
    corner_offsets = _corner_offsets(n_state)            # (2^k, k) int32
    strides = _grid_strides(axis_sizes)                  # (k,) int32

    weight_kv_kr = jnp.asarray(pc.v_weights)[:, None] * jnp.asarray(pc.ret_weights)[None, :]

    ret_names = pc.model.ret_names
    xr_pos = ret_names.index("xr")
    xb_pos = ret_names.index("xb")

    return PCJax(
        wealth_grid=jnp.asarray(pc.wealth_grid),
        s_grid=jnp.asarray(pc.s_grid),
        z_grid=jnp.asarray(pc.z_grid),
        dz=jnp.asarray(pc.dz),
        state_grid=jnp.asarray(pc.state_grid),
        axis_grids=axis_grids,
        axis_sizes=axis_sizes,
        corner_offsets=corner_offsets,
        strides=strides,
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
        rtb_idx=int(pc.model.rtb_index_in_state),
        xr_pos=int(xr_pos),
        xb_pos=int(xb_pos),
    )


def _build_per_age_terminal_kernel(pcj, mp, sc, n_dev):
    """Dispatch on n_dev: vmap-only on single-device hosts, pmap+vmap on multi-device."""
    if n_dev == 1:
        return _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc)
    return _build_per_age_terminal_kernel_pmap(pcj, mp, sc, n_dev)


def _build_per_age_terminal_kernel_pmap(pcj, mp, sc, n_dev):
    """Build a pmap'd terminal kernel that returns ``(N_state_padded, n_w)``
    arrays. Padding is invisible at the call site — caller slices to
    ``[:N_state]`` after gather.
    """
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton))

    log_R_bill, log_x_s, log_x_b = _all_is_log_returns_numpy(pcj)
    state_grid_np = np.asarray(pcj.state_grid)
    N_state = state_grid_np.shape[0]
    n_chunks = int(sc.cell_vmap_chunks)
    if n_chunks == 1:
        pad_n = math.ceil(N_state / n_dev) * n_dev
    else:
        raw_chunk_size = (N_state + n_chunks - 1) // n_chunks
        chunk_size = ((raw_chunk_size + n_dev - 1) // n_dev) * n_dev
        per_dev_chunk = chunk_size // n_dev
        pad_n = chunk_size * n_chunks

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

    if n_chunks == 1:
        per_dev = pad_n // n_dev

        def reshape_for_pmap(arr):
            return arr.reshape((n_dev, per_dev) + arr.shape[1:])

        log_R_bill_pm = jnp.asarray(reshape_for_pmap(log_R_bill_p))
        log_x_s_pm = jnp.asarray(reshape_for_pmap(log_x_s_p))
        log_x_b_pm = jnp.asarray(reshape_for_pmap(log_x_b_p))
        ann_pm = jnp.asarray(reshape_for_pmap(ann_p))
    else:
        log_R_bill_flat = jnp.asarray(log_R_bill_p)
        log_x_s_flat = jnp.asarray(log_x_s_p)
        log_x_b_flat = jnp.asarray(log_x_b_p)
        ann_flat = jnp.asarray(ann_p)

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
        if n_chunks != 1:
            chunk_results = []
            for i in range(n_chunks):
                start = i * chunk_size

                def chunk_for_pmap(arr):
                    return jnp.reshape(
                        arr[start:start + chunk_size],
                        (n_dev, per_dev_chunk) + arr.shape[1:],
                    )

                out = per_dev_solve(
                    chunk_for_pmap(log_R_bill_flat),
                    chunk_for_pmap(log_x_s_flat),
                    chunk_for_pmap(log_x_b_flat),
                    chunk_for_pmap(ann_flat),
                )
                out[0].block_until_ready()
                chunk_results.append(out)

            def collapse_chunk(k):
                flat_chunks = [
                    jnp.reshape(r[k], (chunk_size,) + r[k].shape[2:])
                    for r in chunk_results
                ]
                return jnp.concatenate(flat_chunks, axis=0)[:N_state]

            return tuple(collapse_chunk(k) for k in range(len(chunk_results[0])))

        (c_pm, as_pm, ab_pm,
         ni_pm, nb_pm) = per_dev_solve(log_R_bill_pm, log_x_s_pm, log_x_b_pm, ann_pm)

        # Stay on device — caller threads jnp arrays through to the next kernel.
        def collapse(a):
            arr = jnp.reshape(a, (pad_n,) + a.shape[2:])
            return arr[:N_state]

        return (
            collapse(c_pm), collapse(as_pm), collapse(ab_pm),
            collapse(ni_pm), collapse(nb_pm),
        )

    return call


def _build_chunked_index_arrays(n_z, N_state, n_chunks):
    """Build padded (z_idx, is_idx) arrays for cell-axis chunking.

    Padding is done in NumPy at builder time and converted to jnp once,
    matching the pmap path's strategy. Done outside ``@jit`` so the padded
    arrays are jnp-array constants by the time the kernel traces — no
    in-trace concatenate/full ops, no constant-folding ambiguity.

    Returns
    -------
    z_idx_padded, is_idx_padded : jnp.ndarray, shape ``(chunk_size * n_chunks,)``
    n_cells : int
        Real cell count = ``n_z * N_state`` (used to slice off padding).
    chunk_size : int
        Per-chunk fixed cell count = ``ceil(n_cells / n_chunks)``.
    """
    n_cells = n_z * N_state
    chunk_size = (n_cells + n_chunks - 1) // n_chunks
    n_cells_padded = chunk_size * n_chunks

    cell_idx = np.arange(n_cells, dtype=np.int64)
    z_idx_np = (cell_idx // N_state).astype(np.int64)
    is_idx_np = (cell_idx % N_state).astype(np.int64)

    pad_count = n_cells_padded - n_cells
    if pad_count > 0:
        z_idx_np = np.concatenate(
            [z_idx_np, np.full(pad_count, z_idx_np[-1], dtype=np.int64)]
        )
        is_idx_np = np.concatenate(
            [is_idx_np, np.full(pad_count, is_idx_np[-1], dtype=np.int64)]
        )

    return jnp.asarray(z_idx_np), jnp.asarray(is_idx_np), n_cells, chunk_size


def _build_chunked_pmap_index_arrays(n_z, N_state, n_chunks, n_dev):
    """Build padded pmap-ready cell-index chunks.

    Padding is appended only at the end of the flat cell axis, preserving the
    existing pmap convention: duplicate the final real cell and trim it away
    after solving. Each chunk is rounded to a multiple of ``n_dev`` so it can
    be reshaped to ``(n_dev, per_dev)`` for pmap.
    """
    n_cells = n_z * N_state
    raw_chunk_size = (n_cells + n_chunks - 1) // n_chunks
    chunk_size = ((raw_chunk_size + n_dev - 1) // n_dev) * n_dev
    per_dev = chunk_size // n_dev
    n_cells_padded = chunk_size * n_chunks

    cell_idx = np.arange(n_cells, dtype=np.int64)
    z_idx_np = (cell_idx // N_state).astype(np.int64)
    is_idx_np = (cell_idx % N_state).astype(np.int64)

    pad_count = n_cells_padded - n_cells
    if pad_count > 0:
        z_idx_np = np.concatenate(
            [z_idx_np, np.full(pad_count, z_idx_np[-1], dtype=np.int64)]
        )
        is_idx_np = np.concatenate(
            [is_idx_np, np.full(pad_count, is_idx_np[-1], dtype=np.int64)]
        )

    z_chunks = z_idx_np.reshape(n_chunks, n_dev, per_dev)
    is_chunks = is_idx_np.reshape(n_chunks, n_dev, per_dev)
    return jnp.asarray(z_chunks), jnp.asarray(is_chunks), n_cells, chunk_size


def _chunked_pmap_runner(pmap_chunk_fn, z_chunks_pm, is_chunks_pm,
                         n_cells, chunk_size, n_chunks):
    """Build a Python-level chunk loop around a pmap'd per-chunk solve.

    The loop intentionally lives outside pmap/JIT. Each chunk gets one pmap
    dispatch, then ``block_until_ready`` forces the chunk's intermediates to
    be released before scheduling the next chunk.
    """
    def runner(*kernel_args):
        chunk_results = []
        for i in range(n_chunks):
            out = pmap_chunk_fn(
                z_chunks_pm[i], is_chunks_pm[i], *kernel_args,
            )
            out[0].block_until_ready()
            chunk_results.append(out)
        n_outs = len(chunk_results[0])

        def collapse(k):
            flat_chunks = [
                jnp.reshape(r[k], (chunk_size,) + r[k].shape[2:])
                for r in chunk_results
            ]
            return jnp.concatenate(flat_chunks, axis=0)[:n_cells]

        return tuple(collapse(k) for k in range(n_outs))

    return runner


def _chunked_vmap_runner(jit_chunk_fn, z_idx_padded, is_idx_padded,
                          n_cells, chunk_size, n_chunks):
    """Build a Python-level chunk-loop runner around a JIT'd per-chunk vmap.

    The chunk loop runs in Python (outside any ``@jit`` boundary). Each
    iteration is a separate JIT call. ``.block_until_ready()`` on the first
    output of each chunk forces that chunk's intermediates to be released
    before the next chunk schedules its allocations — the deterministic
    per-chunk memory bound the chunking design intends.

    Why this matters: an earlier version placed the ``for i in range(n_chunks):``
    loop *inside* an ``@jit`` function. Python unrolls the loop at trace time
    and XLA emits one HLO graph spanning all chunks. The XLA scheduler is
    then free to materialise multiple chunks' working memory concurrently,
    which produces correct math (output is bit-identical) but defeats the
    memory bound entirely. Moving the loop to Python land — with each chunk
    being its own ``@jit`` call — restores the bound.

    Parameters
    ----------
    jit_chunk_fn : @jit-compiled callable
        Signature ``jit_chunk_fn(*kernel_args, z_chunk, is_chunk) -> tuple``,
        where each output entry has leading axis ``chunk_size``.
    z_idx_padded, is_idx_padded : jnp.ndarray, shape ``(chunk_size * n_chunks,)``
        Pre-padded index arrays (last cell index repeated to fill).
    n_cells : int
        Real cell count; used to slice off padding after concat.
    chunk_size : int
        Per-chunk fixed cell count = ``ceil(n_cells / n_chunks)``.
    n_chunks : int
        Static Python int. Must be >= 1.

    Returns
    -------
    runner : callable
        ``runner(*kernel_args) -> tuple of (n_cells, ...) arrays``.
    """
    def runner(*kernel_args):
        chunk_results = []
        for i in range(n_chunks):
            start = i * chunk_size
            z_chunk = z_idx_padded[start:start + chunk_size]
            is_chunk = is_idx_padded[start:start + chunk_size]
            out = jit_chunk_fn(*kernel_args, z_chunk, is_chunk)
            # Block on this chunk's first output before scheduling the next.
            # Without this, JAX's async dispatch lets multiple chunks queue
            # up and XLA may allocate them concurrently — defeating the bound.
            out[0].block_until_ready()
            chunk_results.append(out)
        n_outs = len(chunk_results[0])
        return tuple(
            jnp.concatenate([r[k] for r in chunk_results], axis=0)[:n_cells]
            for k in range(n_outs)
        )
    return runner


def _build_per_age_terminal_kernel_vmap_only(pcj, mp, sc):
    """Single-device terminal kernel. Drops the pmap padding/reshape/collapse
    so XLA can fuse the per-i_s solve into one kernel. Output: ``(N_state, n_w)``.
    """
    init_a_s = jnp.float64(sc.init_alpha_s)
    init_a_b = jnp.float64(sc.init_alpha_b)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton))

    # Pre-build per-i_s log-return tensors (NumPy → jnp) once. Same prep as
    # the pmap path; only the padding/reshape-for-pmap step is dropped.
    log_R_bill, log_x_s, log_x_b = _all_is_log_returns_numpy(pcj)

    n_chunks = int(sc.cell_vmap_chunks)
    N_state = log_R_bill.shape[0]

    # Pre-pad on the host so the kernel sees jnp constants. Same strategy as
    # the pmap path (np.concatenate at build time).
    if n_chunks == 1:
        chunk_size = N_state
        N_padded = N_state
    else:
        chunk_size = (N_state + n_chunks - 1) // n_chunks
        N_padded = chunk_size * n_chunks

    def pad0_np(arr_np, target_n):
        if arr_np.shape[0] == target_n:
            return arr_np
        last = arr_np[-1:]
        extra = target_n - arr_np.shape[0]
        return np.concatenate([arr_np] + [last] * extra, axis=0)

    log_R_bill_jnp = jnp.asarray(pad0_np(log_R_bill, N_padded))
    log_x_s_jnp = jnp.asarray(pad0_np(log_x_s, N_padded))
    log_x_b_jnp = jnp.asarray(pad0_np(log_x_b, N_padded))
    ann_jnp = jnp.asarray(pad0_np(np.asarray(pcj.annuity_factors), N_padded))

    def per_i_s(log_Rb, lxs, lxb, A):
        return _solve_terminal_at_i_s(
            log_Rb, lxs, lxb, pcj.weight_kv_kr, A,
            pcj.s_grid, pcj.wealth_grid,
            init_a_s, init_a_b,
            mp.gamma, mp.beta, mp.b_bar, mp.delta,
            pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
            *static,
        )

    # per_chunk: one @jit shared by both K=1 fast path and K>1 chunked path.
    # Same trace structure for both => bit-identical math regardless of K.
    @jit
    def per_chunk(log_Rb_c, lxs_c, lxb_c, ann_c):
        return vmap(per_i_s)(log_Rb_c, lxs_c, lxb_c, ann_c)

    if n_chunks == 1:
        # Fast path: single per_chunk call with the full unpadded tensors.
        # No chunk loop, no inter-chunk block.
        def call(_unused_age_idx=None):
            return per_chunk(log_R_bill_jnp, log_x_s_jnp, log_x_b_jnp, ann_jnp)
        return call

    # Chunked path: per_chunk called K times in a Python-level loop with
    # .block_until_ready() between chunks. Same memory-bounding rationale as
    # `_chunked_vmap_runner` (see that helper's docstring). Terminal chunks
    # over the i_s tensor inputs directly (no index arrays) so it inlines
    # its own loop rather than reusing the index-array runner.
    def call(_unused_age_idx=None):
        chunk_results = []
        for i in range(n_chunks):
            start = i * chunk_size
            log_Rb_c = log_R_bill_jnp[start:start + chunk_size]
            lxs_c = log_x_s_jnp[start:start + chunk_size]
            lxb_c = log_x_b_jnp[start:start + chunk_size]
            ann_c = ann_jnp[start:start + chunk_size]
            out = per_chunk(log_Rb_c, lxs_c, lxb_c, ann_c)
            # Block on this chunk's first output before scheduling the next,
            # so its working memory is released before the next allocates.
            out[0].block_until_ready()
            chunk_results.append(out)
        n_outs = len(chunk_results[0])
        return tuple(
            jnp.concatenate([r[k] for r in chunk_results], axis=0)[:N_state]
            for k in range(n_outs)
        )

    return call


def _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state, per_is_tensors):
    """Dispatch on n_dev: vmap-only on single-device hosts, pmap+vmap on multi-device.

    ``per_is_tensors`` is the result of ``_precompute_per_is_tensors(pcj)``,
    hoisted by the caller so retirement / working / boundary builders share it.
    """
    if n_dev == 1:
        return _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state, per_is_tensors)
    return _build_per_age_retirement_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state, per_is_tensors)


def _build_per_age_retirement_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state, per_is_tensors):
    """Build a pmap'd retirement kernel. Returns a callable
    ``call(c_next, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr)
    -> (n_z, N_state, n_w)``.

    ``init_a_s_arr``/``init_a_b_arr`` are (n_z, N_state, n_w) policy arrays
    from the previous (older) age. The Newton init at cell (z_idx, i_s) is
    gathered as ``[z_idx, i_s, w_ref_idx]`` (mid-wealth slice), giving each
    cell a single warm-started scalar shared across the savings vmap.
    """
    gather_dtype = _resolve_gather_dtype(sc)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton),
              gather_dtype)

    # Per-i_s precompute is supplied by the caller (single trace shared across
    # retirement / working / boundary builders).
    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    n_w = pcj.wealth_grid.shape[0]
    w_ref_idx = n_w // 2

    n_chunks = int(sc.cell_vmap_chunks)
    if n_chunks == 1:
        n_cells = n_z * N_state
        pad_n = math.ceil(n_cells / n_dev) * n_dev
        per_dev = pad_n // n_dev

        cell_idx = np.arange(n_cells, dtype=np.int64)
        cell_idx_padded = np.concatenate([cell_idx, np.full(pad_n - n_cells, cell_idx[-1])])
        z_idx_padded = (cell_idx_padded // N_state).astype(np.int64)
        is_idx_padded = (cell_idx_padded % N_state).astype(np.int64)
        z_pm = jnp.asarray(z_idx_padded.reshape(n_dev, per_dev))
        is_pm = jnp.asarray(is_idx_padded.reshape(n_dev, per_dev))
    else:
        z_chunks_pm, is_chunks_pm, n_cells, chunk_size = (
            _build_chunked_pmap_index_arrays(n_z, N_state, n_chunks, n_dev)
        )

    @partial(pmap, in_axes=(0, 0, None, None, None, None, None))
    def per_dev_solve(
        z_block, is_block, c_next, pension_next_by_z, psi_per_z,
        init_a_s_arr, init_a_b_arr,
    ):
        def per_cell(z_idx, i_s):
            init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
            init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]
            return _solve_retirement_at_cell(
                z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
                log_R_bill_all[i_s], log_x_s_all[i_s], log_x_b_all[i_s],
                j_corners_all[i_s], w_corners_all[i_s],
                pcj.weight_kv_kr, pcj.annuity_factors,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s_cell, init_a_b_cell,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )
        return vmap(per_cell)(z_block, is_block)

    if n_chunks != 1:
        runner = _chunked_pmap_runner(
            per_dev_solve, z_chunks_pm, is_chunks_pm, n_cells, chunk_size, n_chunks,
        )

    def call(c_next_jnp, pension_next_by_z, psi_per_z, init_a_s_arr, init_a_b_arr):
        if n_chunks != 1:
            c_flat, s_flat, b_flat, ni_flat, nb_flat = runner(
                c_next_jnp, pension_next_by_z, psi_per_z,
                init_a_s_arr, init_a_b_arr,
            )
            return (
                jnp.reshape(c_flat, (n_z, N_state, -1)),
                jnp.reshape(s_flat, (n_z, N_state, -1)),
                jnp.reshape(b_flat, (n_z, N_state, -1)),
                jnp.reshape(ni_flat, (n_z, N_state, -1)),
                jnp.reshape(nb_flat, (n_z, N_state, -1)),
            )

        (c_pm, as_pm, ab_pm,
         ni_pm, nb_pm) = per_dev_solve(
            z_pm, is_pm, c_next_jnp, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )
        # (n_dev, per_dev, n_w) -> (pad_n, n_w) -> (n_cells, n_w) -> (n_z, N_state, n_w)
        # Stay on device.
        def collapse(a):
            flat = jnp.reshape(a, (pad_n,) + a.shape[2:])
            return jnp.reshape(flat[:n_cells], (n_z, N_state) + a.shape[2:])

        return (
            collapse(c_pm), collapse(as_pm), collapse(ab_pm),
            collapse(ni_pm), collapse(nb_pm),
        )

    return call


def _build_per_age_retirement_kernel_vmap_only(pcj, mp, sc, n_z, N_state, per_is_tensors):
    """Single-device retirement kernel. Output shape matches the pmap path:
    ``(n_z, N_state, n_w)`` per policy array.

    Cell-axis chunking (``sc.cell_vmap_chunks``) is implemented with the
    Python-level chunk loop in ``_chunked_vmap_runner`` — each chunk is its
    own ``@jit`` call so peak HBM is bounded at one chunk's working memory
    regardless of XLA's scheduler choices. ``cell_vmap_chunks=1`` keeps the
    fast path: a single ``@jit'd vmap`` over all cells, no chunk-loop wrapper.
    """
    gather_dtype = _resolve_gather_dtype(sc)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton),
              gather_dtype)

    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    n_w = pcj.wealth_grid.shape[0]
    w_ref_idx = n_w // 2

    n_chunks = int(sc.cell_vmap_chunks)
    z_idx_padded, is_idx_padded, n_cells, chunk_size = _build_chunked_index_arrays(
        n_z, N_state, n_chunks,
    )

    def per_cell(z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
                 init_a_s_arr, init_a_b_arr):
        init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
        init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]
        return _solve_retirement_at_cell(
            z_idx, i_s, c_next, pension_next_by_z, psi_per_z,
            log_R_bill_all[i_s], log_x_s_all[i_s], log_x_b_all[i_s],
            j_corners_all[i_s], w_corners_all[i_s],
            pcj.weight_kv_kr, pcj.annuity_factors,
            pcj.s_grid, pcj.wealth_grid,
            init_a_s_cell, init_a_b_cell,
            mp.gamma, mp.beta, mp.b_bar, mp.delta,
            pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
            *static,
        )

    # per_chunk: one @jit'd function shared by both K=1 (called once with the
    # full padded indices) and K>1 (called K times via _chunked_vmap_runner).
    # Using one trace for both paths guarantees K=1 and K>1 produce
    # bit-identical math — separate fast-path traces can diverge in subtle
    # XLA-fusion ways even when the algorithm is the same.
    @jit
    def per_chunk(c_next, pension_next_by_z, psi_per_z,
                   init_a_s_arr, init_a_b_arr, z_chunk, is_chunk):
        return vmap(
            per_cell, in_axes=(0, 0, None, None, None, None, None),
        )(
            z_chunk, is_chunk,
            c_next, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )

    if n_chunks == 1:
        # Fast path: one per_chunk call with the full unpadded indices. No
        # chunk-loop wrapper, no inter-chunk block — zero overhead vs the
        # pre-chunking baseline for 5⁴ runs.
        z_full = z_idx_padded[:n_cells]
        is_full = is_idx_padded[:n_cells]

        def call(c_next_jnp, pension_next_by_z, psi_per_z,
                  init_a_s_arr, init_a_b_arr):
            c_flat, s_flat, b_flat, ni_flat, nb_flat = per_chunk(
                c_next_jnp, pension_next_by_z, psi_per_z,
                init_a_s_arr, init_a_b_arr,
                z_full, is_full,
            )
            return (
                jnp.reshape(c_flat, (n_z, N_state, -1)),
                jnp.reshape(s_flat, (n_z, N_state, -1)),
                jnp.reshape(b_flat, (n_z, N_state, -1)),
                jnp.reshape(ni_flat, (n_z, N_state, -1)),
                jnp.reshape(nb_flat, (n_z, N_state, -1)),
            )
        return call

    # Chunked path: same per_chunk, called K times via _chunked_vmap_runner.
    # The chunk for-loop runs in Python with .block_until_ready() between
    # chunks so each chunk's working memory is freed before the next allocates.
    runner = _chunked_vmap_runner(
        per_chunk, z_idx_padded, is_idx_padded, n_cells, chunk_size, n_chunks,
    )

    def call(c_next_jnp, pension_next_by_z, psi_per_z,
              init_a_s_arr, init_a_b_arr):
        c_flat, s_flat, b_flat, ni_flat, nb_flat = runner(
            c_next_jnp, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )
        return (
            jnp.reshape(c_flat, (n_z, N_state, -1)),
            jnp.reshape(s_flat, (n_z, N_state, -1)),
            jnp.reshape(b_flat, (n_z, N_state, -1)),
            jnp.reshape(ni_flat, (n_z, N_state, -1)),
            jnp.reshape(nb_flat, (n_z, N_state, -1)),
        )

    return call


def _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next, per_is_tensors):
    """Dispatch on n_dev: vmap-only on single-device hosts, pmap+vmap on multi-device.

    ``per_is_tensors`` is the result of ``_precompute_per_is_tensors(pcj)``,
    hoisted by the caller so retirement / working / boundary builders share it.
    """
    if n_dev == 1:
        return _build_per_age_working_kernel_vmap_only(pcj, mp, sc, n_z, N_state, use_pension_next, per_is_tensors)
    return _build_per_age_working_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next, per_is_tensors)


def _build_per_age_working_kernel_pmap(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next, per_is_tensors):
    """Build a pmap'd working-age kernel. The boundary case
    (work -> retirement, ``use_pension_next == True``) is a separate trace
    selected by the orchestrator.

    Newton init at each (z_idx, i_s) cell is gathered from
    ``init_a_s_arr[z_idx, i_s, w_ref_idx]`` (and similarly for a_b),
    where ``init_*_arr`` is the previous (older) age's converged policy.
    """
    gather_dtype = _resolve_gather_dtype(sc)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton),
              gather_dtype)

    # Per-i_s precompute is supplied by the caller (single trace shared across
    # retirement / working / boundary builders).
    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    n_w = pcj.wealth_grid.shape[0]
    w_ref_idx = n_w // 2

    n_chunks = int(sc.cell_vmap_chunks)
    if n_chunks == 1:
        n_cells = n_z * N_state
        pad_n = math.ceil(n_cells / n_dev) * n_dev
        per_dev = pad_n // n_dev

        cell_idx = np.arange(n_cells, dtype=np.int64)
        cell_idx_padded = np.concatenate([cell_idx, np.full(pad_n - n_cells, cell_idx[-1])])
        z_idx_padded = (cell_idx_padded // N_state).astype(np.int64)
        is_idx_padded = (cell_idx_padded % N_state).astype(np.int64)
        z_pm = jnp.asarray(z_idx_padded.reshape(n_dev, per_dev))
        is_pm = jnp.asarray(is_idx_padded.reshape(n_dev, per_dev))
    else:
        z_chunks_pm, is_chunks_pm, n_cells, chunk_size = (
            _build_chunked_pmap_index_arrays(n_z, N_state, n_chunks, n_dev)
        )

    @partial(pmap, in_axes=(0, 0, None, None, None, None, None, None))
    def per_dev_solve(
        z_block, is_block, c_next,
        income_next_table_z, pension_next_by_z, psi_per_z,
        init_a_s_arr, init_a_b_arr,
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

            init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
            init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]

            return _solve_working_at_cell(
                z_idx, i_s, c_next,
                income_table,
                iz_lo, frac_z,
                pcj.eta_weights, pcj.eps_weights,
                psi_per_z,
                log_R_bill_all[i_s], log_x_s_all[i_s], log_x_b_all[i_s],
                j_corners_all[i_s], w_corners_all[i_s],
                pcj.weight_kv_kr, pcj.annuity_factors,
                pcj.s_grid, pcj.wealth_grid,
                init_a_s_cell, init_a_b_cell,
                mp.gamma, mp.beta, mp.b_bar, mp.delta,
                pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
                *static,
            )

        return vmap(per_cell)(z_block, is_block)

    if n_chunks != 1:
        runner = _chunked_pmap_runner(
            per_dev_solve, z_chunks_pm, is_chunks_pm, n_cells, chunk_size, n_chunks,
        )

    def call(
        c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
        init_a_s_arr, init_a_b_arr,
    ):
        if n_chunks != 1:
            c_flat, s_flat, b_flat, ni_flat, nb_flat = runner(
                c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
                init_a_s_arr, init_a_b_arr,
            )
            return (
                jnp.reshape(c_flat, (n_z, N_state, -1)),
                jnp.reshape(s_flat, (n_z, N_state, -1)),
                jnp.reshape(b_flat, (n_z, N_state, -1)),
                jnp.reshape(ni_flat, (n_z, N_state, -1)),
                jnp.reshape(nb_flat, (n_z, N_state, -1)),
            )

        (c_pm, as_pm, ab_pm,
         ni_pm, nb_pm) = per_dev_solve(
            z_pm, is_pm, c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )
        # Stay on device.
        def collapse(a):
            flat = jnp.reshape(a, (pad_n,) + a.shape[2:])
            return jnp.reshape(flat[:n_cells], (n_z, N_state) + a.shape[2:])

        return (
            collapse(c_pm), collapse(as_pm), collapse(ab_pm),
            collapse(ni_pm), collapse(nb_pm),
        )

    return call


def _build_per_age_working_kernel_vmap_only(pcj, mp, sc, n_z, N_state, use_pension_next, per_is_tensors):
    """Single-device working-age kernel (and work->retirement boundary case
    when ``use_pension_next``). Output shape matches the pmap path:
    ``(n_z, N_state, n_w)`` per policy array.

    Cell-axis chunking (``sc.cell_vmap_chunks``) is implemented via
    ``_chunked_vmap_runner`` — Python-level chunk loop, ``.block_until_ready()``
    between chunks, deterministic per-chunk HBM bound. ``cell_vmap_chunks=1``
    keeps the fast path of a single ``@jit'd vmap`` over all cells.
    """
    gather_dtype = _resolve_gather_dtype(sc)
    static = (sc.tol, sc.max_iter, sc.max_backtrack_iter,
              sc.line_search_max_step, sc.singular_det, sc.grad_step_size,
              sc.grad_denom_eps, sc.tiny_savings, sc.euler_inv_floor,
              sc.min_consumption, sc.egm_anchor,
              bool(sc.use_fori_newton),
              gather_dtype)

    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is_tensors

    n_w = pcj.wealth_grid.shape[0]
    w_ref_idx = n_w // 2

    n_chunks = int(sc.cell_vmap_chunks)
    z_idx_padded, is_idx_padded, n_cells, chunk_size = _build_chunked_index_arrays(
        n_z, N_state, n_chunks,
    )

    def per_cell(z_idx, i_s, c_next, income_next_table_z, pension_next_by_z,
                  psi_per_z, init_a_s_arr, init_a_b_arr):
        z_now = pcj.z_grid[z_idx]
        z_next = mp.rho * z_now + pcj.eta_nodes
        iz_lo, frac_z = vmap(bracket_uniform, in_axes=(0, None, None, None))(
            z_next, pcj.z_grid[0], pcj.dz, pcj.z_grid.shape[0]
        )

        if use_pension_next:
            pension_at_eta = (
                (1.0 - frac_z) * pension_next_by_z[iz_lo]
                + frac_z * pension_next_by_z[iz_lo + 1]
            )
            income_table = pension_at_eta[:, None] * jnp.ones_like(pcj.eps_weights)[None, :]
        else:
            income_table = income_next_table_z[z_idx]

        init_a_s_cell = init_a_s_arr[z_idx, i_s, w_ref_idx]
        init_a_b_cell = init_a_b_arr[z_idx, i_s, w_ref_idx]

        return _solve_working_at_cell(
            z_idx, i_s, c_next,
            income_table,
            iz_lo, frac_z,
            pcj.eta_weights, pcj.eps_weights,
            psi_per_z,
            log_R_bill_all[i_s], log_x_s_all[i_s], log_x_b_all[i_s],
            j_corners_all[i_s], w_corners_all[i_s],
            pcj.weight_kv_kr, pcj.annuity_factors,
            pcj.s_grid, pcj.wealth_grid,
            init_a_s_cell, init_a_b_cell,
            mp.gamma, mp.beta, mp.b_bar, mp.delta,
            pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
            *static,
        )

    # per_chunk: one @jit shared by both K=1 (called once with full padded
    # indices) and K>1 (called K times via the runner). One trace shared by
    # both paths guarantees bit-identical math.
    @jit
    def per_chunk(
        c_next, income_next_table_z, pension_next_by_z, psi_per_z,
        init_a_s_arr, init_a_b_arr, z_chunk, is_chunk,
    ):
        return vmap(
            per_cell, in_axes=(0, 0, None, None, None, None, None, None),
        )(
            z_chunk, is_chunk,
            c_next, income_next_table_z, pension_next_by_z,
            psi_per_z, init_a_s_arr, init_a_b_arr,
        )

    if n_chunks == 1:
        # Fast path: single per_chunk call with the full unpadded indices.
        z_full = z_idx_padded[:n_cells]
        is_full = is_idx_padded[:n_cells]

        def call(
            c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        ):
            c_flat, s_flat, b_flat, ni_flat, nb_flat = per_chunk(
                c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
                init_a_s_arr, init_a_b_arr,
                z_full, is_full,
            )
            return (
                jnp.reshape(c_flat, (n_z, N_state, -1)),
                jnp.reshape(s_flat, (n_z, N_state, -1)),
                jnp.reshape(b_flat, (n_z, N_state, -1)),
                jnp.reshape(ni_flat, (n_z, N_state, -1)),
                jnp.reshape(nb_flat, (n_z, N_state, -1)),
            )
        return call

    # Chunked path: per_chunk called K times via _chunked_vmap_runner. Chunk
    # loop runs in Python with .block_until_ready() between chunks.
    runner = _chunked_vmap_runner(
        per_chunk, z_idx_padded, is_idx_padded, n_cells, chunk_size, n_chunks,
    )

    def call(
        c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
        init_a_s_arr, init_a_b_arr,
    ):
        c_flat, s_flat, b_flat, ni_flat, nb_flat = runner(
            c_next_jnp, income_next_table, pension_next_by_z, psi_per_z,
            init_a_s_arr, init_a_b_arr,
        )
        return (
            jnp.reshape(c_flat, (n_z, N_state, -1)),
            jnp.reshape(s_flat, (n_z, N_state, -1)),
            jnp.reshape(b_flat, (n_z, N_state, -1)),
            jnp.reshape(ni_flat, (n_z, N_state, -1)),
            jnp.reshape(nb_flat, (n_z, N_state, -1)),
        )

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
    if int(sc.cell_vmap_chunks) < 1:
        raise ValueError(
            f"SolverConfig.cell_vmap_chunks must be >= 1, got {sc.cell_vmap_chunks}."
        )
    delta = sc.delta_bequest if sc.delta_bequest >= 0.0 else DELTA_BEQUEST

    solve_control, control_active = _normalize_solve_control(model, pc, solve_control)

    if verbose >= 1:
        n_dev_log = len(jax.devices())
        pattern = "vmap-only (single-device)" if n_dev_log == 1 else f"pmap+vmap ({n_dev_log} devices)"
        print(f"\n{'='*70}")
        print(f"LIFECYCLE PORTFOLIO SOLVER  (JAX, EGM + 2D Newton)")
        print(f"  Devices: {jax.devices()}")
        print(f"  Cell-batching pattern: {pattern}")
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
    # Compute per-i_s log-return / state-bracket tensors once and share across
    # the three non-terminal builders. Independent of z; depends only on pcj.
    per_is_tensors = _precompute_per_is_tensors(pcj)
    terminal_kernel = _build_per_age_terminal_kernel(pcj, mp, sc, n_dev)
    retirement_kernel = _build_per_age_retirement_kernel(pcj, mp, sc, n_dev, n_z, N_state, per_is_tensors)
    working_kernel = _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next=False, per_is_tensors=per_is_tensors)
    boundary_kernel = _build_per_age_working_kernel(pcj, mp, sc, n_dev, n_z, N_state, use_pension_next=True, per_is_tensors=per_is_tensors)

    # ---- Output arrays ----
    # Policies live on device as list[jnp.ndarray | None] across ages, so the
    # backward induction never round-trips through host memory between ages.
    # Materialised to NumPy at checkpoint/return boundaries via
    # _materialize_policy_lists.
    shape = (n_age, n_z, N_state, n_w)
    C_list = [None] * n_age
    S_list = [None] * n_age
    B_list = [None] * n_age
    solved_age_mask = np.zeros(n_age, dtype=bool)

    # ---- Per-age diagnostics (post-hoc only) ----
    age_max_foc = np.zeros(n_age)
    age_newton_fail = np.zeros(n_age, dtype=np.int64)

    # Per-age Newton-iter / backtrack-iter counts, one entry per (cell, savings).
    # Terminal is z-invariant so its entries are (N_state, n_savings) —
    # broadcast-equivalent to (n_z, N_state, n_savings) for histogram aggregation.
    # Lists hold device or host arrays; converted to NumPy at the histogram
    # aggregation step at end.
    newton_iter_per_age = [None] * n_age
    backtrack_iter_per_age = [None] * n_age

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
            if Cc.shape != shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch at {ckpt_dir}: "
                    f"got {Cc.shape}, expected {shape}. "
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
            # Upload solved slabs to device once at resume; each remains
            # device-resident for the rest of the solve.
            for t in range(n_age):
                if ckpt_mask[t]:
                    C_list[t] = jnp.asarray(Cc[t])
                    S_list[t] = jnp.asarray(Sc[t])
                    B_list[t] = jnp.asarray(Bc[t])
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
    # Pick per-axis midpoints, not the flat-index midpoint. For a state
    # grid of shape (n0, n1, n2, n3) (e.g. (6,6,6,6)), N_state // 2 unravels
    # to (n0//2, 0, 0, 0) in C-order — which means cy at center but spr/rtb/
    # y_1 pinned to axis index 0 (extreme tail). The probe values were then
    # corner-cell readings, not state-grid-center readings, masking the
    # correct mean-state economics under noisy tail-cell extremes.
    state_axis_sizes = tuple(g.shape[0] for g in pc.state_bracket_grids)
    mid_per_axis = tuple(n // 2 for n in state_axis_sizes)
    i_s_med = int(np.ravel_multi_index(mid_per_axis, state_axis_sizes))

    # ---- Terminal age ----
    if not solved_age_mask[-1]:
        if verbose >= 1:
            print(f"\n  Terminal condition (age {terminal_age}) ... ", end="", flush=True)
        c_T, s_T, b_T, ni_T, nb_T = terminal_kernel()
        # Broadcast across z (terminal policy is z-invariant — bequest only).
        # jnp.broadcast_to stays on device. On the pmap path (n_dev > 1) the
        # next kernel reads it via in_axes=None and materialises lazily; on
        # the vmap-only path (n_dev == 1) the broadcast threads through the
        # JIT trace and the indexing gather inside the kernel materialises
        # only what it needs.
        C_list[-1] = jnp.broadcast_to(c_T[None, :, :], (n_z, N_state, n_w))
        S_list[-1] = jnp.broadcast_to(s_T[None, :, :], (n_z, N_state, n_w))
        B_list[-1] = jnp.broadcast_to(b_T[None, :, :], (n_z, N_state, n_w))
        # Terminal iter counts are (N_state,) — z-invariant. Materialise to
        # NumPy now (~5KB) so the orchestrator's aggregator never holds a
        # device array for them.
        newton_iter_per_age[-1] = np.asarray(ni_T)
        backtrack_iter_per_age[-1] = np.asarray(nb_T)
        solved_age_mask[-1] = True
        if verbose >= 1:
            # Small reduction (~5KB transfer) for the print line; cheap.
            print(f"done  [c range: {float(c_T.min()):.3f}-{float(c_T.max()):.3f}]")
    elif verbose >= 1:
        print(f"\n  Terminal condition (age {terminal_age}) ... loaded from checkpoint")

    # ---- Backward induction header ----
    if verbose >= 1:
        print(f"\n{'='*100}")
        print(f"  Live policy probe: z=z_grid[{i_z_med}], state per-axis center {mid_per_axis} (i_s={i_s_med}), wealth={progress_wealth_label}")
        hdr = (f" {'Age':>3}  {'Phase':<6} {'Time':>6}  "
               f"{'alpha_s':>7}  {'alpha_b':>7}  {'a_bill':>7}  {'W':>6}  {'c/W':>5}")
        print(hdr)
        print(f"{'='*100}")

    t_start = time.time()
    solve_status = "complete"

    # Hoist all per-age tables to device once. Slicing inside the loop
    # (e.g. survival_jnp[t, :]) stays on-device; avoids ~250 host->device
    # dispatch points per solve and reduces GPU pipeline fragmentation.
    pension_table_jnp = jnp.asarray(pc.pension_after_tax)
    pension_dummy_z_jnp = jnp.zeros(n_z, dtype=jnp.float64)
    survival_jnp = jnp.asarray(pc.survival_probs_2d)
    working_income_next_jnp = jnp.asarray(pc.working_income_next)  # (n_age, n_z, n_eta, n_eps)

    # Backward-age warm-start init arrays. When the toggle is off, every cell
    # at every age is seeded from the canonical scalar (init_alpha_s,
    # init_alpha_b); we express that through the same kernel signature by
    # passing constant (n_z, N_state, n_w) arrays so the kernel trace is
    # identical regardless of the toggle.
    init_arr_shape = (n_z, N_state, n_w)
    cold_init_a_s_arr = jnp.full(init_arr_shape, sc.init_alpha_s, dtype=jnp.float64)
    cold_init_a_b_arr = jnp.full(init_arr_shape, sc.init_alpha_b, dtype=jnp.float64)

    try:
        for t in reversed(range(n_age - 1)):
            age = ages[t]
            if youngest_age_to_solve is not None and age < youngest_age_to_solve:
                solve_status = "stopped_early"
                break
            if solved_age_mask[t]:
                continue

            psi_t = survival_jnp[t, :]
            # c_next_jnp is the previous age's policy, already on device.
            c_next_jnp = C_list[t + 1]

            # Backward-age warm-start: previous (older) age's converged policy
            # at the same (z, state) cell, gathered at mid-wealth inside the
            # kernel. Toggle off -> uniform cold init across all cells.
            if sc.use_backward_age_warm_start:
                init_a_s_arr = S_list[t + 1]
                init_a_b_arr = B_list[t + 1]
            else:
                init_a_s_arr = cold_init_a_s_arr
                init_a_b_arr = cold_init_a_b_arr

            if age >= retire_age:
                pension_next = pension_table_jnp[t + 1, :]
                c_t, s_t, b_t, ni_t, nb_t = retirement_kernel(
                    c_next_jnp, pension_next, psi_t, init_a_s_arr, init_a_b_arr,
                )
                label = "RETIRE"
            else:
                use_pen = (age == retire_age - 1)
                if use_pen:
                    pension_next = pension_table_jnp[t + 1, :]
                    income_table = jnp.zeros((n_z, pc.n_eta, pc.n_eps))   # ignored on this branch
                    c_t, s_t, b_t, ni_t, nb_t = boundary_kernel(
                        c_next_jnp, income_table, pension_next, psi_t,
                        init_a_s_arr, init_a_b_arr,
                    )
                else:
                    pension_next = pension_dummy_z_jnp
                    income_table = working_income_next_jnp[t + 1]  # (n_z, n_eta, n_eps)
                    c_t, s_t, b_t, ni_t, nb_t = working_kernel(
                        c_next_jnp, income_table, pension_next, psi_t,
                        init_a_s_arr, init_a_b_arr,
                    )
                label = "WORK  "

            # Store device arrays directly — no D->H round trip.
            C_list[t] = c_t
            S_list[t] = s_t
            B_list[t] = b_t
            # Iter counts are tiny (n_z * N_state ints) — materialise per age
            # so device memory isn't pinned by the histogram aggregator.
            newton_iter_per_age[t] = np.asarray(ni_t)
            backtrack_iter_per_age[t] = np.asarray(nb_t)
            solved_age_mask[t] = True

            if verbose >= 1:
                elapsed = time.time() - t_start
                probe_w = float(progress_wealth_by_age[t])
                # Materialise only the (n_w,) probe slice — ~1KB per array.
                # Single device_get over the tuple = one D->H sync per age,
                # not three (avoids draining the dispatch pipeline 3x).
                s_slice, b_slice, c_slice = jax.device_get((
                    s_t[i_z_med, i_s_med, :],
                    b_t[i_z_med, i_s_med, :],
                    c_t[i_z_med, i_s_med, :],
                ))
                probe_as = _interp_progress_policy_at_wealth(s_slice, pc.wealth_grid, probe_w)
                probe_ab = _interp_progress_policy_at_wealth(b_slice, pc.wealth_grid, probe_w)
                probe_c = _interp_progress_policy_at_wealth(c_slice, pc.wealth_grid, probe_w)
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
                    # Histograms also at periodic checkpoints, so partial
                    # bundles carry max_iter calibration data even when the
                    # run is interrupted before the post-loop aggregator at
                    # line 2784 can run. ~1ms cost per checkpoint.
                    ni_diag_ck, nb_diag_ck = _build_iter_histograms(
                        newton_iter_per_age, backtrack_iter_per_age, ages, solved_age_mask,
                    )
                    diag["newton_iter_histogram"] = ni_diag_ck
                    diag["backtrack_iter_histogram"] = nb_diag_ck
                    # Materialise to NumPy at checkpoint boundary (single
                    # D->H pass for the slabs solved so far).
                    C_ckpt, S_ckpt, B_ckpt = _materialize_policy_lists(
                        C_list, S_list, B_list, shape, solved_age_mask,
                    )
                    last_saved_bundle_path = str(_save_policy_checkpoint(
                        checkpoint_path, C_ckpt, S_ckpt, B_ckpt, diag,
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

    # ---- Newton / backtrack iter histograms ----
    # Per-age entries are either None (unsolved age) or NumPy arrays — terminal
    # is (N_state,), non-terminal is (n_z, N_state). Flatten each, then build
    # aggregate p50/p95/p99/max stats and per-age p99. Under fori_loop both
    # max_iter and max_backtrack_iter are wall cost regardless of cell
    # convergence, so these stats let the user set them from data.
    ni_diag, nb_diag = _build_iter_histograms(
        newton_iter_per_age, backtrack_iter_per_age, ages, solved_age_mask,
    )
    diagnostics["newton_iter_histogram"] = ni_diag
    diagnostics["backtrack_iter_histogram"] = nb_diag

    # Single materialisation point: device-resident policy list -> NumPy.
    # All downstream consumers (final save, sanity check, return value) read
    # from these arrays.
    C_mat, S_mat, B_mat = _materialize_policy_lists(
        C_list, S_list, B_list, shape, solved_age_mask,
    )

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
        nih = diagnostics["newton_iter_histogram"]
        bth = diagnostics["backtrack_iter_histogram"]
        if nih.get("n_cells", 0) > 0:
            print(
                f"  Newton iters:    p50={nih['p50']:.0f}  p95={nih['p95']:.0f}  "
                f"p99={nih['p99']:.0f}  max={nih['max']}  "
                f"(max_iter={sc.max_iter})"
            )
            print(
                f"  Backtrack iters: p50={bth['p50']:.1f}  p95={bth['p95']:.1f}  "
                f"p99={bth['p99']:.1f}  max={bth['max']}  "
                f"(max_backtrack_iter={sc.max_backtrack_iter})"
            )
        print(f"{'='*100}\n")

    # _materialize_policy_lists already filled unsolved ages with NaN, so
    # _mask_unsolved_ages_in_place would be a no-op; skip it.

    if solve_status == "interrupted" and not return_partial_on_interrupt:
        raise KeyboardInterrupt

    return C_mat, S_mat, B_mat, diagnostics


def _build_iter_histograms(newton_iter_per_age, backtrack_iter_per_age,
                            ages, solved_age_mask):
    """Aggregate Newton-iter and backtrack-iter counts into a diag-friendly dict.

    Each per-age entry is either ``None`` (unsolved) or a NumPy array of shape
    ``(N_state, n_savings)`` (terminal) / ``(n_z, N_state, n_savings)``
    (non-terminal). The savings axis is per-real-Newton-call (the s=0 anchor
    is excluded); collapsing earlier (e.g. ``jnp.max`` over savings per cell)
    saturates the histogram because the highest-savings Newton solves are
    the ones that hit ``max_iter``. All entries flatten to 1-D and
    concatenate; per-age stats are computed on the flattened slice.

    Returns a tuple ``(newton_diag, backtrack_diag)`` of dicts with keys
    ``p50``, ``p95``, ``p99``, ``max``, ``per_age_p99``, ``per_age_max``,
    ``per_age_ages``, and ``n_cells``. When no ages were solved, returns
    empty stats with ``n_cells=0``.
    """
    def _stats_for(per_age_list):
        per_age_p99 = []
        per_age_max = []
        per_age_age = []
        flat_chunks = []
        for t, arr in enumerate(per_age_list):
            if arr is None or not solved_age_mask[t]:
                continue
            arr_flat = np.asarray(arr).ravel()
            if arr_flat.size == 0:
                continue
            flat_chunks.append(arr_flat)
            per_age_p99.append(float(np.percentile(arr_flat, 99)))
            per_age_max.append(int(arr_flat.max()))
            per_age_age.append(int(ages[t]))
        if not flat_chunks:
            return {
                "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0,
                "per_age_p99": [], "per_age_max": [], "per_age_ages": [],
                "n_cells": 0,
            }
        all_flat = np.concatenate(flat_chunks)
        return {
            "p50": float(np.median(all_flat)),
            "p95": float(np.percentile(all_flat, 95)),
            "p99": float(np.percentile(all_flat, 99)),
            "max": int(all_flat.max()),
            "per_age_p99": per_age_p99,
            "per_age_max": per_age_max,
            "per_age_ages": per_age_age,
            "n_cells": int(all_flat.size),
        }

    return _stats_for(newton_iter_per_age), _stats_for(backtrack_iter_per_age)


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

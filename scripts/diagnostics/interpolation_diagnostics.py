"""Empirical diagnostics for z-axis interpolation on saved policy bundles.

This script focuses on the object that the working-age solver actually
interpolates in z: next-period consumption `C_mat[t, iz, i_s, i_w]`.

Implemented diagnostics
-----------------------
1. Local curvature proxy:
       r_curv = |c[i+1] - 2 c[i] + c[i-1]| / |c[i]|
   On a uniform z-grid this is the dimensionless quantity discussed in the
   Catmull-Rom handoff. The implied midpoint relative error of linear
   interpolation is approximately r_curv / 8.

2. Leave-one-out reconstruction at held-out interior z nodes:
   - linear in c
   - Catmull-Rom in c
   - linear in log(c), mapped back to c
   - linear in 1/c, mapped back to c

The diagnostics are aggregated over all financial states, but partitioned by
age bucket and wealth-grid bucket so we can see where cubic interpolation is
actually buying accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lifecycle.discretization import discretize_income_ar1_mixture, get_eta_quadrature_mixture
from lifecycle.policy_io import load_policy_bundle


def _catmull_rom(p0, p1, p2, p3, f):
    """Generic Catmull-Rom evaluation, vectorized over numpy arrays."""
    f2 = f * f
    f3 = f2 * f
    return (
        p1
        + 0.5 * f * (-p0 + p2)
        + 0.5 * f2 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
        + 0.5 * f3 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)
    )


def _catmull_rom_midpoint(p0, p1, p2, p3):
    """Catmull-Rom with f=0.5, vectorized over numpy arrays."""
    return _catmull_rom(p0, p1, p2, p3, 0.5)


def _pchip_slope_pair(d_left, d_right, h_left, h_right):
    """Fritsch-Carlson / PCHIP interior slope on possibly nonuniform spacing."""
    same_sign = (d_left > 0.0) & (d_right > 0.0) | ((d_left < 0.0) & (d_right < 0.0))
    w1 = 2.0 * h_right + h_left
    w2 = h_right + 2.0 * h_left
    with np.errstate(divide="ignore", invalid="ignore"):
        harm = (w1 + w2) / (w1 / d_left + w2 / d_right)
    return np.where(same_sign, harm, 0.0)


def _pchip_interp(p0, p1, p2, p3, f, h0=1.0, h1=1.0, h2=1.0):
    """Local PCHIP/Hermite interpolation on [p1, p2], vectorized over arrays."""
    d0 = (p1 - p0) / h0
    d1 = (p2 - p1) / h1
    d2 = (p3 - p2) / h2
    m1 = _pchip_slope_pair(d0, d1, h0, h1)
    m2 = _pchip_slope_pair(d1, d2, h1, h2)
    t = f
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p1 + h10 * h1 * m1 + h01 * p2 + h11 * h1 * m2


def _unwrap_metadata_value(obj):
    """Extract raw values from summary-style metadata entries when present."""
    if isinstance(obj, dict) and "values" in obj:
        return obj["values"]
    return obj


def _diagnostics_summary(metadata):
    summary = metadata.get("diagnostics_summary", {})
    return summary if isinstance(summary, dict) else {}


def _extract_configs(metadata, n_age, n_w):
    """Best-effort extraction of ages, working mask, and wealth grid."""
    run_config = metadata.get("run_config", {})
    base = run_config.get("base_config", {})
    disc = _extract_disc_config(metadata)
    summary = _diagnostics_summary(metadata)

    start_age = base.get("start_age")
    terminal_age = base.get("terminal_age")
    retire_age = base.get("retire_age")

    ages = None
    if start_age is not None and terminal_age is not None:
        ages_try = np.arange(int(start_age), int(terminal_age) + 1, dtype=int)
        if ages_try.size == n_age:
            ages = ages_try

    if ages is None:
        solved_mask = _extract_solved_age_mask(metadata, n_age)
        youngest_solved = summary.get("youngest_solved_age")
        oldest_solved = summary.get("oldest_solved_age")
        if (
            solved_mask is not None
            and youngest_solved is not None
            and oldest_solved is not None
        ):
            solved_idx = np.flatnonzero(solved_mask)
            if solved_idx.size > 0:
                idx_span = int(solved_idx[-1] - solved_idx[0])
                age_span = int(oldest_solved) - int(youngest_solved)
                if idx_span == age_span:
                    start_age_inferred = int(youngest_solved) - int(solved_idx[0])
                    ages = start_age_inferred + np.arange(n_age, dtype=int)

    if ages is None:
        ages = np.arange(n_age, dtype=int)

    wealth_grid = None
    wealth_min = disc.get("wealth_min")
    wealth_max = disc.get("wealth_max")
    if wealth_min is not None and wealth_max is not None:
        wealth_grid = np.geomspace(float(wealth_min), float(wealth_max), n_w)
    else:
        wealth_grid = np.arange(n_w, dtype=float)

    return ages, retire_age, wealth_grid


def _reconstruct_z_process(metadata, n_z_fallback):
    """Reconstruct the exact z-grid and eta quadrature when metadata is sufficient."""
    run_config = metadata.get("run_config", {})
    base = run_config.get("base_config", {})
    disc = _extract_disc_config(metadata)

    required_base = ("rho", "pz", "mu_eta1", "sigma_eta1", "sigma_eta2")
    required_disc = ("n_z", "n_stds", "n_eta_nodes")

    if not all(k in base for k in required_base):
        return None
    if not all(k in disc for k in required_disc):
        return None

    rho = float(base["rho"])
    pz = float(base["pz"])
    mu_eta1 = float(base["mu_eta1"])
    sigma_eta1 = float(base["sigma_eta1"])
    sigma_eta2 = float(base["sigma_eta2"])
    mu_eta2 = float(base.get("mu_eta2", 0.0))
    n_z = int(disc.get("n_z", n_z_fallback))
    n_stds = float(disc["n_stds"])
    n_eta_nodes = int(disc["n_eta_nodes"])

    z_grid, _ = discretize_income_ar1_mixture(
        rho=rho,
        p=pz,
        mu1=mu_eta1,
        sigma1=sigma_eta1,
        mu2=mu_eta2,
        sigma2=sigma_eta2,
        N=n_z,
        n_stds=n_stds,
    )
    model_stub = SimpleNamespace(
        pz=pz,
        mu_eta1=mu_eta1,
        sigma_eta1=sigma_eta1,
        sigma_eta2=sigma_eta2,
    )
    eta_nodes, eta_weights = get_eta_quadrature_mixture(model_stub, n_nodes=n_eta_nodes)
    return {
        "rho": rho,
        "z_grid": z_grid,
        "dz": float(z_grid[1] - z_grid[0]),
        "eta_nodes": np.asarray(eta_nodes, dtype=float),
        "eta_weights": np.asarray(eta_weights, dtype=float),
    }


def _extract_disc_config(metadata):
    run_config = metadata.get("run_config", {})
    disc = run_config.get("discretization_config", {})
    if not disc:
        disc = metadata.get("disc_config", {})
    if not disc:
        disc = _diagnostics_summary(metadata).get("disc_config", {})
    return disc


def _extract_solved_age_mask(metadata, n_age):
    summary = _diagnostics_summary(metadata)
    mask = metadata.get("solved_age_mask")
    if mask is None:
        mask = summary.get("solved_age_mask")
    mask = _unwrap_metadata_value(mask)
    if mask is None:
        return None

    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (n_age,):
        return None
    return arr


def _extract_state_grid_sizes(metadata):
    disc = _extract_disc_config(metadata)
    sizes = disc.get("state_grid_sizes")
    if sizes is None:
        return None
    return tuple(int(x) for x in sizes)


def _infer_state_axis_labels(metadata):
    disc = _extract_disc_config(metadata)
    mode = disc.get("state_grid_mode")
    if mode in ("cholesky", "principal"):
        return ["u0", "u1", "u2"]

    run_config = metadata.get("run_config", {})
    var_config = run_config.get("var_config", {})
    names = var_config.get("variable_names")
    idx = var_config.get("state_indices")
    if isinstance(names, list) and isinstance(idx, list):
        try:
            labels = [str(names[int(i)]) for i in idx]
            if len(labels) == 3:
                return labels
        except Exception:
            pass

    return ["axis0", "axis1", "axis2"]


def _split_contiguous_range(start, stop, n_chunks):
    """Split [start, stop) into up to n_chunks contiguous slices."""
    if stop <= start:
        return []
    chunks = np.array_split(np.arange(start, stop, dtype=int), n_chunks)
    out = []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        out.append(slice(int(chunk[0]), int(chunk[-1]) + 1))
    return out


def _age_label(age_slice, ages, age_labels_are_real):
    lo = int(ages[age_slice.start])
    hi = int(ages[age_slice.stop - 1])
    if age_labels_are_real:
        return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return f"t{lo}-t{hi}" if lo != hi else f"t{lo}"


def _wealth_label(w_slice, wealth_grid, wealth_labels_are_real):
    lo = wealth_grid[w_slice.start]
    hi = wealth_grid[w_slice.stop - 1]
    if wealth_labels_are_real:
        return f"{lo:.2e}-{hi:.2e}"
    return f"w{w_slice.start}-w{w_slice.stop - 1}"


def _build_age_buckets(ages, retire_age, working_only):
    """Contiguous age buckets, with final working year isolated when possible."""
    n_age = ages.size
    age_labels_are_real = not np.array_equal(ages, np.arange(n_age))

    if working_only and retire_age is not None:
        work_idx = np.flatnonzero(ages < int(retire_age))
        if work_idx.size == 0:
            raise ValueError("No working-age slices found in the bundle.")

        trans_idx = np.flatnonzero(ages == int(retire_age) - 1)
        if trans_idx.size == 1 and trans_idx[0] in work_idx:
            trans_t = int(trans_idx[0])
            regular = _split_contiguous_range(int(work_idx[0]), trans_t, 3)
            buckets = []
            for sl in regular:
                buckets.append((_age_label(sl, ages, age_labels_are_real), sl))
            trans_slice = slice(trans_t, trans_t + 1)
            buckets.append((_age_label(trans_slice, ages, age_labels_are_real), trans_slice))
            return buckets

        work_slice = slice(int(work_idx[0]), int(work_idx[-1]) + 1)
        parts = _split_contiguous_range(work_slice.start, work_slice.stop, 4)
        return [(_age_label(sl, ages, age_labels_are_real), sl) for sl in parts]

    parts = _split_contiguous_range(0, n_age, 5)
    return [(_age_label(sl, ages, age_labels_are_real), sl) for sl in parts]


def _build_wealth_buckets(wealth_grid, n_buckets):
    n_w = wealth_grid.size
    wealth_labels_are_real = not np.array_equal(wealth_grid, np.arange(n_w, dtype=float))
    parts = _split_contiguous_range(0, n_w, n_buckets)
    return [(_wealth_label(sl, wealth_grid, wealth_labels_are_real), sl) for sl in parts]


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64), dtype=np.float64))


def _compute_metrics(C_view):
    """Compute curvature and leave-one-out metrics for a C_mat view."""
    # C_view shape: (n_age_bucket, n_z, n_state, n_w_bucket)
    if C_view.shape[1] < 5:
        raise ValueError("Need n_z >= 5 for cubic leave-one-out diagnostics.")

    c_mid = C_view[:, 1:-1, :, :]
    curv = np.abs(C_view[:, 2:, :, :] - 2.0 * c_mid + C_view[:, :-2, :, :]) / c_mid
    curv_flat = np.ravel(curv)

    truth = C_view[:, 2:-2, :, :]
    left = C_view[:, 1:-3, :, :]
    right = C_view[:, 3:-1, :, :]

    linear_pred = 0.5 * (left + right)
    cubic_pred = _catmull_rom_midpoint(
        C_view[:, :-4, :, :],
        left,
        right,
        C_view[:, 4:, :, :],
    )
    pchip_pred = _pchip_interp(
        C_view[:, :-4, :, :],
        left,
        right,
        C_view[:, 4:, :, :],
        0.5,
        h0=1.0,
        h1=2.0,
        h2=1.0,
    )
    log_linear_pred = np.exp(0.5 * (np.log(left) + np.log(right)))
    inv_linear_pred = 1.0 / (0.5 * (1.0 / left + 1.0 / right))

    err_linear = (linear_pred - truth) / truth
    err_cubic = (cubic_pred - truth) / truth
    err_pchip = (pchip_pred - truth) / truth
    err_log = (log_linear_pred - truth) / truth
    err_inv = (inv_linear_pred - truth) / truth

    abs_lin = np.abs(err_linear)
    abs_cub = np.abs(err_cubic)
    abs_pchip = np.abs(err_pchip)
    abs_log = np.abs(err_log)

    rmse_linear = _rmse(err_linear)
    rmse_cubic = _rmse(err_cubic)
    rmse_pchip = _rmse(err_pchip)
    rmse_log = _rmse(err_log)
    rmse_inv = _rmse(err_inv)

    return {
        "n_curv": int(curv_flat.size),
        "n_loo": int(truth.size),
        "curv_med_pct": 100.0 * float(np.median(curv_flat)),
        "curv_p95_pct": 100.0 * float(np.quantile(curv_flat, 0.95)),
        "curv_p99_pct": 100.0 * float(np.quantile(curv_flat, 0.99)),
        "curv_share_gt_0p5": 100.0 * float(np.mean(curv_flat > 0.005)),
        "curv_share_gt_5": 100.0 * float(np.mean(curv_flat > 0.05)),
        "midpoint_p95_pct": 12.5 * float(np.quantile(curv_flat, 0.95)),
        "rmse_linear_pct": 100.0 * rmse_linear,
        "rmse_cubic_pct": 100.0 * rmse_cubic,
        "rmse_pchip_pct": 100.0 * rmse_pchip,
        "rmse_loglin_pct": 100.0 * rmse_log,
        "rmse_invlin_pct": 100.0 * rmse_inv,
        "ratio_lin_cubic": rmse_linear / rmse_cubic if rmse_cubic > 0.0 else np.nan,
        "ratio_lin_pchip": rmse_linear / rmse_pchip if rmse_pchip > 0.0 else np.nan,
        "ratio_cubic_pchip": rmse_cubic / rmse_pchip if rmse_pchip > 0.0 else np.nan,
        "ratio_lin_loglin": rmse_linear / rmse_log if rmse_log > 0.0 else np.nan,
        "ratio_cubic_loglin": rmse_cubic / rmse_log if rmse_log > 0.0 else np.nan,
        "cubic_win_pct": 100.0 * float(np.mean(abs_cub < abs_lin)),
        "pchip_win_pct": 100.0 * float(np.mean(abs_pchip < abs_lin)),
        "pchip_beats_cubic_pct": 100.0 * float(np.mean(abs_pchip < abs_cub)),
        "loglin_beats_cubic_pct": 100.0 * float(np.mean(abs_log < abs_cub)),
    }


def _compute_cubic_overshoot_metrics(C_view, tol_rel=1e-12):
    """Scan interior z-intervals for Catmull-Rom overshoot beyond endpoint values.

    For each interval [z_i, z_{i+1}] where Catmull-Rom is active, this checks
    whether the cubic leaves the closed interval spanned by the two endpoint
    node values. Overshoot magnitude is scaled by the average endpoint level.
    """
    n_age_bucket, n_z, _, _ = C_view.shape
    if n_z < 4:
        raise ValueError("Need n_z >= 4 for Catmull-Rom overshoot diagnostics.")

    total_intervals = 0
    total_over = 0
    total_gt_0p1 = 0
    total_gt_1 = 0
    sum_rel_over = 0.0
    max_rel_over = 0.0

    for iz_lo in range(1, n_z - 2):
        p0 = C_view[:, iz_lo - 1, :, :]
        p1 = C_view[:, iz_lo, :, :]
        p2 = C_view[:, iz_lo + 1, :, :]
        p3 = C_view[:, iz_lo + 2, :, :]

        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        scale = np.maximum(0.5 * (np.abs(p1) + np.abs(p2)), 1e-12)

        a1 = 0.5 * (-p0 + p2)
        a2 = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
        a3 = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)

        A = 3.0 * a3
        B = 2.0 * a2
        C = a1

        rel_max = np.zeros_like(scale)
        quad_mask = np.abs(A) > 1e-14
        disc = B * B - 4.0 * A * C
        disc_clipped = np.maximum(disc, 0.0)
        sqrt_disc = np.sqrt(disc_clipped)

        def _update_rel_max(f, valid_mask):
            nonlocal rel_max
            valid = valid_mask & np.isfinite(f) & (f > 0.0) & (f < 1.0)
            if not np.any(valid):
                return
            f_eval = np.where(valid, f, 0.0)
            f2 = f_eval * f_eval
            f3 = f2 * f_eval
            c_val = p1 + a1 * f_eval + a2 * f2 + a3 * f3
            over = np.maximum(lo - c_val, c_val - hi)
            rel = over / scale
            rel = np.where(valid, rel, 0.0)
            rel_max = np.maximum(rel_max, rel)

        with np.errstate(divide="ignore", invalid="ignore"):
            root1 = (-B + sqrt_disc) / (2.0 * A)
            root2 = (-B - sqrt_disc) / (2.0 * A)
        _update_rel_max(root1, quad_mask & (disc >= 0.0))
        _update_rel_max(root2, quad_mask & (disc >= 0.0))

        lin_mask = ~quad_mask & (np.abs(B) > 1e-14)
        with np.errstate(divide="ignore", invalid="ignore"):
            lin_root = -C / B
        _update_rel_max(lin_root, lin_mask)

        positive = rel_max > tol_rel
        total_intervals += int(rel_max.size)
        total_over += int(np.count_nonzero(positive))
        total_gt_0p1 += int(np.count_nonzero(rel_max > 0.001))
        total_gt_1 += int(np.count_nonzero(rel_max > 0.01))
        if np.any(positive):
            sum_rel_over += float(np.sum(rel_max[positive], dtype=np.float64))
            max_rel_over = max(max_rel_over, float(np.max(rel_max)))

    share_over = total_over / total_intervals if total_intervals > 0 else np.nan
    mean_rel_over = sum_rel_over / total_over if total_over > 0 else 0.0

    return {
        "n_intervals": int(total_intervals),
        "overshoot_share_pct": 100.0 * share_over,
        "overshoot_mean_pct": 100.0 * mean_rel_over,
        "overshoot_max_pct": 100.0 * max_rel_over,
        "overshoot_gt_0p1_pct": 100.0 * (total_gt_0p1 / total_intervals) if total_intervals > 0 else np.nan,
        "overshoot_gt_1_pct": 100.0 * (total_gt_1 / total_intervals) if total_intervals > 0 else np.nan,
    }


def _compute_mpc_cubic_scan_metrics(C_view, wealth_grid, tol_abs=1e-12):
    """Scan the z-cubic applied to wealth-slope MPC objects inside _interp_z_wealth.

    Builds nodewise wealth slopes
        mpc[z, ..., iw] = (C[z, ..., iw+1] - C[z, ..., iw]) / dw_iw
    and then checks, along each interior z-interval where Catmull-Rom is used:
    1. whether the cubic leaves the endpoint interval [min(mpc_i, mpc_{i+1}),
       max(mpc_i, mpc_{i+1})]
    2. whether the cubic leaves the economically admissible band [0, 1], which
       would activate the clip in solver.py.
    """
    n_age_bucket, n_z, _, n_w = C_view.shape
    if n_z < 4:
        raise ValueError("Need n_z >= 4 for MPC cubic diagnostics.")
    if n_w < 2:
        raise ValueError("Need at least two wealth nodes for MPC diagnostics.")

    dw = np.diff(np.asarray(wealth_grid, dtype=np.float64))
    if dw.size != n_w - 1:
        raise ValueError("wealth_grid length does not match C_view wealth dimension.")

    mpc_nodes = np.diff(C_view, axis=3) / dw.reshape(1, 1, 1, -1)
    node_oob = (mpc_nodes < -tol_abs) | (mpc_nodes > 1.0 + tol_abs)

    total_intervals = 0
    shape_over_count = 0
    shape_over_gt_1bp = 0
    clip_count = 0
    clip_gt_1bp = 0
    sum_shape_over = 0.0
    sum_clip = 0.0
    max_shape_over = 0.0
    max_clip = 0.0

    for iz_lo in range(1, n_z - 2):
        p0 = mpc_nodes[:, iz_lo - 1, :, :]
        p1 = mpc_nodes[:, iz_lo, :, :]
        p2 = mpc_nodes[:, iz_lo + 1, :, :]
        p3 = mpc_nodes[:, iz_lo + 2, :, :]

        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        scale = np.maximum(0.5 * (np.abs(p1) + np.abs(p2)), 1e-12)

        a1 = 0.5 * (-p0 + p2)
        a2 = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
        a3 = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)

        A = 3.0 * a3
        B = 2.0 * a2
        C = a1

        shape_rel_max = np.zeros_like(scale)
        clip_abs_max = np.zeros_like(scale)
        quad_mask = np.abs(A) > 1e-14
        disc = B * B - 4.0 * A * C
        disc_clipped = np.maximum(disc, 0.0)
        sqrt_disc = np.sqrt(disc_clipped)

        def _update_scan(f, valid_mask):
            nonlocal shape_rel_max, clip_abs_max
            valid = valid_mask & np.isfinite(f) & (f > 0.0) & (f < 1.0)
            if not np.any(valid):
                return
            f_eval = np.where(valid, f, 0.0)
            f2 = f_eval * f_eval
            f3 = f2 * f_eval
            m_val = p1 + a1 * f_eval + a2 * f2 + a3 * f3
            shape_over = np.maximum(lo - m_val, m_val - hi) / scale
            clip_over = np.maximum(np.maximum(-m_val, 0.0), np.maximum(m_val - 1.0, 0.0))
            shape_rel_max = np.maximum(shape_rel_max, np.where(valid, shape_over, 0.0))
            clip_abs_max = np.maximum(clip_abs_max, np.where(valid, clip_over, 0.0))

        with np.errstate(divide="ignore", invalid="ignore"):
            root1 = (-B + sqrt_disc) / (2.0 * A)
            root2 = (-B - sqrt_disc) / (2.0 * A)
        _update_scan(root1, quad_mask & (disc >= 0.0))
        _update_scan(root2, quad_mask & (disc >= 0.0))

        lin_mask = ~quad_mask & (np.abs(B) > 1e-14)
        with np.errstate(divide="ignore", invalid="ignore"):
            lin_root = -C / B
        _update_scan(lin_root, lin_mask)

        shape_positive = shape_rel_max > tol_abs
        clip_positive = clip_abs_max > tol_abs

        total_intervals += int(shape_rel_max.size)
        shape_over_count += int(np.count_nonzero(shape_positive))
        shape_over_gt_1bp += int(np.count_nonzero(shape_rel_max > 0.001))
        clip_count += int(np.count_nonzero(clip_positive))
        clip_gt_1bp += int(np.count_nonzero(clip_abs_max > 0.001))

        if np.any(shape_positive):
            sum_shape_over += float(np.sum(shape_rel_max[shape_positive], dtype=np.float64))
            max_shape_over = max(max_shape_over, float(np.max(shape_rel_max)))
        if np.any(clip_positive):
            sum_clip += float(np.sum(clip_abs_max[clip_positive], dtype=np.float64))
            max_clip = max(max_clip, float(np.max(clip_abs_max)))

    mean_shape_over = sum_shape_over / shape_over_count if shape_over_count > 0 else 0.0
    mean_clip = sum_clip / clip_count if clip_count > 0 else 0.0

    return {
        "n_nodes": int(mpc_nodes.size),
        "node_oob_pct": 100.0 * float(np.mean(node_oob)),
        "node_neg_pct": 100.0 * float(np.mean(mpc_nodes < -tol_abs)),
        "node_gt1_pct": 100.0 * float(np.mean(mpc_nodes > 1.0 + tol_abs)),
        "n_intervals": int(total_intervals),
        "shape_over_pct": 100.0 * (shape_over_count / total_intervals) if total_intervals > 0 else np.nan,
        "shape_over_gt_0p1_pct": 100.0 * (shape_over_gt_1bp / total_intervals) if total_intervals > 0 else np.nan,
        "shape_mean_pct": 100.0 * mean_shape_over,
        "shape_max_pct": 100.0 * max_shape_over,
        "clip_bind_pct": 100.0 * (clip_count / total_intervals) if total_intervals > 0 else np.nan,
        "clip_bind_gt_0p1_pct": 100.0 * (clip_gt_1bp / total_intervals) if total_intervals > 0 else np.nan,
        "clip_mean_abs_pctpt": 100.0 * mean_clip,
        "clip_max_abs_pctpt": 100.0 * max_clip,
    }


def _compute_actual_znext_metrics(C_next_view, z_process, gamma, path_filter="all"):
    """Compare candidate interpolants at the actual off-grid z_next cloud used by the FOC.

    C_next_view shape: (n_current_ages_bucket, n_z, n_state, n_w_bucket)
    path_filter   : "all" | "cubic" | "edge"
    """
    z_grid = z_process["z_grid"]
    dz = float(z_process["dz"])
    rho = float(z_process["rho"])
    eta_nodes = z_process["eta_nodes"]
    eta_weights = z_process["eta_weights"]
    n_z = z_grid.size

    tot_w_count = 0.0
    tot_w_count_cubic = 0.0
    sum_sq_lin = 0.0
    sum_sq_pchip = 0.0
    sum_sq_log = 0.0
    sum_sq_inv = 0.0
    sum_abs_lin = 0.0
    sum_abs_pchip = 0.0
    sum_abs_log = 0.0
    sum_abs_inv = 0.0
    sum_sq_mu_lin = 0.0
    sum_sq_mu_pchip = 0.0
    sum_sq_mu_log = 0.0
    sum_sq_mu_inv = 0.0
    max_abs_lin = 0.0
    max_abs_pchip = 0.0
    max_abs_log = 0.0
    max_abs_inv = 0.0

    for z_idx in range(n_z):
        z_cur = z_grid[z_idx]
        for eta, eta_w in zip(eta_nodes, eta_weights):
            z_next = rho * z_cur + eta
            iz_lo = int((z_next - z_grid[0]) / dz)
            iz_lo = max(0, min(iz_lo, n_z - 2))
            frac_z = (z_next - z_grid[iz_lo]) / dz
            frac_z = max(0.0, min(1.0, frac_z))
            use_cubic = (iz_lo >= 1) and (iz_lo + 2 < n_z)

            if path_filter == "cubic" and not use_cubic:
                continue
            if path_filter == "edge" and use_cubic:
                continue

            v0 = C_next_view[:, iz_lo, :, :]
            v1 = C_next_view[:, iz_lo + 1, :, :]
            c_linear = (1.0 - frac_z) * v0 + frac_z * v1
            if use_cubic:
                c_pchip = _pchip_interp(
                    C_next_view[:, iz_lo - 1, :, :],
                    v0,
                    v1,
                    C_next_view[:, iz_lo + 2, :, :],
                    frac_z,
                )
            else:
                c_pchip = c_linear
            c_loglin = np.exp((1.0 - frac_z) * np.log(v0) + frac_z * np.log(v1))
            c_invlin = 1.0 / ((1.0 - frac_z) / v0 + frac_z / v1)

            if use_cubic:
                c_current = _catmull_rom(
                    C_next_view[:, iz_lo - 1, :, :],
                    v0,
                    v1,
                    C_next_view[:, iz_lo + 2, :, :],
                    frac_z,
                )
            else:
                c_current = c_linear

            diff_lin = (c_linear - c_current) / c_current
            diff_pchip = (c_pchip - c_current) / c_current
            diff_log = (c_loglin - c_current) / c_current
            diff_inv = (c_invlin - c_current) / c_current

            mu_ratio_lin = np.power(c_current / c_linear, gamma) - 1.0
            mu_ratio_pchip = np.power(c_current / c_pchip, gamma) - 1.0
            mu_ratio_log = np.power(c_current / c_loglin, gamma) - 1.0
            mu_ratio_inv = np.power(c_current / c_invlin, gamma) - 1.0

            count = diff_lin.size
            w = float(eta_w)
            block_weight = w * count
            tot_w_count += block_weight
            if use_cubic:
                tot_w_count_cubic += block_weight

            abs_lin = np.abs(diff_lin)
            abs_pchip = np.abs(diff_pchip)
            abs_log = np.abs(diff_log)
            abs_inv = np.abs(diff_inv)

            sum_sq_lin += w * float(np.sum(diff_lin * diff_lin, dtype=np.float64))
            sum_sq_pchip += w * float(np.sum(diff_pchip * diff_pchip, dtype=np.float64))
            sum_sq_log += w * float(np.sum(diff_log * diff_log, dtype=np.float64))
            sum_sq_inv += w * float(np.sum(diff_inv * diff_inv, dtype=np.float64))

            sum_abs_lin += w * float(np.sum(abs_lin, dtype=np.float64))
            sum_abs_pchip += w * float(np.sum(abs_pchip, dtype=np.float64))
            sum_abs_log += w * float(np.sum(abs_log, dtype=np.float64))
            sum_abs_inv += w * float(np.sum(abs_inv, dtype=np.float64))

            sum_sq_mu_lin += w * float(np.sum(mu_ratio_lin * mu_ratio_lin, dtype=np.float64))
            sum_sq_mu_pchip += w * float(np.sum(mu_ratio_pchip * mu_ratio_pchip, dtype=np.float64))
            sum_sq_mu_log += w * float(np.sum(mu_ratio_log * mu_ratio_log, dtype=np.float64))
            sum_sq_mu_inv += w * float(np.sum(mu_ratio_inv * mu_ratio_inv, dtype=np.float64))

            max_abs_lin = max(max_abs_lin, float(np.max(abs_lin)))
            max_abs_pchip = max(max_abs_pchip, float(np.max(abs_pchip)))
            max_abs_log = max(max_abs_log, float(np.max(abs_log)))
            max_abs_inv = max(max_abs_inv, float(np.max(abs_inv)))

    if tot_w_count == 0.0:
        return None

    return {
        "n_eval_eff": int(round(tot_w_count)),
        "cubic_share_pct": 100.0 * tot_w_count_cubic / tot_w_count,
        "rmse_lin_pct": 100.0 * np.sqrt(sum_sq_lin / tot_w_count),
        "rmse_pchip_pct": 100.0 * np.sqrt(sum_sq_pchip / tot_w_count),
        "rmse_log_pct": 100.0 * np.sqrt(sum_sq_log / tot_w_count),
        "rmse_inv_pct": 100.0 * np.sqrt(sum_sq_inv / tot_w_count),
        "mae_lin_pct": 100.0 * (sum_abs_lin / tot_w_count),
        "mae_pchip_pct": 100.0 * (sum_abs_pchip / tot_w_count),
        "mae_log_pct": 100.0 * (sum_abs_log / tot_w_count),
        "mae_inv_pct": 100.0 * (sum_abs_inv / tot_w_count),
        "mu_rmse_lin_pct": 100.0 * np.sqrt(sum_sq_mu_lin / tot_w_count),
        "mu_rmse_pchip_pct": 100.0 * np.sqrt(sum_sq_mu_pchip / tot_w_count),
        "mu_rmse_log_pct": 100.0 * np.sqrt(sum_sq_mu_log / tot_w_count),
        "mu_rmse_inv_pct": 100.0 * np.sqrt(sum_sq_mu_inv / tot_w_count),
        "maxabs_lin_pct": 100.0 * max_abs_lin,
        "maxabs_pchip_pct": 100.0 * max_abs_pchip,
        "maxabs_log_pct": 100.0 * max_abs_log,
        "maxabs_inv_pct": 100.0 * max_abs_inv,
    }


def _reshape_state_axis(C_view, state_grid_sizes):
    """Reshape flat state axis to explicit 3D state-grid axes."""
    n0, n1, n2 = [int(x) for x in state_grid_sizes]
    expected = n0 * n1 * n2
    if C_view.shape[2] != expected:
        raise ValueError(
            "state_grid_sizes do not match flat state dimension: "
            f"{state_grid_sizes} -> {expected}, but C_view.shape[2]={C_view.shape[2]}"
        )
    return C_view.reshape(C_view.shape[0], C_view.shape[1], n0, n1, n2, C_view.shape[3])


def _compute_state_axis_metrics(C_view, state_grid_sizes, axis):
    """Compute curvature and leave-one-out metrics along one state-grid axis.

    The diagnostic works in the interpolation coordinates used by the solver's
    trilinear lookup, where each axis is uniformly spaced.
    """
    arr = _reshape_state_axis(C_view, state_grid_sizes)
    axis_len = int(state_grid_sizes[axis])
    if axis_len < 3:
        raise ValueError(f"Need axis length >= 3 for state-axis diagnostics, got {axis_len}")

    arr = np.moveaxis(arr, 2 + axis, 2)  # -> (age, z, axis, other0, other1, w)

    c_mid = arr[:, :, 1:-1, :, :, :]
    curv = np.abs(arr[:, :, 2:, :, :, :] - 2.0 * c_mid + arr[:, :, :-2, :, :, :]) / c_mid
    curv_flat = np.ravel(curv)

    truth = c_mid
    left = arr[:, :, :-2, :, :, :]
    right = arr[:, :, 2:, :, :, :]

    linear_pred = 0.5 * (left + right)
    log_linear_pred = np.exp(0.5 * (np.log(left) + np.log(right)))
    inv_linear_pred = 1.0 / (0.5 * (1.0 / left + 1.0 / right))

    err_linear = (linear_pred - truth) / truth
    err_log = (log_linear_pred - truth) / truth
    err_inv = (inv_linear_pred - truth) / truth

    rmse_linear = _rmse(err_linear)
    rmse_log = _rmse(err_log)
    rmse_inv = _rmse(err_inv)

    return {
        "n_curv": int(curv_flat.size),
        "n_loo": int(truth.size),
        "curv_med_pct": 100.0 * float(np.median(curv_flat)),
        "curv_p95_pct": 100.0 * float(np.quantile(curv_flat, 0.95)),
        "curv_p99_pct": 100.0 * float(np.quantile(curv_flat, 0.99)),
        "curv_share_gt_0p5": 100.0 * float(np.mean(curv_flat > 0.005)),
        "curv_share_gt_5": 100.0 * float(np.mean(curv_flat > 0.05)),
        "midpoint_p95_pct": 12.5 * float(np.quantile(curv_flat, 0.95)),
        "rmse_linear_pct": 100.0 * rmse_linear,
        "rmse_loglin_pct": 100.0 * rmse_log,
        "rmse_invlin_pct": 100.0 * rmse_inv,
        "ratio_lin_loglin": rmse_linear / rmse_log if rmse_log > 0.0 else np.nan,
        "loglin_win_pct": 100.0 * float(np.mean(np.abs(err_log) < np.abs(err_linear))),
    }


def _format_table(rows):
    headers = [
        "age",
        "wealth",
        "curv_p95%",
        "mid_lin_p95%",
        "lin_rmse%",
        "cubic_rmse%",
        "loglin_rmse%",
        "invlin_rmse%",
        "lin/cubic",
        "cubic/loglin",
        "cubic_win%",
        "loglin<cubic%",
    ]
    lines = []
    widths = [max(len(h), 12) for h in headers]
    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["curv_p95_pct"]:.3f}',
            f'{row["midpoint_p95_pct"]:.3f}',
            f'{row["rmse_linear_pct"]:.3f}',
            f'{row["rmse_cubic_pct"]:.3f}',
            f'{row["rmse_loglin_pct"]:.3f}',
            f'{row["rmse_invlin_pct"]:.3f}',
            f'{row["ratio_lin_cubic"]:.2f}',
            f'{row["ratio_cubic_loglin"]:.2f}',
            f'{row["cubic_win_pct"]:.1f}',
            f'{row["loglin_beats_cubic_pct"]:.1f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["curv_p95_pct"]:.3f}',
            f'{row["midpoint_p95_pct"]:.3f}',
            f'{row["rmse_linear_pct"]:.3f}',
            f'{row["rmse_cubic_pct"]:.3f}',
            f'{row["rmse_loglin_pct"]:.3f}',
            f'{row["rmse_invlin_pct"]:.3f}',
            f'{row["ratio_lin_cubic"]:.2f}',
            f'{row["ratio_cubic_loglin"]:.2f}',
            f'{row["cubic_win_pct"]:.1f}',
            f'{row["loglin_beats_cubic_pct"]:.1f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def _format_actual_table(rows):
    headers = [
        "segment",
        "path",
        "cubic_share%",
        "lin_rmse%",
        "log_rmse%",
        "inv_rmse%",
        "lin_mae%",
        "log_mae%",
        "mu_lin_rmse%",
        "mu_log_rmse%",
        "max|lin|%",
        "max|log|%",
    ]
    widths = [max(len(h), 12) for h in headers]
    lines = []

    for row in rows:
        vals = [
            row["segment"],
            row["path"],
            f'{row["cubic_share_pct"]:.1f}',
            f'{row["rmse_lin_pct"]:.3f}',
            f'{row["rmse_log_pct"]:.3f}',
            f'{row["rmse_inv_pct"]:.3f}',
            f'{row["mae_lin_pct"]:.3f}',
            f'{row["mae_log_pct"]:.3f}',
            f'{row["mu_rmse_lin_pct"]:.3f}',
            f'{row["mu_rmse_log_pct"]:.3f}',
            f'{row["maxabs_lin_pct"]:.3f}',
            f'{row["maxabs_log_pct"]:.3f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["segment"],
            row["path"],
            f'{row["cubic_share_pct"]:.1f}',
            f'{row["rmse_lin_pct"]:.3f}',
            f'{row["rmse_log_pct"]:.3f}',
            f'{row["rmse_inv_pct"]:.3f}',
            f'{row["mae_lin_pct"]:.3f}',
            f'{row["mae_log_pct"]:.3f}',
            f'{row["mu_rmse_lin_pct"]:.3f}',
            f'{row["mu_rmse_log_pct"]:.3f}',
            f'{row["maxabs_lin_pct"]:.3f}',
            f'{row["maxabs_log_pct"]:.3f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def _format_state_axis_table(rows):
    headers = [
        "axis",
        "age",
        "curv_p95%",
        "mid_lin_p95%",
        "lin_rmse%",
        "loglin_rmse%",
        "invlin_rmse%",
        "lin/loglin",
        "loglin_win%",
    ]
    widths = [max(len(h), 12) for h in headers]
    lines = []

    for row in rows:
        vals = [
            row["axis"],
            row["age"],
            f'{row["curv_p95_pct"]:.3f}',
            f'{row["midpoint_p95_pct"]:.3f}',
            f'{row["rmse_linear_pct"]:.3f}',
            f'{row["rmse_loglin_pct"]:.3f}',
            f'{row["rmse_invlin_pct"]:.3f}',
            f'{row["ratio_lin_loglin"]:.2f}',
            f'{row["loglin_win_pct"]:.1f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["axis"],
            row["age"],
            f'{row["curv_p95_pct"]:.3f}',
            f'{row["midpoint_p95_pct"]:.3f}',
            f'{row["rmse_linear_pct"]:.3f}',
            f'{row["rmse_loglin_pct"]:.3f}',
            f'{row["rmse_invlin_pct"]:.3f}',
            f'{row["ratio_lin_loglin"]:.2f}',
            f'{row["loglin_win_pct"]:.1f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def _format_overshoot_table(rows):
    headers = [
        "age",
        "wealth",
        "over% ",
        "over>0.1% ",
        "over>1% ",
        "mean_over% ",
        "max_over% ",
    ]
    widths = [max(len(h), 12) for h in headers]
    lines = []

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["overshoot_share_pct"]:.3f}',
            f'{row["overshoot_gt_0p1_pct"]:.3f}',
            f'{row["overshoot_gt_1_pct"]:.3f}',
            f'{row["overshoot_mean_pct"]:.3f}',
            f'{row["overshoot_max_pct"]:.3f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["overshoot_share_pct"]:.3f}',
            f'{row["overshoot_gt_0p1_pct"]:.3f}',
            f'{row["overshoot_gt_1_pct"]:.3f}',
            f'{row["overshoot_mean_pct"]:.3f}',
            f'{row["overshoot_max_pct"]:.3f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def _format_mpc_scan_table(rows):
    headers = [
        "age",
        "wealth",
        "node_oob%",
        "shape_over%",
        "shape>0.1%",
        "clip_bind%",
        "clip>0.1%",
        "mean_clip_pp",
        "max_clip_pp",
    ]
    widths = [max(len(h), 12) for h in headers]
    lines = []

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["node_oob_pct"]:.3f}',
            f'{row["shape_over_pct"]:.3f}',
            f'{row["shape_over_gt_0p1_pct"]:.3f}',
            f'{row["clip_bind_pct"]:.3f}',
            f'{row["clip_bind_gt_0p1_pct"]:.3f}',
            f'{row["clip_mean_abs_pctpt"]:.3f}',
            f'{row["clip_max_abs_pctpt"]:.3f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["node_oob_pct"]:.3f}',
            f'{row["shape_over_pct"]:.3f}',
            f'{row["shape_over_gt_0p1_pct"]:.3f}',
            f'{row["clip_bind_pct"]:.3f}',
            f'{row["clip_bind_gt_0p1_pct"]:.3f}',
            f'{row["clip_mean_abs_pctpt"]:.3f}',
            f'{row["clip_max_abs_pctpt"]:.3f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def _format_pchip_table(rows):
    headers = [
        "age",
        "wealth",
        "pchip_rmse%",
        "lin/pchip",
        "cubic/pchip",
        "pchip<lin%",
        "pchip<cubic%",
    ]
    widths = [max(len(h), 12) for h in headers]
    lines = []

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["rmse_pchip_pct"]:.3f}',
            f'{row["ratio_lin_pchip"]:.2f}',
            f'{row["ratio_cubic_pchip"]:.2f}',
            f'{row["pchip_win_pct"]:.1f}',
            f'{row["pchip_beats_cubic_pct"]:.1f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["age"],
            row["wealth"],
            f'{row["rmse_pchip_pct"]:.3f}',
            f'{row["ratio_lin_pchip"]:.2f}',
            f'{row["ratio_cubic_pchip"]:.2f}',
            f'{row["pchip_win_pct"]:.1f}',
            f'{row["pchip_beats_cubic_pct"]:.1f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def _format_actual_pchip_table(rows):
    headers = [
        "segment",
        "path",
        "pchip_rmse%",
        "pchip_mae%",
        "mu_pchip_rmse%",
        "max|pchip|%",
    ]
    widths = [max(len(h), 12) for h in headers]
    lines = []

    for row in rows:
        vals = [
            row["segment"],
            row["path"],
            f'{row["rmse_pchip_pct"]:.3f}',
            f'{row["mae_pchip_pct"]:.3f}',
            f'{row["mu_rmse_pchip_pct"]:.3f}',
            f'{row["maxabs_pchip_pct"]:.3f}',
        ]
        for i, val in enumerate(vals):
            widths[i] = max(widths[i], len(val))

    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header)
    lines.append(rule)

    for row in rows:
        vals = [
            row["segment"],
            row["path"],
            f'{row["rmse_pchip_pct"]:.3f}',
            f'{row["mae_pchip_pct"]:.3f}',
            f'{row["mu_rmse_pchip_pct"]:.3f}',
            f'{row["maxabs_pchip_pct"]:.3f}',
        ]
        lines.append("  ".join(val.ljust(widths[i]) for i, val in enumerate(vals)))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle",
        nargs="?",
        default="saved_runs/constrained_grid5x5x5_nz11",
        help="Path to saved policy bundle directory (default: %(default)s)",
    )
    parser.add_argument(
        "--include-retirement",
        action="store_true",
        help="Include retirement ages in the bucket table (default: working ages only).",
    )
    parser.add_argument(
        "--wealth-buckets",
        type=int,
        default=4,
        help="Number of contiguous wealth-grid buckets (default: %(default)s)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save the aggregate results as JSON.",
    )
    args = parser.parse_args()

    bundle = Path(args.bundle)
    C_mat, _, _, _, metadata = load_policy_bundle(bundle)
    n_age, n_z, n_state, n_w = C_mat.shape

    if n_z < 5:
        raise ValueError(f"Bundle has n_z={n_z}; need at least 5 for cubic leave-one-out.")

    ages, retire_age, wealth_grid = _extract_configs(metadata, n_age, n_w)
    solved_age_mask = _extract_solved_age_mask(metadata, n_age)

    active_age_idx = np.arange(n_age, dtype=int)
    if solved_age_mask is not None and np.any(solved_age_mask):
        active_age_idx = np.flatnonzero(solved_age_mask)

    ages_active = ages[active_age_idx]
    C_active = C_mat[active_age_idx, :, :, :]

    working_only = not args.include_retirement
    if retire_age is not None and not np.any(ages_active < int(retire_age)):
        working_only = False

    age_buckets = _build_age_buckets(
        ages=ages_active,
        retire_age=retire_age,
        working_only=working_only,
    )
    wealth_buckets = _build_wealth_buckets(wealth_grid, args.wealth_buckets)
    z_process = _reconstruct_z_process(metadata, n_z_fallback=n_z)
    state_grid_sizes = _extract_state_grid_sizes(metadata)
    state_axis_labels = _infer_state_axis_labels(metadata) if state_grid_sizes is not None else None

    summary_rows = []

    # Global row first.
    global_age_slice = slice(age_buckets[0][1].start, age_buckets[-1][1].stop)
    global_wealth_slice = slice(0, n_w)
    global_metrics = _compute_metrics(C_active[global_age_slice, :, :, global_wealth_slice])
    global_metrics["age"] = "ALL"
    global_metrics["wealth"] = "ALL"
    summary_rows.append(global_metrics)

    # Age-only rows.
    for age_label, age_slice in age_buckets:
        metrics = _compute_metrics(C_active[age_slice, :, :, :])
        metrics["age"] = age_label
        metrics["wealth"] = "ALL"
        summary_rows.append(metrics)

    bucket_rows = []
    for age_label, age_slice in age_buckets:
        for wealth_label, wealth_slice in wealth_buckets:
            metrics = _compute_metrics(C_active[age_slice, :, :, wealth_slice])
            metrics["age"] = age_label
            metrics["wealth"] = wealth_label
            bucket_rows.append(metrics)

    overshoot_rows = []
    overshoot_global = _compute_cubic_overshoot_metrics(C_active[global_age_slice, :, :, global_wealth_slice])
    overshoot_global["age"] = "ALL"
    overshoot_global["wealth"] = "ALL"
    overshoot_rows.append(overshoot_global)

    for age_label, age_slice in age_buckets:
        metrics = _compute_cubic_overshoot_metrics(C_active[age_slice, :, :, :])
        metrics["age"] = age_label
        metrics["wealth"] = "ALL"
        overshoot_rows.append(metrics)

    overshoot_bucket_rows = []
    for age_label, age_slice in age_buckets:
        for wealth_label, wealth_slice in wealth_buckets:
            metrics = _compute_cubic_overshoot_metrics(C_active[age_slice, :, :, wealth_slice])
            metrics["age"] = age_label
            metrics["wealth"] = wealth_label
            overshoot_bucket_rows.append(metrics)

    mpc_scan_rows = []
    mpc_global = _compute_mpc_cubic_scan_metrics(C_active[global_age_slice, :, :, global_wealth_slice], wealth_grid)
    mpc_global["age"] = "ALL"
    mpc_global["wealth"] = "ALL"
    mpc_scan_rows.append(mpc_global)

    for age_label, age_slice in age_buckets:
        metrics = _compute_mpc_cubic_scan_metrics(C_active[age_slice, :, :, :], wealth_grid)
        metrics["age"] = age_label
        metrics["wealth"] = "ALL"
        mpc_scan_rows.append(metrics)

    mpc_scan_bucket_rows = []
    for age_label, age_slice in age_buckets:
        for wealth_label, wealth_slice in wealth_buckets:
            wealth_grid_bucket = wealth_grid[wealth_slice]
            metrics = _compute_mpc_cubic_scan_metrics(
                C_active[age_slice, :, :, wealth_slice],
                wealth_grid_bucket,
            )
            metrics["age"] = age_label
            metrics["wealth"] = wealth_label
            mpc_scan_bucket_rows.append(metrics)

    print("=" * 110)
    print("Z-Interpolation Diagnostics")
    print("=" * 110)
    print(f"Bundle        : {bundle}")
    print(f"C_mat shape   : {C_mat.shape}   (n_state={n_state})")
    if solved_age_mask is not None and np.any(solved_age_mask):
        print(
            f"Solved ages   : {int(ages_active[0])}-{int(ages_active[-1])} "
            f"({active_age_idx.size}/{n_age} slices)"
        )
    if retire_age is not None:
        if working_only:
            mode = "working ages only"
        elif solved_age_mask is not None and np.any(solved_age_mask):
            mode = "solved ages only"
        else:
            mode = "all ages"
        print(f"Age filter    : {mode}   (retire_age={retire_age})")
    else:
        if solved_age_mask is not None and np.any(solved_age_mask):
            print("Age filter    : metadata missing retire_age; using solved-age buckets")
        else:
            print("Age filter    : metadata missing retire_age; using raw time-index buckets")
    print(
        "Metric note   : midpoint linear relative error is approximately curv_p95% / 8 "
        "(reported as mid_lin_p95%)."
    )
    print(
        "Interpretation: lin/cubic > 1 means Catmull-Rom improves leave-one-out accuracy; "
        "cubic/loglin > 1 means linear-in-log(c) beats cubic-in-c."
    )

    print("\nSummary")
    print(_format_table(summary_rows))

    print("\nAge x Wealth Buckets")
    print(_format_table(bucket_rows))

    print("\nPCHIP Leave-One-Out Summary")
    print(
        "These rows compare a local shape-preserving cubic Hermite / PCHIP-style predictor "
        "against the same held-out z-node truth used above.\n"
        "`lin/pchip > 1` means PCHIP improves on linear; `cubic/pchip > 1` means it beats "
        "the current Catmull-Rom leave-one-out predictor."
    )
    print(_format_pchip_table(summary_rows))

    print("\nPCHIP Age x Wealth Buckets")
    print(_format_pchip_table(bucket_rows))

    print("\nCubic Overshoot Summary")
    print(
        "These rows count interior z-intervals where the Catmull-Rom curve leaves the "
        "closed interval spanned by its two endpoint node values.\n"
        "`over%` is the fraction of intervals with any overshoot; "
        "`mean_over%` and `max_over%` scale the overshoot by the average endpoint level."
    )
    print(_format_overshoot_table(overshoot_rows))

    print("\nCubic Overshoot Age x Wealth Buckets")
    print(_format_overshoot_table(overshoot_bucket_rows))

    print("\nMPC Cubic Scan")
    print(
        "These rows rebuild the nodewise wealth slopes used as `mpc_z*` in the cubic branch of "
        "_interp_z_wealth and scan the interior z-cubics they define.\n"
        "`node_oob%` is the share of node slopes outside [0,1]. "
        "`clip_bind%` is the share of interior cubic intervals whose interior extrema leave [0,1], "
        "so the solver's MPC clip would bind inside the interval."
    )
    print(_format_mpc_scan_table(mpc_scan_rows))

    print("\nMPC Cubic Scan Age x Wealth Buckets")
    print(_format_mpc_scan_table(mpc_scan_bucket_rows))

    state_axis_rows = []
    if state_grid_sizes is not None:
        print("\nState-Axis Summary")
        print(
            "These diagnostics reshape the flat financial-state index back to its 3D grid and "
            "run the same local-curvature and centered leave-one-out tests along each axis.\n"
            "For cholesky grids, the axes are transformed interpolation coordinates; otherwise "
            "they correspond to the stored state-variable axes."
        )

        for axis, axis_label in enumerate(state_axis_labels):
            if int(state_grid_sizes[axis]) < 3:
                continue

            metrics = _compute_state_axis_metrics(C_active[global_age_slice, :, :, :], state_grid_sizes, axis)
            metrics["axis"] = axis_label
            metrics["age"] = "ALL"
            state_axis_rows.append(metrics)

            for age_label, age_slice in age_buckets:
                metrics = _compute_state_axis_metrics(C_active[age_slice, :, :, :], state_grid_sizes, axis)
                metrics["axis"] = axis_label
                metrics["age"] = age_label
                state_axis_rows.append(metrics)

        print(_format_state_axis_table(state_axis_rows))
    else:
        print(
            "\nState-Axis Summary\n"
            "Skipped: bundle metadata is missing `state_grid_sizes`, so the flat state index "
            "cannot be reshaped back to explicit axes."
        )

    actual_rows = []
    if z_process is not None and retire_age is not None:
        run_config = metadata.get("run_config", {})
        gamma = float(run_config.get("base_config", {}).get("gamma", 3.0))
        current_age_idx = active_age_idx[active_age_idx < n_age - 1]
        current_age_idx = current_age_idx[ages[current_age_idx] < int(retire_age)]
        ages_current = ages[current_age_idx]
        age_labels_are_real = not np.array_equal(ages_current, np.arange(ages_current.size))

        # Global rows over all current working ages, using continuation slices t+1.
        if current_age_idx.size > 0:
            C_next_all = C_mat[current_age_idx + 1, :, :, :]
            for path in ("all", "cubic", "edge"):
                metrics = _compute_actual_znext_metrics(C_next_all, z_process, gamma, path_filter=path)
                metrics["segment"] = "ALL"
                metrics["path"] = path.upper()
                actual_rows.append(metrics)

            # Per-age-bucket rows, all paths only.
            for age_label, age_slice in age_buckets:
                idx = active_age_idx[np.arange(age_slice.start, age_slice.stop, dtype=int)]
                idx = idx[idx < n_age - 1]
                idx = idx[ages[idx] < int(retire_age)]
                if idx.size == 0:
                    continue
                C_next_view = C_mat[idx + 1, :, :, :]
                metrics = _compute_actual_znext_metrics(C_next_view, z_process, gamma, path_filter="all")
                metrics["segment"] = age_label
                metrics["path"] = "ALL"
                actual_rows.append(metrics)

            print("\nActual Off-Grid z_next Comparison")
            print(
                "This section evaluates the continuation policy at the exact off-grid "
                "z_next = rho*z + eta points used in the working-age FOC.\n"
                "Reported errors are relative differences versus the current live scheme "
                "(Catmull-Rom on interior intervals, linear fallback on edges)."
            )
            print(_format_actual_table(actual_rows))

            print("\nActual Off-Grid PCHIP Comparison")
            print(
                "These rows isolate the PCHIP-style candidate from the table above. "
                "Errors are relative to the current live scheme at the exact off-grid "
                "working-age z_next cloud."
            )
            print(_format_actual_pchip_table(actual_rows))
        else:
            print(
                "\nActual Off-Grid z_next Comparison\n"
                "Skipped: no solved working-age slices are available in this bundle."
            )
    else:
        print(
            "\nActual Off-Grid z_next Comparison\n"
            "Skipped: bundle metadata is missing enough z-process information or "
            "retirement timing to reconstruct the exact working-age z_next cloud."
        )

    if args.json_out is not None:
        payload = {
            "bundle": str(bundle),
            "shape": list(C_mat.shape),
            "summary_rows": summary_rows,
            "bucket_rows": bucket_rows,
            "overshoot_rows": overshoot_rows,
            "overshoot_bucket_rows": overshoot_bucket_rows,
            "mpc_scan_rows": mpc_scan_rows,
            "mpc_scan_bucket_rows": mpc_scan_bucket_rows,
            "state_axis_rows": state_axis_rows,
            "actual_rows": actual_rows,
        }
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON results to {out_path}")


if __name__ == "__main__":
    main()

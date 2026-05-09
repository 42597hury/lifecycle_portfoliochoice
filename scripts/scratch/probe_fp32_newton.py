"""Scratch probe for fp32 Newton arithmetic.

This script intentionally does not modify production solver code. It runs the
same tiny lifecycle window twice:

1. Current mixed-precision baseline: gather/interp may be fp32, FOC/Newton fp64.
2. Monkey-patched fp32 Newton: CCV return, FOC sums, Jacobian terms, and the
   Newton residual/Jacobian values are produced in fp32.

The output is a compact JSON file for the handoff report.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


# Force the single-device vmap path before lifecycle imports JAX. This makes the
# CPU-side precision comparison reproducible and avoids pmap padding artifacts.
os.environ.setdefault("LIFECYCLE_DISABLE_VIRTUAL_CPUS", "1")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from configs._canonical import BASE_CONFIG, CANONICAL_SOLVER  # noqa: E402
from lifecycle.model import DiscretizationConfig, SolveControl  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.solver import run_lifecycle_solver  # noqa: E402
from lifecycle.var import build_nominal_system1_var_config_hardcoded  # noqa: E402
import lifecycle.solver as solver_mod  # noqa: E402


AGE_RE = re.compile(r"^\s*(?P<age>\d+)\s+(?P<phase>RETIRE|WORK)\s+(?P<elapsed>[0-9.]+)s")


class Tee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self._stream = stream

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._stream.flush()
        return super().write(s)

    def flush(self) -> None:
        self._stream.flush()
        super().flush()


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if hasattr(x, "_asdict"):
        return _jsonable(x._asdict())
    return x


def _f32(x):
    return jnp.asarray(x, dtype=jnp.float32)


def _maybe_return(x, return_foc64: bool):
    return x.astype(jnp.float64) if return_foc64 else x


def _bequest_mu_and_mup_f32(W, A, gamma, b_bar, delta):
    W = _f32(W)
    A = _f32(A)
    gamma = _f32(gamma)
    b_bar = _f32(b_bar)
    delta = _f32(delta)
    C_bar = W / A + delta
    mu = b_bar * C_bar ** (-gamma) / A
    mup = -gamma * mu / (A * C_bar)
    return mu, mup


def _ccv_log_return_and_grad_f32(alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
                                  sigma2_xr, sigma2_xb, sigma_xrxb):
    alpha_s = _f32(alpha_s)
    alpha_b = _f32(alpha_b)
    log_R_bill = _f32(log_R_bill)
    log_x_s = _f32(log_x_s)
    log_x_b = _f32(log_x_b)
    sigma2_xr = _f32(sigma2_xr)
    sigma2_xb = _f32(sigma2_xb)
    sigma_xrxb = _f32(sigma_xrxb)

    r_p = (
        log_R_bill
        + alpha_s * log_x_s
        + alpha_b * log_x_b
        + _f32(0.5) * (alpha_s * sigma2_xr + alpha_b * sigma2_xb)
        - _f32(0.5) * (
            alpha_s * alpha_s * sigma2_xr
            + _f32(2.0) * alpha_s * alpha_b * sigma_xrxb
            + alpha_b * alpha_b * sigma2_xb
        )
    )
    R_p = jnp.exp(r_p)
    dr_da_s = log_x_s + sigma2_xr * (_f32(0.5) - alpha_s) - alpha_b * sigma_xrxb
    dr_da_b = log_x_b + sigma2_xb * (_f32(0.5) - alpha_b) - alpha_s * sigma_xrxb
    return R_p, dr_da_s, dr_da_b


def _make_terminal_foc_jac_ccv_f32(return_foc64: bool):
    def terminal_foc_jac_ccv_f32(
        alpha_s, alpha_b, s_val, A_is,
        log_R_bill, log_x_s, log_x_b,
        weight_kv_kr,
        sigma2_xr, sigma2_xb, sigma_xrxb,
        gamma, b_bar, delta,
    ):
        R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad_f32(
            alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
            sigma2_xr, sigma2_xb, sigma_xrxb,
        )
        s_val = _f32(s_val)
        weight_kv_kr = _f32(weight_kv_kr)
        sigma2_xr = _f32(sigma2_xr)
        sigma2_xb = _f32(sigma2_xb)
        sigma_xrxb = _f32(sigma_xrxb)

        sR_p = s_val * R_p
        mu, mup = _bequest_mu_and_mup_f32(sR_p, A_is, gamma, b_bar, delta)
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
        return tuple(_maybe_return(x, return_foc64) for x in (foc_s, foc_b, J_ss, J_bb, J_sb, V_dot))

    return terminal_foc_jac_ccv_f32


def _make_retirement_foc_jac_ccv_f32(return_foc64: bool):
    def retirement_foc_jac_ccv_f32(
        alpha_s, alpha_b, s_val, psi_z,
        log_R_bill, log_x_s, log_x_b, weight_kv_kr,
        w_corners,
        c_corners_at_z, wealth_grid,
        pension_next_z,
        A_is,
        sigma2_xr, sigma2_xb, sigma_xrxb,
        gamma, b_bar, delta, min_consumption,
        gather_dtype=jnp.float64,
    ):
        R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad_f32(
            alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
            sigma2_xr, sigma2_xb, sigma_xrxb,
        )
        s_val = _f32(s_val)
        psi_z = _f32(psi_z)
        pension_next_z = _f32(pension_next_z)
        weight_kv_kr = _f32(weight_kv_kr)
        sigma2_xr = _f32(sigma2_xr)
        sigma2_xb = _f32(sigma2_xb)
        sigma_xrxb = _f32(sigma_xrxb)

        sR_p = s_val * R_p
        x_next = sR_p + pension_next_z
        mu_bq, mup_bq = _bequest_mu_and_mup_f32(sR_p, A_is, gamma, b_bar, delta)

        wealth_grid_g = solver_mod._cast_for_gather(wealth_grid, gather_dtype)
        c_corners_at_z_g = solver_mod._cast_for_gather(c_corners_at_z, gather_dtype)
        w_corners_g = solver_mod._cast_for_gather(w_corners, gather_dtype)

        n_w = wealth_grid_g.shape[0]

        def per_kv_kr(x_scalar, c_kv, w_kv):
            x_scalar_g = solver_mod._cast_for_gather(jnp.asarray(x_scalar), gather_dtype)
            iw = jnp.clip(jnp.searchsorted(wealth_grid_g, x_scalar_g, side="right") - 1, 0, n_w - 2)
            iw_hi = iw + 1
            x0 = wealth_grid_g[iw]
            x1 = wealth_grid_g[iw_hi]
            inv_dw = jnp.asarray(1.0, dtype=gather_dtype) / (x1 - x0)
            fw = (x_scalar_g - x0) * inv_dw

            c_w0 = c_kv[:, iw]
            c_w1 = c_kv[:, iw_hi]
            c_per_corner = c_w0 + (c_w1 - c_w0) * fw
            slope_per_corner = (c_w1 - c_w0) * inv_dw

            c_g = jnp.sum(w_kv * c_per_corner)
            mpc_g = jnp.sum(w_kv * slope_per_corner)
            c = jnp.maximum(c_g.astype(jnp.float64), min_consumption)
            mpc = jnp.clip(mpc_g.astype(jnp.float64), 0.0, 1.0)
            return c, mpc

        c_at_xn, mpc_at_xn = jax.vmap(
            lambda c_kv, w_kv, x_row: jax.vmap(per_kv_kr, in_axes=(0, None, None))(
                x_row, c_kv, w_kv
            ),
            in_axes=(0, 0, 0),
        )(c_corners_at_z_g, w_corners_g, x_next)

        c_at_xn = _f32(c_at_xn)
        mpc_at_xn = _f32(mpc_at_xn)
        gamma32 = _f32(gamma)
        mu_alive = c_at_xn ** (-gamma32)
        mup_alive = -gamma32 * mu_alive / c_at_xn * mpc_at_xn

        prob_death = _f32(1.0) - psi_z
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
        return tuple(_maybe_return(x, return_foc64) for x in (foc_s, foc_b, J_ss, J_bb, J_sb, e_sum))

    return retirement_foc_jac_ccv_f32


def _make_working_foc_jac_ccv_f32(return_foc64: bool):
    def working_foc_jac_ccv_f32(
        alpha_s, alpha_b, s_val, psi_z,
        log_R_bill, log_x_s, log_x_b, weight_kv_kr,
        w_corners,
        c_corners_T, wealth_grid,
        income_next_table,
        eta_iz_lo, eta_frac_z,
        eta_weights, eps_weights,
        A_is,
        sigma2_xr, sigma2_xb, sigma_xrxb,
        gamma, b_bar, delta, min_consumption,
        gather_dtype=jnp.float64,
    ):
        R_p, dr_da_s, dr_da_b = _ccv_log_return_and_grad_f32(
            alpha_s, alpha_b, log_R_bill, log_x_s, log_x_b,
            sigma2_xr, sigma2_xb, sigma_xrxb,
        )
        s_val = _f32(s_val)
        psi_z = _f32(psi_z)
        weight_kv_kr = _f32(weight_kv_kr)
        income_next_table = _f32(income_next_table)
        eta_weights = _f32(eta_weights)
        eps_weights = _f32(eps_weights)
        sigma2_xr = _f32(sigma2_xr)
        sigma2_xb = _f32(sigma2_xb)
        sigma_xrxb = _f32(sigma_xrxb)

        sR_p = s_val * R_p
        mu_bq, mup_bq = _bequest_mu_and_mup_f32(sR_p, A_is, gamma, b_bar, delta)
        prob_death = _f32(1.0) - psi_z

        dRp_das = R_p * dr_da_s
        dRp_dab = R_p * dr_da_b
        bequest_factor = weight_kv_kr * prob_death

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

        x_next = sR_p[:, :, None, None] + income_next_table[None, None, :, :]

        def per_kv(c_kv, w_kv, x_kv):
            def at_eta_eps(x_scalar, iz, fz):
                return solver_mod._interp_c_and_mpc_at_cell(
                    c_kv, w_kv, iz, fz, x_scalar, wealth_grid, min_consumption,
                    gather_dtype=gather_dtype,
                )

            def per_kr(x_kr):
                def per_keta(x_row, iz, fz):
                    return jax.vmap(at_eta_eps, in_axes=(0, None, None))(x_row, iz, fz)
                return jax.vmap(per_keta, in_axes=(0, 0, 0))(x_kr, eta_iz_lo, eta_frac_z)
            return jax.vmap(per_kr)(x_kv)

        c_at_xn, mpc_at_xn = jax.vmap(per_kv)(c_corners_T, w_corners, x_next)
        c_at_xn = _f32(c_at_xn)
        mpc_at_xn = _f32(mpc_at_xn)
        gamma32 = _f32(gamma)

        mu_alive = c_at_xn ** (-gamma32)
        mup_alive = -gamma32 * mu_alive / c_at_xn * mpc_at_xn
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
        J_sb_al = jnp.sum(jac_lin_al * dRp_dab_b * dRp_das_b + extra_sb_al)

        out = (
            foc_s_bq + foc_s_al,
            foc_b_bq + foc_b_al,
            J_ss_bq + J_ss_al,
            J_bb_bq + J_bb_al,
            J_sb_bq + J_sb_al,
            e_bq + e_al,
        )
        return tuple(_maybe_return(x, return_foc64) for x in out)

    return working_foc_jac_ccv_f32


@contextlib.contextmanager
def fp32_newton_patch(return_foc64: bool):
    originals = {
        "_ccv_log_return_and_grad": solver_mod._ccv_log_return_and_grad,
        "terminal_foc_jac_ccv": solver_mod.terminal_foc_jac_ccv,
        "retirement_foc_jac_ccv": solver_mod.retirement_foc_jac_ccv,
        "working_foc_jac_ccv": solver_mod.working_foc_jac_ccv,
    }
    solver_mod._ccv_log_return_and_grad = _ccv_log_return_and_grad_f32
    solver_mod.terminal_foc_jac_ccv = _make_terminal_foc_jac_ccv_f32(return_foc64)
    solver_mod.retirement_foc_jac_ccv = _make_retirement_foc_jac_ccv_f32(return_foc64)
    solver_mod.working_foc_jac_ccv = _make_working_foc_jac_ccv_f32(return_foc64)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(solver_mod, name, fn)


def build_probe_model(args):
    base_config = dict(BASE_CONFIG)
    base_config["start_age"] = args.youngest_age
    disc = DiscretizationConfig(
        n_wealth=args.n_wealth,
        wealth_min=0.13,
        wealth_max=200.0,
        n_savings=args.n_savings,
        state_grid_sizes=tuple(args.state_grid),
        state_grid_mode="cholesky",
        state_n_stds=(2.0, 2.25, 2.0, 2.25),
        n_z=args.n_z,
        n_eps_nodes=2,
        n_eta_nodes=2,
        n_ret_nodes_1d=(2, 2),
        n_state_quad_nodes=tuple(args.state_quad),
    )
    var_config = build_nominal_system1_var_config_hardcoded()
    model = build_model(base_config, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    sc = CANONICAL_SOLVER._replace(
        max_iter=args.max_iter,
        gather_precision="f32",
        use_fori_newton=True,
    )
    solve_control = None
    return model, pc, sc, solve_control


def parse_age_timings(log_text: str):
    rows = []
    for line in log_text.splitlines():
        m = AGE_RE.match(line)
        if m:
            rows.append({
                "age": int(m.group("age")),
                "phase": m.group("phase"),
                "elapsed_s": float(m.group("elapsed")),
            })
    prev = 0.0
    for row in rows:
        row["delta_s"] = row["elapsed_s"] - prev
        prev = row["elapsed_s"]
    return rows


def summarize_age_timings(rows):
    out = {"n_progress_ages": len(rows)}
    if not rows:
        return out
    deltas = np.asarray([r["delta_s"] for r in rows], dtype=float)
    out.update({
        "first_nonterminal_age_s": float(deltas[0]),
        "median_nonterminal_age_s": float(np.median(deltas)),
        "p95_nonterminal_age_s": float(np.percentile(deltas, 95)),
    })
    for phase in ["RETIRE", "WORK"]:
        vals = np.asarray([r["delta_s"] for r in rows if r["phase"] == phase], dtype=float)
        if vals.size:
            out[f"median_{phase.lower()}_age_s"] = float(np.median(vals))
            out[f"p95_{phase.lower()}_age_s"] = float(np.percentile(vals, 95))
            out[f"n_{phase.lower()}_ages"] = int(vals.size)
    return out


def run_one(label: str, model, pc, sc, solve_control, patch: bool, return_foc64: bool):
    print(f"\n=== {label} ===", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)
    print(f"Config: n_age={pc.n_age}, n_z={pc.n_z}, N_state={pc.N_state}, n_w={pc.n_w}, n_s={pc.n_s}", flush=True)
    tee = Tee(sys.stdout)
    t0 = time.time()
    ctx = fp32_newton_patch(return_foc64) if patch else contextlib.nullcontext()
    with ctx, contextlib.redirect_stdout(tee):
        C, S, B, diag = run_lifecycle_solver(
            model, pc, sc, verbose=1, solve_control=solve_control,
        )
    wall = time.time() - t0
    age_rows = parse_age_timings(tee.getvalue())
    solved = np.asarray(diag["solved_age_mask"], dtype=bool)
    C_s = C[solved]
    S_s = S[solved]
    B_s = B[solved]
    return {
        "label": label,
        "wall_s_outer": float(wall),
        "solver_wall_s": float(diag.get("wall_time_sec", np.nan)),
        "age_timing_rows": age_rows,
        "age_timing_summary": summarize_age_timings(age_rows),
        "diag": {
            "solve_status": diag.get("solve_status"),
            "n_ages_solved": int(diag.get("n_ages_solved", 0)),
            "youngest_solved_age": diag.get("youngest_solved_age"),
            "oldest_solved_age": diag.get("oldest_solved_age"),
            "total_newton_failures": int(diag.get("total_newton_failures", -1)),
            "worst_foc_resid": float(diag.get("worst_foc_resid", np.nan)),
            "newton_iter_histogram": _jsonable(diag.get("newton_iter_histogram")),
            "backtrack_iter_histogram": _jsonable(diag.get("backtrack_iter_histogram")),
        },
        "policy_summary": {
            "nan_C": int(np.isnan(C_s).sum()),
            "nan_S": int(np.isnan(S_s).sum()),
            "nan_B": int(np.isnan(B_s).sum()),
            "inf_C": int(np.isinf(C_s).sum()),
            "inf_S": int(np.isinf(S_s).sum()),
            "inf_B": int(np.isinf(B_s).sum()),
            "alpha_s_min": float(np.nanmin(S_s)),
            "alpha_s_max": float(np.nanmax(S_s)),
            "alpha_b_min": float(np.nanmin(B_s)),
            "alpha_b_max": float(np.nanmax(B_s)),
        },
        "arrays": (C, S, B, solved),
    }


def diff_stats(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {
        "max": float(np.nanmax(d)),
        "median": float(np.nanmedian(d)),
        "p95": float(np.nanpercentile(d, 95)),
        "p99": float(np.nanpercentile(d, 99)),
        "mean": float(np.nanmean(d)),
    }


def compare_runs(base, test):
    mask = np.asarray(base["arrays"][3]) & np.asarray(test["arrays"][3])
    C0, S0, B0, _ = base["arrays"]
    C1, S1, B1, _ = test["arrays"]
    C0 = C0[mask]
    S0 = S0[mask]
    B0 = B0[mask]
    C1 = C1[mask]
    S1 = S1[mask]
    B1 = B1[mask]
    alpha_abs = np.maximum(np.abs(S0 - S1), np.abs(B0 - B1))
    bill0 = 1.0 - S0 - B0
    bill1 = 1.0 - S1 - B1
    return {
        "n_common_solved_ages": int(mask.sum()),
        "n_policy_points": int(S0.size),
        "C_abs": diff_stats(C0, C1),
        "S_abs": diff_stats(S0, S1),
        "B_abs": diff_stats(B0, B1),
        "bill_abs": diff_stats(bill0, bill1),
        "alpha_component_abs": {
            "max": float(np.nanmax(alpha_abs)),
            "median": float(np.nanmedian(alpha_abs)),
            "p95": float(np.nanpercentile(alpha_abs, 95)),
            "p99": float(np.nanpercentile(alpha_abs, 99)),
        },
        "wall_ratio_solver": float(test["solver_wall_s"] / base["solver_wall_s"]),
        "speedup_solver": float(base["solver_wall_s"] / test["solver_wall_s"]),
        "wall_delta_solver_s": float(test["solver_wall_s"] - base["solver_wall_s"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--youngest-age", type=int, default=67)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--n-wealth", type=int, default=12)
    parser.add_argument("--n-savings", type=int, default=12)
    parser.add_argument("--n-z", type=int, default=3)
    parser.add_argument("--state-grid", type=int, nargs=4, default=(2, 3, 2, 3))
    parser.add_argument("--state-quad", type=int, nargs=4, default=(2, 3, 2, 3))
    parser.add_argument("--return-foc64", action="store_true",
                        help="Cast fp32 FOC/Jac outputs back to fp64 before Newton step.")
    parser.add_argument("--out", default="docs/scans/fp32_newton_probe_results.json")
    args = parser.parse_args()

    model, pc, sc, solve_control = build_probe_model(args)
    print(f"Probe output: {args.out}", flush=True)

    base = run_one("baseline_gather_f32_newton_f64", model, pc, sc, solve_control, patch=False, return_foc64=args.return_foc64)
    test = run_one("patched_gather_f32_newton_f32", model, pc, sc, solve_control, patch=True, return_foc64=args.return_foc64)
    cmp = compare_runs(base, test)

    for run in (base, test):
        run.pop("arrays", None)

    result = {
        "probe": {
            "date": "2026-05-07",
            "jax_devices": [str(d) for d in jax.devices()],
            "return_foc64": bool(args.return_foc64),
            "args": vars(args),
            "solver_config": _jsonable(sc),
            "disc_config": _jsonable(pc.disc_config),
            "solve_control": _jsonable(solve_control),
        },
        "baseline": _jsonable(base),
        "fp32_newton": _jsonable(test),
        "comparison": _jsonable(cmp),
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()

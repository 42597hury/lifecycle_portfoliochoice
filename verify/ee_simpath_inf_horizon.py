"""verify/ee_simpath_inf_horizon.py -- Sim-path consumption-Euler diagnostic for
inf-horizon (stationary) policy bundles.

The inf-horizon solver (lifecycle/inf_horizon_solver.py) writes a single
stationary slab of shape (n_z, N_state, n_w) for the no-mortality, no-pension,
no-bequest retirement-only Bellman fixed point. verify/ee_simpath.py expects a
4-D lifecycle bundle and a multi-period simulator, so it doesn't apply here.

This script samples N "households" from the stationary state distribution
(jointly: state from the VAR's stationary normal, z from the lifecycle z_grid's
stationary distribution, wealth log-uniform across [wealth_min, wealth_max]),
interpolates the stationary policy to get (c, alpha_s, alpha_b), then
re-evaluates retirement_foc_jac_ccv at each sampled state with bumped
quadrature (pc_eval). Reports consumption-Euler residuals
(``ee = 1 - (beta * e_sum)^(-1/gamma) / c``).

Inf-horizon overrides (matching inf_horizon_solver.run_infinite_horizon_solver):
    psi_z      = ones(n_z)        (no mortality)
    pension    = zeros(n_z)       (no pension/income)
    b_bar      = 0                (no bequest motive)
    delta      = solver_config.delta_bequest

Usage
-----
    python verify/ee_simpath_inf_horizon.py <bundle> \
        [--eval-mode {same,next_finer,double}] \
        [--n-samples 16384] [--seed 42]

Output
------
    <bundle>/ee_simpath.json
    <bundle>/ee_simpath.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import numpy as np
import scipy.linalg as la

import jax
import jax.numpy as jnp
from jax import jit, vmap

from lifecycle.model import (
    DELTA_BEQUEST,
    DiscretizationConfig,
    SolverConfig,
)
from lifecycle.policy_io import load_policy_bundle
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import (
    _build_step_log_returns,
    _build_step_state_brackets,
    _pc_to_jnp,
    bracket_uniform,
    interp_1d_lin_extrap,
    retirement_foc_jac_ccv,
)
from verify._diag_helpers import build_bundle_var_config


LOG10_EE_FLOOR = -16.0


# =============================================================================
# Bundle loading + config rehydration
# =============================================================================

def _resolve_bundle_path(bundle_arg: str) -> Path:
    p = Path(bundle_arg)
    if p.is_dir():
        return p
    p2 = Path("saved_runs") / bundle_arg
    if p2.is_dir():
        return p2
    p3 = Path("saved_runs") / "inf_horizon" / bundle_arg
    if p3.is_dir():
        return p3
    raise FileNotFoundError(
        f"Bundle not found. Tried: {p}, {p2}, {p3}"
    )


def _list_to_tuple_recursive(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_list_to_tuple_recursive(x) for x in v)
    return v


def _rehydrate_disc_config(d: dict) -> DiscretizationConfig:
    tuple_fields = {
        "state_grid_sizes",
        "n_state_quad_nodes",
        "n_ret_nodes_1d",
        "state_n_stds",
        "ret_lobatto_Z",
        "state_lobatto_Z",
    }
    valid = set(DiscretizationConfig._fields)
    kwargs = {}
    for k, v in d.items():
        if k not in valid:
            continue
        kwargs[k] = _list_to_tuple_recursive(v) if k in tuple_fields else v
    return DiscretizationConfig(**kwargs)


def _rehydrate_solver_config(d: dict | None) -> SolverConfig:
    if d is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    return SolverConfig(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# Eval-rule construction (mirrors verify/ee_simpath.py)
# =============================================================================

def _coerce_K_per_axis(value: Any, n_axes: int) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n_axes
    seq = list(value)
    if len(seq) != n_axes:
        raise ValueError(f"Expected length-{n_axes} sequence, got {seq!r}")
    return tuple(int(v) for v in seq)


def _build_eval_disc(base: DiscretizationConfig, model, mode: str) -> DiscretizationConfig:
    n_state = int(model.n_state)
    n_ret = int(model.n_ret)
    ret_base = _coerce_K_per_axis(base.n_ret_nodes_1d, n_ret)
    state_base = _coerce_K_per_axis(base.n_state_quad_nodes, n_state)
    eta_base = int(base.n_eta_nodes)
    eps_base = int(base.n_eps_nodes)

    if mode == "same":
        ret_eval, state_eval = ret_base, state_base
        eta_eval, eps_eval = eta_base, eps_base
    elif mode == "next_finer":
        ret_eval = tuple(max(1, k + 2) for k in ret_base)
        state_eval = tuple(max(1, k + 1) for k in state_base)
        eta_eval = max(1, eta_base + 2)
        eps_eval = max(1, eps_base + 2)
    elif mode == "double":
        ret_eval = tuple(max(1, 2 * k) for k in ret_base)
        state_eval = tuple(max(1, 2 * k) for k in state_base)
        eta_eval = max(1, 2 * eta_base)
        eps_eval = max(1, 2 * eps_base)
    else:
        raise ValueError(f"Unknown eval mode: {mode!r}")

    return base._replace(
        n_ret_nodes_1d=ret_eval,
        n_state_quad_nodes=state_eval,
        n_eta_nodes=eta_eval,
        n_eps_nodes=eps_eval,
    )


# =============================================================================
# Stationary state-coord sampling (analytic VAR(1) stationary normal)
# =============================================================================

def _sample_stationary_state_coords(model, n_samples: int, rng: np.random.Generator,
                                     n_stds_clip: float) -> np.ndarray:
    """Draw state_coords from the VAR's stationary multivariate normal.

    s_t+1 = Phi_0_state + Phi_11 s_t + N(0, Sigma_ss) -- assuming Phi_11
    has spectral radius < 1, the stationary distribution is N(mu, Sigma) with:
        mu  = (I - Phi_11)^{-1} Phi_0_state
        Sigma = solve_discrete_lyapunov(Phi_11, Sigma_ss)

    Samples are clipped to within ``n_stds_clip`` per-axis stds of mu so we
    don't draw extreme tails the policy never sees during fixed-point iteration.
    """
    Phi_0 = np.asarray(model.Phi_0_state, dtype=np.float64)
    Phi_11 = np.asarray(model.Phi_11, dtype=np.float64)
    Sigma_ss = np.asarray(model.Sigma_ss, dtype=np.float64)
    n_state = Phi_0.shape[0]

    I = np.eye(n_state)
    mu = la.solve(I - Phi_11, Phi_0)
    Sigma = la.solve_discrete_lyapunov(Phi_11, Sigma_ss)
    Sigma = 0.5 * (Sigma + Sigma.T)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(n_state))

    z_std = rng.standard_normal(size=(n_samples, n_state))
    samples = mu[None, :] + z_std @ L.T

    sigma_axes = np.sqrt(np.diag(Sigma))
    if n_stds_clip > 0.0 and np.all(np.isfinite(sigma_axes)) and np.all(sigma_axes > 0.0):
        lo = mu - n_stds_clip * sigma_axes
        hi = mu + n_stds_clip * sigma_axes
        samples = np.clip(samples, lo, hi)

    return np.ascontiguousarray(samples, dtype=np.float64), mu, sigma_axes


def _sample_z_stationary(model, pc, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample continuous z from the stationary normal of the AR(1) z process.

    With n_z=1 (the inf-horizon canonical config) the z_grid collapses to a
    single point -- we just return that constant. Otherwise we mirror the
    simulator's "stationary" mode: sample an index from the binned normal
    proportional to the stationary std of z, then return z_grid[idx]. Returns
    continuous values (z_grid points; brackets in the FOC handle the rest).
    """
    if pc.n_z == 1:
        return np.full(n_samples, float(pc.z_grid[0]), dtype=np.float64)

    # AR(1): z_t+1 = rho z_t + eta. Stationary variance:
    mu_eta2_eff = -(model.pz / (1.0 - model.pz)) * model.mu_eta1
    var_eta = (
        model.pz * (model.sigma_eta1 ** 2 + model.mu_eta1 ** 2)
        + (1 - model.pz) * (model.sigma_eta2 ** 2 + mu_eta2_eff ** 2)
    )
    sigma_z = np.sqrt(var_eta / (1.0 - model.rho ** 2))
    z = rng.normal(loc=0.0, scale=sigma_z, size=n_samples)
    z_min = float(pc.z_grid[0])
    z_max = float(pc.z_grid[-1])
    return np.clip(z, z_min, z_max).astype(np.float64)


def _sample_wealth_loguniform(pc, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Log-uniform sample over the bundle's [wealth_min, wealth_max]."""
    w_min = float(pc.wealth_grid[0])
    w_max = float(pc.wealth_grid[-1])
    log_w = rng.uniform(np.log(w_min), np.log(w_max), size=n_samples)
    return np.exp(log_w).astype(np.float64)


# =============================================================================
# Stationary-policy interpolation (multilinear in s, bilinear in z*w)
# =============================================================================

def _build_policy_interp_kernel(pcj_solver, model):
    """JIT'd batched interpolation of (C, S, B) at continuous (z, s, w) under
    the stationary inf-horizon slab. Returns (c, alpha_s, alpha_b) per sample.

    The interpolation scheme matches lifecycle.simulation._trilinear_z_w_lookup
    (multilinear in state corners, bilinear in (z, w)) so the EE we compute is
    consistent with what the solver/sim sees.
    """
    n_state = int(model.n_state)
    n_corners = 1 << n_state
    corner_offsets_np = np.zeros((n_corners, n_state), dtype=np.int32)
    for c in range(n_corners):
        for d in range(n_state):
            corner_offsets_np[c, d] = (c >> (n_state - 1 - d)) & 1
    axis_sizes = pcj_solver.axis_sizes
    strides_np = np.empty(n_state, dtype=np.int32)
    s = 1
    for d in range(n_state - 1, -1, -1):
        strides_np[d] = s
        s *= int(axis_sizes[d])
    corner_offsets = jnp.asarray(corner_offsets_np)
    strides = jnp.asarray(strides_np)
    axis_grids = pcj_solver.axis_grids
    bracket_shift = pcj_solver.state_bracket_shift
    bracket_L_inv = pcj_solver.state_bracket_L_inv
    z_grid = pcj_solver.z_grid
    z_lo = pcj_solver.z_grid[0]
    dz = pcj_solver.dz
    n_z = pcj_solver.z_grid.shape[0]
    wealth_grid = pcj_solver.wealth_grid

    def _bracket_axis(grid, val):
        n = grid.shape[0]
        lo = jnp.clip(jnp.searchsorted(grid, val, side="right") - 1, 0, n - 2)
        denom = grid[lo + 1] - grid[lo]
        frac = jnp.clip((val - grid[lo]) / denom, 0.0, 1.0)
        return lo, frac

    z_collapses = (n_z == 1)

    @jit
    def interp_one(C_slab, S_slab, B_slab, z, s_t, w):
        # z bracket -- skipped when n_z=1 (inf-horizon canonical config). With
        # n_z=1, bracket_uniform's iz_hi=1 would index out of bounds and return
        # NaN under JAX's gather semantics on CPU, poisoning everything.
        if z_collapses:
            iz_lo = jnp.int32(0)
            iz_hi = jnp.int32(0)
            frac_z = jnp.float64(0.0)
        else:
            iz_lo, frac_z = bracket_uniform(z, z_lo, dz, n_z)
            iz_hi = iz_lo + 1

        # state bracket on Cholesky-decorrelated coords
        b_vec = bracket_L_inv @ (s_t - bracket_shift)
        lo_list = []
        frac_list = []
        for d in range(n_state):
            lo_d, f_d = _bracket_axis(axis_grids[d], b_vec[d])
            lo_list.append(lo_d)
            frac_list.append(f_d)
        lo = jnp.stack(lo_list)
        frac = jnp.stack(frac_list)

        per_axis = jnp.where(corner_offsets > 0, frac[None, :], 1.0 - frac[None, :])
        w_corners = jnp.prod(per_axis, axis=1)
        idx_per_axis = lo[None, :] + corner_offsets
        j_corners = jnp.sum(idx_per_axis * strides[None, :], axis=1).astype(jnp.int32)

        def per_corner(arr, j):
            v_lo = interp_1d_lin_extrap(w, wealth_grid, arr[iz_lo, j, :])
            v_hi = interp_1d_lin_extrap(w, wealth_grid, arr[iz_hi, j, :])
            return (1.0 - frac_z) * v_lo + frac_z * v_hi

        def reduce_arr(arr):
            vals = vmap(lambda j, wt: wt * per_corner(arr, j))(j_corners, w_corners)
            return jnp.sum(vals)

        c = reduce_arr(C_slab)
        a_s = reduce_arr(S_slab)
        a_b = reduce_arr(B_slab)
        return c, a_s, a_b

    @jit
    def batch(C_slab, S_slab, B_slab, z_arr, s_arr, w_arr):
        return vmap(interp_one, in_axes=(None, None, None, 0, 0, 0))(
            C_slab, S_slab, B_slab, z_arr, s_arr, w_arr,
        )

    return batch


# =============================================================================
# FOC re-evaluation (retirement only)
# =============================================================================

def _build_inf_horizon_ee_kernel(pcj_solver, pcj_eval, model, sc, delta):
    """JIT'd per-sample EE kernel for the inf-horizon retirement-only Bellman.

    Inputs per sample:
        z          continuous z (used to bracket on solver z_grid)
        state      state_coords (n_state,)
        c, a_s, a_b, sav   policy outputs at (z, state, w)
        C_slab     (n_z, N_state, n_w) stationary policy used as C_next

    Overrides applied to match inf_horizon_solver:
        psi_z = 1.0 (per z), pension_next = 0.0 (per z), b_bar = 0
    """
    gamma = jnp.float64(model.gamma)
    beta = jnp.float64(model.beta)
    b_bar = jnp.float64(0.0)
    delta_j = jnp.float64(delta)
    min_consumption = jnp.float64(sc.min_consumption)

    z_collapses = (int(pcj_solver.z_grid.shape[0]) == 1)

    @jit
    def per_sample(z, state, c, alpha_s, alpha_b, savings, C_slab, A_const):
        if z_collapses:
            iz_lo = jnp.int32(0)
            iz_hi = jnp.int32(0)
            frac_z = jnp.float64(0.0)
        else:
            iz_lo, frac_z = bracket_uniform(
                z, pcj_solver.z_grid[0], pcj_solver.dz, pcj_solver.z_grid.shape[0],
            )
            iz_hi = iz_lo + 1
        psi_z = jnp.float64(1.0)
        pension_next_z = jnp.float64(0.0)

        s_next, j_corners, w_corners = _build_step_state_brackets(
            state,
            pcj_solver.Phi_0_state,
            pcj_solver.Phi_11,
            pcj_eval.v_nodes,
            pcj_solver.axis_grids,
            pcj_solver.axis_sizes,
            pcj_solver.corner_offsets,
            pcj_solver.strides,
            pcj_solver.state_bracket_shift,
            pcj_solver.state_bracket_L_inv,
        )
        log_R_bill, log_x_s, log_x_b = _build_step_log_returns(
            state,
            pcj_eval.M_v_nodes,
            pcj_eval.ret_nodes,
            pcj_eval.const_r,
            pcj_eval.A_r,
            pcj_solver.y_1_idx,
            pcj_solver.xr_pos,
            pcj_solver.xb_pos,
        )

        c_lo = C_slab[iz_lo, j_corners, :]
        c_hi = C_slab[iz_hi, j_corners, :]
        c_corners_at_z = (1.0 - frac_z) * c_lo + frac_z * c_hi

        _, _, _, _, _, e_sum = retirement_foc_jac_ccv(
            alpha_s, alpha_b, savings, psi_z,
            log_R_bill, log_x_s, log_x_b, pcj_eval.weight_kv_kr,
            w_corners, c_corners_at_z, pcj_solver.wealth_grid,
            pension_next_z, A_const,
            pcj_eval.sigma2_xr, pcj_eval.sigma2_xb, pcj_eval.sigma_xrxb,
            gamma, b_bar, delta_j, min_consumption,
        )

        beta_e = beta * e_sum
        valid = jnp.logical_and(beta_e > 0.0, c > 0.0)
        c_implied = jnp.where(
            valid,
            jnp.power(jnp.where(valid, beta_e, 1.0), -1.0 / gamma),
            jnp.nan,
        )
        ee = jnp.where(valid, 1.0 - c_implied / c, jnp.nan)
        return ee, valid, e_sum

    @jit
    def batch(z_arr, state_arr, c_arr, as_arr, ab_arr, sav_arr, C_slab, A_const):
        return vmap(
            per_sample,
            in_axes=(0, 0, 0, 0, 0, 0, None, None),
        )(z_arr, state_arr, c_arr, as_arr, ab_arr, sav_arr, C_slab, A_const)

    return batch


# =============================================================================
# Aggregation
# =============================================================================

def _stats_log10_abs_ee(ee: np.ndarray, valid: np.ndarray) -> dict:
    if not np.any(valid):
        return {
            "count": 0,
            "mean_log10_abs_ee": None,
            "median_log10_abs_ee": None,
            "p95_log10_abs_ee": None,
            "p99_log10_abs_ee": None,
            "max_log10_abs_ee": None,
            "frac_below_neg6": None,
            "frac_below_neg5": None,
            "frac_below_neg4": None,
        }
    abs_ee = np.abs(ee[valid])
    log_abs = np.log10(np.maximum(abs_ee, 10.0 ** LOG10_EE_FLOOR))
    return {
        "count": int(log_abs.size),
        "mean_log10_abs_ee": float(np.mean(log_abs)),
        "median_log10_abs_ee": float(np.median(log_abs)),
        "p95_log10_abs_ee": float(np.percentile(log_abs, 95.0)),
        "p99_log10_abs_ee": float(np.percentile(log_abs, 99.0)),
        "max_log10_abs_ee": float(np.max(log_abs)),
        "frac_below_neg6": float(np.sum(log_abs < -6.0) / log_abs.size),
        "frac_below_neg5": float(np.sum(log_abs < -5.0) / log_abs.size),
        "frac_below_neg4": float(np.sum(log_abs < -4.0) / log_abs.size),
    }


GATES = {
    "publication": {"mean_gate": -5.0, "max_gate": -3.0},
    "welfare":     {"mean_gate": -4.0, "max_gate": -2.0},
}


def _build_gates(stats_all: dict, stats_unc: dict) -> list[dict]:
    out = []
    for cell_set, st in (("all", stats_all), ("unconstrained", stats_unc)):
        for grade, thr in GATES.items():
            mean_v = st["mean_log10_abs_ee"]
            max_v = st["max_log10_abs_ee"]
            if mean_v is None or max_v is None:
                passed = None
            else:
                passed = (mean_v <= thr["mean_gate"]) and (max_v <= thr["max_gate"])
            out.append({
                "cell_set": cell_set,
                "grade": grade,
                "mean_log10_abs_ee": mean_v,
                "max_log10_abs_ee": max_v,
                "mean_gate": thr["mean_gate"],
                "max_gate": thr["max_gate"],
                "pass": passed,
            })
    return out


# =============================================================================
# Markdown
# =============================================================================

def _format_md(summary: dict) -> str:
    L: list[str] = []
    L.append(f"# Inf-horizon sim-path EE: `{Path(summary['bundle_path']).name}`")
    L.append("")
    L.append(f"- Bundle: `{summary['bundle_path']}`")
    L.append(f"- Eval mode: `{summary['eval_mode']}`")
    L.append(f"- N samples: `{summary['n_samples']}`")
    L.append(f"- Seed: `{summary['seed']}`")
    L.append(f"- Solver tol: `{summary['tolerance_used_at_solve']:.1e}`")
    L.append(f"- Inf-horizon overrides: psi=1, pension=0, b_bar=0, delta={summary['delta_bequest_used']}")
    sd = summary["solver_disc_config"]
    ed = summary["eval_disc_config"]
    L.append(f"- Solver quadrature: ret={sd['n_ret_nodes_1d']}, "
             f"state={sd['n_state_quad_nodes']}, "
             f"eta={sd['n_eta_nodes']}, eps={sd['n_eps_nodes']}")
    L.append(f"- Eval quadrature:   ret={ed['n_ret_nodes_1d']}, "
             f"state={ed['n_state_quad_nodes']}, "
             f"eta={ed['n_eta_nodes']}, eps={ed['n_eps_nodes']}")
    L.append(f"- Sampling: state ~ stationary VAR (clip {summary['state_clip_n_stds']}σ); "
             f"z ~ stationary AR(1) on z_grid; wealth ~ log-uniform")
    L.append(f"- Kink tol (s/x): `{summary['kink_tol']}`")
    L.append(f"- log10|EE| floor: `{LOG10_EE_FLOOR}`")
    L.append("")

    def _fmt(x):
        return "n/a" if x is None else f"{x:.4f}"

    for cell_set in ("all", "unconstrained"):
        st = summary[f"stats_{cell_set}"]
        L.append(f"## Stats ({cell_set})")
        L.append("")
        L.append(f"- Count            : `{st['count']}`")
        L.append(f"- Mean log10|EE|   : `{_fmt(st['mean_log10_abs_ee'])}`")
        L.append(f"- Median log10|EE| : `{_fmt(st['median_log10_abs_ee'])}`")
        L.append(f"- P95 log10|EE|    : `{_fmt(st['p95_log10_abs_ee'])}`")
        L.append(f"- P99 log10|EE|    : `{_fmt(st['p99_log10_abs_ee'])}`")
        L.append(f"- Max log10|EE|    : `{_fmt(st['max_log10_abs_ee'])}`")
        if st["frac_below_neg6"] is not None:
            L.append(f"- Frac < -6        : `{st['frac_below_neg6']*100:.2f}%`")
            L.append(f"- Frac < -5        : `{st['frac_below_neg5']*100:.2f}%`")
            L.append(f"- Frac < -4        : `{st['frac_below_neg4']*100:.2f}%`")
        L.append("")

    L.append("## Pass / fail gates")
    L.append("")
    L.append("| cell_set | grade | mean | max | mean_gate | max_gate | pass |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | :---: |")
    for g in summary["gates"]:
        m = "-" if g["mean_log10_abs_ee"] is None else f"{g['mean_log10_abs_ee']:.3f}"
        mx = "-" if g["max_log10_abs_ee"] is None else f"{g['max_log10_abs_ee']:.3f}"
        passed = "-" if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
        L.append(f"| {g['cell_set']} | {g['grade']} | {m} | {mx} | "
                 f"{g['mean_gate']:.1f} | {g['max_gate']:.1f} | {passed} |")
    L.append("")

    L.append("## Sampling diagnostics")
    L.append("")
    L.append(f"- Stationary state mean : `{summary['state_mu']}`")
    L.append(f"- Stationary state std  : `{summary['state_sigma_per_axis']}`")
    L.append(f"- Wealth range          : `[{summary['wealth_min']:.3f}, {summary['wealth_max']:.3f}]`")
    L.append(f"- Constrained samples   : `{summary['n_constrained']}` "
             f"({summary['frac_constrained']*100:.2f}%)")
    L.append(f"- Invalid samples       : `{summary['n_invalid']}`")
    L.append("")
    return "\n".join(L)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("bundle", help="Inf-horizon bundle dir or bare name "
                        "(also looked up under saved_runs/inf_horizon/<name>/).")
    parser.add_argument("--eval-mode", choices=("same", "next_finer", "double"),
                        default="next_finer")
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-clip-n-stds", type=float, default=3.0,
                        help="Per-axis clip on stationary state samples (σ multiples). 0 disables.")
    parser.add_argument("--kink-tol", type=float, default=1e-3,
                        help="Samples with savings/wealth < this are flagged constrained.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    bundle_path = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle_path}", flush=True)

    print("Loading bundle...", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle_path)
    print(f"  Policy shape: C={C.shape}, dtype={C.dtype}", flush=True)

    spec = metadata.get("wealth_dynamics_spec", "?")
    if spec != "ccv_log":
        raise ValueError(
            f"Bundle wealth_dynamics_spec={spec!r}; this diagnostic requires ccv_log."
        )
    solver_kind = metadata.get("run_config", {}).get("solver_kind", "?")
    if solver_kind != "infinite_horizon":
        raise ValueError(
            f"Bundle solver_kind={solver_kind!r}; this script targets infinite_horizon. "
            "Use verify/ee_simpath.py for lifecycle bundles."
        )

    run_config = metadata["run_config"]
    base_config = run_config["base_config"]
    disc_solver = _rehydrate_disc_config(run_config["discretization_config"])
    disc_solver = disc_config_with_bundle_wealth_grid(disc_solver, bundle_path, metadata)
    solver_config = _rehydrate_solver_config(run_config.get("solver_config"))
    delta = solver_config.delta_bequest if solver_config.delta_bequest >= 0.0 else DELTA_BEQUEST

    print("Rebuilding model + solver-side precompute...", flush=True)
    var_config = build_bundle_var_config(metadata, bundle_path)
    model = build_model(base_config, var_config, verbose=False)
    pc_solver = build_precompute(model, disc_solver, verbose=False)

    expected_shape = (pc_solver.n_z, pc_solver.N_state, pc_solver.n_w)
    if C.shape != expected_shape:
        raise RuntimeError(
            f"Policy shape {C.shape} != rebuilt {expected_shape}; bundle/config mismatch."
        )

    print(f"Building eval-rule precompute (mode={args.eval_mode})...", flush=True)
    disc_eval = _build_eval_disc(disc_solver, model, args.eval_mode)
    pc_eval = build_precompute(model, disc_eval, verbose=False)
    print(
        f"  Solver q: ret={disc_solver.n_ret_nodes_1d} state={disc_solver.n_state_quad_nodes} "
        f"eta={disc_solver.n_eta_nodes} eps={disc_solver.n_eps_nodes}",
        flush=True,
    )
    print(
        f"  Eval q  : ret={disc_eval.n_ret_nodes_1d} state={disc_eval.n_state_quad_nodes} "
        f"eta={disc_eval.n_eta_nodes} eps={disc_eval.n_eps_nodes}",
        flush=True,
    )

    pcj_solver = _pc_to_jnp(pc_solver, delta)
    pcj_eval = _pc_to_jnp(pc_eval, delta)

    # ---- Sample ----
    rng = np.random.default_rng(int(args.seed))
    print(f"\nSampling {args.n_samples} initial states...", flush=True)
    state_samples, state_mu, state_sigma = _sample_stationary_state_coords(
        model, int(args.n_samples), rng, float(args.state_clip_n_stds),
    )
    z_samples = _sample_z_stationary(model, pc_solver, int(args.n_samples), rng)
    w_samples = _sample_wealth_loguniform(pc_solver, int(args.n_samples), rng)

    print(f"  state_mu          : {state_mu}", flush=True)
    print(f"  state_sigma_axes  : {state_sigma}", flush=True)
    print(f"  z range (sampled) : [{z_samples.min():.3f}, {z_samples.max():.3f}]", flush=True)
    print(f"  w range (sampled) : [{w_samples.min():.3f}, {w_samples.max():.3f}]", flush=True)

    # ---- Interpolate policy ----
    print("\nInterpolating stationary policy at samples...", flush=True)
    interp_kernel = _build_policy_interp_kernel(pcj_solver, model)
    C_jnp = jnp.asarray(C)
    S_jnp = jnp.asarray(S)
    B_jnp = jnp.asarray(B)
    z_j = jnp.asarray(z_samples)
    s_j = jnp.asarray(state_samples)
    w_j = jnp.asarray(w_samples)

    t0 = time.time()
    c_j, as_j, ab_j = interp_kernel(C_jnp, S_jnp, B_jnp, z_j, s_j, w_j)
    c_j.block_until_ready()
    print(f"  Interp wall: {time.time() - t0:.2f}s", flush=True)

    c_np = np.asarray(jax.device_get(c_j))
    as_np = np.asarray(jax.device_get(as_j))
    ab_np = np.asarray(jax.device_get(ab_j))

    # Flat-extrap upper-edge rescale (matches simulator)
    w_max_grid = float(pc_solver.wealth_grid[-1])
    above = w_samples > w_max_grid
    c_np = np.where(above, c_np * (w_samples / w_max_grid), c_np)
    c_np = np.clip(c_np, 0.0, np.maximum(w_samples, 0.0))
    sav_np = w_samples - c_np

    # ---- FOC re-evaluation ----
    print("\nRe-evaluating retirement FOC at samples (eval-rule quadrature)...", flush=True)
    ee_kernel = _build_inf_horizon_ee_kernel(pcj_solver, pcj_eval, model, solver_config, delta)
    A_const = jnp.float64(1.0)  # annuity factor unused (b_bar=0)
    t0 = time.time()
    ee_j, valid_j, esum_j = ee_kernel(
        z_j, s_j,
        jnp.asarray(c_np), jnp.asarray(as_np), jnp.asarray(ab_np), jnp.asarray(sav_np),
        C_jnp, A_const,
    )
    ee_j.block_until_ready()
    print(f"  FOC wall: {time.time() - t0:.2f}s", flush=True)
    ee_np = np.asarray(jax.device_get(ee_j))
    valid_np = np.asarray(jax.device_get(valid_j))
    valid_np = valid_np & np.isfinite(ee_np)

    # KKT-slack flag
    is_constrained = np.zeros_like(valid_np)
    pos = w_samples > 1e-12
    is_constrained[pos] = (sav_np[pos] / w_samples[pos]) < args.kink_tol
    n_invalid = int((~valid_np).sum())
    n_constrained = int((is_constrained & valid_np).sum())

    stats_all = _stats_log10_abs_ee(ee_np, valid_np)
    stats_unc = _stats_log10_abs_ee(ee_np, valid_np & (~is_constrained))
    gates = _build_gates(stats_all, stats_unc)

    summary = {
        "bundle_path": str(bundle_path),
        "wealth_dynamics_spec": metadata.get("wealth_dynamics_spec"),
        "tolerance_used_at_solve": float(solver_config.tol),
        "max_iter_used_at_solve": int(solver_config.max_iter),
        "delta_bequest_used": float(delta),
        "eval_mode": args.eval_mode,
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "state_clip_n_stds": float(args.state_clip_n_stds),
        "kink_tol": float(args.kink_tol),
        "log10_ee_floor": float(LOG10_EE_FLOOR),
        "solver_disc_config": disc_solver._asdict(),
        "eval_disc_config": disc_eval._asdict(),
        "wealth_min": float(pc_solver.wealth_grid[0]),
        "wealth_max": float(pc_solver.wealth_grid[-1]),
        "state_mu": [float(x) for x in state_mu.tolist()],
        "state_sigma_per_axis": [float(x) for x in state_sigma.tolist()],
        "n_invalid": n_invalid,
        "n_constrained": n_constrained,
        "frac_constrained": (n_constrained / max(1, int(valid_np.sum()))),
        "stats_all": stats_all,
        "stats_unconstrained": stats_unc,
        "gates": gates,
    }

    def _stringify(v):
        if isinstance(v, tuple):
            return [_stringify(x) for x in v]
        if isinstance(v, dict):
            return {k: _stringify(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_stringify(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    summary["solver_disc_config"] = _stringify(summary["solver_disc_config"])
    summary["eval_disc_config"] = _stringify(summary["eval_disc_config"])

    # ---- stdout summary ----
    print("\n" + "=" * 72, flush=True)
    print("Inf-horizon sim-path EE summary", flush=True)
    print("=" * 72, flush=True)
    print(f"  Bundle      : {bundle_path}", flush=True)
    print(f"  Eval mode   : {args.eval_mode}", flush=True)
    print(f"  N samples   : {args.n_samples}", flush=True)
    print(f"  Invalid     : {n_invalid}", flush=True)
    print(f"  Constrained : {n_constrained} ({summary['frac_constrained']*100:.2f}%)", flush=True)
    print("\n  All-cells stats:", flush=True)
    if stats_all["mean_log10_abs_ee"] is not None:
        print(f"    mean={stats_all['mean_log10_abs_ee']:.3f}  "
              f"median={stats_all['median_log10_abs_ee']:.3f}  "
              f"p95={stats_all['p95_log10_abs_ee']:.3f}  "
              f"p99={stats_all['p99_log10_abs_ee']:.3f}  "
              f"max={stats_all['max_log10_abs_ee']:.3f}", flush=True)
    print("  Unconstrained stats:", flush=True)
    if stats_unc["mean_log10_abs_ee"] is not None:
        print(f"    mean={stats_unc['mean_log10_abs_ee']:.3f}  "
              f"median={stats_unc['median_log10_abs_ee']:.3f}  "
              f"p95={stats_unc['p95_log10_abs_ee']:.3f}  "
              f"p99={stats_unc['p99_log10_abs_ee']:.3f}  "
              f"max={stats_unc['max_log10_abs_ee']:.3f}", flush=True)
    print("\n  Gates:", flush=True)
    for g in gates:
        if g["pass"] is None:
            continue
        verdict = "PASS" if g["pass"] else "FAIL"
        print(f"    {g['cell_set']:14s} {g['grade']:11s}  "
              f"mean={g['mean_log10_abs_ee']:.3f} (gate {g['mean_gate']:.1f})  "
              f"max={g['max_log10_abs_ee']:.3f} (gate {g['max_gate']:.1f})  -> {verdict}",
              flush=True)
    print("=" * 72, flush=True)

    suffix = args.out_suffix
    if not args.no_save:
        out_path = bundle_path / f"ee_simpath{suffix}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved JSON: {out_path}", flush=True)
    if not args.no_markdown:
        md_path = bundle_path / f"ee_simpath{suffix}.md"
        md_path.write_text(_format_md(summary), encoding="utf-8")
        print(f"Saved markdown: {md_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

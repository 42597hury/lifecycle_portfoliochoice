"""Rank quadrature axes by fixed-policy FOC sensitivity.

This diagnostic answers: "If I spend extra quadrature nodes on exactly one
axis, which axis changes the policy FOC the most?"

It loads a saved policy bundle, rebuilds its model/precompute, and evaluates
the saved policy under:

  1. the solver quadrature rule; and
  2. one-axis-refined evaluation rules.

For each refined axis it reports the FOC displacement and the local implied
portfolio correction

    delta_alpha = -J_eval^{-1} (F_axis - F_base),

where F=(FOC_stock, FOC_bond) and J is the 2x2 portfolio-FOC Jacobian returned
by the same retirement FOC kernel used by the solver.  The first pass is aimed
at infinite-horizon / stationary bundles, so it uses the retirement FOC with
psi=1 and pension=0 by default and ignores n_z economics.

Examples
--------
    python scripts/analysis/quad_axis_sensitivity.py \
        saved_runs/inf_horizon/full_system_inf_horizon_grid8x8x8_nz1_y1lob_calib1

    # Smoke-test on a retirement-age slice from a lifecycle bundle:
    python scripts/analysis/quad_axis_sensitivity.py \
        saved_runs/full/full_system_grid5x5x5_nz11_jax_benchmark \
        --age 67 --state-sample 64 --wealth-sample 24

Outputs
-------
Writes JSON and Markdown next to the bundle by default:

    <bundle>/quad_axis_sensitivity.json
    <bundle>/quad_axis_sensitivity.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from lifecycle.model import DELTA_BEQUEST, DiscretizationConfig, SolverConfig
from lifecycle.policy_io import load_policy_bundle
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid
from lifecycle.precompute import build_model, build_precompute
from lifecycle.solver import (
    _pc_to_jnp,
    _precompute_per_is_tensors,
    retirement_foc_jac_ccv,
)
from verify._diag_helpers import build_bundle_var_config


# ---------------------------------------------------------------------------
# Bundle/config helpers
# ---------------------------------------------------------------------------

def _resolve_bundle_path(bundle_arg: str) -> Path:
    p = Path(bundle_arg)
    if p.is_dir():
        return p
    for root in (Path("saved_runs"), Path("saved_runs") / "inf_horizon"):
        p2 = root / bundle_arg
        if p2.is_dir():
            return p2
    raise FileNotFoundError(f"Bundle not found: {bundle_arg}")


def _list_to_tuple_recursive(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_list_to_tuple_recursive(x) for x in v)
    return v


def _rehydrate_disc_config(d: dict[str, Any]) -> DiscretizationConfig:
    tuple_fields = {
        "state_grid_sizes",
        "n_state_quad_nodes",
        "n_ret_nodes_1d",
        "state_n_stds",
        "ret_lobatto_Z",
        "state_lobatto_Z",
    }
    valid = set(DiscretizationConfig._fields)
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k not in valid:
            continue
        kwargs[k] = _list_to_tuple_recursive(v) if k in tuple_fields else v
    return DiscretizationConfig(**kwargs)


def _rehydrate_solver_config(d: dict[str, Any] | None) -> SolverConfig:
    if d is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    return SolverConfig(**{k: v for k, v in d.items() if k in valid})


def _rebuild_model_pc(metadata: dict[str, Any], bundle_path: Path | None = None):
    run_config = metadata.get("run_config")
    if run_config is None:
        raise ValueError("Bundle metadata lacks run_config; cannot rebuild model/precompute.")
    base_config = run_config.get("base_config")
    disc_dict = run_config.get("discretization_config")
    if base_config is None or disc_dict is None:
        raise ValueError("Bundle run_config must contain base_config and discretization_config.")

    disc = _rehydrate_disc_config(disc_dict)
    if bundle_path is not None:
        disc = disc_config_with_bundle_wealth_grid(disc, bundle_path, metadata)
    solver_config = _rehydrate_solver_config(run_config.get("solver_config"))
    var_config = build_bundle_var_config(metadata, bundle_path)
    model = build_model(base_config, var_config, verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    return model, pc, disc, solver_config, run_config


def _coerce_axis_tuple(value: Any, n_axes: int) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n_axes
    out = tuple(int(v) for v in value)
    if len(out) != n_axes:
        raise ValueError(f"Expected {n_axes} entries, got {out}")
    return out


def _coerce_lobatto_tuple(value: Any, n_axes: int) -> tuple[float | None, ...]:
    if value is None:
        return (None,) * n_axes
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value),) * n_axes
    out = tuple(None if v is None else float(v) for v in value)
    if len(out) != n_axes:
        raise ValueError(f"Expected {n_axes} Lobatto entries, got {out}")
    return out


def _bumped_lobatto_tuple(
    lobatto: tuple[float | None, ...],
    K: tuple[int, ...],
    axis: int,
    policy: str,
) -> tuple[float | None, ...]:
    """Handle K+2 on a Lobatto axis when K would leave the supported set."""
    out = list(lobatto)
    if out[axis] is None:
        return tuple(out)
    if int(K[axis]) in (3, 5, 7):
        return tuple(out)
    if policy == "preserve":
        return tuple(out)
    if policy in ("drop-refined", "drop-invalid"):
        out[axis] = None
        return tuple(out)
    raise ValueError(f"Unknown lobatto policy: {policy}")


def _make_axis_eval_disc(
    base: DiscretizationConfig,
    model,
    kind: str,
    axis: int,
    bump: int,
    lobatto_policy: str,
) -> tuple[DiscretizationConfig, dict[str, Any]]:
    """Return a disc_config with exactly one quadrature axis bumped."""
    if kind == "ret":
        n_axes = int(model.n_ret)
        K0 = _coerce_axis_tuple(base.n_ret_nodes_1d, n_axes)
        Z0 = _coerce_lobatto_tuple(base.ret_lobatto_Z, n_axes)
        K = list(K0)
        K[axis] = int(K[axis] + bump)
        Z = _bumped_lobatto_tuple(Z0, tuple(K), axis, lobatto_policy)
        return (
            base._replace(n_ret_nodes_1d=tuple(K), ret_lobatto_Z=Z),
            {"base_K": list(K0), "eval_K": list(K), "base_Z": list(Z0), "eval_Z": list(Z)},
        )
    if kind == "state":
        n_axes = int(model.n_state)
        K0 = _coerce_axis_tuple(base.n_state_quad_nodes, n_axes)
        Z0 = _coerce_lobatto_tuple(base.state_lobatto_Z, n_axes)
        K = list(K0)
        K[axis] = int(K[axis] + bump)
        Z = _bumped_lobatto_tuple(Z0, tuple(K), axis, lobatto_policy)
        return (
            base._replace(n_state_quad_nodes=tuple(K), state_lobatto_Z=Z),
            {"base_K": list(K0), "eval_K": list(K), "base_Z": list(Z0), "eval_Z": list(Z)},
        )
    raise ValueError(f"Unknown axis kind: {kind}")


# ---------------------------------------------------------------------------
# Theory/loadings helpers
# ---------------------------------------------------------------------------

def _axis_theory(model) -> dict[str, list[dict[str, Any]]]:
    ret_L = np.linalg.cholesky(0.5 * (np.asarray(model.Sigma_r_cond) + np.asarray(model.Sigma_r_cond).T))
    state_L = np.linalg.cholesky(0.5 * (np.asarray(model.Sigma_ss) + np.asarray(model.Sigma_ss).T))
    ret_names = list(model.ret_names)
    state_names = list(model.state_names)

    ret_rows = []
    for d in range(model.n_ret):
        loading = ret_L[:, d]
        ret_rows.append({
            "kind": "ret",
            "axis": d,
            "name": ret_names[d],
            "loading_to_returns": {ret_names[i]: float(loading[i]) for i in range(model.n_ret)},
            "norm": float(np.linalg.norm(loading)),
            "abs_xr": float(abs(loading[ret_names.index("xr")])) if "xr" in ret_names else None,
            "abs_xb": float(abs(loading[ret_names.index("xb")])) if "xb" in ret_names else None,
        })

    state_rows = []
    M = np.asarray(model.M, dtype=float)
    for d in range(model.n_state):
        state_loading = state_L[:, d]
        ret_effect = M @ state_loading
        state_rows.append({
            "kind": "state",
            "axis": d,
            "name": state_names[d],
            "loading_to_state": {state_names[i]: float(state_loading[i]) for i in range(model.n_state)},
            "return_mean_effect": {ret_names[i]: float(ret_effect[i]) for i in range(model.n_ret)},
            "return_effect_norm": float(np.linalg.norm(ret_effect)),
            "abs_effect_xr": float(abs(ret_effect[ret_names.index("xr")])) if "xr" in ret_names else None,
            "abs_effect_xb": float(abs(ret_effect[ret_names.index("xb")])) if "xb" in ret_names else None,
        })
    return {"ret": ret_rows, "state": state_rows}


# ---------------------------------------------------------------------------
# Policy-slice and point-sampling helpers
# ---------------------------------------------------------------------------

def _policy_slice(C: np.ndarray, S: np.ndarray, B: np.ndarray, pc, args):
    """Return C_now/S_now/B_now/C_next as 3D arrays."""
    if C.ndim == 3:
        return C, S, B, C, {"source": "3d_stationary", "age": None, "age_index": None}

    if C.ndim != 4:
        raise ValueError(f"Expected 3D or 4D policy arrays, got C.ndim={C.ndim}")

    ages = np.asarray(pc.ages)
    if args.age_index is not None:
        t = int(args.age_index)
    elif args.age is not None:
        matches = np.flatnonzero(ages == int(args.age))
        if matches.size == 0:
            raise ValueError(f"Age {args.age} not found in pc.ages")
        t = int(matches[0])
    else:
        matches = np.flatnonzero(ages >= int(pc.model.retire_age))
        if matches.size == 0:
            raise ValueError("No retirement-age slice found; pass --age-index.")
        t = int(matches[0])

    if t < 0 or t >= C.shape[0]:
        raise ValueError(f"age index {t} out of range")
    if args.stationary_slice:
        t_next = t
    else:
        t_next = min(t + 1, C.shape[0] - 1)
    return (
        C[t], S[t], B[t], C[t_next],
        {
            "source": "4d_age_slice",
            "age": int(ages[t]),
            "age_index": int(t),
            "next_age": int(ages[t_next]),
            "next_age_index": int(t_next),
            "stationary_slice": bool(args.stationary_slice),
        },
    )


def _sample_indices(n: int, sample: int, rng: np.random.Generator, probs=None) -> np.ndarray:
    if sample <= 0 or sample >= n:
        return np.arange(n, dtype=np.int64)
    if probs is not None:
        p = np.asarray(probs, dtype=float)
        p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
        p = p / p.sum() if p.sum() > 0 else None
    else:
        p = None
    idx = rng.choice(np.arange(n), size=int(sample), replace=False, p=p)
    idx.sort()
    return idx.astype(np.int64)


def _wealth_indices(n_w: int, sample: int, trim: int) -> np.ndarray:
    lo = max(0, int(trim))
    hi = n_w - 1
    if sample <= 0 or sample >= (hi - lo + 1):
        return np.arange(lo, hi + 1, dtype=np.int64)
    return np.unique(np.round(np.linspace(lo, hi, int(sample))).astype(np.int64))


def _build_points(pc, args):
    rng = np.random.default_rng(int(args.seed))
    z_idx = np.arange(pc.n_z, dtype=np.int64)
    if args.z_sample > 0 and args.z_sample < pc.n_z:
        z_idx = _sample_indices(pc.n_z, args.z_sample, rng)

    state_probs = getattr(pc, "state_stationary_probs", None)
    probs = state_probs if args.state_sample_mode == "stationary" else None
    s_idx = _sample_indices(pc.N_state, int(args.state_sample), rng, probs=probs)
    w_idx = _wealth_indices(pc.n_w, int(args.wealth_sample), int(args.trim_wealth))

    # Cartesian product as flat point arrays.
    zz, ss, ww = np.meshgrid(z_idx, s_idx, w_idx, indexing="ij")
    return zz.ravel().astype(np.int64), ss.ravel().astype(np.int64), ww.ravel().astype(np.int64)


# ---------------------------------------------------------------------------
# JAX FOC evaluator
# ---------------------------------------------------------------------------

def _build_point_foc_kernel(pc, model, solver_config, delta: float, chunk_size: int):
    pcj = _pc_to_jnp(pc, delta)
    per_is = _precompute_per_is_tensors(pcj)
    log_R_bill_all, log_x_s_all, log_x_b_all, j_corners_all, w_corners_all = per_is

    gamma = jnp.float64(model.gamma)
    b_bar = jnp.float64(0.0)
    delta_j = jnp.float64(delta)
    min_consumption = jnp.float64(solver_config.min_consumption)
    psi_one = jnp.float64(1.0)
    pension_zero = jnp.float64(0.0)

    def one_point(z_idx, i_s, i_w, C_now, S_now, B_now, C_next):
        c = C_now[z_idx, i_s, i_w]
        alpha_s = S_now[z_idx, i_s, i_w]
        alpha_b = B_now[z_idx, i_s, i_w]
        wealth = pcj.wealth_grid[i_w]
        savings = wealth - c
        savings_safe = jnp.where(savings > 0.0, savings, 1e-12)

        j_corners_i = j_corners_all[i_s]
        w_corners_i = w_corners_all[i_s]
        c_corners_at_z = C_next[z_idx, j_corners_i, :]
        A_is = pcj.annuity_factors[i_s]

        fs, fb, Jss, Jbb, Jsb, e_sum = retirement_foc_jac_ccv(
            alpha_s, alpha_b, savings_safe, psi_one,
            log_R_bill_all[i_s], log_x_s_all[i_s], log_x_b_all[i_s],
            pcj.weight_kv_kr,
            w_corners_i, c_corners_at_z, pcj.wealth_grid,
            pension_zero, A_is,
            pcj.sigma2_xr, pcj.sigma2_xb, pcj.sigma_xrxb,
            gamma, b_bar, delta_j, min_consumption,
        )
        valid = jnp.logical_and(jnp.isfinite(c), savings > jnp.float64(1e-10))
        return fs, fb, Jss, Jbb, Jsb, e_sum, valid

    @jit
    def per_chunk(z_idx, i_s, i_w, C_now, S_now, B_now, C_next):
        return vmap(one_point, in_axes=(0, 0, 0, None, None, None, None))(
            z_idx, i_s, i_w, C_now, S_now, B_now, C_next
        )

    def run(z_idx_np, i_s_np, i_w_np, C_now, S_now, B_now, C_next):
        Cj = jnp.asarray(C_now)
        Sj = jnp.asarray(S_now)
        Bj = jnp.asarray(B_now)
        Cnextj = jnp.asarray(C_next)
        outs = []
        n = int(z_idx_np.size)
        for start in range(0, n, chunk_size):
            sl = slice(start, min(start + chunk_size, n))
            chunk_out = per_chunk(
                jnp.asarray(z_idx_np[sl]),
                jnp.asarray(i_s_np[sl]),
                jnp.asarray(i_w_np[sl]),
                Cj, Sj, Bj, Cnextj,
            )
            chunk_out[0].block_until_ready()
            outs.append(tuple(np.asarray(jax.device_get(x)) for x in chunk_out))
        return tuple(np.concatenate([o[k] for o in outs], axis=0) for k in range(7))

    return run


# ---------------------------------------------------------------------------
# Scoring/reporting
# ---------------------------------------------------------------------------

def _safe_implied_delta(Fs, Fb, Jss, Jbb, Jsb, valid, det_floor: float):
    det = Jss * Jbb - Jsb * Jsb
    ok = valid & np.isfinite(det) & (np.abs(det) > det_floor)
    da_s = np.full_like(Fs, np.nan, dtype=float)
    da_b = np.full_like(Fb, np.nan, dtype=float)
    # -inv(J) F for [[Jss,Jsb],[Jsb,Jbb]]
    da_s[ok] = -(Jbb[ok] * Fs[ok] - Jsb[ok] * Fb[ok]) / det[ok]
    da_b[ok] = -(-Jsb[ok] * Fs[ok] + Jss[ok] * Fb[ok]) / det[ok]
    return da_s, da_b, ok


def _stats(x: np.ndarray) -> dict[str, float | int | None]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"n": 0, "median": None, "p90": None, "p95": None, "p99": None, "max": None, "rms": None}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }


def _axis_label(kind: str, axis: int, model) -> str:
    names = model.ret_names if kind == "ret" else model.state_names
    return f"{kind}[{axis}]={names[axis]}"


def _score_axis(
    kind: str,
    axis: int,
    model,
    base_foc,
    eval_foc,
    config_delta: dict[str, Any],
    theory_row: dict[str, Any],
    det_floor: float,
) -> dict[str, Any]:
    fs0, fb0, *_rest0, valid0 = base_foc
    fs1, fb1, Jss1, Jbb1, Jsb1, _e1, valid1 = eval_foc
    valid = valid0 & valid1
    dFs = fs1 - fs0
    dFb = fb1 - fb0

    da_s, da_b, ok = _safe_implied_delta(dFs, dFb, Jss1, Jbb1, Jsb1, valid, det_floor)
    delta_norm = np.sqrt(da_s * da_s + da_b * da_b)
    foc_shift_norm = np.sqrt(dFs * dFs + dFb * dFb)
    eval_foc_norm = np.sqrt(fs1 * fs1 + fb1 * fb1)

    out = {
        "kind": kind,
        "axis": int(axis),
        "label": _axis_label(kind, axis, model),
        "config_delta": config_delta,
        "theory": theory_row,
        "n_points": int(valid.size),
        "n_valid_base_eval": int(np.sum(valid)),
        "n_valid_implied_delta": int(np.sum(ok)),
        "foc_shift_norm": _stats(foc_shift_norm[valid]),
        "eval_foc_norm": _stats(eval_foc_norm[valid]),
        "implied_delta_alpha_norm": _stats(delta_norm),
        "implied_delta_alpha_s_abs": _stats(np.abs(da_s)),
        "implied_delta_alpha_b_abs": _stats(np.abs(da_b)),
    }
    # Main ranking score: p95 movement in portfolio-share units.
    p95 = out["implied_delta_alpha_norm"]["p95"]
    out["rank_score_p95_delta_alpha"] = float(p95) if p95 is not None else float("nan")
    return out


def _format_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Quadrature Axis Sensitivity",
        "",
        f"Bundle: `{summary['bundle_path']}`",
        "",
        "## Ranking",
        "",
        "| rank | axis | p95 ||delta alpha|| | rms ||delta alpha|| | p95 |delta alpha_b| | theory xb loading |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(summary["ranked_axes"], start=1):
        p95 = row["implied_delta_alpha_norm"]["p95"]
        rms = row["implied_delta_alpha_norm"]["rms"]
        pb = row["implied_delta_alpha_b_abs"]["p95"]
        if row["kind"] == "ret":
            xb = row["theory"].get("abs_xb")
        else:
            xb = row["theory"].get("abs_effect_xb")
        lines.append(
            f"| {i} | `{row['label']}` | {p95:.3e} | {rms:.3e} | {pb:.3e} | {xb:.3e} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "`delta_alpha = -J_eval^{-1}(F_axis - F_base)` is a fixed-policy,",
        "first-order correction. It ranks where an extra quadrature node is",
        "most likely to move the solved policy; it is not a substitute for",
        "re-solving the top candidate refinements.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("bundle", help="Bundle directory or bare saved_runs[/inf_horizon] name.")
    parser.add_argument("--axis-kind", choices=("all", "ret", "state"), default="all")
    parser.add_argument("--bump", type=int, default=2, help="K increment on the refined axis.")
    parser.add_argument(
        "--lobatto-policy", choices=("drop-invalid", "drop-refined", "preserve"),
        default="drop-invalid",
        help="What to do if a bumped Lobatto axis leaves supported K={3,5,7}.",
    )
    parser.add_argument("--age", type=int, default=None, help="Age to slice from a lifecycle bundle.")
    parser.add_argument("--age-index", type=int, default=None, help="Age index to slice from a lifecycle bundle.")
    parser.add_argument(
        "--stationary-slice", action="store_true",
        help="For 4D bundles, use C[t] as C_next instead of C[t+1].",
    )
    parser.add_argument("--state-sample", type=int, default=0, help="0 means all state cells.")
    parser.add_argument(
        "--state-sample-mode", choices=("uniform", "stationary"), default="uniform",
        help="Sampling probabilities when --state-sample is positive.",
    )
    parser.add_argument("--wealth-sample", type=int, default=64, help="0 means all wealth cells.")
    parser.add_argument("--trim-wealth", type=int, default=2)
    parser.add_argument("--z-sample", type=int, default=0, help="0 means all z cells.")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--det-floor", type=float, default=1e-18)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    bundle = _resolve_bundle_path(args.bundle)
    print(f"Bundle: {bundle}", flush=True)
    C, S, B, _diag, metadata = load_policy_bundle(bundle)
    print(f"  policy shape={C.shape}, dtype={C.dtype}", flush=True)

    print("Rebuilding model/precompute...", flush=True)
    t0 = time.time()
    model, pc_base, disc_base, solver_config, run_config = _rebuild_model_pc(
        metadata, bundle
    )
    delta = solver_config.delta_bequest if solver_config.delta_bequest >= 0 else DELTA_BEQUEST
    print(
        f"  setup {time.time() - t0:.1f}s | state={model.state_names} ret={model.ret_names} "
        f"| N_state={pc_base.N_state} n_z={pc_base.n_z} n_w={pc_base.n_w}",
        flush=True,
    )

    C_now, S_now, B_now, C_next, slice_meta = _policy_slice(C, S, B, pc_base, args)
    expected = (pc_base.n_z, pc_base.N_state, pc_base.n_w)
    if C_now.shape != expected:
        raise RuntimeError(f"Policy slice shape {C_now.shape} != rebuilt precompute shape {expected}")

    z_idx, s_idx, w_idx = _build_points(pc_base, args)
    print(
        f"Scoring {z_idx.size:,} points "
        f"({np.unique(z_idx).size} z x {np.unique(s_idx).size} state x {np.unique(w_idx).size} wealth sampled)",
        flush=True,
    )

    theory = _axis_theory(model)
    print("Evaluating base FOC...", flush=True)
    base_runner = _build_point_foc_kernel(pc_base, model, solver_config, delta, int(args.chunk_size))
    base_foc = base_runner(z_idx, s_idx, w_idx, C_now, S_now, B_now, C_next)

    axis_plan: list[tuple[str, int]] = []
    if args.axis_kind in ("all", "ret"):
        axis_plan.extend(("ret", i) for i in range(model.n_ret))
    if args.axis_kind in ("all", "state"):
        axis_plan.extend(("state", i) for i in range(model.n_state))

    results = []
    for kind, axis in axis_plan:
        label = _axis_label(kind, axis, model)
        print(f"\nAxis {label}: building eval precompute...", flush=True)
        disc_eval, config_delta = _make_axis_eval_disc(
            disc_base, model, kind, axis, int(args.bump), args.lobatto_policy
        )
        t_axis = time.time()
        pc_eval = build_precompute(model, disc_eval, verbose=False)
        runner = _build_point_foc_kernel(pc_eval, model, solver_config, delta, int(args.chunk_size))
        print(
            f"  eval nodes: state={pc_eval.n_state_quad}, ret={pc_eval.n_ret_quad}; running FOC...",
            flush=True,
        )
        eval_foc = runner(z_idx, s_idx, w_idx, C_now, S_now, B_now, C_next)
        theory_row = theory[kind][axis]
        row = _score_axis(
            kind, axis, model, base_foc, eval_foc, config_delta, theory_row, float(args.det_floor)
        )
        results.append(row)
        p95 = row["implied_delta_alpha_norm"]["p95"]
        pb = row["implied_delta_alpha_b_abs"]["p95"]
        print(
            f"  score p95||dalpha||={p95:.3e}, p95|dalpha_b|={pb:.3e}, "
            f"valid={row['n_valid_implied_delta']:,}/{row['n_points']:,}, "
            f"wall={time.time() - t_axis:.1f}s",
            flush=True,
        )

    ranked = sorted(
        results,
        key=lambda r: (-np.nan_to_num(r["rank_score_p95_delta_alpha"], nan=-np.inf), r["kind"], r["axis"]),
    )
    summary = {
        "bundle_path": str(bundle),
        "created_from_script": "scripts/analysis/quad_axis_sensitivity.py",
        "slice": slice_meta,
        "base_disc_config": _jsonable(disc_base._asdict()),
        "state_names": list(model.state_names),
        "ret_names": list(model.ret_names),
        "axis_theory": theory,
        "point_sample": {
            "n_points": int(z_idx.size),
            "z_unique": int(np.unique(z_idx).size),
            "state_unique": int(np.unique(s_idx).size),
            "wealth_unique": int(np.unique(w_idx).size),
            "trim_wealth": int(args.trim_wealth),
            "state_sample": int(args.state_sample),
            "wealth_sample": int(args.wealth_sample),
            "z_sample": int(args.z_sample),
        },
        "ranked_axes": ranked,
    }

    print("\nRanking by p95 implied portfolio correction:", flush=True)
    for i, row in enumerate(ranked, start=1):
        p95 = row["implied_delta_alpha_norm"]["p95"]
        rms = row["implied_delta_alpha_norm"]["rms"]
        pb = row["implied_delta_alpha_b_abs"]["p95"]
        print(f"  {i:2d}. {row['label']:<18} p95={p95:.3e} rms={rms:.3e} p95|db|={pb:.3e}", flush=True)

    if not args.no_save:
        json_path = args.output_json or (bundle / "quad_axis_sensitivity.json")
        md_path = args.output_md or (bundle / "quad_axis_sensitivity.md")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        md_path.write_text(_format_md(summary), encoding="utf-8")
        print(f"\nSaved JSON: {json_path}", flush=True)
        print(f"Saved markdown: {md_path}", flush=True)

    return 0


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _jsonable(v):
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


if __name__ == "__main__":
    raise SystemExit(main())

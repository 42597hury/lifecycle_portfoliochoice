"""
_diag_gridpoint_ee.py -- On-grid Euler-error sweep over a saved policy bundle.

Reproduces the sweep that produced `diagnostics_gridpoint_ee_7x7x7_same.md`.
For each (age, iz, i_s, iw) on a structured probe set, evaluates the FOC
residual at the *saved* policy values using the same FOC kernel as
`_diag_euler_errors.py`, with the same `--eval-mode` semantics.

Probe set (defaults match the original 13,770-point report when run against
a 7x7x7, n_z=9, n_w=150 bundle):

- ages: solved range, minus terminal age
- z indices: low / mid / high = [0, n_z//2, n_z-1]
- state cube: low / mid / high per axis = 27 corners
- wealth indices: [0, 15, 75, 134, 149]

Usage
-----
python -m scripts.diagnostics._diag_gridpoint_ee \\
  --model-bundle saved_runs/<full-lifecycle-bundle> \\
  saved_runs/checkpoints/<bundle> \\
  --eval-mode same \\
  --markdown-out diagnostics_gridpoint_ee_log1p_same.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics._diag_euler_errors import (
    _evaluate_age_errors,
    _load_bundle_context,
)


def _state_cube_indices(state_grid_sizes: tuple[int, ...]) -> np.ndarray:
    """Return flat state indices spanning low/mid/high per axis (27 corners)."""
    per_axis = []
    for n in state_grid_sizes:
        per_axis.append([0, n // 2, n - 1])
    out = []
    N1 = int(state_grid_sizes[1])
    N2 = int(state_grid_sizes[2])
    for i0 in per_axis[0]:
        for i1 in per_axis[1]:
            for i2 in per_axis[2]:
                out.append(i0 * N1 * N2 + i1 * N2 + i2)
    return np.asarray(out, dtype=np.int64)


def _build_probe_arrays(ctx, args) -> dict[str, Any]:
    pc = ctx.pc_policy
    model = ctx.model
    n_w = pc.n_w

    # Wealth indices.
    if args.wealth_indices:
        wealth_idx = np.asarray(args.wealth_indices, dtype=np.int64)
    else:
        default = [0, 15, 75, 134, 149]
        wealth_idx = np.asarray([w for w in default if w < n_w], dtype=np.int64)
    if (wealth_idx >= n_w).any():
        raise ValueError(f"wealth_indices out of range for n_w={n_w}")

    # Z indices.
    if args.z_indices:
        z_idx = np.asarray(args.z_indices, dtype=np.int64)
    else:
        z_idx = np.asarray([0, pc.n_z // 2, pc.n_z - 1], dtype=np.int64)
    if (z_idx >= pc.n_z).any():
        raise ValueError(f"z_indices out of range for n_z={pc.n_z}")

    # State indices (low/mid/high cube).
    state_idx = _state_cube_indices(tuple(int(s) for s in pc.state_grid_sizes))

    # Ages: solved window minus terminal.
    solved_ages = ctx.ages[ctx.solved_mask]
    ages = np.asarray(
        [int(a) for a in solved_ages if int(a) < int(model.terminal_age)],
        dtype=np.int64,
    )
    if args.ages:
        requested = set(int(a) for a in args.ages)
        ages = np.asarray(
            sorted(int(a) for a in ages if int(a) in requested),
            dtype=np.int64,
        )

    return {
        "ages": ages,
        "z_idx": z_idx,
        "state_idx": state_idx,
        "wealth_idx": wealth_idx,
    }


def _summarize(rows: list[dict[str, Any]], gate_levels=(-4, -5, -6)) -> dict[str, Any]:
    log_abs = np.array([r["log10_abs_ee"] for r in rows if r["valid"]])
    abs_ee = np.array([r["abs_ee"] for r in rows if r["valid"]])
    summary = {
        "n_total": len(rows),
        "n_valid": int(log_abs.size),
        "mean_log10_abs_ee": float(np.mean(log_abs)) if log_abs.size else float("nan"),
        "p95_log10_abs_ee": float(np.percentile(log_abs, 95.0)) if log_abs.size else float("nan"),
        "max_log10_abs_ee": float(np.max(log_abs)) if log_abs.size else float("nan"),
        "min_log10_abs_ee": float(np.min(log_abs)) if log_abs.size else float("nan"),
    }
    for level in gate_levels:
        summary[f"share_below_{level}"] = int(np.sum(log_abs < level))
    return summary


def _format_summary_md(s: dict[str, Any]) -> list[str]:
    out = []
    out.append(f"- Mean `log10|EE|`: `{s['mean_log10_abs_ee']:.4f}`")
    out.append(f"- P95 `log10|EE|`: `{s['p95_log10_abs_ee']:.4f}`")
    out.append(f"- Max `log10|EE|`: `{s['max_log10_abs_ee']:.4f}`")
    out.append(f"- Min `log10|EE|`: `{s['min_log10_abs_ee']:.4f}`")
    for k, v in s.items():
        if k.startswith("share_below_"):
            level = k.split("_")[-1]
            out.append(f"- Share with `log10|EE| < {level}`: `{v}/{s['n_valid']}`")
    return out


def _phase_summary(rows: list[dict[str, Any]], retire_age: int) -> list[dict[str, Any]]:
    by_phase: dict[str, list] = {"working": [], "retirement": []}
    for r in rows:
        if not r["valid"]:
            continue
        phase = "retirement" if r["age"] >= retire_age else "working"
        by_phase[phase].append(r["log10_abs_ee"])
    out = []
    for phase, vals in by_phase.items():
        if not vals:
            continue
        a = np.asarray(vals)
        out.append({
            "phase": phase,
            "count": int(a.size),
            "mean": float(np.mean(a)),
            "p95": float(np.percentile(a, 95.0)),
            "max": float(np.max(a)),
            "min": float(np.min(a)),
        })
    return out


def _worst_table(rows: list[dict[str, Any]], n: int = 12) -> list[dict[str, Any]]:
    valid = [r for r in rows if r["valid"]]
    valid.sort(key=lambda r: -r["log10_abs_ee"])  # largest log10|EE| first
    return valid[:n]


def _state_index_to_tuple(i_s: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    N1 = int(sizes[1])
    N2 = int(sizes[2])
    i0 = i_s // (N1 * N2)
    rem = i_s % (N1 * N2)
    i1 = rem // N2
    i2 = rem % N2
    return (i0, i1, i2)


def run(args: argparse.Namespace) -> tuple[Any, list[dict[str, Any]]]:
    ctx = _load_bundle_context(
        bundle_path=Path(args.bundle),
        model_bundle=Path(args.model_bundle),
        eval_mode=args.eval_mode,
        ret_override=tuple(args.eval_ret_nodes) if args.eval_ret_nodes else None,
        state_override=tuple(args.eval_state_nodes) if args.eval_state_nodes else None,
        eta_override=args.eval_eta_nodes,
        eps_override=args.eval_eps_nodes,
    )
    pc_eval = ctx.pc_eval
    model = ctx.model

    probes = _build_probe_arrays(ctx, args)
    ages = probes["ages"]
    z_idx = probes["z_idx"]
    state_idx = probes["state_idx"]
    wealth_idx = probes["wealth_idx"]

    rows: list[dict[str, Any]] = []
    pension_row = np.ascontiguousarray(
        pc_eval.pension_after_tax[int(model.retire_age - model.start_age), :]
    )
    N1 = len(pc_eval.state_bracket_grids[1])
    N2 = len(pc_eval.state_bracket_grids[2])

    age_to_t = {int(a): int(a) - int(model.start_age) for a in ages}

    for age in ages:
        age_int = int(age)
        t = age_to_t[age_int]
        # Build per-age cross-section (n_z * n_state * n_w probes).
        n_probe = z_idx.size * state_idx.size * wealth_idx.size
        z_age = np.empty(n_probe, dtype=np.float64)
        state_coords_age = np.empty((n_probe, int(model.n_state)), dtype=np.float64)
        c_age = np.empty(n_probe, dtype=np.float64)
        alpha_s_age = np.empty(n_probe, dtype=np.float64)
        alpha_b_age = np.empty(n_probe, dtype=np.float64)
        savings_age = np.empty(n_probe, dtype=np.float64)
        meta_per_probe = []

        k = 0
        for iz in z_idx:
            for i_s in state_idx:
                for iw in wealth_idx:
                    z_age[k] = pc_eval.z_grid[iz]
                    state_coords_age[k, :] = pc_eval.state_grid[i_s, :]
                    c_val = float(ctx.C[t, iz, i_s, iw])
                    a_s_val = float(ctx.S[t, iz, i_s, iw])
                    a_b_val = float(ctx.B[t, iz, i_s, iw])
                    w_val = float(pc_eval.wealth_grid[iw])
                    s_val = max(w_val - c_val, 0.0)
                    c_age[k] = c_val
                    alpha_s_age[k] = a_s_val
                    alpha_b_age[k] = a_b_val
                    savings_age[k] = s_val
                    meta_per_probe.append({
                        "age": age_int,
                        "iz": int(iz),
                        "i_s": int(i_s),
                        "iw": int(iw),
                        "x": w_val,
                        "c": c_val,
                        "stock": a_s_val,
                        "bond": a_b_val,
                        "state_tuple": _state_index_to_tuple(int(i_s), tuple(int(s) for s in pc_eval.state_grid_sizes)),
                    })
                    k += 1

        household_idx = np.arange(n_probe, dtype=np.int64)

        ee, valid = _evaluate_age_errors(
            household_idx,
            z_age,
            state_coords_age,
            c_age,
            alpha_s_age,
            alpha_b_age,
            savings_age,
            age_int,
            int(model.retire_age),
            np.ascontiguousarray(ctx.C[t + 1]),
            np.ascontiguousarray(pc_eval.wealth_grid),
            np.ascontiguousarray(pc_eval.z_grid),
            float(pc_eval.dz),
            float(pc_eval.log_det_profile[t + 1]),
            pension_row,
            np.ascontiguousarray(pc_eval.survival_probs_2d[t, :]),
            float(model.rho),
            np.ascontiguousarray(pc_eval.eta_nodes),
            np.ascontiguousarray(pc_eval.eta_weights),
            np.ascontiguousarray(pc_eval.eps_nodes),
            np.ascontiguousarray(pc_eval.eps_weights),
            np.ascontiguousarray(pc_eval.v_nodes),
            np.ascontiguousarray(pc_eval.v_weights),
            np.ascontiguousarray(pc_eval.M_v_nodes),
            np.ascontiguousarray(pc_eval.const_r),
            np.ascontiguousarray(pc_eval.A_r),
            np.ascontiguousarray(model.Phi_0_state),
            np.ascontiguousarray(model.Phi_11),
            np.ascontiguousarray(pc_eval.state_bracket_shift),
            np.ascontiguousarray(pc_eval.state_bracket_L_inv),
            np.ascontiguousarray(pc_eval.state_bracket_grids[0]),
            np.ascontiguousarray(pc_eval.state_bracket_grids[1]),
            np.ascontiguousarray(pc_eval.state_bracket_grids[2]),
            int(N1),
            int(N2),
            np.ascontiguousarray(pc_eval.exp_ret_bill),
            np.ascontiguousarray(pc_eval.exp_ret_stock),
            np.ascontiguousarray(pc_eval.exp_ret_bond),
            np.ascontiguousarray(pc_eval.ret_weights),
            float(model.gamma),
            float(model.beta),
            int(model.b_bar),
            int(model.y_1_index_in_state),
            int(model.spr_index_in_state),
        )

        for j in range(n_probe):
            ee_val = float(ee[j])
            is_valid = bool(valid[j])
            abs_ee = abs(ee_val) if is_valid else float("nan")
            log10_abs = float(np.log10(max(abs_ee, 1e-16))) if is_valid else float("nan")
            meta = meta_per_probe[j]
            meta.update({
                "valid": is_valid,
                "ee": ee_val,
                "abs_ee": abs_ee,
                "log10_abs_ee": log10_abs,
                "phase": "retirement" if age_int >= int(model.retire_age) else "working",
            })
            rows.append(meta)

    return ctx, rows


def _markdown_report(ctx, rows, args, probes_info: dict[str, Any]) -> str:
    pc = ctx.pc_policy
    sizes = tuple(int(s) for s in pc.state_grid_sizes)
    summary = _summarize(rows)
    phase_rows = _phase_summary(rows, int(ctx.model.retire_age))
    worst = _worst_table(rows, n=15)

    out: list[str] = []
    out.append(f"# Grid-Point Euler Sweep: `{ctx.label}`")
    out.append("")
    out.append(f"- Bundle: `{ctx.path}`")
    out.append(f"- Eval mode: `{args.eval_mode}`")
    out.append(f"- Policy quadrature: ret=`{ctx.disc_policy.n_ret_nodes_1d}`, state=`{ctx.disc_policy.n_state_quad_nodes}`")
    out.append(f"- Eval quadrature:   ret=`{ctx.disc_eval.n_ret_nodes_1d}`, state=`{ctx.disc_eval.n_state_quad_nodes}`")
    out.append(f"- Ages tested: `{int(probes_info['ages'].min())}-{int(probes_info['ages'].max())}` ({probes_info['ages'].size} ages)")
    out.append(f"- z indices: `{list(int(v) for v in probes_info['z_idx'])}`")
    out.append(f"- State cube points: `{probes_info['state_idx'].size}` (low/mid/high per axis on a {sizes[0]}x{sizes[1]}x{sizes[2]} grid)")
    out.append(f"- Wealth indices: `{list(int(v) for v in probes_info['wealth_idx'])}`")
    out.append(f"- Total grid points: `{summary['n_total']}`  (valid: `{summary['n_valid']}`)")
    out.append("")

    out.append("## Summary")
    out.append("")
    for line in _format_summary_md(summary):
        out.append(line)
    out.append("")

    out.append("## By Phase")
    out.append("")
    out.append("| phase | count | mean | p95 | max | min |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in phase_rows:
        out.append(
            f"| {r['phase']} | {r['count']} | {r['mean']:.4f} | {r['p95']:.4f} | {r['max']:.4f} | {r['min']:.4f} |"
        )
    out.append("")

    out.append("## Worst Grid Points")
    out.append("")
    out.append("| age | phase | iz | state | iw | x | c | stock | bond | log10|EE| | abs EE |")
    out.append("| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in worst:
        st = r["state_tuple"]
        out.append(
            f"| {r['age']} | {r['phase']} | {r['iz']} | "
            f"({st[0]}, {st[1]}, {st[2]}) | {r['iw']} | "
            f"{r['x']:.4f} | {r['c']:.6f} | {r['stock']:.4f} | {r['bond']:.4f} | "
            f"{r['log10_abs_ee']:.4f} | {r['abs_ee']:.4e} |"
        )
    out.append("")
    return "\n".join(out) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    p.add_argument("--model-bundle", required=True)
    p.add_argument("--eval-mode", choices=("same", "next_finer", "double"), default="same")
    p.add_argument("--eval-ret-nodes", nargs="+", type=int, default=None)
    p.add_argument("--eval-state-nodes", nargs="+", type=int, default=None)
    p.add_argument("--eval-eta-nodes", type=int, default=None)
    p.add_argument("--eval-eps-nodes", type=int, default=None)
    p.add_argument("--ages", nargs="+", type=int, default=None)
    p.add_argument("--z-indices", nargs="+", type=int, default=None)
    p.add_argument("--wealth-indices", nargs="+", type=int, default=None)
    p.add_argument("--markdown-out", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx, rows = run(args)
    probes_info = {
        "ages": np.asarray(sorted(set(r["age"] for r in rows)), dtype=np.int64),
        "z_idx": np.asarray(sorted(set(r["iz"] for r in rows)), dtype=np.int64),
        "state_idx": np.asarray(sorted(set(r["i_s"] for r in rows)), dtype=np.int64),
        "wealth_idx": np.asarray(sorted(set(r["iw"] for r in rows)), dtype=np.int64),
    }

    summary = _summarize(rows)
    print(f"Bundle: {ctx.label}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Probe count: {summary['n_total']} (valid {summary['n_valid']})")
    for line in _format_summary_md(summary):
        print(line[2:])

    if args.markdown_out:
        report = _markdown_report(ctx, rows, args, probes_info)
        Path(args.markdown_out).write_text(report, encoding="utf-8")
        print(f"\nWrote markdown report to {args.markdown_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

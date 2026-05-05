"""Bundle-level state-support / clipping diagnostic.

Simulates a saved policy bundle over its solved age window and reports how often
the continuous financial state lands outside the bundle's cholesky-grid support
in transformed bracket coordinates.

This is the direct diagnostic for "did widening state_n_stds actually reduce
boundary clipping at the states the policy visits?".
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
    _load_bundle_context,
    _maybe_build_warm_start,
    _simulate_bundle_window,
)


def _phase_for_age(age: int, retire_age: int) -> str:
    return "working" if age < retire_age else "retirement"


def _state_bracket_coords(state_coords: np.ndarray,
                          shift: np.ndarray,
                          L_inv: np.ndarray) -> np.ndarray:
    ds = np.asarray(state_coords, dtype=float) - np.asarray(shift, dtype=float)
    return ds @ np.asarray(L_inv, dtype=float).T


def _summarize_rows(ages: np.ndarray,
                    alive: np.ndarray,
                    b_coords: np.ndarray,
                    halfwidths: np.ndarray,
                    retire_age: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outside_axis = np.abs(b_coords) > halfwidths.reshape(1, 1, 3)
    outside_joint = np.any(outside_axis, axis=2)
    wealth_mask = np.asarray(alive, dtype=bool)

    rows: list[dict[str, Any]] = []
    for t, age in enumerate(ages):
        alive_t = wealth_mask[:, t]
        n_alive = int(alive_t.sum())
        if n_alive == 0:
            rows.append({
                "age": int(age),
                "phase": _phase_for_age(int(age), retire_age),
                "alive": 0,
                "outside_joint": np.nan,
                "outside_u0": np.nan,
                "outside_u1": np.nan,
                "outside_u2": np.nan,
                "p95_abs_u0": np.nan,
                "p95_abs_u1": np.nan,
                "p95_abs_u2": np.nan,
            })
            continue

        b_t = b_coords[alive_t, t, :]
        out_t = outside_axis[alive_t, t, :]
        rows.append({
            "age": int(age),
            "phase": _phase_for_age(int(age), retire_age),
            "alive": n_alive,
            "outside_joint": float(np.any(out_t, axis=1).mean()),
            "outside_u0": float(out_t[:, 0].mean()),
            "outside_u1": float(out_t[:, 1].mean()),
            "outside_u2": float(out_t[:, 2].mean()),
            "p95_abs_u0": float(np.quantile(np.abs(b_t[:, 0]), 0.95)),
            "p95_abs_u1": float(np.quantile(np.abs(b_t[:, 1]), 0.95)),
            "p95_abs_u2": float(np.quantile(np.abs(b_t[:, 2]), 0.95)),
        })

    alive_flat = wealth_mask.reshape(-1)
    out_joint_flat = outside_joint.reshape(-1)[alive_flat]
    out_axis_flat = outside_axis.reshape(-1, 3)[alive_flat]
    b_flat = b_coords.reshape(-1, 3)[alive_flat]

    summary = {
        "overall_joint": float(out_joint_flat.mean()) if out_joint_flat.size else np.nan,
        "overall_u0": float(out_axis_flat[:, 0].mean()) if out_axis_flat.size else np.nan,
        "overall_u1": float(out_axis_flat[:, 1].mean()) if out_axis_flat.size else np.nan,
        "overall_u2": float(out_axis_flat[:, 2].mean()) if out_axis_flat.size else np.nan,
        "overall_p95_abs_u0": float(np.quantile(np.abs(b_flat[:, 0]), 0.95)) if b_flat.size else np.nan,
        "overall_p95_abs_u1": float(np.quantile(np.abs(b_flat[:, 1]), 0.95)) if b_flat.size else np.nan,
        "overall_p95_abs_u2": float(np.quantile(np.abs(b_flat[:, 2]), 0.95)) if b_flat.size else np.nan,
    }
    return rows, summary


def _phase_summary(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    phase_rows = [r for r in rows if r["phase"] == phase and r["alive"] > 0]
    if not phase_rows:
        return {
            "phase": phase,
            "alive_total": 0,
            "outside_joint": np.nan,
            "outside_u0": np.nan,
            "outside_u1": np.nan,
            "outside_u2": np.nan,
        }

    weights = np.array([r["alive"] for r in phase_rows], dtype=float)
    def wavg(key: str) -> float:
        vals = np.array([r[key] for r in phase_rows], dtype=float)
        return float(np.average(vals, weights=weights))

    return {
        "phase": phase,
        "alive_total": int(weights.sum()),
        "outside_joint": wavg("outside_joint"),
        "outside_u0": wavg("outside_u0"),
        "outside_u1": wavg("outside_u1"),
        "outside_u2": wavg("outside_u2"),
    }


def _fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100.0 * x:.2f}%"


def _fmt_num(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def _markdown(ctx,
              args: argparse.Namespace,
              halfwidths: np.ndarray,
              rows: list[dict[str, Any]],
              summary: dict[str, Any],
              phase_rows: list[dict[str, Any]],
              warm_start_label: str | None) -> str:
    lines: list[str] = []
    lines.append(f"# State-Clipping Report: `{ctx.label}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Bundle: `{ctx.path}`")
    lines.append(f"- Solved window: ages `{int(ctx.ages[ctx.first_solved_t])}-{int(ctx.ages[ctx.last_solved_t])}`")
    lines.append(f"- Simulation households: `{args.n_simulations}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- Return draw mode: `{args.return_draw_mode}`")
    lines.append(f"- Initial z: `{args.initial_z}`")
    lines.append(f"- Initial state: `{args.initial_state}`")
    lines.append(f"- Partial init mode: `{args.partial_init_mode}`")
    if warm_start_label is not None:
        lines.append(f"- Warm-start source: `{warm_start_label}`")
    lines.append(f"- Grid half-widths in bracket coords: `{tuple(float(x) for x in halfwidths)}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Joint outside share: `{_fmt_pct(summary['overall_joint'])}`")
    lines.append(f"- Axis 0 outside share: `{_fmt_pct(summary['overall_u0'])}`")
    lines.append(f"- Axis 1 outside share: `{_fmt_pct(summary['overall_u1'])}`")
    lines.append(f"- Axis 2 outside share: `{_fmt_pct(summary['overall_u2'])}`")
    lines.append(f"- P95 `|u|`: `({_fmt_num(summary['overall_p95_abs_u0'])}, {_fmt_num(summary['overall_p95_abs_u1'])}, {_fmt_num(summary['overall_p95_abs_u2'])})`")
    lines.append("")
    lines.append("## By Phase")
    lines.append("")
    lines.append("| phase | alive total | joint outside | axis0 | axis1 | axis2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in phase_rows:
        lines.append(
            f"| {r['phase']} | {r['alive_total']} | {_fmt_pct(r['outside_joint'])} | "
            f"{_fmt_pct(r['outside_u0'])} | {_fmt_pct(r['outside_u1'])} | {_fmt_pct(r['outside_u2'])} |"
        )
    lines.append("")
    lines.append("## By Age")
    lines.append("")
    lines.append("| age | phase | alive | joint outside | axis0 | axis1 | axis2 | p95 |u0| | p95 |u1| | p95 |u2| |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['age']} | {r['phase']} | {r['alive']} | {_fmt_pct(r['outside_joint'])} | "
            f"{_fmt_pct(r['outside_u0'])} | {_fmt_pct(r['outside_u1'])} | {_fmt_pct(r['outside_u2'])} | "
            f"{_fmt_num(r['p95_abs_u0'])} | {_fmt_num(r['p95_abs_u1'])} | {_fmt_num(r['p95_abs_u2'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", help="Policy bundle to evaluate.")
    p.add_argument("--model-bundle", required=True, help="Bundle with full run_config.")
    p.add_argument("--n-simulations", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--return-draw-mode", choices=("monte_carlo", "quadrature"), default="monte_carlo")
    p.add_argument("--initial-x", type=float, default=None)
    p.add_argument("--initial-wealth", type=float, default=0.1)
    p.add_argument(
        "--initial-wealth-distribution",
        choices=("point_mass", "normal"),
        default="point_mass",
    )
    p.add_argument("--initial-wealth-normal-std", type=float, default=0.0)
    p.add_argument(
        "--initial-z",
        choices=("median", "stationary", "normal"),
        default="median",
    )
    p.add_argument("--initial-z-normal-std", type=float, default=0.652)
    p.add_argument(
        "--initial-state",
        choices=("median", "stationary", "uniform"),
        default="median",
    )
    p.add_argument(
        "--partial-init-mode",
        choices=("centered", "warm_start"),
        default="centered",
    )
    p.add_argument("--markdown-out", type=str, default=None)
    args = p.parse_args()

    ctx = _load_bundle_context(
        bundle_path=Path(args.bundle),
        model_bundle=Path(args.model_bundle),
        eval_mode="same",
        ret_override=None,
        state_override=None,
        eta_override=None,
        eps_override=None,
    )

    warm_start = None
    warm_start_label = None
    if args.partial_init_mode == "warm_start":
        warm_start, warm_start_label = _maybe_build_warm_start(ctx, args)

    wealth_distribution = None if args.initial_wealth_distribution == "point_mass" else args.initial_wealth_distribution

    sim = _simulate_bundle_window(
        ctx=ctx,
        n_simulations=args.n_simulations,
        seed=args.seed,
        return_draw_mode=args.return_draw_mode,
        initial_x=args.initial_x,
        initial_wealth=args.initial_wealth,
        initial_wealth_distribution=wealth_distribution,
        initial_wealth_normal_std=args.initial_wealth_normal_std,
        initial_z=args.initial_z,
        initial_z_normal_std=args.initial_z_normal_std,
        initial_state=args.initial_state,
        warm_start=warm_start,
    )

    halfwidths = np.array([float(np.max(np.abs(g))) for g in ctx.pc_policy.state_bracket_grids], dtype=float)
    b_coords = _state_bracket_coords(
        sim["state_coords"],
        ctx.pc_policy.state_bracket_shift,
        ctx.pc_policy.state_bracket_L_inv,
    )
    rows, summary = _summarize_rows(
        ages=np.asarray(sim["ages"], dtype=int),
        alive=np.asarray(sim["alive"], dtype=bool),
        b_coords=b_coords,
        halfwidths=halfwidths,
        retire_age=int(ctx.model.retire_age),
    )

    phases = sorted(set(r["phase"] for r in rows))
    phase_rows = [_phase_summary(rows, phase) for phase in phases]

    print("=" * 88)
    print("Bundle State-Clipping Diagnostic")
    print("=" * 88)
    print(f"Bundle          : {ctx.path}")
    print(f"Solved window   : {int(ctx.ages[ctx.first_solved_t])}-{int(ctx.ages[ctx.last_solved_t])}")
    print(f"Grid halfwidths : {tuple(float(x) for x in halfwidths)}")
    print(f"Sim households  : {args.n_simulations}")
    print(f"Joint outside   : {_fmt_pct(summary['overall_joint'])}")
    print(f"Axis outside    : u0={_fmt_pct(summary['overall_u0'])}  u1={_fmt_pct(summary['overall_u1'])}  u2={_fmt_pct(summary['overall_u2'])}")
    print(f"P95 |u|         : ({_fmt_num(summary['overall_p95_abs_u0'])}, {_fmt_num(summary['overall_p95_abs_u1'])}, {_fmt_num(summary['overall_p95_abs_u2'])})")
    print()
    print("By phase")
    for r in phase_rows:
        print(
            f"  {r['phase']:<10} alive={r['alive_total']:>6}  "
            f"joint={_fmt_pct(r['outside_joint']):>8}  "
            f"u0={_fmt_pct(r['outside_u0']):>8}  "
            f"u1={_fmt_pct(r['outside_u1']):>8}  "
            f"u2={_fmt_pct(r['outside_u2']):>8}"
        )
    print()
    print("Selected ages")
    for r in rows:
        if r["alive"] <= 0:
            continue
        if r["age"] in {int(sim["ages"][0]), int(sim["ages"][1]), 75, 85, 95, int(sim["ages"][-1])}:
            print(
                f"  age {r['age']:>2}: alive={r['alive']:>4}  joint={_fmt_pct(r['outside_joint'])}  "
                f"u0={_fmt_pct(r['outside_u0'])}  u1={_fmt_pct(r['outside_u1'])}  u2={_fmt_pct(r['outside_u2'])}"
            )

    if args.markdown_out:
        out = Path(args.markdown_out)
        out.write_text(_markdown(ctx, args, halfwidths, rows, summary, phase_rows, warm_start_label), encoding="utf-8")
        print(f"\nWrote markdown report to {out}")


if __name__ == "__main__":
    main()

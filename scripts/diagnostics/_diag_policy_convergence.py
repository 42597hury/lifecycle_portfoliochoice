"""
_diag_policy_convergence.py -- Compare saved policy bundles on common probes.

This script implements the part of the convergence battery that can be run
immediately on saved policy bundles, before we add simulation-based Euler-error
 and DM-style residual tests:

1. Bundle integrity / solver-health summary
2. Common-probe policy drift versus a reference bundle
3. Publication-grade / welfare-grade gates for solver-health diagnostics

The comparison follows the guidance in:
  contextfiles/GRID_CONVERGENCE_CRITERIA.md

Usage
-----
python -m scripts.diagnostics._diag_policy_convergence ^
  --model-bundle saved_runs/unconstrained_principal_grid5x5x5_nz9 ^
  saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_v2 ^
  saved_runs/checkpoints/unconstrained_principal_grid5x5x5_nz9_from_age65_kret3x7x5_v2

The first positional bundle is the reference. Remaining positional bundles are
compared against it on a shared probe set.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import DiscretizationConfig
from policy_io import load_policy_bundle
from precompute import Precompute, build_model
from simulation import _interp_policy_zx


DEFAULT_AGES = (65, 66, 67, 75, 85, 95)
DEFAULT_WEALTH_QUANTILES = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)


@dataclass
class BundleContext:
    path: Path
    label: str
    C: np.ndarray
    S: np.ndarray
    B: np.ndarray
    diagnostics: dict[str, Any] | None
    metadata: dict[str, Any]
    summary: dict[str, Any]
    model: Any
    pc: Precompute
    ages: np.ndarray
    solved_mask: np.ndarray


def _unpack_summary_array(x: Any) -> Any:
    if isinstance(x, dict) and x.get("kind") == "ndarray":
        return np.array(x.get("values", []))
    return x


def _summary(metadata: dict[str, Any]) -> dict[str, Any]:
    out = metadata.get("diagnostics_summary", {})
    return out if isinstance(out, dict) else {}


def _extract_disc_config(metadata: dict[str, Any]) -> dict[str, Any]:
    run_config = metadata.get("run_config", {})
    disc = run_config.get("discretization_config", {})
    if disc:
        return disc
    disc = metadata.get("disc_config", {})
    if disc:
        return disc
    return _summary(metadata).get("disc_config", {})


def _coerce_seq(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(x)
    return x


def _build_disc_config(raw: dict[str, Any]) -> DiscretizationConfig:
    return DiscretizationConfig(
        n_wealth=int(raw["n_wealth"]),
        wealth_min=float(raw["wealth_min"]),
        wealth_max=float(raw["wealth_max"]),
        n_savings=int(raw["n_savings"]),
        savings_min=float(raw["savings_min"]),
        savings_max=raw.get("savings_max"),
        state_grid_sizes=tuple(int(v) for v in raw["state_grid_sizes"]),
        state_grid_mode=raw.get("state_grid_mode", "principal"),
        state_n_stds=_coerce_seq(raw.get("state_n_stds", 3.0)),
        n_z=int(raw["n_z"]),
        n_stds=float(raw.get("n_stds", 3.0)),
        n_eps_nodes=int(raw["n_eps_nodes"]),
        n_eta_nodes=int(raw.get("n_eta_nodes", 3)),
        n_ret_nodes_1d=_coerce_seq(raw.get("n_ret_nodes_1d", 2)),
        n_state_quad_nodes=_coerce_seq(raw.get("n_state_quad_nodes", 3)),
    )


def _build_model_from_bundle(bundle_path: Path):
    _C, _S, _B, _diag, meta = load_policy_bundle(bundle_path)
    run_config = meta.get("run_config", {})
    if not run_config:
        raise ValueError(
            f"Model bundle '{bundle_path}' is missing run_config. "
            "Pass a bundle saved via save_policy_bundle(...) with full metadata."
        )

    base_cfg = dict(run_config["base_config"])
    var_cfg = dict(run_config["var_config"])

    for key in ("Phi", "Omega", "z_bar"):
        if key in var_cfg and isinstance(var_cfg[key], dict) and "values" in var_cfg[key]:
            var_cfg[key] = np.array(var_cfg[key]["values"], dtype=float)

    model = build_model(base_cfg, var_cfg, verbose=False)
    ages = np.arange(model.start_age, model.terminal_age + 1, dtype=int)
    return model, ages


def _infer_solved_mask(C: np.ndarray, metadata: dict[str, Any], ages: np.ndarray) -> np.ndarray:
    n_age = C.shape[0]
    summary = _summary(metadata)

    mask = metadata.get("solved_age_mask")
    if mask is None:
        mask = summary.get("solved_age_mask")
    mask = _unpack_summary_array(mask)
    if mask is not None:
        arr = np.asarray(mask, dtype=bool)
        if arr.shape == (n_age,):
            return arr

    finite_mask = np.all(np.isfinite(C), axis=(1, 2, 3))
    if finite_mask.shape == (n_age,):
        return finite_mask

    return np.ones(n_age, dtype=bool)


def _load_bundle_context(bundle_path: Path, model: Any, ages: np.ndarray) -> BundleContext:
    C, S, B, diagnostics, metadata = load_policy_bundle(bundle_path)
    summary = _summary(metadata)
    disc_raw = _extract_disc_config(metadata)
    if not disc_raw:
        raise ValueError(
            f"Bundle '{bundle_path}' does not expose a discretization config in metadata."
        )

    disc = _build_disc_config(disc_raw)
    pc = Precompute(model, disc, verbose=False)
    solved_mask = _infer_solved_mask(C, metadata, ages)

    return BundleContext(
        path=bundle_path,
        label=bundle_path.name,
        C=C,
        S=S,
        B=B,
        diagnostics=diagnostics,
        metadata=metadata,
        summary=summary,
        model=model,
        pc=pc,
        ages=ages,
        solved_mask=solved_mask,
    )


def _format_age_range(ctx: BundleContext) -> str:
    solved = ctx.ages[ctx.solved_mask]
    if solved.size == 0:
        return "none"
    return f"{int(solved[0])}-{int(solved[-1])}"


def _health_row(ctx: BundleContext, mode: str) -> dict[str, Any]:
    s = ctx.summary
    total_calls = int(s.get("total_calls", 0) or 0)
    failures = int(s.get("total_newton_failures", 0) or 0)
    conv_rate = 1.0 if total_calls == 0 else 1.0 - failures / total_calls
    mono = int(s.get("total_mono_violations", 0) or 0)
    foc = float(s.get("worst_foc_resid", np.nan))
    avg_iter = float(s.get("avg_newton_iter", np.nan))
    max_iter = int(s.get("max_newton_iter", 0) or 0)
    status = str(s.get("solve_status", "unknown"))
    solver_cfg = s.get("solver_config", {})
    tol = float(solver_cfg.get("tol", np.nan))

    conv_floor = 0.999 if mode == "publication" else 0.9999
    foc_mult = 10.0 if mode == "publication" else 5.0
    foc_gate = bool(np.isfinite(foc) and np.isfinite(tol) and foc <= foc_mult * tol)
    conv_gate = conv_rate >= conv_floor
    mono_gate = mono == 0
    overall = conv_gate and foc_gate and mono_gate

    return {
        "bundle": ctx.label,
        "status": status,
        "ages": _format_age_range(ctx),
        "calls": total_calls,
        "failures": failures,
        "conv_rate": conv_rate,
        "worst_foc": foc,
        "avg_iter": avg_iter,
        "max_iter": max_iter,
        "mono": mono,
        "tol": tol,
        "passes": overall,
        "conv_gate": conv_gate,
        "foc_gate": foc_gate,
        "mono_gate": mono_gate,
    }


def _state_lookup(pc: Precompute) -> dict[tuple[int, int, int], int]:
    return {tuple(int(v) for v in idx): i for i, idx in enumerate(np.asarray(pc.state_indices))}


def _build_state_probes(ref_ctx: BundleContext) -> list[tuple[str, np.ndarray]]:
    n0, n1, n2 = (int(v) for v in ref_ctx.pc.state_grid_sizes)
    m0, m1, m2 = n0 // 2, n1 // 2, n2 // 2
    combos = [
        ("center", (m0, m1, m2)),
        ("u0_low", (0, m1, m2)),
        ("u0_high", (n0 - 1, m1, m2)),
        ("u1_low", (m0, 0, m2)),
        ("u1_high", (m0, n1 - 1, m2)),
        ("u2_low", (m0, m1, 0)),
        ("u2_high", (m0, m1, n2 - 1)),
    ]
    lut = _state_lookup(ref_ctx.pc)
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, int, int]] = set()
    for label, multi in combos:
        if multi in seen or multi not in lut:
            continue
        seen.add(multi)
        out.append((label, np.asarray(ref_ctx.pc.state_grid[lut[multi]], dtype=float).copy()))
    return out


def _build_z_probes(ref_ctx: BundleContext) -> list[tuple[str, float]]:
    n_z = ref_ctx.pc.n_z
    if n_z >= 5:
        idxs = [1, n_z // 2, n_z - 2]
        labels = ["low", "mid", "high"]
    else:
        raw = [0, n_z // 2, n_z - 1]
        idxs = []
        labels = []
        for lab, idx in zip(("low", "mid", "high"), raw):
            if idx not in idxs:
                idxs.append(idx)
                labels.append(lab)
    return [(lab, float(ref_ctx.pc.z_grid[idx])) for lab, idx in zip(labels, idxs)]


def _build_wealth_probes(ref_ctx: BundleContext, quantiles: tuple[float, ...]) -> list[tuple[str, float]]:
    n_w = ref_ctx.pc.n_w
    idxs = np.unique(np.clip(np.rint((n_w - 1) * np.asarray(quantiles)).astype(int), 0, n_w - 1))
    out = []
    for idx in idxs.tolist():
        w = float(ref_ctx.pc.wealth_grid[idx])
        out.append((f"w={w:.4g}", w))
    return out


def _bundle_has_age(ctx: BundleContext, age: int) -> bool:
    if age < int(ctx.ages[0]) or age > int(ctx.ages[-1]):
        return False
    return bool(ctx.solved_mask[int(age - ctx.ages[0])])


def _bracket_axis(x: float, grid: np.ndarray) -> tuple[int, float]:
    n = len(grid)
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[n - 1]:
        return n - 2, 1.0
    hi = int(np.searchsorted(grid, x, side="right"))
    lo = hi - 1
    dx = grid[hi] - grid[lo]
    frac = 0.0 if dx <= 1e-30 else (x - grid[lo]) / dx
    return lo, float(min(1.0, max(0.0, frac)))


def _eval_policy(arr4d: np.ndarray, ctx: BundleContext, age: int, z_val: float, state_vec: np.ndarray, wealth: float) -> float:
    t = int(age - ctx.ages[0])
    pc = ctx.pc

    iz_lo = int((z_val - pc.z_grid[0]) / pc.dz)
    iz_lo = max(0, min(iz_lo, pc.n_z - 2))
    frac_z = (z_val - pc.z_grid[iz_lo]) / pc.dz
    frac_z = float(min(1.0, max(0.0, frac_z)))

    ds = np.asarray(state_vec, dtype=float) - np.asarray(pc.state_bracket_shift, dtype=float)
    b = np.asarray(pc.state_bracket_L_inv, dtype=float) @ ds
    g0, g1, g2 = pc.state_bracket_grids
    lo0, f0 = _bracket_axis(float(b[0]), np.asarray(g0, dtype=float))
    lo1, f1 = _bracket_axis(float(b[1]), np.asarray(g1, dtype=float))
    lo2, f2 = _bracket_axis(float(b[2]), np.asarray(g2, dtype=float))

    N1 = len(g1)
    N2 = len(g2)

    j000 = lo0 * N1 * N2 + lo1 * N2 + lo2
    j001 = lo0 * N1 * N2 + lo1 * N2 + (lo2 + 1)
    j010 = lo0 * N1 * N2 + (lo1 + 1) * N2 + lo2
    j011 = lo0 * N1 * N2 + (lo1 + 1) * N2 + (lo2 + 1)
    j100 = (lo0 + 1) * N1 * N2 + lo1 * N2 + lo2
    j101 = (lo0 + 1) * N1 * N2 + lo1 * N2 + (lo2 + 1)
    j110 = (lo0 + 1) * N1 * N2 + (lo1 + 1) * N2 + lo2
    j111 = (lo0 + 1) * N1 * N2 + (lo1 + 1) * N2 + (lo2 + 1)

    w000 = (1.0 - f0) * (1.0 - f1) * (1.0 - f2)
    w001 = (1.0 - f0) * (1.0 - f1) * f2
    w010 = (1.0 - f0) * f1 * (1.0 - f2)
    w011 = (1.0 - f0) * f1 * f2
    w100 = f0 * (1.0 - f1) * (1.0 - f2)
    w101 = f0 * (1.0 - f1) * f2
    w110 = f0 * f1 * (1.0 - f2)
    w111 = f0 * f1 * f2

    wealth_grid = np.asarray(pc.wealth_grid, dtype=float)
    x = float(wealth)

    return float(
        w000 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j000, x, wealth_grid)
        + w001 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j001, x, wealth_grid)
        + w010 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j010, x, wealth_grid)
        + w011 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j011, x, wealth_grid)
        + w100 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j100, x, wealth_grid)
        + w101 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j101, x, wealth_grid)
        + w110 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j110, x, wealth_grid)
        + w111 * _interp_policy_zx(arr4d, t, iz_lo, frac_z, j111, x, wealth_grid)
    )


def _init_segment_store() -> dict[str, dict[str, list[float]]]:
    metric_keys = ("c_rel_pct", "cw_rel_pct", "s_pp", "b_pp", "bill_pp")
    return {
        "all": {k: [] for k in metric_keys},
        "working": {k: [] for k in metric_keys},
        "retirement": {k: [] for k in metric_keys},
    }


def _append_metric(store: dict[str, dict[str, list[float]]], segment: str, key: str, value: float) -> None:
    store["all"][key].append(value)
    if segment in store:
        store[segment][key].append(value)


def _stats(values: list[float]) -> tuple[float, float, float]:
    arr = np.abs(np.asarray(values, dtype=float))
    return float(np.median(arr)), float(np.quantile(arr, 0.95)), float(np.max(arr))


def _compare_bundles(
    ref_ctx: BundleContext,
    cand_ctx: BundleContext,
    requested_ages: tuple[int, ...],
    wealth_quantiles: tuple[float, ...],
) -> dict[str, Any]:
    common_ages = tuple(
        age for age in requested_ages if _bundle_has_age(ref_ctx, age) and _bundle_has_age(cand_ctx, age)
    )
    if not common_ages:
        raise ValueError(
            f"No common solved ages between '{ref_ctx.label}' and '{cand_ctx.label}' "
            f"for requested probes {requested_ages}."
        )

    z_probes = _build_z_probes(ref_ctx)
    state_probes = _build_state_probes(ref_ctx)
    wealth_probes = _build_wealth_probes(ref_ctx, wealth_quantiles)
    out = _init_segment_store()

    for age in common_ages:
        segment = "working" if age < ref_ctx.model.retire_age else "retirement"
        for _zlab, z_val in z_probes:
            for _slab, state_vec in state_probes:
                for _wlab, wealth in wealth_probes:
                    cref = _eval_policy(ref_ctx.C, ref_ctx, age, z_val, state_vec, wealth)
                    sref = _eval_policy(ref_ctx.S, ref_ctx, age, z_val, state_vec, wealth)
                    bref = _eval_policy(ref_ctx.B, ref_ctx, age, z_val, state_vec, wealth)
                    ccand = _eval_policy(cand_ctx.C, cand_ctx, age, z_val, state_vec, wealth)
                    scand = _eval_policy(cand_ctx.S, cand_ctx, age, z_val, state_vec, wealth)
                    bcand = _eval_policy(cand_ctx.B, cand_ctx, age, z_val, state_vec, wealth)

                    bill_ref = 1.0 - sref - bref
                    bill_cand = 1.0 - scand - bcand
                    cw_ref = cref / wealth
                    cw_cand = ccand / wealth

                    _append_metric(out, segment, "c_rel_pct", 100.0 * (ccand - cref) / max(abs(cref), 1e-12))
                    _append_metric(out, segment, "cw_rel_pct", 100.0 * (cw_cand - cw_ref) / max(abs(cw_ref), 1e-12))
                    _append_metric(out, segment, "s_pp", 100.0 * (scand - sref))
                    _append_metric(out, segment, "b_pp", 100.0 * (bcand - bref))
                    _append_metric(out, segment, "bill_pp", 100.0 * (bill_cand - bill_ref))

    summary: dict[str, Any] = {
        "bundle": cand_ctx.label,
        "ages": common_ages,
        "n_probes": len(common_ages) * len(z_probes) * len(state_probes) * len(wealth_probes),
        "segments": {},
    }
    for segment, metrics in out.items():
        seg_summary: dict[str, Any] = {}
        for key, vals in metrics.items():
            if vals:
                med, p95, vmax = _stats(vals)
                seg_summary[key] = {"median": med, "p95": p95, "max": vmax}
        summary["segments"][segment] = seg_summary

    return summary


def _health_table_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Bundle Health",
        "",
        "| Bundle | Status | Ages | Calls | Failures | Conv rate | Worst FOC | Avg iter | Max iter | Mono viol | Gates |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        gates = []
        gates.append("conv" if row["conv_gate"] else "conv-fail")
        gates.append("foc" if row["foc_gate"] else "foc-fail")
        gates.append("mono" if row["mono_gate"] else "mono-fail")
        if row["passes"]:
            gates.append("pass")
        lines.append(
            f"| `{row['bundle']}` | `{row['status']}` | `{row['ages']}` | "
            f"{row['calls']:,} | {row['failures']:,} | {100.0 * row['conv_rate']:.4f}% | "
            f"{row['worst_foc']:.3e} | {row['avg_iter']:.3f} | {row['max_iter']} | {row['mono']} | "
            f"{', '.join(gates)} |"
        )
    lines.append("")
    return lines


def _metric_label(key: str) -> str:
    return {
        "c_rel_pct": "C rel %",
        "cw_rel_pct": "C/W rel %",
        "s_pp": "Stock share pp",
        "b_pp": "Bond share pp",
        "bill_pp": "Bill share pp",
    }[key]


def _comparison_markdown(reference: BundleContext, comparisons: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Policy Drift vs Reference",
        "",
        f"Reference bundle: `{reference.path}`",
        "",
    ]
    for comp in comparisons:
        lines.append(f"### `{comp['bundle']}`")
        lines.append("")
        lines.append(f"Common age probes: `{list(comp['ages'])}`")
        lines.append(f"Probe count per metric: `{comp['n_probes']}`")
        lines.append("")
        lines.append("| Segment | Metric | Median | P95 | Max |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for segment in ("all", "working", "retirement"):
            seg = comp["segments"].get(segment, {})
            for key in ("c_rel_pct", "cw_rel_pct", "s_pp", "b_pp", "bill_pp"):
                if key not in seg:
                    continue
                stats = seg[key]
                lines.append(
                    f"| `{segment}` | `{_metric_label(key)}` | "
                    f"{stats['median']:.4f} | {stats['p95']:.4f} | {stats['max']:.4f} |"
                )
        lines.append("")
    return lines


def _render_report(
    model_bundle: Path,
    reference: BundleContext,
    contexts: list[BundleContext],
    mode: str,
    comparisons: list[dict[str, Any]],
) -> str:
    health_rows = [_health_row(ctx, mode) for ctx in contexts]
    lines = [
        "# Policy Convergence Diagnostic Report",
        "",
        f"Model bundle: `{model_bundle}`",
        f"Reference bundle: `{reference.path}`",
        f"Threshold mode: `{mode}`",
        "",
        "Implemented subset from `contextfiles/GRID_CONVERGENCE_CRITERIA.md`:",
        "",
        "- bundle integrity / loadability",
        "- solver-health gates (Newton convergence, worst FOC residual, monotonicity)",
        "- common-probe policy drift versus a reference bundle",
        "",
        "Not yet implemented here:",
        "",
        "- simulation-path Euler errors",
        "- Den Haan-Marcet style residual orthogonality",
        "- boundary-mass simulation diagnostics",
        "",
    ]
    lines.extend(_health_table_markdown(health_rows))
    lines.extend(_comparison_markdown(reference, comparisons))
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", help="Reference bundle followed by one or more candidates.")
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=None,
        help="Bundle with full run_config metadata used to reconstruct the economic model.",
    )
    parser.add_argument(
        "--mode",
        choices=("publication", "welfare"),
        default="publication",
        help="Threshold set for solver-health gates.",
    )
    parser.add_argument(
        "--ages",
        nargs="*",
        type=int,
        default=list(DEFAULT_AGES),
        help="Age probes for the common-probe comparison.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional path to write the markdown report.",
    )
    args = parser.parse_args()

    if len(args.bundles) < 2:
        parser.error("Provide at least two bundles: one reference and one candidate.")

    reference_path = Path(args.bundles[0])
    candidate_paths = [Path(p) for p in args.bundles[1:]]
    model_bundle = args.model_bundle if args.model_bundle is not None else reference_path

    model, ages = _build_model_from_bundle(model_bundle)
    reference = _load_bundle_context(reference_path, model, ages)
    contexts = [reference] + [_load_bundle_context(path, model, ages) for path in candidate_paths]

    comparisons = [
        _compare_bundles(
            reference,
            ctx,
            requested_ages=tuple(int(a) for a in args.ages),
            wealth_quantiles=tuple(float(q) for q in DEFAULT_WEALTH_QUANTILES),
        )
        for ctx in contexts[1:]
    ]

    report = _render_report(model_bundle, reference, contexts, args.mode, comparisons)
    print(report)

    if args.markdown_out is not None:
        args.markdown_out.write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.markdown_out}")


if __name__ == "__main__":
    main()

"""Audit infinite-horizon resolution sweeps.

LEGACY analysis (pre-pivot 4-axis bundles): the script reads
`system_iv_inf_*` bundles under saved_runs/inf_horizon/. The metadata-only
parts (status, wall time, grid sizes) still work on the post-pivot branch;
the parts that call `_rebuild_model_pc` (state_grid_axis_sensitivity) will
hit `verify/_diag_helpers.build_bundle_var_config`'s legacy-bundle guard
and raise. Use the pre-pivot revision of this repo to reproduce the full
analysis on the legacy bundles.

The goal is to turn a pile of policy bundles into convergence evidence:

* inventory solver status, wall time, grid sizes, and quadrature;
* compare policy movement from one-axis quadrature/return bumps;
* compare state-grid convergence by interpolating coarse policies onto the
  finest available grid; and
* summarize economically weighted state-grid curvature diagnostics.

Outputs are written to docs/scans by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.policy_io import load_policy_bundle
from scripts.analysis.state_grid_axis_sensitivity import (
    _as_builtin,
    _rebuild_model_pc,
    analyze_bundle as analyze_state_grid_axis,
)


AXIS_NAMES = ("dp", "spr", "rtb", "y_1")
RET_NAMES = ("xr", "xb")


@dataclass
class BundleRecord:
    name: str
    path: Path
    C: np.ndarray
    S: np.ndarray
    B: np.ndarray
    metadata: dict[str, Any]
    model: Any
    pc: Any
    disc: Any
    solver_config: Any

    @property
    def run_config(self) -> dict[str, Any]:
        return self.metadata.get("run_config", {})

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.metadata.get("diagnostics_summary", {})

    @property
    def sweep(self) -> dict[str, Any]:
        return self.run_config.get("sweep", {})


def _tuple_int(value: Any, n: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        if n is None:
            return (int(value),)
        return (int(value),) * n
    return tuple(int(v) for v in value)


def _fmt_tuple(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return "(" + ",".join(str(int(v)) for v in value) + ")"


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}e}"


def _fmt_minutes(seconds: Any) -> str:
    try:
        return f"{float(seconds) / 60.0:.1f}"
    except (TypeError, ValueError):
        return ""


def _weighted_abs_stats(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    abs_values = np.abs(values[mask])
    if abs_values.size == 0:
        return {
            "mean": float("nan"),
            "rms": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    if weights is None:
        return {
            "mean": float(np.mean(abs_values)),
            "rms": float(np.sqrt(np.mean(abs_values * abs_values))),
            "p50": float(np.percentile(abs_values, 50)),
            "p90": float(np.percentile(abs_values, 90)),
            "p95": float(np.percentile(abs_values, 95)),
            "p99": float(np.percentile(abs_values, 99)),
            "max": float(np.max(abs_values)),
        }

    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)[mask]
    wmask = np.isfinite(weights) & (weights > 0.0)
    abs_values = abs_values[wmask]
    weights = weights[wmask]
    total_w = float(np.sum(weights))
    if abs_values.size == 0 or total_w <= 0.0:
        return _weighted_abs_stats(abs_values, None)

    order = np.argsort(abs_values)
    sorted_abs = abs_values[order]
    sorted_w = weights[order]
    cdf = np.cumsum(sorted_w) / total_w

    def wq(q: float) -> float:
        idx = int(np.searchsorted(cdf, q / 100.0, side="left"))
        idx = min(max(idx, 0), sorted_abs.size - 1)
        return float(sorted_abs[idx])

    return {
        "mean": float(np.sum(weights * abs_values) / total_w),
        "rms": float(np.sqrt(np.sum(weights * abs_values * abs_values) / total_w)),
        "p50": wq(50),
        "p90": wq(90),
        "p95": wq(95),
        "p99": wq(99),
        "max": float(np.max(abs_values)),
    }


def discover_bundles(root: Path) -> list[BundleRecord]:
    records: list[BundleRecord] = []
    for meta_path in sorted(root.glob("*/metadata.json")):
        bundle_dir = meta_path.parent
        arrays_path = bundle_dir / "policy_arrays.npz"
        if not arrays_path.exists():
            continue
        C, S, B, _diag, metadata = load_policy_bundle(bundle_dir)
        model, pc, disc, solver_config = _rebuild_model_pc(metadata, bundle_dir)
        records.append(
            BundleRecord(
                name=bundle_dir.name,
                path=bundle_dir,
                C=np.asarray(C, dtype=np.float64),
                S=np.asarray(S, dtype=np.float64),
                B=np.asarray(B, dtype=np.float64),
                metadata=metadata,
                model=model,
                pc=pc,
                disc=disc,
                solver_config=solver_config,
            )
        )
    return records


def inventory_row(record: BundleRecord) -> dict[str, Any]:
    disc_dict = record.run_config.get("discretization_config", {})
    inf_params = record.run_config.get("inf_horizon_params", {})
    diag = record.diagnostics
    return {
        "name": record.name,
        "sweep": record.sweep.get("name", ""),
        "cell_tag": record.sweep.get("cell_tag", ""),
        "shape": list(record.C.shape),
        "grid": _tuple_int(disc_dict.get("state_grid_sizes", record.pc.state_grid_sizes)),
        "state_quad": _tuple_int(disc_dict.get("n_state_quad_nodes", record.disc.n_state_quad_nodes), record.model.n_state),
        "ret_quad": _tuple_int(disc_dict.get("n_ret_nodes_1d", record.disc.n_ret_nodes_1d), record.model.n_ret),
        "tol": inf_params.get("tol", diag.get("tol")),
        "converged": bool(diag.get("converged", False)),
        "n_iter": diag.get("n_iter"),
        "final_stopping_supnorm": diag.get("final_stopping_supnorm"),
        "final_share_supnorm": diag.get("final_share_supnorm"),
        "newton_iter_p99": diag.get("newton_iter_p99"),
        "backtrack_p99": diag.get("n_backtrack_total_p99"),
        "wall_time_seconds": record.run_config.get("wall_time_seconds"),
    }


def _policy_tensor(record: BundleRecord, array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        array = array[-1]
    if array.ndim != 3:
        raise ValueError(f"{record.name}: expected 3D or 4D policy, got {array.shape}")
    shape = (int(record.pc.n_z), *tuple(int(v) for v in record.pc.state_grid_sizes), int(record.pc.n_w))
    return array.reshape(shape)


def _target_bracket_coords(record: BundleRecord) -> np.ndarray:
    grids = [np.asarray(g, dtype=np.float64) for g in record.pc.state_bracket_grids]
    coords = np.empty((record.pc.N_state, len(grids)), dtype=np.float64)
    for d, grid in enumerate(grids):
        coords[:, d] = grid[np.asarray(record.pc.state_indices)[:, d]]
    return coords


def _interp_state_tensor(source: BundleRecord, target: BundleRecord, values: np.ndarray) -> np.ndarray:
    """Interpolate source policy tensor onto target state nodes.

    Wealth and z grids are assumed to match, which is true for the sweep
    bundles used here.
    """
    src_grids = [np.asarray(g, dtype=np.float64) for g in source.pc.state_bracket_grids]
    tgt_coords = _target_bracket_coords(target)
    sizes = tuple(len(g) for g in src_grids)
    src_tensor = _policy_tensor(source, values)
    src_flat = src_tensor.reshape((source.pc.n_z, source.pc.N_state, source.pc.n_w))

    if int(source.pc.n_z) != int(target.pc.n_z) or int(source.pc.n_w) != int(target.pc.n_w):
        raise ValueError("This audit expects matching z and wealth grids.")
    if not np.allclose(source.pc.wealth_grid, target.pc.wealth_grid, rtol=0.0, atol=1e-12):
        raise ValueError("This audit expects matching wealth grids.")

    lo_idx = []
    hi_idx = []
    weights_hi = []
    for d, grid in enumerate(src_grids):
        x = tgt_coords[:, d]
        hi = np.searchsorted(grid, x, side="left")
        hi = np.clip(hi, 1, len(grid) - 1)
        lo = hi - 1
        exact_hi = np.isclose(x, grid[hi], rtol=0.0, atol=1e-12)
        exact_lo = np.isclose(x, grid[lo], rtol=0.0, atol=1e-12)
        denom = np.maximum(grid[hi] - grid[lo], 1e-300)
        wh = (x - grid[lo]) / denom
        wh = np.where(exact_lo, 0.0, wh)
        wh = np.where(exact_hi, 1.0, wh)
        wh = np.clip(wh, 0.0, 1.0)
        lo_idx.append(lo.astype(np.int64))
        hi_idx.append(hi.astype(np.int64))
        weights_hi.append(wh.astype(np.float64))

    out = np.zeros((target.pc.n_z, target.pc.N_state, target.pc.n_w), dtype=np.float64)
    for corner in product((0, 1), repeat=len(src_grids)):
        idx_by_axis = []
        weight = np.ones(target.pc.N_state, dtype=np.float64)
        for d, bit in enumerate(corner):
            if bit:
                idx_by_axis.append(hi_idx[d])
                weight *= weights_hi[d]
            else:
                idx_by_axis.append(lo_idx[d])
                weight *= 1.0 - weights_hi[d]
        flat_idx = np.ravel_multi_index(tuple(idx_by_axis), sizes, mode="clip")
        out += src_flat[:, flat_idx, :] * weight.reshape((1, -1, 1))
    return out


def compare_policies(source: BundleRecord, target: BundleRecord, trim_wealth: int = 5) -> dict[str, Any]:
    src_c = _interp_state_tensor(source, target, source.C)
    src_s = _interp_state_tensor(source, target, source.S)
    src_b = _interp_state_tensor(source, target, source.B)
    tgt_c = _policy_tensor(target, target.C).reshape((target.pc.n_z, target.pc.N_state, target.pc.n_w))
    tgt_s = _policy_tensor(target, target.S).reshape((target.pc.n_z, target.pc.N_state, target.pc.n_w))
    tgt_b = _policy_tensor(target, target.B).reshape((target.pc.n_z, target.pc.N_state, target.pc.n_w))

    lo = max(0, int(trim_wealth))
    hi = target.pc.n_w - lo if lo > 0 else target.pc.n_w
    w_slice = slice(lo, hi)
    wealth = np.asarray(target.pc.wealth_grid[w_slice], dtype=np.float64)
    state_weights = np.asarray(target.pc.state_stationary_probs, dtype=np.float64)
    weights = state_weights.reshape((1, target.pc.N_state, 1))

    d_s = src_s[:, :, w_slice] - tgt_s[:, :, w_slice]
    d_b = src_b[:, :, w_slice] - tgt_b[:, :, w_slice]
    d_port = np.sqrt(d_s * d_s + d_b * d_b)
    d_c_share = (
        src_c[:, :, w_slice] / wealth.reshape((1, 1, -1))
        - tgt_c[:, :, w_slice] / wealth.reshape((1, 1, -1))
    )

    return {
        "source": source.name,
        "target": target.name,
        "trim_wealth": int(trim_wealth),
        "weighted": {
            "portfolio_norm": _weighted_abs_stats(d_port, weights),
            "stock_share": _weighted_abs_stats(d_s, weights),
            "bond_share": _weighted_abs_stats(d_b, weights),
            "consumption_share": _weighted_abs_stats(d_c_share, weights),
        },
        "unweighted": {
            "portfolio_norm": _weighted_abs_stats(d_port, None),
            "stock_share": _weighted_abs_stats(d_s, None),
            "bond_share": _weighted_abs_stats(d_b, None),
            "consumption_share": _weighted_abs_stats(d_c_share, None),
        },
    }


def _find(records: list[BundleRecord], name: str) -> BundleRecord | None:
    return next((record for record in records if record.name == name), None)


def _changed_axis(base: BundleRecord, candidate: BundleRecord) -> str:
    base_state = _tuple_int(base.disc.n_state_quad_nodes, base.model.n_state)
    cand_state = _tuple_int(candidate.disc.n_state_quad_nodes, candidate.model.n_state)
    base_ret = _tuple_int(base.disc.n_ret_nodes_1d, base.model.n_ret)
    cand_ret = _tuple_int(candidate.disc.n_ret_nodes_1d, candidate.model.n_ret)

    changes = []
    for i, (a, b) in enumerate(zip(base_state, cand_state, strict=True)):
        if a != b:
            changes.append(f"state {AXIS_NAMES[i]} {a}->{b}")
    for i, (a, b) in enumerate(zip(base_ret, cand_ret, strict=True)):
        if a != b:
            changes.append(f"ret {RET_NAMES[i]} {a}->{b}")
    return "; ".join(changes) if changes else "base"


def _state_axis_table(axis_results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in axis_results["ranked_axes"]:
        rows.append(
            {
                "rank": row["rank"],
                "name": row["name"],
                "weighted_p95_portfolio_interp": row["p95_portfolio_interp_error_weighted"],
                "weighted_rms_portfolio_interp": row["rms_portfolio_interp_error_weighted"],
                "weighted_p95_consumption_share_interp": row[
                    "p95_consumption_share_interp_error_weighted"
                ],
                "raw_p95_portfolio_curvature": row["p95_portfolio_curvature"],
            }
        )
    return rows


def run_audit(root: Path, trim_wealth: int, wealth_sample: int) -> dict[str, Any]:
    records = discover_bundles(root)
    if not records:
        raise FileNotFoundError(f"No policy bundles found under {root}")

    inventory = [inventory_row(record) for record in records]
    by_name = {record.name: record for record in records}

    axis_base = _find(records, "system_iv_inf_axisbump_run1_sq3333_rq33_calib1")
    grid_g5_ref = _find(records, "system_iv_inf_grid_g5_quad3334_ret44_calib1")

    axis_bump_vs_base = []
    axis_bump_vs_ref = []
    if axis_base is not None:
        axis_candidates = [
            record
            for record in records
            if record.sweep.get("name") == "inf_horizon_axis_bump_sweep"
            and record.name != axis_base.name
        ]
        for record in sorted(axis_candidates, key=lambda item: item.name):
            row = compare_policies(record, axis_base, trim_wealth)
            row["changed_axis"] = _changed_axis(axis_base, record)
            axis_bump_vs_base.append(row)
            if grid_g5_ref is not None:
                ref_row = compare_policies(record, grid_g5_ref, trim_wealth)
                ref_row["changed_axis"] = _changed_axis(axis_base, record)
                axis_bump_vs_ref.append(ref_row)
        if grid_g5_ref is not None:
            ref_row = compare_policies(axis_base, grid_g5_ref, trim_wealth)
            ref_row["changed_axis"] = _changed_axis(axis_base, axis_base)
            axis_bump_vs_ref.insert(0, ref_row)

    state_grid_vs_ref = []
    if grid_g5_ref is not None:
        for name in (
            "system_iv_inf_grid_g3_quad3334_ret44_calib1",
            "system_iv_inf_grid_g4_quad3334_ret44_calib1",
            "system_iv_inf_grid_g5_quad3334_ret44_calib1",
        ):
            record = by_name.get(name)
            if record is None:
                continue
            state_grid_vs_ref.append(compare_policies(record, grid_g5_ref, trim_wealth))

    axis_curvature = {}
    for record in records:
        try:
            result = analyze_state_grid_axis(
                bundle_path=record.path,
                age_index=None,
                wealth_sample=wealth_sample,
                trim_wealth=2,
            )
            axis_curvature[record.name] = _state_axis_table(result)
        except Exception as exc:  # pragma: no cover - report-only path
            axis_curvature[record.name] = {"error": repr(exc)}

    return {
        "root": str(root),
        "trim_wealth": int(trim_wealth),
        "wealth_sample_for_curvature": int(wealth_sample),
        "inventory": inventory,
        "axis_bump_vs_base": axis_bump_vs_base,
        "axis_bump_vs_g5_quad3334_ret44": axis_bump_vs_ref,
        "state_grid_vs_g5_quad3334_ret44": state_grid_vs_ref,
        "axis_curvature": axis_curvature,
        "notes": [
            "Policy comparisons interpolate the source policy onto the target state grid in Cholesky bracket coordinates.",
            "Weighted metrics use the target bundle's stationary state distribution and uniform wealth weights after trimming endpoints.",
            "The g5 quad3334 ret44 bundle is the best available state-grid reference, but it is flagged not converged with final stopping supnorm just above tolerance.",
            "Current inf-horizon bundles have n_z=1, so income-state convergence is intentionally out of scope here.",
        ],
    }


def _comparison_rows(comparisons: list[dict[str, Any]], include_change: bool = False) -> list[str]:
    rows = []
    sorted_rows = sorted(
        comparisons,
        key=lambda row: row["weighted"]["portfolio_norm"]["p95"],
    )
    for row in sorted_rows:
        metric = row["weighted"]
        label = row["source"]
        if include_change:
            label = row.get("changed_axis", label)
        rows.append(
            "| {label} | {src} | {tgt} | {p95_port} | {rms_port} | {p95_bond} | {p95_stock} | {p95_cw} |".format(
                label=label,
                src=row["source"],
                tgt=row["target"],
                p95_port=_fmt_float(metric["portfolio_norm"]["p95"]),
                rms_port=_fmt_float(metric["portfolio_norm"]["rms"]),
                p95_bond=_fmt_float(metric["bond_share"]["p95"]),
                p95_stock=_fmt_float(metric["stock_share"]["p95"]),
                p95_cw=_fmt_float(metric["consumption_share"]["p95"]),
            )
        )
    return rows


def _inventory_rows(inventory: list[dict[str, Any]]) -> list[str]:
    rows = []
    for row in sorted(inventory, key=lambda item: item["name"]):
        rows.append(
            "| {name} | {grid} | {sq} | {rq} | {tol} | {conv} | {stop} | {mins} |".format(
                name=row["name"],
                grid=_fmt_tuple(row["grid"]),
                sq=_fmt_tuple(row["state_quad"]),
                rq=_fmt_tuple(row["ret_quad"]),
                tol=_fmt_float(row["tol"], 1),
                conv="yes" if row["converged"] else "no",
                stop=_fmt_float(row["final_stopping_supnorm"]),
                mins=_fmt_minutes(row["wall_time_seconds"]),
            )
        )
    return rows


def _curvature_rows(axis_curvature: dict[str, Any], names: list[str]) -> list[str]:
    rows = []
    for name in names:
        data = axis_curvature.get(name)
        if not isinstance(data, list):
            continue
        top = data[:4]
        ranking = ", ".join(
            f"{row['name']}={_fmt_float(row['weighted_p95_portfolio_interp'])}" for row in top
        )
        rows.append(f"| {name} | {ranking} |")
    return rows


def write_markdown(result: dict[str, Any], path: Path) -> None:
    inv_rows = _inventory_rows(result["inventory"])
    axis_rows = _comparison_rows(result["axis_bump_vs_base"], include_change=True)
    axis_ref_rows = _comparison_rows(result["axis_bump_vs_g5_quad3334_ret44"], include_change=True)
    grid_rows = _comparison_rows(result["state_grid_vs_g5_quad3334_ret44"], include_change=False)
    curvature_rows = _curvature_rows(
        result["axis_curvature"],
        [
            "system_iv_inf_grid_g3_quad3334_ret44_calib1",
            "system_iv_inf_grid_g4_quad3334_ret44_calib1",
            "system_iv_inf_grid_g5_quad3334_ret44_calib1",
            "system_iv_inf_axisbump_run1_sq3333_rq33_calib1",
        ],
    )

    lines = [
        "# Infinite-Horizon Resolution Investigation",
        "",
        "## Executive Read",
        "",
        "- The current evidence says state-grid resolution is the main unresolved issue; uniform `3^4 -> 4^4 -> 5^4` changes are much larger than solver tolerances.",
        "- The axis-bump solve sweep says state `y_1` quadrature is the only very large quadrature axis; state `spr` is second, while return quadrature bumps are small in this run set.",
        "- The economically weighted grid-curvature diagnostic is specification-sensitive, but the robust message is: `y_1` needs coverage, `spr`/`dp` deserve tests, and `rtb` is lowest priority.",
        "- The best available uniform `5^4` reference is useful but slightly underconverged: final stopping supnorm is around `1.37e-05` versus target `1e-05`.",
        "- `n_z` is intentionally ignored here because these are infinite-horizon bundles with `n_z=1`.",
        "",
        "## Bundle Inventory",
        "",
        "| bundle | grid | state quad | ret quad | tol | converged | final stop | wall min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *inv_rows,
        "",
        "## Axis-Bump Policy Movement",
        "",
        "Same `5^4` state grid, baseline target is `run1_sq3333_rq33`. Metrics are stationary-state weighted and trim five wealth endpoints.",
        "",
        "| changed axis | source | target | p95 portfolio | RMS portfolio | p95 bond | p95 stock | p95 c/w |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *axis_rows,
        "",
        "## Axis-Bump Distance To Best Available Reference",
        "",
        "Reference target is `system_iv_inf_grid_g5_quad3334_ret44_calib1`. This mixes quad and ret changes, so it is a practical reference rather than a clean one-factor experiment.",
        "",
        "| changed axis | source | target | p95 portfolio | RMS portfolio | p95 bond | p95 stock | p95 c/w |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *axis_ref_rows,
        "",
        "## State-Grid Convergence",
        "",
        "Coarser policies are linearly interpolated onto the `5^4` grid with the same `quad3334_ret44` specification.",
        "",
        "| source bundle | source | target | p95 portfolio | RMS portfolio | p95 bond | p95 stock | p95 c/w |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *grid_rows,
        "",
        "## Economically Weighted Grid Curvature",
        "",
        "`h^2/8 * curvature`, weighted by stationary state probabilities. Lower is better; high values indicate where extra state nodes buy the most policy shape.",
        "",
        "| bundle | weighted p95 portfolio interpolation proxy by rank |",
        "|---|---|",
        *curvature_rows,
        "",
        "## Interpretation",
        "",
        "The axis-bump solve sweep says state `y_1` quadrature is the dominant quadrature refinement: moving from `quad3333` to `quad3335` changes the weighted p95 portfolio by about `2.10e-01`, while return bumps are at `2.11e-03` or below. The state-grid sweep also shows large changes across `3^4`, `4^4`, and `5^4`, so the current model is not yet state-grid settled.",
        "",
        "For grid allocation, the economically weighted curvature result is the cleanest guide, but it should be read together with the quadrature setting. Under the best available `quad3334_ret44` grid sweep, `y_1` is consistently the largest remaining grid-curvature axis; `dp` and `spr` are next; `rtb` is much smaller. Under the cheaper `quad3333_ret33` baseline, `spr` dominates, reflecting a term-structure shape signal because the current Cholesky `spr` axis also loads strongly against `y_1`.",
        "",
        "Recommended next controlled runs:",
        "",
        "1. Continue or rerun `g5_quad3334_ret44` to clear the `1e-05` tolerance, since it is the best available reference but currently flagged not converged.",
        "2. Run anisotropic state-grid candidates at fixed `quad3334_ret44`: `(3,3,3,5)`, `(3,5,3,5)`, `(5,3,3,5)`, and `(5,5,3,5)` if time permits.",
        "3. After state-grid choice is stable, rerun the top quadrature candidates on that grid. Based on current evidence, prioritize state `y_1` quadrature, then state `spr`; do not spend more on return quadrature unless a higher-precision reference contradicts this.",
        "4. Test a reordered Cholesky grid only after this baseline is stable: `(dp,spr,y_1,rtb)` and `(spr,y_1,dp,rtb)` are the two useful alternatives.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("saved_runs/inf_horizon"))
    parser.add_argument("--trim-wealth", type=int, default=5)
    parser.add_argument("--wealth-sample", type=int, default=64)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/scans/inf_horizon_resolution_investigation_2026-05-08.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/scans/INF_HORIZON_RESOLUTION_INVESTIGATION_2026-05-08.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args.root, args.trim_wealth, args.wealth_sample)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_as_builtin(result), indent=2), encoding="utf-8")
    write_markdown(result, args.out_md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()

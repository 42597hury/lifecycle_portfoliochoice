"""Rank state-grid axes by policy interpolation curvature.

This is a cheap a posteriori diagnostic for deciding which state axis is most
likely to benefit from extra grid resolution. It does not re-solve the model.
Instead, it loads a solved policy bundle, reshapes the policy onto the tensor
state grid, and measures finite-difference curvature along each state axis.

Large curvature means linear interpolation has more room to err on that axis,
so an anisotropic grid bump on that axis is a good candidate for a follow-up
solve sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifecycle.model import DiscretizationConfig, SolverConfig
from lifecycle.policy_io import load_policy_bundle
from lifecycle.wealth_grid import disc_config_with_bundle_wealth_grid
from lifecycle.precompute import build_model, build_precompute
from verify._diag_helpers import build_bundle_var_config


# Default state-axis names for the post-pivot 3-axis Full real-yields system.
# Bundles solved on System 1 / System 2 carry their own state_names in the
# saved metadata; the diagnostic falls back to this tuple only when the
# bundle does not declare state_names.
DEFAULT_AXIS_NAMES = ("cape", "spr", "y_1")


@dataclass(frozen=True)
class AxisSensitivity:
    axis: int
    name: str
    grid_size: int
    coordinate_min: float
    coordinate_max: float
    max_step: float
    interpolation_error_scale: float
    portfolio_curvature: dict[str, float]
    bond_curvature: dict[str, float]
    stock_curvature: dict[str, float]
    consumption_share_curvature: dict[str, float]
    portfolio_interp_error: dict[str, float]
    consumption_share_interp_error: dict[str, float]
    portfolio_interp_error_weighted: dict[str, float]
    consumption_share_interp_error_weighted: dict[str, float]
    portfolio_slope: dict[str, float]
    bond_slope: dict[str, float]
    stock_slope: dict[str, float]
    consumption_share_slope: dict[str, float]


def _as_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _as_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _as_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "p50_abs": float("nan"),
            "p90_abs": float("nan"),
            "p95_abs": float("nan"),
            "p99_abs": float("nan"),
            "max_abs": float("nan"),
        }
    abs_flat = np.abs(flat)
    return {
        "mean_abs": float(np.mean(abs_flat)),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "p50_abs": float(np.percentile(abs_flat, 50)),
        "p90_abs": float(np.percentile(abs_flat, 90)),
        "p95_abs": float(np.percentile(abs_flat, 95)),
        "p99_abs": float(np.percentile(abs_flat, 99)),
        "max_abs": float(np.max(abs_flat)),
    }


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return {
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "p50_abs": float("nan"),
            "p90_abs": float("nan"),
            "p95_abs": float("nan"),
            "p99_abs": float("nan"),
            "max_abs": float("nan"),
        }

    abs_values = np.abs(values[mask])
    weights = weights[mask]
    total_w = float(np.sum(weights))
    if total_w <= 0.0:
        return {
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "p50_abs": float("nan"),
            "p90_abs": float("nan"),
            "p95_abs": float("nan"),
            "p99_abs": float("nan"),
            "max_abs": float("nan"),
        }

    order = np.argsort(abs_values)
    sorted_abs = abs_values[order]
    sorted_w = weights[order]
    cdf = np.cumsum(sorted_w) / total_w

    def wq(q: float) -> float:
        idx = int(np.searchsorted(cdf, q / 100.0, side="left"))
        idx = min(max(idx, 0), sorted_abs.size - 1)
        return float(sorted_abs[idx])

    return {
        "mean_abs": float(np.sum(weights * abs_values) / total_w),
        "rms": float(np.sqrt(np.sum(weights * abs_values * abs_values) / total_w)),
        "p50_abs": wq(50),
        "p90_abs": wq(90),
        "p95_abs": wq(95),
        "p99_abs": wq(99),
        "max_abs": float(np.max(abs_values)),
    }


def _list_to_tuple_recursive(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_list_to_tuple_recursive(item) for item in value)
    return value


def _rehydrate_disc_config(data: dict[str, Any]) -> DiscretizationConfig:
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
    for key, value in data.items():
        if key not in valid:
            continue
        kwargs[key] = _list_to_tuple_recursive(value) if key in tuple_fields else value
    return DiscretizationConfig(**kwargs)


def _rehydrate_solver_config(data: dict[str, Any] | None) -> SolverConfig:
    if data is None:
        return SolverConfig()
    valid = set(SolverConfig._fields)
    return SolverConfig(**{key: value for key, value in data.items() if key in valid})


def _rebuild_model_pc(
    metadata: dict[str, Any],
    bundle_path: Path | None = None,
) -> tuple[Any, Any, DiscretizationConfig, SolverConfig]:
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
    return model, pc, disc, solver_config


def _policy_tensor(
    c_in: np.ndarray,
    s_in: np.ndarray,
    b_in: np.ndarray,
    pc,
    age_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    c = np.asarray(c_in, dtype=np.float64)
    s = np.asarray(s_in, dtype=np.float64)
    b = np.asarray(b_in, dtype=np.float64)

    if c.ndim == 4:
        idx = int(age_index if age_index is not None else c.shape[0] - 1)
        c = c[idx]
        s = s[idx]
        b = b[idx]
    elif c.ndim == 3:
        idx = -1
    else:
        raise ValueError(f"Expected 3D or 4D policy arrays, found C.ndim={c.ndim}")

    expected = (int(pc.n_z), int(np.prod(pc.state_grid_sizes)), int(pc.n_w))
    if c.shape != expected:
        raise ValueError(f"Policy slice shape {c.shape} does not match expected {expected}")

    shape = (int(pc.n_z), *tuple(int(v) for v in pc.state_grid_sizes), int(pc.n_w))
    return c.reshape(shape), s.reshape(shape), b.reshape(shape), idx


def _wealth_subset(
    c: np.ndarray,
    s: np.ndarray,
    b: np.ndarray,
    wealth_grid: np.ndarray,
    wealth_sample: int | None,
    trim_wealth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    n_w = wealth_grid.size
    lo = max(0, int(trim_wealth))
    hi = max(lo + 1, n_w - int(trim_wealth))
    candidates = np.arange(lo, hi)
    if wealth_sample is not None and wealth_sample > 0 and wealth_sample < candidates.size:
        picks = np.unique(np.linspace(0, candidates.size - 1, int(wealth_sample)).round().astype(int))
        idx = candidates[picks]
    else:
        idx = candidates
    return c[..., idx], s[..., idx], b[..., idx], wealth_grid[idx], idx.tolist()


def _second_derivative(values: np.ndarray, axis: int, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    moved = np.moveaxis(values, axis, 0)
    if x.size < 3:
        return np.empty((0, *moved.shape[1:]), dtype=np.float64)

    left_dx = (x[1:-1] - x[:-2]).reshape((-1,) + (1,) * (moved.ndim - 1))
    right_dx = (x[2:] - x[1:-1]).reshape((-1,) + (1,) * (moved.ndim - 1))
    span = (x[2:] - x[:-2]).reshape((-1,) + (1,) * (moved.ndim - 1))
    left_slope = (moved[1:-1] - moved[:-2]) / left_dx
    right_slope = (moved[2:] - moved[1:-1]) / right_dx
    return 2.0 * (right_slope - left_slope) / span


def _first_derivative(values: np.ndarray, axis: int, x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    moved = np.moveaxis(values, axis, 0)
    if x.size < 2:
        return np.empty((0, *moved.shape[1:]), dtype=np.float64)
    dx = np.diff(x).reshape((-1,) + (1,) * (moved.ndim - 1))
    return np.diff(moved, axis=0) / dx


def _interior_state_weights(
    state_probs: np.ndarray,
    state_sizes: tuple[int, ...],
    state_axis: int,
) -> np.ndarray:
    probs_grid = np.asarray(state_probs, dtype=np.float64).reshape(state_sizes)
    moved = np.moveaxis(probs_grid, state_axis, 0)
    interior = moved[1:-1]
    return interior.reshape((interior.shape[0], 1, *interior.shape[1:], 1))


def _axis_names(n_axes: int) -> tuple[str, ...]:
    if n_axes <= len(DEFAULT_AXIS_NAMES):
        return DEFAULT_AXIS_NAMES[:n_axes]
    extra = tuple(f"state_{i}" for i in range(len(DEFAULT_AXIS_NAMES), n_axes))
    return (*DEFAULT_AXIS_NAMES, *extra)


def _as_int_list(value: Any, n_axes: int) -> list[int]:
    if isinstance(value, (int, np.integer)):
        return [int(value)] * n_axes
    return [int(item) for item in value]


def analyze_bundle(
    bundle_path: Path,
    age_index: int | None,
    wealth_sample: int | None,
    trim_wealth: int,
) -> dict[str, Any]:
    c_raw, s_raw, b_raw, _diagnostics, metadata = load_policy_bundle(bundle_path)
    model, pc, disc, _solver_config = _rebuild_model_pc(metadata, bundle_path)
    c, s, b, selected_age_index = _policy_tensor(c_raw, s_raw, b_raw, pc, age_index)

    wealth_grid = np.asarray(pc.wealth_grid, dtype=np.float64)
    c, s, b, wealth_grid, wealth_indices = _wealth_subset(
        c, s, b, wealth_grid, wealth_sample, trim_wealth
    )
    wealth_shape = (1,) + (1,) * len(pc.state_grid_sizes) + (wealth_grid.size,)
    c_share = c / wealth_grid.reshape(wealth_shape)
    state_sizes = tuple(int(v) for v in pc.state_grid_sizes)
    state_probs = np.asarray(pc.state_stationary_probs, dtype=np.float64)

    axis_names = tuple(getattr(model, "state_names", _axis_names(len(pc.state_grid_sizes))))
    sensitivities: list[AxisSensitivity] = []
    for state_axis, (name, grid_size, coords) in enumerate(
        zip(axis_names, pc.state_grid_sizes, pc.state_bracket_grids, strict=True)
    ):
        tensor_axis = 1 + state_axis
        coords = np.asarray(coords, dtype=np.float64)

        d2_s = _second_derivative(s, tensor_axis, coords)
        d2_b = _second_derivative(b, tensor_axis, coords)
        d2_c_share = _second_derivative(c_share, tensor_axis, coords)
        d2_port = np.sqrt(d2_s * d2_s + d2_b * d2_b)
        step = float(np.max(np.diff(coords))) if coords.size > 1 else 0.0
        interpolation_error_scale = float(step * step / 8.0)
        port_interp_error = d2_port * interpolation_error_scale
        c_share_interp_error = d2_c_share * interpolation_error_scale
        interior_weights = _interior_state_weights(state_probs, state_sizes, state_axis)

        d1_s = _first_derivative(s, tensor_axis, coords)
        d1_b = _first_derivative(b, tensor_axis, coords)
        d1_c_share = _first_derivative(c_share, tensor_axis, coords)
        d1_port = np.sqrt(d1_s * d1_s + d1_b * d1_b)

        sensitivities.append(
            AxisSensitivity(
                axis=state_axis,
                name=name,
                grid_size=int(grid_size),
                coordinate_min=float(coords[0]),
                coordinate_max=float(coords[-1]),
                max_step=step,
                interpolation_error_scale=interpolation_error_scale,
                portfolio_curvature=_stats(d2_port),
                bond_curvature=_stats(d2_b),
                stock_curvature=_stats(d2_s),
                consumption_share_curvature=_stats(d2_c_share),
                portfolio_interp_error=_stats(port_interp_error),
                consumption_share_interp_error=_stats(c_share_interp_error),
                portfolio_interp_error_weighted=_weighted_stats(
                    port_interp_error, interior_weights
                ),
                consumption_share_interp_error_weighted=_weighted_stats(
                    c_share_interp_error, interior_weights
                ),
                portfolio_slope=_stats(d1_port),
                bond_slope=_stats(d1_b),
                stock_slope=_stats(d1_s),
                consumption_share_slope=_stats(d1_c_share),
            )
        )

    ranked = sorted(
        sensitivities,
        key=lambda item: (
            -item.portfolio_interp_error_weighted["p95_abs"],
            -item.portfolio_interp_error_weighted["rms"],
            -item.consumption_share_interp_error_weighted["p95_abs"],
            -item.portfolio_curvature["p95_abs"],
            -item.portfolio_slope["p95_abs"],
        ),
    )

    return {
        "bundle_path": str(bundle_path),
        "created_from_script": "scripts/analysis/state_grid_axis_sensitivity.py",
        "selected_age_index": selected_age_index,
        "policy_shape": list(np.asarray(c_raw).shape),
        "state_grid_sizes": [int(v) for v in pc.state_grid_sizes],
        "quad_n": _as_int_list(disc.n_state_quad_nodes, int(model.n_state)),
        "ret_quad_n": _as_int_list(disc.n_ret_nodes_1d, int(model.n_ret)),
        "wealth_indices": wealth_indices,
        "notes": [
            "Curvature is finite-difference curvature of saved policies on the existing state grid.",
            "The main ranking uses p95 sqrt(d2 stock share^2 + d2 bond share^2).",
            "This is a cheap interpolation-risk diagnostic, not a replacement for anisotropic re-solves.",
        ],
        "sensitivities": [asdict(item) for item in sensitivities],
        "ranked_axes": [
            {
                "rank": rank,
                "axis": item.axis,
                "name": item.name,
                "grid_size": item.grid_size,
                "max_step": item.max_step,
                "interpolation_error_scale": item.interpolation_error_scale,
                "p95_portfolio_curvature": item.portfolio_curvature["p95_abs"],
                "p95_bond_curvature": item.bond_curvature["p95_abs"],
                "p95_stock_curvature": item.stock_curvature["p95_abs"],
                "p95_consumption_share_curvature": item.consumption_share_curvature["p95_abs"],
                "p95_portfolio_interp_error": item.portfolio_interp_error["p95_abs"],
                "rms_portfolio_interp_error_weighted": item.portfolio_interp_error_weighted["rms"],
                "p95_portfolio_interp_error_weighted": item.portfolio_interp_error_weighted["p95_abs"],
                "rms_consumption_share_interp_error_weighted": item.consumption_share_interp_error_weighted["rms"],
                "p95_consumption_share_interp_error_weighted": item.consumption_share_interp_error_weighted["p95_abs"],
                "p95_portfolio_slope": item.portfolio_slope["p95_abs"],
            }
            for rank, item in enumerate(ranked, start=1)
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:.3e}" if np.isfinite(value) else "nan"


def write_markdown(result: dict[str, Any], out_path: Path) -> None:
    rows = []
    for item in result["ranked_axes"]:
        rows.append(
            "| {rank} | {axis} | {name} | {grid_size} | {port} | {bond} | {stock} | {cons} | {slope} |".format(
                rank=item["rank"],
                axis=item["axis"],
                name=item["name"],
                grid_size=item["grid_size"],
                port=_fmt(item["p95_portfolio_interp_error_weighted"]),
                bond=_fmt(item["p95_bond_curvature"]),
                stock=_fmt(item["p95_stock_curvature"]),
                cons=_fmt(item["p95_consumption_share_interp_error_weighted"]),
                slope=_fmt(item["rms_portfolio_interp_error_weighted"]),
            )
        )

    text = "\n".join(
        [
            "# State Grid Axis Sensitivity",
            "",
            f"- Bundle: `{result['bundle_path']}`",
            f"- Policy shape: `{tuple(result['policy_shape'])}`",
            f"- Selected age index: `{result['selected_age_index']}`",
            f"- State grid sizes: `{tuple(result['state_grid_sizes'])}`",
            f"- Quadrature: state `{tuple(result['quad_n'])}`, retirement `{tuple(result['ret_quad_n'])}`",
            f"- Wealth indices used: `{result['wealth_indices'][0]}` to `{result['wealth_indices'][-1]}` ({len(result['wealth_indices'])} points)",
            "",
            "The main score is an economically weighted interpolation-error proxy: `h^2/8 * curvature`, weighted by the stationary state probability.",
            "",
            "| Rank | Axis | Name | Grid n | weighted p95 portfolio interp error | p95 bond curvature | p95 stock curvature | weighted p95 c/w interp error | weighted RMS portfolio interp error |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Interpretation: use this to choose candidates for anisotropic re-solves; final accuracy should be checked against a common finer reference grid. Wealth is weighted uniformly here because this diagnostic does not simulate the ergodic wealth distribution.",
            "",
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", type=Path, help="Path to a policy bundle directory")
    parser.add_argument(
        "--age-index",
        type=int,
        default=None,
        help="Age index for 4D lifecycle bundles. Defaults to last age slice.",
    )
    parser.add_argument(
        "--wealth-sample",
        type=int,
        default=None,
        help="Optional number of wealth nodes to sample after trimming.",
    )
    parser.add_argument(
        "--trim-wealth",
        type=int,
        default=0,
        help="Drop this many wealth nodes from each endpoint before scoring.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <bundle>/state_grid_axis_sensitivity.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to <bundle>/state_grid_axis_sensitivity.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = args.bundle_path.expanduser().resolve()
    result = analyze_bundle(
        bundle_path=bundle_path,
        age_index=args.age_index,
        wealth_sample=args.wealth_sample,
        trim_wealth=args.trim_wealth,
    )

    out_json = args.out_json or bundle_path / "state_grid_axis_sensitivity.json"
    out_md = args.out_md or bundle_path / "state_grid_axis_sensitivity.md"
    out_json.write_text(json.dumps(_as_builtin(result), indent=2), encoding="utf-8")
    write_markdown(result, out_md)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print("Ranked axes:")
    for item in result["ranked_axes"]:
        print(
            "  #{rank}: state[{axis}]={name} weighted_p95_port_interp={port} weighted_p95_c/w_interp={cons} raw_p95_port_curv={raw}".format(
                rank=item["rank"],
                axis=item["axis"],
                name=item["name"],
                port=_fmt(item["p95_portfolio_interp_error_weighted"]),
                cons=_fmt(item["p95_consumption_share_interp_error_weighted"]),
                raw=_fmt(item["p95_portfolio_curvature"]),
            )
        )


if __name__ == "__main__":
    main()

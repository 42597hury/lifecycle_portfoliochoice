"""Generate a curvature-adapted Bakhvalov wealth grid from a policy bundle.

This is an opt-in experiment tool. It does not change solver math; it only
builds a static predefined wealth grid that can be passed through
DiscretizationConfig.wealth_grid_path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from configs._canonical import BASE_CONFIG  # noqa: E402
from lifecycle.model import DiscretizationConfig  # noqa: E402
from lifecycle.policy_io import load_policy_bundle  # noqa: E402
from lifecycle.precompute import build_model, build_precompute  # noqa: E402
from lifecycle.predictability_ablation import get_predictability_system_spec  # noqa: E402
from lifecycle.simulation import simulate_lifecycle  # noqa: E402
from lifecycle.var import build_real_full_var_config_hardcoded  # noqa: E402
from lifecycle.wealth_grid import (  # noqa: E402
    disc_config_with_bundle_wealth_grid,
    legacy_log1p_wealth_grid,
    load_wealth_grid_from_bundle,
    validate_wealth_grid,
    wealth_grid_hash,
    wealth_grid_spacing_stats,
)


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


def _parse_density_age(value: str) -> str | int:
    """Accept the literal "all" or an integer age. Range is checked later."""
    if value is None:
        return "all"
    text = str(value).strip()
    if text.lower() == "all":
        return "all"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--density-age must be 'all' or an integer age, got {value!r}."
        ) from exc


def _resolve_density_age_idx(
    density_age: str | int,
    *,
    start_age: int,
    terminal_age: int,
) -> int | None:
    """Convert a parsed --density-age into the simulation t-index, or None for 'all'."""
    if isinstance(density_age, str):
        if density_age == "all":
            return None
        raise ValueError(
            f"density_age string must be 'all', got {density_age!r}."
        )
    age = int(density_age)
    if age < int(start_age) or age > int(terminal_age):
        raise ValueError(
            f"--density-age={age} is outside the model age range "
            f"[{int(start_age)}, {int(terminal_age)}]."
        )
    return age - int(start_age)


def _parse_smoothing_window(smoothing: str) -> int:
    match = re.fullmatch(r"moving-average-(\d+)", str(smoothing).strip())
    if match is None:
        raise ValueError(
            f"Unsupported smoothing {smoothing!r}; expected moving-average-<odd window>."
        )
    window = int(match.group(1))
    if window < 1 or window % 2 == 0:
        raise ValueError("Moving-average smoothing window must be a positive odd integer.")
    return window


def _moving_average_edge(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / float(window), dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _second_derivative_last_axis(values: np.ndarray, wealth_grid: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(wealth_grid, dtype=np.float64)
    h_left = (x[1:-1] - x[:-2]).reshape((1,) * (y.ndim - 1) + (-1,))
    h_right = (x[2:] - x[1:-1]).reshape((1,) * (y.ndim - 1) + (-1,))
    span = (x[2:] - x[:-2]).reshape((1,) * (y.ndim - 1) + (-1,))
    left_slope = (y[..., 1:-1] - y[..., :-2]) / h_left
    right_slope = (y[..., 2:] - y[..., 1:-1]) / h_right
    return 2.0 * (right_slope - left_slope) / span


def _normalize_curvature(curvature: np.ndarray) -> np.ndarray:
    out = np.abs(np.asarray(curvature, dtype=np.float64))
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out)
    scale = float(np.nanpercentile(out[finite], 95))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = float(np.nanmax(out[finite]))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    out = out / scale
    out[~np.isfinite(out)] = np.nan
    return out


def _combine_policy_curvatures(
    C: np.ndarray,
    S: np.ndarray,
    B: np.ndarray,
    wealth_grid: np.ndarray,
    multi_policy: str,
) -> np.ndarray:
    curvatures = []
    for arr in (C, S, B):
        curvatures.append(
            _normalize_curvature(_second_derivative_last_axis(arr, wealth_grid))
        )
    stack = np.stack(curvatures, axis=0)
    if multi_policy == "worst-of":
        return np.nanmax(stack, axis=0)
    if multi_policy == "mean":
        return np.nanmean(stack, axis=0)
    raise ValueError(f"Unknown multi_policy={multi_policy!r}")


def _build_var_config(run_config: dict[str, Any]) -> dict[str, Any]:
    pab = run_config.get("predictability_ablation", {}) or {}
    system = pab.get("system_code")
    if system is not None:
        spec = get_predictability_system_spec(system)
        # Real-yields pivot: spec carries builder_name (late-bound) instead
        # of a callable. Resolve via importlib so this script works on any
        # spec produced by the new ablation module.
        import importlib
        var_module = importlib.import_module("lifecycle.var")
        builder = getattr(var_module, spec.builder_name)
        var_config, _var_res, _var_data = builder(
            csv_path=str(REPO / "data" / "var_dataset.csv")
        )
        return var_config
    return build_real_full_var_config_hardcoded()


def _build_model_pc(bundle: Path, metadata: dict[str, Any]):
    run_config = metadata.get("run_config")
    if run_config is None:
        raise ValueError("Bundle metadata has no run_config; cannot simulate weights.")
    disc = _rehydrate_disc_config(run_config["discretization_config"])
    disc = disc_config_with_bundle_wealth_grid(disc, bundle, metadata)
    base_config = run_config.get("base_config", dict(BASE_CONFIG))
    model = build_model(base_config, _build_var_config(run_config), verbose=False)
    pc = build_precompute(model, disc, verbose=False)
    return model, pc


def _density_weights_from_simulation(
    C: np.ndarray,
    S: np.ndarray,
    B: np.ndarray,
    model,
    pc,
    *,
    n_simulations: int,
    seed: int,
    initial_wealth: float,
    age_idx: int | None = None,
) -> np.ndarray:
    if C.ndim != 4:
        raise ValueError("density_weighted aggregation requires lifecycle 4D policy arrays.")
    sim = simulate_lifecycle(
        C, S, B, pc, model,
        n_simulations=int(n_simulations),
        initial_wealth=float(initial_wealth),
        initial_z="stationary",
        initial_state="stationary",
        seed=int(seed),
        verbose=False,
    )
    alive = np.asarray(sim["alive"], dtype=bool)
    x = np.asarray(sim["x"], dtype=np.float64)
    z_idx = np.asarray(sim["z_idx"], dtype=np.int64)
    state_idx = np.asarray(sim["state_idx"], dtype=np.int64)

    weights = np.zeros(C.shape[:-1] + (pc.n_w - 2,), dtype=np.float64)
    interior_w = np.asarray(pc.wealth_grid[1:-1], dtype=np.float64)
    if age_idx is None:
        ages_to_accumulate = range(C.shape[0])
    else:
        ages_to_accumulate = (int(age_idx),)
    for t in ages_to_accumulate:
        mask = alive[:, t] & np.isfinite(x[:, t])
        if not mask.any():
            continue
        w = x[mask, t]
        z = np.clip(z_idx[mask, t], 0, pc.n_z - 1)
        s = np.clip(state_idx[mask, t], 0, pc.N_state - 1)
        hi = np.searchsorted(interior_w, w, side="left")
        hi = np.clip(hi, 0, interior_w.size - 1)
        lo = np.clip(hi - 1, 0, interior_w.size - 1)
        choose_hi = np.abs(interior_w[hi] - w) <= np.abs(interior_w[lo] - w)
        wi = np.where(choose_hi, hi, lo)
        np.add.at(weights, (np.full_like(wi, t), z, s, wi), 1.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Simulation produced no alive weighted observations.")
    return weights / total


def _aggregate_monitor(
    combined_curvature: np.ndarray,
    aggregation: str,
    weights: np.ndarray | None,
    density_floor_frac: float = 0.03,
) -> np.ndarray:
    curv = np.asarray(combined_curvature, dtype=np.float64)
    curv_filled = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
    axes = tuple(range(curv.ndim - 1))
    if aggregation == "density_weighted":
        if weights is None:
            raise ValueError("density_weighted aggregation requires simulation weights.")
        monitor = np.sum(curv_filled * weights, axis=axes)
    elif aggregation == "max":
        monitor = np.nanmax(curv, axis=axes)
    elif aggregation == "mean":
        monitor = np.nanmean(curv, axis=axes)
    else:
        raise ValueError(f"Unknown aggregation={aggregation!r}")
    monitor = np.nan_to_num(monitor, nan=0.0, posinf=0.0, neginf=0.0)
    max_monitor = float(np.max(monitor)) if monitor.size else 0.0
    floor = max(1e-14, float(density_floor_frac) * max_monitor)
    return np.maximum(monitor, floor)


def _grid_from_monitor(
    reference_w: np.ndarray,
    monitor_interior: np.ndarray,
    n_wealth: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    monitor_full = np.empty_like(reference_w, dtype=np.float64)
    monitor_full[1:-1] = np.asarray(monitor_interior, dtype=np.float64)
    monitor_full[0] = monitor_full[1]
    monitor_full[-1] = monitor_full[-2]
    smoothed = _moving_average_edge(monitor_full, window)
    M = np.sqrt(np.maximum(smoothed, 1e-300))
    increments = 0.5 * (M[1:] + M[:-1]) * np.diff(reference_w)
    F = np.concatenate(([0.0], np.cumsum(increments)))
    if not np.isfinite(F[-1]) or F[-1] <= 0.0:
        raise ValueError("Monitor integral is not positive and finite.")
    targets = np.linspace(0.0, F[-1], int(n_wealth))
    grid = np.interp(targets, F, reference_w)
    grid[0] = reference_w[0]
    grid[-1] = reference_w[-1]
    return np.ascontiguousarray(grid, dtype=np.float64), smoothed


def _sensitivity_stats(primary: np.ndarray, alternatives: dict[int, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    primary_diff = np.diff(primary)
    for window, grid in alternatives.items():
        node_shift = np.abs(grid - primary)
        spacing_shift = np.abs(np.diff(grid) - primary_diff) / np.maximum(primary_diff, 1e-300)
        out[str(window)] = {
            "max_abs_node_shift": float(np.max(node_shift)),
            "max_relative_spacing_shift": float(np.max(spacing_shift)),
        }
    return out


def _write_plot(out_png: Path, wealth_grid: np.ndarray, log_grid: np.ndarray) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    i_new = np.arange(wealth_grid.size)
    i_log = np.arange(log_grid.size)
    axes[0].plot(i_log, log_grid, label="log1p", lw=1.8)
    axes[0].plot(i_new, wealth_grid, label="Bakhvalov", lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("node index")
    axes[0].set_ylabel("wealth W")
    axes[0].set_title("Wealth nodes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(log_grid[:-1], np.diff(log_grid), label="log1p", lw=1.8)
    axes[1].plot(wealth_grid[:-1], np.diff(wealth_grid), label="Bakhvalov", lw=1.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("left node W")
    axes[1].set_ylabel("adjacent spacing")
    axes[1].set_title("Spacing profile")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Bakhvalov Wealth Grid",
        "",
        f"- Source bundle: `{summary['source_bundle']}`",
        f"- n_wealth: `{summary['n_wealth']}`",
        f"- Aggregation: `{summary['aggregation']}`",
        f"- Multi-policy: `{summary['multi_policy']}`",
        f"- Smoothing: `{summary['smoothing']}`",
        f"- Density age: `{summary.get('density_age')}`",
        f"- Density initial wealth: `{summary.get('density_initial_wealth')}`",
        f"- Hash: `{summary['wealth_grid_hash']}`",
        "",
        "## Float32 Spacing",
        "",
        f"- min_diff64: `{summary['spacing']['min_diff64']:.6e}`",
        f"- min_diff32: `{summary['spacing']['min_diff32']:.6e}`",
        f"- min_rel_diff32: `{summary['spacing']['min_rel_diff32']:.6e}`",
        f"- non-positive fp32 gaps: `{summary['spacing']['n_nonpositive_diff32']}`",
        "",
        "## Smoothing Sensitivity",
        "",
    ]
    for window, stats in summary["smoothing_sensitivity"].items():
        lines.append(
            f"- window {window}: max node shift "
            f"`{stats['max_abs_node_shift']:.6e}`, max relative spacing shift "
            f"`{stats['max_relative_spacing_shift']:.6e}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPO / "saved_runs" / "ablations" / "system_i_grid7_nz70_calib1",
    )
    parser.add_argument("--n-wealth", type=int, required=True)
    parser.add_argument(
        "--aggregation",
        choices=("density_weighted", "max", "mean"),
        default="density_weighted",
    )
    parser.add_argument(
        "--multi-policy",
        choices=("worst-of", "mean"),
        default="worst-of",
    )
    parser.add_argument("--smoothing", default="moving-average-9")
    parser.add_argument("--sensitivity-windows", type=int, nargs="*", default=[5, 15])
    parser.add_argument("--n-simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--density-initial-wealth",
        type=float,
        default=1.0,
        help="Initial wealth (in AWI units) seeded into simulate_lifecycle when "
             "building density weights.",
    )
    parser.add_argument(
        "--density-age",
        type=_parse_density_age,
        default=67,
        help="Either 'all' for lifecycle-mass weighting or an integer age in "
             "[start_age, terminal_age] for a single-age cross-section snapshot.",
    )
    parser.add_argument(
        "--density-floor-frac",
        type=float,
        default=0.03,
        help="Floor on the aggregated curvature monitor as a fraction of its max. "
             "Avoids leaving tiny-curvature regions with zero density (which would "
             "starve them of grid nodes). Default 0.03.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .npz path for the generated wealth grid.",
    )
    args = parser.parse_args(argv)

    bundle = args.bundle
    C, S, B, _diag, metadata = load_policy_bundle(bundle)
    C = np.asarray(C, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    reference_w = load_wealth_grid_from_bundle(bundle, metadata)

    primary_window = _parse_smoothing_window(args.smoothing)
    sensitivity_windows = sorted(set([primary_window, *args.sensitivity_windows]))
    for window in sensitivity_windows:
        if window < 1 or window % 2 == 0:
            raise ValueError("Sensitivity windows must be positive odd integers.")

    model = pc = None
    weights = None
    density_age_idx: int | None = None
    if args.aggregation == "density_weighted":
        model, pc = _build_model_pc(bundle, metadata)
        density_age_idx = _resolve_density_age_idx(
            args.density_age,
            start_age=model.start_age,
            terminal_age=model.terminal_age,
        )
        weights = _density_weights_from_simulation(
            C, S, B, model, pc,
            n_simulations=args.n_simulations,
            seed=args.seed,
            initial_wealth=args.density_initial_wealth,
            age_idx=density_age_idx,
        )

    combined = _combine_policy_curvatures(C, S, B, reference_w, args.multi_policy)
    monitor = _aggregate_monitor(combined, args.aggregation, weights, args.density_floor_frac)

    grids_by_window: dict[int, np.ndarray] = {}
    smooth_by_window: dict[int, np.ndarray] = {}
    for window in sensitivity_windows:
        grid, smoothed_monitor = _grid_from_monitor(
            reference_w, monitor, int(args.n_wealth), window
        )
        validate_wealth_grid(
            grid,
            n_wealth=int(args.n_wealth),
            wealth_min=float(reference_w[0]),
            wealth_max=float(reference_w[-1]),
            source=f"Bakhvalov wealth grid window={window}",
        )
        grids_by_window[window] = grid
        smooth_by_window[window] = smoothed_monitor

    wealth_grid = grids_by_window[primary_window]
    grid_hash = wealth_grid_hash(wealth_grid)
    spacing = wealth_grid_spacing_stats(wealth_grid)
    sensitivity = _sensitivity_stats(
        wealth_grid,
        {w: g for w, g in grids_by_window.items() if w != primary_window},
    )

    density_age_str = (
        "all" if args.density_age == "all" else str(int(args.density_age))
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        wealth_grid=wealth_grid,
        wealth_grid_hash=grid_hash,
        source_bundle=str(bundle),
        aggregation=str(args.aggregation),
        multi_policy=str(args.multi_policy),
        smoothing=str(args.smoothing),
        n_wealth=int(args.n_wealth),
        wealth_min=float(reference_w[0]),
        wealth_max=float(reference_w[-1]),
        monitor_wealth=reference_w,
        monitor=monitor,
        smoothed_monitor=smooth_by_window[primary_window],
        density_age=density_age_str,
        density_initial_wealth=float(args.density_initial_wealth),
    )

    log_grid = legacy_log1p_wealth_grid(args.n_wealth, reference_w[0], reference_w[-1])
    plot_path = args.out.with_suffix(".png")
    json_path = args.out.with_suffix(".json")
    md_path = args.out.with_suffix(".md")
    _write_plot(plot_path, wealth_grid, log_grid)

    summary = {
        "source_bundle": str(bundle),
        "output_grid": str(args.out),
        "plot": str(plot_path),
        "n_wealth": int(args.n_wealth),
        "wealth_min": float(reference_w[0]),
        "wealth_max": float(reference_w[-1]),
        "aggregation": str(args.aggregation),
        "multi_policy": str(args.multi_policy),
        "smoothing": str(args.smoothing),
        "wealth_grid_hash": grid_hash,
        "spacing": spacing,
        "smoothing_sensitivity": sensitivity,
        "n_simulations": int(args.n_simulations) if args.aggregation == "density_weighted" else None,
        "seed": int(args.seed) if args.aggregation == "density_weighted" else None,
        "density_age": density_age_str if args.aggregation == "density_weighted" else None,
        "density_initial_wealth": (
            float(args.density_initial_wealth)
            if args.aggregation == "density_weighted"
            else None
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)

    print(f"Wrote grid: {args.out}", flush=True)
    print(f"Hash: {grid_hash}", flush=True)
    print(f"Plot: {plot_path}", flush=True)
    print(f"Summary: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

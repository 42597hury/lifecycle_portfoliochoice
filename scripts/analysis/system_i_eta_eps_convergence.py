"""
System I × (n_eta, n_eps) sensitivity convergence analysis.

Reads the three System I ablation bundles solved at fixed n_z=30 with varying
working-age income-shock quadrature density:

    saved_runs/ablations/system_i_grid7_nz30_eta3eps4_calib1/
    saved_runs/ablations/system_i_grid7_nz30_eta4eps5_calib1/
    saved_runs/ablations/system_i_grid7_nz30_eta6eps6_calib1/

Treats (n_eta=6, n_eps=6) as the reference. All three bundles share shape
(78, 30, 7, 180), so no interpolation is required — element-wise comparison
is valid.

Reports sup-norm / RMS / relative divergence and per-axis profiles for the
consumption / risky-share / bond-share policies. Splits the divergence into
working ages (22..66, indices 0..44) and retirement (67..99, indices 45..77)
to confirm the sanity check that retirement-age policies are independent of
(n_eta, n_eps).

Usage:
    python scripts/analysis/system_i_eta_eps_convergence.py
        [--bundles-root saved_runs/ablations]
        [--output-dir docs/scans]
        [--fig-dir docs/scans/figures]
        [--no-plots]

Writes:
    {output-dir}/system_i_eta_eps_convergence_metrics.json
    {fig-dir}/eta_eps_convergence_curves.png
    {fig-dir}/eta_eps_per_age_divergence.png
    {fig-dir}/eta_eps_per_z_divergence.png
    {fig-dir}/eta_eps_per_wealth_divergence.png
    {fig-dir}/eta_eps_probe_alpha_vs_age.png

Read-only with respect to lifecycle/. Does not re-solve any bundle.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lifecycle.policy_io import load_policy_bundle  # noqa: E402

CONFIGS: tuple[tuple[int, int], ...] = ((3, 4), (4, 5), (6, 6))
REFERENCE_CONFIG: tuple[int, int] = (6, 6)
BUNDLE_NAME_TEMPLATE = "system_i_grid7_nz30_eta{n_eta}eps{n_eps}_calib1"

# Lifecycle conventions (verified from base_config in metadata.json):
START_AGE = 22
RETIRE_AGE = 67
TERMINAL_AGE = 99
RETIRE_AGE_IDX = RETIRE_AGE - START_AGE  # 45 — first retirement-age index
N_AGES = TERMINAL_AGE - START_AGE + 1    # 78


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def _bundle_path(bundles_root: Path, n_eta: int, n_eps: int) -> Path:
    return bundles_root / BUNDLE_NAME_TEMPLATE.format(n_eta=n_eta, n_eps=n_eps)


def _read_disc_config(bundle: Path) -> dict[str, Any]:
    diag_path = bundle / "diagnostics.pkl"
    with diag_path.open("rb") as f:
        diag = pickle.load(f)
    return dict(diag.get("disc_config", {})), diag


def load_one(bundles_root: Path, cfg: tuple[int, int]) -> dict[str, Any]:
    n_eta, n_eps = cfg
    bundle = _bundle_path(bundles_root, n_eta, n_eps)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle}")
    C, S, B, diag, metadata = load_policy_bundle(bundle)
    disc, full_diag = _read_disc_config(bundle)
    return {
        "config": cfg,
        "bundle": bundle,
        "C": np.asarray(C),
        "S": np.asarray(S),
        "B": np.asarray(B),
        "metadata": metadata,
        "disc_config": disc,
        "diagnostics": full_diag,
    }


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_bundle(b: dict[str, Any]) -> list[str]:
    """Return list of warning strings; empty == clean."""
    cfg = b["config"]
    warnings: list[str] = []
    disc = b["disc_config"]
    if int(disc.get("n_eta_nodes", -1)) != cfg[0]:
        warnings.append(
            f"{cfg}: disc_config.n_eta_nodes={disc.get('n_eta_nodes')} != requested {cfg[0]}"
        )
    if int(disc.get("n_eps_nodes", -1)) != cfg[1]:
        warnings.append(
            f"{cfg}: disc_config.n_eps_nodes={disc.get('n_eps_nodes')} != requested {cfg[1]}"
        )
    if int(disc.get("n_z", -1)) != 30:
        warnings.append(f"{cfg}: disc_config.n_z={disc.get('n_z')} != 30")
    nf = b["diagnostics"].get("total_newton_failures", None)
    if nf not in (0, None):
        warnings.append(f"{cfg}: total_newton_failures={nf} (expected 0)")
    status = b["diagnostics"].get("solve_status", None)
    if status != "complete":
        warnings.append(f"{cfg}: solve_status={status!r} (expected 'complete')")
    state_names = (
        b["metadata"].get("run_config", {})
        .get("predictability_ablation", {})
        .get("state_names")
    )
    if state_names != ["rtb"] and state_names != ("rtb",):
        warnings.append(f"{cfg}: state_names={state_names!r} (expected ('rtb',))")
    for name in ("C", "S", "B"):
        arr = b[name]
        if arr.shape != (N_AGES, 30, 7, 180):
            warnings.append(f"{cfg}: {name} shape={arr.shape} != (78,30,7,180)")
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            warnings.append(f"{cfg}: {name} contains {n_bad} non-finite cells")
    return warnings


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _abs_delta_metrics(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Scalar summaries of element-wise |coarse - ref|."""
    sup = float(np.max(delta))
    p99 = float(np.percentile(delta, 99))
    p95 = float(np.percentile(delta, 95))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    mean_abs = float(np.mean(delta))
    ref_abs = np.abs(ref)
    median_nz = float(np.median(ref_abs[ref_abs > 0]) if np.any(ref_abs > 0) else 0.0)
    threshold = max(1e-8, 1e-3 * median_nz)
    mask = ref_abs > threshold
    if mask.any():
        rel = delta[mask] / ref_abs[mask]
        sup_rel = float(np.max(rel))
        p99_rel = float(np.percentile(rel, 99))
    else:
        sup_rel = float("nan")
        p99_rel = float("nan")
    return {
        "sup": sup,
        "p99": p99,
        "p95": p95,
        "rms": rms,
        "mean_abs": mean_abs,
        "sup_rel": sup_rel,
        "p99_rel": p99_rel,
        "rel_threshold": float(threshold),
    }


def _per_axis_max(delta: np.ndarray) -> dict[str, list[float]]:
    """Max |coarse - ref| reduced along three of four axes."""
    return {
        "per_age": [float(x) for x in np.max(delta, axis=(1, 2, 3))],
        "per_z": [float(x) for x in np.max(delta, axis=(0, 2, 3))],
        "per_state": [float(x) for x in np.max(delta, axis=(0, 1, 3))],
        "per_wealth": [float(x) for x in np.max(delta, axis=(0, 1, 2))],
    }


def _argmax_indices(delta: np.ndarray) -> dict[str, int]:
    """Return (age, z, state, wealth) indices of the worst cell."""
    flat = int(np.argmax(delta))
    age, z, st, w = np.unravel_index(flat, delta.shape)
    return {"age_idx": int(age), "z_idx": int(z), "state_idx": int(st),
            "wealth_idx": int(w), "age_value": int(START_AGE + age)}


def compute_pair_metrics(
    coarse: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    cfg = coarse["config"]
    metrics: dict[str, Any] = {
        "config_coarse": list(cfg),
        "config_reference": list(reference["config"]),
        "shape": list(reference["C"].shape),
    }
    for name in ("C", "S", "B"):
        coarse_arr = coarse[name]
        ref_arr = reference[name]
        if coarse_arr.shape != ref_arr.shape:
            raise ValueError(
                f"Shape mismatch for {name}: coarse {coarse_arr.shape} vs ref {ref_arr.shape}"
            )
        delta = np.abs(coarse_arr - ref_arr)
        scalar = _abs_delta_metrics(delta, ref_arr)
        per_axis = _per_axis_max(delta)
        argmax = _argmax_indices(delta)
        metrics[name] = {**scalar, **per_axis, "argmax": argmax}
    return metrics


def split_working_retirement(
    coarse: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Compute sup divergence separately on working / retirement age slabs.

    Retirement policies do not depend on (n_eta, n_eps) — the FOC at retirement
    integrates only over return shocks, not working-age income shocks. So
    retirement-age divergence should be exactly 0 modulo float-rounding from
    a different upstream computation feeding the same age slab. If non-zero,
    that is a flag.
    """
    out: dict[str, Any] = {}
    for name in ("C", "S", "B"):
        delta = np.abs(coarse[name] - reference[name])
        out[name] = {
            "working_sup": float(np.max(delta[:RETIRE_AGE_IDX])),
            "working_rms": float(np.sqrt(np.mean(delta[:RETIRE_AGE_IDX] ** 2))),
            "retirement_sup": float(np.max(delta[RETIRE_AGE_IDX:])),
            "retirement_rms": float(np.sqrt(np.mean(delta[RETIRE_AGE_IDX:] ** 2))),
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(bundles: dict[tuple[int, int], dict],
                metrics_payload: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    pairs = metrics_payload["pairs"]
    coarse_keys = [k for k in pairs.keys()]  # e.g. "(3, 4)", "(4, 5)"
    coarse_cfgs = [tuple(pairs[k]["config_coarse"]) for k in coarse_keys]
    products = [eta * eps for (eta, eps) in coarse_cfgs]
    # Reference itself: divergence 0 at product 36
    ref_eta, ref_eps = REFERENCE_CONFIG
    products_full = products + [ref_eta * ref_eps]

    # --- 1. Convergence curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("sup", "Sup-norm |coarse − ref|"),
        ("rms", "RMS |coarse − ref|"),
        ("p99", "99th-pctile |coarse − ref|"),
        ("sup_rel", "Relative sup |coarse − ref| / |ref|"),
    ]
    colors = {"C": "C0", "S": "C1", "B": "C2"}
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name in ("C", "S", "B"):
            vals = [pairs[k][name][key] for k in coarse_keys]
            # Sort by product (ascending)
            order = np.argsort(products)
            x_sorted = [products[i] for i in order]
            v_sorted = [vals[i] for i in order]
            ax.plot(x_sorted, v_sorted, marker="o", label=name, color=colors[name])
        ax.set_xscale("log")
        if key != "sup_rel":
            ax.set_yscale("log")
        ax.set_xlabel("n_eta × n_eps (log scale)")
        ax.set_ylabel(title)
        ax.set_title(title + f" vs (n_eta={ref_eta}, n_eps={ref_eps})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        # Annotate with config labels
        for cfg, prod, k in zip(coarse_cfgs, products, coarse_keys):
            ax.annotate(
                f"({cfg[0]},{cfg[1]})",
                xy=(prod, pairs[k]["C"][key]),
                xytext=(3, 3), textcoords="offset points", fontsize=8,
            )
    fig.suptitle(
        "System I × (n_eta, n_eps) convergence — coarse vs (6, 6) reference",
        fontsize=13,
    )
    fig.tight_layout()
    out = fig_dir / "eta_eps_convergence_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")

    # --- 2. Per-age, per-z, per-wealth divergence (one panel per array)
    for axis_key, axis_label, fname in [
        ("per_age", "age (index from start_age=22)", "eta_eps_per_age_divergence.png"),
        ("per_z", "z index (n_z=30)", "eta_eps_per_z_divergence.png"),
        ("per_wealth", "wealth index (n_w=180)", "eta_eps_per_wealth_divergence.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        cmap = plt.get_cmap("viridis")
        for ax, name in zip(axes, ("C", "S", "B")):
            for i, k in enumerate(coarse_keys):
                vals = pairs[k][name][axis_key]
                x = np.arange(len(vals))
                cfg = pairs[k]["config_coarse"]
                ax.plot(x, vals,
                        color=cmap(i / max(1, len(coarse_keys) - 1)),
                        label=f"({cfg[0]},{cfg[1]})", lw=1.6)
            if axis_key == "per_age":
                ax.axvline(RETIRE_AGE_IDX, color="k", ls=":", lw=0.8,
                           label=f"retire (age {RETIRE_AGE})")
            ax.set_yscale("log")
            ax.set_xlabel(axis_label)
            ax.set_ylabel(f"sup |{name}_coarse − {name}_ref|")
            ax.set_title(f"{name}: divergence per {axis_label}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Per-{axis_label} divergence vs (n_eta=6, n_eps=6)"
        )
        fig.tight_layout()
        out = fig_dir / fname
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")

    # --- 3. Probe-cell α_s, α_b vs age
    # Use per-axis midpoint convention: z=mid, state=mid, wealth=mid.
    ref_b = bundles[REFERENCE_CONFIG]
    n_z = ref_b["S"].shape[1]
    n_state = ref_b["S"].shape[2]
    n_wealth = ref_b["S"].shape[3]
    z_mid = n_z // 2
    state_mid = n_state // 2
    wealth_mid = n_wealth // 2
    ages = np.arange(START_AGE, TERMINAL_AGE + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    cfgs_sorted = sorted(bundles.keys(), key=lambda c: c[0] * c[1])
    for ax, name, title in zip(axes, ("S", "B"), ("α_s (risky share)", "α_b (bond share)")):
        for i, cfg in enumerate(cfgs_sorted):
            arr = bundles[cfg][name]
            curve = arr[:, z_mid, state_mid, wealth_mid]
            ls = "-" if cfg == REFERENCE_CONFIG else "--"
            ax.plot(ages, curve,
                    color=cmap(i / max(1, len(cfgs_sorted) - 1)),
                    label=f"(n_eta={cfg[0]}, n_eps={cfg[1]})",
                    ls=ls, lw=1.6)
        ax.axvline(RETIRE_AGE, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.set_xlabel("age")
        ax.set_ylabel(title)
        ax.set_title(
            f"{title} at probe cell "
            f"(z idx={z_mid}/{n_z}, state idx={state_mid}/{n_state}, "
            f"wealth idx={wealth_mid}/{n_wealth})"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Probe-cell portfolio policies across (n_eta, n_eps)")
    fig.tight_layout()
    out = fig_dir / "eta_eps_probe_alpha_vs_age.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles-root", type=Path,
        default=REPO / "saved_runs" / "ablations",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "docs" / "scans",
    )
    parser.add_argument(
        "--metrics-name", type=str,
        default="system_i_eta_eps_convergence_metrics.json",
    )
    parser.add_argument(
        "--fig-dir", type=Path,
        default=REPO / "docs" / "scans" / "figures",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    bundles_root: Path = args.bundles_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bundles from: {bundles_root}")
    bundles: dict[tuple[int, int], dict[str, Any]] = {}
    for cfg in CONFIGS:
        bundles[cfg] = load_one(bundles_root, cfg)
        sh = bundles[cfg]["C"].shape
        wall = bundles[cfg]["diagnostics"].get("wall_time_sec", float("nan"))
        print(f"  (n_eta={cfg[0]}, n_eps={cfg[1]}): shape={sh}, wall={wall:.1f}s")

    print("\nValidation gates:")
    any_warnings = False
    validation: dict[str, list[str]] = {}
    for cfg, b in bundles.items():
        ws = validate_bundle(b)
        validation[str(cfg)] = ws
        if ws:
            any_warnings = True
            for w in ws:
                print(f"  WARN {w}")
    if not any_warnings:
        print("  all clean.")

    ref = bundles[REFERENCE_CONFIG]

    print(f"\nDivergence vs (n_eta=6, n_eps=6) reference:")
    pairs: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, Any]] = {}
    for cfg in CONFIGS:
        if cfg == REFERENCE_CONFIG:
            continue
        m = compute_pair_metrics(bundles[cfg], ref)
        s = split_working_retirement(bundles[cfg], ref)
        pairs[str(cfg)] = m
        splits[str(cfg)] = s
        print(
            f"  ({cfg[0]},{cfg[1]}) | "
            f"sup_C={m['C']['sup']:.4e} sup_S={m['S']['sup']:.4e} sup_B={m['B']['sup']:.4e} | "
            f"rms_C={m['C']['rms']:.3e} | "
            f"rel_sup_C={m['C']['sup_rel']:.3e}"
        )
        for name in ("C", "S", "B"):
            print(
                f"      {name}: working_sup={s[name]['working_sup']:.4e}, "
                f"retirement_sup={s[name]['retirement_sup']:.4e}"
            )

    # Distribution snapshot
    distribution = {}
    for cfg, b in bundles.items():
        distribution[str(cfg)] = {
            "S_quantiles": [float(q) for q in np.quantile(b["S"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "B_quantiles": [float(q) for q in np.quantile(b["B"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "C_quantiles": [float(q) for q in np.quantile(b["C"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "S_min": float(b["S"].min()), "S_max": float(b["S"].max()),
            "B_min": float(b["B"].min()), "B_max": float(b["B"].max()),
            "C_min": float(b["C"].min()), "C_max": float(b["C"].max()),
        }

    wall_times = {
        str(cfg): float(b["diagnostics"].get("wall_time_sec", float("nan")))
        for cfg, b in bundles.items()
    }

    payload = {
        "bundles_root": str(bundles_root),
        "configs": [list(c) for c in CONFIGS],
        "reference_config": list(REFERENCE_CONFIG),
        "reference_shape": list(ref["C"].shape),
        "retire_age_idx": RETIRE_AGE_IDX,
        "start_age": START_AGE,
        "terminal_age": TERMINAL_AGE,
        "validation": validation,
        "pairs": pairs,
        "split_working_retirement": splits,
        "distributions": distribution,
        "wall_time_sec": wall_times,
    }
    metrics_path = output_dir / args.metrics_name
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote metrics to {metrics_path}")

    if not args.no_plots:
        print("\nGenerating figures ...")
        _make_plots(bundles, payload, args.fig_dir)
        print(f"All figures written under {args.fig_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

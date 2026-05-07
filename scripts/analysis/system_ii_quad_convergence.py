"""
System II quadrature-density sensitivity convergence analysis.

Reads the System II ablation bundles solved at fixed n_z=15, (n_eta, n_eps)=(3, 4),
state_grid_sizes=(7, 7), n_ret_nodes_1d=(3, 3) with varying state-quadrature
density:

    saved_runs/ablations/system_ii_grid7x7_nz15_sq3x3_rq3x3_calib1/   # baseline
    saved_runs/ablations/system_ii_grid7x7_nz15_sq4x4_rq3x3_calib1/   # uniform state refinement
    saved_runs/ablations/system_ii_grid7x7_nz15_sq3x3_rq4x4_calib1/   # ret refinement
    saved_runs/ablations/system_ii_grid7x7_nz15_sq3x5_rq3x3_calib1/   # y_1 K-bump (no Lobatto)

Treats sq3x3_rq3x3 (the cheapest) as the baseline; the question is "is this
enough?". All four bundles share shape (78, 15, 49, 180) so element-wise
comparison is valid (no interpolation).

Reports sup-norm / RMS / relative divergence and per-axis profiles for the
consumption / risky-share / bond-share policies. Splits divergence into working
ages (22..66, indices 0..44) and retirement (67..99, indices 45..77). The 49-cell
state axis is reshaped as (7, 7) = (n_rtb, n_y1) for the per-state heatmap.

Bundles missing on disk are skipped gracefully — pairwise metrics are only
computed for available pairs, and the output JSON / verdicts mark which
comparisons were not feasible.

Usage:
    python scripts/analysis/system_ii_quad_convergence.py
        [--bundles-root saved_runs/ablations]
        [--output-dir docs/scans]
        [--fig-dir docs/scans/figures]
        [--no-plots]

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

# (label, bundle dirname, n_state_quad, n_ret_nodes_1d, role)
CONFIGS: tuple[tuple[str, str, tuple[int, int], tuple[int, int], str], ...] = (
    ("sq3x3_rq3x3", "system_ii_grid7x7_nz15_sq3x3_rq3x3_calib1", (3, 3), (3, 3), "baseline"),
    ("sq4x4_rq3x3", "system_ii_grid7x7_nz15_sq4x4_rq3x3_calib1", (4, 4), (3, 3), "state_uniform"),
    ("sq3x3_rq4x4", "system_ii_grid7x7_nz15_sq3x3_rq4x4_calib1", (3, 3), (4, 4), "ret_refine"),
    ("sq3x5_rq3x3", "system_ii_grid7x7_nz15_sq3x5_rq3x3_calib1", (3, 5), (3, 3), "y1_kbump"),
)
BASELINE_LABEL = "sq3x3_rq3x3"

# Compares to run if bundles available: (coarse_label, ref_label, effect_name)
PAIR_PLAN: tuple[tuple[str, str, str], ...] = (
    ("sq4x4_rq3x3", "sq3x3_rq3x3", "state_quad_uniform_refinement"),
    ("sq3x3_rq4x4", "sq3x3_rq3x3", "ret_quad_refinement"),
    ("sq3x5_rq3x3", "sq3x3_rq3x3", "y1_axis_kbump_vs_baseline"),
    ("sq3x5_rq3x3", "sq4x4_rq3x3", "kbump_vs_uniform_state_refinement"),
)

# Lifecycle conventions:
START_AGE = 22
RETIRE_AGE = 67
TERMINAL_AGE = 99
RETIRE_AGE_IDX = RETIRE_AGE - START_AGE  # 45 — first retirement-age index
N_AGES = TERMINAL_AGE - START_AGE + 1    # 78

# State-grid layout: 49 = 7 (rtb) × 7 (y_1), C-order (rtb outer, y_1 inner).
N_RTB = 7
N_Y1 = 7
EXPECTED_SHAPE = (N_AGES, 15, N_RTB * N_Y1, 180)


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def _load_diagnostics(bundle: Path) -> dict[str, Any]:
    diag_path = bundle / "diagnostics.pkl"
    if not diag_path.exists():
        return {}
    with diag_path.open("rb") as f:
        return pickle.load(f)


def load_one(bundles_root: Path, dirname: str) -> dict[str, Any] | None:
    bundle = bundles_root / dirname
    if not bundle.exists():
        return None
    C, S, B, _diag, metadata = load_policy_bundle(bundle)
    diag = _load_diagnostics(bundle)
    return {
        "bundle": bundle,
        "C": np.asarray(C),
        "S": np.asarray(S),
        "B": np.asarray(B),
        "metadata": metadata,
        "diagnostics": diag,
        "disc_config": dict(diag.get("disc_config", {})),
    }


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def validate_bundle(b: dict[str, Any], label: str,
                    expected_n_state_quad: tuple[int, int],
                    expected_n_ret: tuple[int, int]) -> list[str]:
    """Return list of warning strings; empty == clean."""
    warnings: list[str] = []
    disc = b["disc_config"]
    nsq = disc.get("n_state_quad_nodes")
    if nsq is not None and tuple(nsq) != expected_n_state_quad:
        warnings.append(
            f"{label}: disc_config.n_state_quad_nodes={tuple(nsq)} != expected {expected_n_state_quad}"
        )
    nrq = disc.get("n_ret_nodes_1d")
    if nrq is not None and tuple(nrq) != expected_n_ret:
        warnings.append(
            f"{label}: disc_config.n_ret_nodes_1d={tuple(nrq)} != expected {expected_n_ret}"
        )
    if int(disc.get("n_z", -1)) != 15:
        warnings.append(f"{label}: disc_config.n_z={disc.get('n_z')} != 15")
    if int(disc.get("n_eta_nodes", -1)) != 3:
        warnings.append(f"{label}: disc_config.n_eta_nodes={disc.get('n_eta_nodes')} != 3")
    if int(disc.get("n_eps_nodes", -1)) != 4:
        warnings.append(f"{label}: disc_config.n_eps_nodes={disc.get('n_eps_nodes')} != 4")
    nf = b["diagnostics"].get("total_newton_failures", None)
    if nf not in (0, None):
        warnings.append(f"{label}: total_newton_failures={nf} (expected 0)")
    status = b["diagnostics"].get("solve_status", None)
    if status not in (None, "complete"):
        warnings.append(f"{label}: solve_status={status!r} (expected 'complete')")
    state_names = (
        b["metadata"].get("run_config", {})
        .get("predictability_ablation", {})
        .get("state_names")
    )
    if state_names not in (["rtb", "y_1"], ("rtb", "y_1")):
        warnings.append(f"{label}: state_names={state_names!r} (expected ('rtb', 'y_1'))")
    for name in ("C", "S", "B"):
        arr = b[name]
        if arr.shape != EXPECTED_SHAPE:
            warnings.append(f"{label}: {name} shape={arr.shape} != {EXPECTED_SHAPE}")
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            warnings.append(f"{label}: {name} contains {n_bad} non-finite cells")
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
    rtb_idx, y1_idx = divmod(int(st), N_Y1)
    return {
        "age_idx": int(age),
        "z_idx": int(z),
        "state_flat_idx": int(st),
        "rtb_idx": rtb_idx,
        "y1_idx": y1_idx,
        "wealth_idx": int(w),
        "age_value": int(START_AGE + age),
    }


def compute_pair_metrics(
    coarse: dict[str, Any], reference: dict[str, Any],
    coarse_label: str, ref_label: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "coarse_label": coarse_label,
        "reference_label": ref_label,
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

    State-quadrature density should affect retirement-age policies because the
    state vector still includes (rtb, y_1) at retirement (y_1 is the persistent
    income-shock state and matters for retirement consumption via the projected
    retirement income). Unlike the (n_eta, n_eps) sweep, retirement divergence
    here is informative, not a sanity check.
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

def _make_plots(bundles: dict[str, dict],
                metrics_payload: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    pairs = metrics_payload["pairs"]

    # --- 1. Convergence-curve: sup-norm vs total quad nodes (state×ret).
    # x = n_state_quad_product * n_ret_quad_product (a convenient cost proxy).
    cfg_by_label = {label: (nsq, nrq) for (label, _dn, nsq, nrq, _r) in CONFIGS}
    nodes_total: dict[str, int] = {}
    for label, (nsq, nrq) in cfg_by_label.items():
        nodes_total[label] = int(nsq[0] * nsq[1] * nrq[0] * nrq[1])

    pairs_against_baseline = [k for k, v in pairs.items()
                              if v["reference_label"] == BASELINE_LABEL]
    if pairs_against_baseline:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
        for ax, name in zip(axes, ("C", "S", "B")):
            xs = []
            ys = []
            labels = []
            for k in pairs_against_baseline:
                m = pairs[k]
                cl = m["coarse_label"]
                xs.append(nodes_total[cl])
                ys.append(m[name]["sup"])
                labels.append(cl)
            order = np.argsort(xs)
            xs_o = [xs[i] for i in order]
            ys_o = [ys[i] for i in order]
            labels_o = [labels[i] for i in order]
            ax.plot(xs_o, ys_o, marker="o", color="C0", lw=1.6)
            for x, y, lbl in zip(xs_o, ys_o, labels_o):
                ax.annotate(lbl, xy=(x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=8)
            ax.set_xlabel("n_state_quad × n_ret_quad (total quad nodes)")
            ax.set_ylabel(f"sup |{name}_coarse - {name}_baseline|")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{name}: divergence vs {BASELINE_LABEL}")
            ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(
            "System II quadrature sweep — sup-norm divergence vs baseline (sq3x3_rq3x3)"
        )
        fig.tight_layout()
        out = fig_dir / "system_ii_quad_convergence_curves.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")

    # --- 2. Per-axis (age, z, wealth) divergence line plots.
    axis_specs = [
        ("per_age", "age (index from start_age=22)", "system_ii_quad_per_age_divergence.png"),
        ("per_z", "z index (n_z=15)", "system_ii_quad_per_z_divergence.png"),
        ("per_wealth", "wealth index (n_w=180)", "system_ii_quad_per_wealth_divergence.png"),
    ]
    cmap = plt.get_cmap("viridis")
    pair_keys = list(pairs.keys())
    if pair_keys:
        for axis_key, axis_label, fname in axis_specs:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, name in zip(axes, ("C", "S", "B")):
                for i, k in enumerate(pair_keys):
                    m = pairs[k]
                    vals = m[name][axis_key]
                    x = np.arange(len(vals))
                    color = cmap(i / max(1, len(pair_keys) - 1))
                    label = f"{m['coarse_label']} vs {m['reference_label']}"
                    ax.plot(x, vals, color=color, label=label, lw=1.4)
                if axis_key == "per_age":
                    ax.axvline(RETIRE_AGE_IDX, color="k", ls=":", lw=0.8,
                               label=f"retire (age {RETIRE_AGE})")
                ax.set_yscale("log")
                ax.set_xlabel(axis_label)
                ax.set_ylabel(f"sup |{name}|")
                ax.set_title(f"{name}: per-{axis_label}")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(loc="best", fontsize=7)
            fig.suptitle(f"Per-{axis_label} divergence — System II quad sweep")
            fig.tight_layout()
            out = fig_dir / fname
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"  wrote {out}")

    # --- 3. State heatmap: 7×7 grid of (rtb, y_1) cells.
    # For each pair, reshape per_state -> (n_rtb, n_y1) and render.
    if pair_keys:
        n_pairs = len(pair_keys)
        for name in ("C", "S", "B"):
            fig, axes = plt.subplots(1, n_pairs, figsize=(4.6 * n_pairs, 4.2),
                                      squeeze=False)
            for j, k in enumerate(pair_keys):
                m = pairs[k]
                per_state = np.asarray(m[name]["per_state"]).reshape(N_RTB, N_Y1)
                ax = axes[0, j]
                im = ax.imshow(per_state, origin="lower", aspect="auto",
                                cmap="viridis")
                ax.set_xlabel("y_1 index (0..6)")
                ax.set_ylabel("rtb index (0..6)")
                ax.set_title(f"{m['coarse_label']} vs {m['reference_label']}")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                              label=f"sup |{name}|")
            fig.suptitle(f"{name}: per-state-cell sup-divergence (7×7 grid, rtb × y_1)")
            fig.tight_layout()
            out = fig_dir / f"system_ii_quad_state_heatmap_{name}.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"  wrote {out}")

    # --- 4. Probe-cell α_s / α_b vs age (one curve per available bundle).
    available_labels = [k for k in bundles if bundles[k] is not None]
    if available_labels:
        ref_b = bundles[available_labels[0]]
        n_z = ref_b["S"].shape[1]
        n_state = ref_b["S"].shape[2]
        n_wealth = ref_b["S"].shape[3]
        z_mid = n_z // 2
        state_mid = n_state // 2
        wealth_mid = n_wealth // 2
        ages = np.arange(START_AGE, TERMINAL_AGE + 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
        for ax, name, title in zip(
            axes, ("S", "B", "C"),
            ("α_s (risky share)", "α_b (bond share)", "C (consumption)"),
        ):
            for i, lbl in enumerate(available_labels):
                arr = bundles[lbl][name]
                curve = arr[:, z_mid, state_mid, wealth_mid]
                ls = "-" if lbl == BASELINE_LABEL else "--"
                ax.plot(ages, curve,
                        color=cmap(i / max(1, len(available_labels) - 1)),
                        label=lbl, ls=ls, lw=1.5)
            ax.axvline(RETIRE_AGE, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.set_xlabel("age")
            ax.set_ylabel(title)
            ax.set_title(f"{title} probe (z={z_mid}/{n_z}, state={state_mid}/{n_state}, w={wealth_mid}/{n_wealth})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("Probe-cell policies across System II quad-sweep bundles")
        fig.tight_layout()
        out = fig_dir / "system_ii_quad_probe_vs_age.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _bucket(sup_C: float, sup_S: float, sup_B: float) -> str:
    """Crude GREEN/YELLOW/RED bucket on max sup-norm divergence.

    Thresholds chosen to mirror the System I eta-eps convention:
      - GREEN:  max(sup_C, sup_S, sup_B) < 1e-3  → policies coincide to 1e-3
      - YELLOW: 1e-3 ≤ max < 1e-2          → small but non-trivial divergence
      - RED:    max ≥ 1e-2                 → material divergence
    """
    worst = max(sup_C, sup_S, sup_B)
    if worst < 1e-3:
        return "GREEN"
    if worst < 1e-2:
        return "YELLOW"
    return "RED"


def make_verdicts(pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Construct the three sub-verdicts described in the handoff."""
    verdict_map: dict[str, str] = {
        "state_quad_uniform_refinement": "PENDING — bundle not on disk",
        "ret_quad_refinement": "PENDING — bundle not on disk",
        "y1_kbump_vs_uniform": "PENDING — bundle not on disk",
    }

    if "sq4x4_rq3x3__vs__sq3x3_rq3x3" in pairs:
        m = pairs["sq4x4_rq3x3__vs__sq3x3_rq3x3"]
        verdict_map["state_quad_uniform_refinement"] = _bucket(
            m["C"]["sup"], m["S"]["sup"], m["B"]["sup"]
        )
    if "sq3x3_rq4x4__vs__sq3x3_rq3x3" in pairs:
        m = pairs["sq3x3_rq4x4__vs__sq3x3_rq3x3"]
        verdict_map["ret_quad_refinement"] = _bucket(
            m["C"]["sup"], m["S"]["sup"], m["B"]["sup"]
        )

    # K-bump vs uniform: we want "(3,5) ≤ (4,4) in divergence vs baseline" for
    # the K-bump to be a free win. Compare the two coarse-vs-baseline pairs.
    p_uni = pairs.get("sq4x4_rq3x3__vs__sq3x3_rq3x3")
    p_bump = pairs.get("sq3x5_rq3x3__vs__sq3x3_rq3x3")
    if p_uni is not None and p_bump is not None:
        worst_uni = max(p_uni["C"]["sup"], p_uni["S"]["sup"], p_uni["B"]["sup"])
        worst_bump = max(p_bump["C"]["sup"], p_bump["S"]["sup"], p_bump["B"]["sup"])
        if worst_bump <= 0.5 * worst_uni:
            verdict_map["y1_kbump_vs_uniform"] = (
                "GREEN — K-bump materially lower under baseline-divergence proxy"
            )
        elif worst_bump <= 1.5 * worst_uni:
            verdict_map["y1_kbump_vs_uniform"] = "YELLOW — K-bump ≈ uniform"
        else:
            verdict_map["y1_kbump_vs_uniform"] = "RED — K-bump worse than uniform"

    return verdict_map


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
        default="system_ii_quad_convergence_metrics.json",
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

    print(f"Loading System II quad-sweep bundles from: {bundles_root}")
    bundles: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    for label, dirname, _nsq, _nrq, _role in CONFIGS:
        b = load_one(bundles_root, dirname)
        bundles[label] = b
        if b is None:
            missing.append(dirname)
            print(f"  MISSING: {label} -> {dirname}")
            continue
        sh = b["C"].shape
        wall = b["diagnostics"].get("wall_time_sec", float("nan"))
        nsq_actual = b["disc_config"].get("n_state_quad_nodes")
        nrq_actual = b["disc_config"].get("n_ret_nodes_1d")
        print(f"  {label}: shape={sh}, n_state_quad={nsq_actual}, "
              f"n_ret_nodes={nrq_actual}, wall={wall:.1f}s")

    print("\nValidation gates:")
    any_warnings = False
    validation: dict[str, list[str]] = {}
    for label, dirname, nsq, nrq, _role in CONFIGS:
        if bundles[label] is None:
            validation[label] = ["bundle missing on disk"]
            continue
        ws = validate_bundle(bundles[label], label, nsq, nrq)
        validation[label] = ws
        if ws:
            any_warnings = True
            for w in ws:
                print(f"  WARN {w}")
    if not any_warnings and not missing:
        print("  all clean.")

    print("\nPairwise divergences:")
    pairs: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, Any]] = {}
    skipped_pairs: list[dict[str, str]] = []
    for coarse_lbl, ref_lbl, effect in PAIR_PLAN:
        if bundles.get(coarse_lbl) is None or bundles.get(ref_lbl) is None:
            print(f"  SKIP {coarse_lbl} vs {ref_lbl} ({effect}): missing bundle(s)")
            skipped_pairs.append(
                {"coarse_label": coarse_lbl, "reference_label": ref_lbl, "effect": effect}
            )
            continue
        m = compute_pair_metrics(bundles[coarse_lbl], bundles[ref_lbl],
                                  coarse_lbl, ref_lbl)
        s = split_working_retirement(bundles[coarse_lbl], bundles[ref_lbl])
        m["effect"] = effect
        pair_key = f"{coarse_lbl}__vs__{ref_lbl}"
        pairs[pair_key] = m
        splits[pair_key] = s
        print(
            f"  {coarse_lbl} vs {ref_lbl} ({effect}) | "
            f"sup_C={m['C']['sup']:.4e} sup_S={m['S']['sup']:.4e} sup_B={m['B']['sup']:.4e} | "
            f"rms_C={m['C']['rms']:.3e} | rel_sup_C={m['C']['sup_rel']:.3e}"
        )
        for name in ("C", "S", "B"):
            print(
                f"      {name}: working_sup={s[name]['working_sup']:.4e}, "
                f"retirement_sup={s[name]['retirement_sup']:.4e}"
            )

    verdicts = make_verdicts(pairs)
    print("\nVerdicts:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")

    distribution: dict[str, dict[str, Any]] = {}
    for label in bundles:
        b = bundles[label]
        if b is None:
            continue
        distribution[label] = {
            "S_quantiles": [float(q) for q in np.quantile(b["S"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "B_quantiles": [float(q) for q in np.quantile(b["B"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "C_quantiles": [float(q) for q in np.quantile(b["C"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
            "S_min": float(b["S"].min()), "S_max": float(b["S"].max()),
            "B_min": float(b["B"].min()), "B_max": float(b["B"].max()),
            "C_min": float(b["C"].min()), "C_max": float(b["C"].max()),
        }

    wall_times = {
        label: float(b["diagnostics"].get("wall_time_sec", float("nan")))
        for label, b in bundles.items() if b is not None
    }

    payload = {
        "bundles_root": str(bundles_root),
        "configs": [
            {"label": lbl, "dirname": dn, "n_state_quad": list(nsq),
             "n_ret_nodes_1d": list(nrq), "role": role,
             "available": bundles[lbl] is not None}
            for (lbl, dn, nsq, nrq, role) in CONFIGS
        ],
        "missing_bundles": missing,
        "baseline_label": BASELINE_LABEL,
        "expected_shape": list(EXPECTED_SHAPE),
        "retire_age_idx": RETIRE_AGE_IDX,
        "start_age": START_AGE,
        "terminal_age": TERMINAL_AGE,
        "validation": validation,
        "pairs": pairs,
        "split_working_retirement": splits,
        "skipped_pairs": skipped_pairs,
        "verdicts": verdicts,
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

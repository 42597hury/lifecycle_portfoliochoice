"""Structural analysis of Newton failures across recent solved bundles.

Reads diagnostics.pkl + metadata.json from a curated bundle list, extracts
per-age fail counts and Newton/backtrack iter histograms, and prints/plots
a structural summary used by docs/scans/NEWTON_FAILURE_STRUCTURE_2026-05-08.md.

Read-only. No solver calls.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
SAVED = REPO / "saved_runs"
OUTDIR = REPO / "docs" / "scans" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


# Curated bundle list. Tuples: (label, kind, path, max_iter_hint).
BUNDLES = [
    # System I wealth-point sweep (post-fix, ~10% fail rate)
    ("S1_w180_log", "lifecycle", SAVED / "ablations/system_i_grid7_nz25_w180_log_calib1"),
    ("S1_w180_bakh", "lifecycle", SAVED / "ablations/system_i_grid7_nz25_w180_bakh_calib1"),
    ("S1_w90_log", "lifecycle", SAVED / "ablations/system_i_grid7_nz25_w90_log_calib1"),
    ("S1_w90_bakh", "lifecycle", SAVED / "ablations/system_i_grid7_nz25_w90_bakh_calib1"),
    ("S1_w60_log", "lifecycle", SAVED / "ablations/system_i_grid7_nz25_w60_log_calib1"),
    ("S1_w120_log", "lifecycle", SAVED / "ablations/system_i_grid7_nz25_w120_log_calib1"),
    # System II grid 7x7 quad sweep (post-fix audit)
    ("S2_7x7_sq3x3_rq3x3", "lifecycle", SAVED / "ablations/system_ii_grid7x7_nz15_sq3x3_rq3x3_calib1"),
    ("S2_7x7_sq4x4_rq3x3", "lifecycle", SAVED / "ablations/system_ii_grid7x7_nz15_sq4x4_rq3x3_calib1"),
    ("S2_7x7_sq3x5_rq3x3", "lifecycle", SAVED / "ablations/system_ii_grid7x7_nz15_sq3x5_rq3x3_calib1"),
    ("S2_7x7_sq3x3_rq4x4", "lifecycle", SAVED / "ablations/system_ii_grid7x7_nz15_sq3x3_rq4x4_calib1"),
    # Inf-horizon (System IV, post-fix)
    ("S4_inf_5x5x5x5_axis1", "inf", SAVED / "inf_horizon/system_iv_inf_axisbump_run1_sq3333_rq33_calib1"),
    ("S4_inf_3x3x3x3", "inf", SAVED / "inf_horizon/system_iv_inf_grid_g3_quad3334_ret44_calib1"),
    ("S4_inf_4x4x4x4", "inf", SAVED / "inf_horizon/system_iv_inf_grid_g4_quad3334_ret44_calib1"),
    ("S4_inf_5x5x5x5", "inf", SAVED / "inf_horizon/system_iv_inf_grid_g5_quad3334_ret44_calib1"),
]


def load_bundle(path: Path) -> dict:
    meta_p = path / "metadata.json"
    diag_p = path / "diagnostics.pkl"
    out = {"path": path}
    if not meta_p.exists():
        return out
    with open(meta_p) as f:
        meta = json.load(f)
    out["meta"] = meta
    if diag_p.exists():
        try:
            with open(diag_p, "rb") as f:
                diag = pickle.load(f)
            out["diag"] = diag
        except Exception as exc:
            out["diag_error"] = str(exc)
    return out


def summarize(label: str, kind: str, path: Path) -> dict | None:
    if not path.exists():
        return None
    b = load_bundle(path)
    if "meta" not in b:
        return None
    meta = b["meta"]
    diag_summary = meta.get("diagnostics_summary", {})
    solver_cfg = diag_summary.get("solver_config", {})
    disc_cfg = diag_summary.get("disc_config", {})
    shape = meta.get("shape")  # lifecycle: (78,n_z,N_state,n_w); inf: (1,N_state,n_w)
    nh = diag_summary.get("newton_iter_histogram", {})
    bh = diag_summary.get("backtrack_iter_histogram", {})
    age_fail = diag_summary.get("age_newton_fail", {})
    age_fail_vals = np.asarray(age_fail.get("values", [])) if isinstance(age_fail, dict) else None
    total_fail = int(diag_summary.get("total_newton_failures", 0))
    n_cells = int(nh.get("n_cells", 0))

    out = {
        "label": label,
        "kind": kind,
        "path": str(path),
        "shape": shape,
        "max_iter": solver_cfg.get("max_iter"),
        "max_backtrack_iter": solver_cfg.get("max_backtrack_iter"),
        "tol": solver_cfg.get("tol"),
        "n_z": disc_cfg.get("n_z"),
        "state_grid_sizes": disc_cfg.get("state_grid_sizes"),
        "n_state_quad_nodes": disc_cfg.get("n_state_quad_nodes"),
        "n_ret_nodes_1d": disc_cfg.get("n_ret_nodes_1d"),
        "n_savings": disc_cfg.get("n_savings"),
        "wall_time_sec": diag_summary.get("wall_time_sec"),
        "newton_p50": nh.get("p50"),
        "newton_p95": nh.get("p95"),
        "newton_p99": nh.get("p99"),
        "newton_max": nh.get("max"),
        "newton_n_cells": n_cells,
        "backtrack_p50": bh.get("p50"),
        "backtrack_p95": bh.get("p95"),
        "backtrack_p99": bh.get("p99"),
        "backtrack_max": bh.get("max"),
        "total_fail": total_fail,
        "fail_rate": (total_fail / n_cells) if n_cells else None,
        "age_fail": age_fail_vals,
        "newton_per_age_p99": np.asarray(nh.get("per_age_p99", []))
        if "per_age_p99" in nh
        else (np.asarray(nh.get("per_iter_p99", [])) if "per_iter_p99" in nh else None),
        "newton_per_age_max": np.asarray(nh.get("per_age_max", []))
        if "per_age_max" in nh
        else (np.asarray(nh.get("per_iter_max", [])) if "per_iter_max" in nh else None),
        "backtrack_per_age_p99": np.asarray(bh.get("per_age_p99", []))
        if "per_age_p99" in bh
        else (np.asarray(bh.get("per_iter_p99", [])) if "per_iter_p99" in bh else None),
        "backtrack_per_age_max": np.asarray(bh.get("per_age_max", []))
        if "per_age_max" in bh
        else (np.asarray(bh.get("per_iter_max", [])) if "per_iter_max" in bh else None),
    }
    return out


def extract_per_age_diag(path: Path) -> dict | None:
    """Read full diagnostics.pkl to get per_age_n_iter_dist if present."""
    diag_p = path / "diagnostics.pkl"
    if not diag_p.exists():
        return None
    try:
        with open(diag_p, "rb") as f:
            diag = pickle.load(f)
    except Exception as exc:
        return {"error": str(exc)}
    keys = list(diag.keys()) if isinstance(diag, dict) else []
    return {"keys": keys, "diag": diag}


def main() -> None:
    rows = []
    for label, kind, path in BUNDLES:
        s = summarize(label, kind, path)
        if s is None:
            print(f"[skip] {label}: missing -> {path}")
            continue
        rows.append(s)

    print()
    print("=" * 110)
    print("Summary")
    print("=" * 110)
    hdr = (
        f"{'label':28s}  {'shape':24s}  {'mxIt':>4s}  {'mxBk':>4s}  "
        f"{'cells':>10s}  {'fails':>9s}  {'rate%':>6s}  "
        f"{'nP50':>4s} {'nP95':>4s} {'nP99':>4s} {'nMx':>4s}  "
        f"{'bP50':>5s} {'bP95':>5s} {'bP99':>5s} {'bMx':>5s}"
    )
    print(hdr)
    for r in rows:
        rate = (r["fail_rate"] or 0) * 100
        print(
            f"{r['label']:28s}  {str(r['shape']):24s}  {str(r['max_iter']):>4s}  "
            f"{str(r['max_backtrack_iter']):>4s}  "
            f"{r['newton_n_cells']:>10d}  {r['total_fail']:>9d}  {rate:>6.2f}  "
            f"{r['newton_p50']:>4.0f} {r['newton_p95']:>4.0f} {r['newton_p99']:>4.0f} "
            f"{r['newton_max']:>4.0f}  "
            f"{r['backtrack_p50']:>5.0f} {r['backtrack_p95']:>5.0f} "
            f"{r['backtrack_p99']:>5.0f} {r['backtrack_max']:>5.0f}"
        )

    # ---- Per-age fail-rate plots (lifecycle bundles only) ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_abs, ax_rate = axes
    for r in rows:
        if r["kind"] != "lifecycle":
            continue
        af = r["age_fail"]
        if af is None or af.size == 0:
            continue
        ax_abs.plot(np.arange(22, 22 + af.size), af, label=r["label"], lw=1.0)
        # cells per age
        shape = r["shape"]
        # lifecycle shape: (n_age, n_z, N_state, n_w/n_savings)
        # cells per age in histogram = n_z * N_state * n_savings (s=0 anchor stripped)
        cells_per_age = (
            shape[1] * shape[2] * (r["n_savings"] or shape[3])
        ) if shape and len(shape) == 4 else None
        if cells_per_age:
            ax_rate.plot(
                np.arange(22, 22 + af.size),
                af / cells_per_age * 100,
                label=r["label"],
                lw=1.0,
            )
    ax_abs.set_ylabel("Newton failures (count)")
    ax_abs.set_yscale("log")
    ax_abs.legend(fontsize=7, loc="upper right", ncol=2)
    ax_abs.grid(True, alpha=0.3)
    ax_rate.set_ylabel("Failure rate (% of cells per age)")
    ax_rate.set_xlabel("Age")
    ax_rate.axvline(67, color="red", lw=0.8, ls="--", alpha=0.6)
    ax_rate.text(67.2, ax_rate.get_ylim()[0] * 1.05 if ax_rate.get_ylim()[0] > 0 else 0.1,
                 "retire age 67", color="red", fontsize=8)
    ax_rate.grid(True, alpha=0.3)
    fig.suptitle("Newton failures vs age — lifecycle bundles (post-8bfaec9 fix)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "newton_fail_per_age.png", dpi=140)
    plt.close(fig)
    print(f"[ok] wrote {OUTDIR / 'newton_fail_per_age.png'}")

    # ---- Per-age newton_p99 / max for lifecycle ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for r in rows:
        if r["kind"] != "lifecycle":
            continue
        v = r["newton_per_age_max"]
        if v is None or v.size == 0:
            continue
        ax.plot(np.arange(22, 22 + v.size), v, label=r["label"], lw=1.0)
    ax.set_ylabel("Newton iter (per-age max)")
    ax.set_xlabel("Age")
    ax.axvline(67, color="red", lw=0.8, ls="--", alpha=0.6)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-age Newton-iter max — lifecycle bundles")
    fig.tight_layout()
    fig.savefig(OUTDIR / "newton_iter_max_per_age.png", dpi=140)
    plt.close(fig)
    print(f"[ok] wrote {OUTDIR / 'newton_iter_max_per_age.png'}")

    # ---- Per-iter (inf-horizon) backtrack & failures plot ----
    inf_rows = [r for r in rows if r["kind"] == "inf"]
    if inf_rows:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax_iter_p99, ax_bk_p99 = axes
        for r in inf_rows:
            v = r["newton_per_age_p99"]
            if v is not None and v.size > 0:
                ax_iter_p99.plot(np.arange(v.size), v, label=r["label"], lw=1.0)
            v = r["backtrack_per_age_p99"]
            if v is not None and v.size > 0:
                ax_bk_p99.plot(np.arange(v.size), v, label=r["label"], lw=1.0)
        ax_iter_p99.set_ylabel("Newton-iter p99")
        ax_iter_p99.legend(fontsize=8)
        ax_iter_p99.grid(True, alpha=0.3)
        ax_bk_p99.set_ylabel("Backtrack p99 (per-cell sum)")
        ax_bk_p99.set_xlabel("Bellman iter index")
        ax_bk_p99.grid(True, alpha=0.3)
        fig.suptitle("Inf-horizon: per-iter Newton/backtrack histograms")
        fig.tight_layout()
        fig.savefig(OUTDIR / "newton_inf_horizon_per_iter.png", dpi=140)
        plt.close(fig)
        print(f"[ok] wrote {OUTDIR / 'newton_inf_horizon_per_iter.png'}")

    # ---- Save a JSON of the table for the markdown ----
    json_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        rr["age_fail_head"] = r["age_fail"][:5].tolist() if r["age_fail"] is not None and r["age_fail"].size else None
        rr["age_fail_tail"] = r["age_fail"][-5:].tolist() if r["age_fail"] is not None and r["age_fail"].size else None
        rr["age_fail_argmax"] = int(np.argmax(r["age_fail"])) if r["age_fail"] is not None and r["age_fail"].size else None
        rr["age_fail_max"] = int(np.max(r["age_fail"])) if r["age_fail"] is not None and r["age_fail"].size else None
        json_rows.append(rr)

    out_json = REPO / "docs/scans/newton_failure_structure_2026-05-08.json"
    with open(out_json, "w") as f:
        json.dump(json_rows, f, indent=2, default=str)
    print(f"[ok] wrote {out_json}")


if __name__ == "__main__":
    main()

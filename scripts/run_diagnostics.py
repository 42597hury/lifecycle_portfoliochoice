"""Auto-run the EE diagnostic battery on a freshly-solved bundle.

Designed to be invoked at the end of `scripts/run_solve.py` (i.e. on the EC2
instance, immediately after `save_policy_bundle`, before S3 upload), but it
also works as a standalone retrofit on any bundle dir on disk.

Runs the four headline diagnostics from `docs/workflows/EE_DIAGNOSTIC_WORKFLOW.md`
in order:

    1. _diag_arbitrage_quadsweep   (~30s, sanity: quadrature wiring + min R_p)
    2. _diag_invalid_cells         (~5min, per-cell breakdown of invalidity)
    3. _diag_gridpoint_ee          (~3min, gridpoint EE under next_finer)
    4. _diag_euler_errors          (~5min, sim-path EE vs publication gates)

Outputs land in <bundle_dir>/diagnostics_reports/ as markdown plus a
<bundle_dir>/diagnostics_summary.json manifest of status + report paths.

Each diagnostic is isolated in try/except: a failure in one does NOT abort
the others, and never aborts the caller. The summary records which ones
passed/failed and the exception text.

Usage (standalone):
    python scripts/run_diagnostics.py saved_runs/<bundle-name>
    python scripts/run_diagnostics.py <bundle> --skip arbitrage invalid_cells
    python scripts/run_diagnostics.py <bundle> --n-simulations 1000     # smaller smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Initial-condition defaults for the sim-path diagnostic. These match the
# documented convention in EE_DIAGNOSTIC_WORKFLOW.md §8 / §3 and are written
# explicitly into diagnostics_summary.json so a future reader can reproduce
# the run from the bundle alone.
SIMPATH_DEFAULTS = {
    "n_simulations": 5000,
    "eval_households_per_age": 256,
    "initial_z": "stationary",
    "initial_state": "median",
    "partial_init_mode": "centered",
    "eval_mode": "next_finer",
}

GRIDPOINT_DEFAULTS = {
    "eval_mode": "next_finer",
}


DIAGNOSTIC_KEYS = ("arbitrage", "invalid_cells", "gridpoint_ee", "simpath_ee")


def _diag_argv(name: str, bundle: Path, model_bundle: Path, reports_dir: Path,
               label: str, simpath_args: dict[str, Any]) -> tuple[list[str], Path]:
    """Build the argv list and the markdown-output path for one diagnostic."""
    if name == "arbitrage":
        out = reports_dir / f"diagnostics_arbitrage_{label}.md"
        argv = [
            "--bundle", str(bundle),
            "--markdown-out", str(out),
        ]
    elif name == "invalid_cells":
        out = reports_dir / f"diagnostics_invalid_cells_{label}.md"
        argv = [
            "--bundle", str(bundle),
            "--model-bundle", str(model_bundle),
            "--eval-mode", "next_finer",
            "--markdown-out", str(out),
        ]
    elif name == "gridpoint_ee":
        out = reports_dir / f"diagnostics_gridpoint_ee_{label}_nextfiner.md"
        argv = [
            str(bundle),
            "--model-bundle", str(model_bundle),
            "--eval-mode", GRIDPOINT_DEFAULTS["eval_mode"],
            "--markdown-out", str(out),
        ]
    elif name == "simpath_ee":
        out = reports_dir / f"diagnostics_simpath_ee_{label}_nextfiner.md"
        argv = [
            str(bundle),
            "--model-bundle", str(model_bundle),
            "--eval-mode", simpath_args["eval_mode"],
            "--n-simulations", str(simpath_args["n_simulations"]),
            "--eval-households-per-age", str(simpath_args["eval_households_per_age"]),
            "--initial-z", simpath_args["initial_z"],
            "--initial-state", simpath_args["initial_state"],
            "--partial-init-mode", simpath_args["partial_init_mode"],
            "--markdown-out", str(out),
        ]
        if simpath_args.get("initial_x") is not None:
            argv += ["--initial-x", str(simpath_args["initial_x"])]
    else:
        raise ValueError(f"Unknown diagnostic: {name}")
    return argv, out


def _invoke(name: str, argv: list[str]) -> None:
    """Call a diagnostic's main(argv) in-process. Raises on failure."""
    if name == "arbitrage":
        from scripts.diagnostics._diag_arbitrage_quadsweep import main as _main
    elif name == "invalid_cells":
        from scripts.diagnostics._diag_invalid_cells import main as _main
    elif name == "gridpoint_ee":
        from scripts.diagnostics._diag_gridpoint_ee import main as _main
    elif name == "simpath_ee":
        from scripts.diagnostics._diag_euler_errors import main as _main
    else:
        raise ValueError(f"Unknown diagnostic: {name}")
    rc = _main(argv)
    if rc not in (None, 0):
        raise RuntimeError(f"{name} returned non-zero exit code: {rc}")


def run_diagnostics(
    bundle_dir: Path,
    model_bundle: Path | None = None,
    skip: tuple[str, ...] = (),
    simpath_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the four-diagnostic battery on `bundle_dir`. Always returns a manifest dict.

    - bundle_dir: the freshly-saved bundle directory (must contain metadata.json)
    - model_bundle: bundle that supplies run_config metadata; defaults to bundle_dir
    - skip: subset of DIAGNOSTIC_KEYS to skip
    - simpath_overrides: dict that overrides keys in SIMPATH_DEFAULTS (e.g.
        {"n_simulations": 1000} for smoke runs)
    """
    bundle_dir = Path(bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir does not exist: {bundle_dir}")
    model_bundle = Path(model_bundle).resolve() if model_bundle else bundle_dir

    label = bundle_dir.name
    reports_dir = bundle_dir / "diagnostics_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    simpath_args = dict(SIMPATH_DEFAULTS)
    if simpath_overrides:
        simpath_args.update(simpath_overrides)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "bundle_label": label,
        "bundle_dir": str(bundle_dir),
        "model_bundle_dir": str(model_bundle),
        "diagnostic_eval_mode": "next_finer",
        "simpath_initial_conditions": {
            "n_simulations": simpath_args["n_simulations"],
            "eval_households_per_age": simpath_args["eval_households_per_age"],
            "initial_z": simpath_args["initial_z"],
            "initial_state": simpath_args["initial_state"],
            "initial_x": simpath_args.get("initial_x"),
            "partial_init_mode": simpath_args["partial_init_mode"],
        },
        "diagnostics": {},
    }

    overall_t0 = time.time()
    for name in DIAGNOSTIC_KEYS:
        if name in skip:
            print(f"[diagnostics] {name}: SKIPPED (--skip)")
            manifest["diagnostics"][name] = {"status": "skipped"}
            continue

        argv, out_path = _diag_argv(
            name, bundle_dir, model_bundle, reports_dir, label, simpath_args,
        )
        print(f"[diagnostics] {name}: starting -> {out_path.name}")
        t0 = time.time()
        try:
            _invoke(name, argv)
            elapsed = time.time() - t0
            print(f"[diagnostics] {name}: ok ({elapsed:.1f}s)")
            manifest["diagnostics"][name] = {
                "status": "ok",
                "elapsed_seconds": round(elapsed, 1),
                "report_path": str(out_path.relative_to(bundle_dir)),
                "argv": argv,
            }
        except Exception as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            print(f"[diagnostics] {name}: FAILED after {elapsed:.1f}s: {exc}")
            print(tb)
            manifest["diagnostics"][name] = {
                "status": "failed",
                "elapsed_seconds": round(elapsed, 1),
                "error": str(exc),
                "traceback": tb,
                "argv": argv,
            }

    manifest["total_runtime_seconds"] = round(time.time() - overall_t0, 1)

    statuses = [d["status"] for d in manifest["diagnostics"].values()]
    if all(s in ("ok", "skipped") for s in statuses) and "ok" in statuses:
        manifest["overall_status"] = "complete"
    elif any(s == "ok" for s in statuses):
        manifest["overall_status"] = "partial"
    else:
        manifest["overall_status"] = "failed"

    summary_path = bundle_dir / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[diagnostics] wrote summary: {summary_path}")
    print(f"[diagnostics] overall status: {manifest['overall_status']} "
          f"(total {manifest['total_runtime_seconds']:.1f}s)")

    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", type=Path,
                   help="Path to the bundle directory (must contain metadata.json).")
    p.add_argument("--model-bundle", type=Path, default=None,
                   help="Bundle supplying run_config metadata; defaults to `bundle`.")
    p.add_argument("--skip", nargs="*", default=[], choices=list(DIAGNOSTIC_KEYS),
                   help="Diagnostic names to skip (any subset of "
                        f"{', '.join(DIAGNOSTIC_KEYS)}).")
    p.add_argument("--n-simulations", type=int, default=None,
                   help=f"Override sim-path n_simulations "
                        f"(default {SIMPATH_DEFAULTS['n_simulations']}).")
    p.add_argument("--eval-households-per-age", type=int, default=None,
                   help=f"Override sim-path eval cap "
                        f"(default {SIMPATH_DEFAULTS['eval_households_per_age']}).")
    p.add_argument("--initial-z", default=None,
                   choices=("median", "stationary", "normal"))
    p.add_argument("--initial-state", default=None,
                   choices=("median", "stationary"))
    p.add_argument("--initial-x", type=float, default=None)
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.n_simulations is not None:
        overrides["n_simulations"] = args.n_simulations
    if args.eval_households_per_age is not None:
        overrides["eval_households_per_age"] = args.eval_households_per_age
    if args.initial_z is not None:
        overrides["initial_z"] = args.initial_z
    if args.initial_state is not None:
        overrides["initial_state"] = args.initial_state
    if args.initial_x is not None:
        overrides["initial_x"] = args.initial_x

    manifest = run_diagnostics(
        args.bundle,
        model_bundle=args.model_bundle,
        skip=tuple(args.skip),
        simpath_overrides=overrides,
    )
    return 0 if manifest["overall_status"] in ("complete", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())

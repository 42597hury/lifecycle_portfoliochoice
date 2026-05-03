"""Run the lifecycle solver from a config file and save the bundle.

Usage:
    python scripts/run_solve.py configs/<name>.py [options]

The config is a Python file defining: PREDICTABILITY_SYSTEM, base_config,
disc_config_template, solver_config, and (optionally) BUNDLE_SUFFIX.
See configs/system_iv_5x5x5.py for a template.

Bundle is written to ./saved_runs/<name>/ using the same naming scheme as
main.ipynb. If --bucket (or $S3_BUCKET) is set, the bundle is also synced to
s3://<bucket>/saved_runs/<name>/.

Crash safety
------------
When --bucket is set, this script also:

  1. Looks for an existing per-config checkpoint at
     s3://<bucket>/checkpoints/<bundle-name>/, downloads it locally to
     ./saved_runs/checkpoints/<bundle-name>/, and resumes from the last solved
     age. The solver refuses to resume if grid/quadrature shapes don't match.
  2. Configures the solver to write a per-age checkpoint to
     ./saved_runs/checkpoints/<bundle-name>/ (sidecar uploader on the EC2 host
     syncs it to S3 every 60s -- see scripts/ec2_userdata.sh).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("run_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_bundle_name(system_label, base_config, disc_config, suffix):
    constraint = "constrained" if base_config["constrained"] else "unconstrained"
    grid = "x".join(str(s) for s in disc_config.state_grid_sizes)
    return (
        f"{system_label}_{constraint}_{disc_config.state_grid_mode}"
        f"_grid{grid}_nz{disc_config.n_z}{suffix}"
    )


def upload_bundle(bundle_dir: Path, bucket: str) -> None:
    dest = f"s3://{bucket}/saved_runs/{bundle_dir.name}/"
    print(f"[upload] {bundle_dir} -> {dest}")
    subprocess.run(
        ["aws", "s3", "cp", "--recursive", str(bundle_dir), dest],
        check=True,
    )
    print("[upload] done")


def download_checkpoint_from_s3(bucket: str, bundle_name: str, local_dir: Path) -> bool:
    """If a checkpoint for this bundle exists in S3, sync it locally.

    Returns True if a checkpoint was downloaded, False if none exists.
    """
    s3_uri = f"s3://{bucket}/checkpoints/{bundle_name}/"
    # Probe with `aws s3 ls` (returns non-zero exit if prefix is empty/missing).
    cp = subprocess.run(
        ["aws", "s3", "ls", s3_uri],
        capture_output=True, text=True,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        print(f"[checkpoint] no existing checkpoint at {s3_uri}")
        return False

    print(f"[checkpoint] found checkpoint at {s3_uri} -- syncing to {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", s3_uri, str(local_dir)],
        check=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path,
                        help="Path to a Python config file (see configs/).")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"),
                        help="S3 bucket for upload (default: $S3_BUCKET).")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip S3 upload even if --bucket is set.")
    parser.add_argument("--bundle-suffix", default=None,
                        help="Override BUNDLE_SUFFIX from the config "
                             "(e.g. '_ec2test' for trial runs).")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "saved_runs",
                        help="Directory to write the bundle into.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable per-age checkpointing and S3 resume "
                             "(default: checkpoint to saved_runs/checkpoints/"
                             "<bundle>/ and resume from S3 if a checkpoint "
                             "exists there).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from lifecycle.precompute import build_model, Precompute
    from lifecycle.predictability_ablation import prepare_predictability_system
    from lifecycle.solver import run_lifecycle_solver
    from lifecycle.model import SolveControl
    from lifecycle.policy_io import save_policy_bundle

    var_csv = PROJECT_ROOT / "data" / "var_dataset.csv"
    if not var_csv.exists():
        raise FileNotFoundError(f"Missing VAR dataset: {var_csv}")

    print(f"[run_solve] config: {args.config}")
    print(f"[run_solve] system: {cfg.PREDICTABILITY_SYSTEM}")

    system_setup = prepare_predictability_system(
        cfg.PREDICTABILITY_SYSTEM,
        csv_path=str(var_csv),
        disc_config_template=cfg.disc_config_template,
    )
    var_config = system_setup["var_config"]
    disc_config = system_setup["disc_config"]
    system_label = system_setup["system_label"]

    suffix = (
        args.bundle_suffix if args.bundle_suffix is not None
        else getattr(cfg, "BUNDLE_SUFFIX", "_v2")
    )
    bundle_name = build_bundle_name(system_label, cfg.base_config, disc_config, suffix)
    bundle_dir = args.output_root / bundle_name
    print(f"[run_solve] bundle: {bundle_dir}")

    # --- Checkpointing ---
    checkpoint_dir = args.output_root / "checkpoints" / bundle_name
    solve_control = None
    if not args.no_checkpoint:
        # Try to resume from S3 if a checkpoint already exists for this bundle.
        if args.bucket:
            try:
                download_checkpoint_from_s3(args.bucket, bundle_name, checkpoint_dir)
            except subprocess.CalledProcessError as exc:
                # Don't fail the whole run if checkpoint sync fails -- just
                # warn and start fresh. The local solver still writes
                # checkpoints below; they'll be uploaded by the sidecar.
                print(f"[checkpoint] S3 sync failed ({exc}); starting fresh")

        # Per-config SolveControl, identical across all 10 configs.
        # Override hook: if a config defines SOLVE_CONTROL, that wins.
        default_sc = SolveControl(
            youngest_age_to_solve=None,
            checkpoint_path=str(checkpoint_dir),
            checkpoint_every_n_ages=1,
            save_on_interrupt=True,
            return_partial_on_interrupt=True,
            progress_wealth_source="scf_median",
        )
        cfg_sc = getattr(cfg, "SOLVE_CONTROL", None)
        if cfg_sc is not None:
            # Merge: respect any explicit fields the config sets, but always
            # override checkpoint_path so it points to this bundle's dir.
            solve_control = cfg_sc._replace(checkpoint_path=str(checkpoint_dir))
        else:
            solve_control = default_sc
        print(f"[run_solve] checkpoint dir: {checkpoint_dir}")
        print(f"[run_solve] solve_control: {solve_control}")

    model = build_model(cfg.base_config, var_config, verbose=True)
    pc = Precompute(model, disc_config)

    t0 = time.time()
    print("[run_solve] solving...")
    C_mat, S_mat, B_mat, diagnostics = run_lifecycle_solver(
        model, pc,
        solver_config=cfg.solver_config,
        solve_control=solve_control,
    )
    print(f"[run_solve] solve done in {time.time() - t0:.1f}s")

    constraint_label = "constrained" if cfg.base_config["constrained"] else "unconstrained"
    run_config = {
        "base_config": cfg.base_config,
        "discretization_config": disc_config._asdict(),
        "solver_config": cfg.solver_config._asdict(),
        "var_config": var_config,
        "predictability_ablation": {
            "system_code": system_setup["system_code"],
            "system_label": system_label,
            "system_title": system_setup["system_title"],
            "var_builder_name": system_setup["var_builder_name"],
            "state_names": list(system_setup["state_names"]),
        },
        "constrained": bool(model.constrained),
        "solve_label": constraint_label,
    }
    save_policy_bundle(
        bundle_dir,
        C_mat, S_mat, B_mat,
        diagnostics=diagnostics,
        run_config=run_config,
        overwrite=True,
    )
    print(f"[run_solve] saved bundle: {bundle_dir}")

    if args.no_upload:
        print("[run_solve] --no-upload set; skipping S3")
    elif args.bucket:
        upload_bundle(bundle_dir, args.bucket)
    else:
        print("[run_solve] no --bucket or $S3_BUCKET; skipping S3")


if __name__ == "__main__":
    main()

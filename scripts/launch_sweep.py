"""Launch one EC2 instance per config in a sweep directory.

Usage:
    python scripts/launch_sweep.py configs/sweep_2026_05/ --key-name hugo-thesis

What it does:
    1. Finds every *.py file in the given directory (skipping __init__.py and _*.py).
    2. Launches one EC2 instance per config, in parallel.
    3. Each instance runs scripts/run_solve.py with its config and uploads the
       bundle to s3://<bucket>/saved_runs/<bundle-name>/.
    4. Each instance auto-terminates on success; stays up on failure for SSH.

Each config in the sweep must produce a DISTINCT bundle name. If two configs
produce the same bundle name (same system, grid, mode, n_z, suffix), they will
overwrite each other in S3. Use BUNDLE_SUFFIX to differentiate runs that share
grid/constraint settings (e.g. "_gamma3", "_gamma5", "_high_beta").

After launching, watch progress per-instance the same way as a single run:
    aws s3 cp s3://<bucket>/launches/<launch-id>/userdata.log - | Select-Object -Last 50

When all are done:
    aws s3 sync s3://<bucket>/saved_runs/ ./saved_runs/

Use --dry-run to preview without launching.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT = PROJECT_ROOT / "scripts" / "launch_run.py"


def discover_configs(sweep_dir: Path) -> list[Path]:
    if not sweep_dir.is_dir():
        sys.exit(f"sweep dir not found or not a directory: {sweep_dir}")
    configs = sorted(
        p for p in sweep_dir.glob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("_")
    )
    if not configs:
        sys.exit(f"no configs found in {sweep_dir} (looking for *.py files)")
    return configs


def launch_one(config_path: Path, base_args: list[str]) -> tuple[Path, int, str]:
    cmd = [sys.executable, str(LAUNCH_SCRIPT), str(config_path), *base_args]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    return config_path, cp.returncode, (cp.stdout + cp.stderr).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sweep_dir", type=Path,
                    help="Directory of config *.py files; one launch per file.")
    ap.add_argument("--instance-type", default="c6i.4xlarge")
    ap.add_argument("--key-name", default=None)
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-parallel", type=int, default=10,
                    help="Max simultaneous launches (default 10).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List configs that would launch; don't launch.")
    args = ap.parse_args()

    configs = discover_configs(args.sweep_dir)
    print(f"[sweep] found {len(configs)} configs in {args.sweep_dir}:")
    for c in configs:
        print(f"        - {c.name}")

    if args.dry_run:
        print("[sweep] dry-run; not launching.")
        return

    base_args = ["--instance-type", args.instance_type]
    if args.key_name:
        base_args += ["--key-name", args.key_name]
    if args.bucket:
        base_args += ["--bucket", args.bucket]
    if args.region:
        base_args += ["--region", args.region]

    print(f"[sweep] launching {len(configs)} instances of {args.instance_type} "
          f"(max {args.max_parallel} concurrent)...")
    print()

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(launch_one, c, base_args): c for c in configs}
        for fut in concurrent.futures.as_completed(futures):
            cfg, code, out = fut.result()
            status = "OK" if code == 0 else f"FAILED (exit {code})"
            print(f"--- {cfg.name}: {status} ---")
            print(out)
            print()
            if code != 0:
                failures.append(cfg.name)

    ok_count = len(configs) - len(failures)
    print(f"[sweep] launched {ok_count}/{len(configs)} instances")
    if failures:
        print(f"[sweep] FAILED: {failures}")
        sys.exit(1)

    print()
    print("[sweep] follow-up:")
    print("  # See all running thesis instances:")
    print("  aws ec2 describe-instances --filters \"Name=tag:Name,Values=thesis-*\" "
          "\"Name=instance-state-name,Values=running\" "
          "--query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`].Value|[0]]' "
          "--output table --region eu-north-1")
    print("  # When all are done, pull bundles:")
    print("  aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/")


if __name__ == "__main__":
    main()

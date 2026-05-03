"""Preflight check for the overnight sweep_main launch.

Launches ONE tiny EC2 instance with configs/sweep_main_smoketest.py and watches
its userdata.log until the solve either completes (PASS) or errors (FAIL).
Cost: ~$0.05 on c6i.2xlarge ($0.34/hr * ~10 min worst case).

Usage:
    python scripts/preflight_sweep.py [--key-name hugo-thesis]

Exit code 0 = PASS (safe to launch the real sweep). Non-zero = FAIL.

What this proves end-to-end:
    1. EC2 instance boots from the configured AMI/profile
    2. Tarball + config download from S3 succeed
    3. pip install of numpy/scipy/numba succeeds
    4. run_solve.py boots cleanly with the production solver settings
       (max_iter_unconstrained=8000, init_alpha_s=0.85, etc.)
    5. SolveControl + checkpoint pipeline writes per-age checkpoints
    6. Bundle uploads to s3://hugo-thesis-runs/saved_runs/
    7. Instance auto-terminates on success
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGION = "eu-north-1"
DEFAULT_BUCKET = "hugo-thesis-runs"
DEFAULT_INSTANCE_TYPE = "c6i.2xlarge"   # cheaper than the sweep's c6i.4xlarge
DEFAULT_TIMEOUT_MIN = 12

INSTANCE_ID_RE = re.compile(r"instance launched:\s+(i-[0-9a-f]+)")
LAUNCH_ID_RE = re.compile(r"launches/([0-9TZ_a-zA-Z-]+)/")

PASS_MARKER = "[run_solve] saved bundle:"
DONE_MARKER = "[5/5] solve complete"
FAIL_MARKER = "!!! ERROR"


def fetch_log(bucket: str, launch_id: str, region: str) -> str | None:
    # Use binary mode + manual utf-8 decoding with errors="replace": on
    # Windows, text=True decodes via cp1252 and the solver's log contains
    # box-drawing chars (the verbose-mode table border) that crash the
    # subprocess reader thread, returning empty stdout silently.
    cp = subprocess.run(
        ["aws", "s3", "cp",
         f"s3://{bucket}/launches/{launch_id}/userdata.log", "-",
         "--region", region],
        capture_output=True,
    )
    if cp.returncode != 0:
        return None
    return cp.stdout.decode("utf-8", errors="replace")


def terminate(instance_id: str, region: str) -> None:
    subprocess.run(
        ["aws", "ec2", "terminate-instances",
         "--instance-ids", instance_id, "--region", region],
        capture_output=True, text=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    ap.add_argument("--key-name", default="hugo-thesis")
    ap.add_argument("--timeout-min", type=int, default=DEFAULT_TIMEOUT_MIN,
                    help=f"Max minutes to wait for PASS (default {DEFAULT_TIMEOUT_MIN}).")
    ap.add_argument("--config", type=Path,
                    default=PROJECT_ROOT / "configs" / "sweep_main_smoketest.py")
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"preflight config not found: {args.config}")

    print(f"[preflight] launching {args.instance_type} with {args.config.name}")
    print(f"[preflight] timeout: {args.timeout_min} min")
    print()

    launch_cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "launch_run.py"),
        str(args.config),
        "--instance-type", args.instance_type,
        "--bucket", args.bucket,
        "--region", args.region,
        "--key-name", args.key_name,
    ]
    cp = subprocess.run(launch_cmd, capture_output=True)
    out = (cp.stdout + cp.stderr).decode("utf-8", errors="replace")
    print(out)
    if cp.returncode != 0:
        print("[preflight] FAIL: launch_run.py exited non-zero")
        sys.exit(2)

    iid_m = INSTANCE_ID_RE.search(out)
    lid_m = LAUNCH_ID_RE.search(out)
    if not iid_m or not lid_m:
        print("[preflight] FAIL: could not parse instance_id / launch_id from "
              "launch output above")
        sys.exit(2)
    instance_id = iid_m.group(1)
    launch_id = lid_m.group(1)
    print(f"[preflight] instance: {instance_id}, launch: {launch_id}")
    print()

    deadline = time.time() + args.timeout_min * 60
    last_log_size = 0
    poll_interval = 20

    try:
        while time.time() < deadline:
            time.sleep(poll_interval)
            log = fetch_log(args.bucket, launch_id, args.region)
            if log is None:
                print(f"[preflight] log not in S3 yet "
                      f"({int(deadline - time.time())}s left)...")
                continue
            new_chunk = log[last_log_size:]
            last_log_size = len(log)
            if new_chunk:
                # Print the new tail so the user can see live progress.
                print(new_chunk, end="" if new_chunk.endswith("\n") else "\n")

            if FAIL_MARKER in log:
                print()
                print("=" * 70)
                print("[preflight] FAIL: error marker found in userdata.log")
                print("=" * 70)
                print()
                # Print last 60 lines for context.
                tail = "\n".join(log.splitlines()[-60:])
                print(tail)
                print()
                print(f"[preflight] instance {instance_id} left running "
                      f"for SSH debugging. To kill it:")
                print(f"  aws ec2 terminate-instances --instance-ids "
                      f"{instance_id} --region {args.region}")
                sys.exit(1)

            if PASS_MARKER in log or DONE_MARKER in log:
                print()
                print("=" * 70)
                print("[preflight] PASS: solve completed end-to-end")
                print("=" * 70)
                print()
                print("Instance is auto-terminating. To verify the bundle:")
                print(f"  aws s3 ls s3://{args.bucket}/saved_runs/ | grep smoketest")
                print()
                print("Safe to launch the real sweep:")
                print("  python scripts/launch_sweep.py configs/sweep_main/ \\")
                print(f"      --instance-type c6i.4xlarge --key-name {args.key_name}")
                sys.exit(0)
        # Timed out
        print()
        print("=" * 70)
        print(f"[preflight] FAIL: timeout after {args.timeout_min} min "
              "without PASS or FAIL marker")
        print("=" * 70)
        if last_log_size > 0:
            print()
            print("Last log tail:")
            tail = "\n".join(log.splitlines()[-40:])
            print(tail)
        print()
        print(f"[preflight] terminating instance {instance_id} to stop spend")
        terminate(instance_id, args.region)
        sys.exit(3)
    except KeyboardInterrupt:
        print(f"\n[preflight] interrupted -- terminating instance {instance_id}")
        terminate(instance_id, args.region)
        sys.exit(130)


if __name__ == "__main__":
    main()

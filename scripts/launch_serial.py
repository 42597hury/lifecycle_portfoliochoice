"""Launch ONE EC2 instance to solve a SEQUENCE of configs back-to-back.

Usage:
    python scripts/launch_serial.py --configs configs/sweep_main/03_state33.py \\
        configs/sweep_main/04_state33_cap.py configs/sweep_main/05_state44_cap.py \\
        configs/sweep_main/09_grid9_state33_cap.py \\
        --instance-type c6i.24xlarge --key-name hugo-thesis

Difference from launch_run.py:
    - Accepts multiple configs (run in the order given)
    - Uses scripts/ec2_userdata_serial.sh as bootstrap
    - The instance solves each config in order, resuming from S3 checkpoints
      if available; one config's failure doesn't block the rest.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARBALL_EXCLUDES = {
    "__pycache__", ".git", ".pytest_cache", ".claude", ".vscode",
    "saved_runs", "archive", "_t9_min_rport_map.npz",
}

INSTANCE_HOURLY_USD = {
    "c6i.4xlarge": 0.68, "c6i.8xlarge": 1.36, "c6i.16xlarge": 2.72,
    "c6i.24xlarge": 4.08, "c6i.32xlarge": 5.44,
}

DEFAULT_INSTANCE_TYPE = "c6i.24xlarge"
DEFAULT_REGION = "eu-north-1"
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "hugo-thesis-runs")
DEFAULT_AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64"
DEFAULT_INSTANCE_PROFILE = "thesis-ec2-runner"
DEFAULT_ROOT_VOLUME_GB = 50  # bigger than single-run since 4 bundles will land


def aws_text(cmd: list[str]) -> str:
    cp = subprocess.run(cmd, capture_output=True)
    if cp.returncode != 0:
        sys.stderr.write(
            f"\n[launch] aws CLI failed (exit {cp.returncode}):\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr:  {cp.stderr.decode('utf-8', errors='replace').strip()}\n"
        )
        sys.exit(cp.returncode)
    return cp.stdout.decode("utf-8", errors="replace").strip()


def aws_json(cmd: list[str]) -> dict:
    cp = subprocess.run(cmd, capture_output=True)
    if cp.returncode != 0:
        sys.stderr.write(
            f"\n[launch] aws CLI failed (exit {cp.returncode}):\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr:  {cp.stderr.decode('utf-8', errors='replace').strip()}\n"
        )
        sys.exit(cp.returncode)
    return json.loads(cp.stdout.decode("utf-8", errors="replace"))


def lookup_ami(region: str) -> str:
    return aws_text([
        "aws", "ssm", "get-parameter",
        "--name", DEFAULT_AMI_PARAM,
        "--region", region,
        "--query", "Parameter.Value", "--output", "text",
    ])


def make_tarball(out_path: Path) -> None:
    print(f"[launch] building tarball: {out_path}")

    def filter_fn(tarinfo):
        for part in tarinfo.name.split("/"):
            if part in TARBALL_EXCLUDES:
                return None
        return tarinfo

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(PROJECT_ROOT, arcname="thesis", filter=filter_fn)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[launch] tarball size: {size_mb:.1f} MB")


def render_userdata(template: str, **subs) -> str:
    out = template
    for key, val in subs.items():
        out = out.replace(f"{{{{ {key} }}}}", str(val))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--configs", nargs="+", type=Path, required=True,
                    help="Configs to solve, in order. Each is a .py file.")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--ami", default=None)
    ap.add_argument("--key-name", default=None)
    ap.add_argument("--instance-profile", default=DEFAULT_INSTANCE_PROFILE)
    ap.add_argument("--security-group-ids", nargs="*", default=None)
    ap.add_argument("--root-volume-gb", type=int, default=DEFAULT_ROOT_VOLUME_GB)
    ap.add_argument("--label", default=None,
                    help="Short label for the launch_id (default: 'serial').")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for cfg in args.configs:
        if not cfg.exists():
            sys.exit(f"config not found: {cfg}")

    label = args.label or "serial"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    launch_id = f"{stamp}_{label}"
    s3_launch_prefix = f"launches/{launch_id}"
    s3_url = f"s3://{args.bucket}/{s3_launch_prefix}"

    config_basenames = [c.name for c in args.configs]
    config_list_str = " ".join(config_basenames)

    print(f"[launch] launch_id:      {launch_id}")
    print(f"[launch] configs ({len(args.configs)} in order):")
    for i, c in enumerate(args.configs, 1):
        print(f"  [{i}] {c.name}")
    print(f"[launch] bucket:         {args.bucket} ({args.region})")
    price = INSTANCE_HOURLY_USD.get(args.instance_type, "?")
    price_str = f"${price}/hr" if isinstance(price, (int, float)) else f"({price})"
    print(f"[launch] type:           {args.instance_type} ~{price_str} on-demand in eu-north-1")

    template_path = PROJECT_ROOT / "scripts" / "ec2_userdata_serial.sh"
    template = template_path.read_text()
    userdata = render_userdata(
        template,
        S3_BUCKET=args.bucket,
        S3_LAUNCH_PREFIX=s3_launch_prefix,
        REGION=args.region,
        LAUNCH_ID=launch_id,
        CONFIG_LIST=config_list_str,
    )

    if args.dry_run:
        print("[launch] dry run: skipping S3 + EC2 calls.")
        print("\n--- USER DATA (rendered) ---")
        print(userdata)
        return

    ami = args.ami or lookup_ami(args.region)
    print(f"[launch] AMI:            {ami}")

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "project.tar.gz"
        make_tarball(tar_path)
        print(f"[launch] uploading tarball + {len(args.configs)} configs to {s3_url}/")
        subprocess.run(
            ["aws", "s3", "cp", str(tar_path),
             f"{s3_url}/project.tar.gz", "--region", args.region],
            check=True,
        )
        for cfg in args.configs:
            subprocess.run(
                ["aws", "s3", "cp", str(cfg),
                 f"{s3_url}/configs/{cfg.name}", "--region", args.region],
                check=True,
            )

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                     encoding="utf-8", newline="\n") as f:
        f.write(userdata)
        userdata_path = f.name

    bdm = json.dumps([{
        "DeviceName": "/dev/xvda",
        "Ebs": {
            "VolumeSize": args.root_volume_gb,
            "VolumeType": "gp3",
            "DeleteOnTermination": True,
        },
    }])

    run_cmd = [
        "aws", "ec2", "run-instances",
        "--region", args.region,
        "--image-id", ami,
        "--instance-type", args.instance_type,
        "--instance-initiated-shutdown-behavior", "terminate",
        "--block-device-mappings", bdm,
        "--user-data", f"fileb://{userdata_path}",
        "--iam-instance-profile", f"Name={args.instance_profile}",
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{{Key=Name,Value=thesis-{launch_id}}}]",
        "--count", "1",
    ]
    if args.key_name:
        run_cmd.extend(["--key-name", args.key_name])
    if args.security_group_ids:
        run_cmd.extend(["--security-group-ids", *args.security_group_ids])

    result = aws_json(run_cmd)
    instance_id = result["Instances"][0]["InstanceId"]

    print()
    print(f"[launch] OK -- instance launched: {instance_id}")
    print(f"[launch] tag:               thesis-{launch_id}")
    print(f"[launch] launch artifacts:  {s3_url}/")
    print()
    print("--- follow-up commands ---")
    print(f"# tail bootstrap log:")
    print(f"  aws s3 cp {s3_url}/userdata.log - | tail -80")
    print(f"# pull bundles when finished:")
    print(f"  aws s3 sync s3://{args.bucket}/saved_runs/ ./saved_runs/")
    print(f"# manually terminate:")
    print(f"  aws ec2 terminate-instances --instance-ids {instance_id} "
          f"--region {args.region}")


if __name__ == "__main__":
    main()

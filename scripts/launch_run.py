"""Launch one EC2 instance to run a single solve.

Usage:
    python scripts/launch_run.py configs/<name>.py [options]

What it does:
    1. Tarballs the project (excluding heavy/junk dirs).
    2. Uploads tarball + config to s3://<bucket>/launches/<launch_id>/.
    3. Resolves the latest Amazon Linux 2023 AMI for the region.
    4. Calls `aws ec2 run-instances` with scripts/ec2_userdata.sh as user-data.
    5. Prints the instance id and follow-up commands.

The instance:
    - downloads the project + config from S3
    - installs deps (numpy, scipy, numba)
    - runs scripts/run_solve.py (which uploads the bundle to S3)
    - on success: terminates itself
    - on failure: stays up for SSH debugging

One-time setup required:
    - IAM role named 'thesis-ec2-runner' attached as instance profile,
      with policy AmazonS3FullAccess (or scoped equivalent).
    - (optional) EC2 key pair for SSH access; pass --key-name to use it.
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

# Paths to skip when tarballing the project. Matched against any segment of
# the in-archive name, so this covers nested __pycache__ etc.
TARBALL_EXCLUDES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".claude",
    ".vscode",
    "saved_runs",
    "archive",
    "_t9_min_rport_map.npz",
}

# Approximate eu-north-1 on-demand prices ($/hour) — for printing only.
INSTANCE_HOURLY_USD = {
    "c6i.2xlarge": 0.34,
    "c6i.4xlarge": 0.68,
    "c6i.8xlarge": 1.36,
    "c6i.16xlarge": 2.72,
    "c7i.8xlarge": 1.53,
    "c7i.16xlarge": 3.05,
    "c7i.48xlarge": 9.16,
}

DEFAULT_INSTANCE_TYPE = "c6i.4xlarge"
DEFAULT_REGION = "eu-north-1"
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "hugo-thesis-runs")
DEFAULT_AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64"
DEFAULT_INSTANCE_PROFILE = "thesis-ec2-runner"
DEFAULT_ROOT_VOLUME_GB = 30


def aws_text(cmd: list[str]) -> str:
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stderr.write(
            f"\n[launch] aws CLI failed (exit {cp.returncode}):\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr:  {cp.stderr.strip()}\n"
        )
        sys.exit(cp.returncode)
    return cp.stdout.strip()


def aws_json(cmd: list[str]) -> dict:
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stderr.write(
            f"\n[launch] aws CLI failed (exit {cp.returncode}):\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr:  {cp.stderr.strip()}\n"
        )
        sys.exit(cp.returncode)
    return json.loads(cp.stdout)


def lookup_ami(region: str) -> str:
    return aws_text([
        "aws", "ssm", "get-parameter",
        "--name", DEFAULT_AMI_PARAM,
        "--region", region,
        "--query", "Parameter.Value",
        "--output", "text",
    ])


def make_tarball(out_path: Path) -> None:
    print(f"[launch] building tarball: {out_path}")

    def filter_fn(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        for part in tarinfo.name.split("/"):
            if part in TARBALL_EXCLUDES:
                return None
        return tarinfo

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(PROJECT_ROOT, arcname="thesis", filter=filter_fn)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[launch] tarball size: {size_mb:.1f} MB")


def render_userdata(template: str, **subs: object) -> str:
    out = template
    for key, val in subs.items():
        out = out.replace(f"{{{{ {key} }}}}", str(val))
    return out


def ensure_bucket(bucket: str, region: str) -> None:
    """Create the S3 bucket if it doesn't exist (idempotent)."""
    cp = subprocess.run(
        ["aws", "s3api", "head-bucket", "--bucket", bucket, "--region", region],
        capture_output=True, text=True,
    )
    if cp.returncode == 0:
        return
    print(f"[launch] creating bucket s3://{bucket} in {region}")
    subprocess.run(
        ["aws", "s3api", "create-bucket", "--bucket", bucket, "--region", region,
         "--create-bucket-configuration", f"LocationConstraint={region}"],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", type=Path, help="Path to a config file in configs/.")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--ami", default=None,
                    help="AMI ID; if omitted, latest Amazon Linux 2023 is used.")
    ap.add_argument("--key-name", default=None,
                    help="EC2 key pair name for SSH access (optional).")
    ap.add_argument("--instance-profile", default=DEFAULT_INSTANCE_PROFILE,
                    help="IAM instance profile granting S3 write.")
    ap.add_argument("--security-group-ids", nargs="*", default=None,
                    help="Security group IDs (default: VPC default SG).")
    ap.add_argument("--root-volume-gb", type=int, default=DEFAULT_ROOT_VOLUME_GB)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + render user-data; don't launch.")
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"config not found: {args.config}")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    launch_id = f"{stamp}_{args.config.stem}"
    s3_launch_prefix = f"launches/{launch_id}"
    s3_url = f"s3://{args.bucket}/{s3_launch_prefix}"

    print(f"[launch] launch_id: {launch_id}")
    print(f"[launch] config:    {args.config}")
    print(f"[launch] bucket:    {args.bucket} ({args.region})")
    price = INSTANCE_HOURLY_USD.get(args.instance_type, "?")
    price_str = f"${price}/hr" if isinstance(price, (int, float)) else f"({price})"
    print(f"[launch] type:      {args.instance_type} ~{price_str} on-demand in eu-north-1")

    template_path = PROJECT_ROOT / "scripts" / "ec2_userdata.sh"
    template = template_path.read_text()
    userdata = render_userdata(
        template,
        S3_BUCKET=args.bucket,
        S3_LAUNCH_PREFIX=s3_launch_prefix,
        REGION=args.region,
        LAUNCH_ID=launch_id,
    )

    if args.dry_run:
        print("[launch] dry run: skipping S3 + EC2 calls.")
        # Build (but don't upload) the tarball so we can report its size.
        with tempfile.TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "project.tar.gz"
            make_tarball(tar_path)
        print("\n--- USER DATA (rendered) ---")
        print(userdata)
        return

    ensure_bucket(args.bucket, args.region)

    ami = args.ami or lookup_ami(args.region)
    print(f"[launch] AMI:       {ami}")

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "project.tar.gz"
        make_tarball(tar_path)

        print(f"[launch] uploading tarball + config to {s3_url}/")
        subprocess.run(
            ["aws", "s3", "cp", str(tar_path),
             f"{s3_url}/project.tar.gz", "--region", args.region],
            check=True,
        )
        subprocess.run(
            ["aws", "s3", "cp", str(args.config),
             f"{s3_url}/config.py", "--region", args.region],
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
    print(f"[launch] OK — instance launched: {instance_id}")
    print(f"[launch] tag:               thesis-{launch_id}")
    print(f"[launch] launch artifacts:  {s3_url}/")
    print()
    print("--- follow-up commands ---")
    print(f"# instance state + IP:")
    print(f"  aws ec2 describe-instances --instance-ids {instance_id} "
          f"--region {args.region} "
          f"--query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' "
          f"--output text")
    print()
    print(f"# tail bootstrap log (after a couple minutes):")
    print(f"  aws s3 cp {s3_url}/userdata.log - | tail -50")
    print()
    print(f"# pull bundles when the run finishes:")
    print(f"  aws s3 sync s3://{args.bucket}/saved_runs/ ./saved_runs/")
    print()
    print(f"# manually terminate (if it's stuck in error state):")
    print(f"  aws ec2 terminate-instances --instance-ids {instance_id} "
          f"--region {args.region}")


if __name__ == "__main__":
    main()

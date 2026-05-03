"""Run a sweep with bounded parallelism. Maintains a queue of configs and
dispatches each to its own EC2 instance such that at most --max-parallel
are running at once. As each instance terminates, the next config in the
queue is launched automatically.

Usage:
    python scripts/launch_queue.py configs/sweep_xxx/ \\
        --max-parallel 2 --instance-type c6i.4xlarge --key-name hugo-thesis

Lifecycle:
    - Keep this script running until it prints "[queue] all done".
    - Close the terminal / Ctrl+C to stop dispatching NEW launches; instances
      already running on EC2 continue independently and finish on their own.
    - If an instance runs >--max-runtime-min minutes it's auto-terminated
      (likely stuck) and its slot is freed for the next config.

When done, pull bundles with:
    aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REGION = "eu-north-1"
DEFAULT_POLL_SEC = 60
DEFAULT_MAX_RUNTIME_MIN = 240   # 4 hours; long-tail safety net

INSTANCE_ID_RE = re.compile(r"instance launched:\s+(i-[0-9a-f]+)")
LAUNCH_ID_RE = re.compile(r"launches/([0-9TZ_a-zA-Z-]+)/")


@dataclass
class ActiveLaunch:
    config: Path
    instance_id: str
    launch_id: str
    started_at: datetime.datetime


def discover_configs(sweep_dir: Path) -> list[Path]:
    if not sweep_dir.is_dir():
        sys.exit(f"sweep dir not found or not a directory: {sweep_dir}")
    configs = sorted(
        p for p in sweep_dir.glob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("_")
    )
    if not configs:
        sys.exit(f"no configs found in {sweep_dir}")
    return configs


def launch_one(config: Path, base_args: list[str]) -> tuple[str | None, str | None, str]:
    """Run launch_run.py for one config; return (instance_id, launch_id, output)."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "launch_run.py"),
        str(config), *base_args,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    out = (cp.stdout + cp.stderr).strip()
    if cp.returncode != 0:
        return None, None, out
    iid_m = INSTANCE_ID_RE.search(out)
    lid_m = LAUNCH_ID_RE.search(out)
    return (
        iid_m.group(1) if iid_m else None,
        lid_m.group(1) if lid_m else None,
        out,
    )


def describe_instance_states(
    instance_ids: list[str], region: str
) -> dict[str, str]:
    if not instance_ids:
        return {}
    cmd = [
        "aws", "ec2", "describe-instances",
        "--instance-ids", *instance_ids,
        "--region", region,
        "--query", "Reservations[].Instances[].[InstanceId,State.Name]",
        "--output", "json",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        sys.stderr.write(f"[queue] warn: describe-instances failed: "
                         f"{cp.stderr.strip()}\n")
        return {}
    rows = json.loads(cp.stdout)
    return {row[0]: row[1] for row in rows}


def terminate_instance(instance_id: str, region: str) -> None:
    cmd = ["aws", "ec2", "terminate-instances",
           "--instance-ids", instance_id, "--region", region]
    subprocess.run(cmd, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--instance-type", default="c6i.4xlarge")
    ap.add_argument("--key-name", default=None)
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_SEC,
                    help=f"Seconds between state polls (default {DEFAULT_POLL_SEC})")
    ap.add_argument("--max-runtime-min", type=int, default=DEFAULT_MAX_RUNTIME_MIN,
                    help=f"Auto-terminate after this many minutes "
                         f"(default {DEFAULT_MAX_RUNTIME_MIN}). Frees a queue slot.")
    args = ap.parse_args()

    configs = discover_configs(args.sweep_dir)
    queue: deque[Path] = deque(configs)
    total = len(configs)

    base_args = ["--instance-type", args.instance_type]
    if args.key_name: base_args += ["--key-name", args.key_name]
    if args.bucket:   base_args += ["--bucket", args.bucket]
    if args.region:   base_args += ["--region", args.region]

    print(f"[queue] {total} configs in {args.sweep_dir}")
    print(f"[queue] max parallel: {args.max_parallel}")
    print(f"[queue] instance: {args.instance_type}")
    print(f"[queue] poll: {args.poll_interval}s | max-runtime: "
          f"{args.max_runtime_min} min (auto-terminate)")
    print()

    active: dict[str, ActiveLaunch] = {}
    completed: list[ActiveLaunch] = []
    timed_out: list[ActiveLaunch] = []
    failed_launch: list[Path] = []

    while queue or active:
        # 1. Dispatch as many as we can
        while queue and len(active) < args.max_parallel:
            cfg = queue.popleft()
            launched_idx = total - len(queue) - len(active)
            print(f"[queue] launching {cfg.name} ({launched_idx + 1}/{total})...")
            iid, lid, out = launch_one(cfg, base_args)
            if iid is None or lid is None:
                print(f"[queue] LAUNCH FAILED for {cfg.name}:")
                print(out)
                failed_launch.append(cfg)
                continue
            active[iid] = ActiveLaunch(
                config=cfg,
                instance_id=iid,
                launch_id=lid,
                started_at=datetime.datetime.now(datetime.timezone.utc),
            )
            print(f"[queue]   -> {iid}, launch {lid}")

        # 2. Status line
        print(f"[queue] active: {len(active)}/{args.max_parallel} | "
              f"queued: {len(queue)} | done: {len(completed)}/{total}"
              + (f" | timed-out: {len(timed_out)}" if timed_out else "")
              + (f" | launch-fail: {len(failed_launch)}" if failed_launch else ""))

        if not active:
            break

        # 3. Wait, then poll states
        time.sleep(args.poll_interval)
        states = describe_instance_states(list(active.keys()), args.region)
        now = datetime.datetime.now(datetime.timezone.utc)

        for iid in list(active.keys()):
            launch = active[iid]
            state = states.get(iid, "unknown")
            age_min = (now - launch.started_at).total_seconds() / 60

            # Done if state has progressed past running/pending
            if state not in ("running", "pending"):
                active.pop(iid)
                completed.append(launch)
                print(f"[queue] DONE: {launch.config.name} "
                      f"(state={state}, runtime={age_min:.1f} min)")
                continue

            # Time-out check on still-running instances
            if state == "running" and age_min > args.max_runtime_min:
                print(f"[queue] TIMEOUT: {launch.config.name} ran "
                      f"{age_min:.1f} min (>{args.max_runtime_min}); "
                      f"terminating {iid}")
                terminate_instance(iid, args.region)
                active.pop(iid)
                timed_out.append(launch)

    print()
    print(f"[queue] ===== sweep finished =====")
    print(f"[queue] completed:     {len(completed)}/{total}")
    if failed_launch:
        print(f"[queue] launch failed: {len(failed_launch)}")
        for f in failed_launch:
            print(f"          - {f.name}")
    if timed_out:
        print(f"[queue] timed out:     {len(timed_out)}")
        for t in timed_out:
            print(f"          - {t.config.name} ({t.instance_id}, {t.launch_id})")
    print()
    print("Pull all bundles:")
    print(f"  aws s3 sync s3://hugo-thesis-runs/saved_runs/ ./saved_runs/")


if __name__ == "__main__":
    main()

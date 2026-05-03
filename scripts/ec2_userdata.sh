#!/bin/bash
# scripts/ec2_userdata.sh — runs once as root on EC2 instance startup.
#
# This file is a TEMPLATE — scripts/launch_run.py substitutes:
#   {{ S3_BUCKET }} {{ S3_LAUNCH_PREFIX }} {{ REGION }} {{ LAUNCH_ID }}
#
# Flow:
#   1. install deps
#   2. download tarball + config from S3
#   3. pip install -r requirements.txt
#   4. run scripts/run_solve.py (which uploads bundle to s3://bucket/saved_runs/)
#   5. on success: shutdown -h now (instance terminates)
#   on failure: leaves instance up for SSH debugging; uploads log
#
# Crash safety:
#   - A 24-hour self-destruct timer is armed on first boot. If the solve hasn't
#     completed by then (longest config is ~20h), the instance shuts itself
#     down regardless. This keeps a single stuck instance from running away.
#   - A checkpoint-uploader sidecar syncs ./saved_runs/checkpoints/ to
#     s3://${S3_BUCKET}/checkpoints/ every 60 seconds, so per-age progress is
#     durable in S3 within 60s of being written. run_solve.py picks up an
#     existing S3 checkpoint at startup and resumes.

set -euo pipefail

# Mirror everything to /var/log/thesis-userdata.log
exec > >(tee -a /var/log/thesis-userdata.log) 2>&1

S3_BUCKET="{{ S3_BUCKET }}"
S3_LAUNCH_PREFIX="{{ S3_LAUNCH_PREFIX }}"
REGION="{{ REGION }}"
LAUNCH_ID="{{ LAUNCH_ID }}"

# Self-destruct after 24 hours regardless of solve state. Longest config is
# ~20h; this is a hard safety bound on per-instance cost. Detached so the rest
# of this script can proceed.
MAX_RUNTIME_SEC=86400
( sleep "${MAX_RUNTIME_SEC}" \
  && echo "!!! self-destruct timer fired after ${MAX_RUNTIME_SEC}s; shutting down" \
  && shutdown -h now ) &
SELF_DESTRUCT_PID=$!

upload_log() {
    aws s3 cp /var/log/thesis-userdata.log \
        "s3://${S3_BUCKET}/${S3_LAUNCH_PREFIX}/userdata.log" \
        --region "${REGION}" 2>/dev/null || true
}

upload_checkpoints() {
    # Sync any per-age checkpoint bundles to S3. Path is relative to the
    # workdir set by the solve step below; if the dir doesn't exist yet
    # (early in the run) the sync is a no-op.
    if [ -d /opt/thesis/thesis/saved_runs/checkpoints ]; then
        aws s3 sync /opt/thesis/thesis/saved_runs/checkpoints/ \
            "s3://${S3_BUCKET}/checkpoints/" \
            --region "${REGION}" 2>/dev/null || true
    fi
}

# Periodically upload the log (every 30s) and the checkpoints (every 60s)
# while the script runs.
(while true; do sleep 30; upload_log; done) &
PERIODIC_LOG_PID=$!
(while true; do sleep 60; upload_checkpoints; done) &
PERIODIC_CKPT_PID=$!

cleanup() {
    kill "${PERIODIC_LOG_PID}" 2>/dev/null || true
    kill "${PERIODIC_CKPT_PID}" 2>/dev/null || true
    kill "${SELF_DESTRUCT_PID}" 2>/dev/null || true
    upload_checkpoints
    upload_log
}

on_error() {
    echo "!!! ERROR at line $1 -- instance will NOT terminate; SSH in to debug."
    cleanup
    exit 1
}

trap cleanup EXIT
trap 'on_error $LINENO' ERR

echo "=========================================="
echo "thesis solve runner: ${LAUNCH_ID}"
date -u
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id || echo unknown)
INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type || echo unknown)
echo "instance: ${INSTANCE_ID} (${INSTANCE_TYPE})"
echo "self-destruct timer: ${MAX_RUNTIME_SEC}s (~24h)"
echo "=========================================="

echo "--- [1/5] installing system packages ---"
dnf install -y python3.11 python3.11-pip git tar gzip

echo "--- [2/5] downloading project + config ---"
mkdir -p /opt/thesis
cd /opt/thesis
aws s3 cp "s3://${S3_BUCKET}/${S3_LAUNCH_PREFIX}/project.tar.gz" project.tar.gz \
    --region "${REGION}"
tar xzf project.tar.gz
cd thesis
aws s3 cp "s3://${S3_BUCKET}/${S3_LAUNCH_PREFIX}/config.py" run_config.py \
    --region "${REGION}"

echo "--- [3/5] pip install ---"
python3.11 -m pip install -r requirements.txt

echo "--- [4/5] running solve ---"
export AWS_DEFAULT_REGION="${REGION}"
export S3_BUCKET="${S3_BUCKET}"
python3.11 scripts/run_solve.py run_config.py --bucket "${S3_BUCKET}"

echo "--- [5/5] solve complete; shutting down ---"
date -u
upload_checkpoints
upload_log
shutdown -h now

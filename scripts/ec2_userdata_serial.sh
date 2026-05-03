#!/bin/bash
# scripts/ec2_userdata_serial.sh — serial-config bootstrap for one EC2 instance.
#
# Substitutions: {{ S3_BUCKET }} {{ S3_LAUNCH_PREFIX }} {{ REGION }}
#                {{ LAUNCH_ID }} {{ CONFIG_LIST }}
#
# CONFIG_LIST is a space-separated list of config filenames present under
# s3://${S3_BUCKET}/${S3_LAUNCH_PREFIX}/configs/. Each is downloaded and
# solved in order with run_solve.py (which auto-resumes from S3 checkpoints).
#
# Difference from ec2_userdata.sh:
#   - No `set -e`: one config's failure must NOT kill the remaining configs.
#   - Loops through configs; logs per-config success/failure.
#   - Self-destructs after 24h or after final config completes.

set -uo pipefail
exec > >(tee -a /var/log/thesis-userdata.log) 2>&1

S3_BUCKET="{{ S3_BUCKET }}"
S3_LAUNCH_PREFIX="{{ S3_LAUNCH_PREFIX }}"
REGION="{{ REGION }}"
LAUNCH_ID="{{ LAUNCH_ID }}"
CONFIGS="{{ CONFIG_LIST }}"

# 24h hard cap on instance runtime (longest single line ~13h; gives headroom).
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
    if [ -d /opt/thesis/thesis/saved_runs/checkpoints ]; then
        aws s3 sync /opt/thesis/thesis/saved_runs/checkpoints/ \
            "s3://${S3_BUCKET}/checkpoints/" \
            --region "${REGION}" 2>/dev/null || true
    fi
}

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
trap cleanup EXIT

echo "=========================================="
echo "thesis SERIAL solve runner: ${LAUNCH_ID}"
date -u
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id || echo unknown)
INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type || echo unknown)
echo "instance: ${INSTANCE_ID} (${INSTANCE_TYPE})"
echo "self-destruct timer: ${MAX_RUNTIME_SEC}s (~24h)"
echo "configs to run sequentially: ${CONFIGS}"
echo "=========================================="

echo "--- [boot/1] installing system packages ---"
dnf install -y python3.11 python3.11-pip git tar gzip

echo "--- [boot/2] downloading project ---"
mkdir -p /opt/thesis
cd /opt/thesis
aws s3 cp "s3://${S3_BUCKET}/${S3_LAUNCH_PREFIX}/project.tar.gz" project.tar.gz \
    --region "${REGION}"
tar xzf project.tar.gz
cd thesis

echo "--- [boot/3] pip install ---"
python3.11 -m pip install -r requirements.txt

export AWS_DEFAULT_REGION="${REGION}"
export S3_BUCKET="${S3_BUCKET}"

SUCCESSES=()
FAILURES=()
IDX=0
N_CONFIGS=$(echo "${CONFIGS}" | wc -w)

for cfg in ${CONFIGS}; do
    IDX=$((IDX + 1))
    echo ""
    echo "=================================================================="
    echo "[${IDX}/${N_CONFIGS}] STARTING config: ${cfg}"
    date -u
    echo "=================================================================="
    aws s3 cp "s3://${S3_BUCKET}/${S3_LAUNCH_PREFIX}/configs/${cfg}" \
        "run_${cfg}" --region "${REGION}"
    set +e
    python3.11 scripts/run_solve.py "run_${cfg}" --bucket "${S3_BUCKET}"
    rc=$?
    set -e
    if [ ${rc} -eq 0 ]; then
        echo ""
        echo "[${IDX}/${N_CONFIGS}] STATUS: ${cfg} OK"
        SUCCESSES+=("${cfg}")
    else
        echo ""
        echo "!!! [${IDX}/${N_CONFIGS}] STATUS: ${cfg} FAILED (exit ${rc}) -- continuing to next config"
        FAILURES+=("${cfg}")
    fi
    upload_checkpoints
    upload_log
done

echo ""
echo "=================================================================="
echo "SERIAL RUN COMPLETE"
echo "successes (${#SUCCESSES[@]}): ${SUCCESSES[*]:-}"
echo "failures  (${#FAILURES[@]}): ${FAILURES[*]:-}"
date -u
echo "=================================================================="

upload_checkpoints
upload_log
shutdown -h now

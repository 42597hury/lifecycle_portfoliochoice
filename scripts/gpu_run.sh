#!/bin/bash
# scripts/gpu_run.sh — End-to-end GPU benchmark wrapper.
#
# Wraps a lifecycle-solver run with cache-aware S3 plumbing:
#   1. Pull the JAX persistent compilation cache from S3 (idempotent — skip
#      if local cache already non-empty).
#   2. Run the solver script (default: verify/benchmark_bundle.py). The solver
#      itself uploads the policy bundle to S3 if S3_BUCKET is set.
#   3. Push the (now-populated) JAX cache back to S3 on success.
#   4. Archive the run log to S3.
#
# The cache is keyed by hardware tier so CPU and GPU runs (and different GPU
# generations) don't pollute each other's caches. The tier is auto-detected
# from `nvidia-smi` output unless overridden via CACHE_TIER env var.
#
# Required env vars:
#   S3_BUCKET                  S3 bucket for cache, bundle, log
#                              (also picked up by verify/benchmark_bundle.py
#                              for its own bundle upload)
#
# Optional env vars:
#   AWS_REGION                 default: eu-north-1
#   LIFECYCLE_JAX_CACHE_DIR    default: ~/.cache/jax_lifecycle
#   CACHE_TIER                 default: auto-detected (gpu-h100-sxm5,
#                              gpu-h100-pcie, gpu-a100, gpu-v100, gpu-other,
#                              cpu). Override to force a specific prefix.
#   RUN_SCRIPT                 default: verify/benchmark_bundle.py
#   CACHE_VERSION              default: v1. Bump to invalidate stale cache
#                              after a JAX upgrade.
#
# Usage:
#   export S3_BUCKET=hugo-thesis-runs
#   bash scripts/gpu_run.sh
#
# Exit code: forwarded from the solver script (non-zero -> still archives
# the log, but skips the cache push).

set -euo pipefail

command -v aws >/dev/null 2>&1 || {
    echo "[gpu_run] FATAL: aws CLI not found. Install with 'sudo apt-get install -y awscli' (Ubuntu) or 'sudo dnf install -y aws-cli' (AL2023)."
    exit 2
}

# --- Required ---
: "${S3_BUCKET:?S3_BUCKET must be set (e.g. hugo-thesis-runs)}"

# --- Defaults ---
AWS_REGION="${AWS_REGION:-eu-north-1}"
LIFECYCLE_JAX_CACHE_DIR="${LIFECYCLE_JAX_CACHE_DIR:-$HOME/.cache/jax_lifecycle}"
RUN_SCRIPT="${RUN_SCRIPT:-verify/benchmark_bundle.py}"
CACHE_VERSION="${CACHE_VERSION:-v1}"

# --- Auto-detect hardware tier (if not overridden) ---
detect_tier() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cpu"
        return
    fi
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]' || true)
    if [ -z "$name" ]; then
        echo "cpu"
    elif echo "$name" | grep -q "h100" && echo "$name" | grep -q "sxm"; then
        echo "gpu-h100-sxm5"
    elif echo "$name" | grep -q "h100"; then
        echo "gpu-h100-pcie"
    elif echo "$name" | grep -q "b200"; then
        echo "gpu-b200"
    elif echo "$name" | grep -q "a100"; then
        echo "gpu-a100"
    elif echo "$name" | grep -q "v100"; then
        echo "gpu-v100"
    elif echo "$name" | grep -q "a10\b"; then
        echo "gpu-a10"
    else
        echo "gpu-other"
    fi
}

CACHE_TIER="${CACHE_TIER:-$(detect_tier)}"
CACHE_PREFIX="jax-cache/${CACHE_VERSION}/${CACHE_TIER}"
LOG_PATH="/tmp/jax-gpu-run.log"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_S3_KEY="run-logs/${TIMESTAMP}_${CACHE_TIER}.log"

echo "================================================================="
echo "GPU run wrapper"
echo "  S3_BUCKET:    $S3_BUCKET"
echo "  AWS_REGION:   $AWS_REGION"
echo "  Cache tier:   $CACHE_TIER (s3://$S3_BUCKET/$CACHE_PREFIX/)"
echo "  Cache dir:    $LIFECYCLE_JAX_CACHE_DIR"
echo "  Run script:   $RUN_SCRIPT"
echo "  Timestamp:    $TIMESTAMP"
echo "================================================================="

# --- Phase 4: Pull cache (idempotent — skip if local already populated) ---
mkdir -p "$LIFECYCLE_JAX_CACHE_DIR"
if [ -z "$(ls -A "$LIFECYCLE_JAX_CACHE_DIR" 2>/dev/null)" ]; then
    echo "[gpu_run] pulling JAX cache from s3://$S3_BUCKET/$CACHE_PREFIX/ ..."
    if ! aws s3 sync "s3://$S3_BUCKET/$CACHE_PREFIX/" "$LIFECYCLE_JAX_CACHE_DIR" \
        --region "$AWS_REGION" 2>&1; then
        echo "[gpu_run] cache pull failed (continuing without cache)"
    fi
else
    echo "[gpu_run] local cache at $LIFECYCLE_JAX_CACHE_DIR is non-empty; skipping pull"
fi

# --- Phase 5: Run solver. The solver picks up S3_BUCKET and uploads the
#    bundle itself; we don't double-handle that. ---
export LIFECYCLE_JAX_CACHE_DIR
export S3_BUCKET
export AWS_REGION

echo "[gpu_run] running $RUN_SCRIPT (output also -> $LOG_PATH) ..."
RC=0
python "$RUN_SCRIPT" 2>&1 | tee "$LOG_PATH" || RC=${PIPESTATUS[0]}

echo "[gpu_run] $RUN_SCRIPT exited with rc=$RC"

# --- Phase 6: Push cache back (only on success — bad runs may have written
#    poisoned entries) ---
if [ "$RC" -eq 0 ]; then
    echo "[gpu_run] pushing JAX cache to s3://$S3_BUCKET/$CACHE_PREFIX/ ..."
    aws s3 sync "$LIFECYCLE_JAX_CACHE_DIR" "s3://$S3_BUCKET/$CACHE_PREFIX/" \
        --size-only --region "$AWS_REGION" || echo "[gpu_run] cache push failed (non-fatal)"
else
    echo "[gpu_run] run failed; skipping cache push"
fi

# --- Phase 7: Archive log (always — failure logs are the most valuable) ---
echo "[gpu_run] archiving log to s3://$S3_BUCKET/$LOG_S3_KEY ..."
aws s3 cp "$LOG_PATH" "s3://$S3_BUCKET/$LOG_S3_KEY" \
    --region "$AWS_REGION" || echo "[gpu_run] log push failed (non-fatal)"

echo "================================================================="
if [ "$RC" -eq 0 ]; then
    echo "[gpu_run] SUCCESS"
    echo "  Bundle:  s3://$S3_BUCKET/saved_runs/<bundle-name>/  (uploaded by $RUN_SCRIPT)"
    echo "  Cache:   s3://$S3_BUCKET/$CACHE_PREFIX/"
    echo "  Log:     s3://$S3_BUCKET/$LOG_S3_KEY"
    echo ""
    echo "  Pull bundle locally:"
    echo "    bash scripts/download_bundle.sh <bundle-name>"
else
    echo "[gpu_run] FAILED (rc=$RC) — see log at s3://$S3_BUCKET/$LOG_S3_KEY"
fi
echo "================================================================="

exit "$RC"

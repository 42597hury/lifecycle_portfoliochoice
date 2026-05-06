#!/bin/bash
# scripts/download_bundle.sh — Pull a policy bundle from S3 to the local laptop.
#
# Usage:
#   bash scripts/download_bundle.sh <bundle-name>
#
#   e.g. bash scripts/download_bundle.sh \
#       system_iv_full_var_unconstrained_cholesky_grid9x9x9_nz11_jax_benchmark
#
# Optional env vars:
#   S3_BUCKET     default: hugo-thesis-runs
#   AWS_REGION    default: eu-north-1
#   LOCAL_DIR     default: ./saved_runs
#
# Companion to scripts/gpu_run.sh on the GPU side.

set -euo pipefail

BUNDLE_NAME="${1:?Usage: $0 <bundle-name>}"
S3_BUCKET="${S3_BUCKET:-hugo-thesis-runs}"
AWS_REGION="${AWS_REGION:-eu-north-1}"
LOCAL_DIR="${LOCAL_DIR:-./saved_runs}"

S3_URI="s3://${S3_BUCKET}/saved_runs/${BUNDLE_NAME}/"
LOCAL_TARGET="${LOCAL_DIR}/${BUNDLE_NAME}/"

mkdir -p "$LOCAL_TARGET"

echo "Pulling $S3_URI -> $LOCAL_TARGET"
aws s3 sync "$S3_URI" "$LOCAL_TARGET" --region "$AWS_REGION"

echo ""
echo "Done. Contents of $LOCAL_TARGET:"
ls -lh "$LOCAL_TARGET"

cat <<EOF

Load in Python:

    from lifecycle.policy_io import load_policy_bundle
    C, S, B, diag, run_config = load_policy_bundle("${LOCAL_TARGET%/}")

EOF

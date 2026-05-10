#!/bin/bash
# Overnight lifecycle chain — Shi-10 theta=0 then theta=1, ~22-24 hr total
set -uo pipefail
cd /home/ubuntu/thesisscripts_JAX
mkdir -p logs

run() {
  local name="$1"; shift
  echo ""
  echo "========================================================="
  echo "===== START $name $(date -u +'%Y-%m-%dT%H:%M:%SZ') ====="
  echo "========================================================="
  python3 "$@" 2>&1 | tee logs/overnight_${name}.log | tail -40
  local rc=${PIPESTATUS[0]}
  echo ""
  echo "===== END $name (exit=$rc) $(date -u +'%Y-%m-%dT%H:%M:%SZ') ====="
  echo ""
  return 0  # always continue (don't abort theta=1 if theta=0 errored after partial bundle)
}

run "theta0" verify/overnight_lifecycle_shi10_theta0.py
run "theta1" verify/overnight_lifecycle_shi10_theta1.py

echo "===== OVERNIGHT CHAIN COMPLETE $(date -u +'%Y-%m-%dT%H:%M:%SZ') ====="
touch logs/OVERNIGHT_CHAIN_COMPLETE

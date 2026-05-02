#!/usr/bin/env bash
# Run the gridpoint Euler diagnostic against the bequest-patch smoke bundle
# in both eval modes. Markdown reports are written next to the project root.
set -euo pipefail

BUNDLE="saved_runs/checkpoints/system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_from_age87_kret3x7x5_ns2p0x2p25x2p25_log1p_pathB_v1"

if [ ! -d "$BUNDLE" ]; then
  echo "ERROR: bundle not found: $BUNDLE" >&2
  exit 1
fi

echo "[diag] same mode"
python -m scripts.diagnostics._diag_gridpoint_ee \
  --model-bundle "$BUNDLE" \
  "$BUNDLE" \
  --eval-mode same \
  --markdown-out diagnostics_gridpoint_ee_log1p_pathB_same.md

echo "[diag] next_finer mode"
python -m scripts.diagnostics._diag_gridpoint_ee \
  --model-bundle "$BUNDLE" \
  "$BUNDLE" \
  --eval-mode next_finer \
  --markdown-out diagnostics_gridpoint_ee_log1p_pathB_nextfiner.md

echo "[diag] done"

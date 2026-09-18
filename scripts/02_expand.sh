#!/usr/bin/env bash
# Stage 2: expand selected layers into models/expanded/{MODEL_ID}-adept-k4
# Usage:
#   export EXPAND_LAYERS=22,23,24,25   # optional override
#   bash ADEPT/scripts/02_expand.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -z "${EXPAND_LAYERS}" ]]; then
  if [[ ! -f "${IMP_OUT}/suggested_expand_layers.txt" ]]; then
    adept_log "ERROR: no EXPAND_LAYERS and missing ${IMP_OUT}/suggested_expand_layers.txt"
    exit 1
  fi
  EXPAND_LAYERS="$(tr -d '[:space:]' < "${IMP_OUT}/suggested_expand_layers.txt")"
fi

NUM_EXPAND="$(echo "${EXPAND_LAYERS}" | tr ',' '\n' | grep -c '[0-9]' || true)"
adept_log "=== 02_expand layers=${EXPAND_LAYERS} (k=${NUM_EXPAND}) -> ${EXP_MODEL} ==="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
adept_run "02_expand" \
  "${LLAMAFACTORY_PYTHON}" "${REPO_ROOT}/ADEPT/expand.py" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${EXP_MODEL}" \
    --expand_layers "${EXPAND_LAYERS}"

# Persist effective expand layers for train configs
echo "${EXPAND_LAYERS}" > "${IMP_OUT}/effective_expand_layers.txt"
adept_log "Wrote ${IMP_OUT}/effective_expand_layers.txt"

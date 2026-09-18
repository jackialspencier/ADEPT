#!/usr/bin/env bash
# Stage 1: ADEPT layer importance (top_k_expand default 4).
# Usage:
#   export REPO_ROOT=/path/to/LlamaFactory_baseline
#   bash ADEPT/scripts/01_calc_importance.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

adept_log "=== 01_calc_importance MODEL=${MODEL} OUT=${IMP_OUT} k=${TOP_K_EXPAND} ==="
if [[ ! -f "${ADEPT_EVAL_DATA_PATH}" ]]; then
  adept_log "ERROR: missing ADEPT_EVAL_DATA_PATH=${ADEPT_EVAL_DATA_PATH}"
  exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
  adept_log "ERROR: missing base model ${MODEL}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
adept_run "01_calc_importance" \
  "${LLAMAFACTORY_PYTHON}" "${REPO_ROOT}/ADEPT/calc_importance.py" \
    --model_name_or_path "${MODEL}" \
    --data_path "${ADEPT_EVAL_DATA_PATH}" \
    --output_dir "${IMP_OUT}" \
    --batch_size "${IMPORTANCE_BATCH_SIZE}" \
    --max_length 2048 \
    --top_k_expand "${TOP_K_EXPAND}"

if [[ -f "${IMP_OUT}/suggested_expand_layers.txt" ]]; then
  adept_log "suggested_expand_layers=$(tr -d '\n' < "${IMP_OUT}/suggested_expand_layers.txt")"
fi

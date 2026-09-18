#!/usr/bin/env bash
# Stage 3: train ADEPT on one domain (freeze expanded blocks + dynamic LR callback).
# Usage:
#   bash ADEPT/scripts/03_train_domain.sh billsum
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

DOMAIN="${1:?domain required (billsum|sql_create_context|fin_instruct|pubmedqa|dialogsum)}"
DATASET_NAME="${DOMAIN}_train"
# Persist under MODELS_ROOT (OBS-mapped on ModelArts via OUTPUT_DIRE / OUTPUT_MODELS)
TRAIN_OUT="${TRAINED_ROOT}/${MODEL_ID}/adept/${DOMAIN}"
CFG_SRC="${CFG_DIR}/gemma2_adept_${DOMAIN}.yaml"

if [[ ! -d "${EXP_MODEL}" ]]; then
  adept_log "ERROR: expanded model missing: ${EXP_MODEL} (run 02_expand first)"
  exit 1
fi
if [[ ! -f "${CFG_SRC}" ]]; then
  adept_log "ERROR: missing yaml ${CFG_SRC}"
  exit 1
fi
if [[ ! -f "${ADEPT_EVAL_DATA_PATH}" ]]; then
  adept_log "ERROR: missing ADEPT_EVAL_DATA_PATH=${ADEPT_EVAL_DATA_PATH}"
  exit 1
fi

# Resolve expand count for freeze_trainable_layers
if [[ -z "${EXPAND_LAYERS:-}" && -f "${IMP_OUT}/effective_expand_layers.txt" ]]; then
  EXPAND_LAYERS="$(tr -d '[:space:]' < "${IMP_OUT}/effective_expand_layers.txt")"
elif [[ -z "${EXPAND_LAYERS:-}" && -f "${IMP_OUT}/suggested_expand_layers.txt" ]]; then
  EXPAND_LAYERS="$(tr -d '[:space:]' < "${IMP_OUT}/suggested_expand_layers.txt")"
fi
NUM_EXPAND="${TOP_K_EXPAND}"
if [[ -n "${EXPAND_LAYERS:-}" ]]; then
  NUM_EXPAND="$(echo "${EXPAND_LAYERS}" | tr ',' '\n' | grep -c '[0-9]' || true)"
fi

mkdir -p "${TRAIN_OUT}" "$(dirname "${TRAIN_OUT}")"
CFG_RUNTIME="${LOG_DIR}/runtime_gemma2_adept_${DOMAIN}.yaml"

# Rewrite paths to absolute REPO_ROOT for Huawei Cloud portability
"${LLAMAFACTORY_PYTHON}" - <<PY
from pathlib import Path
src = Path("${CFG_SRC}")
dst = Path("${CFG_RUNTIME}")
text = src.read_text(encoding="utf-8")
replacements = {
    "model_name_or_path:": "model_name_or_path: ${EXP_MODEL}",
    "dataset_dir:": "dataset_dir: ${DATA_DIR}",
    "dataset:": "dataset: ${DATASET_NAME}",
    "output_dir:": "output_dir: ${TRAIN_OUT}",
    "freeze_trainable_layers:": "freeze_trainable_layers: ${NUM_EXPAND}",
}
out_lines = []
for line in text.splitlines():
    stripped = line.lstrip()
    key = None
    for k in replacements:
        if stripped.startswith(k):
            key = k
            break
    if key is None:
        out_lines.append(line)
    else:
        indent = line[: len(line) - len(stripped)]
        out_lines.append(indent + replacements[key])
dst.write_text("\\n".join(out_lines) + "\\n", encoding="utf-8")
print(f"Wrote {dst}")
PY

adept_log "=== 03_train domain=${DOMAIN} dataset=${DATASET_NAME} out=${TRAIN_OUT} freeze=${NUM_EXPAND} ==="
adept_log "ADEPT_EVAL_DATA_PATH=${ADEPT_EVAL_DATA_PATH}"
adept_log "ADEPT_IMPORTANCE_EVAL_STEPS=${ADEPT_IMPORTANCE_EVAL_STEPS}"

cd "${ADEPT_LF}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TRAIN}"
export REPO_ROOT ADEPT_EVAL_DATA_PATH ADEPT_IMPORTANCE_EVAL_STEPS
export PYTHONPATH=src${PYTHONPATH:+:${PYTHONPATH}}

adept_run "03_train_${DOMAIN}" \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TRAIN}" \
      REPO_ROOT="${REPO_ROOT}" \
      ADEPT_EVAL_DATA_PATH="${ADEPT_EVAL_DATA_PATH}" \
      ADEPT_IMPORTANCE_EVAL_STEPS="${ADEPT_IMPORTANCE_EVAL_STEPS}" \
      PYTHONPATH=src \
  "${LLAMAFACTORY_PYTHON}" src/train.py "${CFG_RUNTIME}"

adept_log "Train finished: ${TRAIN_OUT}"

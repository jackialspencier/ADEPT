#!/usr/bin/env bash
# Stage 4: 12-metric eval for one ADEPT domain checkpoint + aggregate.
# Domain eval + HumanEval share DOMAIN_GPU; public lm-eval uses BENCHMARK_GPU.
# When DOMAIN_GPU == BENCHMARK_GPU, evaluation runs serially to avoid OOM.
#
# Usage:
#   export DOMAIN_GPU=0 BENCHMARK_GPU=1   # 2-GPU (recommended)
#   # or DOMAIN_GPU=0 BENCHMARK_GPU=0     # 1-GPU serial
#   bash ADEPT/scripts/04_eval_domain.sh billsum
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

DOMAIN="${1:?domain required}"
# Checkpoints / metrics under OBS-mapped roots (see shared/scripts/cloud_paths.sh)
TRAIN_OUT="${TRAINED_ROOT}/${MODEL_ID}/adept/${DOMAIN}"
EVAL_OUT="${RESULTS_ROOT}/${MODEL_ID}/adept/${DOMAIN}/eval"
export METHOD_NAME=adept
export HUMANEVAL_GPU="${HUMANEVAL_GPU:-${DOMAIN_GPU}}"
export LOG_DIR
export LOG_PREFIX="eval_${DOMAIN}"
export LM_EVAL_BATCH_SIZE

if [[ ! -d "${TRAIN_OUT}" ]]; then
  adept_log "ERROR: checkpoint missing: ${TRAIN_OUT}"
  exit 1
fi
mkdir -p "${EVAL_OUT}"

adept_log "=== 04_eval domain=${DOMAIN} ckpt=${TRAIN_OUT} ==="
adept_log "RESULTS_ROOT=${RESULTS_ROOT} MODELS_ROOT=${MODELS_ROOT}"
adept_log "DOMAIN_GPU=${DOMAIN_GPU} (domain+HumanEval) BENCHMARK_GPU=${BENCHMARK_GPU} (public)"

cd "${LF_DIR}"
export DOMAIN_GPU BENCHMARK_GPU HUMANEVAL_GPU METHOD_NAME MODEL_ID REPO_ROOT LLAMAFACTORY_PYTHON
export RESULTS_ROOT MODELS_ROOT BASES_ROOT TRAINED_ROOT

# Force serial schedule when same physical GPU is used for both streams
if [[ "${DOMAIN_GPU}" == "${BENCHMARK_GPU}" ]]; then
  export ADEPT_EVAL_SERIAL=1
  adept_log "Same GPU for domain and public -> serial eval (ADEPT_EVAL_SERIAL=1)"
fi

adept_run "04_eval_${DOMAIN}" \
  env DOMAIN_GPU="${DOMAIN_GPU}" \
      BENCHMARK_GPU="${BENCHMARK_GPU}" \
      HUMANEVAL_GPU="${HUMANEVAL_GPU}" \
      METHOD_NAME=adept \
      MODEL_ID="${MODEL_ID}" \
      REPO_ROOT="${REPO_ROOT}" \
      RESULTS_ROOT="${RESULTS_ROOT}" \
      MODELS_ROOT="${MODELS_ROOT}" \
      LOG_DIR="${LOG_DIR}" \
      LOG_PREFIX="eval_${DOMAIN}" \
      LLAMAFACTORY_PYTHON="${LLAMAFACTORY_PYTHON}" \
      LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
      ADEPT_EVAL_SERIAL="${ADEPT_EVAL_SERIAL:-0}" \
  bash scripts/ewc/run_pretrained_eval.sh \
    "${TRAIN_OUT}" \
    "${EVAL_OUT}" \
    "${MAX_SAMPLES}"

if [[ -f "${EVAL_OUT}/summary_12_metrics.csv" ]]; then
  adept_log "Summary: ${EVAL_OUT}/summary_12_metrics.csv"
  cat "${EVAL_OUT}/summary_12_metrics.csv" | tee -a "${LOG_DIR}/pipeline.log"
else
  adept_log "WARN: missing ${EVAL_OUT}/summary_12_metrics.csv"
fi

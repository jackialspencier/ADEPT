#!/usr/bin/env bash
# Shared env for ADEPT Huawei-cloud / local pipelines.
# Source from other scripts:  source "$(dirname "$0")/env.sh"
set -euo pipefail

_ADEPT_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ADEPT_DIR="$(cd "${_ADEPT_SCRIPTS_DIR}/.." && pwd)"
# ADEPT/ -> LlamaFactory_baseline/
export REPO_ROOT="${REPO_ROOT:-$(cd "${_ADEPT_DIR}/.." && pwd)}"

# OBS-mapped write roots (ModelArts OUTPUT_DIRE / OUTPUT_RESULTS / OUTPUT_MODELS)
# shellcheck source=../../shared/scripts/cloud_paths.sh
source "${REPO_ROOT}/shared/scripts/cloud_paths.sh"

export MODEL_ID="${MODEL_ID:-gemma-2-2b-it}"
export MODEL="${MODEL:-${BASES_ROOT}/${MODEL_ID}}"
export EXP_MODEL="${EXP_MODEL:-${EXPANDED_ROOT}/${MODEL_ID}-adept-k4}"
export IMP_OUT="${IMP_OUT:-${RESULTS_ROOT}/${MODEL_ID}/adept/importance}"
export LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/${MODEL_ID}/adept/logs}"
export CFG_DIR="${CFG_DIR:-${REPO_ROOT}/ADEPT/LLaMA-Factory/examples/adept}"
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/LlamaFactory/data}"
export ADEPT_LF="${ADEPT_LF:-${REPO_ROOT}/ADEPT/LLaMA-Factory}"
export LF_DIR="${LF_DIR:-${REPO_ROOT}/LlamaFactory}"

export ADEPT_EVAL_DATA_PATH="${ADEPT_EVAL_DATA_PATH:-${REPO_ROOT}/shared/data/adept_general_competence.json}"
export ADEPT_IMPORTANCE_EVAL_STEPS="${ADEPT_IMPORTANCE_EVAL_STEPS:-500}"
export TOP_K_EXPAND="${TOP_K_EXPAND:-4}"
# Override suggested layers, e.g. EXPAND_LAYERS=22,23,24,25 (paper-aligned for gemma-2-2b-it)
export EXPAND_LAYERS="${EXPAND_LAYERS:-}"

# GPUs: domain+HumanEval share DOMAIN_GPU; public lm-eval uses BENCHMARK_GPU
export DOMAIN_GPU="${DOMAIN_GPU:-0}"
export BENCHMARK_GPU="${BENCHMARK_GPU:-1}"
export HUMANEVAL_GPU="${HUMANEVAL_GPU:-${DOMAIN_GPU}}"
export TRAIN_GPU="${TRAIN_GPU:-0}"
export CUDA_VISIBLE_DEVICES_TRAIN="${CUDA_VISIBLE_DEVICES_TRAIN:-${TRAIN_GPU}}"

export METHOD_NAME="${METHOD_NAME:-adept}"
export MAX_SAMPLES="${MAX_SAMPLES:-1000}"
export LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-4}"
export IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE:-2}"

# Domains (train dataset name = ${domain}_train)
export ADEPT_DOMAINS="${ADEPT_DOMAINS:-billsum sql_create_context fin_instruct pubmedqa dialogsum}"

# Stage skips (1=skip)
export SKIP_IMPORTANCE="${SKIP_IMPORTANCE:-0}"
export SKIP_EXPAND="${SKIP_EXPAND:-0}"
export SKIP_TRAIN="${SKIP_TRAIN:-0}"
export SKIP_EVAL="${SKIP_EVAL:-0}"

# Python: prefer LLAMAFACTORY_PYTHON, else conda env llm-baselines, else python
_resolve_python() {
  if [[ -n "${LLAMAFACTORY_PYTHON:-}" && -x "${LLAMAFACTORY_PYTHON}" ]]; then
    echo "${LLAMAFACTORY_PYTHON}"
    return
  fi
  local candidates=(
    "/opt/conda/envs/llm-baselines/bin/python"
    "${HOME}/anaconda3/envs/llm-baselines/bin/python"
    "${HOME}/miniconda3/envs/llm-baselines/bin/python"
    "/home/wanghejia/anaconda3/envs/llm-baselines/bin/python"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -x "${p}" ]]; then
      echo "${p}"
      return
    fi
  done
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "python"
}

export LLAMAFACTORY_PYTHON="$(_resolve_python)"
export PATH="$(dirname "${LLAMAFACTORY_PYTHON}"):${PATH}"

mkdir -p "${LOG_DIR}" "${IMP_OUT}" "${EXPANDED_ROOT}" "${TRAINED_ROOT}"

adept_log() {
  local msg="$1"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] ${msg}" | tee -a "${LOG_DIR}/pipeline.log"
}

adept_run() {
  # Usage: adept_run LOG_BASENAME -- command args...
  local log_base="$1"
  shift
  local log_file="${LOG_DIR}/${log_base}.log"
  adept_log "START ${log_base} -> ${log_file}"
  adept_log "CMD: $*"
  set +e
  (
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') START ${log_base} ====="
    echo "CMD: $*"
    echo "PWD: $(pwd)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "RESULTS_ROOT=${RESULTS_ROOT}"
    echo "MODELS_ROOT=${MODELS_ROOT}"
    echo "LLAMAFACTORY_PYTHON=${LLAMAFACTORY_PYTHON}"
    echo "================================================"
    "$@"
  ) >"${log_file}" 2>&1
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    adept_log "FAIL ${log_base} exit=${rc} (tail below)"
    tail -n 40 "${log_file}" | tee -a "${LOG_DIR}/pipeline.log" || true
    return "${rc}"
  fi
  adept_log "DONE ${log_base}"
  return 0
}

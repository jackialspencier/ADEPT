#!/usr/bin/env bash
# Full ADEPT pipeline for Huawei Cloud / local:
#   importance -> expand -> for each domain: train + 12-metric eval
#
# Usage (foreground, recommended under nohup on cloud):
#   export REPO_ROOT=/path/to/LlamaFactory_baseline
#   export MODEL_ID=gemma-2-2b-it
#   export DOMAIN_GPU=0 BENCHMARK_GPU=1 TRAIN_GPU=0
#   export EXPAND_LAYERS=22,23,24,25   # optional; else from importance
#   export SKIP_IMPORTANCE=1 SKIP_EXPAND=1  # if already done
#   bash ADEPT/scripts/run_adept_all_domains.sh
#
# Or background:
#   nohup bash ADEPT/scripts/run_adept_all_domains.sh \
#     > $REPO_ROOT/results/$MODEL_ID/adept/logs/nohup_pipeline.out 2>&1 &
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

adept_log "############################################################"
adept_log "ADEPT all-domains pipeline"
adept_log "REPO_ROOT=${REPO_ROOT}"
adept_log "MODEL_ID=${MODEL_ID}"
adept_log "MODEL=${MODEL}"
adept_log "EXP_MODEL=${EXP_MODEL}"
adept_log "DOMAINS=${ADEPT_DOMAINS}"
adept_log "TRAIN_GPU=${TRAIN_GPU} DOMAIN_GPU=${DOMAIN_GPU} BENCHMARK_GPU=${BENCHMARK_GPU}"
adept_log "HUMANEVAL_GPU=${HUMANEVAL_GPU} (tied to domain GPU)"
adept_log "SKIP_IMPORTANCE=${SKIP_IMPORTANCE} SKIP_EXPAND=${SKIP_EXPAND} SKIP_TRAIN=${SKIP_TRAIN} SKIP_EVAL=${SKIP_EVAL}"
adept_log "LLAMAFACTORY_PYTHON=${LLAMAFACTORY_PYTHON}"
adept_log "############################################################"

if [[ "${SKIP_IMPORTANCE}" != "1" ]]; then
  bash "${SCRIPT_DIR}/01_calc_importance.sh"
else
  adept_log "SKIP importance"
fi

if [[ "${SKIP_EXPAND}" != "1" ]]; then
  bash "${SCRIPT_DIR}/02_expand.sh"
else
  adept_log "SKIP expand"
  if [[ -n "${EXPAND_LAYERS}" ]]; then
    echo "${EXPAND_LAYERS}" > "${IMP_OUT}/effective_expand_layers.txt"
    adept_log "Forced EXPAND_LAYERS=${EXPAND_LAYERS} into effective_expand_layers.txt"
  fi
fi

if [[ ! -d "${EXP_MODEL}" ]]; then
  adept_log "ERROR: expanded model not found: ${EXP_MODEL}"
  exit 1
fi

i=0
total=0
for _d in ${ADEPT_DOMAINS}; do
  total=$((total + 1))
done

for DOMAIN in ${ADEPT_DOMAINS}; do
  i=$((i + 1))
  adept_log "======= DOMAIN ${i}/${total}: ${DOMAIN} ======="
  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    bash "${SCRIPT_DIR}/03_train_domain.sh" "${DOMAIN}"
  else
    adept_log "SKIP train ${DOMAIN}"
  fi
  if [[ "${SKIP_EVAL}" != "1" ]]; then
    bash "${SCRIPT_DIR}/04_eval_domain.sh" "${DOMAIN}"
  else
    adept_log "SKIP eval ${DOMAIN}"
  fi
  adept_log "======= DOMAIN ${DOMAIN} DONE ======="
done

adept_log "############################################################"
adept_log "ALL DOMAINS FINISHED"
adept_log "Summaries under: ${REPO_ROOT}/results/${MODEL_ID}/adept/*/eval/summary_12_metrics.csv"
adept_log "Master log: ${LOG_DIR}/pipeline.log"
adept_log "############################################################"

# Optional: collect one-line CSV rows into tables/
TABLE_DIR="${REPO_ROOT}/results/${MODEL_ID}/tables"
mkdir -p "${TABLE_DIR}"
OUT_CSV="${TABLE_DIR}/adept_all_domains_12_metrics.csv"
HEADER_WRITTEN=0
: > "${OUT_CSV}.tmp"
for DOMAIN in ${ADEPT_DOMAINS}; do
  SRC="${REPO_ROOT}/results/${MODEL_ID}/adept/${DOMAIN}/eval/summary_12_metrics.csv"
  if [[ -f "${SRC}" ]]; then
    if [[ "${HEADER_WRITTEN}" -eq 0 ]]; then
      # add domain column
      head -n 1 "${SRC}" | awk -F',' 'BEGIN{OFS=","} {print "domain",$0}' > "${OUT_CSV}.tmp"
      HEADER_WRITTEN=1
    fi
    tail -n +2 "${SRC}" | awk -F',' -v d="${DOMAIN}" 'BEGIN{OFS=","} {print d,$0}' >> "${OUT_CSV}.tmp"
  else
    adept_log "WARN: missing summary for ${DOMAIN}: ${SRC}"
  fi
done
if [[ -s "${OUT_CSV}.tmp" ]]; then
  mv "${OUT_CSV}.tmp" "${OUT_CSV}"
  adept_log "Wrote combined table ${OUT_CSV}"
  cat "${OUT_CSV}" | tee -a "${LOG_DIR}/pipeline.log"
fi

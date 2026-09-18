# ADEPT Huawei Cloud / Local Scripts

Scripts live in `ADEPT/scripts/`. They orchestrate:

1. importance → 2. expand → 3. train × 5 domains → 4. 12-metric eval × 5 domains

HumanEval shares the **domain GPU**; public lm-eval uses `BENCHMARK_GPU`.  
If `DOMAIN_GPU == BENCHMARK_GPU`, eval runs **serially** (domain → HumanEval → public).

## Layout

| Script | Role |
|--------|------|
| `env.sh` | Shared env / logging helpers |
| `01_calc_importance.sh` | Layer importance (`--top_k_expand 4`) |
| `02_expand.sh` | Block expansion |
| `03_train_domain.sh <domain>` | Train one domain |
| `04_eval_domain.sh <domain>` | 12-metric eval + aggregate |
| `run_adept_all_domains.sh` | Full pipeline |

Domains: `billsum sql_create_context fin_instruct pubmedqa dialogsum`

## Prerequisites

- `llm-baselines` Python (or set `LLAMAFACTORY_PYTHON`)
- Base weights: `$REPO_ROOT/models/bases/$MODEL_ID`
- Data: `$REPO_ROOT/LlamaFactory/data` + `shared/data/adept_general_competence.json`
- HumanEval: scores in-process by default (`HUMANEVAL_SCORE_MODE=local`, Huawei Cloud friendly).
  Optional Docker isolation: `HUMANEVAL_SCORE_MODE=docker` + image `llamafactory-humaneval:0.4.12`

## Full pipeline (Huawei Cloud, 2 GPUs)

```bash
export REPO_ROOT=${MA_JOB_DIR}/LlamaFactory_baseline   # change on cloud
# REQUIRED on ModelArts: Training Output env (e.g. OUTPUT_DIRE=/home/ma-user/modelarts/)
# Scripts nest models/ + results/ under that path (see shared/scripts/cloud_paths.sh).
export CLOUD_REQUIRE_OUTPUT=1
export MODEL_ID=gemma-2-2b-it
export TRAIN_GPU=0
export DOMAIN_GPU=0          # domain eval + HumanEval
export BENCHMARK_GPU=1       # public lm-eval
export ADEPT_EVAL_DATA_PATH=$REPO_ROOT/shared/data/adept_general_competence.json
export ADEPT_IMPORTANCE_EVAL_STEPS=500
export TOP_K_EXPAND=4
# Paper-aligned expand for gemma-2-2b-it (optional; else from importance):
export EXPAND_LAYERS=22,23,24,25

# If importance+expand already done:
# export SKIP_IMPORTANCE=1 SKIP_EXPAND=1

# After cloud_paths resolves RESULTS_ROOT from OUTPUT_DIRE:
source $REPO_ROOT/shared/scripts/cloud_paths.sh
mkdir -p $RESULTS_ROOT/$MODEL_ID/adept/logs
cd $REPO_ROOT
bash ADEPT/scripts/run_adept_all_domains.sh \
  > $RESULTS_ROOT/$MODEL_ID/adept/logs/nohup_pipeline.out 2>&1

# Logs / ckpts persist under:
#   $RESULTS_ROOT/$MODEL_ID/adept/logs/
#   $MODELS_ROOT/trained/$MODEL_ID/adept/{domain}/
```

## Single GPU

```bash
export DOMAIN_GPU=0 BENCHMARK_GPU=0 TRAIN_GPU=0
# ADEPT_EVAL_SERIAL is auto-enabled when GPUs match
bash ADEPT/scripts/run_adept_all_domains.sh
```

## Resume / partial runs

```bash
# Skip finished stages
export SKIP_IMPORTANCE=1 SKIP_EXPAND=1
# Only remaining domains:
export ADEPT_DOMAINS="fin_instruct pubmedqa dialogsum"
bash ADEPT/scripts/run_adept_all_domains.sh

# Or one domain:
bash ADEPT/scripts/03_train_domain.sh dialogsum
bash ADEPT/scripts/04_eval_domain.sh dialogsum
```

## Watch logs

```bash
LOG=$REPO_ROOT/results/$MODEL_ID/adept/logs
tail -f $LOG/pipeline.log
tail -f $LOG/01_calc_importance.log
tail -f $LOG/02_expand.log
tail -f $LOG/03_train_billsum.log
tail -f $LOG/04_eval_billsum.log
# eval sub-logs (from run_pretrained_eval.sh):
tail -f $LOG/eval_billsum_domain_gpu0.log
tail -f $LOG/eval_billsum_public_gpu1.log
tail -f $LOG/eval_billsum_humaneval_gpu0.log
```

## Outputs

- Weights: `models/trained/$MODEL_ID/adept/{domain}/`
- Eval: `results/$MODEL_ID/adept/{domain}/eval/summary_12_metrics.{csv,json}`
- Combined: `results/$MODEL_ID/tables/adept_all_domains_12_metrics.csv`

## Env cheat sheet

| Variable | Default | Meaning |
|----------|---------|---------|
| `REPO_ROOT` | parent of `ADEPT/` | Repo root |
| `MODEL_ID` | `gemma-2-2b-it` | Base model id |
| `EXPAND_LAYERS` | from importance | e.g. `22,23,24,25` |
| `TOP_K_EXPAND` | `4` | Importance k |
| `ADEPT_IMPORTANCE_EVAL_STEPS` | `500` | Dynamic LR period |
| `DOMAIN_GPU` | `0` | Domain + HumanEval |
| `BENCHMARK_GPU` | `1` | Public lm-eval |
| `TRAIN_GPU` | `0` | Train / expand / importance |
| `LLAMAFACTORY_PYTHON` | auto `llm-baselines` | Python binary |
| `SKIP_IMPORTANCE` / `SKIP_EXPAND` / `SKIP_TRAIN` / `SKIP_EVAL` | `0` | Skip stages |

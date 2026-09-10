#!/usr/bin/env bash
# 14 new arms: seven-checkpoint full eval+diagnostics, then final-only fine-tuning.
# No downloads, signals, cleanup, or burn management occur in this script.
set -euo pipefail

TASK_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_PROJECT_DIR"
TASK_RUN_ROOT=${1:?Pass a fresh absolute output directory}
[[ "$TASK_RUN_ROOT" == /mnt/local/_outputs/deep-llms_th2/swt/* ]]
[[ ! -e "$TASK_RUN_ROOT" && ! -L "$TASK_RUN_ROOT" ]]
test "$(hostname)" = thiennh-p6-oish-worker-0

source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$CONDA_DEFAULT_ENV" = swt_eval
TASK_PYTHON=/mnt/local/conda-py311/envs/swt_eval/bin/python
test "$(command -v python)" = "$TASK_PYTHON"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID

TASK_TRAIN=/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_14arms_5k_s42_20260909_a01
TASK_BENCH=/mnt/local/_data/deep-llms_th2/benchmarks/hf
TASK_DATA=/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B
TASK_TOKENIZER=/mnt/local/_models/deep-llms_th2/Qwen3-0.6B
TASK_CACHE=/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000
TASK_BUNDLE=/mnt/local/_data/deep-llms_th2/swt/diagnostics/qwen_en_pool_20260909_a01
TASK_GPUS=(0 1 2 3 4 5 6 7)
TASK_ARMS=(T768 P512-128-384 FixedResidual WNW A640 A768 A768-Direct \
    D-1024 O1024-I232 O1024-I256 C-Direct D-Direct T512 O1280)
TASK_STEPS=(250 500 1000 2000 3000 4000 5000)

"$TASK_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
mkdir "$TASK_RUN_ROOT"
exec > >(tee "$TASK_RUN_ROOT/pipeline.log") 2>&1
trap 'TASK_EXIT=$?; echo "SWT_NEXT_EVAL_FINETUNE_FAILED exit=$TASK_EXIT; subsequent stages not started" >&2' ERR
date -u

"$TASK_PYTHON" scripts/verify_manifest.py verify --root "$TASK_PROJECT_DIR" \
    --manifest resources/next_capacity_eval_source_20260910.json
"$TASK_PYTHON" scripts/verify_manifest.py verify --root "$TASK_BENCH" \
    --manifest resources/english_core_benchmark_files_20260908.json
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest discover -s tests -p test_evaluation_pipeline.py -v
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest discover -s tests -p test_diagnostics.py -v
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest discover -s tests -p test_next_capacity_eval_handoff.py -v

CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -u -m scripts.next_capacity_eval_handoff prepare \
    --training-root "$TASK_TRAIN" --benchmarks "$TASK_BENCH" \
    --eval-data "$TASK_DATA/eval" --tokenizer "$TASK_TOKENIZER" \
    --cache "$TASK_CACHE" --bundle "$TASK_BUNDLE" \
    --output "$TASK_RUN_ROOT/inputs_verified.json"

TASK_EVAL_CHECKPOINTS=()
TASK_FINAL_CHECKPOINTS=()
for TASK_ARM in "${TASK_ARMS[@]}"; do
    for TASK_STEP in "${TASK_STEPS[@]}"; do
        TASK_EVAL_CHECKPOINTS+=("${TASK_ARM}_step${TASK_STEP}=$TASK_TRAIN/$TASK_ARM/checkpoint-$TASK_STEP")
    done
    TASK_FINAL_CHECKPOINTS+=("${TASK_ARM}_step5000=$TASK_TRAIN/$TASK_ARM/checkpoint-5000")
done
test "${#TASK_EVAL_CHECKPOINTS[@]}" = 98
test "${#TASK_FINAL_CHECKPOINTS[@]}" = 14

TASK_EVAL=("$TASK_PYTHON" -u -m eval.eval_parallel \
    --checkpoints "${TASK_EVAL_CHECKPOINTS[@]}" \
    --dataset-root "$TASK_BENCH" --tokenizer-name "$TASK_TOKENIZER" \
    --languages en --precision bf16 --benchmark-batch-size 8 \
    --gpus "${TASK_GPUS[@]}" --eval-dir "$TASK_DATA/eval" --batch-size 1 \
    --preprocessing-num-workers 160 --preprocessing-batch-size 1000 \
    --preprocessing-cache-dir "$TASK_CACHE" --diagnostic-bundle "$TASK_BUNDLE" \
    --diagnostics frequency spectra gradients --output-dir "$TASK_RUN_ROOT/eval")
TASK_FINETUNE=("$TASK_PYTHON" -u -m finetune.run_all \
    --checkpoints "${TASK_FINAL_CHECKPOINTS[@]}" \
    --dataset-root "$TASK_BENCH" --tokenizer-name "$TASK_TOKENIZER" \
    --languages en --precision bf16 --benchmark-batch-size 8 \
    --gpus "${TASK_GPUS[@]}" --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 --train-language en --output-dir "$TASK_RUN_ROOT/finetune")

"${TASK_EVAL[@]}" --dry-run > "$TASK_RUN_ROOT/eval_plan.json"
"${TASK_FINETUNE[@]}" --dry-run > "$TASK_RUN_ROOT/finetune_plan.json"
"$TASK_PYTHON" - "$TASK_RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
evaluation = json.loads((root/'eval_plan.json').read_text())
finetune = json.loads((root/'finetune_plan.json').read_text())
assert len(evaluation) == 490 and len({row['name'] for row in evaluation}) == 490
assert {row['stage'] for row in evaluation} == {'ppl', 'benchmarks', 'diagnostics'}
assert sum(row['stage'] == 'diagnostics' for row in evaluation) == 294
assert len(finetune) == 126 and len({row['name'] for row in finetune}) == 126
assert all(Path(row['checkpoint']).name == 'checkpoint-5000' for row in finetune)
print('PLANS_VERIFIED eval=490 diagnostics=294 final_finetune=126', flush=True)
PY

copy_accelerate_config() {
    local TASK_ACCELERATE_TARGET
    TASK_ACCELERATE_TARGET=$("$TASK_PYTHON" -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
    mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
    cp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
    cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
    "$TASK_PYTHON" - "$TASK_ACCELERATE_TARGET" <<'PY'
import sys
from accelerate.commands.config.config_args import load_config_from_file
config = load_config_from_file(sys.argv[1])
assert config.num_processes == 8 and config.mixed_precision == 'bf16'
assert config.distributed_type.value == 'MULTI_GPU' and not config.use_cpu
print('ACCELERATE_CONFIG_COPIED_AND_VERIFIED', sys.argv[1], flush=True)
PY
}

sleep 30
"$TASK_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
copy_accelerate_config
sleep 30
"$TASK_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
echo START_98_CHECKPOINT_FULL_EVALUATION_AND_DIAGNOSTICS
"${TASK_EVAL[@]}"
"$TASK_PYTHON" -u -m scripts.next_capacity_eval_handoff verify-eval \
    --reference "$TASK_RUN_ROOT/inputs_verified.json" --output "$TASK_RUN_ROOT/eval" \
    --summary "$TASK_RUN_ROOT/eval_verified.json"

sleep 30
"$TASK_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
copy_accelerate_config
sleep 30
"$TASK_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
echo START_14_FINAL_CHECKPOINT_ENGLISH_FINETUNING
"${TASK_FINETUNE[@]}"
"$TASK_PYTHON" -u -m scripts.next_capacity_eval_handoff verify-finetune \
    --reference "$TASK_RUN_ROOT/inputs_verified.json" --output "$TASK_RUN_ROOT/finetune" \
    --summary "$TASK_RUN_ROOT/finetune_verified.json"

sleep 30
"$TASK_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
"$TASK_PYTHON" - "$TASK_RUN_ROOT" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from capacity_allocation.data import write_json
root = Path(sys.argv[1])
assert json.loads((root/'eval_verified.json').read_text())['success'] is True
assert json.loads((root/'finetune_verified.json').read_text())['success'] is True
write_json(root/'complete.json', dict(success=True, evaluation_checkpoints=98,
    evaluation_jobs=490, diagnostic_jobs=294, finetune_checkpoints=14,
    finetune_jobs=126, completed_utc=datetime.now(timezone.utc).isoformat()))
PY
echo SWT_NEXT_CAPACITY_EVAL_DIAGNOSTICS_AND_FINETUNE_COMPLETE

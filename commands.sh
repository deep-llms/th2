#1 +120+a
#th2-launch-shared-local-g16-10k-finetune-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_CHECKPOINT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_shared_local_tied_g16_10k_20260824"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

die() {
    echo "ERROR: $*"
    exit 1
}

echo '=== SharedLocal checkpoint-10000 finetune preflight ==='
date -u
hostname
test -x "$TASK_EVAL_PYTHON"
"$TASK_EVAL_PYTHON" -c 'import lm_eval, torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print("FINETUNE_ENV_OK", torch.__version__, "gpus=8")'

echo '=== verify checkpoint and completed evaluation ==='
for filename in \
    config.json model.safetensors embedding.pt trainer_state.json \
    eval_ppl.json eval_benchmarks.json; do
    test -s "$TASK_CHECKPOINT/$filename" \
        || die "missing checkpoint/eval artifact: $TASK_CHECKPOINT/$filename"
done
[[ ! -e "$TASK_CHECKPOINT/output_head.pt" ]] \
    || die 'SharedLocal tied checkpoint unexpectedly has output_head.pt'
grep -F 'Loaded compositional model: arm=shared_local' "$TASK_CHECKPOINT/eval.log"

echo '=== verify offline train and evaluation datasets ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for relpath in \
    Rowan/hellaswag allenai/ai2_arc facebook/xnli \
    alexandrainst/m_arc alexandrainst/m_hellaswag \
    facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x; do
    test -d "$TASK_BENCH_ROOT/$relpath" \
        || die "missing offline dataset: $TASK_BENCH_ROOT/$relpath"
done
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import patch_lm_eval_dataset_paths

patch_lm_eval_dataset_paths(os.environ['LM_EVAL_DATASET_ROOT'])
print('OFFLINE_LM_EVAL_PATHS_OK')
PY

echo '=== verify fresh output, storage, and all eight free B200 GPUs ==='
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to reuse finetune output: $TASK_FINETUNE_OUTPUT"
available_bytes=$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')
[[ "$available_bytes" =~ ^[0-9]+$ ]] || die 'could not determine free storage'
(( available_bytes >= 20000000000 )) \
    || die "less than 20 GB available for finetune artifacts: $available_bytes bytes"

mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
    || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for index in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
        || die "GPU $index is not B200: ${TASK_GPU_NAMES[$index]}"
done
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes active before finetune: ${TASK_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo "FINETUNE_PREFLIGHT_OK fresh_output=$TASK_FINETUNE_OUTPUT available_bytes=$available_bytes"
echo 'PROTOCOL tasks=hellaswag,arc_easy,xnli seeds=42,123,456 jobs=9 epochs=3 bf16'

echo '=== launch SharedLocal checkpoint-10000 finetuning across 8 GPUs ==='
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints shared_local_tied_g16="$TASK_CHECKPOINT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all nine SharedLocal finetune jobs ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" "$TASK_CHECKPOINT" <<'PY'
import json
import math
import os
import sys

output_dir, checkpoint = sys.argv[1:]
arm = 'shared_local_tied_g16'
expected_eval_tasks = {
    'hellaswag': {'hellaswag', 'hellaswag_ar', 'hellaswag_de', 'hellaswag_ru', 'hellaswag_vi'},
    'arc_easy': {'arc_easy', 'arc_ar', 'arc_de', 'arc_ru', 'arc_vi', 'arc_zh'},
    'xnli': {'xnli_en', 'xnli_vi', 'xnli_zh', 'xnli_de', 'xnli_ru', 'xnli_ar'},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for seed in (42, 123, 456):
        stem = f'{task}_{arm}_seed{seed}'
        result_path = os.path.join(output_dir, stem + '.json')
        log_path = os.path.join(output_dir, stem + '.log')
        model_path = os.path.join(output_dir, 'models', stem, 'model_state.pt')
        for path in (result_path, log_path, model_path):
            assert os.path.isfile(path) and os.path.getsize(path) > 0, path
        with open(result_path, encoding='utf-8') as handle:
            result = json.load(handle)
        assert result['checkpoint'] == checkpoint, result['checkpoint']
        assert result['task'] == task
        assert int(result['seed']) == seed
        assert int(result['epochs']) == 3
        assert set(result['eval_results']) == expected_tasks
        assert math.isfinite(float(result['train_time_s']))
        for eval_task, metrics in result['eval_results'].items():
            assert metrics.get('acc') is not None, (result_path, eval_task, metrics)
            assert math.isfinite(float(metrics['acc'])), (result_path, eval_task, metrics)
            if metrics.get('acc_norm') is not None:
                assert math.isfinite(float(metrics['acc_norm']))
        validated += 1
assert validated == 9
summary = os.path.join(output_dir, 'summary.md')
assert os.path.isfile(summary) and os.path.getsize(summary) > 0
print('SHARED_LOCAL_10K_FINETUNE_OK jobs=9 tasks=3 seeds=3')
PY

if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|eval failed:|FAILED \(code|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' \
    "$TASK_FINETUNE_OUTPUT"/*.log; then
    die 'failure signature found in finetune logs'
fi
mapfile -t TASK_FINAL_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after finetune: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
cat "$TASK_FINETUNE_OUTPUT/summary.md"
echo 'TH2 SHARED LOCAL G16 CHECKPOINT-10000 FINETUNE COMPLETE AND VERIFIED; ALL GPUS FREE'

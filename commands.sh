#1 +120+a
#th2-verify-offline-arc-and-launch-recent-finetune-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_INDEPENDENT_CKPT="$TASK_OUTPUT_BASE/lowrank_independent_output_r128/checkpoint-10000"
TASK_SHARED_LOCAL_CKPT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_independent_lr128_shared_local_g16"

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

echo '=== verify the two newly downloaded ARC repositories ==='
date -u
hostname
test -x "$TASK_EVAL_PYTHON"
for relpath in allenai/ai2_arc alexandrainst/m_arc; do
    dataset_dir="$TASK_BENCH_ROOT/$relpath"
    test -d "$dataset_dir"
    file_count=$(find "$dataset_dir" -type f | wc -l)
    byte_count=$(find "$dataset_dir" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
    tree_hash=$(find "$dataset_dir" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
    [[ "$file_count" -gt 0 ]] || die "$relpath contains no regular files"
    [[ "$byte_count" -gt 0 ]] || die "$relpath contains no data"
    printf 'ARC_SNAPSHOT %-30s files=%-5s bytes=%-12s sha256_tree=%s\n' \
        "$relpath" "$file_count" "$byte_count" "$tree_hash"
done

echo '=== load every finetune train dataset and ARC evaluation configuration offline ==='
"$TASK_EVAL_PYTHON" - <<'PY'
from pathlib import Path

from datasets import load_dataset

root = Path('/mnt/local/_data/@PROJECT@/benchmarks/hf')
cases = [
    ('Rowan/hellaswag', None, 'train', 'train_hellaswag'),
    ('allenai/ai2_arc', 'ARC-Easy', 'train', 'train_arc_easy'),
    ('facebook/xnli', 'en', 'train', 'train_xnli_en'),
    ('allenai/ai2_arc', 'ARC-Easy', 'test', 'eval_arc_easy'),
    *[('alexandrainst/m_arc', lang, 'test', f'eval_arc_{lang}')
      for lang in ('ar', 'de', 'ru', 'vi', 'zh')],
]
for relpath, config, split, name in cases:
    dataset = load_dataset(
        str(root / relpath), name=config, split=split, streaming=True
    )
    row = next(iter(dataset))
    assert isinstance(row, dict) and row, (name, row)
    print(f'OFFLINE_FINETUNE_DATA_OK name={name} columns={sorted(row)}')
print('OFFLINE FINETUNE DATA VERIFIED: 3 TRAIN DATASETS, 6 ARC EVAL TASKS')
PY

echo '=== install and validate all offline lm-eval task paths ==='
cd "$TASK_PROJECT_DIR"
"$TASK_EVAL_PYTHON" - <<'PY'
import os

import lm_eval

from eval.benchmarks import _DATASET_PATH_PATCHES, patch_lm_eval_dataset_paths

root = os.environ['LM_EVAL_DATASET_ROOT']
patch_lm_eval_dataset_paths(root)
tasks_dir = os.path.join(os.path.dirname(lm_eval.__file__), 'tasks')
for relative_file, repository, _aliases in _DATASET_PATH_PATCHES:
    expected = f'dataset_path: {os.path.join(root, repository)}'
    with open(os.path.join(tasks_dir, relative_file), encoding='utf-8') as handle:
        lines = [line.strip() for line in handle if line.startswith('dataset_path:')]
    assert lines == [expected], (relative_file, lines, expected)
assert len(_DATASET_PATH_PATCHES) == 16
print('LM_EVAL OFFLINE PATHS VERIFIED: 16 CONFIG FILES')
PY

echo '=== verify all eight B200 GPUs are free ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for index in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
        || die "GPU $index is not a B200: ${TASK_GPU_NAMES[$index]}"
done
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU compute processes are active: ${TASK_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
"$TASK_EVAL_PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print("CUDA_OK devices=8")'
echo 'TH2 ALL 8 B200 GPUS FREE'

echo '=== finetune preflight ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for checkpoint in "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT"; do
    test -d "$checkpoint"
    for filename in config.json embedding.pt trainer_state.json model.safetensors eval_ppl.json eval_benchmarks.json; do
        test -s "$checkpoint/$filename" \
            || die "missing required checkpoint/eval artifact: $checkpoint/$filename"
    done
done
test -s "$TASK_INDEPENDENT_CKPT/output_head.pt"
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to resume or overwrite finetune output: $TASK_FINETUNE_OUTPUT"
available_bytes=$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')
[[ "$available_bytes" =~ ^[0-9]+$ ]] || die 'could not determine available output storage'
(( available_bytes >= 40000000000 )) \
    || die "less than 40 GB available for finetune artifacts: $available_bytes bytes"
echo "fresh_output=$TASK_FINETUNE_OUTPUT available_bytes=$available_bytes"
echo 'protocol=3_tasks_x_2_arms_x_3_seeds=18_jobs; historical batch sizes 16/32; bf16; 3 epochs'

echo '=== launch the matched finetune battery across 8 GPUs ==='
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints \
        lowrank_independent_output_r128="$TASK_INDEPENDENT_CKPT" \
        shared_local_tied_g16="$TASK_SHARED_LOCAL_CKPT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all finetune results ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" <<'PY'
import json
import math
import os
import sys

output_dir = sys.argv[1]
arms = ('lowrank_independent_output_r128', 'shared_local_tied_g16')
seeds = (42, 123, 456)
expected_eval_tasks = {
    'hellaswag': {'hellaswag', 'hellaswag_ar', 'hellaswag_de', 'hellaswag_ru', 'hellaswag_vi'},
    'arc_easy': {'arc_easy', 'arc_ar', 'arc_de', 'arc_ru', 'arc_vi', 'arc_zh'},
    'xnli': {'xnli_en', 'xnli_vi', 'xnli_zh', 'xnli_de', 'xnli_ru', 'xnli_ar'},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for arm in arms:
        for seed in seeds:
            stem = f'{task}_{arm}_seed{seed}'
            result_path = os.path.join(output_dir, stem + '.json')
            log_path = os.path.join(output_dir, stem + '.log')
            model_path = os.path.join(output_dir, 'models', stem, 'model_state.pt')
            assert os.path.isfile(result_path) and os.path.getsize(result_path) > 0, result_path
            assert os.path.isfile(log_path) and os.path.getsize(log_path) > 0, log_path
            assert os.path.isfile(model_path) and os.path.getsize(model_path) > 0, model_path
            with open(result_path, encoding='utf-8') as handle:
                result = json.load(handle)
            assert result['task'] == task, (result_path, result['task'])
            assert int(result['seed']) == seed, (result_path, result['seed'])
            assert expected_tasks == set(result['eval_results']), (
                result_path, result['eval_results'].keys()
            )
            for eval_task, metrics in result['eval_results'].items():
                assert metrics.get('acc') is not None, (result_path, eval_task, metrics)
                assert math.isfinite(float(metrics['acc'])), (result_path, eval_task, metrics)
                if metrics.get('acc_norm') is not None:
                    assert math.isfinite(float(metrics['acc_norm'])), (
                        result_path, eval_task, metrics
                    )
            validated += 1
assert validated == 18
summary_path = os.path.join(output_dir, 'summary.md')
assert os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
print('FINETUNE_RESULTS_OK jobs=18 tasks=3 arms=2 seeds=3')
PY

if grep -HniE 'traceback|out of memory|nan|eval failed:|FAILED \(code' \
    "$TASK_FINETUNE_OUTPUT"/*.log; then
    die 'failure signature found in finetune logs'
fi
mapfile -t TASK_FINAL_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after finetuning: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
cat "$TASK_FINETUNE_OUTPUT/summary.md"
echo 'TH2 FINETUNE COMPLETE AND VERIFIED: 18 JOBS; ALL GPUS FREE'

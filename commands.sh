#1 +120+a
#th2-resume-shared-local-full-then-eval-finetune-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_TRAIN_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TRAIN_OUTPUT="$TASK_OUTPUT_BASE/shared_local_tied_g16"
TASK_RESUME_CHECKPOINT="$TASK_TRAIN_OUTPUT/checkpoint-10000"
TASK_TRAIN_LOG_DIR="$TASK_OUTPUT_BASE/logs/shared_local_tied_g16"
TASK_CONTINUATION_LOG="$TASK_TRAIN_LOG_DIR/shared_local_tied_g16_resume_10000_to_full.log"
TASK_EVAL_LAUNCH_LOG="$TASK_TRAIN_OUTPUT/eval_parallel_full.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_shared_local_tied_g16_full"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_DEST="$HOME/.cache/huggingface/accelerate/default_config.yaml"

export SPARSE_EMB_PYTHON="$TASK_TRAIN_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

die() {
    echo "ERROR: $*"
    exit 1
}

require_gpus_free() {
    local phase=$1
    mapfile -t TASK_GPU_PIDS < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
    )
    [[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
        || die "$phase: GPU compute processes are active: ${TASK_GPU_PIDS[*]}"
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
    echo "$phase: ALL 8 B200 GPUS FREE"
}

echo '=== activate and validate environments ==='
date -u
hostname
test -x "$TASK_CONDA"
test -x "$TASK_TRAIN_PYTHON"
test -x "$TASK_EVAL_PYTHON"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
test "$CONDA_DEFAULT_ENV" = sparse_emb
test "$(command -v python3.11)" = "$TASK_TRAIN_PYTHON"
"$TASK_TRAIN_PYTHON" -c 'import accelerate, datasets, torch, transformers; print("TRAIN_ENV_OK", torch.__version__, accelerate.__version__, transformers.__version__, datasets.__version__)'
"$TASK_EVAL_PYTHON" -c 'import lm_eval, torch; print("EVAL_ENV_OK", torch.__version__)'

echo '=== verify hardware and free GPUs before continuation ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for index in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
        || die "GPU $index is not a B200: ${TASK_GPU_NAMES[$index]}"
done
require_gpus_free PRE_RESUME

echo '=== verify exact checkpoint-10000 resume state ==='
test -d "$TASK_RESUME_CHECKPOINT"
for filename in \
    config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
    embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
    rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_RESUME_CHECKPOINT/$filename" \
        || die "missing resume artifact: $TASK_RESUME_CHECKPOINT/$filename"
done
[[ ! -e "$TASK_RESUME_CHECKPOINT/output_head.pt" ]] \
    || die 'SharedLocal tied checkpoint unexpectedly has output_head.pt'
test -s "$TASK_TRAIN_OUTPUT/train_config.json"

latest_checkpoint=$(find "$TASK_TRAIN_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -printf '%f\n' \
    | sed 's/checkpoint-//' | sort -n | tail -1)
[[ "$latest_checkpoint" == 10000 ]] \
    || die "expected checkpoint-10000 to be latest, found checkpoint-$latest_checkpoint"

"$TASK_TRAIN_PYTHON" - "$TASK_TRAIN_OUTPUT/train_config.json" "$TASK_RESUME_CHECKPOINT" <<'PY'
import json
import os
import sys

import torch

config_path, checkpoint = sys.argv[1:]
with open(config_path, encoding='utf-8') as handle:
    config = json.load(handle)
with open(os.path.join(checkpoint, 'trainer_state.json'), encoding='utf-8') as handle:
    trainer_state = json.load(handle)
training = config['training']
compositional = config['compositional']
assert trainer_state['global_step'] == 10_000, trainer_state['global_step']
assert trainer_state['max_steps'] > 10_000, trainer_state['max_steps']
assert training['num_train_epochs'] == 1, training['num_train_epochs']
assert training['per_device_train_batch_size'] == 16
assert training['gradient_accumulation_steps'] == 4
assert training['bf16'] is True
assert compositional['arm'] == 'shared_local'
assert compositional['shared_rank'] == 64
assert compositional['local_embed_rank'] == 64
assert compositional['num_groups'] == 16
assert compositional['tie_output'] is True
assert compositional['independent_lowrank_output'] is False
scheduler = torch.load(
    os.path.join(checkpoint, 'scheduler.pt'), map_location='cpu', weights_only=True
)
assert scheduler['last_epoch'] == 10_000, scheduler['last_epoch']
print(
    'RESUME_STATE_OK',
    f"global_step={trainer_state['global_step']}",
    f"max_steps={trainer_state['max_steps']}",
    'schedule=original_one_epoch',
)
PY

echo '=== verify model, training data, and untouched final-stage outputs ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for lang in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$lang"
    test "$(find "$TASK_DATA_DIR/$lang" -type f -name '*.arrow' | wc -l)" -gt 0
    test -d "$TASK_EVAL_DIR/$lang"
    test "$(find "$TASK_EVAL_DIR/$lang" -type f -name '*.arrow' | wc -l)" -gt 0
done
for relpath in \
    facebook/xnli facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x \
    Rowan/hellaswag alexandrainst/m_hellaswag \
    allenai/ai2_arc alexandrainst/m_arc; do
    test -d "$TASK_BENCH_ROOT/$relpath" || die "missing offline dataset: $relpath"
done
[[ ! -e "$TASK_CONTINUATION_LOG" ]] \
    || die "refusing to overwrite continuation log: $TASK_CONTINUATION_LOG"
for artifact in \
    "$TASK_TRAIN_OUTPUT/eval.log" \
    "$TASK_TRAIN_OUTPUT/eval_ppl.json" \
    "$TASK_TRAIN_OUTPUT/eval_benchmarks.json" \
    "$TASK_EVAL_LAUNCH_LOG"; do
    [[ ! -e "$artifact" ]] || die "refusing to overwrite final evaluation artifact: $artifact"
done
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to resume or overwrite final finetune output: $TASK_FINETUNE_OUTPUT"

echo '=== install and verify Accelerate configuration ==='
test -s "$TASK_ACCELERATE_SOURCE"
mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"

echo '=== resume SharedLocal G16 from 10000 through the full one-epoch schedule ==='
echo 'NO max_steps; NO stop-at-step; strict automatic resume from the latest checkpoint'
mkdir -p "$TASK_TRAIN_LOG_DIR"
bash scripts/train_shared_local_tied_g16.sh 2>&1 | tee "$TASK_CONTINUATION_LOG"

echo '=== verify completed full training state ==='
grep -F 'Checkpoint detected:' "$TASK_CONTINUATION_LOG" | grep -F 'checkpoint-10000'
grep -F 'Training complete. Model saved to:' "$TASK_CONTINUATION_LOG"
if grep -niE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|NCCL.*(error|timeout)|nan' "$TASK_CONTINUATION_LOG"; then
    die 'failure signature found in SharedLocal continuation log'
fi
for filename in config.json model.safetensors embedding.pt train_config.json trainer_state.json; do
    test -s "$TASK_TRAIN_OUTPUT/$filename" \
        || die "missing final training artifact: $TASK_TRAIN_OUTPUT/$filename"
done
[[ ! -e "$TASK_TRAIN_OUTPUT/output_head.pt" ]] \
    || die 'final SharedLocal tied model unexpectedly has output_head.pt'

"$TASK_TRAIN_PYTHON" - "$TASK_TRAIN_OUTPUT" <<'PY'
import json
import math
import os
import sys

output_dir = sys.argv[1]
with open(os.path.join(output_dir, 'trainer_state.json'), encoding='utf-8') as handle:
    state = json.load(handle)
assert state['global_step'] == state['max_steps'], (
    state['global_step'], state['max_steps']
)
assert state['global_step'] > 10_000, state['global_step']
finite_rows = 0
for row in state['log_history']:
    for key in ('loss', 'grad_norm', 'learning_rate'):
        if key in row:
            assert math.isfinite(float(row[key])), (key, row[key])
    if 'loss' in row:
        finite_rows += 1
assert finite_rows > 0
print(
    'FULL_TRAINING_STATE_OK',
    f"global_step={state['global_step']}",
    f"finite_loss_rows={finite_rows}",
)
PY
require_gpus_free POST_TRAIN

echo '=== patch lm-eval to the offline benchmark snapshots ==='
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import patch_lm_eval_dataset_paths

patch_lm_eval_dataset_paths(os.environ['LM_EVAL_DATASET_ROOT'])
print('OFFLINE_LM_EVAL_PATHS_OK')
PY

echo '=== run full final-model PPL + 26-task evaluation ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "$TASK_TRAIN_OUTPUT" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_EVAL_LAUNCH_LOG"

"$TASK_EVAL_PYTHON" - "$TASK_TRAIN_OUTPUT" <<'PY'
import json
import math
import os
import sys

checkpoint = sys.argv[1]
with open(os.path.join(checkpoint, 'eval_ppl.json'), encoding='utf-8') as handle:
    perplexity = json.load(handle)
with open(os.path.join(checkpoint, 'eval_benchmarks.json'), encoding='utf-8') as handle:
    benchmarks = json.load(handle)
assert set(perplexity) == {'en', 'vi', 'zh', 'ru', 'de', 'ar'}
for language, metrics in perplexity.items():
    assert int(metrics['num_tokens']) > 0, (language, metrics)
    assert math.isfinite(float(metrics['loss'])), (language, metrics)
    assert math.isfinite(float(metrics['perplexity'])), (language, metrics)
assert len(benchmarks) == 26, len(benchmarks)
for task, metrics in benchmarks.items():
    accuracy = metrics.get('acc,none', metrics.get('acc'))
    assert accuracy is not None and math.isfinite(float(accuracy)), (task, metrics)
print('FINAL_EVAL_OK ppl_languages=6 benchmark_tasks=26')
PY
grep -F 'Loaded compositional model: arm=shared_local' "$TASK_TRAIN_OUTPUT/eval.log"
require_gpus_free POST_EVAL

echo '=== run final-model matched finetuning battery: 3 tasks x 3 seeds ==='
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints shared_local_tied_g16_full="$TASK_TRAIN_OUTPUT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all nine final-model finetune jobs ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" <<'PY'
import json
import math
import os
import sys

output_dir = sys.argv[1]
arm = 'shared_local_tied_g16_full'
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
        assert result['task'] == task
        assert int(result['seed']) == seed
        assert set(result['eval_results']) == expected_tasks
        for eval_task, metrics in result['eval_results'].items():
            assert metrics.get('acc') is not None, (result_path, eval_task, metrics)
            assert math.isfinite(float(metrics['acc'])), (result_path, eval_task, metrics)
            if metrics.get('acc_norm') is not None:
                assert math.isfinite(float(metrics['acc_norm']))
        validated += 1
assert validated == 9
summary = os.path.join(output_dir, 'summary.md')
assert os.path.isfile(summary) and os.path.getsize(summary) > 0
print('FINAL_FINETUNE_OK jobs=9 tasks=3 seeds=3')
PY
if grep -HniE 'traceback|out of memory|nan|eval failed:|FAILED \(code' \
    "$TASK_FINETUNE_OUTPUT"/*.log; then
    die 'failure signature found in final-model finetune logs'
fi
require_gpus_free FINAL
cat "$TASK_FINETUNE_OUTPUT/summary.md"
echo 'TH2 SHARED LOCAL FULL TRAINING + EVAL + FINETUNE COMPLETE AND VERIFIED'

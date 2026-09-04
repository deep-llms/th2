#1 +120+a
#th2-readonly-preflight-five-checkpoint-eval-finetune-20260904-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01"
TASK_CHECKPOINTS=(
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460/checkpoint-6500"
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460/checkpoint-10000"
    "$TASK_OUTPUT_BASE/product_code_quota_h6144/checkpoint-6500"
    "$TASK_OUTPUT_BASE/product_code_quota_h6144/checkpoint-10000"
    "$TASK_OUTPUT_BASE/btmos_k3_c256_lb/checkpoint-6500"
)

cd "$TASK_PROJECT"
echo '=== identity and environment ==='
date -u
hostname
test -x "$TASK_EVAL_PYTHON"
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
test -s resources/accelerate_config.yaml
test -s resources/llm_pretrain_burn.py
"$TASK_EVAL_PYTHON" - <<'PY'
import datasets
import lm_eval
import torch
import transformers
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
assert all('B200' in torch.cuda.get_device_name(i) for i in range(8))
print('EVAL_ENV_OK', torch.__version__, transformers.__version__, datasets.__version__)
PY

echo '=== five exact checkpoints ==='
"$TASK_EVAL_PYTHON" - "${TASK_CHECKPOINTS[@]}" <<'PY'
import json
import math
import os
import sys
import torch
from compositional.loading import is_compositional

for checkpoint in sys.argv[1:]:
    assert os.path.isdir(checkpoint), checkpoint
    required = (
        'config.json', 'model.safetensors', 'trainer_state.json',
        'optimizer.pt', 'scheduler.pt', 'embedding.pt',
        *(f'rng_state_{rank}.pth' for rank in range(8)),
    )
    for filename in required:
        path = os.path.join(checkpoint, filename)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, path
    assert is_compositional(checkpoint), checkpoint
    assert not os.path.exists(os.path.join(checkpoint, 'output_head.pt')), checkpoint
    with open(os.path.join(checkpoint, 'trainer_state.json'), encoding='utf-8') as handle:
        state = json.load(handle)
    expected_step = int(os.path.basename(checkpoint).split('-')[1])
    assert int(state['global_step']) == expected_step, (checkpoint, state['global_step'])
    losses = [float(row['loss']) for row in state.get('log_history', []) if 'loss' in row]
    assert losses and all(math.isfinite(value) for value in losses), checkpoint
    embedding = torch.load(
        os.path.join(checkpoint, 'embedding.pt'), map_location='cpu', weights_only=True
    )
    assert embedding and all(torch.isfinite(value).all() for value in embedding.values()), checkpoint
    print(f'CHECKPOINT_OK path={checkpoint} step={expected_step} last_loss={losses[-1]:.6f}')

btmos_10k = os.path.join(os.path.dirname(sys.argv[-1]), 'checkpoint-10000')
assert not os.path.exists(btmos_10k), btmos_10k
print('BTMOS_10000_CORRECTLY_ABSENT_AFTER_INTENTIONAL_STOP')
PY

echo '=== offline eval data and benchmark mirrors ==='
for TASK_LANGUAGE in en vi zh ru de ar; do
    test -d "$TASK_EVAL_DIR/$TASK_LANGUAGE"
    find "$TASK_EVAL_DIR/$TASK_LANGUAGE" -type f -name '*.arrow' -print -quit | grep -q .
    echo "eval_language_ok=$TASK_LANGUAGE"
done
for TASK_RELPATH in \
    facebook/xnli facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x \
    Rowan/hellaswag allenai/ai2_arc \
    alexandrainst/m_arc alexandrainst/m_hellaswag; do
    test -d "$TASK_BENCH_ROOT/$TASK_RELPATH"
    echo "benchmark_ok=$TASK_RELPATH"
done
LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT" "$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import TASK_CONFIGS, patch_lm_eval_dataset_paths
patch_lm_eval_dataset_paths(os.environ['LM_EVAL_DATASET_ROOT'])
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks
print('OFFLINE_TASK_CONFIG_OK tasks=26')
PY

echo '=== result collision and storage check ==='
test ! -e "$TASK_FINETUNE_OUTPUT"
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    for TASK_ARTIFACT in eval.log eval_ppl.json eval_benchmarks.json; do
        test ! -e "$TASK_CHECKPOINT/$TASK_ARTIFACT"
    done
done
df -h "$TASK_OUTPUT_BASE"
TASK_AVAILABLE_BYTES="$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')"
[[ "$TASK_AVAILABLE_BYTES" =~ ^[0-9]+$ ]]
(( TASK_AVAILABLE_BYTES >= 100000000000 ))
echo "STORAGE_OK available_bytes=$TASK_AVAILABLE_BYTES"

echo '=== current burn ownership, one worker per GPU ==='
TASK_ALL_PIDS=()
for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
    mapfile -t TASK_ONE_PIDS < <(
        nvidia-smi -i "$TASK_GPU_INDEX" \
            --query-compute-apps=pid --format=csv,noheader,nounits |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
    test "${#TASK_ONE_PIDS[@]}" -eq 1
    TASK_PID="${TASK_ONE_PIDS[0]}"
    test "$TASK_PID" -ne 1
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    TASK_PARENT="$(tr '\0' ' ' < "/proc/$TASK_PPID/cmdline")"
    [[ "$TASK_PARENT" == *'/tmp/llm_pretrain_burn.py'* ]]
    TASK_ALL_PIDS+=("$TASK_PID")
    echo "gpu=$TASK_GPU_INDEX burn_worker=$TASK_PID burn_parent=$TASK_PPID"
done
mapfile -t TASK_UNIQUE_PIDS < <(printf '%s\n' "${TASK_ALL_PIDS[@]}" | sort -nu)
test "${#TASK_UNIQUE_PIDS[@]}" -eq 8
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 FIVE-CHECKPOINT EVAL/FINETUNE PREFLIGHT READY'

#!/usr/bin/env bash
# Evaluate Tiered RankLift and language-balanced GroupReduce at step 10000,
# using the standard
# three-task/three-seed finetuning battery, and restore the communicating burn.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

gpu_pids_for_indices() {
    local indices="$1"
    nvidia-smi -i "$indices" --query-compute-apps=pid \
        --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

validate_b200_node() {
    mapfile -t TASK_GPU_NAMES < <(
        nvidia-smi --query-gpu=name --format=csv,noheader \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    )
    [[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
        || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
    for TASK_GPU in "${!TASK_GPU_NAMES[@]}"; do
        [[ "${TASK_GPU_NAMES[$TASK_GPU]}" == *B200* ]] \
            || die "GPU $TASK_GPU is not B200: ${TASK_GPU_NAMES[$TASK_GPU]}"
    done
}

require_free_gpus() {
    local stage="$1"
    local process_output
    process_output="$(gpu_pids)" || die "GPU query failed $stage"
    mapfile -t TASK_STAGE_GPU_PIDS < <(printf '%s\n' "$process_output" | sed '/^$/d')
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader
    [[ "${#TASK_STAGE_GPU_PIDS[@]}" -eq 0 ]] \
        || die "GPU compute processes remain $stage: ${TASK_STAGE_GPU_PIDS[*]}"
    echo "ALL 8 B200 GPUS FREE $stage"
}

install_accelerate_config() {
    mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
    cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
    cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
    grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
    grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
    grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
    echo "ACCELERATE_CONFIG_OK target=$TASK_ACCELERATE_TARGET"
}

scan_fatal_logs() {
    local label="$1"
    shift
    local pattern
    pattern='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|FAILED \(code|eval failed:|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
    if grep -HniE "$pattern" "$@"; then
        die "fatal signature found in $label logs"
    fi
}

# Validate that the requested physical GPUs each have exactly one process and
# that every process descends from one common llm_pretrain_burn.py launcher.
burn_launcher_for_indices() {
    local indices="$1"
    local expected="$2"
    "$TASK_BURN_PYTHON" - "$indices" "$expected" "$TASK_BURN_TARGET" <<'PY'
import csv
from pathlib import Path
import subprocess
import sys

indices = [int(value) for value in sys.argv[1].split(',')]
expected = int(sys.argv[2])
burn_path = sys.argv[3]

def output(*args):
    return subprocess.check_output(args, text=True).strip()

def rows(value):
    return [[field.strip() for field in row] for row in csv.reader(value.splitlines()) if row]

def cmdline(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')

def parent(pid):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('PPid:'):
            return int(line.split()[1])
    raise RuntimeError(f'missing PPid for {pid}')

gpu_rows = rows(output(
    'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'
))
index_to_uuid = {int(index): uuid for index, uuid in gpu_rows}
assert all(index in index_to_uuid for index in indices), (indices, index_to_uuid)

app_rows = rows(output(
    'nvidia-smi', '-i', ','.join(map(str, indices)),
    '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'
))
assert len(app_rows) == expected == len(indices), app_rows
seen_indices = set()
launcher_sets = []
pids = []
for uuid, pid_text in app_rows:
    matching = [index for index in indices if index_to_uuid[index] == uuid]
    assert len(matching) == 1, (uuid, matching)
    index = matching[0]
    assert index not in seen_indices, index
    seen_indices.add(index)
    pid = int(pid_text)
    assert pid != 1, pid
    pids.append(pid)
    ancestors = set()
    seen = set()
    cursor = pid
    while cursor > 1 and cursor not in seen:
        seen.add(cursor)
        if burn_path in cmdline(cursor):
            ancestors.add(cursor)
        cursor = parent(cursor)
    assert ancestors, (index, pid)
    launcher_sets.append(ancestors)

common = set.intersection(*launcher_sets)
assert len(common) == 1, (common, launcher_sets)
launcher = next(iter(common))
assert launcher != 1
assert burn_path in cmdline(launcher), cmdline(launcher)
assert seen_indices == set(indices), seen_indices
assert len(pids) == len(set(pids)) == expected, pids
print(launcher)
PY
}

stop_verified_burn() {
    local indices="$1"
    local expected="$2"
    local label="$3"
    local launcher
    local -a workers
    launcher="$(burn_launcher_for_indices "$indices" "$expected")" \
        || die "could not validate $label burn ownership"
    [[ "$launcher" =~ ^[0-9]+$ && "$launcher" -ne 1 ]] \
        || die "unsafe $label burn launcher: $launcher"
    mapfile -t workers < <(gpu_pids_for_indices "$indices")
    [[ "${#workers[@]}" -eq "$expected" ]] \
        || die "$label burn changed after validation"
    for pid in "${workers[@]}"; do
        [[ "$pid" =~ ^[0-9]+$ && "$pid" -ne 1 ]] \
            || die "unsafe $label worker PID: $pid"
    done
    echo "VERIFIED_${label}_BURN launcher=$launcher workers=${workers[*]} indices=$indices"
    # Only signal the exact GPU compute workers. The non-PID-1 spawn launcher
    # exits after its workers terminate; this preserves the runner's PID 1.
    kill -9 "${workers[@]}" 2>/dev/null || true
}

verify_burn_ready() {
    local launcher="$1"
    local indices="$2"
    local expected="$3"
    local world="$4"
    local log="$5"
    local ready=0
    for _ in $(seq 1 300); do
        kill -0 "$launcher" 2>/dev/null || {
            cat "$log" >&2
            die "burn launcher $launcher exited during startup"
        }
        local ready_count
        ready_count="$( { grep -o 'gpu_burn_ready' "$log" 2>/dev/null || true; } | wc -l)"
        mapfile -t TASK_READY_PIDS < <(gpu_pids_for_indices "$indices")
        if [[ "$ready_count" -eq "$expected" && "${#TASK_READY_PIDS[@]}" -eq "$expected" ]]; then
            ready=1
            break
        fi
        sleep 1
    done
    [[ "$ready" -eq 1 ]] || {
        cat "$log" >&2
        die "burn did not become ready on GPUs $indices"
    }
    [[ "$(burn_launcher_for_indices "$indices" "$expected")" == "$launcher" ]] \
        || die "burn ownership mismatch after startup"
    [[ "$( { grep -o "world_size=$world" "$log" || true; } | wc -l)" -eq "$expected" ]] \
        || die "not every burn rank joined world_size=$world"
    for _ in $(seq 1 180); do
        grep -Fq 'gpu_burn_progress' "$log" && break
        sleep 1
    done
    grep -Fq 'gpu_burn_progress' "$log" \
        || die "burn on GPUs $indices made no synchronized compute/communication progress"
}

start_burn() {
    local indices="$1"
    local expected="$2"
    local port="$3"
    local log="$4"
    local pid_file="$5"
    local launcher_variable="$6"
    [[ ! -e "$log" && ! -e "$pid_file" ]] || die "burn output already exists: $log"
    mkdir -p "$(dirname "$log")"
    nohup env \
        CUDA_VISIBLE_DEVICES="$indices" \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT="$port" \
        NCCL_DEBUG=WARN \
        GPU_BURN_MIN_WORLD_SIZE=2 \
        "$TASK_BURN_PYTHON" -u "$TASK_BURN_TARGET" \
        >"$log" 2>&1 </dev/null &
    local launcher=$!
    printf -v "$launcher_variable" '%s' "$launcher"
    printf '%s\n' "$launcher" > "$pid_file"
    echo "burn_launcher=$launcher indices=$indices log=$log"
    verify_burn_ready "$launcher" "$indices" "$expected" "$expected" "$log"
}

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:?SPARSE_EMB_PROJECT_DIR is required}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:?SPARSE_EMB_OUTPUT_BASE is required}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:?SPARSE_EMB_MODEL_DIR is required}"
TASK_EVAL_DIR="${SPARSE_EMB_EVAL_DIR:?SPARSE_EMB_EVAL_DIR is required}"
TASK_BENCH_ROOT="${SPARSE_EMB_BENCH_ROOT:?SPARSE_EMB_BENCH_ROOT is required}"
TASK_EVAL_PYTHON="${SPARSE_EMB_EVAL_PYTHON:?SPARSE_EMB_EVAL_PYTHON is required}"
TASK_CONDA="${SPARSE_EMB_CONDA:?SPARSE_EMB_CONDA is required}"

TASK_TIERED="$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512/checkpoint-10000"
TASK_GROUPREDUCE="$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4/checkpoint-10000"
TASK_CHECKPOINTS=("$TASK_TIERED" "$TASK_GROUPREDUCE")
TASK_LABELS=(tiered_ranklift_c512_s10000 groupreduce_lb_t4_s10000)
TASK_RUN_ID=tiered_c512_groupreduce_lb_10k_20260905_a01
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_${TASK_RUN_ID}.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_${TASK_RUN_ID}"
TASK_COMPLETION_MARKER="$TASK_OUTPUT_BASE/eval_finetune_${TASK_RUN_ID}.complete"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN_SOURCE="$TASK_PROJECT_DIR/resources/llm_pretrain_burn.py"
TASK_BURN_TARGET=/tmp/llm_pretrain_burn.py
TASK_PARTIAL_BURN_LOG="$TASK_OUTPUT_BASE/logs/burn_eval_${TASK_RUN_ID}.log"
TASK_PARTIAL_BURN_PID_FILE="$TASK_OUTPUT_BASE/logs/burn_eval_${TASK_RUN_ID}.pid"
TASK_FULL_BURN_LOG="$TASK_OUTPUT_BASE/logs/burn_final_${TASK_RUN_ID}.log"
TASK_FULL_BURN_PID_FILE="$TASK_OUTPUT_BASE/logs/burn_final_${TASK_RUN_ID}.pid"
TASK_BURN_PYTHON=/usr/bin/python3
TASK_PARTIAL_BURN_LAUNCHER=''
TASK_RECOVERY_BURN_LAUNCHER=''
TASK_FINAL_BURN_STARTED=0

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"

restore_burn_on_failure() {
    local status=$?
    trap - EXIT
    if [[ "$status" -ne 0 && "$TASK_FINAL_BURN_STARTED" -ne 1 ]]; then
        echo "WORKFLOW_FAILED status=$status; attempting safe burn restoration" >&2
        if [[ -n "$TASK_PARTIAL_BURN_LAUNCHER" ]] \
                && kill -0 "$TASK_PARTIAL_BURN_LAUNCHER" 2>/dev/null; then
            if burn_launcher_for_indices 2,3,4,5,6,7 6 >/dev/null 2>&1; then
                stop_verified_burn 2,3,4,5,6,7 6 PARTIAL || true
                sleep 10
            fi
        fi
        local process_output
        if ! process_output="$(gpu_pids)"; then
            echo 'GPU query failed during recovery; no burn started' >&2
            exit "$status"
        fi
        mapfile -t TASK_FAILURE_GPU_PIDS < <(printf '%s\n' "$process_output" | sed '/^$/d')
        if [[ "${#TASK_FAILURE_GPU_PIDS[@]}" -eq 0 ]]; then
            echo 'No GPU jobs remain; restoring full burn after failure.' >&2
            start_burn 0,1,2,3,4,5,6,7 8 29500 \
                "${TASK_FULL_BURN_LOG%.log}_recovery.log" "${TASK_FULL_BURN_PID_FILE%.pid}_recovery.pid" \
                TASK_RECOVERY_BURN_LAUNCHER
            echo "WORKFLOW_FAILED_BURN_RESTORED launcher=$TASK_RECOVERY_BURN_LAUNCHER"
            wait "$TASK_RECOVERY_BURN_LAUNCHER" || true
        else
            echo "GPU jobs remain after failure; not disturbing them: ${TASK_FAILURE_GPU_PIDS[*]}" >&2
        fi
    fi
    exit "$status"
}
trap restore_burn_on_failure EXIT

mkdir -p "$TASK_OUTPUT_BASE/logs"
TASK_WORKFLOW_LOG="$TASK_OUTPUT_BASE/logs/eval_finetune_${TASK_RUN_ID}.log"
[[ ! -e "$TASK_WORKFLOW_LOG" ]] || die "workflow log exists: $TASK_WORKFLOW_LOG"
exec > >(tee "$TASK_WORKFLOW_LOG") 2>&1
echo '=== two-checkpoint eval -> finetune -> burn ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_CONDA"
test -x "$TASK_EVAL_PYTHON"
test -x "$TASK_BURN_PYTHON"
test -s "$TASK_ACCELERATE_SOURCE"
test -s "$TASK_BURN_SOURCE"
validate_b200_node

echo '=== activate and validate eval environment ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate eval
[[ "${CONDA_DEFAULT_ENV:-}" == eval ]] || die 'failed to activate eval environment'
[[ "$(command -v python3.11)" == "$TASK_EVAL_PYTHON" ]] \
    || die "wrong eval Python: $(command -v python3.11)"
"$TASK_EVAL_PYTHON" - <<'PY'
import datasets
import lm_eval
import torch
import transformers
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
assert all('B200' in torch.cuda.get_device_name(i) for i in range(8))
print('EVAL_ENV_OK', torch.__version__, transformers.__version__, datasets.__version__)
PY

echo '=== validate exact tied checkpoints and train configurations ==='
"$TASK_EVAL_PYTHON" - "${TASK_CHECKPOINTS[@]}" <<'PY'
import json
import math
import os
import sys
import torch
from compositional.loading import is_compositional

checkpoints = sys.argv[1:]
expected = {
    checkpoints[0]: ('tiered_ranklift', 10000, {
        'tiered_ranklift_code_dims': '1024,512,192,64',
        'tiered_ranklift_lift_dims': '0,0,320,192',
    }),
    checkpoints[1]: ('groupreduce', 10000, {
        'groupreduce_num_groups': 4,
        'groupreduce_ranks': '1024,512,192,64',
        'mos_components': 1,
    }),
}
for checkpoint, (arm, step, fields) in expected.items():
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
    assert int(state['global_step']) == step, (checkpoint, state['global_step'])
    losses = [float(row['loss']) for row in state.get('log_history', []) if 'loss' in row]
    assert losses and all(math.isfinite(value) for value in losses), checkpoint
    with open(os.path.join(os.path.dirname(checkpoint), 'train_config.json'), encoding='utf-8') as handle:
        comp = json.load(handle)['compositional']
    assert comp.get('arm') == arm and comp.get('tie_output') is True, (checkpoint, comp)
    for key, value in fields.items():
        actual = comp.get(key)
        if isinstance(value, float):
            assert math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12), (checkpoint, key, actual)
        else:
            assert actual == value, (checkpoint, key, actual, value)
    embedding = torch.load(os.path.join(checkpoint, 'embedding.pt'), map_location='cpu', weights_only=True)
    assert embedding and all(torch.isfinite(value).all() for value in embedding.values()), checkpoint
    print(f'CHECKPOINT_OK checkpoint={checkpoint} arm={arm} step={step}')

print('BOTH_CHECKPOINTS_VALID step=10000')
PY

echo '=== verify fresh result paths and offline inputs ==='
[[ ! -e "$TASK_EVAL_LAUNCH_LOG" ]] || die "eval log exists: $TASK_EVAL_LAUNCH_LOG"
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] || die "finetune output exists: $TASK_FINETUNE_OUTPUT"
[[ ! -e "$TASK_COMPLETION_MARKER" ]] || die "completion marker exists: $TASK_COMPLETION_MARKER"
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    for TASK_ARTIFACT in eval.log eval_ppl.json eval_benchmarks.json; do
        [[ ! -e "$TASK_CHECKPOINT/$TASK_ARTIFACT" ]] \
            || die "refusing to overwrite eval artifact: $TASK_CHECKPOINT/$TASK_ARTIFACT"
    done
done
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for TASK_LANGUAGE in en vi zh ru de ar; do
    test -d "$TASK_EVAL_DIR/$TASK_LANGUAGE"
    find "$TASK_EVAL_DIR/$TASK_LANGUAGE" -type f -name '*.arrow' -print -quit | grep -q .
done
for TASK_RELPATH in \
    facebook/xnli facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x \
    Rowan/hellaswag allenai/ai2_arc \
    alexandrainst/m_arc alexandrainst/m_hellaswag; do
    test -d "$TASK_BENCH_ROOT/$TASK_RELPATH" \
        || die "missing offline benchmark dataset: $TASK_RELPATH"
done
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import TASK_CONFIGS, patch_lm_eval_dataset_paths
patch_lm_eval_dataset_paths(os.environ['LM_EVAL_DATASET_ROOT'])
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks
print('OFFLINE_EVAL_INPUTS_OK tasks=26')
PY
TASK_AVAILABLE_BYTES="$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')"
[[ "$TASK_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || die 'could not determine free storage'
(( TASK_AVAILABLE_BYTES >= 100000000000 )) \
    || die "less than 100 GB available: $TASK_AVAILABLE_BYTES bytes"
echo "STORAGE_PREFLIGHT_OK available_bytes=$TASK_AVAILABLE_BYTES"

echo '=== safely stop the current all-GPU burn ==='
TASK_INITIAL_GPU_OUTPUT="$(gpu_pids)" || die 'initial GPU query failed'
mapfile -t TASK_INITIAL_GPU_PIDS < <(printf '%s\n' "$TASK_INITIAL_GPU_OUTPUT" | sed '/^$/d')
if [[ "${#TASK_INITIAL_GPU_PIDS[@]}" -eq 0 ]]; then
    echo 'GPUS_ALREADY_FREE; no process will be signaled'
elif [[ "${#TASK_INITIAL_GPU_PIDS[@]}" -eq 8 ]]; then
    [[ "$(burn_launcher_for_indices 0,1,2,3,4,5,6,7 8)" =~ ^[0-9]+$ ]] \
        || die 'current full burn validation failed'
    stop_verified_burn 0,1,2,3,4,5,6,7 8 FULL
else
    die "unexpected partial GPU ownership before eval: ${TASK_INITIAL_GPU_PIDS[*]}"
fi
sleep 30
require_free_gpus '30 SECONDS AFTER VERIFIED FULL-BURN CANCELLATION'

echo '=== install Accelerate config and current burn implementation ==='
install_accelerate_config
test -s "$TASK_BURN_TARGET"
cmp "$TASK_BURN_SOURCE" "$TASK_BURN_TARGET" || die "runner burn differs from the inspected implementation"
sleep 30
require_free_gpus '30 SECONDS AFTER CONFIG COPY AND BEFORE EVAL'

echo '=== occupy GPUs 2-7 while two eval jobs use GPUs 0-1 ==='
start_burn 2,3,4,5,6,7 6 29537 \
    "$TASK_PARTIAL_BURN_LOG" "$TASK_PARTIAL_BURN_PID_FILE" \
    TASK_PARTIAL_BURN_LAUNCHER
[[ "$TASK_PARTIAL_BURN_LAUNCHER" =~ ^[0-9]+$ ]] \
    || die "invalid partial burn launcher: $TASK_PARTIAL_BURN_LAUNCHER"
for TASK_GPU_INDEX in 0 1; do
    mapfile -t TASK_UNUSED_PIDS < <(gpu_pids_for_indices "$TASK_GPU_INDEX")
    [[ "${#TASK_UNUSED_PIDS[@]}" -eq 0 ]] || die "GPU $TASK_GPU_INDEX is not free before eval"
done

echo '=== run full evaluation for both available checkpoints ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "${TASK_CHECKPOINTS[@]}" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 2 \
    --log "$TASK_EVAL_LAUNCH_LOG"

echo '=== validate both complete evaluations ==='
"$TASK_EVAL_PYTHON" - "${TASK_CHECKPOINTS[@]}" <<'PY'
import json
import math
import os
import sys
expected_languages = {'en', 'vi', 'zh', 'ru', 'de', 'ar'}
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, 'eval_ppl.json'), encoding='utf-8') as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, 'eval_benchmarks.json'), encoding='utf-8') as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == expected_languages, (checkpoint, perplexity.keys())
    from eval.benchmarks import TASK_CONFIGS
    expected_benchmarks = {task for group in TASK_CONFIGS.values() for task in group}
    assert set(benchmarks) == expected_benchmarks, (checkpoint, set(benchmarks) ^ expected_benchmarks)
    for language, metrics in perplexity.items():
        assert int(metrics['num_tokens']) > 0, (checkpoint, language, metrics)
        assert math.isfinite(float(metrics['loss'])), (checkpoint, language, metrics)
        assert math.isfinite(float(metrics['perplexity'])), (checkpoint, language, metrics)
    for task, metrics in benchmarks.items():
        accuracy = metrics.get('acc,none', metrics.get('acc'))
        assert accuracy is not None and math.isfinite(float(accuracy)), (checkpoint, task, metrics)
    print(f'EVAL_JSON_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26')
PY
grep -Fq 'All 2 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
grep -Fq 'Loaded compositional model: arm=tiered_ranklift' "$TASK_TIERED/eval.log"
grep -Fq 'Loaded compositional model: arm=groupreduce' "$TASK_GROUPREDUCE/eval.log"
scan_fatal_logs evaluation "$TASK_EVAL_LAUNCH_LOG" "$TASK_TIERED/eval.log" "$TASK_GROUPREDUCE/eval.log"

echo '=== stop only the verified partial eval burn ==='
[[ "$(burn_launcher_for_indices 2,3,4,5,6,7 6)" == "$TASK_PARTIAL_BURN_LAUNCHER" ]] \
    || die 'partial burn ownership changed during eval'
stop_verified_burn 2,3,4,5,6,7 6 PARTIAL
TASK_PARTIAL_BURN_LAUNCHER=''
sleep 60
require_free_gpus '60 SECONDS AFTER TWO-MODEL EVAL AND PARTIAL-BURN STOP'
install_accelerate_config
sleep 60
require_free_gpus '60 SECONDS AFTER CONFIG COPY AND BEFORE FINETUNE'

echo '=== run standard 18-job finetuning battery across all eight GPUs ==='
echo 'PROTOCOL checkpoints=2 tasks=3 seeds=3 jobs=18 epochs=3 max_parallel_jobs=8'
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints \
        "${TASK_LABELS[0]}=$TASK_TIERED" \
        "${TASK_LABELS[1]}=$TASK_GROUPREDUCE" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all 18 finetuning jobs ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" \
    "${TASK_LABELS[0]}=$TASK_TIERED" \
    "${TASK_LABELS[1]}=$TASK_GROUPREDUCE" <<'PY'
import json
import math
import os
import sys
output_dir = sys.argv[1]
arms = dict(spec.split('=', 1) for spec in sys.argv[2:])
expected_eval_tasks = {
    'hellaswag': {'hellaswag', 'hellaswag_ar', 'hellaswag_de', 'hellaswag_ru', 'hellaswag_vi'},
    'arc_easy': {'arc_easy', 'arc_ar', 'arc_de', 'arc_ru', 'arc_vi', 'arc_zh'},
    'xnli': {'xnli_en', 'xnli_vi', 'xnli_zh', 'xnli_de', 'xnli_ru', 'xnli_ar'},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for arm, checkpoint in arms.items():
        for seed in (42, 123, 456):
            stem = f'{task}_{arm}_seed{seed}'
            paths = (
                os.path.join(output_dir, stem + '.json'),
                os.path.join(output_dir, stem + '.log'),
                os.path.join(output_dir, 'models', stem, 'model_state.pt'),
            )
            assert all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths), paths
            with open(paths[0], encoding='utf-8') as handle:
                result = json.load(handle)
            assert result['checkpoint'] == checkpoint, result['checkpoint']
            assert result['task'] == task and int(result['seed']) == seed, result
            assert int(result['epochs']) == 3, result['epochs']
            assert math.isclose(float(result['lr']), 2e-5), result['lr']
            assert set(result['eval_results']) == expected_tasks, result['eval_results'].keys()
            assert math.isfinite(float(result['train_time_s'])), result['train_time_s']
            for eval_task, metrics in result['eval_results'].items():
                assert metrics.get('acc') is not None, (paths[0], eval_task, metrics)
                assert math.isfinite(float(metrics['acc'])), (paths[0], eval_task, metrics)
                if metrics.get('acc_norm') is not None:
                    assert math.isfinite(float(metrics['acc_norm'])), (paths[0], eval_task, metrics)
            validated += 1
assert validated == 18, validated
summary = os.path.join(output_dir, 'summary.md')
assert os.path.isfile(summary) and os.path.getsize(summary) > 0, summary
print('TWO_CHECKPOINT_FINETUNE_OK jobs=18 tasks=3 seeds=3 epochs=3')
PY
mapfile -t TASK_FINETUNE_LOGS < <(
    find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | sort
)
[[ "${#TASK_FINETUNE_LOGS[@]}" -eq 18 ]] \
    || die "expected 18 finetune logs, found ${#TASK_FINETUNE_LOGS[@]}"
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*tiered_ranklift_c512_s10000*.log; do
    grep -Fq 'Loaded compositional model: arm=tiered_ranklift' "$TASK_LOG" \
        || die "Tiered RankLift loader confirmation missing from $TASK_LOG"
done
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*groupreduce_lb_t4_s10000*.log; do
    grep -Fq 'Loaded compositional model: arm=groupreduce' "$TASK_LOG" \
        || die "GroupReduce loader confirmation missing from $TASK_LOG"
done
scan_fatal_logs finetune "${TASK_FINETUNE_LOGS[@]}"
cat "$TASK_FINETUNE_OUTPUT/summary.md"

echo '=== wait and require all GPUs free before restoring full burn ==='
sleep 60
require_free_gpus '60 SECONDS AFTER TWO-CHECKPOINT FINETUNE'

echo '=== launch and verify current communicating high-memory burn ==='
TASK_FINAL_BURN_LAUNCHER=''
start_burn 0,1,2,3,4,5,6,7 8 29500 \
    "$TASK_FULL_BURN_LOG" "$TASK_FULL_BURN_PID_FILE" \
    TASK_FINAL_BURN_LAUNCHER
[[ "$TASK_FINAL_BURN_LAUNCHER" =~ ^[0-9]+$ ]] \
    || die "invalid full burn launcher: $TASK_FINAL_BURN_LAUNCHER"
TASK_FINAL_BURN_STARTED=1

"$TASK_BURN_PYTHON" - "$TASK_FINAL_BURN_LAUNCHER" <<'PY'
import csv
from pathlib import Path
import subprocess
import sys
launcher = int(sys.argv[1])

def output(*args):
    return subprocess.check_output(args, text=True).strip()
def rows(value):
    return [[field.strip() for field in row] for row in csv.reader(value.splitlines()) if row]
def parent(pid):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('PPid:'):
            return int(line.split()[1])
    raise RuntimeError(pid)
def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False

gpu_rows = rows(output('nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'))
assert len(gpu_rows) == 8, gpu_rows
uuid_to_index = {}
for index_text, uuid, name, used_text, total_text in gpu_rows:
    index = int(index_text)
    used, total = float(used_text), float(total_text)
    assert 'B200' in name, (index, name)
    assert 0.80 <= used / total <= 0.90, (index, used, total)
    uuid_to_index[uuid] = index
app_rows = rows(output('nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'))
assert len(app_rows) == 8, app_rows
seen = set()
for uuid, pid_text in app_rows:
    index = uuid_to_index[uuid]
    pid = int(pid_text)
    assert index not in seen and pid != 1 and descendant(pid), (index, pid)
    seen.add(index)
assert seen == set(range(8)), seen
print('CORRECT_BURN_PROCESS_MEMORY_AND_COMMUNICATION_OK gpus=8 target_fraction=0.85')
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader

"$TASK_EVAL_PYTHON" - "$TASK_COMPLETION_MARKER" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys
path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + '.tmp')
temporary.write_text(
    'status=success\n'
    'checkpoints=2\n'
    'eval_languages=6\n'
    'eval_benchmark_tasks=26\n'
    'finetune_jobs=18\n'
    f'completed_utc={datetime.now(timezone.utc).isoformat()}\n',
    encoding='utf-8',
)
temporary.replace(path)
print(f'COMPLETION_MARKER_WRITTEN path={path}')
PY

trap - EXIT
echo 'TH2 TWO-CHECKPOINT EVAL AND FINETUNE COMPLETE; CORRECT 8-GPU COMMUNICATING BURN ACTIVE'

# Keep the runner job alive and reap the supervised burn if it exits.
# A completed foreground shell may cause the runner to clean up descendants.
echo "SUPERVISING_FINAL_BURN launcher=$TASK_FINAL_BURN_LAUNCHER"
wait "$TASK_FINAL_BURN_LAUNCHER"
die 'final GPU burn exited'

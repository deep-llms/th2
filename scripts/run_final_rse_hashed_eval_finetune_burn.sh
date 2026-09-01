#!/usr/bin/env bash
# Evaluate and finetune the final RSE and hashed Product-Code checkpoint-10k
# models, then restore the runner's communicating high-memory GPU burn.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
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
    mapfile -t TASK_STAGE_GPU_PIDS < <(gpu_pids)
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

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:?SPARSE_EMB_PROJECT_DIR is required}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:?SPARSE_EMB_OUTPUT_BASE is required}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:?SPARSE_EMB_MODEL_DIR is required}"
TASK_EVAL_DIR="${SPARSE_EMB_EVAL_DIR:?SPARSE_EMB_EVAL_DIR is required}"
TASK_BENCH_ROOT="${SPARSE_EMB_BENCH_ROOT:?SPARSE_EMB_BENCH_ROOT is required}"
TASK_EVAL_PYTHON="${SPARSE_EMB_EVAL_PYTHON:?SPARSE_EMB_EVAL_PYTHON is required}"
TASK_CONDA="${SPARSE_EMB_CONDA:?SPARSE_EMB_CONDA is required}"

TASK_RSE_CKPT="$TASK_OUTPUT_BASE/residual_subspace_experts_tied_g12_r120_q80/checkpoint-10000"
TASK_HASHED_CKPT="$TASK_OUTPUT_BASE/product_code_hashed_h2048/checkpoint-10000"
TASK_CHECKPOINTS=("$TASK_RSE_CKPT" "$TASK_HASHED_CKPT")
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_final_rse_hashed_10k_20260901.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_final_rse_hashed_10k_20260901"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN_SOURCE="$TASK_PROJECT_DIR/resources/llm_pretrain_burn.py"
TASK_BURN_TARGET=/tmp/llm_pretrain_burn.py
TASK_BURN_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
TASK_BURN_PYTHON=/usr/bin/python3

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo '=== final RSE + Hashed checkpoint-10k eval -> finetune -> burn ==='
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

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
print(
    "EVAL_ENV_OK",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"datasets={datasets.__version__}",
    "gpus=8",
)
PY

echo '=== validate exact complete tied checkpoints ==='
"$TASK_EVAL_PYTHON" - "$TASK_RSE_CKPT" "$TASK_HASHED_CKPT" <<'PY'
import json
import math
import os
import sys

import torch

from compositional.loading import is_compositional

rse, hashed = sys.argv[1:]
expected = {
    rse: {
        "arm": "residual_subspace_experts",
        "rse_base_rank": 120,
        "rse_expert_rank": 80,
        "rse_num_experts": 12,
        "rse_router_dim": 32,
        "rse_top_k": 2,
    },
    hashed: {
        "arm": "product_code",
        "product_code_head_size": 2048,
        "product_code_num_hashes": 4,
        "product_code_num_buckets": 4096,
        "product_code_assignment": "hashed",
        "product_code_seed": 0,
    },
}
for checkpoint, fields in expected.items():
    assert os.path.isdir(checkpoint), checkpoint
    for filename in (
        "config.json", "model.safetensors", "trainer_state.json",
        "optimizer.pt", "scheduler.pt", "embedding.pt",
        "rng_state_0.pth", "rng_state_1.pth", "rng_state_2.pth",
        "rng_state_3.pth", "rng_state_4.pth", "rng_state_5.pth",
        "rng_state_6.pth", "rng_state_7.pth",
    ):
        path = os.path.join(checkpoint, filename)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, path
    assert is_compositional(checkpoint), checkpoint
    assert not os.path.exists(os.path.join(checkpoint, "output_head.pt")), checkpoint
    with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    assert int(state["global_step"]) == 10000, (checkpoint, state["global_step"])
    loss_rows = [row for row in state.get("log_history", []) if "loss" in row]
    assert loss_rows, checkpoint
    for row in loss_rows:
        for key in ("loss", "grad_norm", "learning_rate"):
            assert math.isfinite(float(row[key])), (checkpoint, key, row[key])
    with open(os.path.join(os.path.dirname(checkpoint), "train_config.json"), encoding="utf-8") as handle:
        comp = json.load(handle)["compositional"]
    assert comp.get("tie_output") is True, (checkpoint, comp)
    for key, value in fields.items():
        assert comp.get(key) == value, (checkpoint, key, comp.get(key), value)
    embedding = torch.load(
        os.path.join(checkpoint, "embedding.pt"), map_location="cpu", weights_only=True
    )
    assert embedding and all(torch.isfinite(value).all() for value in embedding.values()), checkpoint
    print(f"CHECKPOINT_OK checkpoint={checkpoint} arm={fields['arm']} step=10000")
print("TWO_FINAL_CHECKPOINTS_COMPLETE_AND_TIED")
PY

echo '=== verify fresh result paths and offline inputs ==='
[[ ! -e "$TASK_EVAL_LAUNCH_LOG" ]] \
    || die "refusing to overwrite evaluation log: $TASK_EVAL_LAUNCH_LOG"
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to reuse finetune output: $TASK_FINETUNE_OUTPUT"
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
    [[ "$(find "$TASK_EVAL_DIR/$TASK_LANGUAGE" -type f -name '*.arrow' | wc -l)" -gt 0 ]]
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

patch_lm_eval_dataset_paths(os.environ["LM_EVAL_DATASET_ROOT"])
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks
print("OFFLINE_EVAL_INPUTS_OK tasks=26")
PY
TASK_AVAILABLE_BYTES="$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')"
[[ "$TASK_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || die 'could not determine free storage'
(( TASK_AVAILABLE_BYTES >= 50000000000 )) \
    || die "less than 50 GB available: $TASK_AVAILABLE_BYTES bytes"
echo "STORAGE_PREFLIGHT_OK available_bytes=$TASK_AVAILABLE_BYTES"

echo '=== verify and stop only the current runner burn ==='
mapfile -t TASK_INITIAL_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_INITIAL_GPU_PIDS[@]}" -eq 8 ]] \
    || die "expected 8 current burn workers, found ${#TASK_INITIAL_GPU_PIDS[@]}"
test -s "$TASK_BURN_PID_FILE"
TASK_OLD_BURN_LAUNCHER="$(cat "$TASK_BURN_PID_FILE")"
[[ "$TASK_OLD_BURN_LAUNCHER" =~ ^[0-9]+$ && "$TASK_OLD_BURN_LAUNCHER" != 1 ]] \
    || die "invalid burn launcher PID: $TASK_OLD_BURN_LAUNCHER"
kill -0 "$TASK_OLD_BURN_LAUNCHER"
TASK_OLD_BURN_CMD="$(tr '\0' ' ' < "/proc/$TASK_OLD_BURN_LAUNCHER/cmdline")"
[[ "$TASK_OLD_BURN_CMD" == *"$TASK_BURN_TARGET"* ]] \
    || die "unexpected burn launcher command: $TASK_OLD_BURN_CMD"
"$TASK_BURN_PYTHON" - "$TASK_OLD_BURN_LAUNCHER" "${TASK_INITIAL_GPU_PIDS[@]}" <<'PY'
import csv
from pathlib import Path
import subprocess
import sys

launcher = int(sys.argv[1])
pids = [int(value) for value in sys.argv[2:]]

def parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing PPid for {pid}")

def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False

assert launcher != 1
assert len(pids) == len(set(pids)) == 8, pids
assert all(pid != 1 and descendant(pid) for pid in pids), (launcher, pids)
mapping = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
rows = [
    [field.strip() for field in row]
    for row in csv.reader(mapping.splitlines())
    if row
]
assert len(rows) == 8, rows
assert {int(pid_text) for _, pid_text in rows} == set(pids), rows
assert len({uuid for uuid, _ in rows}) == 8, rows
print(f"VERIFIED_RUNNER_BURN_TREE launcher={launcher} workers={pids}")
PY
# All targets were validated above.  Kill the GPU workers and their non-PID-1
# launcher in one operation so a quickly exiting spawn parent cannot be reused
# between separate kill commands.
kill -9 "${TASK_INITIAL_GPU_PIDS[@]}" "$TASK_OLD_BURN_LAUNCHER" 2>/dev/null || true
sleep 30
require_free_gpus '30 SECONDS AFTER VERIFIED BURN CANCELLATION'

echo '=== install Accelerate config and recheck GPUs ==='
install_accelerate_config
sleep 30
require_free_gpus '30 SECONDS AFTER ACCELERATE CONFIG COPY AND BEFORE EVAL'

echo '=== run full eval_parallel for both checkpoint-10000 models ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "${TASK_CHECKPOINTS[@]}" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_EVAL_LAUNCH_LOG"

echo '=== validate both complete evaluations ==='
"$TASK_EVAL_PYTHON" - "${TASK_CHECKPOINTS[@]}" <<'PY'
import json
import math
import os
import sys

expected_languages = {"en", "vi", "zh", "ru", "de", "ar"}
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == expected_languages, (checkpoint, perplexity.keys())
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for language, metrics in perplexity.items():
        assert int(metrics["num_tokens"]) > 0, (checkpoint, language, metrics)
        assert math.isfinite(float(metrics["loss"])), (checkpoint, language, metrics)
        assert math.isfinite(float(metrics["perplexity"])), (checkpoint, language, metrics)
    for task, metrics in benchmarks.items():
        accuracy = metrics.get("acc,none", metrics.get("acc"))
        assert accuracy is not None and math.isfinite(float(accuracy)), (checkpoint, task, metrics)
    print(f"EVAL_JSON_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26")
PY
grep -Fq 'All 2 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
grep -Fq 'Loaded compositional model: arm=residual_subspace_experts' "$TASK_RSE_CKPT/eval.log"
grep -Fq 'Loaded compositional model: arm=product_code' "$TASK_HASHED_CKPT/eval.log"
scan_fatal_logs evaluation \
    "$TASK_EVAL_LAUNCH_LOG" "$TASK_RSE_CKPT/eval.log" "$TASK_HASHED_CKPT/eval.log"

echo '=== wait, require free GPUs, and reinstall Accelerate config ==='
sleep 60
require_free_gpus '60 SECONDS AFTER TWO-MODEL EVAL'
install_accelerate_config
sleep 60
require_free_gpus '60 SECONDS AFTER CONFIG COPY AND BEFORE FINETUNE'

echo '=== run standard 18-job finetuning battery across eight GPUs ==='
echo 'PROTOCOL checkpoints=2 tasks=3 seeds=3 jobs=18 epochs=3 bf16 max_parallel_jobs=8'
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints \
        residual_subspace_experts_tied_g12_r120_q80="$TASK_RSE_CKPT" \
        product_code_hashed_h2048="$TASK_HASHED_CKPT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all 18 finetuning jobs ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" "$TASK_RSE_CKPT" "$TASK_HASHED_CKPT" <<'PY'
import json
import math
import os
import sys

output_dir, rse, hashed = sys.argv[1:]
arms = {
    "residual_subspace_experts_tied_g12_r120_q80": rse,
    "product_code_hashed_h2048": hashed,
}
expected_eval_tasks = {
    "hellaswag": {"hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi"},
    "arc_easy": {"arc_easy", "arc_ar", "arc_de", "arc_ru", "arc_vi", "arc_zh"},
    "xnli": {"xnli_en", "xnli_vi", "xnli_zh", "xnli_de", "xnli_ru", "xnli_ar"},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for arm, checkpoint in arms.items():
        for seed in (42, 123, 456):
            stem = f"{task}_{arm}_seed{seed}"
            paths = (
                os.path.join(output_dir, stem + ".json"),
                os.path.join(output_dir, stem + ".log"),
                os.path.join(output_dir, "models", stem, "model_state.pt"),
            )
            assert all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths), paths
            with open(paths[0], encoding="utf-8") as handle:
                result = json.load(handle)
            assert result["checkpoint"] == checkpoint, result["checkpoint"]
            assert result["task"] == task and int(result["seed"]) == seed, result
            assert int(result["epochs"]) == 3, result["epochs"]
            assert set(result["eval_results"]) == expected_tasks, result["eval_results"].keys()
            assert math.isfinite(float(result["train_time_s"])), result["train_time_s"]
            for eval_task, metrics in result["eval_results"].items():
                assert metrics.get("acc") is not None, (paths[0], eval_task, metrics)
                assert math.isfinite(float(metrics["acc"])), (paths[0], eval_task, metrics)
                if metrics.get("acc_norm") is not None:
                    assert math.isfinite(float(metrics["acc_norm"])), (paths[0], eval_task, metrics)
            validated += 1
assert validated == 18, validated
summary = os.path.join(output_dir, "summary.md")
assert os.path.isfile(summary) and os.path.getsize(summary) > 0, summary
print("TWO_MODEL_FINETUNE_OK jobs=18 tasks=3 seeds=3 epochs=3")
PY
mapfile -t TASK_FINETUNE_LOGS < <(
    find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | sort
)
[[ "${#TASK_FINETUNE_LOGS[@]}" -eq 18 ]] \
    || die "expected 18 finetune logs, found ${#TASK_FINETUNE_LOGS[@]}"
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*_residual_subspace_experts_tied_g12_r120_q80_seed*.log; do
    grep -Fq 'Loaded compositional model: arm=residual_subspace_experts' "$TASK_LOG" \
        || die "RSE loader confirmation missing from $TASK_LOG"
done
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*_product_code_hashed_h2048_seed*.log; do
    grep -Fq 'Loaded compositional model: arm=product_code' "$TASK_LOG" \
        || die "Product-Code loader confirmation missing from $TASK_LOG"
done
scan_fatal_logs finetune "${TASK_FINETUNE_LOGS[@]}"
cat "$TASK_FINETUNE_OUTPUT/summary.md"

echo '=== wait and require all GPUs free before restoring burn ==='
sleep 60
require_free_gpus '60 SECONDS AFTER TWO-MODEL FINETUNE'

echo '=== install and launch current communicating high-memory burn ==='
install -m 0644 "$TASK_BURN_SOURCE" "$TASK_BURN_TARGET"
cmp "$TASK_BURN_SOURCE" "$TASK_BURN_TARGET"
rm -f "$TASK_BURN_LOG" "$TASK_BURN_PID_FILE"
nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29500 \
    NCCL_DEBUG=WARN \
    "$TASK_BURN_PYTHON" -u "$TASK_BURN_TARGET" \
    >"$TASK_BURN_LOG" 2>&1 &
TASK_NEW_BURN_LAUNCHER=$!
printf '%s\n' "$TASK_NEW_BURN_LAUNCHER" > "$TASK_BURN_PID_FILE"
echo "new_burn_launcher=$TASK_NEW_BURN_LAUNCHER log=$TASK_BURN_LOG"

TASK_BURN_READY=0
for _ in $(seq 1 240); do
    kill -0 "$TASK_NEW_BURN_LAUNCHER" 2>/dev/null || {
        cat "$TASK_BURN_LOG" >&2
        die 'burn launcher exited during startup'
    }
    TASK_READY_COUNT="$(grep -Fc 'gpu_burn_ready' "$TASK_BURN_LOG" 2>/dev/null || true)"
    mapfile -t TASK_NEW_GPU_PIDS < <(gpu_pids)
    if [[ "$TASK_READY_COUNT" -eq 8 && "${#TASK_NEW_GPU_PIDS[@]}" -eq 8 ]]; then
        TASK_BURN_READY=1
        break
    fi
    sleep 1
done
[[ "$TASK_BURN_READY" -eq 1 ]] || {
    cat "$TASK_BURN_LOG" >&2
    die 'burn did not become ready on all eight GPUs'
}
for _ in $(seq 1 120); do
    grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG" && break
    sleep 1
done
grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG" \
    || die 'burn made no synchronized compute/communication progress'
[[ "$(grep -Fc 'world_size=8' "$TASK_BURN_LOG")" -eq 8 ]] \
    || die 'not every burn rank joined the eight-GPU process group'

"$TASK_BURN_PYTHON" - "$TASK_NEW_BURN_LAUNCHER" <<'PY'
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
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing PPid for {pid}")

def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False

gpu_rows = rows(output("nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"))
assert len(gpu_rows) == 8, gpu_rows
uuid_to_index = {}
for index_text, uuid, name, used_text, total_text, utilization_text in gpu_rows:
    index = int(index_text)
    used, total = float(used_text), float(total_text)
    assert "B200" in name, (index, name)
    assert 0.80 <= used / total <= 0.90, (index, used, total)
    uuid_to_index[uuid] = index

app_rows = rows(output("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"))
assert len(app_rows) == 8, app_rows
seen = set()
for uuid, pid_text, memory_text in app_rows:
    assert uuid in uuid_to_index, uuid
    index = uuid_to_index[uuid]
    assert index not in seen, index
    pid = int(pid_text)
    assert pid != 1 and descendant(pid), (index, pid, launcher)
    seen.add(index)
assert seen == set(range(8)), seen
print("CORRECT_BURN_PROCESS_AND_MEMORY_MAPPING_OK gpus=8 target_fraction=0.85")
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 RSE HASHED EVAL AND FINETUNE COMPLETE; CORRECT 8-GPU COMMUNICATING BURN ACTIVE'

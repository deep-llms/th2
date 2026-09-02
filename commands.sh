#1 +60+a
#th2-readonly-verify-final-rse-hashed-workflow-20260902-a04
set -euo pipefail

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RUNTIME=/tmp/run_final_rse_hashed_eval_finetune_burn_f954755720e1.sh
TASK_RSE_CKPT="$TASK_OUTPUT_BASE/residual_subspace_experts_tied_g12_r120_q80/checkpoint-10000"
TASK_HASHED_CKPT="$TASK_OUTPUT_BASE/product_code_hashed_h2048/checkpoint-10000"
TASK_EVAL_LOG="$TASK_OUTPUT_BASE/eval_parallel_final_rse_hashed_10k_20260901.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_final_rse_hashed_10k_20260901"
TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11

cd "$TASK_PROJECT_DIR"
echo '=== identity and live GPU state ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== exact workflow stage and GPU ownership ==='
"$TASK_PYTHON" - \
    "$TASK_RUNTIME" "$TASK_FINETUNE_OUTPUT" \
    "$TASK_RSE_CKPT" "$TASK_HASHED_CKPT" <<'PY'
import os
import subprocess
import sys

runtime, finetune_output, rse, hashed = sys.argv[1:]
checkpoints = (rse, hashed)

def args_for(pid):
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.rstrip(b"\0").split(b"\0") if part]

def parent_for(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            return next(int(line.split()[1]) for line in handle if line.startswith("PPid:"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
        return None

def has_script(args, suffix):
    return any(arg == suffix or arg.endswith("/" + suffix) for arg in args[1:])

processes = {}
parents = {}
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    args = args_for(pid)
    if args:
        processes[pid] = args
        parents[pid] = parent_for(pid)

workflow = []
eval_launchers = []
eval_workers = []
finetune_launchers = []
finetune_processes = []
burn_launchers = []
burn_workers = []
for pid, args in processes.items():
    if len(args) >= 2 and os.path.basename(args[0]) in {"bash", "sh"} and args[1] == runtime:
        workflow.append(pid)
    if has_script(args, "eval/eval_parallel.py") and all(path in args for path in checkpoints):
        eval_launchers.append(pid)
    if has_script(args, "eval/eval_checkpoint.py") and any(path in args for path in checkpoints):
        eval_workers.append(pid)
    if has_script(args, "finetune/run_all.py") and finetune_output in args:
        finetune_launchers.append(pid)
    if has_script(args, "finetune/train.py") and finetune_output in args and any(path in args for path in checkpoints):
        finetune_processes.append(pid)
    if any(arg == "/tmp/llm_pretrain_burn.py" for arg in args[1:]):
        burn_launchers.append(pid)

for pid, args in processes.items():
    if "multiprocessing.spawn" in " ".join(args) and parents.get(pid) in burn_launchers:
        burn_workers.append(pid)

if len(finetune_launchers) == 1:
    finetune_trainers = [
        pid for pid in finetune_processes
        if parents.get(pid) == finetune_launchers[0]
    ]
else:
    finetune_trainers = []
finetune_helpers = [
    pid for pid in finetune_processes if pid not in finetune_trainers
]

gpu_text = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True,
)
gpu_pids = sorted({int(line.strip()) for line in gpu_text.splitlines() if line.strip()})

stage_flags = {
    "eval": bool(eval_launchers or eval_workers),
    "finetune": bool(finetune_launchers or finetune_processes),
    "burn": bool(burn_launchers or burn_workers),
}
active = [name for name, enabled in stage_flags.items() if enabled]
assert len(active) <= 1, (active, processes)
stage = active[0] if active else "transition"

if stage != "burn":
    assert len(workflow) == 1, workflow
if stage == "eval":
    assert len(eval_launchers) == 1, eval_launchers
    assert 1 <= len(eval_workers) <= 2, eval_workers
    assert set(gpu_pids) == set(eval_workers), (gpu_pids, eval_workers)
elif stage == "finetune":
    assert len(finetune_launchers) == 1, finetune_launchers
    assert 1 <= len(finetune_trainers) <= 8, finetune_trainers
    assert all(parents.get(pid) in finetune_trainers for pid in finetune_helpers), (
        finetune_helpers, finetune_trainers
    )
    assert set(gpu_pids) == set(finetune_trainers), (gpu_pids, finetune_trainers)
elif stage == "burn":
    assert len(burn_launchers) == 1, burn_launchers
    assert len(burn_workers) == 8, burn_workers
    assert set(gpu_pids) == set(burn_workers), (gpu_pids, burn_workers)
else:
    assert not gpu_pids, gpu_pids

print(f"STAGE={stage}")
print(f"WORKFLOW_PIDS={workflow}")
print(f"EVAL_LAUNCHERS={eval_launchers} EVAL_WORKERS={eval_workers}")
print(
    f"FINETUNE_LAUNCHERS={finetune_launchers} "
    f"FINETUNE_TRAINERS={finetune_trainers} "
    f"FINETUNE_HELPERS={finetune_helpers}"
)
print(f"BURN_LAUNCHERS={burn_launchers} BURN_WORKERS={burn_workers}")
print(f"GPU_PIDS={gpu_pids}")
print("LIVE_PROCESS_AND_GPU_OWNERSHIP_OK")
PY

echo '=== checkpoint and Accelerate invariants ==='
cmp resources/accelerate_config.yaml \
    /mnt/local/.cache/huggingface/accelerate/default_config.yaml
grep -Fxq 'distributed_type: MULTI_GPU' resources/accelerate_config.yaml
grep -Fxq 'mixed_precision: bf16' resources/accelerate_config.yaml
grep -Fxq 'num_processes: 8' resources/accelerate_config.yaml
"$TASK_PYTHON" - "$TASK_RSE_CKPT" "$TASK_HASHED_CKPT" <<'PY'
import json
import os
import sys
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    assert int(state["global_step"]) == 10000, (checkpoint, state["global_step"])
    for filename in ("config.json", "model.safetensors", "embedding.pt"):
        path = os.path.join(checkpoint, filename)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, path
print("CHECKPOINTS_STILL_COMPLETE step=10000 models=2")
PY

echo '=== evaluation artifacts and live tails ==='
for TASK_CKPT in "$TASK_RSE_CKPT" "$TASK_HASHED_CKPT"; do
    echo "checkpoint=$TASK_CKPT"
    for TASK_FILE in eval.log eval_ppl.json eval_benchmarks.json; do
        if [[ -s "$TASK_CKPT/$TASK_FILE" ]]; then
            echo "$TASK_FILE=present bytes=$(stat -c %s "$TASK_CKPT/$TASK_FILE")"
        else
            echo "$TASK_FILE=pending"
        fi
    done
    if [[ -s "$TASK_CKPT/eval.log" ]]; then
        tail -n 35 "$TASK_CKPT/eval.log"
    fi
done
if [[ -s "$TASK_EVAL_LOG" ]]; then
    echo '=== eval_parallel tail ==='
    tail -n 50 "$TASK_EVAL_LOG"
fi

echo '=== finetune artifact counts ==='
TASK_FINETUNE_JSON_COUNT=0
TASK_FINETUNE_LOG_COUNT=0
TASK_FINETUNE_MODEL_COUNT=0
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    TASK_FINETUNE_JSON_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' | wc -l)
    TASK_FINETUNE_LOG_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | wc -l)
fi
if [[ -d "$TASK_FINETUNE_OUTPUT/models" ]]; then
    TASK_FINETUNE_MODEL_COUNT=$(find "$TASK_FINETUNE_OUTPUT/models" -mindepth 2 -maxdepth 2 \
        -type f -name model_state.pt | wc -l)
fi
echo "finetune_json=$TASK_FINETUNE_JSON_COUNT finetune_logs=$TASK_FINETUNE_LOG_COUNT finetune_models=$TASK_FINETUNE_MODEL_COUNT"
if [[ -s "$TASK_FINETUNE_OUTPUT/summary.md" ]]; then
    echo 'finetune_summary=present'
    cat "$TASK_FINETUNE_OUTPUT/summary.md"
else
    echo 'finetune_summary=pending'
fi

echo '=== active finetune commands ==='
mapfile -t TASK_ACTIVE_FINETUNE_PIDS < <(
    pgrep -f 'finetune/train.py' || true
)
for TASK_PID in "${TASK_ACTIVE_FINETUNE_PIDS[@]}"; do
    [[ -r "/proc/$TASK_PID/cmdline" ]] || continue
    printf 'pid=%s cmd=' "$TASK_PID"
    tr '\0' ' ' < "/proc/$TASK_PID/cmdline"
    printf '\n'
done

echo '=== tails of incomplete finetune logs ==='
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    while IFS= read -r TASK_LOG; do
        TASK_JSON="${TASK_LOG%.log}.json"
        [[ -s "$TASK_JSON" ]] && continue
        echo "--- $TASK_LOG bytes=$(stat -c %s "$TASK_LOG") modified=$(stat -c %y "$TASK_LOG") ---"
        tr '\r' '\n' < "$TASK_LOG" | tail -n 12
    done < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | sort)
fi

echo '=== narrow fatal-signature scan ==='
TASK_FATAL_PATTERN='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|FAILED \(code|eval failed:|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
TASK_FATAL_FOUND=0
for TASK_LOG in "$TASK_EVAL_LOG" "$TASK_RSE_CKPT/eval.log" "$TASK_HASHED_CKPT/eval.log"; do
    if [[ -s "$TASK_LOG" ]] && grep -HniE "$TASK_FATAL_PATTERN" "$TASK_LOG"; then
        TASK_FATAL_FOUND=1
    fi
done
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    while IFS= read -r TASK_LOG; do
        if grep -HniE "$TASK_FATAL_PATTERN" "$TASK_LOG"; then
            TASK_FATAL_FOUND=1
        fi
    done < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' -print)
fi
[[ "$TASK_FATAL_FOUND" -eq 0 ]]
echo 'NO_FATAL_SIGNATURE_FOUND'
echo 'TH2 LIVE RSE HASHED EVAL FINETUNE AUDIT COMPLETE; WORKFLOW UNMODIFIED'

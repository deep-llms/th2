#1 +30+a
#th2-check-four-model-workflow-completion-and-burns-20260830-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RUNTIME=/tmp/run_phase1_four_model_eval_finetune_burn_2949bab98ea9.sh
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_nested_groupreduce_two_dense_10k_20260830"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_nested_groupreduce_two_dense_10k_20260830.log"
TASK_CHECKPOINTS=(
    "$TASK_OUTPUT_BASE/nested_ladder_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default/checkpoint-10000"
)
TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11

echo '=== exact process and GPU stage audit (read-only) ==='
date -u
hostname
test -x "$TASK_PYTHON"
TASK_PROCESS_REPORT=$(
    "$TASK_PYTHON" - "$TASK_RUNTIME" "$TASK_FINETUNE_OUTPUT" "${TASK_CHECKPOINTS[@]}" <<'PY'
import os
import subprocess
import sys

runtime, finetune_output, *checkpoints = sys.argv[1:]


def read_args(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.rstrip(b"\0").split(b"\0") if part]


def read_ppid(pid):
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
    args = read_args(pid)
    if args:
        processes[pid] = args
        parents[pid] = read_ppid(pid)

workflow = []
eval_launchers = []
eval_workers = []
finetune_launchers = []
finetune_processes = []
burn_workers = []
for pid, args in processes.items():
    executable = os.path.basename(args[0])
    if executable in {"bash", "sh"} and len(args) >= 2 and args[1] == runtime:
        workflow.append(pid)
    if has_script(args, "eval/eval_parallel.py") and all(path in args for path in checkpoints):
        eval_launchers.append(pid)
    if has_script(args, "eval/eval_checkpoint.py") and any(path in args for path in checkpoints):
        eval_workers.append(pid)
    if has_script(args, "finetune/run_all.py") and finetune_output in args:
        finetune_launchers.append(pid)
    if (
        has_script(args, "finetune/train.py")
        and finetune_output in args
        and any(path in args for path in checkpoints)
    ):
        finetune_processes.append(pid)
    if has_script(args, "scripts/gpu_burn.py"):
        burn_workers.append(pid)

assert len(workflow) == 1, f"expected one workflow parent, found {workflow}"
if len(finetune_launchers) == 1:
    finetune_trainers = [
        pid for pid in finetune_processes if parents.get(pid) == finetune_launchers[0]
    ]
else:
    finetune_trainers = []
finetune_helpers = [pid for pid in finetune_processes if pid not in finetune_trainers]

stage_groups = (
    bool(eval_launchers or eval_workers),
    bool(finetune_launchers or finetune_processes),
    bool(burn_workers),
)
assert sum(stage_groups) <= 1, "eval, finetune, and burn stages overlap"

gpu_output = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True,
)
gpu_pids = sorted({int(line.strip()) for line in gpu_output.splitlines() if line.strip()})
known_gpu_pids = set(eval_workers) | set(finetune_trainers) | set(burn_workers)
unknown = [pid for pid in gpu_pids if pid not in known_gpu_pids]
assert not unknown, f"unexpected GPU processes: {[(pid, processes.get(pid)) for pid in unknown]}"

if eval_launchers or eval_workers:
    stage = "eval"
    assert len(eval_launchers) == 1, eval_launchers
    assert 1 <= len(eval_workers) <= 4, eval_workers
    assert set(gpu_pids) == set(eval_workers), (gpu_pids, eval_workers)
elif finetune_launchers or finetune_processes:
    stage = "finetune"
    assert len(finetune_launchers) == 1, finetune_launchers
    assert 1 <= len(finetune_trainers) <= 8, finetune_trainers
    assert all(parents.get(pid) in finetune_trainers for pid in finetune_helpers)
    assert set(gpu_pids) == set(finetune_trainers), (gpu_pids, finetune_trainers)
elif burn_workers:
    stage = "burn"
    assert len(burn_workers) == 8, burn_workers
    assert all(parents.get(pid) == workflow[0] for pid in burn_workers), (
        burn_workers,
        {pid: parents.get(pid) for pid in burn_workers},
        workflow,
    )
    assert set(gpu_pids) == set(burn_workers), (gpu_pids, burn_workers)
    mapping = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    per_uuid = {}
    for line in mapping.splitlines():
        if not line.strip():
            continue
        uuid, raw_pid = [part.strip() for part in line.split(",", 1)]
        pid = int(raw_pid)
        assert pid in burn_workers, (uuid, pid)
        per_uuid[uuid] = per_uuid.get(uuid, 0) + 1
    assert len(per_uuid) == 8 and set(per_uuid.values()) == {1}, per_uuid
else:
    stage = "validated_handoff_or_wait"
    assert not gpu_pids, gpu_pids

print(f"WORKFLOW_PID={workflow[0]}")
print(f"STAGE={stage}")
print(f"EVAL_LAUNCHERS={len(eval_launchers)} EVAL_WORKERS={len(eval_workers)}")
print(
    f"FINETUNE_LAUNCHERS={len(finetune_launchers)} "
    f"FINETUNE_TRAINERS={len(finetune_trainers)} "
    f"FINETUNE_DATALOADER_HELPERS={len(finetune_helpers)}"
)
print(f"BURN_WORKERS={len(burn_workers)} GPU_PIDS={' '.join(map(str, gpu_pids)) or '<none>'}")
print("PROCESS_AND_GPU_STAGE_OK")
PY
)
printf '%s\n' "$TASK_PROCESS_REPORT"
TASK_STAGE=$(printf '%s\n' "$TASK_PROCESS_REPORT" | sed -n 's/^STAGE=//p')
test -n "$TASK_STAGE"

echo '=== validate all four completed evaluations ==='
test -s "$TASK_EVAL_LAUNCH_LOG"
grep -F 'All 4 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
"$TASK_PYTHON" - "${TASK_CHECKPOINTS[@]}" <<'PY'
import json
import math
import os
import sys

for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == {"en", "vi", "zh", "ru", "de", "ar"}, checkpoint
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for metrics in perplexity.values():
        assert int(metrics["num_tokens"]) > 0
        assert math.isfinite(float(metrics["loss"]))
        assert math.isfinite(float(metrics["perplexity"]))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get("acc,none", metrics.get("acc"))
        assert accuracy is not None and math.isfinite(float(accuracy)), (task, metrics)
print("FOUR_EVALUATIONS_COMPLETE_OK models=4 ppl_languages=6 benchmark_tasks=26")
PY

echo '=== finetune output status ==='
TASK_JSON_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
TASK_LOG_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' 2>/dev/null | wc -l)
TASK_MODEL_COUNT=$(find "$TASK_FINETUNE_OUTPUT/models" -mindepth 2 -maxdepth 2 \
    -type f -name model_state.pt 2>/dev/null | wc -l)
echo "FINETUNE_COUNTS json=$TASK_JSON_COUNT logs=$TASK_LOG_COUNT models=$TASK_MODEL_COUNT"
if [[ -s "$TASK_FINETUNE_OUTPUT/summary.md" ]]; then
    [[ "$TASK_JSON_COUNT" -eq 36 ]]
    [[ "$TASK_LOG_COUNT" -eq 36 ]]
    [[ "$TASK_MODEL_COUNT" -eq 36 ]]
    grep -Fxq '# Fine-tune Benchmark Results (Generative)' "$TASK_FINETUNE_OUTPUT/summary.md"
    grep -Fxq '## hellaswag' "$TASK_FINETUNE_OUTPUT/summary.md"
    grep -Fxq '## arc_easy' "$TASK_FINETUNE_OUTPUT/summary.md"
    grep -Fxq '## xnli' "$TASK_FINETUNE_OUTPUT/summary.md"
    echo 'FOUR_MODEL_FINETUNE_ARTIFACTS_COMPLETE jobs=36'
    cat "$TASK_FINETUNE_OUTPUT/summary.md"
else
    echo 'FINETUNE_SUMMARY_PENDING'
fi

if [[ "$TASK_STAGE" == burn ]]; then
    test -s "$TASK_FINETUNE_OUTPUT/summary.md"
    [[ "$TASK_JSON_COUNT" -eq 36 && "$TASK_LOG_COUNT" -eq 36 && "$TASK_MODEL_COUNT" -eq 36 ]]
    echo 'WORKFLOW_COMPLETE_AND_EIGHT_SUPERVISED_BURNS_ACTIVE'
else
    echo "WORKFLOW_NOT_YET_AT_BURN_STAGE current_stage=$TASK_STAGE"
fi

echo '=== narrow fatal-signature scan ==='
TASK_FATAL_PATTERN='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|FAILED \(code|eval failed:|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
TASK_FATAL_FOUND=0
if grep -HniE "$TASK_FATAL_PATTERN" "$TASK_EVAL_LAUNCH_LOG" "${TASK_CHECKPOINTS[@]/%//eval.log}"; then
    TASK_FATAL_FOUND=1
fi
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    while IFS= read -r TASK_LOG; do
        if grep -HniE "$TASK_FATAL_PATTERN" "$TASK_LOG"; then
            TASK_FATAL_FOUND=1
        fi
    done < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' -print)
fi
[[ "$TASK_FATAL_FOUND" -eq 0 ]] || {
    echo 'ERROR: fatal signature found; active workflow was not modified' >&2
    exit 1
}
echo 'NO_FATAL_SIGNATURE_FOUND'

echo '=== current GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
echo 'TH2 FOUR-MODEL WORKFLOW COMPLETION/BURN STATUS CHECK FINISHED; RUN UNMODIFIED'

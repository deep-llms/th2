#1 +30+a
#th2-audit-live-four-model-eval-finetune-workflow-20260830-a01
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
TASK_AUDIT_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11

echo '=== read-only audit of active four-model workflow ==='
date -u
hostname
test -x "$TASK_AUDIT_PYTHON"

"$TASK_AUDIT_PYTHON" - \
    "$TASK_RUNTIME" "$TASK_FINETUNE_OUTPUT" "${TASK_CHECKPOINTS[@]}" <<'PY'
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


def has_script(args, suffix):
    return any(arg == suffix or arg.endswith("/" + suffix) for arg in args[1:])


processes = {}
for entry in os.scandir("/proc"):
    if entry.name.isdigit():
        pid = int(entry.name)
        args = read_args(pid)
        if args:
            processes[pid] = args

workflow = []
eval_launchers = []
eval_workers = []
finetune_launchers = []
finetune_workers = []
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
        finetune_workers.append(pid)
    if has_script(args, "scripts/gpu_burn.py"):
        burn_workers.append(pid)

groups = {
    "workflow": workflow,
    "eval_launcher": eval_launchers,
    "eval_worker": eval_workers,
    "finetune_launcher": finetune_launchers,
    "finetune_worker": finetune_workers,
    "burn_worker": burn_workers,
}
for label, pids in groups.items():
    print(f"PROCESS_GROUP label={label} count={len(pids)} pids={' '.join(map(str, sorted(pids))) or '<none>'}")
    for pid in sorted(pids):
        print(f"  pid={pid} argv={processes[pid]}")

assert len(workflow) == 1, f"expected exactly one workflow parent, found {workflow}"
active_groups = sum(bool(group) for group in (eval_launchers or eval_workers, finetune_launchers or finetune_workers, burn_workers))
assert active_groups <= 1, "eval, finetune, and burn stages overlap"
if eval_workers:
    assert len(eval_launchers) == 1, eval_launchers
    assert 1 <= len(eval_workers) <= 4, eval_workers
if finetune_workers:
    assert len(finetune_launchers) == 1, finetune_launchers
    assert 1 <= len(finetune_workers) <= 8, finetune_workers
if burn_workers:
    assert len(burn_workers) <= 8, burn_workers

gpu_output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
gpu_pids = sorted({int(line.strip()) for line in gpu_output.splitlines() if line.strip()})
known_gpu_pids = set(eval_workers) | set(finetune_workers) | set(burn_workers)
unknown = [pid for pid in gpu_pids if pid not in known_gpu_pids]
assert not unknown, f"unexpected GPU processes: {[(pid, processes.get(pid)) for pid in unknown]}"
if eval_workers:
    assert set(gpu_pids) == set(eval_workers), (gpu_pids, eval_workers)
if finetune_workers:
    assert set(gpu_pids) == set(finetune_workers), (gpu_pids, finetune_workers)
if burn_workers:
    assert set(gpu_pids) == set(burn_workers), (gpu_pids, burn_workers)

if eval_launchers or eval_workers:
    stage = "eval"
elif finetune_launchers or finetune_workers:
    stage = "finetune"
elif burn_workers:
    stage = "burn"
else:
    stage = "validated_handoff_or_wait"
print(f"LIVE_PROCESS_AUDIT_OK stage={stage} gpu_pids={gpu_pids}")
PY

echo '=== live GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== evaluation launcher and per-checkpoint progress ==='
test -s "$TASK_EVAL_LAUNCH_LOG"
tail -80 "$TASK_EVAL_LAUNCH_LOG"
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    echo "--- checkpoint=$TASK_CHECKPOINT ---"
    test -s "$TASK_CHECKPOINT/eval.log"
    stat --printf='eval_log size=%s modified=%y\n' "$TASK_CHECKPOINT/eval.log"
    tr '\r' '\n' < "$TASK_CHECKPOINT/eval.log" | tail -35
    for TASK_JSON in eval_ppl.json eval_benchmarks.json; do
        if [[ -s "$TASK_CHECKPOINT/$TASK_JSON" ]]; then
            stat --printf='artifact=%n size=%s modified=%y\n' "$TASK_CHECKPOINT/$TASK_JSON"
        else
            echo "PENDING $TASK_CHECKPOINT/$TASK_JSON"
        fi
    done
done

echo '=== narrow fatal-signature scan; read-only ==='
TASK_FATAL_PATTERN='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|FAILED \(code|eval failed:|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
TASK_FATAL_FOUND=0
if grep -HniE "$TASK_FATAL_PATTERN" "$TASK_EVAL_LAUNCH_LOG"; then
    TASK_FATAL_FOUND=1
fi
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    if grep -HniE "$TASK_FATAL_PATTERN" "$TASK_CHECKPOINT/eval.log"; then
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
[[ "$TASK_FATAL_FOUND" -eq 0 ]] || {
    echo 'ERROR: fatal signature found; active workflow was not modified' >&2
    exit 1
}
echo 'NO_FATAL_SIGNATURE_FOUND'
echo 'TH2 LIVE FOUR-MODEL WORKFLOW AUDIT COMPLETE; ACTIVE RUN UNMODIFIED'

#1 +60+a
#th2-ensure-eight-runner-burns-20260902-a01
set -euo pipefail

TASK_BURN_SCRIPT=/tmp/llm_pretrain_burn.py
test -s "$TASK_BURN_SCRIPT"
grep -F 'init_process_group' "$TASK_BURN_SCRIPT"
grep -F 'all_reduce' "$TASK_BURN_SCRIPT"

echo '=== preflight: machine and current GPU ownership ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)

if [[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]]; then
    echo 'No GPU compute process is active; starting the runner burn on all eight GPUs.'
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 "$TASK_BURN_SCRIPT" &
    sleep 15
else
    echo "Found ${#TASK_GPU_PIDS[@]} existing GPU compute process(es); validating ownership without killing anything."
    TASK_PID_CSV="$(IFS=,; echo "${TASK_GPU_PIDS[*]}")"
    python3 - "$TASK_PID_CSV" "$TASK_BURN_SCRIPT" <<'PY'
from pathlib import Path
import sys

pids = [int(value) for value in sys.argv[1].split(",") if value]
needle = sys.argv[2]

def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""

def parent(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[3])
    except (OSError, ValueError, IndexError):
        return 0

unexpected = []
for pid in pids:
    chain = []
    current = pid
    seen = set()
    owned = False
    while current > 1 and current not in seen:
        seen.add(current)
        command = cmdline(current)
        chain.append((current, command))
        if needle in command:
            owned = True
        current = parent(current)
    print(f"gpu_pid={pid} burn_owned={owned} ancestry={chain}")
    if not owned:
        unexpected.append(pid)

if unexpected:
    raise SystemExit(f"REFUSE TO START BURN: unexpected GPU process(es): {unexpected}")
if len(pids) != 8:
    raise SystemExit(f"REFUSE TO MODIFY PARTIAL BURN: expected 8 workers, found {len(pids)}")
print("EXISTING_EIGHT_GPU_BURNS_VERIFIED; NO NEW BURN STARTED")
PY
fi

echo '=== postflight: require one active compute worker per B200 ==='
for TASK_SAMPLE in 1 2 3; do
    echo "sample=$TASK_SAMPLE"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    sleep 3
done

mapfile -t TASK_FINAL_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
test "${#TASK_FINAL_PIDS[@]}" -eq 8
mapfile -t TASK_FINAL_APPS < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits
)
printf '%s\n' "${TASK_FINAL_APPS[@]}"
test "${#TASK_FINAL_APPS[@]}" -eq 8
TASK_FINAL_UUID_COUNT="$(printf '%s\n' "${TASK_FINAL_APPS[@]}" | cut -d, -f1 | sort -u | wc -l)"
test "$TASK_FINAL_UUID_COUNT" -eq 8

echo 'TH2 EIGHT-GPU RUNNER BURN ACTIVE'

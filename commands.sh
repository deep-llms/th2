#1 +60+a
#th2-swt-start-verified-communicating-burn-after-final5k-20260909-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt/bin/python
TASK_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_SESSION=swt_idle_burn_after_final5k_20260909_a01
TASK_PORT=29547
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/burn_after_final5k_20260909_a01
TASK_LOG="$TASK_ROOT/burn.log"
date -u
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
test -s "$TASK_BURN"
test "$(sha256sum "$TASK_BURN" | awk '{print $1}')" = "$(sha256sum resources/llm_pretrain_burn.py | awk '{print $1}')"
test ! -e "$TASK_ROOT"
test ! -L "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    echo 'Refusing duplicate burn session' >&2
    exit 1
fi
"$TASK_PYTHON" - "$TASK_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
print('BURN_PORT_FREE', sys.argv[1])
PY
mkdir -p "$TASK_ROOT"
printf -v TASK_COMMAND 'exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_ADDR=127.0.0.1 MASTER_PORT=%q GPU_BURN_MEMORY_FRACTION=0.85 GPU_BURN_MIN_FREE_GIB=8 GPU_BURN_COMM_TOTAL_MIB=1137 GPU_BURN_COMM_BUCKET_MIB=25 GPU_BURN_APPROX_STEP_SECONDS=0.75 GPU_BURN_MIN_WORLD_SIZE=2 GPU_BURN_MATRIX_SIZE=8192 GPU_BURN_CALIBRATION_GEMMS=64 GPU_BURN_PROGRESS_EVERY=10 %q -u %q >%q 2>&1' \
    "$TASK_PORT" "$TASK_PYTHON" "$TASK_BURN" "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
"$TASK_PYTHON" - "$TASK_LOG" "$TASK_BURN" "$TASK_ROOT" <<'PY'
import json, re, sys, time
from pathlib import Path
from capacity_allocation.data import write_json
from scripts.reclaim_verified_burn import verified_workers
from scripts.gpu_status import snapshot
log, burn, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for _ in range(60):
    text = log.read_text() if log.exists() else ''
    ready = [line for line in text.splitlines() if line.startswith('gpu_burn_ready ')]
    progress = [line for line in text.splitlines() if line.startswith('gpu_burn_progress ')]
    if len(ready) == 8 and len(progress) >= 2:
        assert {int(re.search(r'\brank=(\d+)', line).group(1)) for line in ready} == set(range(8))
        assert all('world_size=8' in line and 'collective_probe_sum=36' in line for line in ready)
        values = [[float(re.search(r'\b'+key+r'=([\d.]+)', line).group(1))
                   for key in ('completed_cycles','completed_collective_payload_gib')]
                  for line in progress[-2:]]
        assert all(b > a > 0 for a,b in zip(values[0], values[1]))
        workers, launcher = verified_workers(burn, list(range(8)))
        assert len(workers) == 8
        status = snapshot()
        assert all(0.83 < gpu['memory_used_mib']/gpu['memory_total_mib'] < 0.87 for gpu in status)
        write_json(root/'burn_verified.json', dict(success=True, launcher=launcher['pid'],
            workers=[w['pid'] for w in workers], gpus=status, progress=progress[-2:]))
        print('ALL_EIGHT_BURNS_MEMORY_AND_COLLECTIVE_PROGRESS_VERIFIED')
        print(json.dumps(status))
        print('\n'.join(progress[-2:]))
        break
    if 'Traceback (most recent call last)' in text:
        raise RuntimeError('Burn failed; inspect retained log')
    time.sleep(5)
else:
    raise RuntimeError('Burn did not establish verified memory and collectives within 300 seconds')
PY
echo SWT_VERIFIED_COMMUNICATING_BURN_RUNNING

#1 +300+a
#th2-swt-next-capacity-eval98-diag294-ft126-20260910-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2

TASK_RUN_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_eval98_diag294_ft126_20260910_a02
TASK_FAILED_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_eval98_diag294_ft126_20260910_a01
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_PYTHON=/usr/bin/python3
TASK_BURN_PORT=29584
TASK_BURN_SESSION=swt_burn_next_capacity_eval_20260910_a02
TASK_GPUS=(0 1 2 3 4 5 6 7)

source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
TASK_SWT_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
test "$(command -v python)" = "$TASK_SWT_PYTHON"
test ! -e "$TASK_RUN_ROOT"
test -s "$TASK_FAILED_ROOT/pipeline.log"
grep -Fq 'SWT_NEXT_EVAL_FINETUNE_FAILED exit=1; subsequent stages not started' "$TASK_FAILED_ROOT/pipeline.log"
test ! -e "$TASK_FAILED_ROOT/inputs_verified.json"
test ! -e "$TASK_FAILED_ROOT/eval_plan.json"
test ! -e "$TASK_FAILED_ROOT/finetune_plan.json"
test -s "$TASK_BURN"
test -x "$TASK_BURN_PYTHON"
command -v tmux >/dev/null
echo '2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174  /tmp/llm_pretrain_burn.py' | sha256sum -c -
"$TASK_SWT_PYTHON" scripts/verify_manifest.py verify --root . \
    --manifest resources/next_capacity_eval_source_20260910.json
"$TASK_SWT_PYTHON" scripts/verify_manifest.py verify \
    --root /mnt/local/_data/deep-llms_th2/benchmarks/hf \
    --manifest resources/english_core_benchmark_files_20260908.json
test -s /mnt/local/_data/deep-llms_th2/swt/diagnostics/qwen_en_pool_20260909_a01/manifest.json
test -s /mnt/local/_outputs/deep-llms_th2/swt/next_capacity_14arms_5k_s42_20260909_a01/training_complete.json
"$TASK_SWT_PYTHON" - "$TASK_RUN_ROOT" <<'PY'
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
usage = os.statvfs(root.parent)
free = usage.f_bavail * usage.f_frsize
assert free >= 250 * 1024**3, f'insufficient free output space: {free/1024**3:.1f} GiB'
print(f'OUTPUT_SPACE_PREFLIGHT_PASSED free_gib={free/1024**3:.1f}', flush=True)
PY

"$TASK_SWT_PYTHON" - "$TASK_BURN_PORT" <<'PY'
import socket, sys
from scripts.gpu_status import snapshot
status = snapshot()
assert len(status) == 8 and all('B200' in row['name'] for row in status)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
print('EVAL_HOST_AND_BURN_PORT_PREFLIGHT_PASSED', flush=True)
PY

# Accept only a free node or the known verified all-GPU burn. The reclaim helper
# refuses unrelated compute processes instead of signaling them.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
    "$TASK_SWT_PYTHON" -u -m scripts.reclaim_verified_burn \
        --burn-path "$TASK_BURN" --gpus "${TASK_GPUS[@]}" --stop
else
    echo 'NO_GPU_COMPUTE_PROCESSES_TO_RECLAIM'
fi
sleep 30
"$TASK_SWT_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free

conda activate swt_eval
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/swt_eval/bin/python
test "$(command -v python)" = "$TASK_EVAL_PYTHON"
TASK_ACCELERATE_TARGET=$("$TASK_EVAL_PYTHON" -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
"$TASK_EVAL_PYTHON" - "$TASK_ACCELERATE_TARGET" <<'PY'
import sys
from accelerate.commands.config.config_args import load_config_from_file
config = load_config_from_file(sys.argv[1])
assert config.num_processes == 8 and config.mixed_precision == 'bf16'
assert config.distributed_type.value == 'MULTI_GPU' and not config.use_cpu
print('ACCELERATE_CONFIG_COPIED_AND_VERIFIED', sys.argv[1], flush=True)
PY
sleep 30
"$TASK_EVAL_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free

bash scripts/eval_diagnostics_finetune_next_capacity_b200.sh "$TASK_RUN_ROOT"

sleep 30
"$TASK_EVAL_PYTHON" scripts/gpu_status.py --gpus "${TASK_GPUS[@]}" --require-free
test -s "$TASK_RUN_ROOT/complete.json"
if tmux has-session -t "$TASK_BURN_SESSION" 2>/dev/null; then
    echo 'Burn session already exists; refusing duplicate' >&2
    exit 1
fi
"$TASK_EVAL_PYTHON" - "$TASK_BURN_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
print('POST_EVAL_BURN_PORT_FREE', flush=True)
PY

TASK_BURN_LOG="$TASK_RUN_ROOT/burn.log"
printf -v TASK_BURN_COMMAND 'exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_ADDR=127.0.0.1 MASTER_PORT=%q GPU_BURN_MEMORY_FRACTION=0.85 GPU_BURN_MIN_FREE_GIB=8 GPU_BURN_COMM_TOTAL_MIB=1137 GPU_BURN_COMM_BUCKET_MIB=25 GPU_BURN_APPROX_STEP_SECONDS=0.75 GPU_BURN_MIN_WORLD_SIZE=2 GPU_BURN_MATRIX_SIZE=8192 GPU_BURN_CALIBRATION_GEMMS=64 GPU_BURN_PROGRESS_EVERY=10 %q -u %q >%q 2>&1' \
    "$TASK_BURN_PORT" "$TASK_BURN_PYTHON" "$TASK_BURN" "$TASK_BURN_LOG"
tmux new-session -d -s "$TASK_BURN_SESSION" "$TASK_BURN_COMMAND"
tmux set-option -w -t "$TASK_BURN_SESSION" remain-on-exit on

"$TASK_EVAL_PYTHON" - "$TASK_BURN_LOG" "$TASK_BURN" "$TASK_RUN_ROOT" <<'PY'
import re, sys, time
from pathlib import Path
from capacity_allocation.data import write_json
from scripts.reclaim_verified_burn import verified_workers
from scripts.gpu_status import snapshot
log = Path(sys.argv[1])
for _ in range(60):
    text = log.read_text() if log.exists() else ''
    ready = [line for line in text.splitlines() if line.startswith('gpu_burn_ready ')]
    progress = [line for line in text.splitlines() if line.startswith('gpu_burn_progress ')]
    if len(ready) == 8 and len(progress) >= 2:
        ranks = {int(re.search(r'\brank=(\d+)', line).group(1)) for line in ready}
        assert ranks == set(range(8))
        assert all('world_size=8' in line and 'collective_probe_sum=36' in line for line in ready)
        values = [[float(re.search(r'\b'+key+r'=([\d.]+)', line).group(1)) for key in
                   ('completed_cycles', 'completed_collective_payload_gib')]
                  for line in progress[-2:]]
        assert all(new > old > 0 for old, new in zip(values[0], values[1]))
        workers, launcher = verified_workers(sys.argv[2], list(range(8)))
        assert len(workers) == 8
        status = snapshot()
        assert all(0.83 < gpu['memory_used_mib']/gpu['memory_total_mib'] < 0.87 for gpu in status)
        write_json(Path(sys.argv[3])/'burn_verified.json', dict(success=True,
            launcher=launcher['pid'], workers=[row['pid'] for row in workers],
            gpus=status, progress=progress[-2:]))
        print('ALL_EIGHT_BURNS_MEMORY_AND_COLLECTIVE_PROGRESS_VERIFIED', flush=True)
        break
    if 'Traceback (most recent call last)' in text:
        raise RuntimeError('Burn failed; inspect its retained log')
    time.sleep(5)
else:
    raise RuntimeError('Burn did not establish verified communication/memory within 300 seconds')
PY

date -u
echo SWT_NEXT_CAPACITY_EVAL_FINETUNE_AND_BURN_HANDOFF_COMPLETE

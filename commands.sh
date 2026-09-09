#1 +60+a
#th2-swt-next-architecture-production-smoke-20260909-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt/bin/python
TASK_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
TASK_TORCHRUN=/mnt/local/conda-py311/envs/swt/bin/torchrun
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/next_architectures_smoke_20260909_a03
TASK_BURN_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/burn_after_next_architectures_smoke_20260909_a03
TASK_BURN_LOG="$TASK_BURN_ROOT/burn.log"
TASK_BURN_SESSION=swt_burn_after_next_architectures_smoke_20260909_a03
TASK_BURN_PORT=29561
TASK_ARMS=(T768 T512 P512-128-384 A640 A768 A768-Direct FixedResidual WNW D-1024 O1024-I256 O1024-I232 O1280 C-Direct D-Direct)
TASK_GPUS=(0 1 2 3 4 5 6 7)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID
test -x "$TASK_PYTHON"
test -x "$TASK_TORCHRUN"
test -s "$TASK_BURN"
test "$(sha256sum capacity_allocation/modeling.py | awk '{print $1}')" = cdaedf19db69c35a4bf58bef8a1b3224647b17359c0fa098d02277df2b26341c
test "$(sha256sum "$TASK_BURN" | awk '{print $1}')" = 2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
test "$(sha256sum resources/llm_pretrain_burn.py | awk '{print $1}')" = 2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174
test ! -e "$TASK_ROOT"
test ! -L "$TASK_ROOT"
test ! -e "$TASK_BURN_ROOT"
test ! -L "$TASK_BURN_ROOT"
if tmux has-session -t "$TASK_BURN_SESSION" 2>/dev/null; then
    echo 'Refusing duplicate burn session' >&2
    exit 1
fi
mkdir -p "$TASK_ROOT"
exec > >(tee "$TASK_ROOT/pipeline.log") 2>&1
date -u
hostname
"$TASK_PYTHON" - <<'PY'
import importlib.metadata as md
import torch
from capacity_allocation.modeling import NEW_ARMS, EXPECTED_COUNTS
assert md.version('transformers') == '5.9.0'
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
assert len(NEW_ARMS) == 14 and all(arm in EXPECTED_COUNTS for arm in NEW_ARMS)
print('DESTINATION_ENV_AND_ALL_NEW_ARMS_VERIFIED', torch.__version__, torch.version.cuda)
PY
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m unittest tests.test_capacity_models -v

# Reclaim only the eight workers descended from the exact reviewed burn path.
"$TASK_PYTHON" -m scripts.reclaim_verified_burn \
    --burn-path "$TASK_BURN" --gpus "${TASK_GPUS[@]}"
"$TASK_PYTHON" -m scripts.reclaim_verified_burn \
    --burn-path "$TASK_BURN" --gpus "${TASK_GPUS[@]}" --stop
sleep 30
"$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free

TASK_SMOKE_EXIT=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$TASK_TORCHRUN" \
    --standalone --nproc_per_node=8 -m scripts.smoke_capacity \
    --device cuda --precision bf16 --production \
    --batch-size 1 --sequence-length 2048 --arms "${TASK_ARMS[@]}" \
    --output "$TASK_ROOT/production_ddp_smoke.json" || TASK_SMOKE_EXIT=$?

if [ "$TASK_SMOKE_EXIT" -eq 0 ]; then
    set +e
    "$TASK_PYTHON" - "$TASK_ROOT/production_ddp_smoke.json" <<'PY'
import json, math, sys
from pathlib import Path
from capacity_allocation.modeling import NEW_ARMS, EXPECTED_COUNTS
path = Path(sys.argv[1])
result = json.loads(path.read_text())
assert result['result'] == 'PASS'
assert result['world_size'] == 8
assert result['precision'] == 'bf16'
assert result['tiny_models'] is False
assert result['batch_size'] == 1 and result['sequence_length'] == 2048
assert [row['arm'] for row in result['experiments']] == list(NEW_ARMS)
for row in result['experiments']:
    assert row['parameters']['total'] == EXPECTED_COUNTS[row['arm']]
    assert len(row['losses']) == 3 and all(math.isfinite(x) for x in row['losses'])
    assert row['peak_allocated_gib'] is not None and row['peak_allocated_gib'] > 0
    assert all(math.isfinite(x) for x in row['final_scales'].values())
print('ALL_14_PRODUCTION_BF16_EIGHT_RANK_SMOKES_VERIFIED')
PY
    TASK_VALIDATE_EXIT=$?
    set -e
    if [ "$TASK_VALIDATE_EXIT" -ne 0 ]; then
        TASK_SMOKE_EXIT=$TASK_VALIDATE_EXIT
    fi
fi

sleep 30
"$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free

# Restore the reviewed persistent eight-rank communicating burn after the test.
"$TASK_PYTHON" - "$TASK_BURN_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
print('BURN_PORT_FREE', sys.argv[1])
PY
mkdir -p "$TASK_BURN_ROOT"
printf -v TASK_BURN_COMMAND 'exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_ADDR=127.0.0.1 MASTER_PORT=%q GPU_BURN_MEMORY_FRACTION=0.85 GPU_BURN_MIN_FREE_GIB=8 GPU_BURN_COMM_TOTAL_MIB=1137 GPU_BURN_COMM_BUCKET_MIB=25 GPU_BURN_APPROX_STEP_SECONDS=0.75 GPU_BURN_MIN_WORLD_SIZE=2 GPU_BURN_MATRIX_SIZE=8192 GPU_BURN_CALIBRATION_GEMMS=64 GPU_BURN_PROGRESS_EVERY=10 %q -u %q >%q 2>&1' \
    "$TASK_BURN_PORT" "$TASK_PYTHON" "$TASK_BURN" "$TASK_BURN_LOG"
tmux new-session -d -s "$TASK_BURN_SESSION" "$TASK_BURN_COMMAND"
tmux set-option -w -t "$TASK_BURN_SESSION" remain-on-exit on
"$TASK_PYTHON" - "$TASK_BURN_LOG" "$TASK_BURN" "$TASK_BURN_ROOT" <<'PY'
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
                   for key in ('completed_cycles', 'completed_collective_payload_gib')]
                  for line in progress[-2:]]
        assert all(b > a > 0 for a, b in zip(values[0], values[1]))
        workers, launcher = verified_workers(burn, list(range(8)))
        assert len(workers) == 8
        status = snapshot()
        assert all(0.83 < gpu['memory_used_mib']/gpu['memory_total_mib'] < 0.87 for gpu in status)
        write_json(root/'burn_verified.json', dict(success=True, launcher=launcher['pid'],
            workers=[worker['pid'] for worker in workers], gpus=status,
            progress=progress[-2:]))
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

if [ "$TASK_SMOKE_EXIT" -ne 0 ]; then
    echo "PRODUCTION_SMOKE_FAILED exit=$TASK_SMOKE_EXIT; communicating burn restored" >&2
    exit "$TASK_SMOKE_EXIT"
fi
echo SWT_NEXT_ARCHITECTURES_PRODUCTION_SMOKE_COMPLETE_AND_BURN_RESTORED

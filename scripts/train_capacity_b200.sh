#!/usr/bin/env bash
# Authorized Qwen English pilot: CPU cache -> reclaim burn -> smoke -> train -> verify -> burn.
set -euo pipefail

TASK_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_PROJECT_DIR"
TASK_RUN_ROOT=${1:?Pass a fresh absolute run directory outside the project}
shift
TASK_ARMS=("$@")
[[ "${#TASK_ARMS[@]}" -gt 0 ]] || { echo 'Specify experiment arms' >&2; exit 1; }
[[ "$TASK_RUN_ROOT" == /mnt/local/_outputs/deep-llms_th2/swt/* ]] || exit 1
[[ ! -e "$TASK_RUN_ROOT" ]] || { echo 'Run root already exists; refusing overwrite' >&2; exit 1; }
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
[[ "$CONDA_DEFAULT_ENV" == swt ]]
TASK_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
[[ "$(command -v python)" == "$TASK_PYTHON" ]]
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TASK_DATA=/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B
TASK_TOKENIZER=/mnt/local/_models/deep-llms_th2/Qwen3-0.6B
TASK_CACHE=/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_PYTHON=/usr/bin/python3
TASK_TRAIN_PORT=29571
TASK_BURN_PORT=29573
TASK_GPUS=(0 1 2 3 4 5 6 7)
TASK_SESSION="swt_burn_$(basename "$TASK_RUN_ROOT")"
test -s "$TASK_BURN"
test -x "$TASK_BURN_PYTHON"
command -v tmux >/dev/null
"$TASK_PYTHON" - "$TASK_BURN" "${TASK_ARMS[@]}" <<'PY'
import hashlib, socket, sys
from pathlib import Path
from scripts.gpu_status import snapshot
from capacity_allocation.modeling import ARMS
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == '2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174'
arms = sys.argv[2:]
assert len(arms) == len(set(arms)) and all(arm in ARMS for arm in arms)
status = snapshot()
assert len(status) == 8 and all('B200' in row['name'] for row in status)
for port in (29571,29573):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',port))
print('HOST_AND_LAUNCH_PREFLIGHT_PASSED', flush=True)
PY
mkdir -p "$TASK_RUN_ROOT"
exec > >(tee "$TASK_RUN_ROOT/pipeline.log") 2>&1
trap 'TASK_EXIT=$?; echo "SWT_PIPELINE_FAILED exit=$TASK_EXIT; no success marker or burn handoff" >&2' ERR
printf '%s\n' "${TASK_ARMS[@]}" > "$TASK_RUN_ROOT/arms.txt"
date -u
hostname
echo "CPU_CACHE_PREPARATION; existing burns remain untouched"
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -u -m scripts.prepare_capacity_cache \
  --data-root "$TASK_DATA" --tokenizer "$TASK_TOKENIZER" --cache-dir "$TASK_CACHE" \
  --workers 160 --output "$TASK_RUN_ROOT/cache_ready.json"

# Revalidate actual burn ownership and PID start times immediately before signals.
"$TASK_PYTHON" -u -m scripts.reclaim_verified_burn --burn-path "$TASK_BURN" \
  --gpus "${TASK_GPUS[@]}" --stop
sleep 30
"$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free
TASK_ACCELERATE_TARGET=$("$TASK_PYTHON" -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
"$TASK_PYTHON" - "$TASK_ACCELERATE_TARGET" <<'PY'
import sys
from accelerate.commands.config.config_args import load_config_from_file
config = load_config_from_file(sys.argv[1])
assert config.num_processes == 8 and config.mixed_precision == 'bf16'
assert config.distributed_type.value == 'MULTI_GPU' and not config.use_cpu
print('ACCELERATE_CONFIG_COPIED_AND_VERIFIED',sys.argv[1], flush=True)
PY
sleep 30
"$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free

# Actual production dimensions, BF16, eight communicating ranks, batch 16 x sequence 2048.
"$TASK_PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port="$TASK_TRAIN_PORT" \
  -m scripts.smoke_capacity --device cuda --precision bf16 --production \
  --batch-size 16 --sequence-length 2048 --arms "${TASK_ARMS[@]}" \
  --output "$TASK_RUN_ROOT/production_smoke.json"
sleep 30
"$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free

for TASK_ARM in "${TASK_ARMS[@]}"; do
  echo "START_TRAINING_ARM $TASK_ARM"
  date -u
  "$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free
  accelerate launch --config_file "$TASK_ACCELERATE_TARGET" --main_process_port "$TASK_TRAIN_PORT" train.py \
    --arm "$TASK_ARM" --num_hidden_layers 6 --tokenizer_name "$TASK_TOKENIZER" \
    --data_dir "$TASK_DATA/train" --eval_data_dir "$TASK_DATA/eval" --languages en \
    --block_size 2048 --preprocessing_num_workers 160 --preprocessing_batch_size 1000 \
    --preprocessing_cache_dir "$TASK_CACHE" --output_dir "$TASK_RUN_ROOT/$TASK_ARM" \
    --num_train_epochs 1 --stop-at-step 10000 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 --bf16 --attn_implementation sdpa \
    --learning_rate 3e-4 --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' --warmup_steps 500 \
    --weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.95 --max_grad_norm 1 \
    --seed 42 --data_seed 42 --ddp_find_unused_parameters false --ddp_timeout 21600 \
    --save_steps 250 --logging_steps 10 --eval_strategy steps --eval_steps 1000 \
    --dataloader_num_workers 8 --report_to none --run_name "swt_${TASK_ARM}_qwen6_10k" \
    2>&1 | tee "$TASK_RUN_ROOT/train_${TASK_ARM}.log"
  sleep 30
  "$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free
  CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" -m scripts.verify_capacity_run \
    --run-dir "$TASK_RUN_ROOT/$TASK_ARM" --arm "$TASK_ARM" --step 10000 --world-size 8 \
    --cache-report "$TASK_RUN_ROOT/cache_ready.json" --output "$TASK_RUN_ROOT/verified_${TASK_ARM}.json"
done

"$TASK_PYTHON" - "$TASK_RUN_ROOT" "${TASK_ARMS[@]}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from capacity_allocation.data import write_json
root = Path(sys.argv[1])
results = [json.loads((root/f'verified_{arm}.json').read_text()) for arm in sys.argv[2:]]
assert all(row['success'] and row['step']==10000 for row in results)
write_json(root/'training_complete.json', dict(success=True, completed_utc=datetime.now(timezone.utc).isoformat(), experiments=results))
print('ALL_TRAINING_COMPLETED_AND_VERIFIED', flush=True)
PY
sleep 30
"$TASK_PYTHON" -m scripts.gpu_status --gpus "${TASK_GPUS[@]}" --require-free
# Persistent own tmux terminal; keep stdin attached. Never background under the exiting runner shell.
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'Burn session already exists; refusing duplicate' >&2
  exit 1
fi
"$TASK_PYTHON" - "$TASK_BURN_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1',int(sys.argv[1])))
PY
TASK_BURN_LOG="$TASK_RUN_ROOT/burn.log"
printf -v TASK_BURN_COMMAND 'exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_ADDR=127.0.0.1 MASTER_PORT=%q GPU_BURN_MEMORY_FRACTION=0.85 GPU_BURN_MIN_FREE_GIB=8 GPU_BURN_COMM_TOTAL_MIB=1137 GPU_BURN_COMM_BUCKET_MIB=25 GPU_BURN_APPROX_STEP_SECONDS=0.75 GPU_BURN_MIN_WORLD_SIZE=2 GPU_BURN_MATRIX_SIZE=8192 GPU_BURN_CALIBRATION_GEMMS=64 GPU_BURN_PROGRESS_EVERY=10 %q -u %q >%q 2>&1' \
  "$TASK_BURN_PORT" "$TASK_BURN_PYTHON" "$TASK_BURN" "$TASK_BURN_LOG"
tmux new-session -d -s "$TASK_SESSION" "$TASK_BURN_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
"$TASK_PYTHON" - "$TASK_BURN_LOG" "$TASK_BURN" "$TASK_RUN_ROOT" <<'PY'
import json, re, sys, time
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
        ranks = {int(re.search(r'\brank=(\d+)',line).group(1)) for line in ready}
        assert ranks == set(range(8))
        assert all('world_size=8' in line and 'collective_probe_sum=36' in line for line in ready)
        values = [[float(re.search(r'\b'+key+r'=([\d.]+)',line).group(1)) for key in
                   ('completed_cycles','completed_collective_payload_gib')] for line in progress[-2:]]
        assert all(b > a > 0 for a,b in zip(values[0],values[1]))
        workers, launcher = verified_workers(sys.argv[2], list(range(8)))
        assert len(workers) == 8
        status = snapshot()
        assert all(0.83 < gpu['memory_used_mib']/gpu['memory_total_mib'] < 0.87 for gpu in status)
        write_json(Path(sys.argv[3])/'burn_verified.json', dict(success=True, launcher=launcher['pid'],
            workers=[row['pid'] for row in workers], gpus=status, progress=progress[-2:]))
        print('ALL_EIGHT_BURNS_MEMORY_AND_COLLECTIVE_PROGRESS_VERIFIED', flush=True)
        break
    if 'Traceback (most recent call last)' in text:
        raise RuntimeError('Burn failed; inspect its retained log')
    time.sleep(5)
else:
    raise RuntimeError('Burn did not establish verified communication/memory within 300 seconds')
PY
date -u
echo SWT_TRAINING_AND_BURN_HANDOFF_COMPLETE

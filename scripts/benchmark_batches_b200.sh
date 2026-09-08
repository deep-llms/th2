#!/usr/bin/env bash
# Authorized batch benchmark only. Does not stop jobs, clean data, or resume training.
set -euo pipefail
TASK_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_PROJECT_DIR"
TASK_ROOT=${1:?Pass a fresh absolute output directory}
[[ "$TASK_ROOT" == /mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_* ]]
[[ ! -e "$TASK_ROOT" ]]
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TASK_PYTHON=/mnt/local/conda-py311/envs/swt/bin/python
test "$(command -v python)" = "$TASK_PYTHON"
python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
mkdir -p "$TASK_ROOT"
exec > >(tee "$TASK_ROOT/pipeline.log") 2>&1
TASK_ACCELERATE=$(python -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
mkdir -p "$(dirname "$TASK_ACCELERATE")"
cp resources/accelerate_config.yaml "$TASK_ACCELERATE"
cmp resources/accelerate_config.yaml "$TASK_ACCELERATE"
echo BATCH_BENCHMARK_ACCELERATE_COPIED_AND_VERIFIED
sleep 30
python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
python - <<'PY'
import socket
from accelerate.commands.config.config_args import load_config_from_file, default_yaml_config_file
config = load_config_from_file(default_yaml_config_file)
assert config.num_processes == 8 and config.mixed_precision == 'bf16'
assert config.distributed_type.value == 'MULTI_GPU'
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 29575))
PY
for TASK_ARM in B0 A128 A256 A512 C D; do
  for TASK_BATCH in 16 32; do
    TASK_ACCUM=$((64 / TASK_BATCH))
    TASK_NAME="${TASK_ARM}_b${TASK_BATCH}_a${TASK_ACCUM}"
    python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
    date -u
    echo "START_BATCH_BENCHMARK $TASK_NAME"
    if accelerate launch --config_file "$TASK_ACCELERATE" --main_process_port 29575 \
      -m scripts.benchmark_capacity_batch --arm "$TASK_ARM" \
      --batch-size "$TASK_BATCH" --accumulation "$TASK_ACCUM" \
      --warmup-updates 5 --measure-updates 20 \
      --data-root /mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B \
      --tokenizer /mnt/local/_models/deep-llms_th2/Qwen3-0.6B \
      --cache-dir /mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000 \
      --cache-report /mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02/cache_ready.json \
      --output-dir "$TASK_ROOT/$TASK_NAME" > "$TASK_ROOT/$TASK_NAME.log" 2>&1; then
      python - "$TASK_ROOT/$TASK_NAME/result.json" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
assert result['success'] and not result['tiny'] and result['world_size'] == 8
assert result['effective_batch'] == 512 and result['measured_updates'] == 20
assert len(result['metrics']['ranks']) == 8
print('PROFILE_RESULT',result['arm'],result['batch_size'],
      result['metrics']['median_optimizer_step_s'],result['metrics']['peak_allocated_gib'],
      result['metrics']['peak_reserved_gib'],flush=True)
PY
    else
      TASK_EXIT=$?
      tail -n 50 "$TASK_ROOT/$TASK_NAME.log"
      # OOM is an expected possible benchmark outcome, never a usable config.
      # Any other failure stops this launcher for investigation.
      python - "$TASK_ROOT/$TASK_NAME.log" "$TASK_ROOT/${TASK_NAME}_oom.json" "$TASK_EXIT" <<'PY'
import json, sys
from pathlib import Path
from capacity_allocation.data import write_json
text = Path(sys.argv[1]).read_text()
if 'CUDA out of memory' not in text:
    raise RuntimeError('Non-OOM benchmark failure; inspect preserved log before continuing')
write_json(Path(sys.argv[2]), dict(success=False, outcome='cuda_oom', exit_code=int(sys.argv[3])))
print('PROFILE_OOM_RECORDED',sys.argv[1],flush=True)
PY
    fi
    sleep 30
    python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
  done
done
python - "$TASK_ROOT" <<'PY'
import json, sys
from pathlib import Path
from capacity_allocation.data import write_json
root = Path(sys.argv[1])
profiles = {}
for arm in ('B0','A128','A256','A512','C','D'):
    for batch, accum in ((16,4),(32,2)):
        name = f'{arm}_b{batch}_a{accum}'
        result, oom = root/name/'result.json', root/f'{name}_oom.json'
        if result.exists() == oom.exists():
            raise RuntimeError(f'Missing/ambiguous profile: {name}')
        profiles[name] = json.loads((result if result.exists() else oom).read_text())
write_json(root/'benchmark_complete.json', dict(completed=True,
    all_profiles_passed=all(row['success'] for row in profiles.values()), profiles=profiles))
print('ALL_BATCH_PROFILES_FINISHED',flush=True)
PY
date -u
echo BATCH_BENCHMARK_QUEUE_COMPLETE_GPUS_FREE

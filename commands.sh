#1 +120+a
#th2-swt-qwen6-fresh-six-arms-5k-then-burn-20260908-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
test "$PWD" = /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=offline
export SWT_STOP_AT_STEP=5000
echo 'SCREENING_CUTOFF=5000 EFFECTIVE_BATCH=512 INPUT_TOKENS_PER_ARM=5242880000 FULL_EPOCH_LR_SCHEDULE'
python scripts/verify_manifest.py verify --root "$PWD" --manifest resources/swt_qwen_launch_5k_20260908.json
python - <<'PY'
import os
from pathlib import Path
for name in (
    '/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a03',
    '/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01',
    '/mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000',
):
    assert not os.path.lexists(name), name
active_names = {'train.py', 'train_capacity_b200.sh', 'scripts.prepare_capacity_cache',
                'benchmark_batches_b200.sh', 'scripts.benchmark_capacity_batch'}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    except (FileNotFoundError, ProcessLookupError):
        continue
    assert not any(Path(arg).name in active_names for arg in args), (proc.name, args)
print('FRESH_PATHS_AND_NO_EXISTING_SWT_QUEUE_VERIFIED', flush=True)
PY
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
sleep 30
python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
TASK_ACCELERATE_TARGET=$(python -c 'from accelerate.commands.config.config_args import default_yaml_config_file; print(default_yaml_config_file)')
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
python - "$TASK_ACCELERATE_TARGET" <<'PY'
import sys
from accelerate.commands.config.config_args import load_config_from_file
config = load_config_from_file(sys.argv[1])
assert config.num_processes == 8 and config.mixed_precision == 'bf16'
assert config.distributed_type.value == 'MULTI_GPU' and not config.use_cpu
print('ACCELERATE_CONFIG_COPIED_AND_VERIFIED', sys.argv[1], flush=True)
PY
sleep 30
python -m scripts.gpu_status --gpus 0 1 2 3 4 5 6 7 --require-free
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p test_capacity_handoff.py -v
echo SWT_FRESH_SIX_ARM_QUEUE_STARTING
# Foreground tested workflow: regenerate English cache, recheck/copy Accelerate,
# production smoke, sequential six-arm training and validation, then persistent
# communicating burns. Any failed stage prevents the success/burn handoff.
bash scripts/train_capacity_b200.sh \
  /mnt/local/_outputs/@PROJECT@/swt/qwen6_allarms_5k_s42_20260908_a01 \
  B0 A128 A256 A512 C D

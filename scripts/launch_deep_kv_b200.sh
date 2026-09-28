#!/usr/bin/env bash
# Run inside an independent tmux session; stdout must go to a durable log.
set -euo pipefail
TASK_ROOT=$1
TASK_INSPECTION=$2
test "$(hostname)" = thiennh-p6-78gg-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TASK_SESSION=$(basename "$TASK_ROOT")
test ! -e "$TASK_ROOT"
mkdir -p "$TASK_ROOT"
python -u - "$TASK_ROOT" <<'PY'
import json, shutil, sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from deep_kv.__main__ import jobs
root = Path(sys.argv[1])
source = Path('resources/accelerate_config.yaml')
destination = Path(default_yaml_config_file)
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination)
assert source.read_bytes() == destination.read_bytes()
config = load_config_from_file(str(destination)).to_dict()
assert config['num_processes'] == 8 and config['mixed_precision'] == 'bf16'
assert config['distributed_type'] == 'MULTI_GPU'
recipe = json.loads(Path('deep_kv.b200.json').read_text())
for key, value in dict(max_steps=28600, warmup_steps=1430, stop_after=2500,
                       per_device_train_batch_size=16, gradient_accumulation_steps=4,
                       block_size=2048).items():
    assert recipe[key] == value, (key, recipe.get(key))
(root / 'recipe.json').write_text(json.dumps(recipe, indent=2))
(root / 'jobs.json').write_text(json.dumps(jobs('deep_kv.b200.json'), indent=2))
(root / 'failure-jobs.json').write_text(json.dumps({'jobs': [{
    'name': 'intentional-failure', 'argv': ['{python}', '-c', 'raise SystemExit(7)'],
    'required_outputs': [{'path': 'never-created.json'}]}]}, indent=2))
print('ACCELERATE_CONFIG_INSTALLED', destination, flush=True)
PY
accelerate env
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_train_then_burn -v
# Verify the crash path on the real GPUs before handing them to the long queue.
set +e
python -u -m scripts.train_then_burn --config "$TASK_ROOT/failure-jobs.json" \
  --output "$TASK_ROOT/rehearsal" --inspection "$TASK_INSPECTION" \
  --host thiennh-p6-78gg-worker-0 --burn-session "${TASK_SESSION}-rehearsal-burn"
TASK_REHEARSAL_RC=$?
set -e
test "$TASK_REHEARSAL_RC" -eq 1
sleep 20
python -u - "$TASK_ROOT" <<'PY'
import json, sys
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress, approved_launcher, BURN_HASH, process
root = Path(sys.argv[1]) / 'rehearsal'
receipt = json.loads((root / 'supervisor.json').read_text())
assert receipt['training_status'] == 'failed' and 'handoff_error' not in receipt
assert receipt['burn']['collective_progress_verified']
assert burn_progress((root / 'burn.log').read_text())
status = snapshot(list(range(8)))
assert all(len(g['pids']) == 1 for g in status)
source = str(Path('resources/llm_pretrain_burn.py').resolve())
assert all(approved_launcher(process(g['pids'][0])['ppid'], {source: BURN_HASH}) for g in status)
print('FAILURE_HANDOFF_VERIFIED_AFTER_SUPERVISOR_EXIT', json.dumps(status), flush=True)
PY
python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" \
  --output "$TASK_ROOT/production" --inspection "$TASK_INSPECTION" \
  --host thiennh-p6-78gg-worker-0 --burn-session "${TASK_SESSION}-final-burn"

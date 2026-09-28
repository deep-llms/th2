#1 +60+a
#th2-78gg-deep-kv-E-2500-20260928-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-78gg-worker-0
cd /mnt/local/@PROJECT@
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/deep-kv-E-2500-20260928-a02
TASK_SESSION=deep-kv-E-2500-20260928-a02
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    echo 'REFUSE: session already exists' >&2
    exit 1
fi
mkdir -p "$TASK_ROOT"
cat > "$TASK_ROOT/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-78gg-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" <<'PY'
import hashlib, importlib.metadata, json, shutil, sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from deep_kv import kv_loss_weight
from deep_kv.__main__ import jobs
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, process

root = Path(sys.argv[1])
previous = Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-2500-20260928-a02')
versions = {key: importlib.metadata.version(key) for key in ('torch', 'transformers', 'accelerate', 'datasets')}
assert versions == dict(torch='2.14.0', transformers='5.9.0', accelerate='1.13.0', datasets='4.8.5'), versions
import torch
assert torch.__version__ == '2.14.0+cu130', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
source = Path('resources/accelerate_config.yaml')
destination = Path(default_yaml_config_file)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    shutil.copy2(destination, root / 'previous_accelerate_config.yaml')
shutil.copy2(source, destination)
assert source.read_bytes() == destination.read_bytes()
config = load_config_from_file(str(destination)).to_dict()
assert config['num_processes'] == 8 and config['mixed_precision'] == 'bf16'
assert config['distributed_type'] == 'MULTI_GPU'
recipe = json.loads(Path('deep_kv.b200.json').read_text())
assert recipe == json.loads((previous / 'recipe.json').read_text()), 'Recipe changed from A-D'
for key, value in dict(max_steps=28600, warmup_steps=1430, stop_after=2500,
                       per_device_train_batch_size=16, gradient_accumulation_steps=4,
                       block_size=2048).items():
    assert recipe[key] == value, (key, recipe.get(key))
assert kv_loss_weight('E') == .3 and kv_loss_weight('D') == 1.
for key in ('config_name', 'tokenizer_name', 'data_dir', 'eval_data_dir'):
    assert Path(recipe[key]).exists(), (key, recipe[key])
inspection = inspect()
assert inspection['host'] == 'thiennh-p6-78gg-worker-0'
assert not inspection['guard_disabled'], 'Another workload holds the GPU guard'
old_gpus = json.loads((previous / 'production/burn-verified.json').read_text())['gpus']
assert [(g['index'], g['uuid']) for g in inspection['gpus']] == [(g['index'], g['uuid']) for g in old_gpus]
approved = {**APPROVED_BURNS, str(Path('resources/llm_pretrain_burn.py').resolve()): BURN_HASH}
for pid in inspection['workers']:
    assert approved_launcher(pid, approved) or approved_launcher(process(pid)['ppid'], approved), pid
queue = jobs('deep_kv.b200.json', arms='E')
assert [j['name'] for j in queue['jobs']] == ['arm-E', 'compare']
for name, value in [('recipe.json', recipe), ('jobs.json', queue), ('gpu_inspection.json', inspection)]:
    (root / name).write_text(json.dumps(value, indent=2))
receipt = dict(versions=versions, accelerate_cache=str(destination),
               accelerate_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               recipe_matches_original_D=True, arm='E', kv_loss_weight=.3,
               inspected_at=inspection['time'], host=inspection['host'])
(root / 'preflight.json').write_text(json.dumps(receipt, indent=2))
print('E_PREFLIGHT_PASSED', json.dumps(receipt), flush=True)
PY
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" \
    --output "$TASK_ROOT/production" --inspection "$TASK_ROOT/gpu_inspection.json" \
    --host thiennh-p6-78gg-worker-0 --burn-session deep-kv-E-2500-20260928-a02-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 100 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

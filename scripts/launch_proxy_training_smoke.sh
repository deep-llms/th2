#!/usr/bin/env bash
# Disposable 50-step A/P6-iso fits through unmodified train.py on eight B200s.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=$2
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline WANDB_PROJECT=deep2shallow NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1042 TOKENIZERS_PARALLELISM=false
python -u - "$TASK_ROOT" "$TASK_HOST" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from deep_kv.__main__ import jobs
from run_experiments import load_jobs
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import APPROVED_BURNS, BURN_HASH, GUARD_HASH, digest, approved_launcher
from train import load_text

root = Path(sys.argv[1])
assert shutil.disk_usage(root).free > 200 * 2**30
recipe = json.loads(Path('proxy_heads.b200.json').read_text())
manifest = json.loads(Path('resources/qwen3_base_assets.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((Path(recipe['tokenizer_name']) / name).read_bytes()).hexdigest() == expected
source = Path('resources/accelerate_config.yaml')
destination = Path(default_yaml_config_file)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    shutil.copy2(destination, root / 'accelerate.previous.yaml')
shutil.copy2(source, destination)
assert source.read_bytes() == destination.read_bytes()
config = load_config_from_file(str(destination)).to_dict()
assert config['num_processes'] == 8 and config['mixed_precision'] == 'bf16'
assert config['distributed_type'] == 'MULTI_GPU'
subprocess.run(['accelerate', 'env'], check=True)
# These are disposable raw-text subsets. Tokenization/packing still happen in
# train.py, using its existing HF cache shared by both smoke arms.
for key, count, name in [('data_dir', 300000, 'train'), ('eval_data_dir', 2048, 'validation')]:
    dataset = load_text(recipe[key])
    assert len(dataset) >= count
    destination = root / 'text' / name
    dataset.select(range(count)).save_to_disk(str(destination))
    recipe[key] = str(destination)
    print('SMOKE_TEXT_READY', name, count, flush=True)
recipe.update(max_steps=100, warmup_steps=5, stop_after=50, proxy_warmup_steps=10,
              preprocessing_num_workers=16, eval_rows=128, monitor_rows=32,
              logging_steps=5, save_steps=25, save_total_limit=2, eval_steps=25,
              eval_on_start=False, dataloader_num_workers=2)
recipe_path = root / 'smoke_recipe.json'
with recipe_path.open('x') as handle:
    json.dump(recipe, handle, indent=2)
items = []
for job in jobs(recipe_path, stop_after=50, arms=('A', 'P6-iso'), seeds=[1042])['jobs']:
    if 'gpus' not in job:
        continue
    arm = job['argv'][job['argv'].index('--arm') + 1]
    if arm != 'A':
        job['argv'] += ['--proxy_module_seed', '1043']
    job['argv'] = ['env', 'PYTHONHASHSEED=1042', *job['argv']]
    job['timeout_seconds'] = 3600
    items.append(job)
    name = 'validate-' + arm
    items.append(dict(name=name, argv=['{python}', '-m', 'scripts.check_fa4_proxy', 'validate',
        '--run-dir', '{run_dir}', '--seed', '1042', '--arms', arm, '--steps', '50',
        '--schedule-steps', '100', '--warmup-steps', '5', '--output', '{run_dir}/' + name + '.json'],
        required_outputs=[dict(path=name+'.json', json_equals={'status':'passed'})]))
with (root / 'jobs.json').open('x') as handle:
    json.dump({'jobs':items}, handle, indent=2)
load_jobs(root / 'jobs.json')
record = inspect()
assert record['host'] == sys.argv[2] and not record['guard_disabled']
assert len(record['gpus']) == 8 and all('B200' in gpu['name'] for gpu in record['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh') == GUARD_HASH
approved = {**APPROVED_BURNS, str(Path.cwd() / 'resources/llm_pretrain_burn.py'): BURN_HASH}
for pid in record['workers']:
    assert approved_launcher(pid, approved) or approved_launcher(process(pid)['ppid'], approved), pid
with (root / 'inspection.json').open('x') as handle:
    json.dump(record, handle, indent=2)
print('FULL_MODEL_SMOKE_PREFLIGHT_PASSED', flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" \
  --output "$TASK_ROOT/supervised" --inspection "$TASK_ROOT/inspection.json" \
  --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

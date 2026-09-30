#1 +60+a
#th2-78gg-AB-seed123-2500-20260930-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-78gg-worker-0
cd /mnt/local/@PROJECT@
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/deep-kv-AB-seed123-2500-20260930-a01
TASK_SESSION=deep-kv-AB-seed123-2500-20260930-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
cat > "$TASK_ROOT/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=123
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" <<'PY'
import hashlib, importlib.metadata, json, shutil, sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from deep_kv.__main__ import jobs
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH, process
from run_experiments import load_jobs
root = Path(sys.argv[1])
for package, expected in [('torch','2.14.0'), ('transformers','5.9.0'), ('accelerate','1.13.0')]:
 assert importlib.metadata.version(package) == expected
assert shutil.disk_usage(root).free > 250 * 2**30
source = Path('resources/accelerate_config.yaml')
destination = Path(default_yaml_config_file)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists(): shutil.copy2(destination, root / 'previous_accelerate_config.yaml')
shutil.copy2(source, destination)
assert source.read_bytes() == destination.read_bytes()
config = load_config_from_file(str(destination)).to_dict()
assert config['num_processes'] == 8 and config['mixed_precision'] == 'bf16'
assert config['distributed_type'] == 'MULTI_GPU'
inspection = inspect()
assert inspection['host'] == 'thiennh-p6-78gg-worker-0' and not inspection['guard_disabled']
assert len(inspection['gpus']) == 8 and all('B200' in g['name'] for g in inspection['gpus'])
assert hashlib.sha256(Path('/mnt/local/_gpu_guard/gpu_guard.sh').read_bytes()).hexdigest() == GUARD_HASH
approved = {**APPROVED_BURNS, str(Path('resources/llm_pretrain_burn.py').resolve()): BURN_HASH}
assert all(approved_launcher(pid, approved) or approved_launcher(process(pid)['ppid'], approved)
           for pid in inspection['workers'])
recipe = json.loads(Path('deep_kv.b200.json').read_text())
for key, expected in dict(max_steps=28600, warmup_steps=1430, per_device_train_batch_size=16,
                          gradient_accumulation_steps=4, block_size=2048, eval_rows=4882).items():
 assert recipe[key] == expected
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'): assert Path(recipe[key]).exists()
original_recipe = dict(recipe)
assert recipe['seed'] == recipe['data_seed'] == 42 and recipe['stop_after'] == 2500
recipe.update(seed=123, data_seed=123)
assert {key for key in recipe if recipe[key] != original_recipe[key]} == {'seed', 'data_seed'}
(root / 'seed_provenance.json').write_text(json.dumps({
 'original_seed': 42, 'seed': 123, 'data_seed': 123, 'python_hash_seed': 123,
 'backbone_seed': 123, 'branch_seed': 124,
 'note': 'Existing derived branch stream remains seed+1. Python/NumPy/Torch use set_seed(123); dataset shuffle and sampler use 123. Saved text splits and fixed evaluation prefix unchanged.',
 'fresh_start': True, 'only_recipe_changes': ['seed', 'data_seed'],
 'original_recipe': original_recipe}, indent=2))
recipe_path = root / 'recipe.json'
recipe_path.write_text(json.dumps(recipe, indent=2))
arms = ['A', 'B']
manifest = jobs(recipe_path, stop_after=2500, arms=arms)
manifest['jobs'].insert(0, {
 'name': 'AB-seed123-smoke', 'gpus': list(range(8)),
 'argv': ['{python}', str(Path('scripts/smoke_deep_kv_functional.py').resolve()),
          '--root', '{run_dir}/smoke', '--recipe', str(recipe_path), '--arms', *arms],
 'required_outputs': [{'path': 'smoke/smoke_complete.json', 'json_equals': {
  'status': 'ok', 'arms': arms, 'resumed_step': 12, 'tokens_per_update': 1048576, 'scientific_result': False}}]})
manifest_path = root / 'jobs.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
load_jobs(manifest_path)
(root / 'gpu_inspection.json').write_text(json.dumps(inspection, indent=2))
(root / 'preflight.json').write_text(json.dumps({'at': inspection['time'], 'host': inspection['host'],
 'arms': arms, 'stop_after': 2500, 'schedule_steps': 28600, 'warmup_steps': 1430,
 'tokens_per_update': 1048576, 'accelerate_cache': str(destination),
 'accelerate_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
 'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
 ['train.py','deep_kv/model.py','deep_kv/training.py','scripts/smoke_deep_kv_functional.py','scripts/train_then_burn.py']}}, indent=2))
print('AB_SEED123_PREFLIGHT_PASSED', json.dumps({'accelerate_cache': str(destination), 'arms': arms}), flush=True)
PY
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/production" \
 --inspection "$TASK_ROOT/gpu_inspection.json" --host thiennh-p6-78gg-worker-0 --burn-session deep-kv-AB-seed123-2500-20260930-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 70 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

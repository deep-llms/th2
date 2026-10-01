#1 +60+a
#th2-78gg-TA-10000-preflight-20261001-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import hashlib, importlib.metadata, json, shutil
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH, process
inspection = inspect()
print('GPU_INSPECTION', json.dumps(inspection), flush=True)
approved = {**APPROVED_BURNS, str(Path('resources/llm_pretrain_burn.py').resolve()): BURN_HASH}
assert not inspection['guard_disabled']
assert all(approved_launcher(pid, approved) or approved_launcher(process(pid)['ppid'], approved)
           for pid in inspection['workers'])
assert hashlib.sha256(Path('/mnt/local/_gpu_guard/gpu_guard.sh').read_bytes()).hexdigest() == GUARD_HASH
for name in ('torch', 'transformers', 'accelerate', 'datasets'):
 print('ENV_VERSION', name, importlib.metadata.version(name), flush=True)
recipe = json.loads(Path('deep_kv.b200.json').read_text())
for key in ('config_name', 'tokenizer_name', 'data_dir', 'eval_data_dir'):
 path = Path(recipe[key]); assert path.exists(), str(path)
 print('INPUT_EXISTS', key, str(path), flush=True)
print('FREE_DISK_GIB', shutil.disk_usage('/mnt/local/_outputs').free / 2**30, flush=True)
root = Path('/mnt/local/_outputs/deep-llms_th2/deep-bottleneck-5000-20260930-a01')
saved_recipe = json.loads((root / 'recipe.json').read_text())
expected = dict(recipe, stop_after=5000, checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=True,
                causal_attention=False, lm_chunk=128, skip_memory_metrics=True)
assert saved_recipe == expected, {k: (saved_recipe.get(k), expected.get(k)) for k in set(saved_recipe) | set(expected) if saved_recipe.get(k) != expected.get(k)}
run = root / 'production/run/Task-Aware-Align'
checkpoint = run / 'checkpoint-5000'
state = json.loads((checkpoint / 'trainer_state.json').read_text())
config = json.loads((run / 'train_config.json').read_text())
result = json.loads((run / 'result.json').read_text())
assert state['global_step'] == 5000 and state['max_steps'] == 28600, state['global_step']
assert config['pilot']['arm'] == result['arm'] == 'Task-Aware-Align' and config['world_size'] == 8
assert result['global_step'] == 5000 and result['status'] == 'stopped'
sizes = {name: (checkpoint / name).stat().st_size for name in
         ['model.safetensors', 'optimizer.pt', 'scheduler.pt', 'training_args.bin', 'trainer_state.json'] + [f'rng_state_{i}.pth' for i in range(8)]}
assert all(sizes.values()), sizes
print('TA_SOURCE_VERIFIED', json.dumps(dict(checkpoint=str(checkpoint), sizes=sizes, eval_lm_loss=result['evaluation']['eval_lm_loss'],
      seed=config['training']['seed'], data_seed=config['training']['data_seed'], train_fingerprint=config['train_fingerprint'],
      checkpoints=sorted(p.name for p in run.glob('checkpoint-*')))), flush=True)
print('READ_ONLY_PREFLIGHT_PASSED', flush=True)
PY

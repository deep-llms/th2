#!/bin/bash
# Run only after fresh GPU ownership inspection and authorized burn reclamation.
set -euo pipefail
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline WANDB_PROJECT=deep2shallow NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1
TASK_SMOKE=/mnt/local/_outputs/deep-llms_th2/deep-kv-refactor-smoke-20260927-a01
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
test ! -e "$TASK_SMOKE"
mkdir -p "$TASK_SMOKE"
exec > >(tee "$TASK_SMOKE/smoke.log") 2>&1
python -u - "$TASK_SMOKE" <<'PY'
import json, shutil, sys
from pathlib import Path
from train import load_text
from accelerate.commands.config.config_args import default_yaml_config_file
root = Path(sys.argv[1])
assert shutil.disk_usage(root).free > 180 * 2**30, 'Insufficient checkpoint disk space'
assert Path(default_yaml_config_file).read_bytes() == Path('resources/accelerate_config.yaml').read_bytes()
config = json.loads(Path('deep_kv.b200.json').read_text())
for key, rows, name in (('data_dir', 20000, 'train'), ('eval_data_dir', 2000, 'eval')):
    ds = load_text(config[key])
    ds = ds.select(range(min(rows, len(ds))))
    dest = root / 'text' / name
    ds.save_to_disk(str(dest))
    config[key] = str(dest)
# Smoke-only budget. Preserve the full model, context, microbatch, accumulation,
# optimizer, and BF16; use a short schedule to exercise cutoff and resume.
config.update(max_steps=4, warmup_steps=1, preprocessing_num_workers=8,
              eval_rows=129, monitor_rows=16, logging_steps=1, save_steps=1,
              eval_steps=1, dataloader_num_workers=2)
(root / 'recipe.json').write_text(json.dumps(config, indent=2))
PY
accelerate env
python -m deep_kv make-jobs --config "$TASK_SMOKE/recipe.json" --stop-after 2 --output "$TASK_SMOKE/jobs.json"
python run_experiments.py --config "$TASK_SMOKE/jobs.json" --run-dir "$TASK_SMOKE/run"
python -u - "$TASK_SMOKE" <<'PY'
import json, shutil, subprocess, sys
from pathlib import Path
from scripts.gpu_status import require_free
root = Path(sys.argv[1])
shutil.copy2(root / 'run/comparison.json', root / 'comparison-step2.json')
queue = json.loads((root / 'jobs.json').read_text())['jobs']
for job in queue[:4]:
    require_free(list(range(8)))
    argv = [x.replace('{python}', sys.executable).replace('{run_dir}', str(root / 'run')) for x in job['argv']]
    argv[argv.index('--stop_after') + 1] = '3'
    with (root / (job['name'] + '-resume.log')).open('x') as log:
        subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
    require_free(list(range(8)))
from deep_kv.report import report
result = report(root / 'run')
assert result['compared_update'] == 3
for arm in 'ABCD':
    path = root / 'run' / arm
    config = json.loads((path / 'train_config.json').read_text())
    value = json.loads((path / 'result.json').read_text())
    assert config['world_size'] == 8 and config['tokens_per_update'] == 1048576
    assert value['input_tokens'] == 3 * 1048576 and value['evaluation']['eval_rows'] == 129
    assert config['model_config']['num_hidden_layers'] == 28
    assert config['training']['bf16'] and config['training']['per_device_train_batch_size'] == 16
    assert (path / 'checkpoint-3/optimizer.pt').is_file()
    assert len(list((path / 'checkpoint-3').glob('rng_state_*.pth'))) == 8
(root / 'smoke_complete.json').write_text(json.dumps({'status': 'ok', 'arms': list('ABCD'),
    'world_size': 8, 'initial_cutoff': 2, 'resumed_step': 3, 'schedule_steps': 4,
    'tokens_per_update': 1048576, 'eval_rows': 129, 'scientific_result': False}, indent=2))
print('DEEP_KV_EIGHT_GPU_SMOKE_OK', flush=True)
PY

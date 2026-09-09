#1 +60+a
#th2-swt-benchmark-inventory-before-full-eval-20260909-c01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7
df -h /mnt/local/_data /mnt/local/_outputs
python - <<'PY'
import hashlib, json
from pathlib import Path
from eval.benchmarks import task_plan
root = Path('/mnt/local/_data/deep-llms_th2/benchmarks/hf')
plan, unavailable = task_plan('en')
assert not unavailable and len(plan) == 78
for repo in sorted({item['repository'] for item in plan}):
    path = root/repo
    print('REPOSITORY', repo, 'EXISTS', path.is_dir(), 'PARQUETS', len(list(path.rglob('*.parquet'))), flush=True)
manifest = json.loads(Path('resources/english_core_benchmark_files_20260908.json').read_text())
counts = dict(ok=0, missing=0, mismatch=0)
for item in manifest['files']:
    path = root/item['path']
    if not path.is_file():
        counts['missing'] += 1
        print('MISSING', item['path'])
    else:
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != item['bytes'] or digest != item['sha256']:
            counts['mismatch'] += 1
            print('MISMATCH', item['path'])
        else:
            counts['ok'] += 1
print('CORE_MANIFEST_INVENTORY', json.dumps(counts), flush=True)
training = Path('/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_5k_s42_20260908_a01')
for arm in ('B0','A128','A256','A512','C','D'):
    for step in (250,500,1000,2000,3000,4000,5000):
        ckpt = training/arm/f'checkpoint-{step}'
        assert (ckpt/'config.json').is_file() and list(ckpt.glob('*.safetensors')), str(ckpt)
print('ALL_42_CHECKPOINTS_PRESENT')
print('SWT_BENCHMARK_INVENTORY_COMPLETE_NO_GPU_WORK_LAUNCHED')
PY

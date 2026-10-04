#1 +60+a
#th2-tjx3-verify-sampled-loader-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone
print('DATA_CHECK_START', datetime.now(timezone.utc).isoformat(), flush=True)
root = Path('/mnt/local/_data/deep-llms_th2/cx_sampled_old')
raw = (root / 'dataset_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'dbba73b7a95ebc530297bac5a2151be6292f080e80e9b3101302a905806dde9a'
manifest = json.loads(raw)
receipt = json.loads((root / 'UPLOAD_COMPLETE.json').read_text())
assert receipt['state'] == 'complete' and receipt['pool_identity'] == manifest['pool_identity']
for name, spec in manifest['files'].items():
    path = root / name
    assert path.is_file() and path.stat().st_size == spec['bytes'], name
print('ALL_MANIFESTED_FILE_SIZES_OK', len(manifest['files']), flush=True)
subset = next(s for s in manifest['subsets'] if s['name'] == 'qwen3_0.6b_base_en_30B')
prefix = 'subsets/' + subset['name'] + '/'
checked = 0
for name, spec in manifest['files'].items():
    if not name.startswith(prefix):
        continue
    digest = hashlib.sha256()
    with (root / name).open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == spec['sha256'], name
    checked += 1
    if checked % 20 == 0:
        print('ENGLISH_HASH_PROGRESS', checked, flush=True)
print('ENGLISH_FILE_HASHES_OK', checked, flush=True)
from train import load_text
from deep_kv.__main__ import jobs
config = json.loads(Path('deep_kv.b200.json').read_text())
for split, key in [('train', 'data_dir'), ('validation', 'eval_data_dir')]:
    directory = root / 'subsets' / subset['name'] / split
    assert Path(config[key]) == directory
    expected_paths = [root / s['path'] for s in subset['shards'][split]]
    assert sorted(directory.glob('shard_*')) == expected_paths
    data = load_text(config[key])
    assert data.column_names == ['text']
    assert len(data) == subset['totals'][split]['documents']
    assert isinstance(data[0]['text'], str) and isinstance(data[-1]['text'], str)
    print('TRAIN_LOADER_OK', split, len(data), 'shards', len(expected_paths), flush=True)
for job in jobs('deep_kv.b200.json')['jobs'][:-1]:
    for key in ('data_dir', 'eval_data_dir'):
        assert job['argv'][job['argv'].index('--' + key) + 1] == config[key]
print('QUEUE_DATA_PATHS_OK', flush=True)
for key in ('config_name', 'tokenizer_name'):
    print('MODEL_ASSET_EXISTS', key, Path(config[key]).exists(), flush=True)
print('DATA_COMPATIBILITY_PASSED', datetime.now(timezone.utc).isoformat(), flush=True)
PY

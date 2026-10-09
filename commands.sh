#1 +60+a
#th2-q359-check-cx-sampled-old-20261009-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/train_env/bin/python3.11 - <<'PY'
from pathlib import Path
import datasets

root = Path('/mnt/local/_data/deep-llms_th2/cx_sampled_old')
assert root.is_dir(), root
print('ROOT', root, flush=True)
print('ROOT_ENTRIES', sorted(p.name for p in root.iterdir()), flush=True)
for split in ('train', 'validation'):
    path = root / 'subsets' / 'qwen3_0.6b_base_en_30B' / split
    assert path.is_dir(), path
    shards = sorted(p for p in path.glob('shard_*') if p.is_dir())
    assert shards, path
    rows = 0
    for shard in shards:
        assert (shard / 'state.json').is_file(), shard
        dataset = datasets.load_from_disk(str(shard))
        assert 'text' in dataset.column_names and len(dataset) > 0, shard
        rows += len(dataset)
    print('SPLIT', split, 'SHARDS', len(shards), 'ROWS', rows, flush=True)
    assert rows > 1000000 if split == 'train' else rows > 1000
print('CX_SAMPLED_OLD_LOCAL_CHECK_OK', flush=True)
PY

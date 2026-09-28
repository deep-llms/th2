#1 +60+a
#th2-78gg-readonly-data-format-20260928-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import json, socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq

print('READ_ONLY_DATA_FORMAT', datetime.now(timezone.utc).isoformat(), socket.gethostname(), flush=True)
base = Path('/mnt/local/_data/deep-llms_th2/data')
raw = base / 'raw'
print('RAW_ROOT', str(raw), 'exists', raw.is_dir(), flush=True)
files = sorted(p for p in raw.rglob('*') if p.is_file())
print('RAW_EXTENSIONS', dict(Counter(p.suffix for p in files)), flush=True)
for path in files[:8]:
    print('RAW_FILE', str(path.relative_to(base)), path.stat().st_size, flush=True)
parquets = [p for p in files if p.suffix == '.parquet']
if parquets:
    path = parquets[0]
    table = pq.ParquetFile(path)
    print('PARQUET_METADATA', json.dumps(dict(path=str(path), rows=table.metadata.num_rows,
          row_groups=table.metadata.num_row_groups, schema=str(table.schema_arrow))), flush=True)
for split in ('train', 'eval'):
    root = base / 'Qwen_Qwen3-0.6B-Base' / split / 'en'
    states = sorted(root.rglob('state.json'))
    print('SAMPLED_SPLIT', split, str(root), 'saved_datasets', len(states), flush=True)
    if states:
        shard = states[0].parent
        print('SAMPLED_FILES', str(shard), sorted(p.name for p in shard.iterdir()), flush=True)
        print('SAMPLED_STATE', (shard / 'state.json').read_text(), flush=True)
        info = json.loads((shard / 'dataset_info.json').read_text())
        print('SAMPLED_FEATURES', json.dumps(info.get('features')), flush=True)
print('READ_ONLY_DATA_FORMAT_COMPLETE', flush=True)
PY

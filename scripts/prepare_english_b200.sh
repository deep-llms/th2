#!/usr/bin/env bash
# Offline CPU-only preparation; never signals or allocates GPU processes.
set -euo pipefail
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_PREFIX" = /mnt/local/conda-py311/envs/swt
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
SWT_TOKENIZER=/mnt/local/_models/deep-llms_th2/gpt2_tokenizer_607a30d
SWT_PACKS=/mnt/local/_data/deep-llms_th2/swt/english_gpt2_10b_seed0_20260907_a01
test ! -e "$SWT_PACKS"
test ! -L "$SWT_PACKS"
date -u
hostname
python scripts/verify_manifest.py verify --root "$SWT_TOKENIZER" \
  --manifest resources/gpt2_tokenizer_607a30d_manifest.json --strict
python - <<'PY'
from pathlib import Path
import json, shutil
source = json.loads(Path('resources/culturax_en_source_b200.json').read_text())
assert len(source['shards']) == 50
for shard in source['shards']:
    p = Path(shard['path'])
    assert p.is_file() and p.stat().st_size == shard['bytes'], p
assert shutil.disk_usage('/mnt/local/_data/deep-llms_th2').free > 150 * 1024**3
print('RAW_INVENTORY_SIZES_AND_FREE_DISK_OK; full source SHA256 checks follow in preparer')
PY
python -m unittest discover -s tests -v
python -m capacity_allocation.data \
  --source-manifest resources/culturax_en_source_b200.json \
  --tokenizer-dir "$SWT_TOKENIZER" \
  --tokenizer-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --output "$SWT_PACKS" \
  --seed 0 --sample-fraction 0.30 \
  --validation-fraction 0.002 --test-fraction 0.002 \
  --min-train-tokens 10000000000 \
  --min-validation-tokens 20000000 --min-test-tokens 20000000
python - "$SWT_PACKS" <<'PY'
import json, sys
from pathlib import Path
from capacity_allocation.data import PackedTokens, sha256
p = Path(sys.argv[1])
m = json.loads((p / 'manifest.json').read_text())
assert m['complete'] and m['sequence_length'] == 2048
assert m['vocab_size'] == 50257 and m['eos_token_id'] == 50256
assert m['source_manifest_sha256'] == sha256(p / 'source_manifest.json')
assert m['documents_sha256'] == sha256(p / 'documents.sqlite')
for split, minimum in [('train', 10_000_000_000), ('validation', 20_000_000), ('test', 20_000_000)]:
    ds = PackedTokens(p, split, verify_hash=True)
    assert m['splits'][split]['packed_tokens'] >= minimum
    for index in (0, len(ds)-1):
        ids = ds[index]['input_ids']
        assert len(ids) == 2048 and int(ids.min()) >= 0 and int(ids.max()) < 50257
print(json.dumps(m['splits'], indent=2))
print('SWT_ENGLISH_GPT2_10B_PACKS_VERIFIED')
print('Near-duplicate audit remains required before research training.')
PY
date -u
echo SWT_SAMPLING_FINISHED_SUCCESSFULLY

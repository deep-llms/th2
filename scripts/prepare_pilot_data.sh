#!/usr/bin/env bash
# Full offline pilot data only. Does not train, download, or manage GPU processes.
set -euo pipefail
test "$(hostname)" = thiennh-p6-8mgy-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
TASK_RAW=/mnt/local/_data/deep-llms_th2/data/raw
TASK_ASSETS=/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-da87bfb608c14b7cf20ba1ce41287e8de496c0cd
TASK_DATA=/mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/ccm_prepare_pilot_v1_20260913_a01
date -u
hostname
command -v python
test -d "$TASK_RAW/en"
test ! -e "$TASK_DATA" && test ! -L "$TASK_DATA"
test ! -e "$TASK_OUTPUT" && test ! -L "$TASK_OUTPUT"
python - "$TASK_ASSETS" <<'PY'
import shutil, sys
import numpy, pyarrow, torch, transformers
from ccm.cli import asset_identity, code_hash
from ccm.contracts import PILOT, require
from ccm.model import pilot_config
require(transformers.__version__ == '5.9.0', 'Untested Transformers version')
require(shutil.disk_usage('/mnt/local').free >= 100*2**30, 'Need at least 100 GiB free working space')
assets = asset_identity(sys.argv[1], 'resources/qwen3_base_assets.json')
config = pilot_config(sys.argv[1])
print('PILOT_DATA_PREFLIGHT_OK', dict(torch=torch.__version__, transformers=transformers.__version__,
      numpy=numpy.__version__, pyarrow=pyarrow.__version__, code_hash=code_hash(),
      base_revision=assets['revision'], layers=config.num_hidden_layers, quotas=PILOT.quotas()), flush=True)
PY
# Read-only observation: existing GPU work must stay untouched during this CPU task.
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
mkdir -p "$(dirname "$TASK_DATA")" "$(dirname "$TASK_OUTPUT")"
mkdir "$TASK_DATA" "$TASK_OUTPUT"
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 > "$TASK_OUTPUT/gpus_before.json"
echo CCM_PILOT_PREPARE_STARTED
python -m ccm prepare \
  --raw-dir "$TASK_RAW" --source-manifest resources/culturax_raw_manifest.tsv \
  --dataset-revision 6a8734bc69fefcbb7735f4f9250f43e4cd7a442e \
  --tokenizer-path "$TASK_ASSETS" --tokenizer-manifest resources/qwen3_base_assets.json \
  --output "$TASK_DATA/corpus" | tee "$TASK_OUTPUT/prepare.log"
date -u
echo CCM_PILOT_VALIDATE_STARTED
python -m ccm validate-data --data "$TASK_DATA/corpus" > "$TASK_OUTPUT/data_validation.json"
echo CCM_PILOT_VOCABULARY_STARTED
python -m ccm vocabulary --data "$TASK_DATA/corpus" --output "$TASK_DATA/vocabulary.npz" \
  | tee "$TASK_OUTPUT/vocabulary.log"
date -u
echo CCM_PILOT_COVERAGE_STARTED
python -m ccm coverage --data "$TASK_DATA/corpus" --vocabulary "$TASK_DATA/vocabulary.npz" \
  --output "$TASK_OUTPUT/coverage.json" | tee "$TASK_OUTPUT/coverage.log"
python - "$TASK_DATA" "$TASK_OUTPUT" "$TASK_ASSETS" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys
from ccm.cli import asset_identity, code_hash
from ccm.contracts import PILOT, require, read_json, write_json, file_hash
from ccm.data import Corpus, validate_splits
from ccm.keys import Vocabulary
from ccm.runtime import require_coverage
data, out, assets_path = map(Path, sys.argv[1:])
corpus = Corpus(data/'corpus')
require(not corpus.meta['engineering'] and corpus.budget == PILOT, 'Wrong scientific budget/mode')
assets = asset_identity(assets_path, 'resources/qwen3_base_assets.json')
require(corpus.meta['provenance']['tokenizer'] == assets, 'Base asset provenance mismatch')
require(corpus.meta['provenance']['source_code_hash'] == code_hash(), 'Source changed during preparation')
documents = validate_splits(corpus)
require(read_json(out/'data_validation.json')['manifest_hash'] == corpus.meta['manifest_hash'], 'Validation mismatch')
batch_counts = {}
for phase, expected in [('common', PILOT.common_steps), ('stage1', PILOT.adapt_steps), ('stage2', PILOT.continue_steps)]:
    count = 0
    for rows in corpus.optimizer_batches(phase, 1017):
        require(sum(len(row['tokens']) for row in rows) == PILOT.batch_tokens, 'Incorrect immutable optimizer batch')
        count += 1
    require(count == expected, 'Incorrect optimizer step count')
    batch_counts[phase] = count
    print('PILOT_BATCHES_VERIFIED', phase, count, flush=True)
vocab = Vocabulary.load(data/'vocabulary.npz')
require(len(vocab.keys) == PILOT.slots, 'Wrong exact vocabulary capacity')
require(vocab.metadata['compile_segment_hash'] == corpus.meta['files']['compile.segments.jsonl'],
        'Vocabulary compilation-segment mismatch')
cov = require_coverage(out/'coverage.json', corpus, vocab)
require(cov['role'] == 'dev' and cov['before_reader_training'], 'Wrong coverage role/timing')
report = dict(success=True, completed_at=datetime.now(timezone.utc).isoformat(),
              corpus_hash=corpus.meta['manifest_hash'], vocabulary_hash=vocab.hash,
              quotas=corpus.meta['quotas'], documents=documents, optimizer_batches=batch_counts,
              slots=len(vocab.keys), eligible_hit_rate=cov['eligible_hit_rate'],
              data_root=str(data), base_assets=str(assets_path), source_code_hash=code_hash(),
              files={str(p): file_hash(p) for p in [
                  data/'corpus/manifest.json', data/'vocabulary.npz', Path(str(data/'vocabulary.npz')+'.json'),
                  out/'data_validation.json', out/'coverage.json']})
write_json(out/'complete.json', report)
print('CCM_PILOT_DATA_VERIFIED', report, flush=True)
PY
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 > "$TASK_OUTPUT/gpus_after.json"
date -u
echo CCM_PILOT_STEP1_COMPLETE_NO_TRAINING_LAUNCHED

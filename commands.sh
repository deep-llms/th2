#1 +60+a
#th2-swt-verify-downloaded-benchmarks-offline-20260909-c01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/benchmark_verify_20260909_c01
test ! -e "$TASK_ROOT"
mkdir "$TASK_ROOT"
exec > >(tee "$TASK_ROOT/pipeline.log") 2>&1
python scripts/verify_manifest.py verify --root /mnt/local/_data/deep-llms_th2/benchmarks/hf --manifest resources/english_core_benchmark_files_20260908.json
python -u - "$TASK_ROOT" <<'PY'
import sys
from pathlib import Path
from eval.runtime import offline
offline()
from eval.benchmarks import task_plan, load_tasks
from capacity_allocation.data import write_json
plan, missing = task_plan('en')
assert len(plan) == 78 and not missing
tasks = load_tasks(plan, '/mnt/local/_data/deep-llms_th2/benchmarks/hf')
coverage = {}
for name, task in tasks.items():
    docs = task.eval_docs
    assert len(docs) > 0, name
    coverage[name] = dict(eval_examples=len(docs), fingerprint=docs._fingerprint)
    print('EVAL_SPLIT', name, len(docs), flush=True)
for name in ('hellaswag', 'arc_easy', 'xnli_en'):
    assert tasks[name].has_training_docs()
    docs = tasks[name].training_docs()
    assert len(docs) > 0
    coverage[name]['training_examples'] = len(docs)
    print('TRAIN_SPLIT', name, len(docs), flush=True)
write_json(Path(sys.argv[1])/'coverage.json', dict(success=True, tasks=coverage))
print('ALL_78_EVAL_TASKS_AND_3_TRAINING_SPLITS_LOAD_OFFLINE', flush=True)
PY
python -u -m scripts.smoke_english_benchmarks --dataset-root /mnt/local/_data/deep-llms_th2/benchmarks/hf --output-dir "$TASK_ROOT/core_smoke" --examples-per-task 2 --arms B0 C
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
sha256sum "$TASK_ROOT/coverage.json" "$TASK_ROOT/core_smoke/complete.json"
echo SWT_ALL_BENCHMARK_INPUTS_VERIFIED_CPU_SMOKE_PASSED

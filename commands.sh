#1 +60+a
#th2-swt-fixed-effective-batch-benchmark-20260908-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
test "$PWD" = /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
test "$CONDA_DEFAULT_ENV" = swt
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
python scripts/verify_manifest.py verify --root "$PWD" --manifest resources/swt_qwen_launch_20260908.json
python - <<'PY'
import hashlib
from pathlib import Path
expected = {
    'scripts/benchmark_capacity_batch.py': '380ddd75b16ed3ff6c15a1717105dc34326c3682bc9dd47a6e257d04baa6f216',
    'scripts/benchmark_batches_b200.sh': 'bd2b40bd01c4ae6a6b4c1dea62f6928dc8f14030b126de549527124671bcdec2',
}
for name, digest in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
print('BATCH_BENCHMARK_SOURCE_VERIFIED',flush=True)
PY
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p test_batch_benchmark.py -v
bash scripts/benchmark_batches_b200.sh /mnt/local/_outputs/@PROJECT@/swt/batch_benchmark_20260908_a01

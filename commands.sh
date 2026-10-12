#1 +60+a
#th2-q359-paired-parallel-enable-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-expanded-single-seed42-20261013-a01
"$TASK_PYTHON" -u -m scripts.refresh_paired_schedule --root "$TASK_ROOT" \
 --expected resources/paired_parallel_refresh_20261013.json --output "$TASK_ROOT/parallel-refresh.json"
cat "$TASK_ROOT/parallel-refresh.json"
"$TASK_PYTHON" -m unittest tests.test_paired_parallel -q
"$TASK_PYTHON" -u - "$TASK_ROOT" <<'REMOTE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for path in (root/'supervised/run/run.json',root/'supervised/run/finetune/run.json'):
 value=json.loads(path.read_text());value['jobs']=value['jobs'][-2:];print(path,json.dumps(value))
print('PARALLEL_PAIRED_SCHEDULER_INSTALLED; active fine-tuning left running',flush=True)
REMOTE

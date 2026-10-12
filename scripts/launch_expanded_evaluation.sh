#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${1:?Pass fresh run root}
TASK_HOST=${2:?Pass verified hostname}
TASK_POSITION_DOCUMENTS=${3:?Pass the frozen position-probe document count}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate eval_fa4
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m pip check
# Controller downloads are asynchronous. Burns remain active until every file is present.
python -u - <<'PY'
import json,time
from pathlib import Path
root=Path('/mnt/local/_data/deep-llms_th2/expanded-20261012')
manifest=json.loads(Path('resources/expanded_inputs_20261012.json').read_text())
files={f['path']:f['bytes'] for repo in manifest['repositories'] for f in repo['files']}
for _ in range(720):
    missing=[p for p,n in files.items() if not (root/p).is_file() or (root/p).stat().st_size!=n]
    if not missing:
        print('CONTROLLER_INPUT_SIZES_COMPLETE; next verifying checksums',flush=True)
        break
    print('WAITING_FOR_CONTROLLER_DOWNLOAD',len(files)-len(missing),'/',len(files),flush=True)
    time.sleep(30)
else:raise RuntimeError('Controller downloads incomplete after six hours; GPUs untouched')
PY
python -u -m scripts.expanded_evaluation_study --root "$TASK_ROOT" --host "$TASK_HOST" --position-documents "$TASK_POSITION_DOCUMENTS"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

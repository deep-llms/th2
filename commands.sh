#1 +60+a
#th2-distill-ownership-cpu-ready-20260924-a01
set -euo pipefail
date -u
PYTHONPATH="$PWD" /usr/bin/python3 -u -m scripts.verified_gpu_reclaim inspect --output /mnt/local/_outputs/@PROJECT@/distill-ownership-20260924-a01.json
TASK_PYTHON=/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11
TASK_LOG=/mnt/local/_outputs/@PROJECT@/distill-cpu-tests-20260924-a01.log
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$TASK_PYTHON" -m unittest discover -s tests -v > "$TASK_LOG" 2>&1
tail -n 8 "$TASK_LOG"
"$TASK_PYTHON" - <<'CHECK'
import json
from pathlib import Path
from pcc.joint_training import code_identity
Path('/mnt/local/_outputs/deep-llms_th2/distill-cpu-ready-20260924-a01.json').write_text(json.dumps({'status':'ok','code':code_identity()})+'\n')
CHECK
echo DISTILL_CPU_READY

#1 +60+a
#th2-readonly-verify-tiered-groupreduce-eval-start-20260905-a01
set -euo pipefail
date -u
TASK_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RUN=tiered_c512_groupreduce_lb_10k_20260905_a01
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
tail -25 "$TASK_BASE/logs/eval_finetune_${TASK_RUN}.log"
/mnt/local/conda-py311/envs/eval/bin/python3.11 - "$TASK_BASE" <<'PY'
from pathlib import Path
import sys
base=Path(sys.argv[1])
for arm in ('tiered_ranklift_lb_t4_c512','groupreduce_matched_lb_t4'):
    path=base/arm/'checkpoint-10000'/'eval.log'
    print('EVAL_LOG',arm)
    if path.exists():
        with path.open('rb') as handle:
            head=handle.read(12000).decode(errors='replace')
            handle.seek(max(0,path.stat().st_size-6000))
            tail=handle.read().decode(errors='replace')
        print('\n'.join(line for line in head.splitlines() if 'Loaded' in line or 'checkpoint' in line or 'arm=' in line))
        print(tail[-3500:])
    else:
        print('not present')
PY
tail -5 "$TASK_BASE/logs/burn_eval_${TASK_RUN}.log"
echo 'TH2 READONLY EVAL START CHECK COMPLETE'

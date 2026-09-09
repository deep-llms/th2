#1 +30+a
#th2-swt-identify-finetune-before-final-only-stop-20260909-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
date -u
hostname
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt_eval/bin/python
python - <<'PY'
import json
from pathlib import Path
from scripts.reclaim_verified_burn import identity
from scripts.gpu_status import snapshot
root = '/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01'
expected = ['bash', 'scripts/eval_finetune_capacity_b200.sh', root]
records = {}
for path in Path('/proc').iterdir():
    if path.name.isdigit() and int(path.name) > 1:
        try:
            records[int(path.name)] = identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
for queue in [r for r in records.values() if r['argv'] == expected]:
    owned = {queue['pid']}
    while True:
        expanded = owned | {p for p,r in records.items() if r['parent'] in owned}
        if expanded == owned:
            break
        owned = expanded
    print('EXACT_PIPELINE', json.dumps(queue))
    print('OWNED_DESCENDANTS', json.dumps([records[p] for p in sorted(owned)]))
print('GPU_STATUS', json.dumps(snapshot()))
ft = Path(root)/'finetune'
print('FINETUNE_COUNTS', len(list(ft.glob('*/training_complete.json'))), len(list(ft.glob('*/result.json'))))
print('FINAL_RESULTS', [p.parent.name for p in ft.glob('*_step5000_*/result.json')])
for stage in ('eval','finetune'):
    p = Path(root)/stage/'complete.json'
    print('COMPLETE', stage, p.exists())
print('PIPELINE_TAIL', '\n'.join((Path(root)/'pipeline.log').read_text().splitlines()[-12:]))
print('FINAL_ONLY_STOP_INSPECTION_COMPLETE')
PY

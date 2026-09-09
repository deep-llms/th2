#1 +60+a
#th2-swt-readonly-runtime-estimate-20260909-1014-a01
set -euo pipefail
date -u
hostname
cd /mnt/local/deep-llms_th2
python3 scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7
python3 - <<'PY'
import json, time
from datetime import datetime, timezone
from pathlib import Path
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01')
for stage in ('eval', 'finetune'):
    folder = root/stage
    plan = folder/'plan.json'
    end = folder/'complete.json'
    if plan.is_file():
        start = plan.stat().st_mtime
        finish = end.stat().st_mtime if end.is_file() else time.time()
        print('STAGE_TIME', stage, 'start_utc', datetime.fromtimestamp(start, timezone.utc).isoformat(), 'hours', round((finish-start)/3600,4), 'complete', end.is_file())
    if end.is_file():
        result = json.loads(end.read_text())
        print('STAGE_COMPLETE', stage, result['success'], len(result['completed']))
    failure = folder/'failed.json'
    if failure.is_file():
        print('STAGE_FAILURE', stage, failure.read_text()[:5000])
for path in sorted((root/'finetune').glob('*/training_complete.json')):
    item = json.loads(path.read_text())
    print('FINETUNE_TIMING', json.dumps(dict(job=path.parent.name, task=item['task'], seconds=item['training']['seconds'], steps=item['training']['steps'], used_examples=item['used_examples'], eval_complete=(path.parent/'result.json').is_file())))
ft = root/'finetune'
print('FINETUNE_COUNTS', 'started_config', len(list(ft.glob('*/run_config.json'))), 'training_complete', len(list(ft.glob('*/training_complete.json'))), 'eval_complete', len(list(ft.glob('*/result.json'))))
print('EVAL_COUNTS', 'ppl', len(list((root/'eval').glob('*/ppl.json'))), 'benchmarks', len(list((root/'eval').glob('*/benchmarks.json'))))
pipeline = (root/'pipeline.log').read_text()
print('PIPELINE_TAIL')
print('\n'.join(pipeline.splitlines()[-18:]))
print('SWT_READONLY_RUNTIME_ESTIMATE_COMPLETE')
PY

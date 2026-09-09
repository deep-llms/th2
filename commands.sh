#1 +30+a
#th2-swt-final5k-readonly-progress-timing-20260909-1246
set -euo pipefail
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
date -u
hostname
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7
python - <<'PY'
import json, time
from pathlib import Path
from datetime import datetime, timezone
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/final5k_diagnostics_finetune_20260909_a01')
def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()
for stage in ('diagnostics', 'finetune'):
    p = root/stage
    plan = p/'plan.json'
    done = p/'complete.json'
    if plan.exists():
        print('STAGE', stage, 'started', utc(plan.stat().st_mtime), 'elapsed_minutes', (time.time()-plan.stat().st_mtime)/60)
    if done.exists():
        d = json.loads(done.read_text())
        print('STAGE_COMPLETE', stage, d['success'], len(d['completed']), utc(done.stat().st_mtime))
    if (p/'failed.json').exists():
        print('STAGE_FAILED', stage, (p/'failed.json').read_text())
for p in sorted((root/'finetune').glob('*/run_config.json')):
    config = json.loads(p.read_text())
    trained, scored = p.parent/'training_complete.json', p.parent/'result.json'
    row = dict(job=p.parent.name, task=config['task'], started_utc=utc(p.stat().st_mtime),
        elapsed_minutes=(time.time()-p.stat().st_mtime)/60, trained=trained.exists(), scored=scored.exists())
    if trained.exists():
        d = json.loads(trained.read_text())
        row.update(train_seconds=d['training']['seconds'], steps=d['training']['steps'], used_examples=d['used_examples'])
    if scored.exists():
        row.update(finished_utc=utc(scored.stat().st_mtime), total_seconds=scored.stat().st_mtime-p.stat().st_mtime)
    print('FINETUNE_JOB', json.dumps(row), flush=True)
for p in sorted((root/'finetune').glob('*_xnli_*.log')):
    if (root/'finetune'/p.stem/'result.json').exists():
        continue
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size-1800))
        print('XNLI_LOG_TAIL', p.name, 'mtime', utc(p.stat().st_mtime), f.read().decode(errors='replace'), flush=True)
print('PIPELINE_COMPLETE', (root/'complete.json').exists())
print('PIPELINE_TAIL', '\n'.join((root/'pipeline.log').read_text().splitlines()[-12:]))
print('FINAL5K_READONLY_PROGRESS_COMPLETE', utc(time.time()))
PY

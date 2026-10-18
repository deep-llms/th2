#1 +30+a
#th2-q359-dot-final-verification-20261018-a02
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
"$TASK_PYTHON" - <<'PY'
import ast, hashlib, json, math, re, time
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.train_then_burn import burn_progress, approved_launcher, BURN_HASH, GUARD_HASH, digest
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a03')
summary=json.loads((root/'supervised/run/checks/summary.json').read_text())
assert summary['status']=='passed' and not summary['failed_required']
assert len(summary['results'])==17
supervisor=json.loads((root/'supervised/supervisor.json').read_text())
assert supervisor['training_status']=='ok' and supervisor['training_returncode']==0
assert 'finished_at' in supervisor and 'handoff_error' not in supervisor
assert supervisor['burn']['collective_progress_verified']
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
source=Path.cwd()/'resources/llm_pretrain_burn.py'
assert digest(source)==BURN_HASH
losses={}
for r in summary['results']:
    if not r['name'].startswith('bench_'):
        continue
    receipt=r['receipt']
    assert r['passed'] and receipt['world_size']==8
    assert receipt['config']['num_hidden_layers']==12
    for name, expected in receipt['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
    lines=(root/'supervised/run/checks'/(r['name']+'.log')).read_text().splitlines()
    rows=[ast.literal_eval(line) for line in lines if line.startswith("{'train_runtime':")]
    assert len(rows)==1
    loss=float(rows[0]['train_loss'])
    assert math.isfinite(loss) and loss>0
    losses[r['name']]=rows[0]
first=inspect()
assert not first['guard_disabled']
expected=supervisor['burn']['gpus']
assert [(g['uuid'],g['pids']) for g in first['gpus']]==[(g['uuid'],g['pids']) for g in expected]
assert len(first['workers'])==8 and all(len(g['pids'])==1 for g in first['gpus'])
for pid in first['workers']:
    assert approved_launcher(first['processes'][str(pid)]['ppid'],{str(source):BURN_HASH})
log=root/'supervised/burn.log'
def latest():
    text=log.read_text()
    assert burn_progress(text)
    values=re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',text)
    return int(values[-1][0]),float(values[-1][1])
before=latest()
print('FINAL_BURN_INITIAL',before,flush=True)
for _ in range(3):
    time.sleep(10)
    after=latest()
    if after[0]>before[0] and after[1]>before[1]:
        break
assert after[0]>before[0] and after[1]>before[1]
second=inspect()
assert not second['guard_disabled']
assert second['workers']==first['workers']
for pid in first['workers']:
    assert first['processes'][str(pid)]==second['processes'][str(pid)]
result=dict(status='passed',summary_sha256=hashlib.sha256((root/'supervised/run/checks/summary.json').read_bytes()).hexdigest(),supervisor=supervisor,training_metrics=losses,burn_cycles_before=before,burn_cycles_after=after,inspection=second)
with (root/'final_verification_a02.json').open('x') as f:json.dump(result,f,indent=2)
print('DOT_FINAL_VERIFICATION '+json.dumps(result),flush=True)
PY

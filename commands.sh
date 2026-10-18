#1 +30+a
#th2-q359-dot-resume-final-20261018-a01
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
"$TASK_PYTHON" - <<'PY'
import hashlib,json,re,time
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH,GUARD_HASH,digest
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-dot-resume-20261018-a01')
checks=root/'supervised/run/checks'
summary=json.loads((checks/'summary.json').read_text())
assert summary['status']=='passed' and len(summary['results'])==16 and summary['world_size']==8
assert all(r['status']=='passed' and r['max_loss_abs']==0 and r['restored_state_exact'] and r['final_state_exact'] for r in summary['results'])
supervisor=json.loads((root/'supervised/supervisor.json').read_text())
assert supervisor['training_status']=='ok' and supervisor['training_returncode']==0 and 'finished_at' in supervisor
assert supervisor['burn']['collective_progress_verified'] and 'handoff_error' not in supervisor
recipe=json.loads((checks/'full/checkpoint-100/training_recipe.json').read_text())
assert recipe['execution']['world_size']==8 and recipe['model']['attention_backend']=='fa4'
assert recipe['training']['max_steps']==200 and recipe['training']['gradient_accumulation_steps']==2
assert recipe['runtime']['torch'].split('+')[0]=='2.14.1'
for step in (20,100,200):
    checkpoint=checks/'full'/f'checkpoint-{step}'
    for name in ['optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{rank}.pth' for rank in range(8)]:
        assert (checkpoint/name).stat().st_size>0,(step,name)
for name,value in recipe['code_sha256'].items():assert digest(name)==value,name
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
source=Path.cwd()/'resources/llm_pretrain_burn.py'
assert digest(source)==BURN_HASH
first=inspect()
assert not first['guard_disabled'] and len(first['workers'])==8
expected=supervisor['burn']['gpus']
assert [(g['uuid'],g['pids']) for g in first['gpus']]==[(g['uuid'],g['pids']) for g in expected]
assert all(len(g['pids'])==1 for g in first['gpus'])
for pid in first['workers']:assert approved_launcher(first['processes'][str(pid)]['ppid'],{str(source):BURN_HASH})
def progress():
    text=(root/'supervised/burn.log').read_text()
    assert burn_progress(text)
    records=re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)',text)
    return int(records[-1][0]),float(records[-1][1])
before=progress()
for _ in range(3):
    time.sleep(10)
    after=progress()
    if after[0]>before[0] and after[1]>before[1]:break
assert after[0]>before[0] and after[1]>before[1]
second=inspect()
assert not second['guard_disabled'] and second['workers']==first['workers']
assert all(first['processes'][str(pid)]==second['processes'][str(pid)] for pid in first['workers'])
result=dict(status='passed',supervisor=supervisor,burn_cycles_before=before,burn_cycles_after=after,inspection=second,
    summary_sha256=digest(checks/'summary.json'),training_recipe=recipe,model_fixture=json.loads((checks/'model.json').read_text()),
    probe_sha256=digest('tests/bdot/resume_gpu_probe.py'))
with (root/'final_verification.json').open('x') as handle:json.dump(result,handle,indent=2)
print('DOT_RESUME_FINAL '+json.dumps(result),flush=True)
PY

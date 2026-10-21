#1 +30+a
#th2-6y3o-dot-v3-health-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import hashlib,json
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.bdot_job import latest_complete_checkpoint
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01')
run=root/'prepared/runs/P1-A-prime-s0'
read=lambda p:json.loads(p.read_text())
health={'live':inspect(),'queue':read(root/'supervised/run/run.json'),'supervisor':read(root/'supervised/supervisor.json')}
recipe=read(run/'training_recipe.json')
health['source_hashes_match']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in recipe['code_sha256'].items())
health['train_args']=read(root/'prepared/P1-A-prime-s0.train.json')
health['routing']=read(root/'prepared/P1-A-prime-s0.model.json')['bdot']
health['metrics']=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines() if l][-15:]
if list(run.glob('checkpoint-*')):
 cp=latest_complete_checkpoint(run)
 state=read(cp/'trainer_state.json')
 health['checkpoint']={'path':str(cp),'global_step':state['global_step'],'max_steps':state['max_steps'],
  'files':{f.name:f.stat().st_size for f in cp.iterdir() if f.is_file()}}
retention=run/'retention.jsonl'
if retention.is_file():health['retention']=[json.loads(l) for l in retention.read_text().splitlines() if l]
health['retained_checkpoints']=[p.name for p in sorted(run.glob('checkpoint-*'))]
print('V3_HEALTH',json.dumps(health),flush=True)
with (root/'launch-health-a01.json').open('x') as f:json.dump(health,f,indent=2)
PY

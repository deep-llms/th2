#1 +30+a
#th2-6y3o-dot-router-curves-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
base=Path('/mnt/local/_outputs/deep-llms_th2')
paths={
'D-s0':base/'6y3o-dot-pilot-20261020-a01/prepared/runs/D',
'D-s1':base/'6y3o-dot-v3-20261021-a01/prepared/runs/D-s1'}
for seed,root in [(0,'6y3o-dot-router-20261022-a01'),(1,'6y3o-dot-router-confirm-20261023-a01')]:
 for arm in ['Partial-25-cosine','Partial-50-cosine-detach']:
  name=arm+'-s'+str(seed);paths[name]=base/root/'prepared/runs'/name
r={'observed_at':datetime.now(timezone.utc).isoformat(),'runs':{}}
read=lambda p:json.loads(p.read_text())
for name,path in paths.items():
 state=read(path/'checkpoint-4768/trainer_state.json')
 assert state['global_step']==4768
 rows=[row for row in state['log_history'] if 'eval_loss' in row]
 fields=['step','eval_loss','eval_num_sequences','eval_num_tokens','eval_off_default_share','eval_reuse_rate','eval_routing_decisions']
 curve=[{k:row[k] for k in fields if k in row} for row in rows]
 config=read(path/'train_config.json')
 model=read(path/'checkpoint-4768/config.json')
 r['runs'][name]={'root':str(path),'curve':curve,'train':read(path/'train_results.json'),'final':read(path/'final_eval.json'),'config':config,'bdot':model.get('bdot'),'state_sha256':hashlib.sha256((path/'checkpoint-4768/trainer_state.json').read_bytes()).hexdigest()}
print('ROUTER_CURVES '+json.dumps(r),flush=True)
REMOTE

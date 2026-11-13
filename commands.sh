#1 +30+a
#th2-wc69-dot-pool-e1b-v15-health-20261113-a03
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" - <<'PY'
from pathlib import Path
import json,time,hashlib
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112');admission=base/'execution-e1b-v15-a02';root=base/'run-e1b-v15-a02'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
def tail(p,n=3):
 if not p.exists():return []
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));x=f.read().decode(errors='replace').splitlines()
 return x[-n:]
out=dict(utc=time.time(),ownership=inspect(),submitted=read(admission/'submitted.json'),supervisor=read(admission/'supervised/supervisor.json'),native=read(admission/'supervised/burn-verified.json'),source_errors=[],data_errors=[])
out['data_admission']=read(admission/'data-admission.json')
out['reclaim']=read(admission/'supervised/reclaim.json')
out['free_before_training']=read(admission/'supervised/gpus-free-before-training.json')
manifest=read(admission/'source-manifest.json')
if manifest:
 out['source_sha256']=sha(admission/'source-manifest.json')
 for p,h in manifest['files'].items():
  if sha(admission/'source'/p)!=h:out['source_errors'].append(p)
kg=read(base/'data/e1/manifest.json')
if kg:
 for p,v in kg['files'].items():
  if sha(base/'data/e1'/p)!=v['sha256']:out['data_errors'].append(p)
out['state']={n:read(root/n) for n in ['status.json','budget.json','tests_summary.json','report_complete.json']};out['runs']={}
for a in ['D','L','SO','DOT']:
 out.setdefault('checks',{})[a]={'core':read(root/('checks-'+a)/'checks.json'),'overfit':read(root/('overfit-'+a+'.json')),'check_logtail':tail(root/('checks-'+a+'.log'),8),'overfit_logtail':tail(root/('overfit-'+a+'.log'),8)}
 for s in [0,1]:
  for prefix in ['first200','E1b']:
   name=f'{prefix}-{a}-s{s}';p=root/name
   if p.exists():out['runs'][name]={'config':read(p/'config.json'),'terminal':read(p/'train_complete.json'),'last_metrics':tail(p/'metrics.jsonl'),'analysis':read(p/'analysis_complete.json'),'logtail':tail(root/(name+'.log'))}
out['exports']=[read(p) for p in sorted((root/'tier1').glob('*.json'))] if (root/'tier1').exists() else []
out['checkpoint_verification']={}
for a in ['D','L','SO','DOT']:
 for seed in [0,1]:
  name=f'first200-{a}-s{seed}';checkpoint=root/name/'final_state';complete=read(checkpoint/'complete.json')
  if complete:
   errors=None  # actual first200 hashes verified by health-a02; metadata only here
   out['checkpoint_verification'][name]={'complete':complete,'hash_errors':errors,'verification_mode':'metadata_only; actual hashes verified health-a02'}
  name=f'E1b-{a}-s{seed}';run=root/name
  if name in out['runs']:
   ev=sorted(run.glob('eval-*.json'),key=lambda p:int(p.stem.split('-')[1]))
   out['runs'][name]['recent_dev']=[read(p) for p in ev[-3:]]
   out['runs'][name]['checkpoints']=[{'path':str(p),'complete':read(p/'complete.json')} for p in sorted(run.glob('checkpoint-*'))]
out['queue_ancestry']={}
for pid in {v['ppid'] for v in out['ownership']['processes'].values()}:
 for _ in range(2):
  try:
   proc=Path('/proc')/str(pid);stat=(proc/'stat').read_text().rsplit(')',1)[1].split();parent=int(stat[1]);argv=(proc/'cmdline').read_bytes().split(b'\0')
   out['queue_ancestry'][str(pid)]={'ppid':parent,'argv':[x.decode(errors='replace') for x in argv if x],'cwd':str((proc/'cwd').resolve())}
   pid=parent
  except (OSError,ValueError):break
out['supervisor_tail']=tail(admission/'supervisor.log',12)
print(json.dumps(out,allow_nan=False))
PY

#1 +60+a
#th2-ohmv-token-order-health-20261118-a02
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,time
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process
base=Path('/mnt/local/_outputs/deep-llms_th2/token_order_20261118')
r=dict(utc=time.time(),ownership=inspect(),state={},runs={})
for f in ('status.json','terminal.json','scientific-complete.json','failure.json','backup.json','REPORT.json','execution/admission.json','execution/submitted.json'):
 p=base/f
 if p.exists():r['state'][f]=json.loads(p.read_text())
if (base/'execution/source.json').exists():
 pin=json.loads((base/'execution/source.json').read_text());r['source_actual']={p:sha(base/'execution/source'/p)==v for p,v in pin.items()}
plan=json.loads(Path('resources/token_order_plan_20261118.json').read_text());r['input_actual']={p:sha(p)==v for p,v in plan['hashes'].items()}
guard=Path('/mnt/local/_gpu_guard/DISABLED');r['guard_owner']=json.loads(guard.read_text()) if guard.exists() else None
r['active_identity']=[]
for job in r['state'].get('status.json',{}).get('active',[]):
 pid=job['pid'];record={'name':job['name'],'pid':pid}
 try:
  actual=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
  record.update(argv=actual,argv_matches=actual==job['argv'],identity=process(pid))
  parent=record['identity']['ppid'];record['parent']=process(parent) if parent>1 else None
 except FileNotFoundError:record['exited_during_snapshot']=True
 r['active_identity'].append(record)
for name in ['D-s0','D-s1','R-s0','R-s1','W-s0','W-s1']:
 run=base/'runs'/name;q={}
 for f in ('probe.json','fit/progress.json','valid/progress.json','head/progress.json','selection.json','dclm/progress.json','fwe/progress.json','complete.json'):
  p=run/f
  if p.exists():q[f]=json.loads(p.read_text())
 log=base/f'fit_eval-{name}.log'
 if log.exists():q['log_tail']=log.read_text()[-2000:]
 r['runs'][name]=q
if (base/'native-verified.json').exists():r['native']=json.loads((base/'native-verified.json').read_text())
if (base/'execution/queue.log').exists():r['queue_tail']=(base/'execution/queue.log').read_text()[-3000:]
print('TOKEN_ORDER_HEALTH',json.dumps(r),flush=True)
PY

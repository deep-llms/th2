#1 +60+a
#th2-ohmv-layer-adaptation-continuation-health-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import GUARD_HASH
from efficient_llm.warm_order.data import sha
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_continuation_a01');gate=base.with_name('layer_adaptation_20261121_gpu_gate_a03')
r=dict(utc=datetime.now(timezone.utc).isoformat(),ownership=inspect(),state={},runs={},studies={})
for name in ('status.json','failure.json','terminal.json','scientific-complete.json','execution/submitted.json','execution/admission.json','backup.json'):
 p=base/name
 if p.exists():r['state'][name]=json.loads(p.read_text())
if (gate/'gpu-admission.json').exists():
 g=json.loads((gate/'gpu-admission.json').read_text());assert sha(gate/'gpu-admission.json')==r['state']['execution/admission.json']['gate_sha256']=='8e599c548a8eea04a18691b84e934ed82af9a88cd6f0d6d8674d4154379a2ba5';r['gate']=dict(actual_sha256=sha(gate/'gpu-admission.json'),branches=g['branches'],reused_registration_sha256=g.get('reused_registration_sha256'))
source=json.loads((base/'execution/source.json').read_text());r['source_checks']={p:sha(base/'execution/source'/p)==h for p,h in source.items()}
for name in ('A-s0','B-s0','A-s1','B-s1'):
 run=base/'runs'/name;item={}
 for f in ('complete.json','checkpoint.json'):
  p=run/f
  if p.exists():item[f]=json.loads(p.read_text())
 p=run/'trace.jsonl'
 if p.exists():
  data=p.read_text();lines=data.splitlines();lines=lines if data.endswith('\n') else lines[:-1]
  item['trace_count']=len(lines);item['last_trace']=json.loads(lines[-1]) if lines else None
 snapshots={}
 for step in (0,10,50,200,500,1000,2000):
  p=run/'snapshots'/f'{step:06d}'/'checkpoint.json'
  if p.exists():
   cp=json.loads(p.read_text());snapshots[str(step)]=dict(metadata=cp,actual_sha_matches=sha(p.with_name('fullstate.pt'))==cp['sha256'])
 item['snapshots']=snapshots
 if 'complete.json' in item:item['terminal_fullstate_sha_matches']=sha(run/'fullstate.pt')==item['complete.json']['fullstate_sha256']
 p=base/('continuation-'+name+'.log')
 if p.exists():item['log_tail']=p.read_text()[-1500:]
 r['runs'][name]=item
for seed in (0,1):
 root=base/'study'/f's{seed}';item={}
 for step in (0,10,50,200,500,1000,2000):
  folder=root/f'step_{step:06d}'
  if (folder/'loss/REPORT.json').exists():
   q=json.loads((folder/'loss/REPORT.json').read_text());item[str(step)]={'contrasts':q['contrasts_nat_per_token'],'sequences':q['sequences']}
 item['terminal_report_exists']=(root/'REPORT.json').exists();p=base/f'study-s{seed}.log'
 if p.exists():item['log_tail']=p.read_text()[-1500:]
 r['studies'][str(seed)]=item
p=base/'execution/queue.log'
if p.exists():r['queue_tail']=p.read_text()[-2000:]
if r['state'].get('terminal.json',{}).get('native_verified'):
 prior=json.loads((base/'native-verified.json').read_text())['samples'][-1];now=r['ownership']
 assert not Path('/mnt/local/_gpu_guard/DISABLED').exists() and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert ownership(prior['gpus'])==ownership(now['gpus']) and all(process(int(pid))==v for pid,v in prior['processes'].items())
 r['native_handoff_current_verified']=True
 if r['state']['terminal.json']['success']:
  assert sha(base/'tier1.tar.gz')==r['state']['backup.json']['archive_sha256']
  r['archive_actual_hash_verified']=True
print('ADAPTATION_CONTINUATION_HEALTH',json.dumps(r),flush=True)
PY

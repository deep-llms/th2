#1 +60+a
#th2-ohmv-adaptation-context-health-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import GUARD_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2/adaptation_context_20261121_a01');old=base.with_name('layer_adaptation_20261121_continuation_a01')
r=dict(utc=datetime.now(timezone.utc).isoformat(),ownership=inspect(),projects={})
for label,root in [('background',base),('adaptation',old)]:
 item={}
 for name in ('status.json','failure.json','terminal.json','scientific-complete.json','execution/submitted.json'):
  p=root/name
  if p.exists():item[name]=json.loads(p.read_text())
 source=root/'execution/source.json'
 if source.exists():
  pins=json.loads(source.read_text());checks={n:sha(root/'execution/source'/n)==h for n,h in pins.items()};assert all(checks.values());item['source_pins_verified']=len(checks);item['source_manifest_sha256']=sha(source)
 if label=='background':
  item['seeds']={}
  for seed in (0,1):
   d=root/'prepared'/f's{seed}'/'study';e={}
   for name in ('progress.json','complete.json','REPORT.json'):
    p=d/name
    if p.exists():e[name]=json.loads(p.read_text())
   gp=d.parent/'gate.json'
   if gp.exists():e['gate.json']=json.loads(gp.read_text())
   log=root/f'study-s{seed}.log'
   if log.exists():e['log_tail']=log.read_text()[-2500:]
   item['seeds'][str(seed)]=e
 p=root/'backup.json'
 if p.exists():
  b=json.loads(p.read_text());item['backup']=dict(archive_sha256=b['archive_sha256'],archive_bytes=b['archive_bytes'],node_replay=b['node_replay']);assert sha(root/'tier1.tar.gz')==b['archive_sha256'];item['archive_actual_hash_verified']=True
  if label=='adaptation':item['backup']['fullstate_copies']=len(b['node_fullstate_copies'])
 if item.get('terminal.json',{}).get('native_verified'):
  item['native_handoff_receipt_sha256']=sha(root/'native-verified.json')
  if label=='background':
   prior=json.loads((root/'native-verified.json').read_text())['samples'][-1];now=r['ownership'];assert ownership(prior['gpus'])==ownership(now['gpus']) and all(process(int(pid))==v for pid,v in prior['processes'].items());assert not Path('/mnt/local/_gpu_guard/DISABLED').exists() and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH;item['current_native_verified']=True
 r['projects'][label]=item
print('ADAPTATION_CONTEXT_HEALTH',json.dumps(r),flush=True)
PY

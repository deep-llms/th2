#1 +60+a
#th2-ohmv-attention-shift-health-20261122-a02
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/attention_shift_20261121_a02')
r=dict(utc=datetime.now(timezone.utc).isoformat(),ownership=inspect(),seeds={})
for name in ('execution/submitted.json','status.json','failure.json','terminal.json','scientific-complete.json'):
 p=base/name
 if p.exists():r[name]=json.loads(p.read_text())
for seed in (0,1):
 root=base/'prepared'/f's{seed}';d={}
 for name in ('complete.json','study/progress.json','study/REPORT.json'):
  p=root/name
  if p.exists():d[name]=json.loads(p.read_text())
 for branch in ('C','S'):
  work=root/'work'/branch;e={}
  for name in ('numerical.json','gate.json'):
   p=work/name
   if p.exists():
    v=json.loads(p.read_text());e[name]=dict(sha256=sha(p),pass_=v['pass_'],kind=v['kind'],step=v.get('step'),exact_resume=v.get('exact_resume'),max_abs=v['max_abs'])
  for name in ('checkpoint.json','complete.json'):
   p=work/'fit'/name
   if p.exists():e[name]=json.loads(p.read_text())
  for name in ('trace.jsonl','validation.jsonl'):
   p=work/'fit'/name
   if p.exists():e[name]=[json.loads(l) for l in p.read_text().splitlines()[-2:]]
  d[branch]=e
 r['seeds'][str(seed)]=d
for name in ('study-s0.log','study-s1.log','execution/queue.log'):
 p=base/name
 if p.exists():r[name]=p.read_text()[-4000:]
p=base/'execution/source.json'
if p.exists():
 pins=json.loads(p.read_text());assert all(sha(base/'execution/source'/n)==h for n,h in pins.items());r['frozen_source_sha256']=sha(p);r['source_pins_verified']=len(pins)
p=base/'backup.json'
if p.exists():
 v=json.loads(p.read_text());r['backup']=dict(archive_sha256=v['archive_sha256'],archive_bytes=v['archive_bytes'],node_replay=v['node_replay'],copies=len(v['node_correction_backups']),members=len(v['files'])+1)
for name in ('backup.json','terminal.json','native-verified.json'):
 p=base/name
 if p.exists():r[name+'_sha256']=sha(p)
text=subprocess.check_output(['ps','-eo','pid,ppid,etimes,args'],text=True)
r['project_processes']=[s for s in text.splitlines() if 'attention_shift' in s and ('node run' in s or 'shift_queue' in s)]
print('ATTENTION_SHIFT_HEALTH',json.dumps(r),flush=True)
PY

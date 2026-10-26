#1 +30+a
#th2-6y3o-efficient-kv-recovery-health-20261027-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'READONLY'
import json,hashlib,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-recovery-20261027-a02')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ['jobs.json','source-manifest.json','accelerate-env.log','supervised/supervisor.json','supervised/burn-verified.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/complete.json','supervised/run/summary.json','supervised/run/smoke/result.json','supervised/run/A/result.json','supervised/run/B/result.json','supervised/run/A/validation.json','supervised/run/B/validation.json']:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)>2000000:raise ValueError('Unexpected oversized result '+rel)
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for p in sorted(root.rglob('*.log')):
 if p.stat().st_size:
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
p=root/'probe/manifest.json'
if p.exists():
 raw=p.read_bytes();m=json.loads(raw)
 old=json.loads(Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-20261027-a03/probe/manifest.json').read_text())
 identities={k:{r[k] for records in old['documents'].values() for r in records} for k in ['text_sha256','tokens_sha256']}
 overlap={k:len(v & {r[k] for r in m['documents']['calibration']}) for k,v in identities.items()}
 print('PROBE_IDENTITY',json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),metadata={k:v for k,v in m.items() if k!='documents'},previous_document_overlap=overlap,identical_validation_records=m['documents']['evaluation']==old['documents']['evaluation'])),flush=True)
processes=subprocess.check_output(['ps','-eo','pid,ppid,etimes,pcpu,pmem,args'],text=True)
for line in processes.splitlines():
 if ('efficient_llm.diagnostic prepare' in line or 'scripts.launch_group_diagnostic' in line or 'efficient_llm.kv_mlp_study' in line) and not any(x in line for x in ['bash -c','sh -c']):print('PROCESS',line[:1500],flush=True)
print('HEALTH_COMPLETE',flush=True)
READONLY

#1 +30+a
#th2-6y3o-dot-grouped-status-20261020-a13
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'REMOTE_PY'
from pathlib import Path
import datetime,hashlib,json,re,time
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-grouped-20261020-a01')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
for name in ['launch.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/checks/summary.json','supervised/burn-verified.json']:
 p=root/name
 if not p.exists():print('MISSING',name,flush=True);continue
 print('RECEIPT_SHA256',name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
 j=json.loads(p.read_text())
 if name.endswith('summary.json'):
  j={**{k:v for k,v in j.items() if k not in ('checks','benchmarks')},'checks':{k:(v.get('status') if isinstance(v,dict) else [x.get('status','recorded') for x in v]) for k,v in j.get('checks',{}).items()},'benchmarks':{k:{q:v[q] for q in ('seconds_per_update','tokens_per_second','peak_allocated_bytes')} for k,v in j.get('benchmarks',{}).items()}}
 print('RECEIPT',name,json.dumps(j),flush=True)
for name in ['supervisor.log','supervised/run/grouped_verification.log']:
 p=root/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));s=f.read().decode(errors='replace')
  print('LOG_TAIL',name,'\n'+s,flush=True)
checks=root/'supervised/run/checks'
logs=sorted(checks.rglob('*.log'),key=lambda p:p.stat().st_mtime) if checks.exists() else []
for p in logs[-2:]:
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3500));s=f.read().decode(errors='replace')
 print('ACTIVE_LOG',str(p.relative_to(root)),'\n'+s,flush=True)
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH
live=inspect();print('GPU_LIVE',json.dumps(live),flush=True)
proof=root/'supervised/burn-verified.json'
if proof.exists():
 j=json.loads(proof.read_text())
 assert j['collective_progress_verified'] and not live['guard_disabled']
 assert {p for g in j['gpus'] for p in g['pids']}==set(live['workers'])
 assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
 assert all(approved_launcher(process(pid)['ppid'],{str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}) for pid in live['workers'])
 log=root/'supervised/burn.log';first=log.read_text();time.sleep(10);second=log.read_text()
 pattern=r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)'
 a=re.findall(pattern,first)[-1];b=re.findall(pattern,second)[-1]
 assert burn_progress(second) and int(b[0])>int(a[0]) and float(b[1])>float(a[1])
 print('LIVE_BURN_COLLECTIVES_VERIFIED',json.dumps({'before':a,'after':b,'guard_released':True}),flush=True)
REMOTE_PY

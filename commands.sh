#1 +30+a
#th2-6y3o-native12-pipeline-status-20261020-a09
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'PY'
from pathlib import Path
import datetime,json,subprocess,resource
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a01')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
print('RESOURCE_LIMITS',json.dumps({'open_files':resource.getrlimit(resource.RLIMIT_NOFILE),'processes':resource.getrlimit(resource.RLIMIT_NPROC),'mem_available':next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))}),flush=True)
for name in ['preparation-failed.json','assets-verified.json','cache-ready.json','launch.json','supervised/run/run.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/resume-check/complete.json','supervised/run/real-smoke25.json','supervised/run/real-smoke50.json','supervised/run/baseline-5000.json']:
 p=root/name
 if not p.is_file():continue
 j=json.loads(p.read_text())
 if name=='assets-verified.json':j={k:j[k] for k in ['status','files_verified','splits']}
 if name=='cache-ready.json':j={k:j[k] for k in ['status','seconds','train_sequences','validation_sequences']}
 if name=='supervised/run/run.json':j={**{k:v for k,v in j.items() if k!='jobs'},'jobs':[{k:v for k,v in x.items() if k!='artifacts'} for x in j['jobs']]}
 if name.endswith(('-smoke25.json','-smoke50.json','-5000.json')):j={k:v for k,v in j.items() if k!='files'}
 print('RECEIPT',name,json.dumps(j),flush=True)
for p in [Path(str(root)+'.prepare.log'),root/'supervisor.log',*(root/'supervised/run').glob('*.log')]:
 if not p.is_file():continue
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-32768));lines=f.read().decode(errors='replace').splitlines()
 print('LOG_TAIL',str(p),flush=True)
 print('\n'.join(lines[-12:]),flush=True)
for session in ['native12_prepare_20261020_a01','native12_train_20261020_a01']:
 exists=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
 if exists:
  r=subprocess.run(['tmux','display-message','-p','-t',session,'#{session_name} pane_dead=#{pane_dead} pane_pid=#{pane_pid}'],capture_output=True,text=True)
  print('SESSION',session,r.returncode,r.stdout.strip(),flush=True)
 else:print('SESSION',session,'absent',flush=True)
from scripts.gpu_process import inspect
live=inspect()
print('GPU_LIVE',json.dumps({'time':live['time'],'gpus':live['gpus'],'guard_disabled':live['guard_disabled']}),flush=True)
PY

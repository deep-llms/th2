#1 +20+a
#th2-wc69-family-f12-health-20261111-a06
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
from pathlib import Path
from scripts.a2_family_f12_health import health
from scripts.a2_family_f10_health import health as f10health
from scripts.a2_family_f11_health import health as f11health
from eval.a2_family_f12 import read
from efficient_llm.diagnostic_data import sha256
import time
def stable_health(fn,arg):
 for attempt in range(3):
  try:
   fn(arg);return
  except (FileNotFoundError,ProcessLookupError) as error:
   print('HEALTH_RETRY',attempt,repr(error),flush=True)
   if attempt==2:raise
   time.sleep(1)
stable_health(health,'resources/a2_family_f12_plan_20261111.json')
stable_health(f11health,'resources/a2_family_f11_plan_20261111.json')
stable_health(f10health,'/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01')
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f12-queue-20261111-a01')
for name in ('prepared_config.json','actual_inputs.json'):
 p=root/name
 if p.is_file():
  value=read(p)
  print('F12_CPU_PROOF',name,sha256(p),'verified',value.get('verified'),'files',len(value.get('files',{})),'heal_rows',value.get('heal_rows'),'f11_source_verified',value.get('f11_source_verified'),flush=True)
p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01/results/a2_family_20261107/f10/verification.json')
print('F10_VERIFICATION_SHA',sha256(p),flush=True)
import subprocess
r=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f11-20261111-a01/results/a2_family_20261107/f11')
for sub in ('nine','new','calibration'):
 for p in sorted((r/sub).glob('*.log')):
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-1200));tail=f.read().decode(errors='replace')
  print('F11_SCORER_LOG',str(p),p.stat().st_size,tail,flush=True)
rows=subprocess.check_output(['ps','-eo','pid=,ppid=,etimes=,pcpu=,stat=,comm='],text=True).splitlines()
owned={152650};changed=True
while changed:
 changed=False
 for line in rows:
  f=line.split()
  if int(f[1]) in owned and int(f[0]) not in owned:owned.add(int(f[0]));changed=True
for line in rows:
 if int(line.split()[0]) in owned:print('F11_PROCESS',line,flush=True)
for name in ('verification.json','backup/verified.json'):
 p=r/name
 if p.is_file():
  value=read(p)
  if 'files' in value:value['file_count']=len(value.pop('files'))
  if isinstance(value.get('backup'),dict) and 'files' in value['backup']:value['backup']['file_count']=len(value['backup'].pop('files'))
  print('F11_FINAL_PROOF',name,sha256(p),value,flush=True)
CODE

#1 +20+a
#th2-wc69-family-f12-health-20261111-a09
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
from pathlib import Path
import time,json
from collections import Counter
from scripts.a2_family_f12_health import health
from eval.a2_family_f12 import read
from efficient_llm.diagnostic_data import sha256
for attempt in range(3):
 try:
  health('resources/a2_family_f12_plan_20261111.json');break
 except (FileNotFoundError,ProcessLookupError) as error:
  print('HEALTH_RETRY',attempt,repr(error),flush=True)
  if attempt==2:raise
  time.sleep(1)
queue=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f12-queue-20261111-a01')
for name in ('prepared_config.json','actual_inputs.json'):
 p=queue/name;v=read(p)
 print('F12_CPU_PROOF',name,sha256(p),'verified',v.get('verified'),'files',len(v.get('files',{})),'heal_rows',v.get('heal_rows'),'f11_source_verified',v.get('f11_source_verified'),flush=True)
r=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f12-20261111-a01/results/a2_family_20261107/f12')
counts=Counter();failed=[];paths=list((r/'masks').glob('*-rank*.json'))+list((r/'heals').glob('*/tests-rank*.json'))
for p in paths:
 v=read(p)
 for check in v.get('checks',[]):
  counts[check['test']]+=1
  if not check['passed']:failed.append(dict(path=str(p),check=check))
 if v.get('nonfinite_values'):failed.append(dict(path=str(p),nonfinite=v['nonfinite_values']))
print('F12_CHECK_SUMMARY',json.dumps(dict(files=len(paths),counts=dict(counts),failures=failed)),flush=True)
for p in sorted((r/'heals').glob('*/result.json')):
 v=read(p)
 print('F12_RESULT',p.parent.name,sha256(p),json.dumps({k:v.get(k) for k in ('flags','training_seconds','elapsed_seconds','registered_run','roundtrip_difference','delta_post','run_id')}),flush=True)
for name in ('verification.json','backup/verified.json'):
 p=r/name
 if p.is_file():
  v=read(p)
  if 'files' in v:v['file_count']=len(v.pop('files'))
  print('F12_FINAL_PROOF',name,sha256(p),json.dumps(v),flush=True)
CODE

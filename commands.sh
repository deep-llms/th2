#1 +20+a
#th2-wc69-family-v17-health-20261109-a09
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import json,datetime,subprocess
from scripts.gpu_process import inspect,process
from scripts.a2_family_v17_pause import digest
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01')
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inspection=inspect(),pause={},pause_log=(root/'pause.log').read_text()[-18000:])
for p in sorted((root/'pause').glob('*.json')):r['pause'][p.name]=dict(path=str(p),sha256=digest(p),value=json.loads(p.read_text()))
for name in ['supervisor.json','burn-verified.json']:
 p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01/supervised')/name
 if p.exists():r[name]=dict(path=str(p),sha256=digest(p),value=json.loads(p.read_text()))
p=Path('/mnt/local/_gpu_guard/DISABLED');r['guard']=json.loads(p.read_text()) if p.exists() else None
r['chains']={}
for g in r['inspection']['gpus']:
 for pid in g['pids']:
  chain=[]
  while pid>1:
   v=process(pid);chain.append(v);pid=v['ppid']
  r['chains'][str(g['index'])]=chain
admission=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-admission-20261109-a02')
run=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02')
r['admission_log']=(admission/'admission.log').read_text()[-18000:] if (admission/'admission.log').exists() else None
r['v17']={}
for rel in ['progress.json','launch.json','jobs.json','source-manifest.json','supervised/supervisor.json','results/a2_family_20261107/config/plan.json','results/a2_family_20261107/manifests/actual_inputs.json','results/a2_family_20261107/manifests/actual_inputs_at_gpu_admission.json','results/a2_family_20261107/progress.json','results/a2_family_20261107/ledger.json','results/a2_family_20261107/backup/verified.json']:
 p=run/rel
 if p.exists():r['v17'][rel]=dict(path=str(p),sha256=digest(p),value=json.loads(p.read_text()))
if (run/'source-manifest.json').exists():
 m=json.loads((run/'source-manifest.json').read_text());bad=[p for p,h in m['files'].items() if digest(run/'source'/p)!=h]
 r['v17_source']=dict(files=len(m['files']),changed=bad)
r['v17_logs']={p.name:p.read_text()[-6500:] for p in sorted((run/'results/a2_family_20261107/logs').glob('*.log'))}
r['v17_checks']={p.relative_to(run).as_posix():json.loads(p.read_text()) for p in (run/'results/a2_family_20261107/heals').glob('*/tests-rank*.json')}
r['v17_results']={}
for p in (run/'results/a2_family_20261107/heals').glob('*/result.json'):
 v=json.loads(p.read_text());r['v17_results'][p.parent.name]=dict(path=str(p),sha256=digest(p),delta_post=v.get('delta_post'),final=v['scores']['final']['loss'] if v['scores']['final'] else None,registered_run=v['registered_run'],flags=v['flags'])
r['v17_supervisor_log']=(run/'supervisor.log').read_text()[-7000:] if (run/'supervisor.log').exists() else None
r['v17_recipes']={}
for p in (run/'results/a2_family_20261107/heals').glob('*/recipe.json'):
 v=json.loads(p.read_text());r['v17_recipes'][p.parent.name]=dict(path=str(p),sha256=digest(p),value={k:v[k] for k in ('run_id','dataset','arm','teacher','initial_parent','block','removed','steps','tokens','lr','training','source','registered_run','spec_sha256')})
import re
r['v17_nonfinite_counts']={p.name:len(re.findall(r'NONFINITE_KD_FLAG|Traceback \(most recent call last\)|loss[\x27\x22]?\s*:\s*(?:nan|inf)',p.read_text(),flags=re.I)) for p in (run/'results/a2_family_20261107/logs').glob('*.log')}
print('V17_HEALTH '+json.dumps(r),flush=True)
PYCPU

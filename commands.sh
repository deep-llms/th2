#1 +20+a
#th2-wc69-family-v17-health-20261109-a01
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
print('V17_HEALTH '+json.dumps(r),flush=True)
PYCPU

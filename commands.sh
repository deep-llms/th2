#1 +60+a
#th2-ohmv-adaptation-directions-cpuhealth-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYHEALTH'
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/adaptation_directions_20261121_a01')
r=dict(utc=datetime.now(timezone.utc).isoformat(),ownership=inspect(),seeds={})
for seed in (0,1):
 root=base/'prepared'/f's{seed}';d={}
 for name in ('calibration/progress.json','calibration/calibration.json','config.json'):
  f=root/name
  if f.exists():
   obj=json.loads(f.read_text())
   d[name]=obj if name.endswith('progress.json') else dict(sha256=sha(f),bytes=f.stat().st_size)
 r['seeds'][str(seed)]=d
for name in ('prepared/prepared.json','cpu-terminal.json'):
 f=base/name
 if f.exists():r[name]=json.loads(f.read_text());r[name+'_sha256']=sha(f)
pins=json.loads((base/'source.json').read_text());assert all(sha(base/'source'/n)==h for n,h in pins.items());r['source_pins_verified']=len(pins)
text=subprocess.check_output(['ps','-eo','pid,ppid,etimes,args'],text=True)
r['preparer_processes']=[s for s in text.splitlines() if 'efficient_llm.warm_order.adaptation_directions_node prepare' in s]
print('ADAPTATION_DIRECTIONS_CPU_HEALTH',json.dumps(r),flush=True)
PYHEALTH

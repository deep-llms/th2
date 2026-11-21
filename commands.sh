#1 +60+a
#th2-ohmv-layer-adaptation-gpu-health-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2/layer_adaptation_20261121_gpu_gate_a01')
r=dict(utc=datetime.now(timezone.utc).isoformat(),ownership=inspect(),state={},runs={})
for name in ('status.json','failure.json','terminal.json','qualification-complete.json','gpu-admission.json','execution/submitted.json'):
 p=base/name
 if p.exists():r['state'][name]=json.loads(p.read_text())
source=json.loads((base/'execution/source.json').read_text());r['source_checks']={p:sha(base/'execution/source'/p)==h for p,h in source.items()}
for name in ('A-s0','B-s0','A-s1','B-s1'):
 run=base/'runs'/name;item={}
 for f in ('complete.json','numerical.json','whole/checkpoint.json','split/checkpoint.json'):
  p=run/f
  if p.exists():item[f]=json.loads(p.read_text())
 for sub in ('whole','split'):
  p=run/sub/'trace.jsonl'
  if p.exists():
   lines=p.read_text().splitlines();item[sub+'_trace_count']=len(lines);item[sub+'_trace_tail']=[json.loads(x) for x in lines[-1:]]
 p=base/('qualify-'+name+'.log')
 if p.exists():item['log_tail']=p.read_text()[-1800:]
 if 'complete.json' in item:
  proof=item['complete.json'];item['actual_hashes']={sub:sha(run/sub/'fullstate.pt')==proof[key] for sub,key in [('whole','whole_fullstate_sha256'),('split','split_fullstate_sha256')]}
  assert all(item['actual_hashes'].values()) and sha(run/'numerical.json')==proof['numerical_sha256']
 r['runs'][name]=item
p=base/'execution/queue.log'
if p.exists():r['queue_tail']=p.read_text()[-1800:]
if r['state'].get('terminal.json',{}).get('native_verified'):
 prior=json.loads((base/'native-verified.json').read_text());native='/tmp/llm_pretrain_burn.py';digest=prior['script_sha256'];now=r['ownership']
 assert not Path('/mnt/local/_gpu_guard/DISABLED').exists() and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert ownership(now['gpus'])==ownership(prior['samples'][-1]['gpus'])
 assert all(process(int(pid))==p for pid,p in prior['samples'][-1]['processes'].items())
 assert sha(native)==digest and len(now['workers'])==8 and all(approved_launcher(process(pid)['ppid'],{native:digest}) for pid in now['workers'])
 assert all(r['source_checks'].values())
 r['native_handoff_current_verified']=True
 if r['state']['terminal.json']['success']:
  assert all('complete.json' in item for item in r['runs'].values())
  assert sha(base/'gpu-admission.json')==r['state']['qualification-complete.json']['gpu_admission_sha256']
  r['gpu_admission_actual_sha256']=sha(base/'gpu-admission.json')
print('ADAPTATION_GPU_HEALTH',json.dumps(r),flush=True)
PY

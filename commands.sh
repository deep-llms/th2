#1 +15+a
#th2-q359-screen-eval-preflight-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json,math,subprocess
from pathlib import Path
from datetime import datetime,timezone
base=Path('/mnt/local/_outputs/deep-llms_th2')
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
def tail(p,n=10):
 if not p.exists():return
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-24000));s=f.read().decode(errors='replace')
 print('TAIL',str(p),'\n'+'\n'.join(s.splitlines()[-n:]),flush=True)
for name in ('q359-proxy-12layer-screen22-20261013-a01','q359-proxy-followup11-20261014-a01'):
 root=base/name;run=root/'supervised/run';print('ROOT',name,flush=True)
 p=root/'supervised/supervisor.json'
 if p.exists():print('SUPERVISOR',p.read_text(),flush=True)
 p=run/'run.json'
 if p.exists():
  x=json.loads(p.read_text());print('STATUS',x['status'],flush=True)
  for j in x['jobs']:
   print('JOB',json.dumps(j),flush=True)
   if j['status'] not in ('ok','pending','queued'):tail(run/(j['name']+'.log'),12)
 tail(root/'supervisor.log',12)
 tail(root/'supervised/burn.log',4)
 patterns=('seed-42/*/result.json',) if 'screen22' in name else ('*/seed-42/*/result.json',)
 for pattern in patterns:
  for p in sorted(run.glob(pattern)):
   r=json.loads(p.read_text());e=r.get('evaluation',{})
   print('RESULT',json.dumps(dict(path=str(p),arm=r.get('arm'),step=r.get('global_step'),status=r.get('status'),evaluation=e,training_cost=r.get('training_cost'),ppl=math.exp(e['eval_lm_loss']) if 'eval_lm_loss' in e else None)),flush=True)
 for relative in ('preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json'):
  p=root/relative
  if p.exists():print('HANDOFF',relative,p.read_text(),flush=True)
 r=subprocess.run(['tmux','list-panes','-t',name,'-F','#{pane_dead} #{pane_pid} #{pane_current_command}'],capture_output=True,text=True)
 print('PANE',r.returncode,r.stdout.strip(),flush=True)
subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv'],check=True)
PYREMOTE
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 - <<'PYENV'
import importlib.metadata as md,json
from pathlib import Path
print('EVAL_VERSIONS',json.dumps({p:md.version(p) for p in ('torch','transformers','accelerate','datasets','lm_eval','flash-attn-4','nvidia-cutlass-dsl')}))
base=Path('/mnt/local/_outputs/deep-llms_th2')
for p in base.glob('q359-expanded*/manifest.json'): print('BENCHMARK_MANIFEST',str(p),p.stat().st_size)
root=base/'q359-proxy-12layer-screen22-20261013-a01/supervised/run/seed-42'
for arm in ('A','P6-iso','P6-iso-deep','P6-iso-adapter','P7-simple','P7-simple-lm'):
 p=root/arm/'checkpoint-2500';c=json.loads((p.parent/'train_config.json').read_text());t=json.loads((p/'trainer_state.json').read_text())
 print('EVAL_CHECKPOINT',arm,p.exists(),t['global_step'],c['model_config']['num_hidden_layers'],c['pilot']['proxy_lambda_max'],(p/'model.safetensors').stat().st_size)
print('DATA_EXISTS',Path('/mnt/local/_data/deep-llms_th2/expanded-20261012').is_dir())
PYENV

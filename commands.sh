#1 +30+a
#th2-6y3o-dot-diagnostics-status-20261021-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
from pathlib import Path
import json,hashlib
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-diagnostics-20261021-a01')
for name in ['preparation-failed.json','launch.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/summary.json']:
 p=root/name
 if p.exists():print('RECEIPT',name,p.read_text().replace('\n',' '),flush=True)
for p in sorted((root/'supervised/run').glob('*/result.json')):
 j=json.loads(p.read_text());s=j['route_study'];h=s['router_health'];u=s['route_diagnostics']
 print('RESULT',p.parent.name,json.dumps(dict(loss=j['loss'],ppl=j['perplexity'],interventions={k:v['loss'] for k,v in s['interventions'].items()},beta=h['beta'],beta_range=h['beta_range'],raw_gap=h['raw_gap'],load=u['module_load_share'],mean_distinct=u['mean_distinct_modules'],reuse=u['reuse_rate'],checks=h['passed'],gradient=s.get('gradient_check'))),flush=True)
 print('FILE_HASH',str(p.relative_to(root)),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
logs=[Path(str(root)+'.prepare.log'),root/'supervisor.log',*sorted((root/'supervised/run').glob('*.log'))]
for p in logs:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2600));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',str(p),'\n'+tail,flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
if (root/'supervised/burn-verified.json').exists():
 from scripts.native_baseline_pipeline import burn_gate
 from scripts.train_then_burn import burn_progress
 burn_gate(root)
 assert burn_progress((root/'supervised/burn.log').read_text())
 print('CURRENT_BURN_HANDOFF_VERIFIED',flush=True)
PY

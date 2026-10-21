#1 +30+a
#th2-6y3o-dot-diagnostics-gradient-status-20261021-a03
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
from pathlib import Path
import json,hashlib,tarfile,io
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-diagnostics-gradient-20261021-a02')
for name in ['preparation-failed.json','launch.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/summary.json']:
 p=root/name
 if p.exists():print('RECEIPT',name,p.read_text().replace('\n',' '),flush=True)
for p in sorted((root/'supervised/run').glob('*/result.json')):
 j=json.loads(p.read_text())['route_study']
 print('GRADIENT_RESULT',p.parent.name,json.dumps({k:j[k] for k in ['gradient_check','centered_gradient_check']}),flush=True)
for p in [Path(str(root)+'.prepare.log'),root/'supervisor.log',*sorted((root/'supervised/run').glob('*.log'))]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2200));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',str(p),'\n'+tail,flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
if (root/'supervised/burn-verified.json').exists():
 from scripts.native_baseline_pipeline import burn_gate
 from scripts.train_then_burn import burn_progress,digest
 burn_gate(root)
 assert burn_progress((root/'supervised/burn.log').read_text())
 print('CURRENT_BURN_HANDOFF_VERIFIED',flush=True)
 if (root/'supervised/run/summary.json').exists() and json.loads((root/'supervised/run/summary.json').read_text())['status']=='ok':
  names=['previous_gradient/P1-A.json','previous_gradient/P1-B.json','previous_gradient/run.json','original_results.tar.gz','original_results_manifest.json','launch.json','accelerate_env.txt','preflight.json',
    'supervised/supervisor.json','supervised/burn-verified.json','supervised/reclaim.json',
    'supervised/gpus-free-before-training.json','supervised/run/run.json',
    'supervised/run/complete.json','supervised/run/summary.json','supervised/run/P1-A/result.json',
    'supervised/run/P1-B/result.json']
  hashes={}
  with tarfile.open(root/'results_bundle.tar.gz','x:gz') as tar:
   for name in names:
    p=root/name
    assert p.is_file() and not p.is_symlink() and p.stat().st_size<20*1024**2
    hashes[name]=digest(p);tar.add(p,arcname=name,recursive=False)
   payload=json.dumps(hashes,sort_keys=True).encode()
   info=tarfile.TarInfo('member_sha256.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
  report=dict(sha256=digest(root/'results_bundle.tar.gz'),files=hashes)
  with (root/'results_bundle_manifest.json').open('x') as f:json.dump(report,f,indent=2)
  print('BUNDLE_READY',json.dumps(report),flush=True)
PY

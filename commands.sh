#1 +30+a
#th2-6y3o-dot-N-final-R-preflight-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'PY'
from pathlib import Path
import datetime,json,hashlib,shutil
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
for name in ['preparation-failed.json','control-comparison.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/N-A.verified.json','supervised/supervisor.json','supervised/burn-verified.json']:
 p=root/name
 if p.exists():print('RECEIPT',name,p.read_text().replace('\n',' '),flush=True)
path=root/'prepared/runs/N-A'
for name in ['train_results.json','final_eval.json']:
 p=path/name
 if p.exists():print('RESULT',name,p.read_text().replace('\n',' '),flush=True)
p=path/'metrics.jsonl'
if p.exists():
 rows=[json.loads(s) for s in p.read_text().splitlines()]
 for r in rows[-3:]:print('LATEST_METRIC',json.dumps(r),flush=True)
state=json.loads((root/'supervised/supervisor.json').read_text())
if state.get('finished_at') and state.get('training_status')=='ok':
 from scripts.native_baseline_pipeline import burn_gate
 from scripts.train_then_burn import burn_progress
 assert (root/'supervised/run/complete.json').is_file()
 assert json.loads((root/'supervised/run/run.json').read_text())['status']=='ok'
 final=path/'checkpoint-4768'
 names=['optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(8)]
 missing=[n for n in names if not (final/n).is_file() or (final/n).stat().st_size==0]
 weights=list(final.glob('*.safetensors'))
 trainer=json.loads((final/'trainer_state.json').read_text())
 check=dict(step=trainer['global_step'],max_steps=trainer['max_steps'],missing=missing,weight_files=len(weights),checkpoints=len(list(path.glob('checkpoint-*'))))
 print('FINAL_CHECK',json.dumps(check),flush=True)
 assert not missing and weights and check['step']==check['max_steps']==4768
 for name in ['train_results.json','final_eval.json','metrics.jsonl']:
  p=path/name;print('RESULT_SHA256',name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
 burn_gate(root)
 assert burn_progress((root/'supervised/burn.log').read_text())
 assert not Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-R-20261021-a01').exists()
 assert shutil.disk_usage('/mnt/local').free>200*1024**3
 print('N_COMPLETE_R_PREFLIGHT_PASSED',flush=True)
else:
 print('N_NOT_YET_COMPLETE_WITH_VERIFIED_HANDOFF',flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
PY

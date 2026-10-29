#1 +30+a
#th2-6y3o-a2-stage1-downstream-replay-verify-20261030-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
cd /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-downstream-20261029-a01/source
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket
from pathlib import Path
from datetime import datetime,timezone
from scripts.a2_remaining import read_locked
from eval.a2_suite import collect_files
from eval.a2_stage1_stats import MODELS,aligned_accuracy_arrays,paired_accuracy_intervals
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-downstream-20261029-a01')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sm=read(root/'source-manifest.json');assert sm['code_commit']=='13686edd9cb8d7914b0f2c44c3c362c435a808f6'
for rel,h in sm['files'].items():assert sha(root/'source'/rel)==h
sv=read(root/'supervised/supervisor.json');assert sv['training_status']=='ok' and sv['training_returncode']==0
terminal=read(root/'supervised/run/complete.json');assert terminal['status']=='ok' and len(terminal['jobs'])==1
job=terminal['jobs'][0];assert job['name']=='stage1-downstream' and job['returncode']==0
for item in job['artifacts']:assert sha(root/'supervised/run'/item['path'])==item['sha256']
plan=read(root/'plan.json');done=read(root/'supervised/run/downstream/complete.json');assert done['status']=='ok' and done['models']==6
ids={n:read_locked(lock) for n,lock in plan['identities'].items()};assert set(ids)==set(MODELS)
actual=collect_files(ids,done['prepared'],done['scores'],done['replays']);assert actual==read_locked(done['results'])
replays={n:read_locked(lock) for n,lock in done['replays'].items()}
paired=paired_accuracy_intervals(aligned_accuracy_arrays(replays),resamples=10000,seed=0)
assert paired==read_locked(done['additional_results']['paired_accuracy'])
print('INDEPENDENT_REPLAY_VERIFIED',json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),source=sm['code_commit'],source_files=len(sm['files']),models=6,requests=78259,tasks=9,score_hashes={n:l['sha256'] for n,l in done['scores'].items()},replay_hashes={n:l['sha256'] for n,l in done['replays'].items()},results=done['results'],paired_accuracy=done['additional_results']['paired_accuracy'],resamples=10000,seed=0)),flush=True)
PY

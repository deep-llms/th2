#1 +30+a
#th2-6y3o-dot-dispatch-opt-20261020-a03
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE_PY'
from pathlib import Path
import importlib.metadata as md,json,socket,subprocess
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import digest,GUARD_HASH,BURN_HASH,APPROVED_BURNS,approved_launcher
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-opt-20261020-a03')
assert not root.exists(),'Choose fresh output root'
previous=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-profile-20261020-a01')
assert json.loads((previous/'supervised/supervisor.json').read_text())['training_status']=='ok'
assert json.loads((previous/'supervised/burn-verified.json').read_text())['collective_progress_verified']
prior=json.loads((previous/'preflight.json').read_text())
for name in ('bdot/model.py','bdot/grouped.py','bdot/trainer.py'):
 assert digest(previous/'reference'/name)==prior['source_sha256'][name]
retry=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-opt-20261020-a02')
assert json.loads((retry/'supervised/burn-verified.json').read_text())['collective_progress_verified']
report=inspect()
assert report['host']=='thiennh-p6-6y3o-worker-0'
assert len(report['gpus'])==8 and all('B200' in x['name'] for x in report['gpus'])
assert not report['guard_disabled'],'Another workload owns the guard'
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert digest('resources/llm_pretrain_burn.py')==BURN_HASH
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
for pid in report['workers']:
 launcher=pid if approved_launcher(pid,approved) else process(pid)['ppid']
 assert launcher>1 and approved_launcher(launcher,approved),'Unrecognized GPU workload'
expected={'torch':'2.14.1','transformers':'5.9.0','accelerate':'1.13.0','datasets':'4.8.5','tokenizers':'0.22.2','safetensors':'0.8.0','flash-attn-4':'4.0.0b33'}
report['versions']={p:md.version(p) for p in expected}
assert all(report['versions'][p].split('+')[0]==v for p,v in expected.items())
report['source_sha256']={name:digest(name) for name in ('bdot/model.py','bdot/grouped.py','bdot/trainer.py','train.py','scripts/bdot_grouped_verify.py')}
root.mkdir(parents=True)
(root/'preflight.json').write_text(json.dumps(report,indent=2))
print('GROUPED_PREFLIGHT',json.dumps(report),flush=True)
print('GROUPED_PREFLIGHT_PASSED',flush=True)
REMOTE_PY
unset CUDA_VISIBLE_DEVICES
/mnt/local/conda-py311/envs/train/bin/python3.11 scripts/bdot_b200_launch.py --root /mnt/local/_outputs/deep-llms_th2/6y3o-dot-dispatch-opt-20261020-a03 --session dot_dispatch_opt_20261020_a03 --host thiennh-p6-6y3o-worker-0 --jobs-config resources/bdot_b200_dispatch.json

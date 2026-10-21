#1 +30+a
#th2-6y3o-dot-v3-evidence-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import hashlib,io,json,tarfile
from pathlib import Path
from scripts.gpu_process import inspect
p=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01')
checks=p/'supervised/run/checks'
live=inspect()
observation={'live':live,'queue':json.loads((p/'supervised/run/run.json').read_text())}
for name in ['supervised/run/checks/summary.json','supervised/supervisor.json','supervised/burn-verified.json']:
 f=p/name
 if f.is_file():observation[name]=json.loads(f.read_text())
run=p/'prepared/runs/P1-A-prime-s0'
for name in ['metrics.jsonl','retention.jsonl']:
 f=run/name
 if f.is_file():observation[name]=[json.loads(l) for l in f.read_text().splitlines() if l][-15:]
observation['checkpoints']={}
for cp in sorted(run.glob('checkpoint-*')):
 if cp.is_dir():
  observation['checkpoints'][cp.name]={f.name:f.stat().st_size for f in cp.iterdir() if f.is_file()}
for f in sorted((p/'supervised/run').glob('*.attempt*.log'),key=lambda f:f.stat().st_mtime)[-1:]:
 print('TRAIN_LOG',f.name,f.read_text(errors='replace')[-10000:],flush=True)
print('V3_OBSERVATION',json.dumps(observation),flush=True)
if observation['supervised/run/checks/summary.json']['status']=='passed':
 (p/'launch-observation-a01.json').write_text(json.dumps(observation,indent=2)+'\n')
 files=[p/s for s in ['initial-preflight.json','preflight.json','launch.json','accelerate_env.txt','gated-jobs.json',
 'prepared/budget.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/checks/summary.json',
 'supervised/run/checks/T1.json','supervised/run/checks/v2_parity.json','supervised/run/checks/v3_numerics.json','launch-observation-a01.json']]
 files+=list((p/'prepared').glob('*.train.json'))+list((p/'prepared').glob('*.model.json'))
 files+=list(checks.glob('T8_*.json'))+list(checks.glob('resume_*/summary.json'))
 files+=list(checks.glob('carry_*/result.json'))+list(checks.glob('carry_*/routes_sample.npz'))
 files+=list(run.glob('*_recipe.json'))+list(run.glob('train_config.json'))
 content={str(f.relative_to(p)):f.read_bytes() for f in sorted(set(files))}
 archive=p/'launch-evidence-a01.tar.gz'
 if archive.exists():raise RuntimeError('Evidence archive already exists')
 with tarfile.open(archive,'w:gz') as t:
  for name,data in content.items():
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
   t.addfile(info,io.BytesIO(data))
 manifest={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
  'members':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in content.items()}}
 with (p/'launch-evidence-a01.manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
 print('EVIDENCE_READY',manifest['sha256'],len(content),archive.stat().st_size,flush=True)
PY

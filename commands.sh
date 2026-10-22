#1 +30+a
#th2-6y3o-dot-mlp-status-20261022-a05
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib
from pathlib import Path
from scripts.gpu_process import inspect
from accelerate.commands.config.config_args import default_config_file
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-mlp-20261022-a01')
read=lambda p:json.loads(p.read_text())
result={'live':inspect()}
for name in ['launch.json','supervised/reclaim.json','supervised/gpus-free-before-training.json',
             'supervised/run/run.json','supervised/supervisor.json','supervised/run/checks/summary.json',
             'supervised/run/complete.json','supervised/run/mlp-results.json']:
 p=root/name
 if p.is_file():result[name]=read(p)
result['accelerate_cache_matches']=Path(default_config_file).read_bytes()==Path('resources/accelerate_b200.yaml').read_bytes()
if (root/'accelerate_env.txt').is_file():result['accelerate_env']=(root/'accelerate_env.txt').read_text()
for name in ['supervisor.log','supervised/run/mlp_acceptance.log']:
 p=root/name
 if p.is_file():result[name]=p.read_text(errors='replace')[-5000:]
checks=root/'supervised/run/checks'
if checks.exists():
 result['check_logs']={p.name:p.read_text(errors='replace')[-2000:] for p in checks.glob('*.log')}
 for folder in checks.glob('resume_*'):
  if folder.is_dir():
   result[str(folder.relative_to(root))]={p.name:p.read_text(errors='replace')[-1800:] for p in folder.glob('*.log')}
for p in sorted((root/'prepared/runs').glob('*')):
 if not p.is_dir():continue
 item={}
 for name in ['train_results.json','final_eval.json']:
  if (p/name).is_file():item[name]=read(p/name)
 if (p/'metrics.jsonl').is_file():item['metrics']=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines() if l][-3:]
 item['checkpoints']=[q.name for q in p.glob('checkpoint-*')]
 if item['checkpoints']:
  from scripts.bdot_job import latest_complete_checkpoint
  try:
   cp=latest_complete_checkpoint(p)
   item['checkpoint']={'path':str(cp),'step':read(cp/'trainer_state.json')['global_step'],
     'files':{f.name:f.stat().st_size for f in cp.iterdir() if f.is_file()}}
  except RuntimeError as error:item['checkpoint_observation']=str(error)
 if (p/'training_recipe.json').is_file():
  item['source_matches']=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in read(p/'training_recipe.json')['code_sha256'].items())
  item['data_matches_D']=read(p/'data_recipe.json')==read(root.parent/'6y3o-dot-pilot-20261020-a01/prepared/runs/D/data_recipe.json')
 result[p.name]=item
for p in (root/'supervised/run').glob('*.attempt*.log'):
 result[p.name]=p.read_text(errors='replace')[-1800:]
print('MLP_STATUS '+json.dumps(result),flush=True)
(root/'status-a05.json').write_text(json.dumps(result,indent=2))

if result.get('supervised/run/checks/summary.json',{}).get('status')=='passed':
 import tarfile,io
 names=['initial-preflight.json','preflight.json','launch.json','accelerate_env.txt','gated-jobs.json',
 'supervised/reclaim.json','supervised/gpus-free-before-training.json','status-a05.json',
 'supervised/run/checks/summary.json','supervised/run/checks/old_parity.json',
 'supervised/run/checks/mlp_numerics.json','supervised/run/checks/resume_Partial-25/summary.json',
 'supervised/run/checks/resume_MLP-only-random/summary.json','supervised/run/checks/carry_D/result.json']
 names += [str(p.relative_to(root)) for pattern in ['prepared/*.json','supervised/run/checks/T8_*.json',
 'prepared/runs/*/training_recipe.json','prepared/runs/*/data_recipe.json'] for p in root.glob(pattern)]
 data={}
 for name in sorted(set(names)):
  p=root/name
  assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
  data[name]=p.read_bytes()
 archive=root/'launch-evidence-a01.tar.gz';manifest=root/'launch-evidence-a01.manifest.json'
 assert not archive.exists() and not manifest.exists()
 with tarfile.open(archive,'x:gz') as tar:
  for name,b in data.items():
   info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o600
   tar.addfile(info,io.BytesIO(b))
 value={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in data.items()}}
 manifest.write_text(json.dumps(value,indent=2))
 print('MLP_EVIDENCE '+json.dumps({'archive':str(archive),'manifest':str(manifest),'sha256':value['sha256'],'files':len(data)}),flush=True)
else:print('MLP_EVIDENCE_PENDING',flush=True)
REMOTE

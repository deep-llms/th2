#1 +30+a
#th2-6y3o-dot-mlp-status-20261022-a03
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
 if (p/'training_recipe.json').is_file():
  item['source_matches']=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in read(p/'training_recipe.json')['code_sha256'].items())
  item['data_matches_D']=read(p/'data_recipe.json')==read(root.parent/'6y3o-dot-pilot-20261020-a01/prepared/runs/D/data_recipe.json')
 result[p.name]=item
for p in (root/'supervised/run').glob('*.attempt*.log'):
 result[p.name]=p.read_text(errors='replace')[-1800:]
print('MLP_STATUS '+json.dumps(result),flush=True)
(root/'status-a03.json').write_text(json.dumps(result,indent=2))
REMOTE

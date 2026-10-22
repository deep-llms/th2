#1 +30+a
#th2-6y3o-dot-C1-health-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import hashlib,json
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.bdot_job import latest_complete_checkpoint
from accelerate.commands.config.config_args import default_config_file
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-C1-20261022-a01')
run=root/'prepared/runs/C1-s0'
read=lambda p:json.loads(p.read_text())
health={'live':inspect()}
for name in ['launch.json','C1-contract.json','supervised/run/run.json','supervised/supervisor.json',
             'supervised/reclaim.json','supervised/gpus-free-before-training.json']:
    p=root/name
    if p.is_file():health[name]=read(p)
health['accelerate_cache_matches']=Path(default_config_file).read_bytes()==Path('resources/accelerate_b200.yaml').read_bytes()
health['accelerate_env']=(root/'accelerate_env.txt').read_text() if (root/'accelerate_env.txt').is_file() else None
for name in ['supervisor.log','supervised/run/C1-s0.log','supervised/run/C1-s0.verified.attempt1.log']:
    p=root/name
    if p.is_file():health[name]=p.read_text(errors='replace')[-5000:]
if (run/'training_recipe.json').is_file():
    recipe=read(run/'training_recipe.json')
    health['source_hashes_match']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in recipe['code_sha256'].items())
    pred=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01')
    health['same_data_recipe_as_C2']=read(run/'data_recipe.json')==read(pred/'prepared/runs/C2-s0/data_recipe.json')
    health['train_args']=read(root/'prepared/C1-s0.train.json')
    health['routing']=read(root/'prepared/C1-s0.model.json')['bdot']
if (run/'metrics.jsonl').is_file():
    health['metrics']=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines() if l][-3:]
if list(run.glob('checkpoint-*')):
    try:
        cp=latest_complete_checkpoint(run);state=read(cp/'trainer_state.json')
        health['checkpoint']={'path':str(cp),'global_step':state['global_step'],'max_steps':state['max_steps'],
            'files':{f.name:f.stat().st_size for f in cp.iterdir() if f.is_file()}}
    except RuntimeError as e:health['checkpoint_observation']=str(e)
print('C1_HEALTH',json.dumps(health),flush=True)
with (root/'launch-health-a01.json').open('x') as f:json.dump(health,f,indent=2)
PY

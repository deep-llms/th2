#1 +60+a
#th2-tjx3-proxy-remaining-2500-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-remaining-2500-20261006-a01
TASK_SESSION=tjx3-proxy-remaining-2500-20261006-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
cat > "$TASK_ROOT/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" <<'PY'
import hashlib, importlib.metadata as md, json, shutil, sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from deep_kv.__main__ import jobs
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH, digest
from run_experiments import load_jobs
from train import load_text
root=Path(sys.argv[1]);project=Path.cwd()
for package,expected in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('flash-attn-4','4.0.0b33'),('datasets','4.8.5'),('nvidia-cutlass-dsl','4.8.0')]:
    assert md.version(package).split('+')[0]==expected,(package,md.version(package))
assert shutil.disk_usage(root).free>500*2**30
inspection=inspect()
assert inspection['host']=='thiennh-p6-tjx3-worker-0' and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination)
assert source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
for key,value in dict(arm='A',attention_backend='fa4',seed=42,data_seed=42,stop_after=2500,max_steps=28600,warmup_steps=1430,
    per_device_train_batch_size=16,gradient_accumulation_steps=4,block_size=2048,
    isolate_documents=True,proxy_screen=True).items():assert recipe[key]==value,(key,recipe.get(key))
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):assert Path(recipe[key]).exists()
for key in ('data_dir','eval_data_dir'):
    raw=load_text(recipe[key]);assert len(raw)>0 and 'text' in raw.column_names
    print('ARROW_TEXT_DATA_VERIFIED',key,len(raw),raw._fingerprint,flush=True)
# Previously validated throughput configuration: micro16, no activation recomputation.
recipe.update(checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False)
recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe,indent=2))
import copy
from pathlib import Path
from deep_kv.__main__ import jobs
from run_experiments import load_jobs

ARMS=['P7-simple','P7','P4-iso','P6','P5','P7-mlp','P7-kq','P7-ems','P4','P6-iso']
BASELINE='/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A'
ROWS='/mnt/local/_outputs/deep-llms_th2/proxy-final-attention-check-20261005-a01/prepared/train.json'

def build(root,recipe):
    recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe,indent=2))
    smoke={**recipe,'logging_steps':1,'eval_rows':32,'monitor_rows':32}
    smoke_path=root/'smoke-recipe.json';smoke_path.write_text(json.dumps(smoke,indent=2))
    manifest={'jobs':[]}
    def placed(job,prefix,name):
        job=copy.deepcopy(job);job['name']=name
        job['argv']=[arg.replace('{run_dir}','{run_dir}/'+prefix) for arg in job['argv']]
        for output in job['required_outputs']:output['path']=prefix+'/'+output['path']
        return job
    for job in jobs(recipe_path,stop_after=2500,arms=['A',*ARMS],seeds=[42],reuse_baselines={42:BASELINE})['jobs']:
        if 'gpus' in job:
            arm=job['argv'][job['argv'].index('--arm')+1]
            result='checks/'+arm+'-numerics.json'
            manifest['jobs'].append(dict(name='numerics-'+arm,gpus=[0],argv=['{python}','-u','-m','scripts.check_fa4_proxy','worker',
                '--arm',arm,'--recipe',str(recipe_path),'--rows',ROWS,'--output','{run_dir}/'+result],
                required_outputs=[dict(path=result,json_equals={'status':'passed'})]))
            smoke_jobs=jobs(smoke_path,stop_after=25,arms=[arm],seeds=[42])['jobs']
            train_job=next(j for j in smoke_jobs if 'gpus' in j)
            manifest['jobs'].append(placed(train_job,'smoke','smoke-'+arm))
            manifest['jobs'].append(dict(name='validate-smoke-'+arm,argv=['{python}','-m','scripts.check_fa4_proxy','validate',
                '--run-dir','{run_dir}/smoke','--steps','25','--arms',arm,'--attention-backend','fa4',
                '--output','{run_dir}/checks/'+arm+'-smoke-validation.json'],
                required_outputs=[dict(path='checks/'+arm+'-smoke-validation.json',json_equals={'status':'passed'})]))
            guard=r'''
import json,sys
from pathlib import Path
from deep_kv.report import comparable_config
smoke,baseline,output=map(Path,sys.argv[1:])
a=json.loads((smoke/'train_config.json').read_text());b=json.loads((baseline/'train_config.json').read_text())
assert a['training']['logging_steps']==1 and a['data']['eval_rows']==a['data']['monitor_rows']==32
assert a['train_fingerprint']==b['train_fingerprint'],'Training data/order changed'
# These are the ONLY smoke-only recipe differences. The validation subset
# has a different fingerprint because it selects 32 rather than 4882 rows.
a['training']['logging_steps']=b['training']['logging_steps']
for key in ('eval_rows','monitor_rows'):a['data'][key]=b['data'][key]
a['eval_fingerprint']=b['eval_fingerprint']
assert comparable_config(a)==comparable_config(b),'Recipe differs from completed A beyond documented smoke settings'
with output.open('x') as f:json.dump(dict(status='passed',train_fingerprint=a['train_fingerprint']),f)
print('MATCHED_PRODUCTION_RECIPE_VERIFIED',sys.argv[1],flush=True)
'''
            manifest['jobs'].append(dict(name='guard-recipe-'+arm,argv=['{python}','-c',guard,
                '{run_dir}/smoke/seed-42/'+arm,BASELINE,'{run_dir}/checks/'+arm+'-recipe.json'],
                required_outputs=[dict(path='checks/'+arm+'-recipe.json',json_equals={'status':'passed'})]))
        manifest['jobs'].append(placed(job,'training','training-'+job['name']))
        if 'gpus' in job:
            manifest['jobs'].append(dict(name='validate-production-'+arm,argv=['{python}','-m','scripts.check_fa4_proxy','validate',
                '--run-dir','{run_dir}/training','--steps','2500','--arms',arm,'--attention-backend','fa4',
                '--output','{run_dir}/checks/'+arm+'-production-validation.json'],
                required_outputs=[dict(path='checks/'+arm+'-production-validation.json',json_equals={'status':'passed'})]))
    path=root/'jobs.json';path.write_text(json.dumps(manifest,indent=2));load_jobs(path)
    production=[j for j in manifest['jobs'] if j['name'].startswith('training-seed-42-arm-')]
    assert len(production)==10 and [j['name'].split('training-seed-42-arm-')[1] for j in production]==ARMS
    assert all(j['gpus']==list(range(8)) for j in production)
    assert len({j['name'] for j in manifest['jobs']})==len(manifest['jobs'])
    return manifest

assert Path(BASELINE,'result.json').is_file()
assert Path(ROWS).is_file() and len(json.loads(Path(ROWS).read_text()))==16
# Immutable local input identities from the established asset manifest.
assets=json.loads(Path('resources/qwen3_base_assets.json').read_text())
for name,expected in assets['files'].items():
    assert digest(Path(recipe['tokenizer_name'])/name)==expected,('Asset changed',name)
manifest=build(root,recipe)
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(host=inspection['host'],at=inspection['time'],arms=ARMS,seed=42,
    fresh_start=True,stop_after=2500,schedule_steps=28600,warmup_steps=1430,tokens_per_update=1048576,
    accelerate_cache=str(destination),accelerate_sha256=digest(source),source_sha256={p:digest(project/p) for p in
    ['train.py','deep_kv/proxy.py','deep_kv/proxy_estimators.py','deep_kv/proxy_training.py','deep_kv/proxy_memory.py','deep_kv/report.py','scripts/check_fa4_proxy.py','scripts/train_then_burn.py']}),indent=2))
print('REMAINING_PROXY_PRODUCTION_PREFLIGHT_PASSED',str(destination),inspection['workers'],flush=True)
PY
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-proxy-remaining-2500-20261006-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 80 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

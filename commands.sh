#1 +60+a
#th2-tjx3-proxy-p1-alpha1-2500-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-p1-alpha1-2500-20261005-a01
TASK_SESSION=tjx3-proxy-p1-alpha1-2500-20261005-a01
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
CUDA_VISIBLE_DEVICES='' python -u - "$TASK_ROOT" <<'PRE'
import json,shutil,sys,importlib.metadata as md
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
root=Path(sys.argv[1]);project=Path.cwd()
assert shutil.disk_usage(root).free>180*2**30
for package,expected in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('flash-attn-4','4.0.0b33'),('datasets','4.8.5'),('nvidia-cutlass-dsl','4.8.0')]:
    assert md.version(package).split('+')[0]==expected
status=inspect()
assert status['host']=='thiennh-p6-tjx3-worker-0' and not status['guard_disabled']
assert len(status['gpus'])==8 and all('B200' in gpu['name'] for gpu in status['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in status['workers'])
(root/'inspection.json').write_text(json.dumps(status,indent=2))
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination)
assert source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
print('PREFLIGHT',status['time'],status['workers'],str(destination),flush=True)
from deep_kv.__main__ import jobs
old=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02/supervised/run/seed-42/P1-block')
saved=json.loads((old/'train_config.json').read_text())
assert json.loads((old.parent/'P1-block-validation.json').read_text())['status']=='passed'
assert saved['train_fingerprint']=='6e4708f1e818fb44' and saved['eval_fingerprint']=='08432871cf987d61'
recipe=json.loads((project/'baseline_a_fa4.b200.json').read_text())
recipe.update(arm='P1-block',proxy_alpha_init=1.)
for key,value in recipe.items():
    if key in ('output_dir','stop_after','proxy_alpha_init'):continue
    values=[saved[section][key] for section in ('model','data','pilot','training') if key in saved[section]]
    if key=='report_to':value=[value] if isinstance(value,str) else value
    assert values and all(v==value for v in values),(key,values,value)
assert saved['pilot'].get('proxy_alpha_init',0.)==0.
assert saved['pilot']['proxy_lambda_max']==.1 and saved['pilot']['proxy_warmup_steps']==250
items=[]
for phase,steps in [('smoke',3),('training',2500)]:
    current={**recipe}
    if phase=='smoke':current.update(logging_steps=1,eval_rows=32,monitor_rows=32)
    path=root/(phase+'-recipe.json');path.write_text(json.dumps(current,indent=2))
    manifest=jobs(path,stop_after=steps,arms=('P1-block',),seeds=[42])
    for job in manifest['jobs']:
        job['name']=phase+'-'+job['name']
        job['argv']=[arg.replace('{run_dir}','{run_dir}/'+phase) for arg in job['argv']]
        for output in job['required_outputs']:output['path']=phase+'/'+output['path']
        job['timeout_seconds']=21600 if 'gpus' in job else 600
        items.append(job)
    output=phase+'/validation.json'
    items.append(dict(name='validate-'+phase,timeout_seconds=600,
        argv=['{python}','-u','-m','scripts.check_fa4_proxy','validate',
              '--run-dir','{run_dir}/'+phase,'--arms','P1-block','--steps',str(steps),
              '--output','{run_dir}/'+output],
        required_outputs=[dict(path=output,json_equals={'status':'passed','steps':steps})]))
(root/'jobs.json').write_text(json.dumps({'jobs':items},indent=2))
print('ALPHA_INIT_RUN',json.dumps(dict(arm='P1-block',alpha_init=1.,lambda_max=.1,
    aux_warmup=250,seed=42,steps=2500,full_schedule=28600,warmup=1430,world_size=8,
    tokens_per_update=1048576,source_recipe=str(old),fresh=True)),flush=True)

PRE
accelerate env
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_proxy_alpha_init -v > "$TASK_ROOT/cpu-tests.log" 2>&1
tail -n 7 "$TASK_ROOT/cpu-tests.log"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-proxy-p1-alpha1-2500-20261005-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 70 "$TASK_ROOT/launch.log"

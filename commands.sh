#1 +60+a
#th2-tjx3-p4-routing-fix-checks-20261006-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/p4-routing-fix-checks-20261006-a02
TASK_SESSION=tjx3-p4-routing-fix-checks-20261006-a02
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
cat > "$TASK_ROOT/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
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
previous=Path('/mnt/local/_outputs/deep-llms_th2/p4-routing-fix-checks-20261006-a01/supervised/supervisor.json')
previous_state=json.loads(previous.read_text())
assert previous_state['training_status']=='failed' and previous_state['burn']['collective_progress_verified']
print('PREVIOUS_HANDOFF_VERIFIED',previous_state['finished_at'],flush=True)
for package,expected in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0')]:
    assert md.version(package).split('+')[0]==expected,(package,md.version(package))
assert shutil.disk_usage(root).free>250*2**30
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
for key,value in dict(arm='A',seed=42,data_seed=42,stop_after=2500,max_steps=28600,warmup_steps=1430,
    per_device_train_batch_size=16,gradient_accumulation_steps=4,block_size=2048,
    isolate_documents=True,proxy_screen=True).items():assert recipe[key]==value,(key,recipe.get(key))
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):assert Path(recipe[key]).exists()
for key in ('data_dir','eval_data_dir'):
    raw=load_text(recipe[key]);assert len(raw)>0 and 'text' in raw.column_names
    print('ARROW_TEXT_DATA_VERIFIED',key,len(raw),raw._fingerprint,flush=True)
# Previously validated throughput configuration: micro16, no activation recomputation.
recipe.update(checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False)
recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe,indent=2))
manifest={'jobs':[]}
recipe.update(logging_steps=1,eval_rows=32,monitor_rows=32)
recipe_path.write_text(json.dumps(recipe,indent=2))
rows=Path('/mnt/local/_outputs/deep-llms_th2/proxy-final-attention-check-20261005-a01/prepared/train.json')
assert rows.is_file() and len(json.loads(rows.read_text()))==16
manifest['jobs'].append(dict(name='cuda-reference',gpus=[0],argv=['{python}','-u','-m','scripts.check_p4_launch',
    '--recipe',str(recipe_path),'--rows',str(rows),'--output','{run_dir}/numerics.json'],
    required_outputs=[dict(path='numerics.json',json_equals={'status':'passed'})]))
for stage,cutoff in [('smoke',25)]:
    for job in jobs(recipe_path,stop_after=cutoff,arms=['A','P4-iso-4h','P4-4h'],seeds=[42])['jobs']:
        job['name']=stage+'-'+job['name']
        job['argv']=[arg.replace('{run_dir}','{run_dir}/'+stage) for arg in job['argv']]
        if stage=='smoke' and 'gpus' in job:
            index=job['argv'].index(str(project/'train.py'))
            job['argv'][index:index+1]=['--module','scripts.profile_proxy_training']
        for output in job['required_outputs']:output['path']=stage+'/'+output['path']
        manifest['jobs'].append(job)
    manifest['jobs'].append(dict(name='validate-smoke',argv=['{python}','-m','scripts.check_fa4_proxy','validate',
        '--run-dir','{run_dir}/smoke','--steps','25','--arms','A','P4-iso-4h','P4-4h',
        '--attention-backend','sdpa','--output','{run_dir}/smoke/validation.json'],
        required_outputs=[dict(path='smoke/validation.json',json_equals={'status':'passed'})]))
    validation=r'''
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);summary={}
for arm in ('A','P4-iso-4h','P4-4h'):
    folder=root/'seed-42'/arm
    state=json.loads((folder/'trainer_state.json').read_text())
    profile=json.loads((folder/'component-profile.json').read_text())
    assert profile['status']=='profiled' and profile['kernel_milliseconds']
    times=[r['seconds_per_update'] for r in state['log_history'] if 'seconds_per_update' in r and 10<=r['step']<=25]
    assert len(times)==16
    summary[arm]=dict(median_seconds_per_update=statistics.median(times),profile=profile)
for arm in ('P4-iso-4h','P4-4h'):
    summary[arm]['overhead_fraction']=summary[arm]['median_seconds_per_update']/summary['A']['median_seconds_per_update']-1
(root/'throughput.json').write_text(json.dumps(dict(status='passed',arms=summary),indent=2))
print('P4_FULL_SIZE_CHECKS_PASSED',json.dumps(summary),flush=True)
'''
    manifest['jobs'].append(dict(name='summarize-throughput',argv=['{python}','-c',validation,'{run_dir}/smoke'],
        required_outputs=[dict(path='smoke/throughput.json',json_equals={'status':'passed'})]))
path=root/'jobs.json';path.write_text(json.dumps(manifest,indent=2));load_jobs(path)
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(host=inspection['host'],at=inspection['time'],arms=['A','P4-iso-4h','P4-4h'],seed=42,
    fresh_start=True,stop_after=25,schedule_steps=28600,warmup_steps=1430,tokens_per_update=1048576,
    accelerate_cache=str(destination),accelerate_sha256=digest(source),source_sha256={p:digest(project/p) for p in
    ['train.py','deep_kv/proxy.py','deep_kv/proxy_estimators.py','deep_kv/proxy_training.py',
     'scripts/check_p4_launch.py','scripts/profile_proxy_training.py','scripts/train_then_burn.py']}),indent=2))
print('P4_CHECKS_PREFLIGHT_PASSED',str(destination),inspection['workers'],flush=True)
PY
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-p4-routing-fix-checks-20261006-a02-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 80 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

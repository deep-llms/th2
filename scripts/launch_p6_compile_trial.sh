#!/usr/bin/env bash
# Bounded experimental compilation study; original weights and environments untouched.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=${2:?Pass verified hostname}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1042 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/compiler-cache"
export TORCHINDUCTOR_COMPILE_THREADS=1
python -u - "$TASK_ROOT" "$TASK_HOST" <<'PY'
import copy,importlib.metadata as md,json,os,shutil,subprocess,sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from run_experiments import load_jobs
from deep_kv.__main__ import jobs
root,host=Path(sys.argv[1]),sys.argv[2];project=Path.cwd()
for p,v in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('datasets','4.8.5'),
            ('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0'),('wandb','0.30.0')]:
    assert md.version(p).split('+')[0]==v,(p,md.version(p))
assert shutil.disk_usage(root).free>200*2**30
with (root/'cpu-tests.log').open('x') as log:
    subprocess.run([sys.executable,'-m','unittest','tests.test_p6_compile_trial','-v'],
        stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
checkpoint=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042/P6-iso/checkpoint-10000')
expected='3e04cc0a27f1e0d5a1e97efdc1e34c11d4e821f9b34fb76898a0652fd8d28442'
assert digest(checkpoint/'model.safetensors')==expected
assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==10000
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
c=load_config_from_file(str(destination)).to_dict()
assert c['num_processes']==8 and c['mixed_precision']=='bf16' and c['distributed_type']=='MULTI_GPU'
subprocess.run(['accelerate','env'],check=True)
recipe=json.loads((project/'proxy_heads.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):assert Path(recipe[key]).exists(),key
for key,value in dict(block_size=2048,per_device_train_batch_size=16,gradient_accumulation_steps=4,
                      max_steps=28600,warmup_steps=1430,attention_backend='fa4',isolate_documents=True,
                      checkpoint_layers=False,checkpoint_aux=False,checkpoint_lm=False).items():
    assert recipe[key]==value,(key,recipe[key])
# Final evaluation is bounded; full production training order/schedule are retained.
recipe.update(eval_rows=32,monitor_rows=32,save_steps=100,save_total_limit=0,seed=1042,data_seed=1042)
path=root/'recipe.json';path.write_text(json.dumps(recipe,indent=2))
items=[dict(name='numerics',gpus=list(range(8)),timeout_seconds=2400,
    argv=['{python}','-m','accelerate.commands.launch','--config_file',str(source),
          '--main_process_port','29651','--module','scripts.p6_compile_trial','gate',
          '--checkpoint',str(checkpoint),'--checkpoint-sha256',expected,'--output','{run_dir}/numerics'],
    required_outputs=[dict(path='numerics/summary.json',json_equals=dict(status='passed',selected='cosine',checkpoint_unchanged=True))])]
for name,arm,mode in [('A-first','A','eager'),('P6-eager-first','P6-iso','eager'),
                      ('P6-compiled-first','P6-iso','selected'),('P6-compiled-second','P6-iso','selected'),
                      ('P6-eager-second','P6-iso','eager'),('A-second','A','eager')]:
    job=copy.deepcopy(next(j for j in jobs(path,stop_after=100,arms=[arm],seeds=[1042])['jobs'] if 'gpus' in j))
    job['name']=name;job['timeout_seconds']=3600
    job['argv']=[v.replace('{run_dir}','{run_dir}/'+name) for v in job['argv']]
    for out in job['required_outputs']:out['path']=name+'/'+out['path']
    index=job['argv'].index(str(project/'train.py'))
    job['argv'][index:index+1]=['--module','scripts.p6_compile_trial','train','--implementation',mode,
                              '--gate','{run_dir}/numerics/summary.json','--']
    for rank in range(8):
        job['required_outputs'].append(dict(path=f'{name}/seed-1042/{arm}/compile-trial-rank{rank}.json',json_equals=dict(status='complete')))
    items.append(job)
items.append(dict(name='summary',argv=['{python}','-m','scripts.p6_compile_trial','summary','--run-dir','{run_dir}'],
                  required_outputs=[dict(path='summary.json',json_equals=dict(status='passed',production_defaults_changed=False))]))
(root/'jobs.json').write_text(json.dumps(dict(jobs=items),indent=2));load_jobs(root/'jobs.json')
inspection=inspect();assert inspection['host']==host and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',checkpoint=str(checkpoint),
    checkpoint_sha256=expected,accelerate_sha256=digest(destination),stages=len(items)),indent=2))
print('P6_COMPILE_PREFLIGHT_PASSED',flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

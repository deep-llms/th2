#1 +120+a
#th2-tjx3-proxy-baseline-A-fa4-2500-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-baseline-A-fa4-2500-20261005-a01
TASK_SESSION=tjx3-proxy-baseline-A-fa4-2500-20261005-a01
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
recipe=json.loads(Path('baseline_a_fa4.b200.json').read_text())
for key,value in dict(arm='A',seed=42,data_seed=42,stop_after=2500,max_steps=28600,warmup_steps=1430,
    per_device_train_batch_size=16,gradient_accumulation_steps=4,block_size=2048,
    isolate_documents=True,proxy_screen=True).items():assert recipe[key]==value,(key,recipe.get(key))
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):assert Path(recipe[key]).exists()
for key in ('data_dir','eval_data_dir'):
    raw=load_text(recipe[key]);assert len(raw)>0 and 'text' in raw.column_names
    print('ARROW_TEXT_DATA_VERIFIED',key,len(raw),raw._fingerprint,flush=True)
# Previously validated throughput configuration: micro16, no activation recomputation.
recipe.update(checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False)
reference=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-2500-20261005-a01')
old=json.loads((reference/'recipe.json').read_text())
difference={k for k in old.keys()|recipe.keys() if old.get(k)!=recipe.get(k)}
assert difference=={'attention_backend','output_dir'},difference
reference=reference/'supervised/run/baseline/seed-42/A'
assert json.loads((reference/'result.json').read_text())['global_step']==2500
from deep_kv.fa4 import load_kernel
_,kernel_metadata=load_kernel()
recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe,indent=2))
manifest={'jobs':[dict(name='production-numerics',gpus=[0],
    argv=['{python}','-u','-m','scripts.validate_fa4_baseline','numerics','--recipe',str(recipe_path),'--output','{run_dir}/numerics.json'],
    required_outputs=[dict(path='numerics.json',json_equals={'status':'passed'})])]}
for stage,cutoff in [('smoke',3),('baseline',2500)]:
    for job in jobs(recipe_path,stop_after=cutoff,arms=['A'],seeds=[42])['jobs']:
        job['name']=stage+'-'+job['name']
        job['argv']=[arg.replace('{run_dir}','{run_dir}/'+stage) for arg in job['argv']]
        if stage=='smoke' and 'gpus' in job:
            job['argv'][job['argv'].index('--logging_steps')+1]='1'
        for output in job['required_outputs']:output['path']=stage+'/'+output['path']
        manifest['jobs'].append(job)
    manifest['jobs'].append(dict(name='validate-'+stage,
        argv=['{python}','-u','-m','scripts.validate_fa4_baseline','validate',
              '--arm','{run_dir}/'+stage+'/seed-42/A','--reference',str(reference),
              '--steps',str(cutoff),'--output','{run_dir}/'+stage+'/validation.json'],
        required_outputs=[dict(path=stage+'/validation.json',json_equals={'status':'passed'})]))
path=root/'jobs.json';path.write_text(json.dumps(manifest,indent=2));load_jobs(path)
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(host=inspection['host'],at=inspection['time'],arm='A',seed=42,
    attention_backend='fa4',kernel=kernel_metadata,reference=str(reference),fresh_start=True,stop_after=2500,schedule_steps=28600,warmup_steps=1430,tokens_per_update=1048576,
    accelerate_cache=str(destination),accelerate_sha256=digest(source),source_sha256={p:digest(project/p) for p in
    ['train.py','deep_kv/proxy.py','deep_kv/proxy_training.py','deep_kv/fa4.py','scripts/validate_fa4_baseline.py','scripts/train_then_burn.py']}),indent=2))
print('BASELINE_PREFLIGHT_PASSED',str(destination),inspection['workers'],flush=True)
PY
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-proxy-baseline-A-fa4-2500-20261005-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 80 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

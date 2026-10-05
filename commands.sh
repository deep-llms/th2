#1 +60+a
#th2-tjx3-proxy-fa4-screen-seed42-2500-20261005-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-fa4-screen-seed42-2500-20261005-a02
TASK_SESSION=tjx3-proxy-fa4-screen-seed42-2500-20261005-a02
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
control=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-restart-control-20261005-a01')
oldroot=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a01')
terminal=json.loads((oldroot/'supervised/supervisor.json').read_text())
assert terminal.get('finished_at') and terminal.get('burn',{}).get('collective_progress_verified')
assert (control/'stop-request.json').is_file()
oldowner=json.loads((control/'inspection.json').read_text())['supervisor']
try: assert process(oldowner['pid'])['start_ticks']!=oldowner['start_ticks']
except FileNotFoundError: pass
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
import hashlib
SCREEN_ARMS=('A','P1-block','P3-block','P1-lambda0','P3-lambda0','P1-flow','V1','V3')
from deep_kv.__main__ import jobs
from deep_kv.report import report,comparable_config
base=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A')
smoke=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-smoke-20261005-a01/supervised/run')
assert hashlib.sha256((base/'train_config.json').read_bytes()).hexdigest()=='2d198d6491282c7d8e748c63b372d5dac1efbe3eaa8bbd73c9e7d2189118f8db'
recipe_path=project/'baseline_a_fa4.b200.json'
recipe=json.loads(recipe_path.read_text())
old=json.loads((base/'train_config.json').read_text())
new=json.loads((smoke/'seed-42/A/train_config.json').read_text())
assert old['train_fingerprint']=='6e4708f1e818fb44' and old['eval_fingerprint']=='08432871cf987d61'
new['training']['logging_steps']=recipe['logging_steps']
for key in ('eval_rows','monitor_rows'):new['data'][key]=recipe[key]
new['eval_fingerprint']=old['eval_fingerprint']  # Full prefix independently recomputed by prior preflight.
assert comparable_config(old)==comparable_config(new)
for key,value in recipe.items():
    if key in ('output_dir','stop_after'):continue
    values=[old[section][key] for section in ('model','data','pilot','training') if key in old[section]]
    assert values,(key,'missing')
    if key=='report_to':value=[value] if isinstance(value,str) else value
    assert all(v==value for v in values),(key,values,value)
report(root/'baseline-preflight',('A',),baseline_dir=base,expected_step=2500,
       expected_seed=42,expected_attention_backend='fa4')
manifest=jobs(recipe_path,stop_after=2500,arms=SCREEN_ARMS,seeds=[42],reuse_baselines={42:str(base)})
training=[job for job in manifest['jobs'] if 'gpus' in job]
assert len(training)==7 and all(job['gpus']==list(range(8)) for job in training)
assert [j['name'] for j in training]==['seed-42-arm-'+arm for arm in SCREEN_ARMS if arm!='A']
assert all('--resume_from_checkpoint' not in j['argv'] for j in training)
expanded=[dict(name='clean-authorized-previous-screen', timeout_seconds=1800,
    argv=['{python}','-u','-m','scripts.clean_proxy_screen_restart','--output','{run_dir}/cleanup.json'],
    required_outputs=[dict(path='cleanup.json',json_equals={'status':'cleaned'})])]
for job in manifest['jobs']:
    job['timeout_seconds']=21600 if 'gpus' in job else 600
    expanded.append(job)
    if 'gpus' in job:
        arm=job['name'].removeprefix('seed-42-arm-')
        assert arm!='A' and arm!='P3-flow'
        output=f'seed-42/{arm}-validation.json'
        expanded.append(dict(name=f'validate-{arm}',timeout_seconds=600,
            argv=['{python}','-u','-m','scripts.check_fa4_proxy','validate','--run-dir','{run_dir}',
                  '--steps','2500','--seed','42','--arms',arm,'--output','{run_dir}/'+output],
            required_outputs=[dict(path=output,json_equals={'status':'passed','steps':2500})]))
manifest['jobs']=expanded
(root/'jobs.json').write_text(json.dumps(manifest,indent=2))
(root/'recipe.json').write_text(json.dumps(recipe,indent=2))
print('SEED42_SCREEN',json.dumps(dict(reused_baseline=str(base),new_arms=[j['name'] for j in training],
    steps=2500,full_schedule=28600,warmup=1430,tokens_per_update=1048576)),flush=True)
PRE
accelerate env
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_fa4_baseline.FA4BaselineTests.test_queue_and_scope_guards \
 tests.test_proxy_training.ProxyTrainingTests.test_report_target_matching_and_external_baseline \
 tests.test_proxy_training.ProxyTrainingTests.test_reused_baseline_queue tests.test_fa4_screen_validation tests.test_proxy_restart_cleanup -v
CUDA_VISIBLE_DEVICES='' python -m scripts.check_fa4_proxy validate \
 --run-dir /mnt/local/_outputs/deep-llms_th2/proxy-fa4-smoke-20261005-a01/supervised/run \
 --output "$TASK_ROOT/revalidated-smokes.json"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-proxy-fa4-screen-seed42-2500-20261005-a02-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 70 "$TASK_ROOT/launch.log"

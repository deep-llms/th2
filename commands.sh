#1 +60+a
#th2-tjx3-proxy-gate-sweep-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-gate-sweep-20261005-a01
TASK_SESSION=tjx3-proxy-gate-sweep-20261005-a01
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
old=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a02')
terminal=json.loads((old/'supervised/supervisor.json').read_text())
assert terminal.get('finished_at') and terminal.get('burn',{}).get('collective_progress_verified')
control=Path('/mnt/local/_outputs/deep-llms_th2/proxy-gate-sweep-control-20261005-a01')
stop=json.loads((control/'stop-request.json').read_text());owner=stop['supervisor']
try:assert process(owner['pid'])['start_ticks']!=owner['start_ticks']
except FileNotFoundError:pass
items=[]
for arm in ('P1-block','P3-block'):
    folder=old/'supervised/run/seed-42'/arm
    prior=json.loads((folder.parent/f'{arm}-validation.json').read_text())
    assert prior['status']=='passed' and prior['steps']==2500
    items.append(dict(name='evaluate-'+arm,gpus=list(range(8)),timeout_seconds=1800,
        argv=['{python}','-m','accelerate.commands.launch','--config_file',str(source),
              '-m','scripts.evaluate_proxy_gates','--arm-dir',str(folder),
              '--output','{run_dir}/'+arm],
        required_outputs=[dict(path=arm+'/summary.json',json_equals={'status':'passed','step':2500,
                              'model_state_unchanged':True,'checkpoint_unchanged':True})]))
(root/'jobs.json').write_text(json.dumps({'jobs':items},indent=2))
print('GATE_SWEEP',json.dumps(dict(arms=['P1-block','P3-block'],multipliers=[1,2,5,1],
    source=str(old),steps=2500,eval_rows=4882,world_size=8)),flush=True)
PRE
accelerate env
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_proxy_gate_sweep -v > "$TASK_ROOT/cpu-tests.log" 2>&1
tail -n 7 "$TASK_ROOT/cpu-tests.log"
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-proxy-gate-sweep-20261005-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 70 "$TASK_ROOT/launch.log"

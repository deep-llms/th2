#1 +60+a
#th2-78gg-BFG-checkpoint-only-resume-20260929-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-78gg-worker-0
cd /mnt/local/@PROJECT@
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/deep-kv-BFG-10000-20260929-a02
TASK_SESSION=deep-kv-BFG-10000-20260929-a02
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" <<'PY'
import hashlib, importlib.metadata, json, shutil, subprocess, sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, process
root=Path(sys.argv[1])
assert importlib.metadata.version('torch')=='2.14.0'
assert importlib.metadata.version('transformers')=='5.9.0'
assert shutil.disk_usage(root).free>250*2**30
source=Path('resources/accelerate_config.yaml')
destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'previous_accelerate_config.yaml')
shutil.copy2(source,destination)
assert source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
inspection=inspect()
assert inspection['host']=='thiennh-p6-78gg-worker-0' and not inspection['guard_disabled']
old=json.loads(Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01/production/burn-verified.json').read_text())['gpus']
assert [(g['index'],g['uuid']) for g in old]==[(g['index'],g['uuid']) for g in inspection['gpus']]
approved={**APPROVED_BURNS,str(Path('resources/llm_pretrain_burn.py').resolve()):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
recipe=json.loads(Path('deep_kv.b200.json').read_text())
for key in ['config_name','tokenizer_name','data_dir','eval_data_dir']:assert Path(recipe[key]).exists()
from scripts.check_optimized_resume import make_jobs
previous_root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a01/production')
handoff=json.loads((previous_root/'supervisor.json').read_text())
assert handoff['training_status']=='failed' and handoff['burn']['collective_progress_verified']
assert not (previous_root/'run/continuation').exists()
(root/'previous-handoff.json').write_text(json.dumps(handoff,indent=2))
source_root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01/production/run')
for arm in 'BFG':
 saved=json.loads((source_root/arm/'train_config.json').read_text())
 result=json.loads((source_root/arm/'result.json').read_text())
 state=json.loads((source_root/arm/'checkpoint-5000/trainer_state.json').read_text())
 assert result['global_step']==state['global_step']==5000
 assert state['max_steps']==saved['training']['max_steps']==28600
 assert saved['training']['warmup_steps']==1430 and saved['world_size']==8
 assert saved['tokens_per_update']==1048576
 assert saved['training']['per_device_train_batch_size']==16
 assert saved['training']['gradient_accumulation_steps']==4
 assert not saved['training']['ignore_data_skip']
 assert saved['data']['block_size']==2048 and saved['data']['eval_rows']==4882
 for name in ['model.safetensors','optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(8)]]:
  assert (source_root/arm/'checkpoint-5000'/name).stat().st_size>0
control_root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a01/production/run/resume-smoke/control')
prior=json.loads((control_root.parent.parent/'run.json').read_text())
assert all(any(j['name']==f'smoke-control-{arm}' and j['status']=='ok' for j in prior['jobs']) for arm in 'BFG')
for arm in 'BFG':
 assert json.loads((control_root/arm/'result.json').read_text())['global_step']==5001
 for rank in range(8):
  assert json.loads((control_root/arm/f'resume-check-{rank}.json').read_text())['status']=='ok'
make_jobs(Path('deep_kv.b200.json').resolve(),root,source_root,5000,10000,control_root)
(root/'gpu_inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps({'accelerate_cache':str(destination),
 'accelerate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'host':inspection['host'],
 'at':inspection['time'],'scope':'checkpoint-only resume gate against completed controls, then B/F/G 5000 to 10000',
 'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
 ['train.py','deep_kv/model.py','scripts/check_optimized_resume.py','scripts/stage_deep_kv_resume.py']}},indent=2))
print('OPTIMIZED_RESUME_PREFLIGHT_PASSED',flush=True)
PY
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/production" \
 --inspection "$TASK_ROOT/gpu_inspection.json" --host thiennh-p6-78gg-worker-0 --burn-session deep-kv-BFG-10000-20260929-a02-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 70 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

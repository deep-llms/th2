#1 +60+a
#th2-tjx3-downstream-2500-20261007-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/downstream-2500-20261007-a01
TASK_SESSION=tjx3-downstream-2500-20261007-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
cat > "$TASK_ROOT/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate eval_fa4
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m pip check
python -u - "$TASK_ROOT" <<'PY'
import hashlib,importlib.metadata as md,json,shutil,subprocess,sys,time
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from run_experiments import load_jobs
root=Path(sys.argv[1]); project=Path.cwd()
for package,expected in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('datasets','4.8.5'),('lm_eval','0.4.10'),('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0')]:
 assert md.version(package).split('+')[0]==expected,(package,md.version(package))
from deep_kv.fa4 import load_kernel
load_kernel()
print('EVAL_ENV_VERIFIED',sys.executable,flush=True)
manifest=project/'resources/downstream_english_20261007.json'
data=Path('/mnt/local/_data/deep-llms_th2/downstream-english-20261007')
spec=json.loads(manifest.read_text());files=[f for repo in spec['repositories'] for f in repo['files']]
# CPU-only waiting: leave the existing burns alone while controller downloads finish.
for attempt in range(60):
 missing=[f['path'] for f in files if not (data/f['path']).is_file() or (data/f['path']).stat().st_size != f['bytes']]
 if not missing:break
 print('WAITING_FOR_CONTROLLER_FILES',len(missing),flush=True);time.sleep(30)
else:raise RuntimeError('Benchmark controller download did not finish; GPUs untouched')
subprocess.run([sys.executable,'-u','-m','scripts.check_downstream_eval','data','--dataset-root',str(data),
 '--dataset-manifest',str(manifest),'--output',str(root/'data-validation.json')],check=True)
assert json.loads((root/'data-validation.json').read_text())==json.loads((project/'resources/downstream_english_reference_20261007.json').read_text()),'B200 data differs from local reference'
base=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A/checkpoint-2500')
parent=Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01/supervised/run/training/seed-42')
arms=['A','P6-iso','P7-simple','P7','P7-mlp','P7-ems','P7-kq','P6','P4','P5','P4-iso']
checkpoints=[base]+[parent/arm/'checkpoint-2500' for arm in arms[1:]]
from deep_kv.report import comparable_config
common=None
for arm,path in zip(arms,checkpoints):
 assert (path/'model.safetensors').stat().st_size>1000000000
 assert json.loads((path/'trainer_state.json').read_text())['global_step']==2500
 recipe=json.loads((path.parent/'train_config.json').read_text());assert recipe['pilot']['arm']==arm
 normalized=comparable_config(recipe)
 if common is None:common=normalized
 else:assert normalized==common
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
c=load_config_from_file(str(destination)).to_dict();assert c['num_processes']==8 and c['mixed_precision']=='bf16'
subprocess.run(['accelerate','env'],check=True)
jobs=[]
for arm,path in zip(arms,checkpoints):
 output='checks/'+arm+'-numerics.json'
 jobs.append(dict(name='numerics-'+arm,gpus=[0],timeout_seconds=600,argv=['{python}','-u','-m','scripts.check_downstream_eval','numerics',
 '--checkpoint',str(path),'--output','{run_dir}/'+output],required_outputs=[dict(path=output,json_equals={'status':'passed','arm':arm,'step':2500})]))
tasks=['hellaswag','xnli','belebele','xstorycloze','paws-x','piqa','arc_easy','arc_challenge','winogrande']
for phase,limit,batch in [('smoke',8,4),('full',None,8)]:
 argv=['{python}','-u','-m','eval.eval_parallel','--checkpoints',*map(str,checkpoints),'--bench-only','--english-only',
 '--tasks',*tasks,'--num-gpus','8','--gpu-ids',*map(str,range(8)),'--batch-size',str(batch),'--num-fewshot','0','--seed','42',
 '--dataset-root',str(data),'--dataset-manifest',str(manifest),'--output-dir','{run_dir}/'+phase,'--log','{run_dir}/'+phase+'-launcher.log']
 if limit:argv+=['--limit',str(limit)]
 jobs.append(dict(name=phase+'-evaluation',gpus=list(range(8)),timeout_seconds=14400,argv=argv,
 required_outputs=[dict(path=phase+'/checkpoints.json')]))
 output=phase+'-validation.json'
 argv=['{python}','-u','-m','scripts.check_downstream_eval','validate','--directory','{run_dir}/'+phase,
 '--data-check',str(root/'data-validation.json'),'--count','11','--output','{run_dir}/'+output]
 if limit:argv+=['--limit',str(limit)]
 jobs.append(dict(name='validate-'+phase,argv=argv,required_outputs=[dict(path=output,json_equals={'status':'passed','limit':limit})]))
(root/'jobs.json').write_text(json.dumps({'jobs':jobs},indent=2));load_jobs(root/'jobs.json')
inspection=inspect();assert inspection['host']=='thiennh-p6-tjx3-worker-0' and not inspection['guard_disabled']
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',arms=arms,tasks=tasks,checkpoints=list(map(str,checkpoints)),
 source_hashes={str(p):digest(p) for p in sorted((project/'eval').glob('*.py'))},accelerate_cache=str(destination),
 dataset_manifest_sha256=digest(manifest),environment=sys.executable),indent=2))
print('DOWNSTREAM_PREFLIGHT_PASSED',arms,flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-downstream-2500-20261007-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 60 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

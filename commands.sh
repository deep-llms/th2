#1 +60+a
#th2-tjx3-supervised-stsb-boolq-20261008-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/supervised-stsb-boolq-20261008-a01
TASK_SESSION=tjx3-supervised-stsb-boolq-20261008-a01
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
from scripts.finetune_study import make_jobs
from eval.benchmarks import local_dataset_paths
from run_experiments import load_jobs
root=Path(sys.argv[1]);project=Path.cwd()
for package,expected in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('datasets','4.8.5'),('lm_eval','0.4.10'),('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0')]:
 assert md.version(package).split('+')[0]==expected,(package,md.version(package))
from deep_kv.fa4 import load_kernel
load_kernel()
manifest=project/'resources/supervised_english_20261008_v2.json'
data=Path('/mnt/local/_data/deep-llms_th2/supervised-english-20261008-v2')
files=[f for repo in json.loads(manifest.read_text())['repositories'] for f in repo['files']]
for attempt in range(60):
 missing=[f['path'] for f in files if not (data/f['path']).is_file() or (data/f['path']).stat().st_size!=f['bytes']]
 if not missing:break
 print('WAITING_FOR_CONTROLLER_FILES',missing,flush=True);time.sleep(30)
else:raise RuntimeError('Controller files incomplete; GPUs untouched')
mappings=local_dataset_paths(data,manifest)
from datasets import load_dataset
for task,counts in [('stsb',{'train':5749,'validation':1500,'test':1379}),('boolq',{'train':9427,'validation':3270})]:
 mapping=mappings[task];ds=load_dataset(mapping['dataset_path'],**mapping['dataset_kwargs'])
 assert {k:len(v) for k,v in ds.items()}==counts
base=Path('/mnt/local/_outputs/deep-llms_th2/proxy-baseline-A-fa4-2500-20261005-a01/supervised/run/baseline/seed-42/A/checkpoint-2500')
parent=Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01/supervised/run/training/seed-42')
checkpoints={'A':str(base),'P6':str(parent/'P6/checkpoint-2500'),'P6-iso':str(parent/'P6-iso/checkpoint-2500'),'P7-simple':str(parent/'P7-simple/checkpoint-2500')}
from deep_kv.report import comparable_config
common=None;hashes={}
for arm,path in checkpoints.items():
 p=Path(path);assert (p/'model.safetensors').stat().st_size>1000000000
 assert json.loads((p/'trainer_state.json').read_text())['global_step']==2500
 recipe=json.loads((p.parent/'train_config.json').read_text());assert recipe['pilot']['arm']==arm
 normalized=comparable_config(recipe)
 if common is None:common=normalized
 else:assert normalized==common
 hashes[arm]=digest(p/'model.safetensors')
 expected={'A':'a9c3b8dce1d9e25699823fd10ef28b71e35f451499be73391a02959d05fd353b',
 'P6':'3d6df7709ad15367b14bddc7c2000e929a0e092a706487326c19feede278acb3',
 'P6-iso':'6bd57582187722caa6e7ca1b504993c40eb9cbc89f086079ca7f8b61d7ded8cf',
 'P7-simple':'d8ed9239c675244fb7737f9cbf6715c0110abf69632176f53b47fe552149bc30'}
 assert hashes[arm]==expected[arm],'Source checkpoint weight mismatch'
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
c=load_config_from_file(str(destination)).to_dict();assert c['num_processes']==8 and c['mixed_precision']=='bf16'
subprocess.run(['accelerate','env'],check=True)
subprocess.run([sys.executable,'-u','-m','scripts.finetune_study','audit-data',
 '--dataset-root',str(data),'--manifest',str(manifest),'--tokenizer',str(base),
 '--output',str(root/'data-audit.json')],check=True)
assert json.loads((root/'data-audit.json').read_text())==json.loads((project/'resources/supervised_reference_20261008.json').read_text()),'Local/remote data audit differs'
jobs=make_jobs(root,project,data,manifest,checkpoints,tasks=('stsb','boolq'));load_jobs(root/'jobs.json')
assert len(jobs)==97
inspection=inspect();assert inspection['host']=='thiennh-p6-tjx3-worker-0' and not inspection['guard_disabled']
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',arms=list(checkpoints),checkpoints=checkpoints,
 checkpoint_sha256=hashes,dataset_manifest_sha256=digest(manifest),environment=sys.executable,
 accelerate_cache=str(destination),stages=len(jobs),tasks=['stsb','boolq'],scipy_version=md.version('scipy'),
 source_hashes={str(p):digest(p) for p in [project/'eval/finetune.py',project/'scripts/finetune_study.py',project/'deep_kv/proxy_memory.py',project/'deep_kv/training.py']} ),indent=2))
print('SUPERVISED_PREFLIGHT_PASSED',list(checkpoints),len(jobs),flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-supervised-stsb-boolq-20261008-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 60 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

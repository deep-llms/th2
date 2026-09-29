#1 +60+a
#th2-78gg-performance-benchmark-20260929-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-78gg-worker-0
cd /mnt/local/@PROJECT@
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/performance-20260929-a01
TASK_SESSION=performance-20260929-a01
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
assert shutil.disk_usage(root).free>180*2**30
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
base={**recipe,'stop_after':18,'eval_strategy':'no','eval_on_start':False,'eval_rows':16,'monitor_rows':16,
      'report_to':'none','logging_steps':6,'disable_tqdm':True}
fast={'checkpoint_layers':False,'lm_chunk':512}
variants=[
 ('current-A','A','original',{},False,False),
 ('native-A','A','original',{'checkpoint_layers':False},True,False),
 ('current-B','B','original',{},False,False),
 ('no-checkpoint-B','B','original',{'checkpoint_layers':False},False,False),
 ('chunk512-B','B','original',{'lm_chunk':512},False,False),
 ('causal-B','B','causal',{},False,False),
 ('fast-B','B','causal',fast,False,False),
 ('batch32-B','B','original',{'per_device_train_batch_size':32,'gradient_accumulation_steps':2},False,False),
 ('current-F','F','original',{},False,False),
 ('fast-F','F','causal',fast,False,False),
 ('current-G','G','original',{},False,False),
 ('fast-G','G','causal',fast,False,False),
 ('perf-env-fast-B','B','causal',fast,False,True),
 ('fa4-B','B','fa4',fast,False,True),
]
items=[]
(root/'configs').mkdir()
for name,arm,backend,extra,native,perf in variants:
 cfg={**base,**extra,'arm':arm,'output_dir':str(root/'production/run'/name),'run_name':'perf-'+name}
 cfgpath=root/'configs'/f'{name}.json';cfgpath.write_text(json.dumps(cfg,indent=2))
 python='/mnt/local/conda-py311/envs/perf_env/bin/python' if perf else sys.executable
 argv=[python,'-m','accelerate.commands.launch','--config_file',str(source.resolve()),'--module',
       'scripts.benchmark_training','worker','--variant',name,'--backend',backend]
 if native:argv.append('--native')
 argv.append(str(cfgpath))
 items.append({'name':name,'gpus':list(range(8)),'timeout_seconds':1800,'argv':argv,
   'required_outputs':[{'path':f'{name}/result.json','json_equals':{'global_step':18,'arm':arm,'status':'stopped'}},
     *[{'path':f'{name}/benchmark-rank{i}.json','json_equals':{'status':'ok','rank':i}} for i in range(8)]]})
items.append({'name':'summarize','argv':['{python}','-m','scripts.benchmark_training','summarize','--root','{run_dir}',
             '--names',*[v[0] for v in variants]],'required_outputs':[{'path':'benchmark-summary.json','json_equals':{'status':'ok'}}]})
for name,value in [('jobs.json',{'jobs':items}),('variants.json',variants),('gpu_inspection.json',inspection)]:
 (root/name).write_text(json.dumps(value,indent=2))
(root/'preflight.json').write_text(json.dumps({'accelerate_cache':str(destination),
 'accelerate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'host':inspection['host'],
 'at':inspection['time'],'variants':len(variants),'tokens_per_update':1048576},indent=2))
print('PERFORMANCE_PREFLIGHT_PASSED',len(variants),'variants',flush=True)
PY
accelerate env
/mnt/local/conda-py311/envs/perf_env/bin/accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/production" \
 --inspection "$TASK_ROOT/gpu_inspection.json" --host thiennh-p6-78gg-worker-0 --burn-session performance-20260929-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 70 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0

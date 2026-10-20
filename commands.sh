#1 +30+a
#th2-6y3o-dot-N-launch-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01
TASK_SESSION=dot_N_prepare_20261020_a01
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.prepare.log"
test ! -e "$TASK_ROOT.prepare.py"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
cat > "$TASK_ROOT.prepare.py" <<'PY'
from pathlib import Path
import datetime,json,os,shutil,socket,subprocess,sys
project=Path.cwd()
sys.path.insert(0,str(project))
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
from scripts.train_then_burn import burn_progress,digest,GUARD_HASH,BURN_HASH
from run_experiments import load_jobs
old=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01')
out=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01')
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
out.mkdir(parents=True,exist_ok=False)
def read(p): return json.loads(Path(p).read_text())
def write(p,j):
 with Path(p).open('x') as f:json.dump(j,f,indent=2)
def run(script,*args,environment=None):
 print('N_PREPARATION',script,flush=True)
 subprocess.run([sys.executable,str(project/script),*map(str,args)],check=True,
                cwd=project,env=environment or os.environ.copy())
try:
 burn_gate(old)
 assert burn_progress((old/'supervised/burn.log').read_text())
 assert read(old/'supervised/run/run.json')['status']=='ok'
 assert read(old/'supervised/run/T12/summary.json')['status']=='passed'
 assert read(old/'supervised/run/T14/summary.json')['status']=='passed'
 assert shutil.disk_usage(out).free>200*1024**3
 previous=read(old/'prepared/P1-A.train.json')
 previous_model=read(old/'prepared/P1-A.model.json')
 training_receipt=read(old/'prepared/runs/P1-A/training_recipe.json')
 for name,expected in training_receipt['code_sha256'].items():
  assert digest(project/name)==expected,('Changed training source',name)
 prepared=out/'prepared'
 accelerate=project/'resources/accelerate_b200.yaml'
 run('scripts/make_bdot_pilot.py','--tokenizer',previous['tokenizer_name'],
     '--data',previous['data_dir'],'--eval-data',previous['eval_data_dir'],
     '--revision',previous['dataset_revision'],'--manifest',previous['validation_manifest'],
     '--accelerate-config',accelerate,'--expert-backend','grouped','--arms','N-A',
     '--output',prepared)
 candidate=read(prepared/'N-A.train.json')
 expected_args={k:v for k,v in previous.items() if k not in ('config_name','output_dir')}
 assert {k:v for k,v in candidate.items() if k not in ('config_name','output_dir')}==expected_args
 expected_model=read(old/'prepared/P1-A.model.json')
 expected_model['bdot']['reuse']=False
 assert read(prepared/'N-A.model.json')==expected_model
 assert candidate['max_steps']==4768 and candidate['warmup_steps']==96
 assert candidate['save_total_limit'] is None and candidate.get('stop_after') is None
 assert not Path(candidate['output_dir']).exists()
 jobs=read(prepared/'jobs.json')
 assert len(jobs['jobs'])==1 and jobs['jobs'][0]['name']=='N-A'
 jobs['jobs'][0]['timeout_seconds']=10800
 write(out/'jobs.json',jobs)
 load_jobs(out/'jobs.json')
 run('scripts/warm_training_cache.py','--arguments',prepared/'N-A.train.json',
     '--output',out/'cache-ready.json')
 cache=read(out/'cache-ready.json');prior_cache=read(old/'cache-ready.json')
 for key in ('train_sequences','validation_sequences','data_recipe','validation_recipe'):
  assert cache[key]==prior_cache[key],('Changed data contract',key)
 write(out/'control-comparison.json',dict(status='passed',
       at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference=str(old),
       training_argument_changes=['config_name','output_dir'],model_change={'bdot.reuse':False},
       training_source_unchanged=True,data_recipe_identical=True,validation_recipe_identical=True,
       validation_manifest_sha256=digest(previous['validation_manifest']),
       steps=candidate['max_steps'],warmup_steps=candidate['warmup_steps'],
       model_config_sha256=digest(prepared/'N-A.model.json'),
       training_args_sha256=digest(prepared/'N-A.train.json')))
 burn_gate(old)
 assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert digest(project/'resources/llm_pretrain_burn.py')==BURN_HASH
 live=inspect()
 assert live['host']=='thiennh-p6-6y3o-worker-0' and len(live['gpus'])==8
 assert all('B200' in g['name'] for g in live['gpus'])
 write(out/'preflight.json',live)
 run('scripts/bdot_b200_launch.py','--root',out,'--host',live['host'],
     '--session','dot_N_train_20261020_a01','--jobs-config',out/'jobs.json',
     environment={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
 print('N_QUEUE_SUBMITTED',flush=True)
except BaseException as error:
 write(out/'preparation-failed.json',dict(status='failed',error=repr(error)))
 raise
PY
printf -v TASK_COMMAND 'exec env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false %q -u %q >%q 2>&1' "$TASK_PYTHON" "$TASK_ROOT.prepare.py" "$TASK_ROOT.prepare.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tail -20 "$TASK_ROOT.prepare.log"
echo N_PREPARATION_SUBMITTED

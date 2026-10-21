#1 +30+a
#th2-6y3o-dot-R-launch-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/6y3o-dot-R-20261021-a01
TASK_SESSION=dot_R_prepare_20261021_a01
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
predecessor=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01')
out=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-R-20261021-a01')
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
out.mkdir(parents=True,exist_ok=False)
def read(p): return json.loads(Path(p).read_text())
def write(p,j):
 with Path(p).open('x') as f:json.dump(j,f,indent=2)
def run(script,*args,environment=None):
 print('R_PREPARATION',script,flush=True)
 subprocess.run([sys.executable,str(project/script),*map(str,args)],check=True,
                cwd=project,env=environment or os.environ.copy())
try:
 burn_gate(predecessor)
 assert burn_progress((predecessor/'supervised/burn.log').read_text())
 assert read(old/'supervised/run/run.json')['status']=='ok'
 assert read(predecessor/'supervised/run/run.json')['status']=='ok'
 assert (predecessor/'supervised/run/complete.json').is_file()
 n_results=read(predecessor/'prepared/runs/N-A/train_results.json')
 assert n_results['global_step']==n_results['schedule_steps']==4768
 n_final=read(predecessor/'prepared/runs/N-A/final_eval.json')
 assert n_final['eval_num_tokens']==9915698 and n_final['eval_num_sequences']==9704
 assert read(old/'supervised/run/T12/summary.json')['status']=='passed'
 assert read(old/'supervised/run/T14/summary.json')['status']=='passed'
 assert shutil.disk_usage(out).free>200*1024**3
 previous=read(old/'prepared/P1-A.train.json')
 training_receipt=read(old/'prepared/runs/P1-A/training_recipe.json')
 for name,expected in training_receipt['code_sha256'].items():
  assert digest(project/name)==expected,('Changed training source',name)
 prepared=out/'prepared'
 accelerate=project/'resources/accelerate_b200.yaml'
 run('scripts/make_bdot_pilot.py','--tokenizer',previous['tokenizer_name'],
     '--data',previous['data_dir'],'--eval-data',previous['eval_data_dir'],
     '--revision',previous['dataset_revision'],'--manifest',previous['validation_manifest'],
     '--accelerate-config',accelerate,'--expert-backend','grouped','--arms','R',
     '--output',prepared)
 candidate=read(prepared/'R.train.json')
 expected_args={k:v for k,v in previous.items() if k not in ('config_name','output_dir')}
 assert {k:v for k,v in candidate.items() if k not in ('config_name','output_dir')}==expected_args
 expected_model=read(old/'prepared/P1-A.model.json')
 expected_model['bdot'].update(routing='random',curriculum='none')
 assert read(prepared/'R.model.json')==expected_model
 assert candidate['max_steps']==4768 and candidate['warmup_steps']==96
 assert candidate['save_total_limit'] is None and candidate.get('stop_after') is None
 assert not Path(candidate['output_dir']).exists()
 jobs=read(prepared/'jobs.json')
 assert len(jobs['jobs'])==1 and jobs['jobs'][0]['name']=='R'
 jobs['jobs'][0]['timeout_seconds']=10800
 write(out/'jobs.json',jobs)
 load_jobs(out/'jobs.json')
 run('scripts/warm_training_cache.py','--arguments',prepared/'R.train.json',
     '--output',out/'cache-ready.json')
 cache=read(out/'cache-ready.json');prior_cache=read(old/'cache-ready.json')
 for key in ('train_sequences','validation_sequences','data_recipe','validation_recipe'):
  assert cache[key]==prior_cache[key],('Changed data contract',key)
 write(out/'control-comparison.json',dict(status='passed',
       at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference=str(old),predecessor=str(predecessor),
       training_argument_changes=['config_name','output_dir'],model_change={'bdot.routing':'random','bdot.curriculum':'none'},
       training_source_unchanged=True,data_recipe_identical=True,validation_recipe_identical=True,
       validation_manifest_sha256=digest(previous['validation_manifest']),
       steps=candidate['max_steps'],warmup_steps=candidate['warmup_steps'],
       model_config_sha256=digest(prepared/'R.model.json'),
       training_args_sha256=digest(prepared/'R.train.json')))
 burn_gate(predecessor)
 assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert digest(project/'resources/llm_pretrain_burn.py')==BURN_HASH
 live=inspect()
 assert live['host']=='thiennh-p6-6y3o-worker-0' and len(live['gpus'])==8
 assert all('B200' in g['name'] for g in live['gpus'])
 write(out/'preflight.json',live)
 run('scripts/bdot_b200_launch.py','--root',out,'--host',live['host'],
     '--session','dot_R_train_20261021_a01','--jobs-config',out/'jobs.json',
     environment={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
 print('R_QUEUE_SUBMITTED',flush=True)
except BaseException as error:
 write(out/'preparation-failed.json',dict(status='failed',error=repr(error)))
 raise
PY
printf -v TASK_COMMAND 'exec env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false %q -u %q >%q 2>&1' "$TASK_PYTHON" "$TASK_ROOT.prepare.py" "$TASK_ROOT.prepare.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tail -20 "$TASK_ROOT.prepare.log"
echo R_PREPARATION_SUBMITTED

#1 +30+a
#th2-6y3o-dot-v3-status-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'PY'
import hashlib,io,json,re,tarfile
from pathlib import Path
from scripts.gpu_process import inspect,process
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH
from scripts.make_bdot_pilot import V3_RUNS
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01')
sup=root/'supervised';run=sup/'run'
read=lambda p:json.loads(p.read_text())
live=inspect();state={'live':live,'queue':read(run/'run.json'),'supervisor':read(sup/'supervisor.json'),'runs':[]}
files=[run/'run.json',sup/'supervisor.json']
for name in ['complete.json','v3-results.json','v3-results.md']:
 p=run/name
 if p.is_file():
  files.append(p)
  if p.suffix=='.json':state[name]=read(p)
for arm,seed in V3_RUNS:
 name=f'{arm}-s{seed}';out=root/'prepared/runs'/name
 row={'name':name}
 for f in ['train_results.json','final_eval.json','retention.jsonl','metrics.jsonl']:
  p=out/f
  if p.is_file():
   files.append(p)
   if f.endswith('.json'):row[f]=read(p)
   elif f=='metrics.jsonl':
    logs=[json.loads(s) for s in p.read_text().splitlines() if s];last=logs[-1]
    row['progress']={k:last[k] for k in ['step','lm_loss','grad_norm','learning_rate','eval_loss','interval_wall_seconds','input_tokens_per_second_interval_wall'] if k in last}
 row['retained_checkpoints']=[p.name for p in sorted(out.glob('checkpoint-*'))]
 cp=out/'checkpoint-4768'
 if cp.is_dir():
  for f in ['trainer_state.json','config.json','training_recipe.json','data_recipe.json','train_config.json']:
   if (cp/f).is_file() and f not in ['training_recipe.json','data_recipe.json']:files.append(cp/f)
  row['final_checkpoint_files']={p.name:p.stat().st_size for p in cp.iterdir() if p.is_file()}
 for p in [run/(name+'.verified.json'),run/name/'result.json',run/name/'routes_sample.npz']:
  if p.is_file():files.append(p)
  if p.name==name+'.verified.json' and p.is_file():row['verified']=read(p)
 state['runs'].append(row)
if (sup/'burn-verified.json').is_file():
 proof=read(sup/'burn-verified.json');files.append(sup/'burn-verified.json');state['burn_receipt']=proof
 try:
  validate_burn_handoff(state['supervisor'],proof,live)
  source=str(Path.cwd()/'resources/llm_pretrain_burn.py')
  assert all(approved_launcher(process(pid)['ppid'],{source:BURN_HASH}) for pid in live['workers'])
  text=(sup/'burn.log').read_text();assert burn_progress(text)
  state['burn_handoff_live_verified']=True
  state['last_burn_progress']=re.findall(r'^.*gpu_burn_progress.*$',text,re.M)[-3:]
 except Exception as e:state['burn_verification_error']=repr(e)
for j in state['queue'].get('jobs',[]):
 if j.get('status')=='failed':
  for p in [run/(j['name']+'.log'),*run.glob(j['name']+'.verified.attempt*.log')]:
   if p.is_file():print('FAILED_LOG',p.name,p.read_text(errors='replace')[-18000:])
print('V3_STATUS',json.dumps(state),flush=True)
status=root/'status-20261022-a01.json'
with status.open('x') as f:json.dump(state,f,indent=2)
files.append(status)
content={str(p.relative_to(root)):p.read_bytes() for p in sorted(set(files))}
archive=root/'results-20261022-a01.tar.gz'
if archive.exists():raise RuntimeError('Refusing existing status archive')
with tarfile.open(archive,'w:gz') as t:
 for name,data in content.items():
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;t.addfile(info,io.BytesIO(data))
manifest={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'members':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in content.items()}}
with (root/'results-20261022-a01.manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('RESULTS_READY',manifest['sha256'],len(content),archive.stat().st_size,flush=True)
PY

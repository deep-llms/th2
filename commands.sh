#1 +30+a
#th2-6y3o-dot-router-followup-queue-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,shutil,subprocess,shlex,sys,time,os
from pathlib import Path
from scripts.gpu_process import inspect
project=Path.cwd()
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-router-20261022-a01')
old=root.parent/'6y3o-dot-mlp-20261022-a01'
pilot=root.parent/'6y3o-dot-pilot-20261020-a01'
staged=project/'queued/router_followup_20261022'
read=lambda p:json.loads(p.read_text())
manifest=staged/'SOURCE_MANIFEST.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='65afe4d86c3e154638d512f2e63c38303d9659b57ef1c4ebf96ff7e857c096c9'
for name,value in read(manifest)['files'].items():
 p=staged/name
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(staged)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==value,name
# Existing training source is verified, never rewritten by this submission.
recipes=list((old/'prepared/runs').glob('*/training_recipe.json'))
assert recipes
for p in recipes:
 for name,value in read(p)['code_sha256'].items():
  assert hashlib.sha256((project/name).read_bytes()).hexdigest()==value,name
before=inspect();assert before['host']=='thiennh-p6-6y3o-worker-0'
session='dot_router_wait_20261022_a01'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
root.mkdir(exist_ok=False)
shutil.copytree(staged,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
(root/'queue-preflight.json').write_text(json.dumps(dict(before=before,predecessor=str(old),
 source_unchanged=True,source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),indent=2))
argv=[sys.executable,'-u',str(root/'source/scripts/bdot_router_followup.py'),'wait',
 '--output',str(root),'--pilot',str(pilot),'--predecessor',str(old),
 '--predecessor-burn-script',str(project/'resources/llm_pretrain_burn.py'),
 '--host',before['host'],'--session','dot_router_20261022_a01']
command='cd '+shlex.quote(str(root/'source'))+' && exec '+shlex.join(['env','CUDA_VISIBLE_DEVICES=',*argv])
command+=' >'+shlex.quote(str(root/'wait.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
for _ in range(30):
 time.sleep(1)
 if (root/'wait-status.json').is_file():break
assert (root/'wait-status.json').is_file(),(root/'wait.log').read_text()[-4000:]
status=read(root/'wait-status.json');assert status['status']=='waiting',status
assert subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_dead}'],text=True).strip()=='0'
jobs=read(root/'gated-jobs.json')['jobs']
assert len(jobs)==14 and jobs[0]['name']=='router_acceptance'
assert all(j.get('gpus')==list(range(8)) for j in jobs[:-1])
assert [j['name'] for j in jobs[1:-1:2]]==['Partial-25-detach-s0','Partial-50-detach-s0','Partial-25-cosine-s0','Partial-50-cosine-s0','Partial-25-cosine-detach-s0','Partial-50-cosine-detach-s0']
for p in recipes:
 assert all(hashlib.sha256((project/n).read_bytes()).hexdigest()==v for n,v in read(p)['code_sha256'].items())
receipt=dict(status='queued',waiter_session=session,root=str(root),wait_status=status,jobs=[j['name'] for j in jobs],
 source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),active_source_unchanged=True,
 snapshot_files=len(read(manifest)['files']),live=inspect())
(root/'queued.json').write_text(json.dumps(receipt,indent=2))
print('ROUTER_QUEUED '+json.dumps(receipt),flush=True)
REMOTE

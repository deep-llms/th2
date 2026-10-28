#1 +30+a
#th2-6y3o-a2-snapshot32b-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import hashlib,json,shutil,socket,subprocess,sys,shlex
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-snapshot32b-20261029-a01')
out=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/snapshots/dclm_first_3p2b_20261029')
old=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a03/source/plan.json')
plan=Path('resources/a2_corpora_30b_20261029.json')
session='a2_snapshot32b_20261029_a01'
assert not root.exists() and not out.exists()
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
assert plan.read_bytes()==old.read_bytes()
assert shutil.disk_usage(out.parent.parent).free>50_000_000_000
root.mkdir();source=root/'source';source.mkdir()
names=['scripts/a2_corpus_snapshot.py','scripts/a2_prepare_corpora.py','efficient_llm/__init__.py','efficient_llm/a2/__init__.py','efficient_llm/a2/data.py','efficient_llm/diagnostic_data.py']
files={}
for name in names:
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,p);files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(plan,source/'plan.json')
(root/'source-manifest.json').write_text(json.dumps(dict(code_commit='d195e58dbe3c9d62f89c466d1ef40c66f0c4f573',files=files,plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),cpu_only=True),indent=2)+'\n')
argv=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+str(source),'nice','-n','10',sys.executable,'-u','-m','scripts.a2_corpus_snapshot','--plan',str(source/'plan.json'),'--corpus','dclm','--output',str(out),'--tokens','3200000000']
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(source),'exec '+shlex.join(argv)+' >'+shlex.quote(str(root/'snapshot.log'))+' 2>&1'],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('CORPUS_SNAPSHOT_SUBMITTED',str(root),flush=True)
PY

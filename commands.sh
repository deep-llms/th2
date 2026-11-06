#1 +20+a
#th2-wc69-family-cpu-prepare-20261107-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=16
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import json,shutil,hashlib,subprocess,socket,shlex,sys,os
from efficient_llm.a2_family.common import write,sha,SPEC_SHA
if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
source=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02');session='a2_family_prepare_20261107_a02'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing preparation session')
root.mkdir(parents=True,exist_ok=False);frozen=root/'source';files={}
selected=[]
for directory in ('efficient_llm','scripts'):
 selected.extend((source/directory).rglob('*.py'))
selected.extend(source/name for name in ['docs/A2_family_spec.md','resources/a2_family_downloads_20261107.json','resources/a2_family_data_inputs_20261107.json','resources/a2_family_cx_reference_20261107.json','resources/a2_family_lr_state_20261107.json'])
for p in selected:
 rel=p.relative_to(source);q=frozen/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);files[str(rel)]=sha(q)
if sha(frozen/'docs/A2_family_spec.md')!=SPEC_SHA:raise ValueError('Wrong spec')
commit='2bad917ee20f9f07557bb156177011933e47cb67';write(root/'source-manifest.json',dict(files=files,code_commit=commit))
cmd='exec '+shlex.join([sys.executable,'-u','-m','scripts.a2_family_prepare','--root',str(root)])+' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),cmd],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
write(root/'launch.json',dict(root=str(root),session=session,code_commit=commit,cpu_only=True,source_manifest_sha256=sha(root/'source-manifest.json')))
print('FAMILY_CPU_PREPARATION_SUBMITTED',root,session,commit,flush=True)
PYCPU

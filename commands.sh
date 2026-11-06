#1 +20+a
#th2-wc69-family-mix-repair-20261107-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=4
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import json,shutil,hashlib,subprocess,socket,shlex,sys,os
from efficient_llm.a2_family.common import write,sha,SPEC_SHA
if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
source=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a02');session='a2_family_mix_20261107_a02'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing preparation session')
root.mkdir(parents=True,exist_ok=False);frozen=root/'source';files={}
selected=[]
for directory in ('efficient_llm','scripts'):
 selected.extend((source/directory).rglob('*.py'))
selected.extend(source/name for name in ['docs/A2_family_spec_v1.3.md','resources/a2_family_mixture_plan_20261107.json','resources/a2_family_mixture_repair_cpu_20261107.json','tests/a2/test_family_mix_repair.py','resources/a2_family_mixture_content_correction_20261107.json','resources/a2_data_rebuild_wc69_20261103.json'])
for p in selected:
 rel=p.relative_to(source);q=frozen/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);files[str(rel)]=sha(q)
if sha(frozen/'docs/A2_family_spec_v1.3.md')!=SPEC_SHA:raise ValueError('Wrong spec')
commit='5635511e014bec64e926335abfb6e9932bd7d4ad';write(root/'source-manifest.json',dict(files=files,code_commit=commit))
proof=json.loads((frozen/'resources/a2_family_mixture_repair_cpu_20261107.json').read_text())
if proof['status']!='PASS' or any(sha(frozen/p)!=v for p,v in proof['files'].items()):raise ValueError('CPU proof identity changed')
if sha(frozen/'resources/a2_family_mixture_plan_20261107.json')!='12b259426fd63e0f4f27da6b039f9b55a39b33fff052ef55738dea7f6cea15e2':raise ValueError('Mixture plan changed')
cx=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02/cx')
if sha(cx/'complete.json')!='c163d1d3ec4c9563dde7085911eddd9b904cf413346abe31a453e1efdeab7d27':raise ValueError('CX identity changed')
cx_info=json.loads((cx/'complete.json').read_text())
for split,info in cx_info['sets'].items():
 if sha(cx/split/'documents.json')!=info['documents_sha256']:raise ValueError('CX document metadata changed')
old=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a01')
if sha(old/'source-manifest.json')!='7b5beba8972ea4a1fa560af3e9a7821033ee715c2a3a0d601ccc4d29683c0b77':raise ValueError('Old CPU source changed')
for rel,h in json.loads((old/'source-manifest.json').read_text())['files'].items():
 if sha(old/'source'/rel)!=h:raise ValueError('Old frozen source changed')
cmd='exec '+shlex.join([sys.executable,'-u','-m','scripts.a2_family_repair_mix','--plan','resources/a2_family_mixture_plan_20261107.json','--output',str(root/'mixture'),'--old-root','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a01/mixture','--corrections','resources/a2_family_mixture_content_correction_20261107.json','--cx-root','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02/cx'])+' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),cmd],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
write(root/'launch.json',dict(root=str(root),session=session,code_commit=commit,cpu_only=True,source_manifest_sha256=sha(root/'source-manifest.json')))
print('FAMILY_CPU_PREPARATION_SUBMITTED',root,session,commit,flush=True)
PYCPU

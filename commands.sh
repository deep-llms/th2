#1 +20+a
#th2-wc69-family-v17-admission-20261109-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import os,sys,shlex,subprocess,socket
from scripts.launch_a2_downstream import freeze_source
from efficient_llm.a2_family_v17.common import read,write,SPEC_SHA
from scripts.a2_family_v14_gate import sha
source=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-admission-20261109-a01')
session='a2_family_v17_admission_20261109_a01';commit='8bd78f4e6305f85525a11d087e458ed3fa93b0c0'
if socket.gethostname()!='thiennh-p6-wc69-worker-0' or root.exists():raise ValueError('Wrong node or existing admission')
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing session')
proof=read(source/'resources/a2_family_v17_cpu_20261109.json')
if not proof['launch_ready'] or proof['status']!='PASS' or proof['spec_sha256']!=SPEC_SHA:raise ValueError('CPU proof incomplete')
for rel,h in proof['files'].items():
 if sha(source/rel)!=h:raise ValueError('Changed tested source: '+rel)
root.mkdir(parents=True,exist_ok=False)
frozen=freeze_source(source,root,commit,extra_files=[*proof['files'],'resources/a2_family_v17_cpu_20261109.json'])
cmd=[sys.executable,'-u','-m','scripts.launch_a2_family_v17','--plan',str(frozen/'resources/a2_family_v17_plan_20261109.json'),
 '--output','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a01','--session','a2_family_v17_20261109_a01','--code-commit',commit]
command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(root/'admission.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],env={**os.environ,'PYTHONPATH':str(frozen)},check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
write(root/'launch.json',dict(code_commit=commit,source_manifest_sha256=sha(root/'source-manifest.json'),session=session,automatic_F8_F7_report_backup_Depth=True))
print('V17_ADMISSION_SUBMITTED',root,commit,flush=True)
PYCPU

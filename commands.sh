#1 +20+a
#th2-wc69-family-v17-pause-20261109-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import os,sys,shlex,subprocess,socket
from scripts.launch_a2_downstream import freeze_source
from scripts.a2_family_v17_pause import request_current,write,digest,WHY_SHA
if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-pause-20261109-a01');session='a2_family_v17_pause_20261109_a01'
if root.exists() or subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing pause request')
if digest('docs/A2_family_spec_v1.7.md')!=WHY_SHA:raise ValueError('Wrong spec')
request=request_current()
root.mkdir(parents=True,exist_ok=False)
frozen=freeze_source(Path.cwd(),root,'v17-pause-a01',extra_files=['docs/A2_family_spec_v1.7.md'])
write(root/'request.json',request)
cmd=[sys.executable,'-u','-m','scripts.a2_family_v17_pause','--request',str(root/'request.json'),'--output',str(root/'pause')]
command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(root/'pause.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],env={**os.environ,'PYTHONPATH':str(frozen)},check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('V17_PAUSE_SUBMITTED',root,session,digest(root/'source-manifest.json'),flush=True)
PYCPU

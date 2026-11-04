#1 +20+a
#th2-wc69-a2-why-data-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
from pathlib import Path
import subprocess,shlex,socket,json
from scripts.launch_a2_downstream import freeze_source
from scripts.a2_why_prepare import OUT,DATA,SHUFFLE
assert socket.gethostname()=='thiennh-p6-wc69-worker-0'
assert not any(p.exists() for p in (OUT,DATA,SHUFFLE))
session='a2_why_data_wc69_20261104_a01'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode != 0
OUT.mkdir()
extra=('docs/A2_why_spec.md','docs/A2_stage1_spec.md','docs/A2_stage1b_fwe_spec.md','resources/a2_corpora_30b_20261029.json','resources/a2_depth_wc69_plan_20261103_a01.json','docs/results/a2_corpora_20261029/fwe_final/fwe_manifest.json','docs/results/a2_stage1b_fwe_20261030/data_final/manifest.json')
frozen=freeze_source(Path.cwd(),OUT,'371686ffd731eeccdc9a80a985272d6e9d1c861c',extra_files=extra)
command='exec '+shlex.join(['env','PYTHONPATH='+str(frozen),'nice','-n','10','/mnt/local/conda-py311/envs/a2_env/bin/python3.11','-u','-m','scripts.a2_why_prepare'])+' >'+shlex.quote(str(OUT/'prepare.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),command],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('WHY_CPU_DATA_SUBMITTED',OUT,flush=True)
PY
sleep 10
tail -c 5000 /mnt/local/_outputs/deep-llms_th2/wc69-a2-why-data-20261104-a01/prepare.log
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
from pathlib import Path
from scripts.gpu_process import inspect
import json
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01')
for name in ('saved_state.json','signal.json','terminal.json'):
 p=root/'pause'/name
 if p.exists(): print('PAUSE',name,p.read_text())
print('PAUSE_TAIL',(root/'pause.log').read_text()[-2500:])
print('GPU',json.dumps(inspect()))
PY

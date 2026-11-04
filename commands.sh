#1 +20+a
#th2-wc69-a2-why-pause-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$TASK_PYTHON" - <<'PY'
from pathlib import Path
import subprocess,shlex,json
from scripts.a2_why_pause_depth import read,verify_owner,WHY_SHA
from scripts.launch_a2_downstream import freeze_source
source=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01')
session='a2_why_pause_wc69_20261104_a01'
request=read(source/'resources/a2_why_pause_request_20261104.json')
assert request['spec_sha256']==WHY_SHA
assert not root.exists()
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode != 0
verify_owner(request)
root.mkdir()
frozen=freeze_source(source,root,'a2aa158',extra_files=('docs/A2_why_spec.md','resources/a2_why_pause_request_20261104.json'))
argv=['/mnt/local/conda-py311/envs/a2_env/bin/python3.11','-u','-m','scripts.a2_why_pause_depth','--request',str(frozen/'resources/a2_why_pause_request_20261104.json'),'--output',str(root/'pause')]
command='cd '+shlex.quote(str(frozen))+' && exec '+shlex.join(argv)+' > '+shlex.quote(str(root/'pause.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print(json.dumps(dict(state='pause_helper_submitted',root=str(root),session=session,spec_sha256=WHY_SHA)),flush=True)
PY
sleep 10
tmux display-message -p -t a2_why_pause_wc69_20261104_a01 '#{pane_dead} #{pane_dead_status} #{pane_pid}'
tail -c 22000 /mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01/pause.log

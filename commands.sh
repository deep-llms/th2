#1 +20+a
#th2-wc69-family-f10-prepare-recovery-20261111-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import subprocess,json,os,shlex
from efficient_llm.diagnostic_data import sha256
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-preparation-20261111-a01');frozen=root/'source'
pane=subprocess.check_output(['tmux','list-panes','-t','a2_family_f10_prepare_20261111_a01','-F','#{pane_dead} #{pane_dead_status}'],text=True).strip()
assert pane=='1 1' and not (root/'prepared').exists(),pane
assert "ModuleNotFoundError: No module named 'efficient_llm'" in (root/'prepare.log').read_text()
for name,h in json.load(open(root/'source-manifest.json'))['files'].items():assert sha256(frozen/name)==h,name
session='a2_family_f10_prepare_20261111_a02'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
shell='exec /mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u '+shlex.quote(str(root/'wait_prepare.py'))+' >'+shlex.quote(str(root/'prepare_recovery_a02.log'))+' 2>&1'
env={**os.environ,'PYTHONPATH':str(frozen),'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],env=env,check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('F10_PREP_RECOVERY_A02_STARTED_SAME_FROZEN_SOURCE_NO_GPU')
PY

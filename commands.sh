#1 +30+a
#th2-6y3o-a2-mech-cpu-completion-20261031-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
import json,os,shlex,subprocess,sys
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-mech-20261031-a01')
session='a2_mech_20261031_a01'
assert root.is_dir() and not (root/'cpu_completion').exists()
subprocess.run(['tmux','has-session','-t',session],check=True)
names=subprocess.check_output(['tmux','list-windows','-t',session,'-F','#{window_name}'],text=True).splitlines()
assert 'cpu_readout' not in names
argv=[sys.executable,'-u','-m','scripts.a2_mech_cpu_completion','--root',str(root),'--code-commit','2661b7873ae526344fbdeb00599ff2e2e1945760']
cmd='exec env CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/local/deep-llms_th2 '+shlex.join(argv)+' >'+shlex.quote(str(root/'cpu_completion_launch.log'))+' 2>&1'
subprocess.run(['tmux','new-window','-d','-t',session,'-n','cpu_readout','-c','/mnt/local/deep-llms_th2',cmd],check=True)
subprocess.run(['tmux','set-option','-w','-t',session+':cpu_readout','remain-on-exit','on'],check=True)
print('MECH_CPU_COMPLETION_SUBMITTED',json.dumps(dict(root=str(root),session=session,window='cpu_readout',no_gpu_actions=True)),flush=True)
PYCPU

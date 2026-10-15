#1 +15+a
#th2-q359-p7-priority-handoff-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 - <<'PYLAUNCH'
import shlex,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01')
source='/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning22-20261015-a03'
session=root.name
if root.exists() or Path(str(root)+'.launch.log').exists():raise RuntimeError('Existing handoff')
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise RuntimeError('Existing session')
env=['NCCL_NVLS_ENABLE=0','WANDB_MODE=offline','WANDB_PROJECT=deep2shallow','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=42','TOKENIZERS_PARALLELISM=false','CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','TORCHINDUCTOR_COMPILE_THREADS=1','TORCHINDUCTOR_CACHE_DIR='+str(root/'inductor'),'TRITON_CACHE_DIR='+str(root/'triton')]
args=['/mnt/local/conda-py311/envs/attention_bench/bin/python3.11','-u','-m','scripts.reorder_p7_tuning','--source',source,'--root',str(root),'--host','thiennh-p6-q359-worker-0']
command='source /mnt/local/conda-py311/etc/profile.d/conda.sh && conda activate attention_bench && exec '+shlex.join(['env',*env,*args])+' >'+shlex.quote(str(root)+'.launch.log')+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,command],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('HANDOFF_WAITER_STARTED',session)
PYLAUNCH
sleep 5
cat /mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01.launch.log

#1 +60+a
#th2-6y3o-efficient-kv-long-launch-20261027-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'LAUNCH'
import json, shlex, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
assert socket.gethostname() == 'thiennh-p6-6y3o-worker-0'
base = Path('/mnt/local/_outputs/deep-llms_th2')
root = base/'6y3o-efficient-kv-long-20261027-a01'
log = base/'6y3o-efficient-kv-long-20261027-a01-prepare.log'
session = 'efficient_kv_long_prepare_20261027_a01'
assert not root.exists() and not log.exists()
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode != 0
argv = [sys.executable, '-u', '-m', 'scripts.launch_group_diagnostic', '--study','kv_long',
 '--output',str(root),'--model','/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6',
 '--data-root','/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513',
 '--host',socket.gethostname(),'--session','efficient_kv_long_20261027_a01',
 '--previous-run',str(base/'6y3o-efficient-kv-recovery-20261027-a02'),
 '--dense-parent-run',str(base/'6y3o-efficient-kv-mlp-20261027-a03'),
 '--predecessor-burn-script',str(base/'6y3o-efficient-kv-qkv-20261027-a01/source/resources/llm_pretrain_burn.py')]
keys = ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE',
        'TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS',
        'TOKENIZERS_PARALLELISM','NCCL_NVLS_ENABLE']
import os
command = 'exec '+shlex.join(['env',*[f'{k}={os.environ[k]}' for k in keys],*argv])+' >'+shlex.quote(str(log))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(Path.cwd()),command],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('LONG_PREPARATION_SUBMITTED',json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),
 session=session,output=str(root),log=str(log))),flush=True)
LAUNCH

#1 +60+a
#th2-6y3o-efficient-kv-recovery-launch-20261027-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
test -x "$TASK_PYTHON"
date -u
hostname
git rev-parse HEAD
"$TASK_PYTHON" -u - <<'VERIFY'
from pathlib import Path
import socket
from scripts.train_then_burn import digest
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-20261027-a03/supervised/run')
expected={
 'A': {'result.json':'546bde083f9c45f16d44b6ea82c4c99a9c75eba33ffe8c67744e3da35cdac4c0',
       'final/config.json':'b03acd66fa4641bf16694645176f135e78a7442386790ee73196aefc3b2b6995',
       'final/model.safetensors':'fe621f7c14de058b20b9ee5dd7e0071ac2c62f352675dfdc21f65af262328be8'},
 'B': {'result.json':'45eebb3ab288c2d62c7ea83848d899242873002224680cb792e4a4506c7c6296',
       'final/config.json':'daa623eedc8e50519ecc0636bb0d6b4d87c17cd212ab2c98025fda30937644c9',
       'final/model.safetensors':'c5f9ef7aea0b4260a9087ec2be998a121f16d4f1693ad5eb88a9a9a7384c487c'}}
for arm,files in expected.items():
 for name,sha in files.items():
  if digest(root/arm/name)!=sha:raise ValueError('Parent changed: '+arm+'/'+name)
print('RECOVERY_PARENT_HASHES_VERIFIED',flush=True)
VERIFY
"$TASK_PYTHON" -u -m scripts.launch_group_diagnostic \
  --study kv_recovery \
  --previous-run /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-20261027-a03 \
  --output /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-recovery-20261027-a01 \
  --model /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6 \
  --data-root /mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513 \
  --host thiennh-p6-6y3o-worker-0 \
  --session efficient_kv_recovery_20261027_a01 \
  --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-inference-20261027-a01/source/resources/llm_pretrain_burn.py

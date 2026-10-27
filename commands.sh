#1 +60+a
#th2-6y3o-efficient-kv-qkv-launch-20261027-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
test -x "$TASK_PYTHON"
date -u
hostname
cd /mnt/local/deep-llms_th2
"$TASK_PYTHON" -u -m scripts.launch_group_diagnostic   --study kv_qkv   --previous-run /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-recovery-20261027-a02   --dense-parent-run /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-20261027-a03   --placement-run /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-placement-20261027-a01   --output /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-qkv-20261027-a01   --model /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6   --data-root /mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513   --host thiennh-p6-6y3o-worker-0   --session efficient_kv_qkv_20261027_a01   --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-placement-20261027-a01/source/resources/llm_pretrain_burn.py

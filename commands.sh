#1 +60+a
#th2-6y3o-efficient-kv-mlp-20261027-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u -m scripts.launch_group_diagnostic \
  --study kv_mlp \
  --output /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-20261027-a01 \
  --model /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6 \
  --data-root /mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513 \
  --host thiennh-p6-6y3o-worker-0 \
  --session efficient_kv_mlp_20261027_a01 \
  --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-window-compact-20261027-a01/source/resources/llm_pretrain_burn.py

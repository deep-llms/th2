#1 +60+a
#th2-6y3o-efficient-kv-downstream-launch-20261028-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/eval/bin/python3.11 -u -m scripts.launch_kv_downstream \
  --parent /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-long-20261027-a01 \
  --output /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-downstream-20261028-a02 \
  --data-root /mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023 \
  --tokenizer /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6 \
  --host thiennh-p6-6y3o-worker-0 --session efficient_kv_downstream_20261028_a02

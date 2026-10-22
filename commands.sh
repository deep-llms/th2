#1 +30+a
#th2-6y3o-dot-mlp-launch-20261022-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /mnt/local/conda-py311/envs/train/bin/python3.11 scripts/bdot_mlp_pipeline.py launch \
  --output /mnt/local/_outputs/deep-llms_th2/6y3o-dot-mlp-20261022-a01 \
  --pilot /mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01 \
  --predecessor /mnt/local/_outputs/deep-llms_th2/6y3o-dot-C1-20261022-a01 \
  --host thiennh-p6-6y3o-worker-0 --session dot_mlp_20261022_a01

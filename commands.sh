#1 +60+a
#th2-q359-dot-verification-20261018-a03
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
test -x "$TASK_PYTHON"
"$TASK_PYTHON" -m scripts.bdot_b200_launch --root /mnt/local/_outputs/deep-llms_th2/q359-dot-verification-20261018-a03 --session dot_verify_20261018_a03

#1 +60+a
#th2-ohmv-warm-downstream-launch-20261118-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.warm_downstream_job launch --commit 6ef3c3d356654c4c041cec64666293e8a80abf3f

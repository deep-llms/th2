#1 +60+a
#th2-ohmv-warm-downstream-launch-20261118-a02
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.warm_downstream_job launch --commit 504a22537d5636d3ab9c3d7368540ba3bfcf448d

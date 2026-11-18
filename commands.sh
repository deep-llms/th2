#1 +60+a
#th2-ohmv-warm-downstream-cpu-recovery-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.warm_downstream_recover launch --commit 0c5f32194827658cc00d3c1b1500f0a354263a5d

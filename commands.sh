#1 +30+a
#th2-ohmv-a2-dot-data-launch-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
date -u
hostname
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_dot_rebuild_data launch --commit 5cbb34d96c41cdc39130428f11dfaca64b1d3376 --source-manifest resources/a2_dot_rebuild_source_20261118.json

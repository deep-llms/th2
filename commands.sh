#1 +30+a
#th2-wc69-dot-pool-defer-scheduler-fix-20261112-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -u -m scripts.dot_pool_defer_a03 --commit d3ed9a62377eb56c0e065fd5c9ed234b4825484a
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -u -m scripts.dot_pool_health --name a02

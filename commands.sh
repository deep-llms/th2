#1 +600+a
#th2-wc69-dot-pool-reschedule-20261112-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -u -m scripts.dot_pool_transition_a02 --commit c6d44961628006060df0742cd78decb5d38b34c4
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -u -m scripts.dot_pool_health --name a02

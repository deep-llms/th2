#1 +30+a
#th2-wc69-dot-rl-health-20261114-a04
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -u -m scripts.dot_rl_health

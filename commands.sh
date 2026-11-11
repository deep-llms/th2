#1 +60+a
#th2-wc69-dot-pool-launch-20261112-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2
/mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -u -m scripts.launch_dot_pool --name a01 --commit 3e1185c77121f3ad1859e523538f84f3fe8b0dfd

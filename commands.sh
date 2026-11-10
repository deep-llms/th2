#1 +20+a
#th2-wc69-family-f9-v110-health-20261110-a05
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_health --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-20261110-a01

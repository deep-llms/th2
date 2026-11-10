#1 +20+a
#th2-wc69-family-f9-v19-preserve-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
nice -n 10 /mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_preserve --prior-root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01 --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v19-preserve-20261110-a01

#1 +20+a
#th2-wc69-family-f9-v110-report-recovery-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 /mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_report_recover --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-20261110-a01/results/a2_family_20261107/downstream --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-report-recovery-20261110-a01 --manifest-sha256 8b0b68bdc69cf6aa7532250e1f368ae37fdcf1b6cebc7bdb099814b72633de3a

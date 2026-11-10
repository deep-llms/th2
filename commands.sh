#1 +20+a
#th2-wc69-family-f9-activate-prepared-20261110-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.activate_a2_family_f9_prepared --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01 --ready-sha256 24a236b405a4b7f2f51153acc9945a2ec8896bb996325e1e416e7b886115d1c5 --attestation-sha256 eb3a64960edc9670e738fa9f51657107e550773e606a02fd81064d3028029d68 --source-commit 2a74421e4dc6f4f6669af1769e1ae07828b031c2 --session a2_family_f9_20261110_a02

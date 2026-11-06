#1 +20+a
#th2-wc69-family-recover-cx-20261107-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_recover_cx \
 --data /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/data \
 --tokenizer-file /mnt/local/_models/deep-llms_th2/a2/dclm/P300/tokenizer.json \
 --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-cx-recovery-20261107-a01

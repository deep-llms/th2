#1 +20+a
#th2-wc69-a2-why-yt1-launch-20261104-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_why \
 --plan resources/a2_why_yt1_plan_20261104_a01.json \
 --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-why-yt1-20261104-a01 \
 --session a2_why_yt1_wc69_20261104_a01 \
 --stage yt1 --code-commit 082f9029887ca949ab3bef810b4f055dbc6f792d

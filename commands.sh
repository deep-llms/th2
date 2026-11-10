#1 +20+a
#th2-wc69-family-f9-launch-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_family_f9 --plan resources/a2_family_f9_plan_20261110.json --output /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01 --session a2_family_f9_20261110_a01 --code-commit "$(git rev-parse HEAD)"

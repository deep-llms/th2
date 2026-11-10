#1 +20+a
#th2-wc69-family-f11-queue-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_family_f11 --mode install --plan resources/a2_family_f11_plan_20261111.json --code-commit 36432091e99089f8bb9c55ea76a29714ac4f4943 --cpu-proof resources/a2_family_f11_cpu_20261111.json

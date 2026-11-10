#1 +20+a
#th2-wc69-family-f12-queue-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_family_f12 --mode install --plan resources/a2_family_f12_plan_20261111.json --code-commit 9651282a3d7c35a13cbc3696e7a5960482690fe4 --cpu-proof resources/a2_family_f12_cpu_20261111.json

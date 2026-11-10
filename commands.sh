#1 +20+a
#th2-wc69-family-f11-health-20261111-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
from scripts.a2_family_f11_health import health
from scripts.a2_family_f10_health import health as f10health
health('resources/a2_family_f11_plan_20261111.json')
f10health('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01')
CODE

#1 +60+a
#th2-6y3o-a2-initial-preflight-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_preflight \
 --host thiennh-p6-6y3o-worker-0 \
 --session a2_initial_preflight_20261028_a01 \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-initial-preflight-20261028-a01 \
 --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs \
 --code-commit 20b61d04d59a603532e31ef00687872be6af7e6d \
 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-review-20261028-a01/source/resources/llm_pretrain_burn.py

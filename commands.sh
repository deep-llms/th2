#1 +60+a
#th2-6y3o-a2-stage1-heals-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_stage1 \
 --stage heals --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs \
 --heal-data /mnt/local/_data/deep-llms_th2/a2_stage1_dclm_shuffle2030_20261029 \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-heals-20261029-a01 \
 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-preflight-20261029-a01/source/resources/llm_pretrain_burn.py \
 --session a2_stage1_heals_20261029_a01 --code-commit 0c1c1488477f1b836989182986bd785fbcd667d8 \
 --preflight-root /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-preflight-20261029-a01 \
 --preflight-sha256 739847de9e35a517fd7240f46b69a47bad7a55183503abfe2668a402b24380ef --spent-seconds 1800

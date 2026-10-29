#1 +30+a
#th2-6y3o-a2-stage1b-fwe-part1-20261030-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_stage1b_heals --phase part1 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-part1-20261030-a01 --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs --heal-data /mnt/local/_data/deep-llms_th2/a2_stage1b_fwe_shuffle2030_20261030 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-preflight-20261030-a01/source/resources/llm_pretrain_burn.py --preflight-root /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-preflight-20261030-a01 --preflight-sha256 0ddedf66933134338cc87f872b28e5df9d8a67ba0cf6460d482c90ebe0321b8b --session a2_stage1b_fwe_part1_20261030_a01 --code-commit 7dd9da088ba5af581c295775819319ce54b2025c --spent-seconds 3152.574 --deadline 2026-10-31T04:06:27.182436+08:00

#1 +60+a
#th2-6y3o-a2-review-smoke-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -c 'from importlib.metadata import version; print("EXISTING_TRAIN_ENV", {x:version(x) for x in ("torch","transformers","accelerate")})'
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -c 'from importlib.metadata import version; print("A2_ENV", {x:version(x) for x in ("torch","transformers","accelerate","datasets","huggingface-hub","tokenizers")})'
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_check \
 --native-code /mnt/local/_models/deep-llms_th2/a2_code_review_20261028 \
 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-downstream-20261028-a02/source/resources/llm_pretrain_burn.py \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-review-20261028-a01 \
 --host thiennh-p6-6y3o-worker-0 --session a2_review_20261028_a01

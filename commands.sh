#1 +60+a
#th2-6y3o-a2-fwe-s1-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_preflight \
  --stage s1-pair --dataset fwe --spent-seconds 5200 \
  --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-fwe-s1-20261028-a01 \
  --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs \
  --prior /mnt/local/_outputs/deep-llms_th2/6y3o-a2-initial-preflight-20261028-a01 \
  --grid /mnt/local/_outputs/deep-llms_th2/6y3o-a2-early-grid-20261028-a01 \
  --code-commit cfeb74c6db5e63d3a1044cda6955d2df16e4891b \
  --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-s1-pair-20261028-a01/source/resources/llm_pretrain_burn.py \
  --host thiennh-p6-6y3o-worker-0 --session a2_fwe_s1_20261028_a01

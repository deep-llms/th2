#1 +60+a
#th2-6y3o-efficient-selective-research-launch-20261026-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=""
/mnt/local/conda-py311/envs/train/bin/python3.11 -u -m scripts.launch_group_diagnostic \
 --study research \
 --host thiennh-p6-6y3o-worker-0 \
 --model /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6 \
 --data-root /mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513 \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-selective-research-20261026-a01 \
 --session efficient_research_20261026_a01 \
 --previous-run /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-context-budget-20261026-a01 \
 --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-efficient-decode-numerics-20261026-a01/source/resources/llm_pretrain_burn.py

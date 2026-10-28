#1 +60+a
#th2-6y3o-a2-branches-20261028-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_preflight \
  --failed-handoff /mnt/local/deep-llms_th2/resources/a2_failed_branch_handoff_20261028.json \
  --stage remaining-prereqs --spent-seconds 18000 \
  --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-branches-20261028-a03 \
  --inputs /mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs \
  --history /mnt/local/deep-llms_th2/resources/a2_completed_surveys_20261028.json \
  --code-commit bb72766458dc06ba5ca2ca275c69f153d7d5a871 \
  --predecessor-burn-script /mnt/local/_outputs/deep-llms_th2/6y3o-a2-branches-20261028-a01/source/resources/llm_pretrain_burn.py \
  --host thiennh-p6-6y3o-worker-0 --session a2_branches_20261028_a03

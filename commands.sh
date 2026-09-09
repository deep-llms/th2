#1 +60+a
#th2-swt-full-42-checkpoint-english-eval-then-finetune-20260909-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
printf '%s  %s\n' b8a3d35ef8db52867a52507f494c41ace823c662e9f5ee44625020c7b6d1018d scripts/eval_finetune_capacity_b200.sh | sha256sum -c -
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_eval_sweep_handoff.py -v
bash scripts/eval_finetune_capacity_b200.sh /mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01

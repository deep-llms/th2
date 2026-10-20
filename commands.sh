#1 +30+a
#th2-6y3o-native12-cx-en-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/6y3o-native12-cx-en-20261020-a01
TASK_SESSION=native12_prepare_20261020_a01
TASK_COMPARE=/mnt/local/_outputs/deep-llms_th2/6y3o-attention-compare-20261020-a01
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.prepare.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
printf -v TASK_COMMAND 'exec env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline %q -u %q --comparison %q --output %q --data-root /mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513 --model-root /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6 --host thiennh-p6-6y3o-worker-0 --session native12_train_20261020_a01 --wait-for-download-seconds 21600 >%q 2>&1' "$TASK_PYTHON" "$PWD/scripts/native_baseline_pipeline.py" "$TASK_COMPARE" "$TASK_ROOT" "$TASK_ROOT.prepare.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
tail -15 "$TASK_ROOT.prepare.log"
echo OFFLINE_DATA_VERIFICATION_AND_GATED_BASELINE_PREPARATION_STARTED

#1 +30+a
#th2-6y3o-dot-v3-launch-20261021-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/6y3o-dot-v3-20261021-a01
TASK_PREPARE_SESSION=dot_v3_prepare_20261021_a01
test -s "$TASK_ROOT/initial-preflight.json"
test ! -e "$TASK_ROOT/prepare.log"
test ! -e "$TASK_ROOT/prepared"
if tmux has-session -t "$TASK_PREPARE_SESSION" 2>/dev/null; then exit 1; fi
printf -v TASK_COMMAND 'cd /mnt/local/deep-llms_th2 && exec env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false %q -u -m scripts.bdot_v3_pipeline --output %q --predecessor /mnt/local/_outputs/deep-llms_th2/6y3o-dot-diagnostics-gradient-20261021-a02 --pilot /mnt/local/_outputs/deep-llms_th2/6y3o-dot-pilot-20261020-a01 --no-reuse /mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01 --random-run /mnt/local/_outputs/deep-llms_th2/6y3o-dot-R-20261021-a01 --host thiennh-p6-6y3o-worker-0 --session dot_v3_20261021_a01 >%q 2>&1' "$TASK_PYTHON" "$TASK_ROOT" "$TASK_ROOT/prepare.log"
tmux new-session -d -s "$TASK_PREPARE_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_PREPARE_SESSION" remain-on-exit on
sleep 5
tail -30 "$TASK_ROOT/prepare.log"
echo V3_PREPARATION_SUBMITTED

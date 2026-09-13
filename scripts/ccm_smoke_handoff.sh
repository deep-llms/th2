#!/usr/bin/env bash
# Run-specific authorized B200 smoke, with verified original-burn restoration.
set -euo pipefail
test "$(hostname)" = thiennh-p6-8mgy-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/ccm_smoke_20260913_a01
TASK_SESSION=ccm_burn_20260913_a01
test -s "$TASK_OUTPUT/data_validation.json"
test -s "$TASK_OUTPUT/coverage.json"
test ! -e "$TASK_OUTPUT/common"
test ! -e "$TASK_OUTPUT/performance"
test ! -e "$TASK_OUTPUT/burn.log"
command -v tmux
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
/usr/bin/python3 -c 'import os,signal; assert hasattr(os,"pidfd_open") and hasattr(signal,"pidfd_send_signal")'
python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",29637)); s.close()'
/usr/bin/python3 scripts/ccm_smoke_gpu_control.py stop-verified-original
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
# This trainer uses torchrun/NCCL, not the template Accelerate configuration.
sleep 30
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
set +e
bash scripts/ccm_smoke_payload.sh > "$TASK_OUTPUT/payload.log" 2>&1
TASK_STATUS=$?
set -e
echo "CCM_PAYLOAD_EXIT=$TASK_STATUS"
sleep 30
# Even on a payload error, restart only after proving no GPU worker remains.
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",29637)); s.close()'
printf -v TASK_BURN_CMD 'exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_ADDR=127.0.0.1 MASTER_PORT=29637 NCCL_DEBUG=INFO /usr/bin/python3 -u /tmp/llm_pretrain_burn.py >%q 2>&1' "$TASK_OUTPUT/burn.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_BURN_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 30
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
python scripts/ccm_smoke_gpu_control.py verify-burn > "$TASK_OUTPUT/burn_verification.json"
sleep 10
python scripts/ccm_smoke_gpu_control.py verify-burn
tail -n 30 "$TASK_OUTPUT/burn.log"
date -u
if [ "$TASK_STATUS" -eq 0 ]; then
    test -s "$TASK_OUTPUT/payload_verified.json"
    echo CCM_B200_SMOKE_SUCCESS_AND_ORIGINAL_BURNS_RESTORED
else
    echo CCM_B200_SMOKE_FAILED_BUT_ORIGINAL_BURNS_RESTORED
fi
exit "$TASK_STATUS"

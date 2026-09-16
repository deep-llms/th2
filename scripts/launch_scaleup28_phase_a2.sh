#!/usr/bin/env bash
set -euo pipefail
test "$PWD" = /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-8mgy-worker-0
date -u
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_SESSION=ccm_scaleup28_phase_a2_20260916_a01
TASK_LOG=/mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a2_20260916_a01.handoff.log
test ! -e "$TASK_LOG"
test ! -L "$TASK_LOG"
test ! -e /mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a2_20260916_a01
test ! -L /mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a2_20260916_a01
test ! -e /mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260916_a01
test ! -L /mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260916_a01
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    echo 'REFUSE: scaleup28 phase-a2 session exists' >&2
    exit 1
fi
# Foreign jobs currently occupy the GPUs; the controller performs the
# authorized identity-pinned clear itself. Read-only snapshot here only.
/usr/bin/python3 scripts/gpu_status.py
printf -v TASK_CMD 'cd /mnt/local/deep-llms_th2 && source /mnt/local/conda-py311/etc/profile.d/conda.sh && conda activate train_env && exec /usr/bin/python3 -u scripts/pilot_scaleup28_phase_a2.py >%q 2>&1' "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "bash -c $(printf '%q' "$TASK_CMD")"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 30
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
tail -n 30 "$TASK_LOG"
echo CCM_SCALEUP28_PHASE_A2_HANDOFF_ALIVE

#1 +60+a
#th2-ohmv-local-order-launch-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
TASK_PYTHON=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_BASE=/mnt/local/_outputs/deep-llms_th2/local_order_20261118
[ "$(hostname)" = thiennh-p6-ohmv-worker-0 ]
[ ! -e "$TASK_BASE/execution" ]
[ ! -e "$TASK_BASE/runs" ]
[ ! -e "$TASK_BASE/launch-tests.log" ]
"$TASK_PYTHON" -m unittest discover -s tests/warm_order -v > "$TASK_BASE/launch-tests.log" 2>&1
tail -n 5 "$TASK_BASE/launch-tests.log"
"$TASK_PYTHON" -m scripts.launch_local_order --commit 4573d73ff7f2d7c2b60fb83160b9103f0203afcf
sleep 5
tail -n 10 "$TASK_BASE/execution/queue.log"

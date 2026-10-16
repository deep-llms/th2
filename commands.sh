#1 +60+a
#th2-q359-ce7454-band-attach-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_VISION=/mnt/local/conda-py311/envs/ce7454_env/bin/python
TASK_CONTROL=/mnt/local/conda-py311/envs/eval_fa4/bin/python
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-ce7454-band-20261016-a01
TASK_OLD=/mnt/local/_outputs/deep-llms_th2/q359-ce7454-scratch-20261016-a04
TASK_SESSION=ce7454_band_20261016_a01
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "$TASK_VISION" -m unittest discover -s ce7454/src -p 'test_band_study.py' -v
printf -v TASK_CMD 'exec env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 %q -u ce7454/src/attach_band_study.py --original %q --output %q >%q 2>&1' "$TASK_CONTROL" "$TASK_OLD" "$TASK_ROOT" "$TASK_ROOT.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 10
cat "$TASK_ROOT.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

#1 +30+a
#th2-q359-dot-grouped-test-20261018-a02
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/eval_fa4/bin/python
"$TASK_PYTHON" scripts/bdot_b200_preflight.py --output /mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a02/preflight.json
"$TASK_PYTHON" scripts/bdot_b200_launch.py \
  --root /mnt/local/_outputs/deep-llms_th2/q359-dot-grouped-20261018-a02 \
  --session dot_grouped_20261018_a02 \
  --jobs-config resources/bdot_b200_grouped.json

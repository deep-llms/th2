#1 +60+a
#th2-joint-v2-reclaim-guard-20260923-a01
set -euo pipefail
date -u
sed -n '1,300p' /mnt/local/_gpu_guard/gpu_guard.sh
PYTHONPATH="$PWD" /usr/bin/python3 -u -m scripts.reclaim_guard_burn_20260923 --authorized-stop --output /mnt/local/_outputs/@PROJECT@/joint-v2-guard-stop-20260923-a01.json
TASK_RUN=/mnt/local/_outputs/@PROJECT@/joint-v2-b200-20260923-a03
cat "$TASK_RUN/runs/run.json"
tail -n 3 "$TASK_RUN/runs/seed-0-Base/train.jsonl"
tail -n 3 "$TASK_RUN/runs/seed-0-Base/validation.jsonl"

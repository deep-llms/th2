#1 +30+a
#th2-joint-v2-live-ownership-20260923-a01
set -euo pipefail
date -u
hostname
/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11 -u -m scripts.inspect_burn
TASK_RUN=/mnt/local/_outputs/@PROJECT@/joint-v2-b200-20260923-a03
cat "$TASK_RUN/runs/run.json"
tail -n 3 "$TASK_RUN/runs/seed-0-Base/train.jsonl"
tail -n 3 "$TASK_RUN/runs/seed-0-Base/validation.jsonl"
if test -e "$TASK_RUN/runs/seed-0-Base/latest.pt"; then
  stat --format='checkpoint_bytes=%s modified=%y' "$TASK_RUN/runs/seed-0-Base/latest.pt"
fi

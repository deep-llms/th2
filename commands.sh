#1 +60+a
#th2-inspect-culturax-sampling-retry-20260823-0230
set -euo pipefail

echo '=== inspect corrected CulturaX sampling retry ==='
date -u
hostname
TASK_OUTPUT_ROOT="/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B"

echo '=== active sampling processes ==='
pgrep -af 'prepare_data.py|sha256sum' || true
ps -eo pid,etime,pcpu,pmem,rss,args | grep -E 'prepare_data.py|sha256sum' | grep -v grep || true

echo '=== output progress ==='
if [ -e "$TASK_OUTPUT_ROOT" ]; then
    du -sh "$TASK_OUTPUT_ROOT"
    for TASK_LANG in en vi zh ru de ar; do
        TASK_SHARDS="$(find "$TASK_OUTPUT_ROOT/train/$TASK_LANG" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' 2>/dev/null | wc -l || true)"
        TASK_TRAIN_ARROW="$(find "$TASK_OUTPUT_ROOT/train/$TASK_LANG" -type f -name '*.arrow' 2>/dev/null | wc -l || true)"
        TASK_EVAL_ARROW="$(find "$TASK_OUTPUT_ROOT/eval/$TASK_LANG" -maxdepth 1 -type f -name '*.arrow' 2>/dev/null | wc -l || true)"
        echo "$TASK_LANG train_shards=$TASK_SHARDS train_arrow_files=$TASK_TRAIN_ARROW eval_arrow_files=$TASK_EVAL_ARROW"
    done
else
    echo 'sample_output=not_created_yet'
fi
df -h /mnt/local/_data/@PROJECT@/data
echo 'TH2 CULTURAX RETRY INSPECTION COMPLETE'

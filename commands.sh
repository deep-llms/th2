#1 +60+a
#th2-verify-runner-smoke-dataset-resolved-path-20260822
set -euo pipefail

echo '=== th2 downloaded dataset verification ==='
date -u
hostname

TASK_PROJECT="${PWD##*/}"
TASK_DATA_DIR="/mnt/local/_data/$TASK_PROJECT/runner_smoke_demo1"
echo "project=$TASK_PROJECT"
echo "dataset_dir=$TASK_DATA_DIR"
test -d "$TASK_DATA_DIR"

echo '=== directory listing ==='
ls -la "$TASK_DATA_DIR"

echo '=== recursive files ==='
find "$TASK_DATA_DIR" -maxdepth 4 -type f -printf '%s bytes  %p\n' | sort

TASK_FILE_COUNT="$(find "$TASK_DATA_DIR" -type f | wc -l)"
TASK_BYTE_COUNT="$(du -sb "$TASK_DATA_DIR" | awk '{print $1}')"
echo "file_count=$TASK_FILE_COUNT"
echo "byte_count=$TASK_BYTE_COUNT"
test "$TASK_FILE_COUNT" -gt 0
test "$TASK_BYTE_COUNT" -gt 0

echo '=== checksums ==='
find "$TASK_DATA_DIR" -type f -print0 | sort -z | xargs -0 -r sha256sum

echo '=== disk usage ==='
du -sh "$TASK_DATA_DIR"
echo 'TH2 DATASET VERIFY OK'

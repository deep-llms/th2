#1 +60+a
#th2-verify-project-token-substitution-20260822
set -euo pipefail

echo '=== th2 downloaded dataset verification ==='
date -u
hostname

TASK_DATA_DIR="/mnt/local/_data/@PROJECT@/runner_smoke_demo1"
echo "dataset_dir=$TASK_DATA_DIR"
TASK_LITERAL_PROJECT_TOKEN='@'"PROJECT"'@'
case "$TASK_DATA_DIR" in
    *"$TASK_LITERAL_PROJECT_TOKEN"*)
        echo 'ERROR: project placeholder was not substituted in the #1 shell body'
        exit 1
        ;;
esac
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
test -s "$TASK_DATA_DIR/data/train.csv"
test -s "$TASK_DATA_DIR/data/test.csv"
echo 'TH2 PROJECT TOKEN AND DATASET VERIFY OK'

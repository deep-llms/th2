#1 +120+a
#th2-export-all-compact-experiment-results-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_EXPORT_NAME=all_experiment_results_20260824
TASK_ARCHIVE="$TASK_EXPORT_DIR/$TASK_EXPORT_NAME.tar.gz"
TASK_CHECKSUM="$TASK_ARCHIVE.sha256"
TASK_FILE_LIST="$TASK_EXPORT_DIR/$TASK_EXPORT_NAME.files"

echo '=== create compact all-experiment result export ==='
date -u
hostname
test -d "$TASK_OUTPUT_BASE"
mkdir -p "$TASK_EXPORT_DIR"
[[ ! -e "$TASK_ARCHIVE" ]] || {
    echo "ERROR: refusing to overwrite $TASK_ARCHIVE"
    exit 1
}
[[ ! -e "$TASK_CHECKSUM" ]] || {
    echo "ERROR: refusing to overwrite $TASK_CHECKSUM"
    exit 1
}
[[ ! -e "$TASK_FILE_LIST" ]] || {
    echo "ERROR: refusing to overwrite $TASK_FILE_LIST"
    exit 1
}

cd "$TASK_OUTPUT_BASE"
find . -type f \
    ! -path './result_exports/*' \
    \( \
        -name 'eval_ppl.json' \
        -o -name 'eval_benchmarks.json' \
        -o -name 'summary.md' \
        -o -name 'anchor_usage.json' \
        -o -name '*bytoken*summary.json' \
        -o -path './finetune*/*.json' \
        -o -path './crosslingual*/*.json' \
        -o -path './*/checkpoint-10000/trainer_state.json' \
        -o -path './*/checkpoint-10000/train_config.json' \
        -o -path './*/train_config.json' \
    \) \
    -print0 \
    | sort -z > "$TASK_FILE_LIST"

TASK_FILE_COUNT=$(tr '\0' '\n' < "$TASK_FILE_LIST" | sed '/^$/d' | wc -l)
[[ "$TASK_FILE_COUNT" -gt 0 ]] || {
    echo 'ERROR: no result artifacts found'
    exit 1
}
tar --null --files-from="$TASK_FILE_LIST" -czf "$TASK_ARCHIVE"
sha256sum "$TASK_ARCHIVE" > "$TASK_CHECKSUM"
TASK_ARCHIVE_BYTES=$(stat -c '%s' "$TASK_ARCHIVE")
(( TASK_ARCHIVE_BYTES < 45000000 )) || {
    echo "ERROR: export exceeds safe pull size: $TASK_ARCHIVE_BYTES bytes"
    exit 1
}

echo "EXPORT_FILE_COUNT=$TASK_FILE_COUNT"
echo "EXPORT_ARCHIVE_BYTES=$TASK_ARCHIVE_BYTES"
cat "$TASK_CHECKSUM"
echo '=== exported result paths ==='
tr '\0' '\n' < "$TASK_FILE_LIST"
echo '=== archive verification ==='
tar -tzf "$TASK_ARCHIVE" | wc -l
gzip -t "$TASK_ARCHIVE"
echo "TH2 COMPACT RESULT EXPORT READY archive=$TASK_ARCHIVE checksum=$TASK_CHECKSUM files=$TASK_FILE_LIST"

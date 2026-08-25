#1 +60+a
#th2-export-all-compact-b200-results-20260825
#!/usr/bin/env bash
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_EXPORT_NAME=all_experiment_results_20260825
TASK_ARCHIVE="$TASK_EXPORT_DIR/$TASK_EXPORT_NAME.tar.gz"
TASK_CHECKSUM="$TASK_ARCHIVE.sha256"
TASK_FILE_LIST="$TASK_EXPORT_DIR/$TASK_EXPORT_NAME.files"

echo '=== create compact structured-result export after matched controls ==='
date -u
hostname
test -d "$TASK_OUTPUT_BASE"
mkdir -p "$TASK_EXPORT_DIR"
for path in "$TASK_ARCHIVE" "$TASK_CHECKSUM" "$TASK_FILE_LIST"; do
    [[ ! -e "$path" ]] || {
        echo "ERROR: refusing to overwrite $path"
        exit 1
    }
done

echo '=== require the four B200 checkpoint-10000 evaluation results ==='
for experiment in \
    lowrank_independent_output_r128 \
    shared_local_tied_g16 \
    global_lowrank_tied_r128_b200 \
    dense_tied_baseline_b200; do
    checkpoint="$TASK_OUTPUT_BASE/$experiment/checkpoint-10000"
    test -s "$checkpoint/eval_ppl.json"
    test -s "$checkpoint/eval_benchmarks.json"
    echo "B200_EVAL_OK experiment=$experiment"
done

echo '=== require both B200 finetune protocols ==='
test -s "$TASK_OUTPUT_BASE/finetune_independent_lr128_shared_local_g16/summary.md"
test "$(find "$TASK_OUTPUT_BASE/finetune_independent_lr128_shared_local_g16" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 18
test -s "$TASK_OUTPUT_BASE/finetune_global_lr_tied_r128_dense_tied_b200_10k_20260825/summary.md"
test "$(find "$TASK_OUTPUT_BASE/finetune_global_lr_tied_r128_dense_tied_b200_10k_20260825" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 18
echo 'B200_FINETUNE_OK protocols=2 jobs=36'

echo '=== collect result-only artifacts; exclude model weights and logs ==='
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
    -print0 | sort -z > "$TASK_FILE_LIST"

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
echo '=== required exported B200 result paths ==='
tr '\0' '\n' < "$TASK_FILE_LIST" \
    | grep -E 'lowrank_independent_output_r128|shared_local_tied_g16|global_lowrank_tied_r128_b200|dense_tied_baseline_b200|finetune_independent_lr128_shared_local_g16|finetune_global_lr_tied_r128_dense_tied_b200_10k_20260825'
echo '=== archive verification ==='
test "$(tar -tzf "$TASK_ARCHIVE" | wc -l)" -eq "$TASK_FILE_COUNT"
gzip -t "$TASK_ARCHIVE"
echo "TH2 COMPACT B200 RESULT EXPORT READY archive=$TASK_ARCHIVE checksum=$TASK_CHECKSUM files=$TASK_FILE_LIST"

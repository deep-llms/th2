#1 +60+a
#th2-tpbw-clean-partial-raw-download-20260921-a01
set -euo pipefail
date -u
hostname
TASK_DIR=/mnt/local/_data/deep-llms_th2/data
echo '--- contents before'
if [ ! -e "$TASK_DIR" ]; then echo download-dir-absent-nothing-to-clean; exit 0; fi
test ! -L "$TASK_DIR"
find "$TASK_DIR" | sort
du -sh "$TASK_DIR" || true
TASK_N=$(find "$TASK_DIR" -type f | wc -l)
test "$TASK_N" -le 60
TASK_BAD=$(find "$TASK_DIR" -type f ! -path "$TASK_DIR/raw/en/*" | wc -l)
test "$TASK_BAD" -eq 0
rm -rf "$TASK_DIR"
echo '--- after'
ls -la /mnt/local/_data/deep-llms_th2/ 2>/dev/null || echo project-data-root-absent
echo '--- models dir (report only, not touched)'
ls -laR /mnt/local/_models/deep-llms_th2/ 2>/dev/null || echo models-root-absent
echo TPBW_RAW_CLEAN_DONE

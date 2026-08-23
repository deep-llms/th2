#1 +60+a
#th2-verify-culturax-sampled-data-20260823
set -euo pipefail

echo '=== verify sampled CulturaX download ==='
date -u
hostname

TASK_DATA_ROOT="/mnt/local/_data/@PROJECT@/data"
TASK_RAW_DIR="$TASK_DATA_ROOT/raw"
TASK_MANIFEST="resources/culturax_raw_manifest.tsv"

case "$TASK_RAW_DIR" in
    /mnt/local/_data/*/data/raw) ;;
    *)
        echo "REFUSE: unexpected raw-data path: $TASK_RAW_DIR"
        exit 1
        ;;
esac

test -d "$TASK_RAW_DIR"
test -s "$TASK_MANIFEST"

TASK_TMP_DIR="$(mktemp -d /tmp/culturax-verify.XXXXXX)"
trap 'rm -rf -- "$TASK_TMP_DIR"' EXIT

echo '=== verify exact parquet path set ==='
awk '!/^#/ {print $3}' "$TASK_MANIFEST" | sort > "$TASK_TMP_DIR/expected_paths.txt"
find "$TASK_RAW_DIR" -type f -name '*.parquet' -printf '%P\n' | sort > "$TASK_TMP_DIR/actual_paths.txt"
diff -u "$TASK_TMP_DIR/expected_paths.txt" "$TASK_TMP_DIR/actual_paths.txt"

TASK_EXPECTED_FILES="$(wc -l < "$TASK_TMP_DIR/expected_paths.txt")"
TASK_ACTUAL_FILES="$(wc -l < "$TASK_TMP_DIR/actual_paths.txt")"
echo "expected_parquet_files=$TASK_EXPECTED_FILES"
echo "actual_parquet_files=$TASK_ACTUAL_FILES"
test "$TASK_EXPECTED_FILES" -eq 75
test "$TASK_ACTUAL_FILES" -eq 75

for TASK_LANG_SPEC in en:50 vi:5 zh:5 ru:5 de:5 ar:5; do
    TASK_LANG="${TASK_LANG_SPEC%%:*}"
    TASK_EXPECTED_LANG_COUNT="${TASK_LANG_SPEC##*:}"
    TASK_ACTUAL_LANG_COUNT="$(find "$TASK_RAW_DIR/$TASK_LANG" -maxdepth 1 -type f -name '*.parquet' | wc -l)"
    echo "$TASK_LANG=$TASK_ACTUAL_LANG_COUNT"
    test "$TASK_ACTUAL_LANG_COUNT" -eq "$TASK_EXPECTED_LANG_COUNT"
done

TASK_UNEXPECTED_FILES="$(find "$TASK_RAW_DIR" -type f ! -path "$TASK_RAW_DIR/.cache/*" ! -name '*.parquet' -print)"
if [ -n "$TASK_UNEXPECTED_FILES" ]; then
    echo 'ERROR: unexpected non-cache files under raw data:'
    printf '%s\n' "$TASK_UNEXPECTED_FILES"
    exit 1
fi

echo '=== verify every file size ==='
while IFS=$'\t' read -r TASK_EXPECTED_HASH TASK_EXPECTED_BYTES TASK_RELATIVE_PATH; do
    case "$TASK_EXPECTED_HASH" in
        ''|'#'*) continue ;;
    esac
    TASK_ACTUAL_BYTES="$(stat -c '%s' "$TASK_RAW_DIR/$TASK_RELATIVE_PATH")"
    if [ "$TASK_ACTUAL_BYTES" != "$TASK_EXPECTED_BYTES" ]; then
        echo "ERROR: size mismatch: $TASK_RELATIVE_PATH expected=$TASK_EXPECTED_BYTES actual=$TASK_ACTUAL_BYTES"
        exit 1
    fi
done < "$TASK_MANIFEST"

TASK_EXPECTED_BYTES="$(awk '!/^#/ {sum += $2} END {printf "%.0f", sum}' "$TASK_MANIFEST")"
TASK_ACTUAL_BYTES="$(find "$TASK_RAW_DIR" -type f -name '*.parquet' -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
echo "expected_bytes=$TASK_EXPECTED_BYTES"
echo "actual_bytes=$TASK_ACTUAL_BYTES"
test "$TASK_EXPECTED_BYTES" = '166107112571'
test "$TASK_ACTUAL_BYTES" = "$TASK_EXPECTED_BYTES"

echo '=== verify all SHA-256 hashes; this reads the full 166 GB ==='
TASK_HASH_START="$SECONDS"
cd "$TASK_RAW_DIR"
sha256sum --quiet -c <(awk '!/^#/ {print $1 "  " $3}' "$OLDPWD/$TASK_MANIFEST")
TASK_HASH_SECONDS="$((SECONDS - TASK_HASH_START))"

echo "sha256_files_verified=$TASK_ACTUAL_FILES"
echo "sha256_elapsed_seconds=$TASK_HASH_SECONDS"
du -sh "$TASK_DATA_ROOT"
echo 'TH2 CULTURAX DATA VERIFICATION OK'

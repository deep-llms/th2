#1 +60+a
#th2-diagnose-culturax-sampling-preflight-20260823-0212
set -u

echo '=== diagnose failed CulturaX sampling preflight ==='
date -u
hostname

TASK_PROJECT_DIR="$PWD"
TASK_PYTHON="/mnt/local/conda/envs/sparse_emb/bin/python"
TASK_DATA_ROOT="/mnt/local/_data/@PROJECT@/data"
TASK_RAW_DIR="$TASK_DATA_ROOT/raw"
TASK_OUTPUT_ROOT="$TASK_DATA_ROOT/Qwen_Qwen3-0.6B"
TASK_MODEL_DIR="/mnt/local/_models/@PROJECT@/Qwen3-0.6B"
TASK_MANIFEST="$TASK_PROJECT_DIR/resources/culturax_raw_manifest.tsv"
TASK_PREPARE_SCRIPT="$TASK_PROJECT_DIR/prepare_data.py"

for TASK_CHECK in \
    "$TASK_PYTHON" \
    "$TASK_DATA_ROOT" \
    "$TASK_RAW_DIR" \
    "$TASK_MODEL_DIR" \
    "$TASK_MODEL_DIR/tokenizer.json" \
    "$TASK_MODEL_DIR/tokenizer_config.json" \
    "$TASK_MANIFEST" \
    "$TASK_PREPARE_SCRIPT"; do
    if [ -e "$TASK_CHECK" ]; then
        echo "EXISTS: $TASK_CHECK"
        ls -ld "$TASK_CHECK"
    else
        echo "MISSING: $TASK_CHECK"
    fi
done

echo '=== model top level ==='
find "$TASK_MODEL_DIR" -mindepth 1 -maxdepth 2 -printf '%y %P -> %l\n' 2>/dev/null | sort | head -100 || true
echo '=== raw/output summary ==='
find "$TASK_RAW_DIR" -type f -name '*.parquet' 2>/dev/null | wc -l
if [ -e "$TASK_OUTPUT_ROOT" ]; then du -sh "$TASK_OUTPUT_ROOT"; else echo 'sample_output=absent'; fi
echo 'TH2 CULTURAX PREFLIGHT DIAGNOSIS COMPLETE'

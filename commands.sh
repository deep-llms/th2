#2 +a -f-/mnt/local/_outputs/deep-llms_th2/result_exports/ranklift_hashedv2_btmos_eval_finetune_partial_20260904.tar.gz,/mnt/local/_outputs/deep-llms_th2/result_exports/ranklift_hashedv2_btmos_eval_finetune_partial_20260904.tar.gz.sha256,/mnt/local/_outputs/deep-llms_th2/result_exports/ranklift_hashedv2_btmos_eval_finetune_partial_20260904.files,/mnt/local/_outputs/deep-llms_th2/result_exports/ranklift_hashedv2_btmos_eval_finetune_partial_20260904.status.txt
#th2-pull-five-checkpoint-eval-and-partial-finetune-results-20260904-a01
set -euo pipefail

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:-/mnt/local/@PROJECT@}"
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_FINETUNE_REL=finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/$TASK_FINETUNE_REL"
TASK_EVAL_LAUNCH_REL=eval_parallel_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01.log
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_EXPORT_STEM=ranklift_hashedv2_btmos_eval_finetune_partial_20260904
TASK_ARCHIVE="$TASK_EXPORT_DIR/$TASK_EXPORT_STEM.tar.gz"
TASK_MANIFEST="$TASK_EXPORT_DIR/$TASK_EXPORT_STEM.files"
TASK_STATUS="$TASK_EXPORT_DIR/$TASK_EXPORT_STEM.status.txt"
TASK_CHECKSUM="$TASK_ARCHIVE.sha256"
TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_CHECKPOINT_RELS=(
    ranklift_tied_c124_m460/checkpoint-6500
    ranklift_tied_c124_m460/checkpoint-10000
    product_code_quota_h6144/checkpoint-6500
    product_code_quota_h6144/checkpoint-10000
    btmos_k3_c256_lb/checkpoint-6500
)
TASK_EXPERIMENT_RELS=(
    ranklift_tied_c124_m460
    product_code_quota_h6144
    btmos_k3_c256_lb
)

cd "$TASK_PROJECT_DIR"
echo '=== export preflight; GPU burn is read-only ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
[[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)" -eq 8 ]]
! pgrep -f '[f]inetune/run_all.py' >/dev/null
! pgrep -f '[f]inetune/train.py' >/dev/null
! pgrep -f '[r]un_five_checkpoint_eval_finetune_burn.sh' >/dev/null

test -x "$TASK_PYTHON"
test -s "$TASK_OUTPUT_BASE/$TASK_EVAL_LAUNCH_REL"
test -d "$TASK_FINETUNE_OUTPUT"

echo '=== validate five complete eval result sets ==='
"$TASK_PYTHON" - "$TASK_OUTPUT_BASE" "${TASK_CHECKPOINT_RELS[@]}" <<'PY'
import json
import math
from pathlib import Path
import sys

base = Path(sys.argv[1])
checkpoints = [base / value for value in sys.argv[2:]]
assert len(checkpoints) == 5
expected_languages = {'en', 'vi', 'zh', 'ru', 'de', 'ar'}
for checkpoint in checkpoints:
    ppl_path = checkpoint / 'eval_ppl.json'
    benchmark_path = checkpoint / 'eval_benchmarks.json'
    log_path = checkpoint / 'eval.log'
    assert all(path.is_file() and path.stat().st_size > 0 for path in (ppl_path, benchmark_path, log_path))
    ppl = json.loads(ppl_path.read_text())
    benchmarks = json.loads(benchmark_path.read_text())
    assert set(ppl) == expected_languages, (checkpoint, ppl.keys())
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for metrics in ppl.values():
        assert int(metrics['num_tokens']) > 0
        assert math.isfinite(float(metrics['loss']))
        assert math.isfinite(float(metrics['perplexity']))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get('acc,none', metrics.get('acc'))
        assert accuracy is not None and math.isfinite(float(accuracy)), (checkpoint, task)
    print(f'EVAL_OK checkpoint={checkpoint} languages=6 benchmarks=26')
PY

echo '=== validate all available finetune results ==='
"$TASK_PYTHON" - "$TASK_FINETUNE_OUTPUT" <<'PY'
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
json_paths = sorted(root.glob('*.json'))
log_paths = sorted(root.glob('*.log'))
model_paths = sorted((root / 'models').glob('*/model_state.pt'))
assert len(json_paths) == 41, len(json_paths)
assert len(log_paths) == 45, len(log_paths)
assert len(model_paths) == 41, len(model_paths)

expected_eval_counts = {'hellaswag': 5, 'arc_easy': 6, 'xnli': 6}
completed_stems = set()
for path in json_paths:
    result = json.loads(path.read_text())
    task = result['task']
    assert task in expected_eval_counts, (path, task)
    assert int(result['seed']) in {42, 123, 456}, path
    assert int(result['epochs']) == 3, path
    assert math.isfinite(float(result['train_time_s'])), path
    assert len(result['eval_results']) == expected_eval_counts[task], path
    for metrics in result['eval_results'].values():
        assert metrics.get('acc') is not None and math.isfinite(float(metrics['acc'])), path
        if metrics.get('acc_norm') is not None:
            assert math.isfinite(float(metrics['acc_norm'])), path
    model_path = root / 'models' / path.stem / 'model_state.pt'
    assert model_path.is_file() and model_path.stat().st_size > 0, model_path
    completed_stems.add(path.stem)

log_stems = {path.stem for path in log_paths}
cancelled = sorted(log_stems - completed_stems)
assert cancelled == [
    'xnli_btmos_k3_c256_lb_s6500_seed456',
    'xnli_hashedv2_h6144_s10000_seed123',
    'xnli_hashedv2_h6144_s10000_seed42',
    'xnli_hashedv2_h6144_s10000_seed456',
], cancelled
print('FINETUNE_AVAILABLE_OK completed=41 expected=45 cancelled=4')
for stem in cancelled:
    print(f'CANCELLED_RESULT_ABSENT stem={stem}')
PY

mkdir -p "$TASK_EXPORT_DIR"
for TASK_PATH in "$TASK_ARCHIVE" "$TASK_MANIFEST" "$TASK_STATUS" "$TASK_CHECKSUM"; do
    [[ ! -e "$TASK_PATH" ]] || {
        echo "ERROR: refusing to overwrite existing export: $TASK_PATH" >&2
        exit 1
    }
done

printf '%s\n' \
    'status=partial_finetune_cancelled_by_user' \
    'eval_checkpoints_complete=5' \
    'eval_languages_per_checkpoint=6' \
    'eval_benchmarks_per_checkpoint=26' \
    'finetune_expected_jobs=45' \
    'finetune_completed_jobs=41' \
    'finetune_cancelled_jobs=4' \
    'cancelled=xnli_hashedv2_h6144_s10000_seed42' \
    'cancelled=xnli_hashedv2_h6144_s10000_seed123' \
    'cancelled=xnli_hashedv2_h6144_s10000_seed456' \
    'cancelled=xnli_btmos_k3_c256_lb_s6500_seed456' \
    > "$TASK_STATUS"

cd "$TASK_OUTPUT_BASE"
{
    printf '%s\n' "$TASK_EVAL_LAUNCH_REL"
    printf '%s\n' "${TASK_EXPORT_STEM}.status.txt" | sed "s#^#result_exports/#"
    for TASK_EXPERIMENT in "${TASK_EXPERIMENT_RELS[@]}"; do
        printf '%s\n' "$TASK_EXPERIMENT/train_config.json"
    done
    for TASK_CHECKPOINT in "${TASK_CHECKPOINT_RELS[@]}"; do
        printf '%s\n' \
            "$TASK_CHECKPOINT/config.json" \
            "$TASK_CHECKPOINT/trainer_state.json" \
            "$TASK_CHECKPOINT/eval.log" \
            "$TASK_CHECKPOINT/eval_ppl.json" \
            "$TASK_CHECKPOINT/eval_benchmarks.json"
    done
    find "$TASK_FINETUNE_REL" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.log' -o -name 'summary.md' \) -print
} | LC_ALL=C sort -u > "$TASK_MANIFEST"

TASK_FILE_COUNT="$(wc -l < "$TASK_MANIFEST")"
[[ "$TASK_FILE_COUNT" -ge 116 && "$TASK_FILE_COUNT" -le 117 ]] || {
    echo "ERROR: unexpected manifest count: $TASK_FILE_COUNT" >&2
    exit 1
}
while IFS= read -r TASK_RELATIVE_PATH; do
    [[ -n "$TASK_RELATIVE_PATH" && -s "$TASK_RELATIVE_PATH" ]] || {
        echo "ERROR: missing or empty export member: $TASK_RELATIVE_PATH" >&2
        exit 1
    }
done < "$TASK_MANIFEST"

tar -czf "$TASK_ARCHIVE" -T "$TASK_MANIFEST"
(
    cd "$TASK_EXPORT_DIR"
    sha256sum "$(basename "$TASK_ARCHIVE")" > "$(basename "$TASK_CHECKSUM")"
    sha256sum -c "$(basename "$TASK_CHECKSUM")"
)
gzip -t "$TASK_ARCHIVE"
[[ "$(tar -tzf "$TASK_ARCHIVE" | wc -l)" -eq "$TASK_FILE_COUNT" ]]
if tar -tzf "$TASK_ARCHIVE" \
        | grep -E '(^|/)(model_state\.pt|model\.safetensors|embedding\.pt|optimizer\.pt|scheduler\.pt)$'; then
    echo 'ERROR: model or optimizer weights entered result archive' >&2
    exit 1
fi
TASK_ARCHIVE_BYTES="$(stat -c '%s' "$TASK_ARCHIVE")"
(( TASK_ARCHIVE_BYTES < 250000000 )) || {
    echo "ERROR: result archive unexpectedly large: $TASK_ARCHIVE_BYTES" >&2
    exit 1
}

echo "EXPORT_FILE_COUNT=$TASK_FILE_COUNT"
echo "EXPORT_ARCHIVE_BYTES=$TASK_ARCHIVE_BYTES"
cat "$TASK_CHECKSUM"
ls -lh "$TASK_ARCHIVE" "$TASK_MANIFEST" "$TASK_STATUS" "$TASK_CHECKSUM"
echo 'TH2 FIVE-CHECKPOINT EVAL AND AVAILABLE FINETUNE RESULT EXPORT READY; GPU BURN UNTOUCHED'

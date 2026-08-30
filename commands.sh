#1 +60+a
#th2-export-four-model-eval-finetune-results-20260830-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_EXPORT_STEM=phase1_four_model_eval_finetune_results_20260830
TASK_ARCHIVE="$TASK_EXPORT_DIR/$TASK_EXPORT_STEM.tar.gz"
TASK_MANIFEST="$TASK_EXPORT_DIR/$TASK_EXPORT_STEM.files"
TASK_CHECKSUM="$TASK_ARCHIVE.sha256"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_nested_groupreduce_two_dense_10k_20260830"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_nested_groupreduce_two_dense_10k_20260830.log"
TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_EXPERIMENTS=(
    nested_ladder_tied_t4
    groupreduce_matched_nested_tied_t4
    dense_tied_baseline_b200
    dense_tied_baseline_b200_ddp_default
)
TASK_CHECKPOINTS=(
    "$TASK_OUTPUT_BASE/nested_ladder_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default/checkpoint-10000"
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

echo '=== validate completed workflow and active burns before export ==='
date -u
hostname
test -x "$TASK_PYTHON"
test -d "$TASK_OUTPUT_BASE"
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 8 ]] \
    || die "expected 8 final burn processes, found ${#TASK_GPU_PIDS[@]}: ${TASK_GPU_PIDS[*]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    [[ "$TASK_PID" -ne 1 ]] || die 'refusing PID 1'
    TASK_CMDLINE=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")
    [[ "$TASK_CMDLINE" == *scripts/gpu_burn.py* ]] \
        || die "GPU PID is not a project burn: $TASK_PID $TASK_CMDLINE"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader

test -s "$TASK_EVAL_LAUNCH_LOG"
grep -F 'All 4 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
test -s "$TASK_FINETUNE_OUTPUT/summary.md"
[[ "$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 36 ]]
[[ "$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | wc -l)" -eq 36 ]]
[[ "$(find "$TASK_FINETUNE_OUTPUT/models" -mindepth 2 -maxdepth 2 -type f -name model_state.pt | wc -l)" -eq 36 ]]

"$TASK_PYTHON" - \
    "$TASK_FINETUNE_OUTPUT" \
    "$TASK_OUTPUT_BASE/nested_ladder_tied_t4/checkpoint-10000" \
    "$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4/checkpoint-10000" \
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000" \
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default/checkpoint-10000" <<'PY'
import glob
import json
import math
import os
import sys

output_dir, nested, groupreduce, dense, dense_ddp = sys.argv[1:]
arms = {
    "nested_ladder_tied_t4": nested,
    "groupreduce_matched_nested_tied_t4": groupreduce,
    "dense_tied_baseline_b200": dense,
    "dense_tied_baseline_b200_ddp_default": dense_ddp,
}
expected_eval_tasks = {
    "hellaswag": {"hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi"},
    "arc_easy": {"arc_easy", "arc_ar", "arc_de", "arc_ru", "arc_vi", "arc_zh"},
    "xnli": {"xnli_en", "xnli_vi", "xnli_zh", "xnli_de", "xnli_ru", "xnli_ar"},
}

for checkpoint in arms.values():
    with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == {"en", "vi", "zh", "ru", "de", "ar"}, checkpoint
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for metrics in perplexity.values():
        assert int(metrics["num_tokens"]) > 0
        assert math.isfinite(float(metrics["loss"]))
        assert math.isfinite(float(metrics["perplexity"]))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get("acc,none", metrics.get("acc"))
        assert accuracy is not None and math.isfinite(float(accuracy)), (checkpoint, task)

result_paths = sorted(glob.glob(os.path.join(output_dir, "*.json")))
log_paths = sorted(glob.glob(os.path.join(output_dir, "*.log")))
assert len(result_paths) == len(log_paths) == 36
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for arm, checkpoint in arms.items():
        for seed in (42, 123, 456):
            path = os.path.join(output_dir, f"{task}_{arm}_seed{seed}.json")
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            assert result["checkpoint"] == checkpoint, path
            assert result["task"] == task, path
            assert int(result["seed"]) == seed, path
            assert int(result["epochs"]) == 3, path
            assert set(result["eval_results"]) == expected_tasks, path
            assert math.isfinite(float(result["train_time_s"])), path
            for metrics in result["eval_results"].values():
                assert metrics.get("acc") is not None and math.isfinite(float(metrics["acc"])), path
                if metrics.get("acc_norm") is not None:
                    assert math.isfinite(float(metrics["acc_norm"])), path
            validated += 1
assert validated == 36
print("FOUR_MODEL_RESULTS_VALID eval_models=4 finetune_jobs=36")
PY

echo '=== narrow fatal scan before export ==='
TASK_FATAL_PATTERN='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|FAILED \(code|eval failed:|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
if grep -HniE "$TASK_FATAL_PATTERN" \
    "$TASK_EVAL_LAUNCH_LOG" "${TASK_CHECKPOINTS[@]/%//eval.log}" \
    "$TASK_FINETUNE_OUTPUT"/*.log; then
    die 'fatal signature found in result logs'
fi

echo '=== create compact result-only archive; exclude all model weights ==='
mkdir -p "$TASK_EXPORT_DIR"
for TASK_PATH in "$TASK_ARCHIVE" "$TASK_MANIFEST" "$TASK_CHECKSUM"; do
    [[ ! -e "$TASK_PATH" ]] || die "refusing to overwrite existing export: $TASK_PATH"
done
cd "$TASK_OUTPUT_BASE"
{
    printf '%s\n' "${TASK_EVAL_LAUNCH_LOG#"$TASK_OUTPUT_BASE/"}"
    for TASK_EXPERIMENT in "${TASK_EXPERIMENTS[@]}"; do
        printf '%s\n' \
            "$TASK_EXPERIMENT/checkpoint-10000/config.json" \
            "$TASK_EXPERIMENT/checkpoint-10000/trainer_state.json" \
            "$TASK_EXPERIMENT/checkpoint-10000/eval.log" \
            "$TASK_EXPERIMENT/checkpoint-10000/eval_ppl.json" \
            "$TASK_EXPERIMENT/checkpoint-10000/eval_benchmarks.json"
        if [[ -s "$TASK_EXPERIMENT/train_config.json" ]]; then
            printf '%s\n' "$TASK_EXPERIMENT/train_config.json"
        fi
    done
    find "${TASK_FINETUNE_OUTPUT#"$TASK_OUTPUT_BASE/"}" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.log' -o -name 'summary.md' \) -print
} | LC_ALL=C sort -u > "$TASK_MANIFEST"

TASK_FILE_COUNT=$(wc -l < "$TASK_MANIFEST")
[[ "$TASK_FILE_COUNT" -ge 94 ]] \
    || die "too few result files in manifest: $TASK_FILE_COUNT"
while IFS= read -r TASK_RELATIVE_PATH; do
    [[ -n "$TASK_RELATIVE_PATH" ]] || die 'empty manifest entry'
    test -s "$TASK_RELATIVE_PATH" || die "missing or empty export file: $TASK_RELATIVE_PATH"
done < "$TASK_MANIFEST"

tar -czf "$TASK_ARCHIVE" -T "$TASK_MANIFEST"
(cd "$TASK_EXPORT_DIR" && sha256sum "$(basename "$TASK_ARCHIVE")" > "$(basename "$TASK_CHECKSUM")")
(cd "$TASK_EXPORT_DIR" && sha256sum -c "$(basename "$TASK_CHECKSUM")")
gzip -t "$TASK_ARCHIVE"
[[ "$(tar -tzf "$TASK_ARCHIVE" | wc -l)" -eq "$TASK_FILE_COUNT" ]]
if tar -tzf "$TASK_ARCHIVE" \
    | grep -E '(^|/)(model_state\.pt|model\.safetensors|embedding\.pt|optimizer\.pt|scheduler\.pt)$'; then
    die 'model weights or optimizer state accidentally included in result archive'
fi
TASK_ARCHIVE_BYTES=$(stat -c '%s' "$TASK_ARCHIVE")
(( TASK_ARCHIVE_BYTES < 250000000 )) \
    || die "result archive unexpectedly exceeds 250 MB: $TASK_ARCHIVE_BYTES"

echo "EXPORT_FILE_COUNT=$TASK_FILE_COUNT"
echo "EXPORT_ARCHIVE_BYTES=$TASK_ARCHIVE_BYTES"
cat "$TASK_CHECKSUM"
ls -lh "$TASK_ARCHIVE" "$TASK_MANIFEST" "$TASK_CHECKSUM"
echo 'TH2 FOUR-MODEL EVAL/FINETUNE RESULT EXPORT READY; GPU BURNS UNTOUCHED'

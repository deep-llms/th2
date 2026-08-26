#1 +120+a
#th2-export-pure-local-results-20260826-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/deep-llms_th2
TASK_CHECKPOINT="$TASK_OUTPUT_BASE/pure_local_tied_g16_r128/checkpoint-10000"
TASK_TRAIN_CONFIG="$TASK_OUTPUT_BASE/pure_local_tied_g16_r128/train_config.json"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_pure_local_tied_g16_r128_10k_20260825.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_pure_local_tied_g16_r128_10k_20260825"
TASK_EXPORT_DIR="$TASK_OUTPUT_BASE/result_exports"
TASK_ARCHIVE_NAME=pure_local_eval_finetune_results_20260826.tar.gz
TASK_MANIFEST_NAME=pure_local_eval_finetune_results_20260826.files
TASK_SHA_NAME=pure_local_eval_finetune_results_20260826.tar.gz.sha256
TASK_ARCHIVE="$TASK_EXPORT_DIR/$TASK_ARCHIVE_NAME"
TASK_MANIFEST="$TASK_EXPORT_DIR/$TASK_MANIFEST_NAME"
TASK_SHA="$TASK_EXPORT_DIR/$TASK_SHA_NAME"
TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11

die() {
    echo "ERROR: $*"
    exit 1
}

echo '=== validate completed Pure-local workflow before compact export ==='
date -u
hostname
test -x "$TASK_PYTHON"

mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 8 ]] \
    || die "workflow is not at final burn stage: expected 8 GPU processes, found ${#TASK_GPU_PIDS[@]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    test -r "/proc/$TASK_PID/cmdline"
    TASK_CMDLINE=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")
    [[ "$TASK_CMDLINE" == *scripts/gpu_burn.py* ]] \
        || die "non-burn GPU process remains: pid=$TASK_PID cmd=$TASK_CMDLINE"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

for TASK_FILE in \
    "$TASK_TRAIN_CONFIG" \
    "$TASK_CHECKPOINT/config.json" \
    "$TASK_CHECKPOINT/trainer_state.json" \
    "$TASK_CHECKPOINT/eval.log" \
    "$TASK_CHECKPOINT/eval_ppl.json" \
    "$TASK_CHECKPOINT/eval_benchmarks.json" \
    "$TASK_EVAL_LAUNCH_LOG" \
    "$TASK_FINETUNE_OUTPUT/summary.md"; do
    test -s "$TASK_FILE" || die "missing result artifact: $TASK_FILE"
done

"$TASK_PYTHON" - "$TASK_CHECKPOINT" "$TASK_FINETUNE_OUTPUT" <<'PY'
import glob
import json
import math
import os
import sys

checkpoint, finetune_output = sys.argv[1:]
with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
    assert int(json.load(handle)["global_step"]) == 10000
with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
    ppl = json.load(handle)
with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
    benchmarks = json.load(handle)
assert set(ppl) == {"en", "vi", "zh", "ru", "de", "ar"}
for language, metrics in ppl.items():
    assert int(metrics["num_tokens"]) > 0, (language, metrics)
    assert math.isfinite(float(metrics["loss"])), (language, metrics)
    assert math.isfinite(float(metrics["perplexity"])), (language, metrics)
assert len(benchmarks) == 26, len(benchmarks)
for task, metrics in benchmarks.items():
    accuracy = metrics.get("acc,none", metrics.get("acc"))
    assert accuracy is not None and math.isfinite(float(accuracy)), (task, metrics)
result_paths = sorted(glob.glob(os.path.join(finetune_output, "*.json")))
log_paths = sorted(glob.glob(os.path.join(finetune_output, "*.log")))
assert len(result_paths) == len(log_paths) == 9, (len(result_paths), len(log_paths))
for result_path in result_paths:
    with open(result_path, encoding="utf-8") as handle:
        result = json.load(handle)
    assert result["checkpoint"] == checkpoint, result["checkpoint"]
    assert int(result["epochs"]) == 3, result["epochs"]
    assert math.isfinite(float(result["train_time_s"])), result["train_time_s"]
print("PURE_LOCAL_RESULTS_VALID jobs=9 ppl_languages=6 benchmark_tasks=26")
PY

grep -F 'Loaded compositional model: arm=pure_local' "$TASK_CHECKPOINT/eval.log"
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*.log; do
    grep -Fq 'Loaded compositional model: arm=pure_local' "$TASK_LOG" \
        || die "Pure-local loader confirmation missing from $TASK_LOG"
done

echo '=== build idempotent compact result archive (no model weights) ==='
mkdir -p "$TASK_EXPORT_DIR"
if [[ -s "$TASK_ARCHIVE" && -s "$TASK_MANIFEST" && -s "$TASK_SHA" ]]; then
    (cd "$TASK_EXPORT_DIR" && sha256sum -c "$TASK_SHA_NAME")
    echo 'TH2 PURE-LOCAL COMPACT RESULT EXPORT OK (EXISTING VERIFIED)'
    exit 0
fi
rm -f "$TASK_ARCHIVE" "$TASK_MANIFEST" "$TASK_SHA"
cd "$TASK_OUTPUT_BASE"
{
    printf '%s\n' \
        pure_local_tied_g16_r128/train_config.json \
        pure_local_tied_g16_r128/checkpoint-10000/config.json \
        pure_local_tied_g16_r128/checkpoint-10000/trainer_state.json \
        pure_local_tied_g16_r128/checkpoint-10000/eval.log \
        pure_local_tied_g16_r128/checkpoint-10000/eval_ppl.json \
        pure_local_tied_g16_r128/checkpoint-10000/eval_benchmarks.json \
        eval_parallel_pure_local_tied_g16_r128_10k_20260825.log
    find finetune_pure_local_tied_g16_r128_10k_20260825 -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.log' -o -name 'summary.md' \) -print
} | LC_ALL=C sort > "$TASK_MANIFEST"
[[ "$(wc -l < "$TASK_MANIFEST")" -eq 26 ]] \
    || die "expected 26 compact result files, found $(wc -l < "$TASK_MANIFEST")"
tar -czf "$TASK_ARCHIVE" -T "$TASK_MANIFEST"
(cd "$TASK_EXPORT_DIR" && sha256sum "$TASK_ARCHIVE_NAME" > "$TASK_SHA_NAME")
(cd "$TASK_EXPORT_DIR" && sha256sum -c "$TASK_SHA_NAME")
tar -tzf "$TASK_ARCHIVE" | grep -E '(^|/)(model_state\.pt|model\.safetensors|embedding\.pt|optimizer\.pt)$' >/dev/null \
    && die 'model weights were accidentally included in compact export'

ls -lh "$TASK_ARCHIVE" "$TASK_MANIFEST" "$TASK_SHA"
echo 'TH2 PURE-LOCAL COMPACT RESULT EXPORT OK'

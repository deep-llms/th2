#1 +120+a
#th2-watch-eval-then-finetune-matched-tied-controls-20260825
#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_GLOBAL_CKPT="$TASK_OUTPUT_BASE/global_lowrank_tied_r128_b200/checkpoint-10000"
TASK_DENSE_CKPT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_global_lr_tied_r128_dense_tied_b200_10k_20260825.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_global_lr_tied_r128_dense_tied_b200_10k_20260825"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

die() {
    echo "ERROR: $*"
    exit 1
}

matched_eval_processes() {
    local proc cmd
    for proc in /proc/[0-9]*; do
        [ -r "$proc/cmdline" ] || continue
        cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
        [[ "$cmd" == *python* ]] || continue
        if [[ "$cmd" == *eval/eval_parallel.py* \
                && "$cmd" == *global_lowrank_tied_r128_b200/checkpoint-10000* \
                && "$cmd" == *dense_tied_baseline_b200/checkpoint-10000* ]] \
            || [[ "$cmd" == *eval/eval_checkpoint.py* \
                && "$cmd" == *global_lowrank_tied_r128_b200/checkpoint-10000* ]] \
            || [[ "$cmd" == *eval/eval_checkpoint.py* \
                && "$cmd" == *dense_tied_baseline_b200/checkpoint-10000* ]]; then
            printf '%s\n' "${proc#/proc/}"
        fi
    done | sort -nu
}

echo '=== monitor the exact current two-checkpoint evaluation ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_EVAL_PYTHON"
for poll in $(seq 1 180); do
    mapfile -t TASK_EVAL_PIDS < <(matched_eval_processes)
    if [[ "${#TASK_EVAL_PIDS[@]}" -eq 0 ]]; then
        echo "eval_processes=0 poll=$poll"
        break
    fi
    latest_global=$(tail -c 200000 "$TASK_GLOBAL_CKPT/eval.log" 2>/dev/null \
        | tr '\r' '\n' | tail -1 || true)
    latest_dense=$(tail -c 200000 "$TASK_DENSE_CKPT/eval.log" 2>/dev/null \
        | tr '\r' '\n' | tail -1 || true)
    echo "eval_poll=$poll processes=${#TASK_EVAL_PIDS[@]} pids=${TASK_EVAL_PIDS[*]}"
    echo "global_tail=$latest_global"
    echo "dense_tail=$latest_dense"
    sleep 60
done
mapfile -t TASK_EVAL_PIDS < <(matched_eval_processes)
[[ "${#TASK_EVAL_PIDS[@]}" -eq 0 ]] \
    || die "evaluation still running after three hours: ${TASK_EVAL_PIDS[*]}"
sleep 10

echo '=== validate both completed full evaluations ==='
"$TASK_EVAL_PYTHON" - "$TASK_GLOBAL_CKPT" "$TASK_DENSE_CKPT" <<'PY'
import json
import math
import os
import sys

expected_languages = {"en", "vi", "zh", "ru", "de", "ar"}
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == expected_languages, (checkpoint, perplexity.keys())
    for language, metrics in perplexity.items():
        assert int(metrics["num_tokens"]) > 0, (checkpoint, language, metrics)
        assert math.isfinite(float(metrics["loss"])), (checkpoint, language, metrics)
        assert math.isfinite(float(metrics["perplexity"])), (checkpoint, language, metrics)
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get("acc,none", metrics.get("acc"))
        assert accuracy is not None and math.isfinite(float(accuracy)), (
            checkpoint, task, metrics
        )
    print(f"EVAL_COMPLETE_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26")
PY
test -s "$TASK_EVAL_LAUNCH_LOG"
grep -F 'All 2 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
grep -F 'Loaded compositional model: arm=lowrank' "$TASK_GLOBAL_CKPT/eval.log"
if grep -Fq 'Loaded compositional model:' "$TASK_DENSE_CKPT/eval.log"; then
    die 'dense checkpoint was incorrectly loaded as compositional'
fi
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|FAILED \(code|Error:' \
    "$TASK_GLOBAL_CKPT/eval.log" "$TASK_DENSE_CKPT/eval.log" "$TASK_EVAL_LAUNCH_LOG"; then
    die 'failure signature found in completed evaluation logs'
fi
echo 'TH2 BOTH MATCHED TIED CONTROL EVALUATIONS COMPLETE AND VALID'

echo '=== require all eight B200 GPUs free after evaluation ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
    || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for index in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
        || die "GPU $index is not B200: ${TASK_GPU_NAMES[$index]}"
done
for attempt in $(seq 1 30); do
    mapfile -t TASK_GPU_PIDS < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
    )
    [[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] && break
    echo "waiting_for_gpu_release attempt=$attempt pids=${TASK_GPU_PIDS[*]}"
    sleep 10
done
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after evaluation: ${TASK_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 ALL 8 B200 GPUS FREE AFTER EVAL AND BEFORE FINETUNE'

echo '=== standard two-checkpoint finetune preflight ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for relpath in \
    Rowan/hellaswag allenai/ai2_arc facebook/xnli \
    alexandrainst/m_arc alexandrainst/m_hellaswag \
    facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x; do
    test -d "$TASK_BENCH_ROOT/$relpath" \
        || die "missing offline dataset: $TASK_BENCH_ROOT/$relpath"
done
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import patch_lm_eval_dataset_paths

patch_lm_eval_dataset_paths(os.environ["LM_EVAL_DATASET_ROOT"])
print("OFFLINE_LM_EVAL_PATHS_OK")
PY
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to reuse finetune output: $TASK_FINETUNE_OUTPUT"
available_bytes=$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')
[[ "$available_bytes" =~ ^[0-9]+$ ]] || die 'could not determine free storage'
(( available_bytes >= 40000000000 )) \
    || die "less than 40 GB available for 18 finetune jobs: $available_bytes bytes"
echo "FINETUNE_PREFLIGHT_OK fresh_output=$TASK_FINETUNE_OUTPUT available_bytes=$available_bytes"
echo 'PROTOCOL checkpoints=2 tasks=3 seeds=3 jobs=18 epochs=3 bf16 max_parallel_jobs=8'

echo '=== launch one 18-job finetune queue across all eight GPUs ==='
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints \
        global_lowrank_tied_r128_b200="$TASK_GLOBAL_CKPT" \
        dense_tied_baseline_b200="$TASK_DENSE_CKPT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all 18 finetune jobs and summary ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" "$TASK_GLOBAL_CKPT" "$TASK_DENSE_CKPT" <<'PY'
import json
import math
import os
import sys

output_dir, global_ckpt, dense_ckpt = sys.argv[1:]
arms = {
    "global_lowrank_tied_r128_b200": global_ckpt,
    "dense_tied_baseline_b200": dense_ckpt,
}
expected_eval_tasks = {
    "hellaswag": {"hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi"},
    "arc_easy": {"arc_easy", "arc_ar", "arc_de", "arc_ru", "arc_vi", "arc_zh"},
    "xnli": {"xnli_en", "xnli_vi", "xnli_zh", "xnli_de", "xnli_ru", "xnli_ar"},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for arm, checkpoint in arms.items():
        for seed in (42, 123, 456):
            stem = f"{task}_{arm}_seed{seed}"
            result_path = os.path.join(output_dir, stem + ".json")
            log_path = os.path.join(output_dir, stem + ".log")
            model_path = os.path.join(output_dir, "models", stem, "model_state.pt")
            for path in (result_path, log_path, model_path):
                assert os.path.isfile(path) and os.path.getsize(path) > 0, path
            with open(result_path, encoding="utf-8") as handle:
                result = json.load(handle)
            assert result["checkpoint"] == checkpoint, result["checkpoint"]
            assert result["task"] == task
            assert int(result["seed"]) == seed
            assert int(result["epochs"]) == 3
            assert set(result["eval_results"]) == expected_tasks
            assert math.isfinite(float(result["train_time_s"]))
            for eval_task, metrics in result["eval_results"].items():
                assert metrics.get("acc") is not None, (result_path, eval_task, metrics)
                assert math.isfinite(float(metrics["acc"])), (result_path, eval_task, metrics)
                if metrics.get("acc_norm") is not None:
                    assert math.isfinite(float(metrics["acc_norm"]))
            validated += 1
assert validated == 18, validated
summary = os.path.join(output_dir, "summary.md")
assert os.path.isfile(summary) and os.path.getsize(summary) > 0, summary
print("MATCHED_TIED_CONTROLS_FINETUNE_OK jobs=18 tasks=3 seeds=3")
PY

grep -lF 'Loaded compositional model: arm=lowrank' \
    "$TASK_FINETUNE_OUTPUT"/*_global_lowrank_tied_r128_b200_seed*.log >/dev/null
if grep -lF 'Loaded compositional model:' \
    "$TASK_FINETUNE_OUTPUT"/*_dense_tied_baseline_b200_seed*.log; then
    die 'a dense finetune job was incorrectly loaded as compositional'
fi
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|eval failed:|FAILED \(code|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' \
    "$TASK_FINETUNE_OUTPUT"/*.log; then
    die 'failure signature found in finetune logs'
fi
mapfile -t TASK_FINAL_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after finetune: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
cat "$TASK_FINETUNE_OUTPUT/summary.md"
echo 'TH2 MATCHED TIED CONTROL FINETUNE COMPLETE AND VERIFIED; ALL GPUS FREE'

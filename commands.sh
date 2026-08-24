#1 +60+a
#th2-check-shared-local-g16-10k-finetune-completion-20260824-2
#!/usr/bin/env bash
set -euo pipefail

TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_CHECKPOINT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_shared_local_tied_g16_10k_20260824"

echo '=== SharedLocal checkpoint-10000 finetune status ==='
date -u
hostname

echo '=== active finetune processes ==='
pgrep -af '[f]inetune/run_all.py|[f]inetune/train.py' || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader \
    || true

echo '=== current artifact counts ==='
test -d "$TASK_FINETUNE_OUTPUT"
echo "RESULT_JSON_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "JOB_LOG_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | wc -l)"
echo "MODEL_STATE_COUNT=$(find "$TASK_FINETUNE_OUTPUT/models" -mindepth 2 -maxdepth 2 -type f -name 'model_state.pt' 2>/dev/null | wc -l)"
if [[ -s "$TASK_FINETUNE_OUTPUT/summary.md" ]]; then
    stat -c 'SUMMARY_PRESENT modified=%y bytes=%s path=%n' "$TASK_FINETUNE_OUTPUT/summary.md"
else
    echo 'SUMMARY_PENDING'
fi

echo '=== per-job log tails ==='
while IFS= read -r log_path; do
    echo "--- $log_path ---"
    tail -20 "$log_path"
done < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' -print | sort)

echo '=== failure-signature scan ==='
TASK_FAILURE_FOUND=0
while IFS= read -r log_path; do
    if grep -niE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|eval failed:|FAILED \(code|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$log_path"; then
        TASK_FAILURE_FOUND=1
    fi
done < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' -print)
[[ "$TASK_FAILURE_FOUND" -eq 0 ]] || {
    echo 'ERROR: failure signature found in finetune logs'
    exit 1
}
echo 'NO FINETUNE FAILURE SIGNATURE FOUND'

if pgrep -f '[f]inetune/run_all.py|[f]inetune/train.py' >/dev/null; then
    echo 'TH2 SHARED LOCAL 10K FINETUNE STATUS: STILL RUNNING'
    exit 0
fi

echo '=== validate all nine completed jobs ==='
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" "$TASK_CHECKPOINT" <<'PY'
import json
import math
import os
import sys

output_dir, checkpoint = sys.argv[1:]
arm = 'shared_local_tied_g16'
expected_eval_tasks = {
    'hellaswag': {'hellaswag', 'hellaswag_ar', 'hellaswag_de', 'hellaswag_ru', 'hellaswag_vi'},
    'arc_easy': {'arc_easy', 'arc_ar', 'arc_de', 'arc_ru', 'arc_vi', 'arc_zh'},
    'xnli': {'xnli_en', 'xnli_vi', 'xnli_zh', 'xnli_de', 'xnli_ru', 'xnli_ar'},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for seed in (42, 123, 456):
        stem = f'{task}_{arm}_seed{seed}'
        result_path = os.path.join(output_dir, stem + '.json')
        log_path = os.path.join(output_dir, stem + '.log')
        model_path = os.path.join(output_dir, 'models', stem, 'model_state.pt')
        for path in (result_path, log_path, model_path):
            assert os.path.isfile(path) and os.path.getsize(path) > 0, path
        with open(result_path, encoding='utf-8') as handle:
            result = json.load(handle)
        assert result['checkpoint'] == checkpoint
        assert result['task'] == task
        assert int(result['seed']) == seed
        assert int(result['epochs']) == 3
        assert set(result['eval_results']) == expected_tasks
        assert math.isfinite(float(result['train_time_s']))
        for eval_task, metrics in result['eval_results'].items():
            assert metrics.get('acc') is not None, (result_path, eval_task, metrics)
            assert math.isfinite(float(metrics['acc'])), (result_path, eval_task, metrics)
            if metrics.get('acc_norm') is not None:
                assert math.isfinite(float(metrics['acc_norm']))
        validated += 1
assert validated == 9
summary = os.path.join(output_dir, 'summary.md')
assert os.path.isfile(summary) and os.path.getsize(summary) > 0
print('SHARED_LOCAL_10K_FINETUNE_RESULTS_OK jobs=9 tasks=3 seeds=3')
PY

mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] || {
    echo "ERROR: GPU processes remain after finetune: ${TASK_GPU_PIDS[*]}"
    exit 1
}
cat "$TASK_FINETUNE_OUTPUT/summary.md"
echo 'TH2 SHARED LOCAL G16 CHECKPOINT-10000 FINETUNE COMPLETE AND VERIFIED; ALL GPUS FREE'

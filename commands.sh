#1 +60+a
#th2-wait-verify-lowrank-tied-finetune-final
echo '=== th2 wait for low-rank tied finetune ==='
date -u
hostname
TASK_OUTPUT_DIR=/opt/dlami/nvme/sparse_emb_outputs/finetune_tied
TASK_ARM=lowrank_tied

for TASK_ATTEMPT in $(seq 1 90); do
    if ! pgrep -f '[f]inetune/run_all.py.*lowrank_tied' >/dev/null; then
        echo "launcher exited at attempt $TASK_ATTEMPT"
        break
    fi
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') still running (attempt $TASK_ATTEMPT/90)"
    sleep 60
done

echo '=== final processes ==='
pgrep -af '[f]inetune/run_all.py|[f]inetune/train.py' || echo 'none'
if pgrep -f '[f]inetune/run_all.py.*lowrank_tied' >/dev/null; then
    echo 'ERROR: timed out waiting for low-rank tied finetune'
    exit 1
fi

echo '=== final artifacts ==='
TASK_JSON_COUNT=$(find "$TASK_OUTPUT_DIR" -maxdepth 1 -type f -name "*_${TASK_ARM}_seed*.json" | wc -l)
TASK_LOG_COUNT=$(find "$TASK_OUTPUT_DIR" -maxdepth 1 -type f -name "*_${TASK_ARM}_seed*.log" | wc -l)
echo "arm_jsons=$TASK_JSON_COUNT"
echo "arm_logs=$TASK_LOG_COUNT"
test "$TASK_JSON_COUNT" -eq 9
test "$TASK_LOG_COUNT" -eq 9
test -s "$TASK_OUTPUT_DIR/summary.md"

python - "$TASK_OUTPUT_DIR" "$TASK_ARM" <<'PY'
import glob
import json
import math
import os
import sys

output_dir, arm = sys.argv[1:]
paths = sorted(glob.glob(os.path.join(output_dir, f"*_{arm}_seed*.json")))
expected_tasks = {"arc_easy", "hellaswag", "xnli"}
expected_seeds = {42, 123, 456}
seen = set()
for path in paths:
    with open(path) as handle:
        result = json.load(handle)
    assert result.get("arm_name") == arm, (path, result.get("arm_name"))
    task = result["task"]
    seed = int(result["seed"])
    assert task in expected_tasks, (path, task)
    assert seed in expected_seeds, (path, seed)
    assert result.get("eval_results"), (path, "empty eval_results")
    for metrics in result["eval_results"].values():
        for key in ("acc", "acc_norm"):
            value = metrics.get(key)
            if value is not None:
                assert math.isfinite(float(value)), (path, key, value)
    seen.add((task, seed))
assert seen == {(task, seed) for task in expected_tasks for seed in expected_seeds}, seen
print(f"validated_jsons={len(paths)} expected_task_seed_pairs={len(seen)}")
PY

echo '=== summary ==='
cat "$TASK_OUTPUT_DIR/summary.md"
echo '=== error scan ==='
if grep -HniE 'traceback|out of memory|nan|eval failed:|FAILED \(code' "$TASK_OUTPUT_DIR"/*_${TASK_ARM}_seed*.log; then
    echo 'ERROR: failure signature found'
    exit 1
else
    echo 'no error signatures'
fi
echo '=== final GPU compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo 'ERROR: GPU compute processes remain'
    exit 1
fi
echo 'TH2 LOWRANK TIED FINETUNE FINAL VERIFIED'

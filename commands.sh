#1 +60+a
#th2-verify-lowrank-tied-eval-10k
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

echo '=== th2 verify low-rank tied checkpoint-10000 evaluation ==='
date -u
hostname
echo "conda=$CONDA_DEFAULT_ENV"

TASK_CHECKPOINT=/opt/dlami/nvme/sparse_emb_outputs/lowrank_tied/checkpoint-10000
TASK_LAUNCH_LOG=/opt/dlami/nvme/sparse_emb_outputs/lowrank_tied/eval_parallel_10k.log

echo '=== evaluation processes ==='
if pgrep -af '[e]val_parallel.py|[e]val_checkpoint.py'; then
    echo 'EVALUATION STILL RUNNING'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    exit 0
fi
echo 'none'
echo '=== GPU compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo '=== result files ==='
ls -lh "$TASK_CHECKPOINT/eval.log" \
    "$TASK_CHECKPOINT/eval_ppl.json" \
    "$TASK_CHECKPOINT/eval_benchmarks.json" \
    "$TASK_LAUNCH_LOG" 2>/dev/null || true

python -c "import json, math; p=json.load(open('$TASK_CHECKPOINT/eval_ppl.json')); b=json.load(open('$TASK_CHECKPOINT/eval_benchmarks.json')); expected={'en','vi','zh','ru','de','ar'}; assert expected <= set(p), (expected-set(p)); assert all(math.isfinite(p[x]['loss']) and math.isfinite(p[x]['perplexity']) and p[x]['num_tokens'] > 0 for x in expected); assert len(b) == 26, len(b); print(f'JSON VALID: 6 PPL languages, {len(b)} benchmark tasks')"
grep -F 'Loaded compositional model: arm=lowrank' "$TASK_CHECKPOINT/eval.log"

echo '=== launcher tail ==='
tail -30 "$TASK_LAUNCH_LOG"
echo '=== eval error scan ==='
grep -Ei 'traceback|out of memory|nan|failed \(code|error:' "$TASK_CHECKPOINT/eval.log" || echo 'no error signatures'
echo 'TH2 LOWRANK TIED EVAL VERIFIED'

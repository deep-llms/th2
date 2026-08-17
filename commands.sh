#1 +60+a
#th2-check-lowrank-tied-finetune-2
echo '=== th2 low-rank tied finetune completion check ==='
date -u
hostname
TASK_OUTPUT_DIR=/opt/dlami/nvme/sparse_emb_outputs/finetune_tied

echo '=== finetune processes ==='
pgrep -af '[f]inetune/run_all.py|[f]inetune/train.py' || echo 'none'
echo '=== GPU compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
echo '=== artifacts so far ==='
echo "logs=$(find "$TASK_OUTPUT_DIR" -maxdepth 1 -type f -name '*.log' 2>/dev/null | wc -l)"
echo "jsons=$(find "$TASK_OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
if [ -f "$TASK_OUTPUT_DIR/summary.md" ]; then
    echo 'summary.md: present'
    cat "$TASK_OUTPUT_DIR/summary.md"
else
    echo 'summary.md: not present'
fi
ls -lh "$TASK_OUTPUT_DIR"/*.log 2>/dev/null | head -12 || true
echo '=== error scan ==='
grep -HniE 'traceback|out of memory|nan|eval failed:|FAILED \(code' "$TASK_OUTPUT_DIR"/*.log 2>/dev/null || echo 'no error signatures'
echo '=== latest log tails ==='
for TASK_LOG in $(ls -t "$TASK_OUTPUT_DIR"/*.log 2>/dev/null | head -3); do
    echo "--- $TASK_LOG"
    tail -8 "$TASK_LOG"
done
echo 'TH2 FINETUNE HEALTH CHECK DONE'

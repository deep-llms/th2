#1 +30+a
#th2-swt-inspect-authorized-cleanup-20260908-a01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name --format=csv
ps -eo pid,ppid,args | grep -E '[t]rain.py|[t]rain_capacity_b200|[b]enchmark_capacity|[b]enchmark_batches|[p]repare_capacity_cache' || true
for task_path in /mnt/local/_outputs/deep-llms_th2/swt /mnt/local/_data/deep-llms_th2/swt /mnt/local/.cache/huggingface/datasets; do
    if [ -e "$task_path" ]; then
        ls -ld "$task_path"
        find "$task_path" -mindepth 1 -maxdepth 2 -type d -print
    fi
done
for task_path in /mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02 /mnt/local/_outputs/deep-llms_th2/swt/batch_benchmark_20260908_a01 /mnt/local/_data/deep-llms_th2/swt/qwen_en_map160_batch1000; do
    if [ -e "$task_path" ]; then
        realpath -e "$task_path"
        du -sh "$task_path"
        find "$task_path" -maxdepth 1 -type f -printf '%f\n' | head -30
    fi
done
echo SWT_CLEANUP_INSPECTION_COMPLETE

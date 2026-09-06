#1 +60+a
#th2-readonly-investigate-burn-lifetime-20260906-a01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
tmux list-panes -a -F '#{session_name} pane=#{pane_id} pid=#{pane_pid} dead=#{pane_dead} exit=#{pane_dead_status} command=#{pane_current_command}' || true
ps -eo pid,ppid,sid,pgid,etime,comm | grep -E 'python|tmux|bash' || true
tail -n 8 /mnt/local/_outputs/@PROJECT@/logs/gpu_burn_ranklift_restore_20260906_a01.log
sha256sum /tmp/llm_pretrain_burn.py resources/llm_pretrain_burn.py
echo '=== original training pane tail if retained ==='
tmux list-panes -a -F '#{session_name} #{pane_id}' | while read -r session pane; do
    case "$session" in
        *train-raw-tiered-and-unified*) tmux capture-pane -p -t "$pane" -S -100 ;;
    esac
done
echo 'TH2 BURN LIFETIME READONLY DIAGNOSTIC COMPLETE'

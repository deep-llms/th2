#1 +120+a
#th2-kill-gpu-procs
pkill -f dummy.py 2>/dev/null || true
sleep 5
# Kill GPU-using python processes by PID from nvidia-smi
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "force-killed $pid"
done
sleep 5
nvidia-smi | grep -E "MiB /|No running"
echo TH2 CLEARED

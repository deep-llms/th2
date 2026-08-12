#1 +120+a
#th2-cancel-and-verify
for i in 1 2 3; do
    pkill -f run_experiments.py 2>/dev/null
    pkill -f train_compositional.py 2>/dev/null
    pkill -f "accelerate launch" 2>/dev/null
    sleep 5
done
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 5
nvidia-smi | grep -E "MiB /|No running"
echo TH2 CANCELLED

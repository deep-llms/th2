#1 +120+a
#th2-kill-all
for i in 1 2 3 4 5; do
    pkill -f dummy.py 2>/dev/null
    pkill -f "python3" 2>/dev/null
    sleep 5
done
nvidia-smi | grep -E "MiB /|No running"
echo TH2 CLEARED

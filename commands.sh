#1 +120+a
#th2-kill-dummy
for i in 1 2 3 4 5; do
    pkill -f dummy.py 2>/dev/null
    sleep 5
done
nvidia-smi | grep -E "MiB /" | head -8
echo TH2 DUMMY KILLED

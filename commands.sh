#1 +120+a
#kill-dummy
for i in 1 2 3 4 5; do
    echo "attempt $i"
    pkill -f dummy.py 2>/dev/null
    sleep 5
done
nvidia-smi
echo "done"

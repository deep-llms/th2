#1 +120+a
#kill-eval
for i in 1 2 3 4 5; do
    echo "attempt $i"
    pkill -f eval_parallel.py 2>/dev/null
    pkill -f eval_checkpoint.py 2>/dev/null
    sleep 5
done
nvidia-smi
echo "done"

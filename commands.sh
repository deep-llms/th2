#1
#kill-eval
for i in 1 2 3 4 5; do
    echo "attempt $i"
    pkill -f eval_checkpoint.py 2>/dev/null
    sleep 5
done
echo "done"

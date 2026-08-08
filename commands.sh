#1 +120+a
#kill-all-training
for i in 1 2 3 4 5; do
    echo "attempt $i"
    pkill -f run_experiments.py 2>/dev/null
    pkill -f train_original_ant.py 2>/dev/null
    pkill -f train_compositional.py 2>/dev/null
    pkill -f "accelerate launch" 2>/dev/null
    sleep 5
done
nvidia-smi
echo "done"

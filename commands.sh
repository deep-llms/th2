#1 +120+a
#th2-eval-lowrank-tied-10k
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

echo '=== th2 low-rank tied checkpoint-10000 eval preflight ==='
date -u
hostname
echo "conda=$CONDA_DEFAULT_ENV"
python --version
nvidia-smi

TASK_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [ -n "$TASK_GPU_PROCESSES" ]; then
    echo 'ERROR: GPU compute processes are active; refusing to start evaluation.'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    exit 1
fi

TASK_CHECKPOINT=/opt/dlami/nvme/sparse_emb_outputs/lowrank_tied/checkpoint-10000
TASK_LAUNCH_LOG=/opt/dlami/nvme/sparse_emb_outputs/lowrank_tied/eval_parallel_10k.log
test -d "$TASK_CHECKPOINT"
test -f "$TASK_CHECKPOINT/config.json"
test -f "$TASK_CHECKPOINT/embedding.pt"
test -f "$TASK_CHECKPOINT/trainer_state.json"

for TASK_RESULT in \
    "$TASK_CHECKPOINT/eval.log" \
    "$TASK_CHECKPOINT/eval_ppl.json" \
    "$TASK_CHECKPOINT/eval_benchmarks.json" \
    "$TASK_LAUNCH_LOG"; do
    if [ -e "$TASK_RESULT" ]; then
        echo "ERROR: existing evaluation artifact found; refusing to overwrite: $TASK_RESULT"
        exit 1
    fi
done

echo '=== launching full PPL + benchmark evaluation ==='
python eval/eval_parallel.py \
    --checkpoints "$TASK_CHECKPOINT" \
    --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_LAUNCH_LOG"

test -f "$TASK_CHECKPOINT/eval_ppl.json"
test -f "$TASK_CHECKPOINT/eval_benchmarks.json"
grep -F 'Loaded compositional model: arm=lowrank' "$TASK_CHECKPOINT/eval.log"
echo 'TH2 LOWRANK TIED EVAL DONE'

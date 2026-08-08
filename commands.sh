#1 +120+a
#cancel-and-clean-prep
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate sparse_emb
sleep 3

echo '=== 1. kill all training ==='
for i in 1 2 3 4 5; do
    echo "attempt $i"
    pkill -f run_experiments.py 2>/dev/null
    pkill -f train_original_ant.py 2>/dev/null
    pkill -f train_compositional.py 2>/dev/null
    pkill -f "accelerate launch" 2>/dev/null
    sleep 5
done

echo '=== 2. state before cleanup ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/

echo '=== 3. delete old run output folders ==='
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant
rm -rf /opt/dlami/nvme/sparse_emb_outputs/v2_attn
rm -rf /opt/dlami/nvme/sparse_emb_outputs/original_ant_old16x
rm -rf /opt/dlami/nvme/sparse_emb_outputs/v2_attn_old16x
rm -rf /opt/dlami/nvme/sparse_emb_outputs/smoke_orig
rm -rf /opt/dlami/nvme/sparse_emb_outputs/logs

echo '=== 4. remove all caches ==='
rm -rf ~/.cache/huggingface/datasets
find /opt/dlami/nvme/sparse_emb_data -name "cache-*" -delete 2>/dev/null
find /opt/dlami/nvme/sparse_emb_data -name "tmp*" -delete 2>/dev/null
echo "caches removed"

echo '=== 5. GPUs empty? ==='
nvidia-smi

echo '=== 6. copy accelerate config ==='
mkdir -p ~/.cache/huggingface/accelerate
cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
cat ~/.cache/huggingface/accelerate/default_config.yaml

echo '=== 7. final state ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/
echo "PREP DONE"

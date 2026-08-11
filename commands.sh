#1
#th2-xling-1
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

python -c "import scipy" 2>/dev/null || pip install -q scipy

O=/opt/dlami/nvme/sparse_emb_outputs
mkdir -p $O/crosslingual
CUDA_VISIBLE_DEVICES=0 python crosslingual/run_crosslingual.py \
    --checkpoints $O/original_ant/checkpoint-10000 \
    --tests t6 t8 probe_b --mexa-sentences 500 \
    --output-dir $O/crosslingual/original_ant \
    > $O/crosslingual/original_ant.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python crosslingual/run_crosslingual.py \
    --checkpoints $O/v2_attn/checkpoint-10000 \
    --tests t6 t8 probe_b --mexa-sentences 500 \
    --output-dir $O/crosslingual/v2_attn \
    > $O/crosslingual/v2_attn.log 2>&1 &
wait

echo '=== original_ant log tail ==='
tail -20 $O/crosslingual/original_ant.log
echo '=== v2_attn log tail ==='
tail -20 $O/crosslingual/v2_attn.log
ls -la $O/crosslingual/*/
echo TH2 XLING DONE

#1 +120+a
#th2-final-verify
echo '=== FM status ==='
sudo systemctl status nvidia-fabricmanager 2>&1 | head -5
echo '=== GPU ==='
nvidia-smi | grep -E "MiB /|No running"
echo '=== CUDA sparse_emb ==='
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate sparse_emb
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count()); x=torch.randn(100,100,device='cuda'); print('ok')"
echo '=== CUDA eval ==='
conda activate eval
python -c "import torch; print('cuda:', torch.cuda.is_available()); x=torch.randn(100,100,device='cuda'); print('ok')"
echo '=== outputs ==='
ls -d /opt/dlami/nvme/sparse_emb_outputs/*/
echo '=== lowrank ckpt-10000 exists ==='
ls /opt/dlami/nvme/sparse_emb_outputs/lowrank/checkpoint-10000/embedding.pt 2>/dev/null && echo YES || echo NO
echo TH2 FINAL OK

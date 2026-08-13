#1 +120+a
#th2-check-new-machine
echo '=== OS ==='
cat /etc/os-release | head -5
uname -a
echo '=== hostname ==='
hostname
echo '=== GPUs ==='
nvidia-smi | head -20
echo '=== disk ==='
df -h | grep -E "nvme|opt|home|Filesystem"
echo '=== conda ==='
which conda 2>/dev/null || echo "no conda"
ls $HOME/miniconda3/envs/ 2>/dev/null || echo "no miniconda3"
echo '=== project dir ==='
ls -la /opt/dlami/nvme/ 2>/dev/null || echo "no /opt/dlami/nvme"
ls -la $HOME/ | head -15
echo '=== data ==='
ls -la /opt/dlami/nvme/sparse_emb_data/ 2>/dev/null || echo "no data dir"
echo '=== outputs ==='
ls -la /opt/dlami/nvme/sparse_emb_outputs/ 2>/dev/null || echo "no outputs dir"
echo '=== python ==='
python3 --version 2>/dev/null
pip3 list 2>/dev/null | grep -E "torch|transformers|accelerate|datasets" | head -5
echo TH2 MACHINE CHECK DONE

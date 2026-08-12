#1 +120+a
#th2-gpu-check
nvidia-smi
pgrep -af "python|accelerate|train" | head -5 || echo "no processes"
ls -la /opt/dlami/nvme/sparse_emb_outputs/ | head -15

#1 +30+a
#th2-ohmv-runner-liveness-check-20261123-a01
set -u
date -u
hostname
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
ls -la --time-style=+%FT%T /mnt/local/_outputs/deep-llms_th2/ | tail -8
ls -la --time-style=+%FT%T /mnt/local/_outputs/deep-llms_th2/anchor_static_20261123_a01 2>&1 | head -12

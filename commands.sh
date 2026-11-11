#1 +30+a
#th2-wc69-dot-pool-prelaunch-health-20261112-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.dot_pool_health
if test -x /mnt/local/conda-py311/envs/dot_pool/bin/python3.11; then
  /mnt/local/conda-py311/envs/dot_pool/bin/python3.11 -c 'import sys,torch,transformers; from importlib.metadata import version; from flash_attn.cute.interface import flash_attn_varlen_func; print("DOT_RUNTIME",sys.executable,torch.__version__,transformers.__version__,version("flash-attn-4"),hasattr(torch.nn.functional,"grouped_mm"))'
fi

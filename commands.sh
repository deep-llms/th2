#1 +60+a
#th2-78gg-flash-attention-env-check-20260929-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json, socket, sys
from datetime import datetime, timezone
import torch
packages={}
for name in ['torch','transformers','accelerate','flash-attn','flash-attn-3','flash-attn-4','triton','nvidia-cudnn-cu13','nvidia-cudnn-cu12']:
 try:packages[name]=metadata.version(name)
 except metadata.PackageNotFoundError:packages[name]=None
relevant=sorted((dist.metadata['Name'],dist.version) for dist in metadata.distributions()
                if any(word in dist.metadata['Name'].lower() for word in ['flash','cutlass','cudnn']))
result={'at':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'python':sys.executable,
        'packages':packages,'related_distributions':relevant,
        'torch_runtime_version':torch.__version__,'torch_cuda_version':torch.version.cuda,
        'flash_attn_module_present':importlib.util.find_spec('flash_attn') is not None,
        'pytorch_flash_attention_built':torch.backends.cuda.is_flash_attention_available(),
        'pytorch_flash_sdp_enabled':torch.backends.cuda.flash_sdp_enabled(),
        'pytorch_mem_efficient_sdp_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),
        'pytorch_cudnn_sdp_enabled':torch.backends.cuda.cudnn_sdp_enabled(),
        'cuda_context_initialized':torch.cuda.is_initialized()}
print('FLASH_ENV_CHECK',json.dumps(result,indent=2),flush=True)
assert not result['cuda_context_initialized']
PY

#1 +30+a
#th2-q359-ce7454-env-verify-20261016-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/ce7454_env/bin/python
"$TASK_PYTHON" -m pip check
CUDA_VISIBLE_DEVICES='' "$TASK_PYTHON" - <<'VERIFY'
import json, sys, torch, torchvision, numpy, scipy, PIL, cv2
from torchvision.models import resnet18
m=resnet18(weights=None,num_classes=3).eval()
torch.set_num_threads(2)
with torch.no_grad(): y=m(torch.zeros(1,3,64,64))
assert y.shape==(1,3) and torch.isfinite(y).all()
print('CE7454_ENV_CPU_VERIFIED',json.dumps({'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'opencv':cv2.__version__,'cuda_build':torch.version.cuda,'architectures':torch.cuda.get_arch_list()}))
VERIFY

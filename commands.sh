#1 +30+a
#th2-6y3o-fa2-torch28-verify-20261020-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python3 - <<'CHECK'
import json
from pathlib import Path
p=Path('/mnt/local/_env_status/train_fa2_t28.status.json')
if p.is_file():
    obj=json.loads(p.read_text())
    print('ENV_INSTALL_STATUS '+json.dumps(obj),flush=True)
else: print('ENV_INSTALL_STATUS_FILE_MISSING',flush=True)
CHECK
python3 scripts/attention_env_inventory.py --host thiennh-p6-6y3o-worker-0 --compare /mnt/local/_outputs/deep-llms_th2/6y3o-attention-study-20261020-a01/inventory-before.json --output /mnt/local/_outputs/deep-llms_th2/6y3o-attention-study-20261020-a01/preflight.json

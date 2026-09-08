#1 +60+a
#th2-swt-stop-verified-queue-preserve-checkpoints-20260908-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
test "$PWD" = /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
python - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('scripts/stop_capacity_queue.py').read_bytes()).hexdigest() == 'd87e5396b728b4e6439a6992c078124c96142c20f1fa5c75b2bf3ee7b68f5397'
PY
python -u -m scripts.stop_capacity_queue --queue-pid 109572 --queue-start 161614436 \
  --run-root /mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02 --stop
python - <<'PY'
from pathlib import Path
from train import validate_resume_checkpoint
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/qwen6_allarms_10k_s42_20260908_a02/B0')
checkpoints = sorted(root.glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]), reverse=True)
for checkpoint in checkpoints:
    try:
        state = validate_resume_checkpoint(checkpoint, 8)
    except (ValueError, OSError) as exc:
        print('RETAINED_INCOMPLETE_CHECKPOINT', checkpoint, str(exc), flush=True)
        continue
    print('LATEST_COMPLETE_RETAINED_CHECKPOINT', checkpoint, state['global_step'], flush=True)
    break
else:
    raise RuntimeError('No complete retained checkpoint')
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo SWT_AUTHORIZED_STOP_DONE_NO_CLEANUP

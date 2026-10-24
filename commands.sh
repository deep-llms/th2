#1 +60+a
#th2-6y3o-dot-stop-burn-recover-20261025-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,time
from pathlib import Path
from scripts.gpu_status import require_free
from scripts.gpu_process import inspect
from scripts.bdot_stop_launch import snapshot
from scripts.train_then_burn import start_burn,release_guard_if_safe,GUARD_HASH
from scripts.native_baseline_pipeline import burn_gate
from run_experiments import write_json,now
old=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-long-20261024-a01')
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-recovery-20261025-a01')
read=lambda p:json.loads(p.read_text())
guard=Path('/mnt/local/_gpu_guard/DISABLED')
marker=guard.read_text();inode=guard.stat().st_ino;owner=json.loads(marker)
assert owner['owner_pid']==519238 and owner['output']==str(old/'supervised')
assert not Path('/proc',str(owner['owner_pid'])).exists(),'Old supervisor still exists; do not compete'
assert hashlib.sha256((guard.parent/'gpu_guard.sh').read_bytes()).hexdigest()==GUARD_HASH
state=read(old/'supervised/supervisor.json')
assert state.get('finished_at') and state.get('handoff_error')=="OSError(22, 'Invalid argument')"
assert not (old/'supervised/burn.log').exists()
before=inspect();require_free(list(range(8)))
original=old/'source'
for rel,h in read(original/'SOURCE_MANIFEST.json')['files'].items():
 assert hashlib.sha256((original/rel).read_bytes()).hexdigest()==h,rel
root.mkdir(exist_ok=False)
snapshot(original,root/'source',('scripts/train_then_burn.py','tests/test_train_then_burn.py'))
out=root/'supervised';out.mkdir()
write_json(root/'recovery.json',dict(predecessor=str(old),original_supervisor=state,stale_guard=owner,before=before))
time.sleep(30)
free=require_free(list(range(8)))
assert guard.stat().st_ino==inode and guard.read_text()==marker
receipt=dict(started_at=now(),training_status='failed',training_error='Predecessor explicitly cancelled by user; burn recovery only',free_after_training=free)
write_json(out/'supervisor.json',receipt)
proof=start_burn(out,'dot_stop_recovery_20261025_a01_burn',root/'source')
receipt.update(burn=proof,finished_at=now())
write_json(out/'supervisor.json',receipt)
assert release_guard_if_safe(guard,inode,marker,out)
burn_gate(root)
print('RECOVERED_BURN_HANDOFF '+json.dumps(receipt),flush=True)
REMOTE

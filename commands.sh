#1 +30+a
#th2-wc69-dot-pool-report-recovery-20261112-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
TASK_PYTHON=/mnt/local/conda-py311/envs/dot_pool/bin/python3.11
"$TASK_PYTHON" - <<'PYFREEZE'
from pathlib import Path
import hashlib,json,shutil
source=Path.cwd();target=Path('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/report-recovery-a01-source')
assert not target.exists()
expected=json.loads((source/'resources/dot_pool_report_recovery_20261112.json').read_text())
for rel,digest in expected['recovery_files'].items():assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==digest
for folder in ['scripts','efficient_llm']:
 for f in sorted((source/folder).rglob('*.py')):
  dest=target/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
shutil.copyfile(source/'resources/dot_pool_report_recovery_20261112.json',target/'expected.json')
for rel,digest in expected['recovery_files'].items():assert hashlib.sha256((target/rel).read_bytes()).hexdigest()==digest
print('CPU_RECOVERY_SOURCE_READY',str(target),flush=True)
PYFREEZE
cd /mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/report-recovery-a01-source
export PYTHONPATH="$PWD"
"$TASK_PYTHON" -u -m scripts.dot_pool_report_recovery --root /mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/run-a03 --output /mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/report-recovery-a01 --expected expected.json

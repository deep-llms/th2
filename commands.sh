#1 +30+a
#th2-6y3o-a2-stage1-downstream-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYRUN'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,sys
source='13686edd9cb8d7914b0f2c44c3c362c435a808f6'
stage=Path('/mnt/local/deep-llms_th2/resources/a2_stage1_terminal_20261029.json')
assert hashlib.sha256(stage.read_bytes()).hexdigest()=='7a815ea063b0b81ba72a81c327384c47a53c1cb0925cad6b28cd4576a96f75fd'
spent=1800+(datetime.now(timezone.utc)-datetime.fromisoformat('2026-10-29T01:21:49.359472+00:00')).total_seconds()
from scripts.launch_a2_stage1_downstream import budget
print('STAGE1_ADMISSION_BUDGET',json.dumps(budget(spent,0,7200,'2026-10-31T04:06:27.182436+08:00')),flush=True)
argv=[sys.executable,'-u','-m','scripts.launch_a2_stage1_downstream',
 '--stage',str(stage),'--stage-sha256','7a815ea063b0b81ba72a81c327384c47a53c1cb0925cad6b28cd4576a96f75fd',
 '--assets','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/assets.json',
 '--assets-sha256','5f3f42b76d1f5be52e95a4a12e736c7aa81fbd121ed2c6daa730078625dc4371',
 '--output','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-downstream-20261029-a01',
 '--predecessor-burn-script','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-heals-20261029-a01/source/resources/llm_pretrain_burn.py',
 '--eval-python','/mnt/local/conda-py311/envs/eval/bin/python3.11',
 '--tokenizer','/mnt/local/_models/deep-llms_th2/a2/dclm/P300',
 '--dataset-root','/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023',
 '--host','thiennh-p6-6y3o-worker-0','--session','a2_stage1_downstream_20261029_a01',
 '--code-commit',source,'--deadline','2026-10-31T04:06:27.182436+08:00',
 '--spent-seconds',str(spent),'--timeout-seconds','7200',
 '--prepared','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-downstream-20261028-a01/supervised/run/downstream/prepared.json',
 '--prepared-sha256','85cf7f2119e35e81a79aea48f58080bdc9ba5371b27accbda0eac52aa6770bbf']
subprocess.run(argv,check=True)
PYRUN

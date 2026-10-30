#1 +30+a
#th2-6y3o-a2-stage1b-fwe-regression-20261030-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYRUN'
from datetime import datetime,timezone
import subprocess,sys,json
from scripts.launch_a2_stage1b_regression import budget
spent=3299.9433463821188+(datetime.now(timezone.utc)-datetime.fromisoformat('2026-10-29T20:19:10.047762+00:00')).total_seconds()
print('REGRESSION_BUDGET',json.dumps(budget(spent,0,1800,'2026-10-31T04:06:27.182436+08:00')),flush=True)
subprocess.run([sys.executable,'-u','-m','scripts.launch_a2_stage1b_regression',
 '--stage','/mnt/local/deep-llms_th2/resources/a2_stage1b_part1_terminal_20261030.json','--stage-sha256','01ad2695a967ac09fde27bdbfd6efc909284427384cd8994877e7666ae74f222',
 '--reference','/mnt/local/deep-llms_th2/resources/a2_stage1b_regression_reference_20261030.json','--reference-sha256','b96a6d60b796fbc71f2bfe42132747480e57bc43407efd3e5045a534978e0187',
 '--assets','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/assets.json',
 '--assets-sha256','5f3f42b76d1f5be52e95a4a12e736c7aa81fbd121ed2c6daa730078625dc4371',
 '--output','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-regression-20261030-a01',
 '--predecessor-burn-script','/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-part1-20261030-a01/source/resources/llm_pretrain_burn.py',
 '--eval-python','/mnt/local/conda-py311/envs/eval/bin/python3.11',
 '--tokenizer','/mnt/local/_models/deep-llms_th2/a2/dclm/P300',
 '--dataset-root','/mnt/local/_data/deep-llms_th2/benchmarks_zero_20261023',
 '--host','thiennh-p6-6y3o-worker-0','--session','a2_stage1b_fwe_regression_20261030_a01',
 '--code-commit','8b78d9e1051407a45be6676820bc57ac87a872e4','--deadline','2026-10-31T04:06:27.182436+08:00',
 '--spent-seconds',str(spent),'--timeout-seconds','1800'],check=True)
PYRUN

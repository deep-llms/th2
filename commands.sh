#1 +60+a
#th2-q359-shared-work-verify-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_REVIEW_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-shared-work-verify-20261014-a01
test ! -e "$TASK_REVIEW_ROOT"
mkdir "$TASK_REVIEW_ROOT"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u -m unittest  tests.test_proxy_shared_work  tests.test_proxy_screen24.Screen24Tests.test_consumer_objective_against_copied_frozen_decoder  tests.test_proxy_screen24.Screen24Tests.test_consumer_gradient_does_not_cross_documents  -v > "$TASK_REVIEW_ROOT/cpu-tests.log" 2>&1 || { cat "$TASK_REVIEW_ROOT/cpu-tests.log"; exit 1; }
cat "$TASK_REVIEW_ROOT/cpu-tests.log"
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - "$TASK_REVIEW_ROOT" <<'PYREMOTE'
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1])
receipt=dict(status='passed',scope='CPU only; no GPU workload or process control',
 timestamp=datetime.now(timezone.utc).isoformat(),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 for p in ('deep_kv/proxy.py','deep_kv/proxy_estimators.py','deep_kv/training.py','tests/test_proxy_shared_work.py')})
(root/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
base=Path('/mnt/local/_outputs/deep-llms_th2')
for name in ('q359-proxy-12layer-screen22-20261013-a01','q359-proxy-followup11-20261014-a01'):
 p=base/name/'supervised/run/run.json'
 if p.exists():
  data=json.loads(p.read_text());print('QUEUE',name,data['status'],[j['name'] for j in data['jobs'] if j['status']=='running'],flush=True)
 else:
  p=base/name/'supervisor.log';print('WAITER',name,p.read_text().splitlines()[-1] if p.exists() else 'no log',flush=True)
PYREMOTE

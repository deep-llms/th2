#!/usr/bin/env bash
# Short synthetic runtime check with the existing verified burn handoff.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=$2
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" "$TASK_HOST" <<'PY'
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
from scripts.verified_gpu_reclaim import inspect, process
from scripts.train_then_burn import APPROVED_BURNS, BURN_HASH, GUARD_HASH, digest, approved_launcher

root = Path(sys.argv[1])
record = inspect()
assert record['host'] == sys.argv[2] and not record['guard_disabled']
assert len(record['gpus']) == 8 and all('B200' in gpu['name'] for gpu in record['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh') == GUARD_HASH
approved = {**APPROVED_BURNS, str(Path.cwd() / 'resources/llm_pretrain_burn.py'): BURN_HASH}
for pid in record['workers']:
    assert approved_launcher(pid, approved) or approved_launcher(process(pid)['ppid'], approved), pid
source = Path('resources/accelerate_config.yaml')
destination = Path(default_yaml_config_file)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    shutil.copy2(destination, root / 'accelerate.previous.yaml')
shutil.copy2(source, destination)
assert source.read_bytes() == destination.read_bytes()
config = load_config_from_file(str(destination)).to_dict()
assert config['num_processes'] == 8 and config['mixed_precision'] == 'bf16'
assert config['distributed_type'] == 'MULTI_GPU'
subprocess.run(['accelerate', 'env'], check=True)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
jobs = {'jobs': [dict(name='fa4-runtime-smoke', gpus=list(range(8)), timeout_seconds=1200,
    argv=[sys.executable, '-m', 'accelerate.commands.launch', '--config_file', str(destination),
          '--main_process_port', str(port), '--module', 'scripts.smoke_fa4_distributed',
          '--output', '{run_dir}/smoke.json'],
    required_outputs=[dict(path='smoke.json', json_equals={'status':'passed','world_size':8})])]}
for name, value in [('inspection.json', record), ('jobs.json', jobs)]:
    with (root / name).open('x') as handle:
        json.dump(value, handle, indent=2)
print('FA4_SMOKE_PREFLIGHT_PASSED', flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" \
  --output "$TASK_ROOT/supervised" --inspection "$TASK_ROOT/inspection.json" \
  --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

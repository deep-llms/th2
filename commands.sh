#1 +300+a
#th2-tjx3-packed-kernels-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PYTHON=/mnt/local/conda-py311/envs/train_env/bin/python
"$TASK_PYTHON" -u - <<'PY'
import json, os, shutil, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
from scripts.train_then_burn import (APPROVED_BURNS, BURN_HASH, GUARD_HASH, digest,
    stop_burns, start_burn, enable_subreaper, clean_owned_children, approved_launcher)
from scripts.verified_gpu_reclaim import inspect
from scripts.gpu_status import require_free
from run_experiments import now, write_json
from accelerate.commands.config.config_args import default_yaml_config_file
assert socket.gethostname() == 'thiennh-p6-tjx3-worker-0'
project=Path.cwd()
root=Path('/mnt/local/_outputs/deep-llms_th2/packed-kernels-20261004-a01')
root.mkdir(exist_ok=False)
assert shutil.which('tmux')
assert digest(project/'resources/llm_pretrain_burn.py') == BURN_HASH
record=inspect();write_json(root/'inspection.json',record)
assert all('B200' in g['name'] for g in record['gpus'])
assert not record['guard_disabled'], 'A workload already holds the GPU guard'
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
for pid in record['workers']:
    parent=record['processes'][str(pid)]['ppid']
    assert approved_launcher(pid,approved) or (parent>1 and approved_launcher(parent,approved)), 'Unrecognized burn'
bench_python='/mnt/local/conda-py311/envs/attention_bench/bin/python'
subprocess.run([bench_python,'-c',"import importlib.metadata as m; import flash_attn.cute.interface; print({p:m.version(p) for p in ['torch','flash-attn-4','nvidia-cutlass-dsl']})"],check=True,timeout=180)
subprocess.run([bench_python,'-m','pip','check'],check=True,timeout=120)
subprocess.run([bench_python,'-m','unittest','tests.test_packed_attention','-v'],check=True,timeout=180)
print('BURN_IDENTITIES_AND_SOURCE_VERIFIED',record['workers'],flush=True)
source=project/'resources/accelerate_config.yaml';cached=Path(default_yaml_config_file)
shutil.copy2(source,cached)
assert cached.read_bytes()==source.read_bytes(), 'Accelerate configuration changed'
print('ACCELERATE_CONFIG_VERIFIED',str(cached),flush=True)
enable_subreaper()
guard=Path('/mnt/local/_gpu_guard/DISABLED');owned=None
if guard.parent.exists():
    assert digest(guard.parent/'gpu_guard.sh')==GUARD_HASH, 'Unknown guard protocol'
    payload=json.dumps({'owner_pid':os.getpid(),'output':str(root),'at':now()})
    with guard.open('x') as f:f.write(payload)
    owned=guard.stat().st_ino
    time.sleep(30)
receipt={'started_at':now(),'tests':[]};attempted=False
def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Test supervisor received signal {signum}')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
try:
    attempted=True
    stop_burns(root,approved)
    require_free(list(range(8)))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','DOCUMENT_TEST_DEVICE':'cuda:0'}
    subprocess.run([sys.executable,'-m','accelerate.commands.accelerate_cli','env'],env=env,check=True,timeout=120)
    bench_python='/mnt/local/conda-py311/envs/attention_bench/bin/python'
    result=subprocess.run([bench_python,'-m','scripts.benchmark_packed_attention',
        '--output',str(root/'kernels.json')],env=env,timeout=1200)
    receipt['benchmark_returncode']=result.returncode
    receipt['passed']=result.returncode==0 and (root/'kernels.json').is_file()

except BaseException as error:
    receipt.update(passed=False,error=repr(error));traceback.print_exc()
finally:
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if attempted:
        try:
            clean_owned_children()
            time.sleep(30)
            require_free(list(range(8)))
            receipt['burn']=start_burn(root,'tjx3-packed-kernels-a01-burn',project)
            print('AUTOMATIC_BURN_RESTORED_AND_VERIFIED',flush=True)
        except BaseException as error:
            receipt['handoff_error']=repr(error);traceback.print_exc()
    if owned is not None and 'burn' in receipt and guard.exists() and guard.stat().st_ino==owned and guard.read_text()==payload:
        guard.unlink()
    receipt['finished_at']=now();write_json(root/'result.json',receipt)
print('PACKED_KERNEL_RESULT',json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt.get('passed') and 'burn' in receipt else 1)
PY

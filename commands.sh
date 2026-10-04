#1 +300+a
#th2-tjx3-document-stability-200-20261004-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
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
root=Path('/mnt/local/_outputs/deep-llms_th2/document-stability-200-20261004-a01')
root.mkdir(exist_ok=False)
work=root/'benchmark'
def run_stage(name, argv, env, timeout):
    print('STAGE_START',name,now(),flush=True)
    log=root/(name+'.log')
    with log.open('x') as out:
        result=subprocess.run(argv,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        print(log.read_text()[-16000:],flush=True)
        raise RuntimeError(f'{name} failed: {result.returncode}')
    print('STAGE_DONE',name,now(),flush=True)
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
subprocess.run([bench_python,'-m','unittest','tests.test_document_training','tests.test_packed_attention','-v'],check=True,timeout=180)
run_stage('prepare',[bench_python,'-m','scripts.benchmark_document_training','prepare','--root',str(work),'--eval-rows','512'],os.environ.copy(),900)
print('BENCHMARK_DATA', (work/'data.json').read_text(),flush=True)
print('BURN_IDENTITIES_AND_SOURCE_VERIFIED',record['workers'],flush=True)
source=project/'resources/accelerate_config.yaml';cached=Path(default_yaml_config_file)
if cached.exists(): shutil.copy2(cached,root/'accelerate.previous.yaml')
cached.parent.mkdir(parents=True,exist_ok=True)
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
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}
    subprocess.run([bench_python,'-m','accelerate.commands.accelerate_cli','env'],env=env,check=True,timeout=120)
    run_stage('correctness',[bench_python,'-m','scripts.benchmark_document_training','correctness',
        '--root',str(work)],{**env,'CUDA_VISIBLE_DEVICES':'0'},1800)
    correctness=json.loads((work/'correctness.json').read_text())
    assert correctness['status']=='passed'
    print('FULL_MODEL_CORRECTNESS',json.dumps(correctness),flush=True)
    for mode in ['sdpa_isolated','fa4_isolated']:
        time.sleep(2);require_free(list(range(8)))
        run_stage(mode,[bench_python,'-m','accelerate.commands.accelerate_cli','launch',
            '--config_file',str(cached),'--module','scripts.benchmark_document_training','train',
            '--root',str(work),'--mode',mode,'--steps','200','--eval-rows','512'],env,1800)
        assert (work/mode/'final_model/model.safetensors').stat().st_size>1000000000
    run_stage('summarize',[bench_python,'-m','scripts.benchmark_document_training','summarize',
        '--root',str(work),'--steps','200','--eval-rows','512','--modes','sdpa_isolated','fa4_isolated'],os.environ.copy(),180)
    receipt['summary']=json.loads((work/'summary.json').read_text())
    receipt['comparison']=json.loads((work/'comparison.json').read_text())
    receipt['passed']=receipt['summary']['status']=='passed'

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
            receipt['burn']=start_burn(root,'tjx3-document-stability-200-a01-burn',project)
            print('AUTOMATIC_BURN_RESTORED_AND_VERIFIED',flush=True)
        except BaseException as error:
            receipt['handoff_error']=repr(error);traceback.print_exc()
    if owned is not None and 'burn' in receipt and guard.exists() and guard.stat().st_ino==owned and guard.read_text()==payload:
        guard.unlink()
    receipt['finished_at']=now();write_json(root/'result.json',receipt)
print('DOCUMENT_TRAINING_RESULT',json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt.get('passed') and 'burn' in receipt else 1)
PY

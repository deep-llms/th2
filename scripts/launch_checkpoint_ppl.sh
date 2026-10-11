#!/usr/bin/env bash
# Evaluate retained checkpoints without training; the existing supervisor owns the burn handoff.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=${2:?Pass verified hostname}
TASK_SOURCE=${3:?Pass completed source run directory}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1042 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" "$TASK_HOST" "$TASK_SOURCE" <<'PY'
import importlib.metadata as md,json,os,shutil,subprocess,sys
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from scripts.checkpoint_ppl_study import make
root,host,source=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
parent=json.loads((source.parent/'supervisor.json').read_text())
assert parent['training_status']=='ok' and parent.get('finished_at')
assert parent['burn']['collective_progress_verified'] is True
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
versions={p:md.version(p) for p in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl','wandb')}
for p,v in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),
            ('datasets','4.8.5'),('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0'),('wandb','0.30.0')]:
    assert versions[p].split('+')[0]==v,(p,versions[p])
assert shutil.disk_usage(root).free>20*2**30
cfg=json.loads((source/'seed-1042/A/train_config.json').read_text())
for section,keys in [('model',('config_name','tokenizer_name')),('data',('data_dir','eval_data_dir'))]:
    for key in keys: assert Path(cfg[section][key]).exists(),key
assets=json.loads(Path('resources/qwen3_base_assets.json').read_text())
for name,expected in assets['files'].items():assert digest(Path(cfg['model']['tokenizer_name'])/name)==expected
config_source=Path('resources/accelerate_config.yaml'); destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(config_source,destination)
assert config_source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
subprocess.run(['accelerate','env'],check=True)
# CPU-only saved-checkpoint evaluation tests in the pinned environment.
# A failure exits before any GPU reclamation.
with (root/'cpu-tests.log').open('x') as log:
    subprocess.run([sys.executable,'-m','unittest',
        'tests.test_checkpoint_ppl_study','-q'],
        stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
print('CHECKPOINT_PPL_TESTS_PASSED',flush=True)
jobs=make(source,root)
inspection=inspect()
assert inspection['host']==host and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved)
           for pid in inspection['workers'])
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',versions=versions,stages=len(jobs),
    source_run=str(source),
    accelerate_cache=str(destination),accelerate_sha256=digest(destination)),indent=2))
print('CHECKPOINT_PPL_PREFLIGHT_PASSED',len(jobs),flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" \
  --output "$TASK_ROOT/supervised" --inspection "$TASK_ROOT/inspection.json" \
  --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

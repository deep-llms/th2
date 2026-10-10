#!/usr/bin/env bash
# Three-checkpoint downstream study using existing evaluation/training entry points.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=${2:?Pass verified hostname}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate eval_fa4
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m pip check
python -u - "$TASK_ROOT" "$TASK_HOST" <<'PY'
import copy,importlib.metadata as md,json,shutil,subprocess,sys,time
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from scripts.stage_deep_kv_resume import digest as file_digest
from scripts.checkpoint_downstream_study import make,GROUPS
from deep_kv.report import comparable_config
root,host=Path(sys.argv[1]),sys.argv[2];project=Path.cwd()
versions={p:md.version(p) for p in ('torch','transformers','accelerate','datasets','lm_eval','flash-attn-4','nvidia-cutlass-dsl')}
for p,v in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),('datasets','4.8.5'),
            ('lm_eval','0.4.10'),('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0')]:
    assert versions[p].split('+')[0]==v,(p,versions[p])
assert shutil.disk_usage(root).free>500*2**30
with (root/'cpu-tests.log').open('x') as log:
    subprocess.run([sys.executable,'-m','unittest','tests.test_fewshot_eval','tests.test_finetune',
        'tests.test_finetune_extended','tests.test_checkpoint_downstream_study','-q'],
        stdout=log,stderr=subprocess.STDOUT,check=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
print('DOWNSTREAM_CPU_TESTS_PASSED',flush=True)
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run')
continuation=Path('/mnt/local/_outputs/deep-llms_th2/q359-A-time-match10800-seed1042-20261010-a01/supervised/run')
for source in (old,continuation):assert json.loads((source/'run.json').read_text())['status']=='ok'
spec={}; common=None; tokenizer=None
for label,arm,step,source in [('A-10000','A',10000,old),('P6-iso-10000','P6-iso',10000,old),('A-10800','A',10800,continuation)]:
    folder=source/'seed-1042'/arm; checkpoint=folder/f'checkpoint-{step}'
    assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==step
    result=json.loads((folder/'result.json').read_text());assert result['arm']==arm and result['global_step']==step
    cfg=json.loads((folder/'train_config.json').read_text())
    assert cfg['pilot']['arm']==arm and cfg['pilot']['attention_backend']=='fa4'
    assert cfg['training']['seed']==cfg['training']['data_seed']==1042
    normalized=comparable_config(cfg);normalized['training']['save_total_limit']=0
    if common is None:common=normalized
    else:assert normalized==common,'Pretraining recipes differ beyond arm/retention'
    if tokenizer is None:tokenizer=cfg['model']['tokenizer_name']
    else:assert tokenizer==cfg['model']['tokenizer_name']
    assert (checkpoint/'model.safetensors').stat().st_size>1000000000
    spec[str(checkpoint)]=dict(label=label,arm=arm,step=step,checkpoint_sha256=file_digest(checkpoint/'model.safetensors'))
print('CHECKPOINT_IDENTITIES_VERIFIED',json.dumps(spec),flush=True)
data_roots=dict(pairs=Path('/mnt/local/_data/deep-llms_th2/downstream-english-20261007'),
                extended=Path('/mnt/local/_data/deep-llms_th2/supervised-english-20261008-v2'))
manifests=dict(pairs=project/'resources/downstream_english_20261007.json',
               extended=project/'resources/supervised_english_20261008_v2.json')
files=[(data_roots[group]/f['path'],f['bytes']) for group,manifest in manifests.items()
       for repo in json.loads(manifest.read_text())['repositories'] for f in repo['files']]
# Controller #d is asynchronous. Leave burns running while inputs arrive.
for attempt in range(120):
    missing=[str(p) for p,size in files if not p.is_file() or p.stat().st_size!=size]
    if not missing:break
    print('WAITING_FOR_CONTROLLER_DOWNLOAD',len(missing),flush=True);time.sleep(30)
else:raise RuntimeError('Benchmark download incomplete; GPUs untouched')
subprocess.run([sys.executable,'-m','scripts.check_downstream_eval','data','--dataset-root',str(data_roots['pairs']),
    '--dataset-manifest',str(manifests['pairs']),'--output',str(root/'data-validation.json')],check=True)
actual=json.loads((root/'data-validation.json').read_text())
assert actual==json.loads((project/'resources/downstream_english_reference_20261007.json').read_text())
subprocess.run([sys.executable,'-m','scripts.finetune_study','audit-data','--dataset-root',str(data_roots['extended']),
    '--manifest',str(manifests['extended']),'--tokenizer',tokenizer,'--output',str(root/'supervised-data-validation.json')],check=True)
assert json.loads((root/'supervised-data-validation.json').read_text())==json.loads((project/'resources/supervised_reference_20261008.json').read_text())
reference=json.loads((project/'resources/fewshot_reference_20261007.json').read_text())
for shot,tasks in GROUPS.items():
    subset=copy.deepcopy(actual);subset['tasks']={t:actual['tasks'][t] for t in tasks}
    (root/f'data-validation-{shot}.json').write_text(json.dumps(subset,indent=2))
    audit=root/f'prompt-audit-{shot}.json'
    subprocess.run([sys.executable,'-m','scripts.check_downstream_eval','fewshot',
        '--dataset-root',str(data_roots['pairs']),'--dataset-manifest',str(manifests['pairs']),
        '--tokenizer',tokenizer,'--tasks',*tasks,'--num-fewshot',str(shot),'--output',str(audit)],check=True)
    assert json.loads(audit.read_text())==reference[str(shot)]
    assert all(v['truncated_documents']==0 for v in reference[str(shot)]['tasks'].values())
print('DATA_AND_PROMPT_AUDITS_PASSED',flush=True)
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
c=load_config_from_file(str(destination)).to_dict()
assert c['num_processes']==8 and c['mixed_precision']=='bf16' and c['distributed_type']=='MULTI_GPU'
subprocess.run(['accelerate','env'],check=True)
items=make(root,project,spec,data_roots,manifests)
inspection=inspect();assert inspection['host']==host and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in inspection['workers'])
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',checkpoints=spec,versions=versions,
    accelerate_sha256=digest(destination),manifest_sha256={k:digest(v) for k,v in manifests.items()},stages=len(items)),indent=2))
print('DOWNSTREAM_PREFLIGHT_PASSED',len(items),flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
  --inspection "$TASK_ROOT/inspection.json" --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

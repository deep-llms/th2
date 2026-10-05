#1 +60+a
#th2-tjx3-sdpa-repeat-200-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/sdpa-repeat-200-20261005-a01
TASK_SESSION=tjx3-sdpa-repeat-200-20261005-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
cat > "$TASK_ROOT/launch.sh" <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$1
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES='' python -u - "$TASK_ROOT" <<'PRE'
import json,shutil,sys,subprocess,hashlib,importlib.metadata as md
from pathlib import Path
from datasets import load_from_disk
import torch
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from run_experiments import load_jobs
root=Path(sys.argv[1]);project=Path.cwd()
reference=Path('/mnt/local/_outputs/deep-llms_th2/document-stability-200-20261004-a01/benchmark')
assert shutil.disk_usage(root).free>40*2**30 and shutil.which('patch')
versions={p:md.version(p) for p in ['torch','transformers','accelerate','datasets']}
assert versions=={'torch':'2.14.1','transformers':'5.9.0','accelerate':'1.13.0','datasets':'4.8.5'},versions
status=inspect();assert status['host']=='thiennh-p6-tjx3-worker-0' and not status['guard_disabled']
assert len(status['gpus'])==8 and all('B200' in g['name'] for g in status['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved) for pid in status['workers'])
(root/'inspection.json').write_text(json.dumps(status,indent=2))
# Restore the exact historical implementation only in this new output directory.
manifest=json.loads((project/'resources/sdpa_repeat_200_source.json').read_text())
patch=project/'resources/sdpa_repeat_200_source.patch';assert digest(patch)==manifest['patch_sha256']
snapshot=root/'historical_source';snapshot.mkdir()
for name,hashes in manifest['files'].items():
    assert digest(project/name)==hashes['base'],name
    target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,target)
subprocess.run(['patch','--batch','--forward','-p1','-d',str(snapshot),'-i',str(patch)],check=True)
for name,hashes in manifest['files'].items():assert digest(snapshot/name)==hashes['historical'],name
# Reuse and hash the original packed pools, not a new sample or new packing pass.
for split,meta_name,count,expected in [('data','data.json',8192,'fe3145a784ea8f12265d3a84f354b8c9912e0a881b3937389db1a1c3b9c8a082'),
    ('eval','eval-data.json',512,'6bff0bc70e47abd91a7203ba776f5727c4dc919c3c2ad59802d4655fe2cca396')]:
    data=load_from_disk(str(reference/split));assert len(data)==count
    h=hashlib.sha256()
    for batch in data.iter(batch_size=256):
        for key in ('input_ids','segments'):h.update(torch.tensor(batch[key],dtype=torch.long).numpy().tobytes())
    assert h.hexdigest()==expected==json.loads((reference/meta_name).read_text())['sha256']
assert json.loads((reference/'correctness.json').read_text())['status']=='passed'
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination);assert source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
(root/'preflight.json').write_text(json.dumps(dict(versions=versions,source=manifest,reference=str(reference),
    data_hashes_verified=True,accelerate_cache=str(destination),accelerate_sha256=digest(source),at=status['time']),indent=2))
# A foreground job runs the historical module, so imports cannot pick up newer code.
script=root/'train.sh'
script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nTASK_RUN=$1\n"+
    "mkdir \"$TASK_RUN/benchmark\"\n"+
    ''.join('ln -s '+str(reference/name)+' \"$TASK_RUN/benchmark/'+name+'\"\n' for name in ['data','eval'])+
    ''.join('cp '+str(reference/name)+' \"$TASK_RUN/benchmark/'+name+'\"\n' for name in ['data.json','eval-data.json','correctness.json'])+
    'cd '+str(snapshot)+'\n'+
    'exec '+sys.executable+' -m accelerate.commands.accelerate_cli launch --config_file '+str(destination)+
    ' --module scripts.benchmark_document_training train --root \"$TASK_RUN/benchmark\" --mode sdpa_isolated --steps 200 --eval-rows 512\n')
subprocess.run(['bash','-n',str(script)],check=True)
manifest_jobs={'jobs':[
    dict(name='sdpa-repeat-200',gpus=list(range(8)),timeout_seconds=1800,
         argv=['bash',str(script),'{run_dir}'],required_outputs=[dict(path='benchmark/sdpa_isolated/rank-0.json',json_equals={'status':'passed'}),
            dict(path='benchmark/sdpa_isolated/final_model/model.safetensors')]),
    dict(name='compare-sdpa-repeat',argv=['{python}','-u','-m','scripts.compare_sdpa_repeat',
         '--reference',str(reference),'--repeat','{run_dir}/benchmark','--output','{run_dir}/comparison.json'],
         required_outputs=[dict(path='comparison.json',json_equals={'status':'completed','matched_streams_and_learning_rates':True})])]}
(root/'jobs.json').write_text(json.dumps(manifest_jobs,indent=2));load_jobs(root/'jobs.json')
print('HISTORICAL_SOURCE_AND_DATA_VERIFIED',status['workers'],str(destination),flush=True)
subprocess.run([sys.executable,'-m','unittest','tests.test_document_training','-v'],cwd=snapshot,check=True)
PRE
accelerate env
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" --output "$TASK_ROOT/supervised" \
 --inspection "$TASK_ROOT/inspection.json" --host thiennh-p6-tjx3-worker-0 \
 --burn-session tjx3-sdpa-repeat-200-20261005-a01-final-burn
LAUNCH
bash -n "$TASK_ROOT/launch.sh"
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' "$TASK_ROOT/launch.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 75 "$TASK_ROOT/launch.log"

#!/usr/bin/env bash
# Fresh seed-1042 A/P6-iso fits; all eight B200s per arm and verified final burn.
set -euo pipefail
TASK_ROOT=$1
TASK_HOST=${2:?Pass the verified target hostname}
cd /mnt/local/deep-llms_th2
test "$(hostname)" = "$TASK_HOST"
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate attention_bench
export NCCL_NVLS_ENABLE=0 WANDB_MODE=offline WANDB_PROJECT=deep2shallow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1042 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u - "$TASK_ROOT" "$TASK_HOST" <<'PY'
import hashlib,importlib.metadata as md,json,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
from scripts.proxy_10k_fresh_queue import make
from train import load_text

root=Path(sys.argv[1]);project=Path.cwd()
versions={p:md.version(p) for p in ('torch','transformers','accelerate','datasets','flash-attn-4','nvidia-cutlass-dsl')}
for p,v in [('torch','2.14.1'),('transformers','5.9.0'),('accelerate','1.13.0'),
            ('datasets','4.8.5'),('flash-attn-4','4.0.0b33'),('nvidia-cutlass-dsl','4.8.0')]:
    assert versions[p].split('+')[0]==v,(p,versions[p])
free=shutil.disk_usage(root).free
assert free>300*2**30,free
recipe=json.loads((project/'proxy_heads.b200.json').read_text())
for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
    assert Path(recipe[key]).exists(),key
assets=json.loads((project/'resources/qwen3_base_assets.json').read_text())
for name,expected in assets['files'].items():
    assert digest(Path(recipe['tokenizer_name'])/name)==expected,name
# Compare downloaded English files with the previously published immutable
# release manifest, before reclaiming GPUs. Hash data in bounded memory.
data_root=Path(recipe['data_dir']).parents[2]
# Fresh-node tokenization and packing retain several full-corpus cache columns.
# Allow space for both cache stages as well as checkpoint writes.
data_free=shutil.disk_usage(data_root).free
assert data_free>2*1024**4,('Need 2 TiB free for fresh full-corpus caches',data_free)
print('FREE_SPACE_GIB',dict(outputs=free/2**30,data=data_free/2**30),flush=True)
manifest_path=data_root/'dataset_manifest.json'
assert digest(manifest_path)=='dbba73b7a95ebc530297bac5a2151be6292f080e80e9b3101302a905806dde9a'
manifest=json.loads(manifest_path.read_text())
entries=[(name,entry) for name,entry in manifest['files'].items()
         if name.startswith('subsets/qwen3_0.6b_base_en_30B/')]
assert entries
def verify_file(item):
    name,entry=item;path=(data_root/name).resolve()
    assert path.is_relative_to(data_root.resolve()) and path.stat().st_size==entry['bytes'],name
    checksum=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):checksum.update(chunk)
    assert checksum.hexdigest()==entry['sha256'],name
    return entry['bytes']
print('VERIFYING_ENGLISH_RELEASE_FILES',len(entries),flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    verified_bytes=sum(pool.map(verify_file,entries))
print('ENGLISH_RELEASE_HASHES_VERIFIED',len(entries),verified_bytes,flush=True)
data={}
for key in ('data_dir','eval_data_dir'):
    ds=load_text(recipe[key]);assert len(ds)>0 and 'text' in ds.column_names
    data[key]=dict(rows=len(ds),fingerprint=ds._fingerprint)
source=project/'resources/accelerate_config.yaml';destination=Path(default_yaml_config_file)
destination.parent.mkdir(parents=True,exist_ok=True)
if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
shutil.copy2(source,destination)
assert source.read_bytes()==destination.read_bytes()
config=load_config_from_file(str(destination)).to_dict()
assert config['num_processes']==8 and config['mixed_precision']=='bf16' and config['distributed_type']=='MULTI_GPU'
subprocess.run(['accelerate','env'],check=True)
inspection=inspect()
assert inspection['host']==sys.argv[2] and not inspection['guard_disabled']
assert len(inspection['gpus'])==8 and all('B200' in g['name'] for g in inspection['gpus'])
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
assert all(approved_launcher(pid,approved) or approved_launcher(process(pid)['ppid'],approved)
           for pid in inspection['workers'])
items=make(project/'proxy_heads.b200.json',root/'jobs.json')
(root/'inspection.json').write_text(json.dumps(inspection,indent=2))
(root/'preflight.json').write_text(json.dumps(dict(status='passed',versions=versions,data=data,
  dataset_manifest_sha256=digest(manifest_path),verified_data_files=len(entries),verified_data_bytes=verified_bytes,
  environment=sys.executable,stages=len(items),free_bytes=free,
  accelerate_cache=str(destination),accelerate_sha256=digest(destination),
  source_sha256={name:digest(project/name) for name in
    ('deep_kv/proxy.py','deep_kv/proxy_training.py','train.py','scripts/proxy_10k_fresh_queue.py',
     'scripts/check_fa4_proxy.py','scripts/train_then_burn.py')}),indent=2))
print('PROXY_10K_PREFLIGHT_PASSED',len(items),inspection['workers'],flush=True)
PY
exec python -u -m scripts.train_then_burn --config "$TASK_ROOT/jobs.json" \
  --output "$TASK_ROOT/supervised" --inspection "$TASK_ROOT/inspection.json" \
  --host "$TASK_HOST" --burn-session "$(basename "$TASK_ROOT")-final-burn"

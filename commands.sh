#1 +60+a
#th2-tjx3-proxy-fa4-restart-inspect-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,os
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect,process
from scripts.train_then_burn import argv
old=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-screen-seed42-2500-20261005-a01')
control=Path('/mnt/local/_outputs/deep-llms_th2/proxy-fa4-restart-control-20261005-a01')
control.mkdir(exist_ok=False)
state=inspect();print('GPU_STATUS',json.dumps(state),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');marker=json.loads(guard.read_text())
assert marker['output']==str(old/'supervised')
owner=process(marker['owner_pid']);args=argv(owner['pid'])
assert 'scripts.train_then_burn' in args and args[args.index('--output')+1]==str(old/'supervised')
chains={}
for pid in state['workers']:
    chain=[];current=pid
    while current!=owner['pid']:
        assert current>1 and len(chain)<10
        item=process(current);chain.append(item);current=item['ppid']
    chains[str(pid)]=chain
assert len(chains)==8
roots=[Path('/mnt/local/_data/deep-llms_th2/cx_sampled_old/subsets/qwen3_0.6b_base_en_30B')/split for split in ('train','validation')]
caches=[];sources=[]
for root in roots:
    assert root.is_dir() and not root.is_symlink()
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),str(p)
        if p.is_file():
            st=p.stat();item=dict(path=str(p),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns)
            (caches if p.name.startswith(('cache-','tmp-')) else sources).append(item)
value=dict(status='inspected',host=state['host'],time=state['time'],old_output=str(old),
    supervisor=owner,supervisor_argv=args,guard=marker,gpu_status=state,worker_chains=chains,
    dataset_roots=list(map(str,roots)),cache_files=caches,source_files=sources)
(control/'inspection.json').write_text(json.dumps(value,indent=2))
print('INSPECTION',json.dumps(dict(supervisor=owner,argv=args,control=str(control),
    cache_count=len(caches),cache_bytes=sum(x['bytes'] for x in caches),sources=sources,
    cache_name_examples=[x['path'] for x in caches[:8]])),flush=True)
print('OLD_SUPERVISOR',(old/'supervised/supervisor.json').read_text(),flush=True)
print('OLD_QUEUE',(old/'supervised/run/run.json').read_text(),flush=True)
PY

#1 +30+a
#th2-6y3o-efficient-migration-inventory-20261026-a01
set -euo pipefail
python3 - <<'REMOTE'
import datetime, hashlib, json, os, socket, stat, subprocess
from pathlib import Path
root=Path('/mnt/local/deep-llms_th2')
def emit(kind, **data):
    print(kind+' '+json.dumps(data, sort_keys=True), flush=True)
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0', 'Unexpected host'
assert not root.is_symlink() and root.resolve()==root==Path.cwd(), 'Unexpected code path'
def describe(p):
    s=p.lstat()
    return {'name':p.name,'kind':'symlink' if p.is_symlink() else 'directory' if p.is_dir() else 'file','bytes':s.st_size}
emit('INVENTORY_START',hostname=socket.gethostname(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cwd=str(root))
for p in sorted(root.iterdir()):
    emit('ROOT_ENTRY',**describe(p))
    if p.is_dir() and not p.is_symlink() and p.name!='.git':
        children=sorted(p.iterdir())
        emit('CHILDREN',parent=p.name,count=len(children),entries=[describe(c) for c in children[:100]])
for base in ['/mnt/local/_data/deep-llms_th2','/mnt/local/_models/deep-llms_th2','/mnt/local/_outputs/deep-llms_th2','/mnt/local/conda-py311/envs','/mnt/local/uvenvs']:
    p=Path(base)
    emit('PRESERVE_EXTERNAL',path=base,exists=p.exists(),entries=[describe(c) for c in sorted(p.iterdir())] if p.is_dir() else [])
# Only expose script paths and process identity, never environment/argument values.
def proc(pid):
    p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))[:-1]
    scripts=[a for a in args if a.endswith('.py') and '\n' not in a]
    return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),'exe':str((p/'exe').resolve()),'cwd':str((p/'cwd').resolve()),'scripts':scripts}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
emit('GPU_INVENTORY',gpus=gpu.splitlines(),apps=apps.splitlines())
for line in apps.splitlines():
    pid=int(line.split(',')[0]); chain=[]; seen=set()
    while pid>1 and pid not in seen:
        seen.add(pid)
        try: item=proc(pid)
        except (FileNotFoundError,ProcessLookupError): break
        chain.append(item); pid=item['ppid']
    emit('GPU_ANCESTRY',chain=chain)
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        item=proc(int(p.name))
        if item['scripts'] and (item['cwd']==str(root) or any(str(root) in a for a in item['scripts'])):
            emit('PROJECT_PROCESS',**item)
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
for p in [root/'resources/llm_pretrain_burn.py',Path('/tmp/llm_pretrain_burn.py')]:
    if p.is_file(): emit('BURN_SOURCE',path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
emit('INVENTORY_COMPLETE',commands_sha256=hashlib.sha256((root/'commands.sh').read_bytes()).hexdigest())
REMOTE

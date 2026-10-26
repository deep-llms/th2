#1 +30+a
#th2-6y3o-efficient-migration-cleanup-20261026-a01
set -euo pipefail
python3 - <<'REMOTE'
import csv, datetime, hashlib, json, os, socket, subprocess, time
from pathlib import Path
ROOT=Path('/mnt/local/deep-llms_th2')
BACKUP=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-migration-20261026-a01')
BURN=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-optimized-20261025-a03/source/resources/llm_pretrain_burn.py')
SHA='2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174'
KEEP={'_run_log_', '_install_active.txt', '_previous_run_status.log', 'commands.sh', '_dl_active.txt'}
EXPECTED={'training_data.py', 'README.md', 'AGENTS.md', 'envs', '_run_log_', 'packed_training.py', 'jobs.training.example.json', 'commands.sh', 'docs', 'bdot', 'configs', 'run_experiments.py', 'scripts', '.gitignore', 'packed_evaluation.py', '_dl_active.txt', 'resources', 'jobs.example.json', 'eval', 'resume_recipe.py', 'tests', 'packed_fa4.py', 'requirements.txt', '_install_active.txt', '_previous_run_status.log', 'packed_fa2.py', 'project.example.json', 'train.py', 'queued'}
GPU_IDENTITIES={'581907': 285074151, '581908': 285074151, '581909': 285074151, '581910': 285074151, '581911': 285074151, '581912': 285074151, '581913': 285074151, '581914': 285074151}
def emit(event, **data):
    print(event+' '+json.dumps(data,sort_keys=True),flush=True)
def process(pid):
    p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    args=(p/'cmdline').read_bytes().decode().split(chr(0))[:-1]
    return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),'args':args}
def gpu_snapshot():
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for row in csv.reader(query.splitlines()):
        pid=int(row[0].strip()); w=process(pid); launcher=process(w['ppid'])
        if GPU_IDENTITIES.get(str(pid))!=w['start_ticks']:
            raise RuntimeError('GPU ownership changed since inventory')
        if launcher['pid']!=581838 or launcher['start_ticks']!=285074033 or str(BURN) not in launcher['args']:
            raise RuntimeError('Unexpected GPU launcher')
        rows.append({'pid':pid,'start_ticks':w['start_ticks'],'gpu_uuid':row[1].strip()})
    if len(rows)!=8 or len({r['gpu_uuid'] for r in rows})!=8:
        raise RuntimeError('Expected one burn worker on each of eight GPUs')
    return sorted(rows,key=lambda r:r['pid'])
def guard_tree(path):
    if path.is_symlink(): return
    if path.is_mount(): raise RuntimeError('Refusing mount: '+str(path))
    if path.is_dir():
        def fail(err): raise err
        for parent,dirs,_ in os.walk(path,followlinks=False,onerror=fail):
            for name in dirs:
                p=Path(parent)/name
                if not p.is_symlink() and p.is_mount(): raise RuntimeError('Refusing nested mount')
def external_identity():
    roots=['/mnt/local/_data/deep-llms_th2','/mnt/local/_models/deep-llms_th2','/mnt/local/conda-py311/envs','/mnt/local/_outputs/deep-llms_th2']
    return {p: {'device':Path(p).stat().st_dev,'inode':Path(p).stat().st_ino,
        'entries':{x.name:[x.lstat().st_dev,x.lstat().st_ino] for x in Path(p).iterdir() if x!=BACKUP}} for p in roots}
if socket.gethostname()!='thiennh-p6-6y3o-worker-0': raise RuntimeError('Unexpected host')
if ROOT.is_symlink() or ROOT.resolve()!=ROOT or Path.cwd()!=ROOT: raise RuntimeError('Unexpected code path')
if BACKUP.exists() or BACKUP.is_symlink(): raise RuntimeError('Backup must be fresh')
if BACKUP.parent.resolve()!=BACKUP.parent or ROOT.stat().st_dev!=BACKUP.parent.stat().st_dev:
    raise RuntimeError('Backup must use the same filesystem and no symlink')
if BURN.is_symlink() or hashlib.sha256(BURN.read_bytes()).hexdigest()!=SHA: raise RuntimeError('Unexpected live burn source')
if {p.name for p in ROOT.iterdir()}!=EXPECTED: raise RuntimeError('Code directory changed since inventory')
command=ROOT/'commands.sh'
if command.is_symlink() or not command.is_file(): raise RuntimeError('Unexpected command file')
command_hash=hashlib.sha256(command.read_bytes()).hexdigest()
targets=[ROOT/name for name in sorted(EXPECTED-KEEP)]
for p in targets: guard_tree(p)
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        info=process(int(p.name)); cwd=(p/'cwd').resolve(strict=True)
        if cwd!=ROOT and cwd.is_relative_to(ROOT): raise RuntimeError('Live process inside old code: '+p.name)
        for arg in info['args'][1:]:
            if not arg.endswith(('.py','.sh')) or '\n' in arg: continue
            candidate=(Path(arg) if arg.startswith('/') else cwd/arg).resolve()
            if candidate!=command and candidate.is_relative_to(ROOT): raise RuntimeError('Live project script: '+p.name)
    except (FileNotFoundError,ProcessLookupError): pass
before=gpu_snapshot(); external=external_identity()
# Save a content manifest so the moved code can be restored or audited.
manifest={}
for top in targets:
    files=[top] if not top.is_dir() else list(top.rglob('*'))
    for p in files:
        if p.is_symlink(): manifest[str(p.relative_to(ROOT))]={'symlink':os.readlink(p)}
        elif p.is_file(): manifest[str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
if gpu_snapshot()!=before: raise RuntimeError('GPU ownership changed during preflight')
BACKUP.mkdir()
(BACKUP/'archived_code').mkdir()
(BACKUP/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
emit('CLEANUP_PREFLIGHT',backup=str(BACKUP),archiving=[p.name for p in targets],preserving=sorted(KEEP),file_count=len(manifest))
for p in targets: p.rename(BACKUP/'archived_code'/p.name)
if {p.name for p in ROOT.iterdir()}!=KEEP: raise RuntimeError('Unexpected remaining entries')
for rel,meta in manifest.items():
    p=BACKUP/'archived_code'/rel
    if 'sha256' in meta and hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']: raise RuntimeError('Archive content mismatch')
    if 'symlink' in meta and os.readlink(p)!=meta['symlink']: raise RuntimeError('Archive link mismatch')
if external_identity()!=external: raise RuntimeError('External directory identities changed')
if hashlib.sha256(command.read_bytes()).hexdigest()!=command_hash: raise RuntimeError('Command changed')
if hashlib.sha256(BURN.read_bytes()).hexdigest()!=SHA: raise RuntimeError('Live burn source changed')
time.sleep(10)
if gpu_snapshot()!=before: raise RuntimeError('GPU ownership changed after cleanup')
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'code_root':str(ROOT),'remaining':sorted(KEEP),'backup':str(BACKUP),'archived_files':len(manifest),'external_directory_identities_unchanged':True,'gpu_processes_unchanged':True,'gpu_processes':before,'commands_sha256':command_hash}
(BACKUP/'cleanup_result.json').write_text(json.dumps(result,indent=2)+'\n')
emit('CLEANUP_COMPLETE',**result)
REMOTE

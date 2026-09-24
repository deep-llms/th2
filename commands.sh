#1 +30+a
#th2-postrun-gpu-identity-20260924-a01
set -euo pipefail
date -u
PYTHONPATH="$PWD" /usr/bin/python3 -u - <<'CHECK'
import hashlib,json,os
from pathlib import Path
from scripts.gpu_status import snapshot
status=snapshot(); print('GPU_STATUS',json.dumps(status),flush=True)
seen=set()
for gpu in status:
    for worker in gpu['pids']:
        pid=worker
        while pid>1 and pid not in seen:
            seen.add(pid);root=Path('/proc')/str(pid)
            try:
                fields=(root/'stat').read_text().rsplit(')',1)[1].split()
                argv=[s.decode(errors='replace') for s in (root/'cmdline').read_bytes().split(b'\0') if s]
                record={'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),'exe':os.readlink(root/'exe'),'command_prefix':argv[:3]}
                record['gpu_environment']={}
                for item in (root/'environ').read_bytes().split(b'\0'):
                    key,_,value=item.partition(b'=')
                    if key in (b'CUDA_VISIBLE_DEVICES',b'BURN_MS',b'IDLE_MS',b'BURN_DIM',b'GRAD_ELEMS'):
                        record['gpu_environment'][key.decode()]=value.decode()
                print('PROCESS',json.dumps(record),flush=True)
                if len(argv)>1 and argv[1].endswith('.py') and 'burn' in Path(argv[1]).name.lower():
                    source=Path(argv[1]);content=source.read_bytes()
                    print('BURN_SOURCE',str(source),hashlib.sha256(content).hexdigest(),flush=True)
                    if len(content)<65536:print(content.decode(),flush=True)
                    print('BURN_STDOUT',os.readlink(root/'fd/1'),flush=True)
                pid=record['ppid']
            except (FileNotFoundError,ProcessLookupError):
                print('PROCESS_EXITED',pid,flush=True);break
print('IDENTITY_INSPECTION_COMPLETE',flush=True)
CHECK

"""Inspect GPU workloads, or stop exactly inspected identities with explicit authorization."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

from scripts.gpu_status import require_free, snapshot


def process(pid):
    if pid <= 1:
        raise ValueError('PID 1 is never a workload target')
    root=Path('/proc')/str(pid)
    fields=(root/'stat').read_text().rsplit(')',1)[1].split()
    command=(root/'cmdline').read_bytes()
    argv=[s.decode(errors='replace') for s in command.split(b'\0') if s]
    kind=None
    if len(argv)>=3 and Path(argv[1]).name=='vllm' and argv[2]=='serve':
        kind='vllm_server'
    elif len(argv)>=2 and Path(argv[1]).name in ('polite_burn.py','llm_pretrain_burn.py'):
        kind='gpu_burn'
    prefix=argv[:3]
    if '-c' in prefix:
        prefix=prefix[:prefix.index('-c')+1]
    return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),
            'command_sha256':hashlib.sha256(command).hexdigest(),'exe':os.readlink(root/'exe'),
            'command_prefix':prefix,'launcher_kind':kind}


def ownership(status):
    return [(g['index'],g['uuid'],sorted(g['pids'])) for g in status]


def inspect():
    status=snapshot(list(range(8)))
    workers=sorted({pid for g in status for pid in g['pids']})
    records={str(pid):process(pid) for pid in workers}
    roots=[]
    for pid in workers:
        ppid=records[str(pid)]['ppid']
        if ppid>1:
            parent=process(ppid)
            if parent['launcher_kind'] in ('vllm_server','gpu_burn'):
                records[str(ppid)]=parent
                roots.append(ppid)
    return {'host':socket.gethostname(),'time':datetime.now(timezone.utc).isoformat(),
            'gpus':status,'workers':workers,'stop_roots':sorted(set(roots)-set(workers)),
            'processes':records,'guard_disabled':Path('/mnt/local/_gpu_guard/DISABLED').exists()}


def validate(expected, status, records, host):
    if expected['host']!=host or host!='thiennh-p6-tpbw-worker-0':
        raise ValueError('Node identity changed')
    if ownership(expected['gpus'])!=ownership(status):
        raise ValueError('GPU ownership changed; re-inspect before stopping')
    if [g['index'] for g in status]!=list(range(8)):
        raise ValueError('Expected all eight physical GPUs')
    targets=expected['workers']+expected['stop_roots']
    if not targets or len(set(targets))!=len(targets) or any(p<=1 for p in targets):
        raise ValueError('Invalid or duplicate inspected process identities')
    if set(expected['workers'])!={pid for g in status for pid in g['pids']}:
        raise ValueError('Worker list differs from inspected GPU processes')
    if set(records)!=set(map(str,targets)) or records!=expected['processes']:
        raise ValueError('A process identity changed; refusing all signals')
    for pid in expected['stop_roots']:
        if records[str(pid)]['launcher_kind'] not in ('vllm_server','gpu_burn'):
            raise ValueError('Only recognized inspected workload launchers can be stopped')
        if not any(records[str(w)]['ppid']==pid for w in expected['workers']):
            raise ValueError('Launcher is not a parent of an inspected GPU worker')


def reclaim(expected, output):
    if output.exists():
        raise ValueError('Reclamation receipt already exists')
    handles={}
    try:
        targets=expected['workers']+expected['stop_roots']
        # Pin identities before the final full validation; never send a group signal.
        for pid in targets:
            if pid<=1:raise ValueError('Refusing PID 1')
            handles[pid]=os.pidfd_open(pid)
        records={str(pid):process(pid) for pid in targets}
        validate(expected,snapshot(list(range(8))),records,socket.gethostname())
        if not Path('/mnt/local/_gpu_guard/DISABLED').is_file():
            raise ValueError('Guard must be held disabled during authorized reclamation')
        print('AUTHORIZED_VERIFIED_STOP',json.dumps(records),flush=True)
        # Gracefully stop the verified serving/burn launchers first so they do
        # not restart their workers. User explicitly authorized these workloads.
        for pid in expected['stop_roots']:
            try:signal.pidfd_send_signal(handles[pid],signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(10)
        for pid in targets:
            try:signal.pidfd_send_signal(handles[pid],signal.SIGKILL)
            except ProcessLookupError:pass
        time.sleep(30)
        after=require_free(list(range(8)))
        with output.open('x') as f:
            json.dump({'status':'ok','host':socket.gethostname(),'stopped':records,
                       'all_eight_gpus_free':True,'after':after},f,indent=2)
        print('ALL_EIGHT_GPUS_VERIFIED_FREE',flush=True)
    finally:
        for handle in handles.values():os.close(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','stop'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--inspection',type=Path)
    parser.add_argument('--authorized-stop',action='store_true')
    args=parser.parse_args()
    if args.mode=='inspect':
        record=inspect()
        with args.output.open('x') as f:json.dump(record,f,indent=2)
        print(json.dumps(record,indent=2),flush=True)
    else:
        if not args.authorized_stop or args.inspection is None:
            parser.error('Stopping requires --authorized-stop and --inspection')
        reclaim(json.loads(args.inspection.read_text()),args.output)


if __name__=='__main__':main()

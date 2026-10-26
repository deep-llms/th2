"""Read-only process identities and Linux PID handles for the owned-job supervisor.

Reused from the previous q359 project; no standalone stop command.
"""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import socket

from scripts.gpu_status import require_free, snapshot


def _pidfd_syscall(number, *args):
    # Linux x86-64 UAPI: pidfd_send_signal=424, pidfd_open=434. Conda may
    # omit Python's wrappers, and dev glibc predates the named libc wrappers.
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('PID-handle fallback requires Linux x86-64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(arg) for arg in args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    """Conda Python can omit os.pidfd_open even on a recent Linux kernel."""
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, pid, 0)


def pidfd_send_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    return _pidfd_syscall(424, fd, sig, 0, 0)


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

"""Single owned handoff for two shared warmups and six paired stack arms."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import traceback

import numpy as np
import torch

from efficient_llm.warm_order.anchor_long import ARMS, END, FORK
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from scripts.anchor_boundary_study import FIT
from scripts.anchor_long_study import prepare, read
from scripts.gpu_process import inspect, ownership, process
from scripts.gpu_status import require_free
from scripts.train_then_burn import approved_launcher, clean_owned_children, enable_subreaper, GUARD_HASH, stop_burns
from scripts.warm_order_queue import NATIVE, NATIVE_SHA, native, spawn, check_source


def stage(base, action, jobs, worker_module='scripts.anchor_long_study'):
    active=[]
    for gpu,(seed,arm) in enumerate(jobs):
        args=[sys.executable,'-u','-m',worker_module,action,'--base',str(base),
              '--seed',str(seed),'--arm',arm]
        active.append(spawn(base,action,f's{seed}-{arm}',gpu,args))
    while active:
        write_json(base/'status.json',dict(stage=action,running=[j['name'] for j in active],utc=time.time()))
        for job in list(active):
            rc=job['child'].poll()
            if rc is None:
                continue
            if rc:
                raise RuntimeError(f"{action} {job['name']} failed rc={rc}")
            active.remove(job)
        if active:
            time.sleep(8)


def check_ownership(proof, expected_hash):
    if sha(proof) != expected_hash:
        raise ValueError('Prior handoff receipt mismatch')
    previous=read(proof)['samples'][-1]
    sample=inspect()
    if sample['guard_disabled'] or sha(NATIVE)!=NATIVE_SHA or len(sample['workers'])!=8:
        raise RuntimeError('Unexpected guard or GPU state')
    if ownership(previous['gpus'])!=ownership(sample['gpus']) or any(process(int(pid))!=record for pid,record in previous['processes'].items()):
        raise RuntimeError('Native owner changed since verified handoff')
    if any(not approved_launcher(process(pid)['ppid'],{NATIVE:NATIVE_SHA}) for pid in sample['workers']):
        raise RuntimeError('GPU ownership unresolved')
    return sample


def run_anchor_arms(base):
    jobs=[(seed,arm) for seed in (0,1) for arm in ARMS]
    stage(base,'qualify',jobs)
    for seed,arm in jobs:
        q=read(base/f's{seed}'/f'qualification-{arm}'/'passed.json')
        if q!={'arm':arm,'seed':seed,'fp32_parity':True,'exact_resume':True}:
            raise ValueError('Missing parity or exact resume')
    stage(base,'fit',[(0,'shared'),(1,'shared')])
    stage(base,'fit',jobs)
    backups={}
    reports={}
    for seed in (0,1):
        for arm in ('shared',*ARMS):
            path=base/f's{seed}'/arm
            end=FORK if arm=='shared' else END
            if read(path/'complete.json')['step']!=end:
                raise ValueError('Fit incomplete')
            pin=read(path/('fork.json' if arm=='shared' else 'terminal.json'))
            state=Path(pin['path'])
            backup=base/'backup'/f's{seed}'/arm/state.name
            backup.parent.mkdir(parents=True)
            shutil.copyfile(state,backup)
            if sha(backup)!=pin['sha256'] or sha(state)!=pin['sha256']:
                raise ValueError('Backup mismatch')
            backups[str(backup)]=pin['sha256']
            if arm!='shared':
                reports[f's{seed}/{arm}']=read(path/'summary.json')
    write_json(base/'backup.json',backups)
    write_json(base/'report.json',reports)


def supervise(base, prepare_inputs, experiment, handoff, expected_hash):
    if socket.gethostname()!='thiennh-p6-ohmv-worker-0' or torch.__version__!='2.14.1+cu130' or np.__version__!='2.4.6':
        raise RuntimeError('Node/runtime mismatch')
    check_source(base)
    if shutil.disk_usage(base).free < 100 * 1024**3:
        raise RuntimeError('Insufficient checkpoint/backup space')
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    if sha(guard.parent/'gpu_guard.sh')!=GUARD_HASH:
        raise RuntimeError('Guard script changed')
    write_json(base/'status.json',dict(stage='CPU preparation'))
    prepare_inputs(base)
    before=check_ownership(handoff,expected_hash)
    write_json(base/'execution/admission.json',dict(ownership=before,utc=time.time()))
    marker=json.dumps(dict(owner_pid=os.getpid(),base=str(base),utc=time.time()))
    with guard.open('x') as f:
        f.write(marker)
    inode=guard.stat().st_ino
    reclaimed=False
    success=False
    def interrupt(*args):
        raise KeyboardInterrupt('Owned experiment interrupted')
    enable_subreaper()
    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    try:
        time.sleep(30)
        current=inspect()
        if ownership(current['gpus'])!=ownership(before['gpus']) or any(process(int(pid))!=r for pid,r in before['processes'].items()):
            raise RuntimeError('Ownership changed during hold')
        reclaimed=True
        stop_burns(base/'execution',{NATIVE:NATIVE_SHA})
        experiment(base)
        success=True
    except BaseException as failure:
        write_json(base/'failure.json',dict(reason=repr(failure),utc=time.time()))
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        if reclaimed:
            clean_owned_children()
            time.sleep(30)
            require_free(list(range(8)))
            native(base,session=base.name+'_native')
        if guard.stat().st_ino!=inode or guard.read_text()!=marker:
            raise RuntimeError('Guard identity changed')
        guard.unlink()
        write_json(base/'terminal.json',dict(success=success,native_verified=reclaimed,utc=time.time()))
    if not success:
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    base=parser.parse_args().base
    try:
        supervise(base,prepare,run_anchor_arms,FIT/'native-verified.json',
                  'a527b46afce2eaf3bc91bc0f1a28a378cadb4d3750f5ffe7d7074dea517b55a2')
    except Exception as error:
        write_json(base/'failure.json',dict(reason=repr(error),utc=time.time()))
        raise

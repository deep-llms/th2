"""Stop only the verified owned proxy-follow-up supervisor, then await burn handoff."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import subprocess
import time

from scripts.gpu_status import snapshot
from scripts.train_then_burn import GPUS
from scripts.verified_gpu_reclaim import ownership,pidfd_open,pidfd_send_signal,process

SESSION='tjx3-proxy-followup-2500-20261008-a01'
BURN_SESSION='proxy-followup-2500-20261008-a01-final-burn'
ROOT=Path('/mnt/local/_outputs/deep-llms_th2/proxy-followup-2500-20261008-a01')
HOST='thiennh-p6-tjx3-worker-0'


def read(path):
    return json.loads(Path(path).read_text())


def handed_off():
    supervised=ROOT/'supervised'
    receipt=supervised/'supervisor.json'
    burn=supervised/'burn-verified.json'
    if not receipt.is_file() or not burn.is_file():
        return False
    a,b=read(receipt),read(burn)
    return (bool(a.get('finished_at')) and 'burn' in a and
            b.get('collective_progress_verified') is True and
            b.get('session')==BURN_SESSION and
            not Path('/mnt/local/_gpu_guard/DISABLED').exists())


def pane_pid():
    r=subprocess.run(['tmux','display-message','-p','-t',SESSION,'#{pane_pid} #{pane_dead}'],
                     capture_output=True,text=True,check=True)
    pid,dead=r.stdout.split()
    if dead!='0':
        raise RuntimeError('Follow-up pane is not live')
    return int(pid)


def belongs_to(pid,ancestor):
    seen=set()
    while pid>1 and pid not in seen:
        if pid==ancestor:
            return True
        seen.add(pid)
        pid=process(pid)['ppid']
    return False


def inspect_owned():
    import socket
    if socket.gethostname()!=HOST:
        raise RuntimeError('Node identity changed')
    supervised=ROOT/'supervised'
    report=read(supervised/'supervisor.json')
    running=read(supervised/'run/run.json')
    if report.get('training_status')!='running' or running.get('status')!='running':
        raise RuntimeError('Follow-up queue is not running; inspect its terminal state')
    pid=pane_pid()
    identity=process(pid)
    argv=[x.decode(errors='replace') for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
    if (pid<=1 or 'scripts.train_then_burn' not in argv
            or '--output' not in argv or argv[argv.index('--output')+1]!=str(supervised)
            or '--config' not in argv or argv[argv.index('--config')+1]!=str(ROOT/'jobs.json')
            or '--host' not in argv or argv[argv.index('--host')+1]!=HOST):
        raise RuntimeError('Tmux pane is not the expected owned supervisor')
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    if not guard.is_file() or read(guard).get('owner_pid')!=pid:
        raise RuntimeError('GPU guard is not owned by this supervisor')
    gpus=snapshot(GPUS)
    workers=sorted({p for g in gpus for p in g['pids']})
    if len(gpus)!=8 or len(workers)!=8 or any(len(g['pids'])!=1 for g in gpus):
        raise RuntimeError('Expected one current training worker on each B200')
    identities={p:process(p) for p in workers}
    if not all(belongs_to(p,pid) for p in workers):
        raise RuntimeError('A GPU worker is not a descendant of the verified supervisor')
    return pid,identity,gpus,identities


def stop(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    if handed_off():
        record={'status':'already_terminal','at':datetime.now(timezone.utc).isoformat(),
                'previous_supervisor':read(ROOT/'supervised/supervisor.json')}
    else:
        pid,identity,gpus,workers=inspect_owned()
        handle=pidfd_open(pid)
        try:
            again,identity2,gpus2,workers2=inspect_owned()
            if (again!=pid or identity2!=identity or ownership(gpus2)!=ownership(gpus)
                    or workers2!=workers):
                raise RuntimeError('Supervisor/GPU worker identities changed before stop')
            print('VERIFIED_OWNED_SUPERVISOR_STOP',pid,identity,flush=True)
            pidfd_send_signal(handle,signal.SIGTERM)
        finally:
            import os
            os.close(handle)
        record={'status':'stop_requested','at':datetime.now(timezone.utc).isoformat(),
                'supervisor':identity,'workers':workers,'gpus':gpus}
        deadline=time.monotonic()+900
        while time.monotonic()<deadline and not handed_off():
            time.sleep(10)
        if not handed_off():
            raise RuntimeError('Stopped queue has not restored verified communicating burns')
    record['previous_supervisor_final']=read(ROOT/'supervised/supervisor.json')
    record['burn_handoff']=read(ROOT/'supervised/burn-verified.json')
    with output.open('x') as handle:
        json.dump(record,handle,indent=2)
    print('PREVIOUS_QUEUE_TERMINAL_AND_BURN_VERIFIED',record['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    stop(a.output)


if __name__=='__main__':
    main()

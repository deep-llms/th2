"""v17 foreground continuation under the existing Family native supervisor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from efficient_llm.a2_family_v17.common import read, write, Checks, SPEC_SHA
from efficient_llm.a2_family_v17.protocol import runs
from efficient_llm.a2_family_v17.readout import report
from scripts.a2_family_v14_gate import sha, read_lock, verify_ready
from scripts.gpu_status import require_free
from scripts.gpu_process import process


def owner(cfg):
    guard = read('/mnt/local/_gpu_guard/DISABLED')
    expected = process(guard['owner_pid'])
    if socket.gethostname()!=cfg['host'] or guard['output']!=cfg['native_output']:
        raise ValueError('v17 native ownership changed')
    pid=os.getpid()
    while pid>1 and pid!=expected['pid']:pid=process(pid)['ppid']
    if pid!=expected['pid']:raise ValueError('Not native child')


def launch(cfg, config, item, events):
    owner(cfg)
    for attempt in range(3):
        try:
            require_free(list(range(8))); break
        except RuntimeError:
            if attempt == 2: raise
            time.sleep(5)
    name = item['run_id']; root = Path(cfg['results_root'])
    if any(e['name']==name for e in events) or (root/'heals'/name).exists():
        raise ValueError('Duplicate v17 attempt')
    write(root/'progress.json', dict(active=name, events=events))
    log = root/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = str(sock.getsockname()[1])
    command = [sys.executable,'-u','-m','accelerate.commands.launch','--config_file',
               'resources/accelerate_b200.yaml','--main_process_port',port,'-m',
               'efficient_llm.a2_family_v17.heal','--config',str(config),'--run-id',name]
    started = time.monotonic()
    with log.open('x') as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}).returncode
    event = dict(name=name,returncode=rc,seconds=time.monotonic()-started,log=str(log))
    events.append(event)
    with (root/'runs.jsonl').open('a') as f:
        f.write(json.dumps(dict(event,spec_sha256=SPEC_SHA))+'\n')
    write(root/'ledger.json',dict(events=events,node_hours=sum(x['seconds'] for x in events)/3600))
    print('v17_RUN_EVENT',json.dumps(event),flush=True)



def execute(cfg,config,receipt):
    root=Path(cfg['results_root']);owner(cfg)
    from scripts.a2_family_v17_prepare import verify_inputs
    verified=verify_inputs(cfg)
    write(root/'manifests/actual_inputs_at_gpu_admission.json',dict(files=verified))
    events=[];checks=Checks(root/'coordinator_checks.json')
    for item in runs():
        launch(cfg,config,item,events)
        checks.attempt('partial_readout',lambda:report(cfg,events))
    decision=report(cfg,events,final=True)
    from scripts.a2_family_v17_backup import backup
    saved=backup(root)
    write(receipt,dict(execution_complete=True,report_complete=True,backup_verified=True,
        coverage_complete=decision['coverage_complete'],events=events,
        report=dict(path=str(root/'REPORT.md'),sha256=sha(root/'REPORT.md')),
        backup_files=saved['files'],scientific_failures_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();cfg=read(a.config)
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong spec')
    root=Path(cfg['results_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(cfg,a.config,a.receipt)

"""Stop old F9 dispatch, drain its current children, then use native handoff.

Only the explicitly pinned coordinator is stopped. GPU children finish normally;
the unchanged native supervisor owns cancellation/cleanup and restoring its burn.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time
from datetime import datetime,timezone
from scripts.gpu_process import process,pidfd_open,pidfd_send_signal,inspect
from efficient_llm.diagnostic_data import sha256
from run_experiments import write_json


def args(pid):
    return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]


def children(pid):
    out={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            r=process(int(p.name))
            if r['ppid']==pid:
                state=(p/'stat').read_text().split(') ',1)[1].split()[0]
                if state!='Z':out[r['pid']]=r
        except (OSError,ValueError):pass
    return out


def validate(expected):
    root=Path(expected['root']);c=expected['coordinator'];s=expected['supervisor']
    if socket.gethostname()!=expected['host']:raise ValueError('Wrong host')
    for record in (c,s):
        if process(record['pid'])!=record:raise ValueError('Pinned process identity changed')
    if c['ppid']!=s['pid']:raise ValueError('Wrong native owner')
    ca=args(c['pid']);sa=args(s['pid'])
    if 'scripts.a2_family_f9_run' not in ca or str(root/'results/a2_family_20261107/downstream/config_admitted_a02.json') not in ca:
        raise ValueError('Not the authorized F9 coordinator')
    if 'scripts.train_then_burn' not in sa or str(root/'supervised') not in sa:
        raise ValueError('Not the authorized native supervisor')
    if Path('/proc',str(c['pid']),'cwd').resolve()!=root/'source':raise ValueError('Coordinator cwd changed')
    marker=json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())
    if marker['owner_pid']!=s['pid'] or marker['output']!=str(root/'supervised'):raise ValueError('Guard ownership mismatch')
    manifest=root/'source-manifest.json'
    if sha256(manifest)!=expected['source_manifest_sha256']:raise ValueError('Manifest changed')
    for name,h in json.loads(manifest.read_text())['files'].items():
        if sha256(root/'source'/name)!=h:raise ValueError('Frozen source changed')
    gpu=inspect()
    if [g['uuid'] for g in gpu['gpus']]!=expected['gpu_uuids']:raise ValueError('GPU UUID mismatch')
    for g in gpu['gpus']:
        for pid in g['pids']:
            if process(pid)['ppid']!=c['pid']:raise ValueError('GPU worker outside pinned coordinator')
    return gpu


def drain(identity):
    e=json.loads(Path(identity).read_text());root=Path(e['root']);out=root/'scope_v110_drain'
    if out.exists():raise ValueError('Existing drain receipt; do not duplicate')
    gpu=validate(e);c=e['coordinator'];s=e['supervisor']
    cf=pidfd_open(c['pid']);sf=pidfd_open(s['pid'])
    try:
        validate(e);out.mkdir()
        def save(name,v):write_json(out/name,dict(v,spec_sha256=e['spec_sha256'],utc=datetime.now(timezone.utc).isoformat()))
        save('requested.json',dict(identity=e,before=gpu,reason='Owner replaces every-model v1.9 with closed-list v1.10'))
        pidfd_send_signal(cf,signal.SIGSTOP)
        for _ in range(50):
            if Path('/proc',str(c['pid']),'stat').read_text().split(') ',1)[1].split()[0] in ('T','t'):break
            time.sleep(.1)
        else:raise RuntimeError('Coordinator stop not acknowledged')
        save('dispatch_stopped.json',dict(coordinator=c,children=children(c['pid'])))
        print('F9_V19_DISPATCH_STOPPED_CURRENT_CHILDREN_DRAINING',str(out),flush=True)
        deadline=time.monotonic()+3600
        while children(c['pid']) and time.monotonic()<deadline:time.sleep(5)
        pending=children(c['pid'])
        if process(s['pid'])!=s:raise ValueError('Native supervisor identity changed')
        save('native_handoff_requested.json',dict(drained=not pending,remaining_children=pending,
             preserved_complete_scores=[str(p) for p in (root/'results/a2_family_20261107/downstream').glob('*/*/complete.json')]))
        pidfd_send_signal(sf,signal.SIGTERM)
        print('F9_V19_NATIVE_HANDOFF_REQUESTED', 'drained',not pending,flush=True)
    finally:
        os.close(cf);os.close(sf)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a=p.parse_args();drain(a.identity)

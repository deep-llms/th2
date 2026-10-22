"""Six router ablations after a verified successful predecessor/burn handoff.

Runs from an immutable source snapshot; never changes the active queue's code.
The waiting process is CPU-only. The existing supervisor owns GPU reclamation,
acceptance, training/evaluation, and the final burn restoration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_experiments import now, write_json
from scripts.make_bdot_pilot import ROUTER_RUNS
from scripts.bdot_mlp_pipeline import prepare


def read(path):return json.loads(Path(path).read_text())


def predecessor_phase(root):
    """No signals; a missing handoff is a wait, a failed queue is a rejection."""
    supervisor=root/'supervised/supervisor.json'
    if not supervisor.is_file():return 'waiting_for_supervisor'
    state=read(supervisor)
    if state.get('training_status')=='failed' or state.get('handoff_error'):
        raise RuntimeError('Predecessor failed; preserving its output and GPU state')
    if state.get('training_status')!='ok' or not state.get('finished_at'):
        return 'waiting_for_completion'
    if read(root/'supervised/run/run.json').get('status')!='ok':
        raise RuntimeError('Predecessor queue has no successful completion')
    report=read(root/'supervised/run/mlp-results.json')
    if report.get('status')!='ok' or len(report.get('runs',[]))!=6:
        raise RuntimeError('Predecessor scientific result is incomplete')
    if not (root/'supervised/burn-verified.json').is_file():return 'waiting_for_burn_proof'
    return 'verify_burn_handoff'


def verify_snapshot():
    manifest=read(ROOT/'SOURCE_MANIFEST.json')
    for name,value in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Invalid source snapshot member: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=value:
            raise ValueError('Source snapshot changed: '+name)


def report(output,pilot):
    before=Path(read(output/'followup.json')['predecessor'])
    dense=read(pilot/'prepared/runs/D/final_eval.json')
    data=read(pilot/'prepared/runs/D/data_recipe.json')
    records=[]
    for arm,_ in ROUTER_RUNS:
        name=arm+'-s0';root=output/'prepared/runs'/name;run=output/'supervised/run'
        receipt=read(run/(name+'.verified.json'));result=read(run/name/'result.json')
        final=read(root/'final_eval.json')
        assert result['status']==receipt['status']=='ok'
        assert result['global_step']==receipt['global_step']==4768
        assert read(root/'data_recipe.json')==data
        assert (result['num_tokens'],result['num_sequences'])==(dense['eval_num_tokens'],dense['eval_num_sequences'])
        assert math.isfinite(result['loss']) and abs(result['loss']-final['eval_loss'])<1e-5
        base='-'.join(arm.split('-')[:2])
        original=read(before/'prepared/runs'/(base+'-s0')/'final_eval.json')['eval_loss']
        records.append(dict(arm=arm,loss=result['loss'],ppl=result['perplexity'],
                            gap_to_D=result['loss']-dense['eval_loss'],gap_to_parent=result['loss']-original))
    write_json(output/'supervised/run/mlp-results.json',dict(status='ok',steps=4768,runs=records,
        scope='Single-seed router-input/metric factorial; previous same-width arms supply the unchanged controls.'))


def wait(a):
    if socket.gethostname()!=a.host:raise ValueError('Wrong node')
    if not a.predecessor or not a.session or not a.predecessor_burn_script:
        raise ValueError('Explicit predecessor, session and old burn script required')
    verify_snapshot()
    # Exclusive claim before generating jobs prevents a second watcher/launch.
    with (a.output/'wait-owner.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),host=a.host,started_at=now()),f)
    followup=dict(predecessor=str(a.predecessor),pilot=str(a.pilot),source=str(ROOT),
                  session=a.session,host=a.host,deadline_seconds=a.timeout)
    write_json(a.output/'followup.json',followup)
    shutil.copytree(ROOT/'reference',a.output/'reference')
    jobs=prepare(a.output,a.pilot,study='router')
    from scripts.native_baseline_pipeline import burn_gate
    from scripts.gpu_process import inspect
    from scripts.train_then_burn import digest,BURN_HASH,burn_progress
    if digest(a.predecessor_burn_script)!=BURN_HASH:raise ValueError('Unknown predecessor burn source')
    deadline=time.monotonic()+a.timeout
    while True:
        if socket.gethostname()!=a.host:raise ValueError('Node changed')
        if (a.output/'CANCEL').exists():raise RuntimeError('Operator cancelled queued follow-up')
        phase=predecessor_phase(a.predecessor)
        write_json(a.output/'wait-status.json',dict(status='waiting',phase=phase,at=now(),**followup))
        print('ROUTER_QUEUE_WAIT '+phase+' '+now(),flush=True)
        if phase=='verify_burn_handoff':
            live=inspect()
            # Supervisor writes its final receipt just before releasing its guard.
            if not live['guard_disabled']:
                burn_gate(a.predecessor)
                if not burn_progress((a.predecessor/'supervised/burn.log').read_text()):
                    raise RuntimeError('No advancing communicating burn evidence')
                verify_snapshot()
                if shutil.disk_usage(a.output).free<300*1024**3:raise RuntimeError('Insufficient free space')
                write_json(a.output/'preflight.json',inspect())
                subprocess.run([sys.executable,str(ROOT/'scripts/bdot_b200_launch.py'),
                    '--root',str(a.output),'--host',a.host,'--session',a.session,
                    '--jobs-config',str(jobs),'--predecessor-burn-script',str(a.predecessor_burn_script)],
                    cwd=ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
                write_json(a.output/'wait-status.json',dict(status='launched',at=now(),**followup))
                print('ROUTER_QUEUE_LAUNCHED',flush=True)
                return
        if time.monotonic()>=deadline:raise TimeoutError('Predecessor did not hand off within deadline')
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['wait','report'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--predecessor',type=Path);p.add_argument('--predecessor-burn-script',type=Path)
    p.add_argument('--host');p.add_argument('--session');p.add_argument('--timeout',type=int,default=43200)
    a=p.parse_args()
    if a.action=='report':report(a.output,a.pilot);return
    try:wait(a)
    except BaseException as error:
        owner=a.output/'wait-owner.json'
        if owner.is_file() and read(owner).get('pid')==os.getpid():
            write_json(a.output/'wait-status.json',dict(status='failed',at=now(),error=repr(error)))
        raise


if __name__=='__main__':main()

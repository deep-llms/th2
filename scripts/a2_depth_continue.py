"""Continue an existing frozen Depth C experiment with a fresh native supervisor.

The recovery handoff describes either exact known burn identities or currently
free original GPUs. It never clears guards, kills a prior session, or edits source.
"""
import fcntl
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from efficient_llm.a2_depth.common import read,write
from efficient_llm.a2_depth.recovery import coordinator_lock
from scripts.a2_remaining import read_locked
from scripts.a2_quant_io import lock
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest,GUARD_HASH
from scripts.launch_a2_depth_partb import predecessor

GUARD=Path('/mnt/local/_gpu_guard/DISABLED')


def validate_existing(output,plan,commit):
    output=Path(output);record=read(output/'launch.json')
    manifest=read_locked(record['source_manifest']);cfg=read_locked(record['config'])
    if manifest['code_commit']!=commit or cfg['code_commit']!=commit:raise ValueError('Changed continuation commit')
    if digest(plan)!=record['input_plan']['sha256']:raise ValueError('Changed continuation plan')
    read_locked(record['input_plan'])
    frozen=output/'source'
    for rel,sha in manifest['files'].items():
        if digest(frozen/rel)!=sha:raise ValueError('Frozen continuation source changed: '+rel)
    if cfg['source_manifest']!=record['source_manifest'] or Path(cfg['results_root'])!=output/'results/a2_depth_20261031':
        raise ValueError('Changed result/source binding')
    return record,cfg,frozen


def admission(output,record,cfg,handoff):
    if socket.gethostname()!='thiennh-p6-6y3o-worker-0':raise ValueError('Wrong continuation host')
    if handoff['mode']=='burn':
        return predecessor({'predecessor':handoff['predecessor']},output)
    if handoff['mode']!='free':raise ValueError('Unknown recovery handoff mode')
    if GUARD.exists():raise ValueError('Native ownership guard still held')
    if digest(GUARD.parent/'gpu_guard.sh')!=GUARD_HASH:raise ValueError('Unknown guard code')
    live=inspect();old=read_locked(record['initial_inspection'])
    if live['host']!=old['host'] or len(live['gpus'])!=8 or any(g['pids'] for g in live['gpus']):
        raise ValueError('Original GPUs not free; no allocation or signals')
    if [(g['index'],g['uuid']) for g in live['gpus']]!=[(g['index'],g['uuid']) for g in old['gpus']]:
        raise ValueError('GPU identity changed')
    return dict(inspection=live,mode='free',predecessor=cfg['predecessor'])


def continue_run(args):
    output=args.output.resolve()
    with (output/'continuation-launch.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record,cfg,frozen=validate_existing(output,args.plan,args.code_commit)
        # The launcher handling this continuation must itself match the frozen code.
        source=Path(__file__).resolve().parents[1]
        for rel in ('scripts/a2_depth_continue.py','scripts/launch_a2_depth_partc.py'):
            if digest(source/rel)!=digest(frozen/rel):raise ValueError('Use the original frozen continuation implementation')
        history=read(output/'continuations.json').get('attempts',[]) if (output/'continuations.json').exists() else []
        for session in [record['session'],*[v['session'] for v in history]]:
            if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:
                status=subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_dead}'],capture_output=True,text=True,check=True).stdout.strip()
                if status!='1':raise ValueError('Prior supervisor session is still active')
        for session in (args.session,args.session+'_burn'):
            if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('New recovery session already exists')
        with coordinator_lock(cfg['results_root']):pass  # No competing coordinator, even between GPU phases.
        handoff=read(args.recovery_handoff);observed=admission(output,record,cfg,handoff)
        attempt=output/'continuations'/f'attempt{len(history)+1:03d}'
        attempt.mkdir(parents=True,exist_ok=False)
        write(attempt/'admission.json',observed)
        jobs=read_locked(record['jobs']);job_lock=write(attempt/'jobs.json',jobs)
        # Reinspect immediately before the native owner; native pidfd checks still follow.
        observed=admission(output,record,cfg,handoff)
        inspection=write(attempt/'inspection.json',observed['inspection'])
        env={**os.environ,**record['environment']}
        cmd=['env',*[k+'='+v for k,v in record['environment'].items()],sys.executable,'-u','-m','scripts.train_then_burn',
             '--config',job_lock['path'],'--output',str(attempt/'supervised'),'--inspection',inspection['path'],
             '--host',socket.gethostname(),'--burn-session',args.session+'_burn',
             '--predecessor-burn-script',observed['predecessor']['burn_script']]
        write(output/'continuations.json',dict(attempts=history+[dict(session=args.session,root=str(attempt),
             handoff=lock(args.recovery_handoff),config=record['config'],source_manifest=record['source_manifest'])]))
        command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(attempt/'supervisor.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',str(frozen),command],env=env,check=True)
        subprocess.run(['tmux','set-option','-w','-t',args.session,'remain-on-exit','on'],check=True)
        print('A2_DEPTH_PART_C_CONTINUED',attempt,flush=True)

"""Wait for verified inputs, pause exact Depth owner, submit Family then Depth.

Runs from a frozen CPU-only source. Any failure is terminal and preserves all
receipts. Never repeats a pause or launch, repairs infrastructure, or signals
unknown work. The already authorized native supervisor handles cancellation.
"""
import argparse
import fcntl
from pathlib import Path
import socket
import subprocess
import sys
import time
from efficient_llm.a2_family.common import read,write,SPEC_SHA
from scripts.a2_family_plan import inputs,pause_request,finalize
from scripts.launch_a2_family import verify_inputs,verify_proof
from scripts.a2_family_overlap import audit
from scripts.a2_remaining import read_locked


def execute(common,mixture,depth,source,output,family_output,session,commit):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    def stage(name,**extra):write(output/'progress.json',dict(stage=name,**extra))
    try:
        verify_proof(source)
        if Path(family_output).exists():raise ValueError('Family output already exists')
        for name in (session,session+'_burn'):
            if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
                raise ValueError('Family session already exists')
        stage('waiting_complete_inputs',Depth_continues=True)
        while not all((Path(p)/'complete.json').is_file() for p in (common,mixture)):
            # A producer's explicit failure is not an instruction to retry it.
            for folder in (common,mixture):
                p=Path(folder)/'complete.json'
                if p.is_file() and read(p).get('status')!='complete':raise ValueError('Input preparation failed: '+str(p))
            time.sleep(30)
        cfg=inputs(common,mixture,depth,source/'resources')
        stage('verifying_actual_input_bytes',Depth_continues=True)
        verify_inputs(cfg,paused=False)
        stage('whole_pool_overlap_audit',Depth_continues=True)
        audit(cfg,output/'overlap.json')
        from scripts.a2_quant_io import lock
        cfg['overlap_preflight']=lock(output/'overlap.json')
        write(output/'inputs.json',cfg)
        verify_proof(source)
        request=pause_request(cfg)
        request_lock=write(output/'pause-request.json',request)
        stage('waiting_safe_checkpoint',arm=request['arm'])
        subprocess.run([sys.executable,'-u','-m','scripts.a2_family_pause_depth',
            '--request',request_lock['path'],'--output',str(output/'pause')],cwd=source,check=True)
        pause=read(output/'pause/terminal.json')
        if pause['state']!='paused' or not pause['handoff_verified']:raise ValueError('Pause/handoff not verified')
        stage('binding_actual_native_handoff',saved_step=pause['global_step'])
        cfg=finalize(cfg,output/'pause');plan=write(output/'plan.json',cfg)
        stage('submitting_family_then_depth',saved_step=pause['global_step'])
        subprocess.run([sys.executable,'-u','-m','scripts.launch_a2_family','--plan',plan['path'],
            '--output',str(family_output),'--session',session,'--code-commit',commit],cwd=source,check=True)
        launch=read(Path(family_output)/'launch.json')
        if not launch['automatic_Depth_resume']:raise ValueError('Depth not queued')
        read_locked(launch['config']);read_locked(launch['jobs'])
        write(output/'complete.json',dict(status='submitted',family_root=str(family_output),
            session=session,automatic_Depth_resume=True,training_start_verified=False))
        stage('native_queue_submitted',training_start_verified=False)
    except BaseException as error:
        write(output/'failure.json',dict(error=repr(error),action='Preserve evidence; no automatic retry or unknown-process intervention'))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('common','mixture','depth','output','family-output','session','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
    source=Path(__file__).resolve().parents[1]
    with (source.parent/'admission.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(a.common,a.mixture,a.depth,source,a.output,a.family_output,a.session,a.code_commit)

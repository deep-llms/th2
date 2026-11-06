"""Continue an existing frozen Depth C experiment with a fresh native supervisor.

The recovery handoff describes either exact known burn identities or currently
free original GPUs. It never clears guards, kills a prior session, or edits source.
"""
from scripts.a2_quant_io import write as write_native
from run_experiments import load_jobs

import fcntl
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import shutil
from efficient_llm.a2_depth.common import read,write
from efficient_llm.a2_depth.recovery import coordinator_lock
from scripts.a2_remaining import read_locked
from scripts.a2_quant_io import lock
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest,GUARD_HASH
from scripts.launch_a2_depth_partb import predecessor

GUARD=Path('/mnt/local/_gpu_guard/DISABLED')
PATCH_FILES={'efficient_llm/a2/measure.py','efficient_llm/a2_depth/heal_long.py'}


def prepare_recovery_source(source,frozen,attempt,patch_path,original_manifest):
    """Version an explicit failure-path fix without changing checkpoint identity."""
    patch=read(patch_path)
    if patch['original_source_manifest']!=original_manifest or patch['status']!='PASS':
        raise ValueError('Recovery proof not bound to original source')
    if set(patch['patches'])!=PATCH_FILES or not patch.get('checks') or not all(x['passed'] for x in patch['checks']):
        raise ValueError('Incomplete recovery validation')
    files=read_locked(original_manifest)['files']
    for rel,item in patch['patches'].items():
        if rel not in files or digest(frozen/rel)!=item['original_sha256'] or digest(source/rel)!=item['patched_sha256']:
            raise ValueError('Recovery source mismatch: '+rel)
    if not {'scripts/a2_depth_continue.py','scripts/launch_a2_depth_partc.py'}.issubset(patch['implementation']):
        raise ValueError('Recovery launcher hashes missing')
    for rel,sha in patch['implementation'].items():
        if digest(source/rel)!=sha:raise ValueError('Recovery launcher/proof mismatch: '+rel)
    patched=attempt/'source';patched.mkdir()
    manifest={}
    for rel,sha in files.items():
        src=source/rel if rel in PATCH_FILES else frozen/rel
        dest=patched/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        actual=digest(dest)
        if actual!=(patch['patches'][rel]['patched_sha256'] if rel in PATCH_FILES else sha):raise ValueError('Recovery copy mismatch')
        manifest[rel]=actual
    proof=write(attempt/'execution-source-manifest.json',dict(original_source_manifest=original_manifest,
        recovery_patch=lock(patch_path),files=manifest,checkpoint_recipe_unchanged=True))
    return patched,proof


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
    host=cfg.get('node_restart',{}).get('host','thiennh-p6-6y3o-worker-0')
    if host not in ('thiennh-p6-6y3o-worker-0','thiennh-p6-wc69-worker-0') or socket.gethostname()!=host:
        raise ValueError('Wrong continuation host')
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
    burn=cfg['predecessor']
    if 'node_restart' in cfg:
        # Free GPUs need no historical B burn on the replacement node.
        # The immutable experiment's own burn is already source-hash verified.
        burn=dict(burn_script=str(Path(output)/'source/resources/llm_pretrain_burn.py'))
    return dict(inspection=live,mode='free',predecessor=burn)


def continue_run(args):
    output=args.output.resolve()
    with (output/'continuation-launch.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record,cfg,frozen=validate_existing(output,args.plan,args.code_commit)
        # The launcher handling this continuation must itself match the frozen code.
        source=Path(__file__).resolve().parents[1]
        patch_path=getattr(args,'recovery_patch',None)
        if not patch_path:
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
        execution_source=frozen;execution_manifest=None
        if patch_path:
            execution_source,execution_manifest=prepare_recovery_source(source,frozen,attempt,patch_path,record['source_manifest'])
        write(attempt/'admission.json',observed)
        jobs=read_locked(record['jobs']);job_lock=write_native(attempt/'jobs.json',jobs)
        load_jobs(Path(job_lock['path']))
        # Reinspect immediately before the native owner; native pidfd checks still follow.
        observed=admission(output,record,cfg,handoff)
        inspection=write(attempt/'inspection.json',observed['inspection'])
        env={**os.environ,**record['environment']}
        run_environment=dict(record['environment'])
        if execution_manifest:
            run_environment.update(PYTHONPATH=str(execution_source),A2_DEPTH_RECOVERY_PATCH=execution_manifest['path'])
        env.update(run_environment)
        cmd=['env',*[k+'='+v for k,v in run_environment.items()],sys.executable,'-u','-m','scripts.train_then_burn',
             '--config',job_lock['path'],'--output',str(attempt/'supervised'),'--inspection',inspection['path'],
             '--host',socket.gethostname(),'--burn-session',args.session+'_burn',
             '--predecessor-burn-script',observed['predecessor']['burn_script']]
        write(output/'continuations.json',dict(attempts=history+[dict(session=args.session,root=str(attempt),
             handoff=lock(args.recovery_handoff),config=record['config'],source_manifest=record['source_manifest'],
             execution_source_manifest=execution_manifest)]))
        command='exec '+shlex.join(cmd)+' >'+shlex.quote(str(attempt/'supervisor.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',str(execution_source),command],env=env,check=True)
        subprocess.run(['tmux','set-option','-w','-t',args.session,'remain-on-exit','on'],check=True)
        print('A2_DEPTH_PART_C_CONTINUED',attempt,flush=True)

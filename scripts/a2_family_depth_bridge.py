"""Native queued handoff to the exact paused Depth config/source, no new owner."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from efficient_llm.a2_family.common import read,write,sha
from scripts.a2_remaining import read_locked
from scripts.gpu_status import require_free


def validate(cfg):
    pause=read_locked(cfg['depth_pause']);saved=read_locked(pause['saved_state'])
    checkpoint=Path(saved['checkpoint']);original=Path(saved['original'])
    for name,digest in saved['hashes'].items():
        if sha(checkpoint/name)!=digest or sha(original/name)!=digest:raise ValueError('Paused native state changed or missing: '+name)
    record=read_locked(cfg['depth_launch']);plan=read_locked(record['config'])
    for directory,lock in [(Path(cfg['depth_root'])/'source',record['source_manifest']),(Path(cfg['depth_execution_source']),cfg['depth_execution_manifest'])]:
        for name,digest in read_locked(lock)['files'].items():
            if sha(directory/name)!=digest:raise ValueError('Frozen Depth source changed: '+name)
    from efficient_llm.a2_depth.resume import select_checkpoint
    from efficient_llm.a2_depth.heal_long import run_identity
    identity=run_identity(plan,pause['arm'])
    trainer=Path(plan['results_root'])/'heals'/f'ST1-dclm-{pause["arm"]}-S1-12.8B-lr1e-3'/'trainer'
    chosen,rejected=select_checkpoint(trainer,identity,8)
    if chosen!=original:raise ValueError('Native resume selection differs from pinned pause state')
    return record,plan,dict(step=saved['global_step'],arm=pause['arm'],checkpoint=str(chosen),rejected=rejected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();cfg=read(a.config)
    # Family report and byte-verified scientific backup must precede resumption.
    backup=read(Path(cfg['results_root'])/'backup/verified.json')
    if not backup['verified']:raise ValueError('Family scientific backup not verified')
    for item in backup['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Family backup changed')
    record,plan,resume=validate(cfg);require_free(list(range(8)))
    source=Path(cfg['depth_execution_source']);environment={**os.environ,**record['environment'],'PYTHONPATH':str(source),'A2_DEPTH_RECOVERY_PATCH':cfg['depth_execution_manifest']['path']}
    write(Path(a.receipt).parent/'depth-resume-admission.json',dict(**resume,config=record['config'],source_manifest=cfg['depth_execution_manifest'],same_native_owner=True))
    command=[sys.executable,'-u','-m','scripts.a2_depth_partc_run','--config',record['config']['path'],'--receipt',a.receipt]
    result=subprocess.run(command,cwd=source,env=environment)
    if result.returncode:raise RuntimeError('Depth continuation returned '+str(result.returncode))

if __name__=='__main__':main()

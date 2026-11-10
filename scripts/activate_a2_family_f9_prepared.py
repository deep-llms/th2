"""Activate already verified F9 preparation after a pre-GPU manifest failure.

Preserves original frozen source, inputs, failed jobs manifest and receipts.
Only new config/jobs/admission records are written; native supervisor handles
all physical GPU ownership checks, exact burn reclaim and final handoff.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from eval.a2_family_f9 import SPEC_SHA,read,write
from scripts.a2_family_f9_run import locked
from scripts.launch_a2_family_f9 import predecessor,save_jobs
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256


def validate_prepared(root,ready_sha,attestation_sha,commit):
    root=Path(root);result=root/'results/a2_family_20261107/downstream'
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Explicit source commit required')
    for name in ('supervised','launch.json','activation_a02.json','jobs_a02.json'):
        if (root/name).exists():raise ValueError('Native launch may already exist; refuse duplicate: '+name)
    ready=locked(dict(path=str(result/'preparation/ready.json'),sha256=ready_sha))
    if not ready['cpu_ready'] or ready['flags'] or ready['models']!=189:raise ValueError('Preparation is not the reviewed complete admission')
    cfg=locked(ready['config'])
    if cfg['spec_sha256']!=SPEC_SHA or cfg['downstream_root']!=str(result):raise ValueError('Preparation identity mismatch')
    attestation=locked(dict(path=str(result/'SOURCE_COMMIT_ATTESTATION.json'),sha256=attestation_sha))
    if not attestation['verified'] or attestation['actual_source_commit']!=commit or not all(attestation['checks'].values()):
        raise ValueError('Source commit attestation invalid')
    if attestation['source_manifest_sha256']!=cfg['source_manifest']['sha256']:raise ValueError('Attestation does not bind this source')
    manifest=locked(cfg['source_manifest'])
    if set(manifest['files'])!=set(attestation['checks']):raise ValueError('Attestation source coverage differs')
    for name,h in manifest['files'].items():
        if sha256(root/'source'/name)!=h:raise ValueError('Frozen source changed: '+name)
    for item in cfg['locked_inputs']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Locked data changed')
    if cfg.get('opensci_cx_root'):
        from efficient_llm.a2.data import PackedRows
        PackedRows(cfg['opensci_cx_root'],'final',verify=True)
    inventory=locked(cfg['inventory'])
    if len(inventory['models'])!=189 or inventory['hash_failures']:raise ValueError('Actual inventory admission differs')
    for common in cfg['benchmarks'].values():
        for key in ('prepared','input_spec','manifest','tokenizer_manifest'):
            if key in common:locked(common[key])
    # Preserve the failed native manifest as the specific reason for recovery.
    failed=read(root/'jobs.json')
    if set(failed)!={'jobs','spec_sha256'} or failed['spec_sha256']!=SPEC_SHA:
        raise ValueError('Not the diagnosed pre-GPU manifest failure')
    cfg['code_commit']=commit
    cfg['source_commit_attestation']=lock(result/'SOURCE_COMMIT_ATTESTATION.json')
    return cfg


def activate(root,ready_sha,attestation_sha,commit,session):
    source=Path(__file__).resolve().parents[1]
    proof=read(source/'resources/a2_family_f9_cpu_admission_20261110.json')
    if proof['status']!='PASS' or not proof['launch_ready']:raise ValueError('CPU admission missing')
    for name,h in proof['files'].items():
        if sha256(source/name)!=h:raise ValueError('Changed admitted activation source: '+name)
    root=Path(root);cfg=validate_prepared(root,ready_sha,attestation_sha,commit)
    if socket.gethostname()!=cfg['host']:raise ValueError('Wrong node')
    for old in ('a2_family_f9_20261110_a01',session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',old],capture_output=True).returncode==0:raise ValueError('Existing F9 session')
    for p in Path('/proc').glob('[0-9]*'):
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(v in args for v in (b'scripts.launch_a2_family_f9',b'scripts.a2_family_f9_prepare',b'scripts.a2_family_f9_run')):
            raise ValueError('F9 preparation/queue still live; no duplicate activation')
    result=Path(cfg['downstream_root']);fresh=predecessor(cfg)
    config=write(result/'config_admitted_a02.json',cfg)
    job=save_jobs(root/'jobs_a02.json',config['path'],result)
    inspection=write(root/'inspection_a02.json',fresh)
    command=[sys.executable,'-u','-m','scripts.train_then_burn','--config',job['path'],
        '--output',str(root/'supervised'),'--inspection',inspection['path'],'--host',cfg['host'],
        '--burn-session',session+'_burn','--predecessor-burn-script',cfg['predecessor_burn_script']]
    write(result/'ADMISSION_CORRECTIONS.json',dict(original_config=lock(result/'config.json'),
        corrected_config=config,original_jobs=lock(root/'jobs.json'),corrected_jobs=job,
        flags=['Original launch had empty commit metadata; all253 source files attested against Git2a74421',
               'Original native jobs manifest had extra spec_sha256 and failed before GPU reclaim; preserved'],
        no_training_or_scoring_repeated=True,source_unchanged=True))
    write(root/'activation_a02.json',dict(config=config,jobs=job,inspection=inspection,session=session,source_commit=commit))
    shell='exec '+shlex.join(command)+' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
    env={**os.environ,'PYTHONPATH':str(root/'source'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'offline'}
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root/'source'),shell],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    print('F9_PREPARED_NATIVE_QUEUE_SUBMITTED',str(root),session,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','ready-sha256','attestation-sha256','source-commit','session'):p.add_argument('--'+k,required=True)
    a=p.parse_args();activate(a.root,a.ready_sha256,a.attestation_sha256,a.source_commit,a.session)

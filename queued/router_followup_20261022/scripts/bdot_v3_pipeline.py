"""Prepare the approved v3 batch, gate it on GPU checks, and use the existing supervisor."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def prepare(output,pilot,no_reuse,random_run):
    previous=json.loads((pilot/'prepared/P1-A.train.json').read_text())
    accelerator=ROOT/'resources/accelerate_b200.yaml'
    prepared=output/'prepared'
    subprocess.run([sys.executable,str(ROOT/'scripts/make_bdot_pilot.py'),'--study','v3',
        '--output',str(prepared),'--tokenizer',previous['tokenizer_name'],
        '--data',previous['data_dir'],'--eval-data',previous['eval_data_dir'],
        '--manifest',previous['validation_manifest'],'--revision',previous['dataset_revision'],
        '--accelerate-config',str(accelerator),'--expert-backend','grouped',
        '--preprocessing-workers',str(previous['preprocessing_num_workers'])],check=True)
    jobs=json.loads((prepared/'jobs.json').read_text())['jobs']
    # Reusing the prior recipe is a hard contract, including data order/cache layout.
    operational={'config_name','output_dir','seed','keep_checkpoint_steps'}
    for path in prepared.glob('*.train.json'):
        new=json.loads(path.read_text())
        changes={k:(previous.get(k),new.get(k)) for k in previous.keys()|new.keys()
                 if k not in operational and previous.get(k)!=new.get(k)}
        if changes:raise ValueError('v3 changes unrelated training settings: '+str(changes))
    run='{run_dir}'
    verification=output/'supervised/run/checks/summary.json'
    checks=dict(name='v3_acceptance',gpus=list(range(8)),timeout_seconds=7200,
        argv=['{python}',str(ROOT/'scripts/bdot_v3_verify.py'),'--output',run+'/checks',
            '--reference',str(output/'reference'),'--pilot',str(pilot),'--no-reuse',str(no_reuse),
            '--random-run',str(random_run),'--accelerate-config',str(accelerator)],
        required_outputs=[dict(path='checks/summary.json',json_equals=dict(status='passed'))])
    summary=dict(name='v3_report',timeout_seconds=300,
        argv=['{python}',str(ROOT/'scripts/bdot_v3_report.py'),'--prepared',str(prepared),
            '--run-dir',run,'--verification',str(verification),'--output',run+'/v3-results.json',
            '--pilot',str(pilot),'--random-run',str(random_run)],
        required_outputs=[dict(path='v3-results.json',json_equals=dict(status='ok'))])
    path=output/'gated-jobs.json'
    with path.open('x') as handle:json.dump(dict(jobs=[checks,*jobs,summary]),handle,indent=2)
    from run_experiments import load_jobs
    load_jobs(path)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','predecessor','pilot','no-reuse','random-run'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--session',required=True)
    a=p.parse_args()
    if socket.gethostname()!=a.host:raise ValueError('Wrong node')
    from scripts.native_baseline_pipeline import burn_gate
    from scripts.gpu_process import inspect
    from scripts.train_then_burn import digest,GUARD_HASH,BURN_HASH,burn_progress
    try:
        initial=json.loads((a.output/'initial-preflight.json').read_text())
        if initial['live']['host']!=a.host:raise ValueError('Wrong preflight node')
        for path,expected in initial['reference_hashes'].items():
            if digest(a.output/'reference'/path)!=expected:raise ValueError('Frozen v2 source changed')
        burn_gate(a.predecessor)
        assert burn_progress((a.predecessor/'supervised/burn.log').read_text())
        if shutil.disk_usage(a.output).free<300*1024**3:raise ValueError('Less than 300 GiB free')
        jobs=prepare(a.output,a.pilot,a.no_reuse,a.random_run)
        burn_gate(a.predecessor)
        assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
        assert digest(ROOT/'resources/llm_pretrain_burn.py')==BURN_HASH
        live=inspect()
        assert live['host']==a.host and len(live['gpus'])==8 and all('B200' in g['name'] for g in live['gpus'])
        with (a.output/'preflight.json').open('x') as handle:json.dump(live,handle,indent=2)
        subprocess.run([sys.executable,str(ROOT/'scripts/bdot_b200_launch.py'),
            '--root',str(a.output),'--host',a.host,'--session',a.session,'--jobs-config',str(jobs)],
            check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
        print('V3_GATED_QUEUE_SUBMITTED',flush=True)
    except BaseException as error:
        (a.output/'preparation-failed.json').write_text(json.dumps(dict(status='failed',error=repr(error))))
        raise


if __name__=='__main__':main()

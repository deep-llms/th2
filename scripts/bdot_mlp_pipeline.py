"""Six approved fixed-attention MLP runs, behind numerical/resume/recipe gates."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.make_bdot_pilot import MLP_RUNS


def read(path):return json.loads(Path(path).read_text())


def prepare(output, pilot):
    previous=read(pilot/'prepared/P1-A.train.json')
    prepared=output/'prepared'
    accelerator=ROOT/'resources/accelerate_b200.yaml'
    subprocess.run([sys.executable,str(ROOT/'scripts/make_bdot_pilot.py'),'--study','mlp',
        '--output',str(prepared),'--tokenizer',previous['tokenizer_name'],'--data',previous['data_dir'],
        '--eval-data',previous['eval_data_dir'],'--manifest',previous['validation_manifest'],
        '--revision',previous['dataset_revision'],'--accelerate-config',str(accelerator),
        '--expert-backend','grouped','--preprocessing-workers',str(previous['preprocessing_num_workers'])],check=True)
    for arm,_ in MLP_RUNS:
        new=read(prepared/(arm+'-s0.train.json'))
        changes={k:(previous.get(k),new.get(k)) for k in previous.keys()|new.keys()
                 if k not in {'config_name','output_dir','keep_checkpoint_steps'} and previous.get(k)!=new.get(k)}
        if changes:raise ValueError('Unrelated training recipe changes: '+str(changes))
    jobs=read(prepared/'jobs.json')['jobs']
    checks=dict(name='mlp_acceptance',gpus=list(range(8)),timeout_seconds=5400,
        argv=['{python}',str(ROOT/'scripts/bdot_mlp_verify.py'),'--output','{run_dir}/checks',
              '--reference',str(output/'reference'),'--pilot',str(pilot),'--prepared',str(prepared),
              '--accelerate-config',str(accelerator)],
        required_outputs=[dict(path='checks/summary.json',json_equals=dict(status='passed'))])
    report=dict(name='mlp_report',timeout_seconds=300,
        argv=['{python}',str(ROOT/'scripts/bdot_mlp_pipeline.py'),'report','--output',str(output),'--pilot',str(pilot)],
        required_outputs=[dict(path='mlp-results.json',json_equals=dict(status='ok'))])
    path=output/'gated-jobs.json'
    with path.open('x') as f:json.dump(dict(jobs=[checks,*jobs,report]),f,indent=2)
    from run_experiments import load_jobs
    load_jobs(path)
    return path


def report(output,pilot):
    run=output/'supervised/run'
    dense=read(pilot/'prepared/runs/D/final_eval.json')
    reference_data=read(pilot/'prepared/runs/D/data_recipe.json')
    records=[]
    for arm,_ in MLP_RUNS:
        name=arm+'-s0'
        root=output/'prepared/runs'/name
        result=read(run/name/'result.json')
        receipt=read(run/(name+'.verified.json'))
        final=read(root/'final_eval.json')
        assert result['status']==receipt['status']=='ok'
        assert result['global_step']==receipt['global_step']==4768
        assert (result['num_sequences'],result['num_tokens'])==(dense['eval_num_sequences'],dense['eval_num_tokens'])
        assert read(root/'data_recipe.json')==reference_data
        assert abs(result['loss']-final['eval_loss'])<1e-5 and math.isfinite(result['loss'])
        assert abs(result['perplexity']-math.exp(result['loss']))<1e-8
        records.append(dict(arm=arm,seed=0,loss=result['loss'],ppl=result['perplexity'],
                            gap_to_D=result['loss']-dense['eval_loss']))
    comparisons=[dict(fraction=f,learned_gain=records[i+1]['loss']-records[i]['loss'])
                 for f,i in ((.25,0),(.5,2),(1.,4))]
    value=dict(status='ok',steps=4768,baseline_loss=dense['eval_loss'],runs=records,pairs=comparisons,
        scope='Exploratory single-seed screen; matching routing curricula, data, active width and backbone parameters.')
    with (run/'mlp-results.json').open('x') as f:json.dump(value,f,indent=2)
    print('MLP_RESULTS '+json.dumps(value),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['launch','prepare','report'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--predecessor',type=Path);p.add_argument('--host');p.add_argument('--session')
    a=p.parse_args()
    if a.action=='report':report(a.output,a.pilot);return
    if a.action=='prepare':prepare(a.output,a.pilot);return
    if not a.host or not a.session or not a.predecessor or socket.gethostname()!=a.host:
        raise ValueError('Explicit matching host, predecessor and session required')
    from scripts.native_baseline_pipeline import burn_gate
    from scripts.gpu_process import inspect
    from scripts.train_then_burn import digest,burn_progress,GUARD_HASH,BURN_HASH
    initial=read(a.output/'initial-preflight.json')
    assert initial['status']=='ok' and initial['live']['host']==a.host
    for path,value in initial['training_code_sha256'].items():
        if path.startswith('bdot/'):
            assert digest(a.output/'reference'/path)==value,path
    burn_gate(a.predecessor)
    assert burn_progress((a.predecessor/'supervised/burn.log').read_text())
    assert shutil.disk_usage(a.output).free>300*1024**3
    jobs=prepare(a.output,a.pilot)
    burn_gate(a.predecessor)
    assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
    assert digest(ROOT/'resources/llm_pretrain_burn.py')==BURN_HASH
    live=inspect()
    assert live['host']==a.host and len(live['gpus'])==8 and all('B200' in g['name'] for g in live['gpus'])
    with (a.output/'preflight.json').open('x') as f:json.dump(live,f,indent=2)
    subprocess.run([sys.executable,str(ROOT/'scripts/bdot_b200_launch.py'),
        '--root',str(a.output),'--host',a.host,'--session',a.session,'--jobs-config',str(jobs)],
        check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    print('MLP_GATED_QUEUE_SUBMITTED',flush=True)


if __name__=='__main__':main()

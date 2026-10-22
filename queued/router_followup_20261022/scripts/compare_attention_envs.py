"""Bounded comparisons on GPUs allocated by train_then_burn; no installations."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_status import require_free

CANDIDATES=[('fa4','train','fa4'), ('fa2_stock','train_fa2_t28','fa2'),
            ('fa2_fixed','train_fa2','fa2')]


def choose(results):
    """Prefer FA4 on ties; require >5% and separation of repeat ranges to switch."""
    valid=[r for r in results if r['eligible']]
    if not valid: raise RuntimeError('No numerically valid environment')
    winner=valid[0]
    for candidate in valid[1:]:
        def times(r,model): return [x['seconds_per_update'] for x in r['benchmarks'] if x['model']==model]
        wn,cn=times(winner,'native'),times(candidate,'native')
        wr,cr=times(winner,'routed'),times(candidate,'routed')
        if (statistics.mean(cn) < .95*statistics.mean(wn) and max(cn) < min(wn)
                and statistics.mean(cr) <= 1.02*statistics.mean(wr)):
            winner=candidate
    return winner['name']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accelerate-config',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    inventory=json.loads(a.inventory.read_text())
    results=[]
    def execute(name,py,argv,gpus,receipt):
        require_free(list(range(8)))
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
             'OMP_NUM_THREADS':'1','PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'}
        start=time.time();print('ATTENTION_CHECK_START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            run=subprocess.run([py,*map(str,argv)],cwd=ROOT,env=env,stdout=log,
                               stderr=subprocess.STDOUT,timeout=1800)
        data=json.loads(receipt.read_text()) if receipt.is_file() else {}
        record=dict(name=name,returncode=run.returncode,seconds=time.time()-start,
                    passed=run.returncode==0 and data.get('status') in ('passed','completed'),receipt=data)
        print('ATTENTION_CHECK_RESULT '+json.dumps({k:v for k,v in record.items() if k!='receipt'}),flush=True)
        require_free(list(range(8)))
        return record
    for name,envname,backend in CANDIDATES:
        py=f'/mnt/local/conda-py311/envs/{envname}/bin/python3.11'
        runtime=inventory['environments'].get(envname,{})
        row=dict(name=name,env=envname,python=py,backend=backend,runtime=runtime,
                 eligible=True,checks=[],benchmarks=[])
        if 'sm_100' not in (runtime.get('compiled_archs') or ''):
            row.update(eligible=False,reason='Missing verified SM100 runtime')
        if name=='fa2_stock' and (runtime.get('cuda') not in ('12.8','12.9') or runtime.get('packages',{}).get('torch','').split('+')[0]!='2.8.0'):
            row.update(eligible=False,reason='Older runtime is not the requested B200-compatible torch 2.8')
        if row['eligible']:
            for kind in ('kernel','native','routed'):
                receipt=a.output/f'{name}_{kind}.json'
                if kind=='kernel': argv=['tests/ml/attention_kernel_probe.py','--backend',backend,'--output',receipt]
                elif kind=='native': argv=['tests/ml/fa4_gpu_probe.py',receipt,'--backend',backend]
                else: argv=['tests/bdot/gpu_probe.py','--routing-only','--stable-router-bias','--'+backend,'--output',receipt]
                check=execute(f'{name}_{kind}',py,argv,[0],receipt);row['checks'].append(check)
                if not check['passed']:
                    row.update(eligible=False,reason=f'{kind} numerical check failed');break
        results.append(row)
        (a.output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    for repeat in range(2):
        candidates=results if repeat==0 else list(reversed(results))
        for row in candidates:
            if not row['eligible']:continue
            for model,cfg,extra in [('native','dense_native',[]),('routed','P1-B',['--progress','.7','--router-prior','0'])]:
                name=f"{row['name']}_{model}_r{repeat+1}";receipt=a.output/(name+'.json')
                argv=['-m','accelerate.commands.launch','--config_file',a.accelerate_config,
                      '--main_process_port','0','scripts/benchmark_bdot.py','--config',f'configs/bdot/{cfg}.json',
                      '--output',receipt,'--device','cuda','--batch-size','16','--accumulation','4',
                      '--length','1024','--document-length','128','--warmup','5','--steps','30',
                      '--bf16','--attention-backend',row['backend'],*extra]
                check=execute(name,row['python'],argv,list(range(8)),receipt)
                if not check['passed']:
                    row.update(eligible=False,reason=f'{name} complete-update benchmark failed');break
                row['benchmarks'].append(dict(model=model,repeat=repeat+1,
                    seconds_per_update=check['receipt']['seconds_per_update'],
                    peak_allocated_bytes=check['receipt']['peak_allocated_bytes']))
                (a.output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    selected=choose(results)
    winner=next(r for r in results if r['name']==selected)
    summary=dict(status='passed',selected=selected,python=winner['python'],backend=winner['backend'],
        policy='Numerical eligibility; FA4 wins ties. Switch only for >5% native speed gain, disjoint repeat ranges, routed regression <=2%.',
        results=results,training_resume_verification_pending=True)
    (a.output/'selection.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('ATTENTION_ENV_SELECTED '+json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()

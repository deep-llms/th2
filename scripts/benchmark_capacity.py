"""Bounded eight-GPU capacity probes, supervised externally by train_then_burn.

Each attempt runs the actual Trainer in a fresh process and fresh output folder.
Only a typed CUDA OOM receipt permits continuing after failure. No production
checkpoint is read or written. Search results are empirical, not OOM guarantees.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_experiments import now, stop_owned_process, write_json
from scripts.gpu_status import require_free
from scripts.train_then_burn import GPUS, enable_subreaper, clean_owned_children


def boundary(probe, start=16, ceiling=64):
    """Find the largest passing integer under the usual monotone-memory assumption."""
    if not probe(start):
        raise ValueError('Starting microbatch must pass')
    low=start
    high=min(low*2,ceiling)
    while probe(high):
        low=high
        if low==ceiling:
            return dict(maximum_tested=low,first_failing=None,bounded=True)
        high=min(high*2,ceiling)
    while high-low>1:
        middle=(low+high)//2
        if probe(middle):
            low=middle
        else:
            high=middle
    return dict(maximum_tested=low,first_failing=high,bounded=False)


def inspect_attempt(path, returncode, variant):
    receipts=list(path.glob('capacity-oom-rank*.json'))
    if returncode:
        if not receipts:
            raise RuntimeError(f'{variant}: unexpected failure {returncode}; inspect attempt log')
        for receipt in receipts:
            value=json.loads(receipt.read_text())
            if value.get('status')!='cuda_oom' or value.get('variant')!=variant or value.get('rank') not in GPUS:
                raise ValueError('Invalid CUDA OOM receipt')
        return dict(status='cuda_oom',returncode=returncode,receipts=[p.name for p in receipts])
    if receipts:
        raise ValueError('Successful attempt contains OOM receipt')
    ranks=[json.loads((path/f'benchmark-rank{i}.json').read_text()) for i in GPUS]
    cfg=json.loads((path/'train_config.json').read_text())
    result=json.loads((path/'result.json').read_text())
    assert result['global_step']==18 and result['status']=='stopped'
    assert all(r['status']=='ok' and r['rank']==i and r['world_size']==8 and r['variant']==variant
               and [s['step'] for s in r['steps']]==list(range(1,19))
               and r['first_batches']==cfg['training']['gradient_accumulation_steps'] for i,r in enumerate(ranks))
    rows=sorted(row for r in ranks for row in r['input_rows'])
    assert len(rows)*2048==cfg['tokens_per_update']
    wall=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(5,18)]
    return dict(status='ok',seconds_per_update=statistics.mean(wall),stdev_seconds=statistics.stdev(wall),
                tokens_per_update=cfg['tokens_per_update'],tokens_per_second=cfg['tokens_per_update']/statistics.mean(wall),
                allocated_peak_gib=max(r['allocated_peak_bytes'] for r in ranks)/2**30,
                reserved_peak_gib=max(r['reserved_peak_bytes'] for r in ranks)/2**30,
                reserved_headroom_gib=min(r['device_total_bytes']-r['reserved_peak_bytes'] for r in ranks)/2**30,
                input_rows=rows,train_fingerprint=cfg['train_fingerprint'],eval_fingerprint=cfg['eval_fingerprint'],
                validations=[r['validation'] for r in ranks])


def run(root, recipe, accelerate_config):
    root.mkdir(exist_ok=False,parents=True)
    enable_subreaper()
    base=json.loads(recipe.read_text())
    results={};common_inputs=None;fingerprints=None
    def attempt(arm, lm=True, aux=True, batch=16, layers=False):
        nonlocal common_inputs,fingerprints
        name=f'{arm}-layers{int(layers)}-lm{int(lm)}-aux{int(aux)}-b{batch}'
        if name in results:
            return results[name]
        require_free(GPUS)
        accum=64//batch if batch in (16,32,64) else 1
        cfg={**base,'arm':arm,'checkpoint_layers':layers,'lm_chunk':512,'stop_after':18,
             'per_device_train_batch_size':batch,'gradient_accumulation_steps':accum,
             'eval_strategy':'no','eval_on_start':False,'eval_rows':16,'monitor_rows':16,
             'report_to':'none','logging_steps':6,'disable_tqdm':True,'ddp_timeout':120,
             'output_dir':str(root/name),'run_name':name}
        cfgpath=root/f'{name}.json';write_json(cfgpath,cfg)
        argv=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(accelerate_config.resolve()),
              '--module','scripts.benchmark_training','worker','--backend','causal',
              '--variant',name,'--disposable']
        if not lm:argv.append('--no-checkpoint-lm')
        if not aux:argv.append('--no-checkpoint-aux')
        argv.append(str(cfgpath))
        print('CAPACITY_START',name,now(),flush=True)
        start=time.monotonic()
        with (root/f'{name}.log').open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=900)
            except BaseException:
                stop_owned_process(child)
                raise
            finally:
                clean_owned_children()
        free=require_free(GPUS)
        result=inspect_attempt(root/name,code,name)
        result.update(arm=arm,checkpoint_layers=layers,checkpoint_lm=lm,checkpoint_aux=aux,
                      microbatch=batch,accumulation=accum,elapsed_seconds=time.monotonic()-start,
                      free_after=free,finished_at=now())
        if result['status']=='ok':
            pair=(result['train_fingerprint'],result['eval_fingerprint'])
            if fingerprints is None:fingerprints=pair
            assert pair==fingerprints
            if result['tokens_per_update']==1048576:
                if common_inputs is None:common_inputs=result['input_rows']
                assert result['input_rows']==common_inputs,'Fixed-global-batch inputs changed'
        results[name]=result
        write_json(root/'progress.json',dict(status='running',results=results))
        print('CAPACITY_RESULT',name,json.dumps({k:v for k,v in result.items() if k not in ('input_rows','validations','free_after')}),flush=True)
        return result

    # Independent ablations, including the combined removal. Full update timing.
    for arm in 'BFG':
        for lm,aux in ((True,True),(False,True)) if arm=='B' else ((True,True),(False,True),(True,False),(False,False)):
            attempt(arm,lm,aux)
    limits={}
    for arm in 'BFG':
        good=[r for r in results.values() if r['arm']==arm and r['status']=='ok' and r['microbatch']==16]
        if not good:raise RuntimeError(f'No passing starting point for {arm}')
        fastest=min(good,key=lambda r:r['seconds_per_update'])
        # Compare capacity of retained loss checkpointing and fastest tested removal.
        flags=dict.fromkeys([(True,True),(fastest['checkpoint_lm'],fastest['checkpoint_aux'])])
        for lm,aux in flags:
            limits[f'{arm}-lm{int(lm)}-aux{int(aux)}']=boundary(
                lambda batch:attempt(arm,lm,aux,batch)['status']=='ok')
    # Larger batch with decoder recomputation: independent comparator.
    attempt('B',batch=64,layers=True)
    compatible=[dict(name=n,**{k:v for k,v in r.items() if k not in ('input_rows','validations','free_after')})
                for n,r in results.items() if r['status']=='ok' and r['tokens_per_update']==1048576
                and r['reserved_headroom_gib']>=8]
    recommended={arm:min((r for r in compatible if r['arm']==arm),key=lambda r:r['seconds_per_update']) for arm in 'BFG'}
    write_json(root/'capacity-summary.json',dict(status='ok',limits=limits,recommended=recommended,results=results,
        note='18-step empirical capacity; >=8 GiB reserved-memory headroom heuristic. No production or long-run OOM guarantee.'))
    print('CAPACITY_COMPLETE',json.dumps(dict(limits=limits,recommended=recommended)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--accelerate-config',type=Path,required=True)
    args=p.parse_args()
    run(args.root.resolve(),args.recipe,args.accelerate_config)

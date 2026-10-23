"""Evaluate the four STOP branches concurrently, one independent GPU per model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_stop_adaptation import read,write


def summarize(results):
    reference=results['D-continue']['final'][0]['loss']
    def speed(r,policy):
        return next(x['decode_ms'] for x in r['latency']
                    if x['batch']==1 and x['context']==128 and x['policy']==policy)
    reference_time=speed(results['D-continue'],dict(policy='none'))
    points=[]
    for name,r in results.items():
        chosen=r['chosen']['policy']
        final=next(x for x in r['final'] if x['policy']==chosen)
        elapsed=speed(r,chosen)
        cache_ok=all(x['passed'] for x in r['cached_checks'].values())
        points.append(dict(branch=name,chosen=chosen,loss=final['loss'],perplexity=final['perplexity'],
            loss_gap_to_D_continue=final['loss']-reference,decode_ms=elapsed,
            speedup_vs_D_continue=reference_time/elapsed,cache_check_passed=cache_ok,
            practical_gate=cache_ok and final['loss']<=reference+.01 and elapsed<=.9*reference_time))
    return dict(status='ok',points=points,reference_loss=reference,reference_decode_ms=reference_time,
        scope='Single-pair adaptation pilot, not a statistical equivalence claim. Compare router STOP against dense STOP too.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    checkpoints=read(a.root/'branches.json');processes=[];handles=[]
    try:
        for gpu,(name,cp) in enumerate(checkpoints.items()):
            arm='D' if name.startswith('D-') else 'Partial-25-cosine'
            log=(a.output/(name+'.log')).open('x');handles.append(log)
            argv=[sys.executable,str(Path(__file__).resolve().parents[1]/'eval/stop_study.py'),
                '--checkpoint',cp,'--parent-recipe',str(a.parent/'prepared'/(arm+'-s0.train.json')),
                '--output',str(a.output/name)]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1'}
            processes.append(subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('STOP evaluation worker failed; inspect preserved worker logs')
            time.sleep(1)
        if any(p.returncode for p in processes):raise RuntimeError('STOP evaluation failed')
        results={name:read(a.output/name/'result.json') for name in checkpoints}
        if any(r['status']!='ok' for r in results.values()):raise ValueError('Invalid result')
        write(a.output/'summary.json',dict(**summarize(results),results=results))
    finally:
        # Only children launched here; never signal GPU-derived PIDs/groups.
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in handles:log.close()


if __name__=='__main__':main()

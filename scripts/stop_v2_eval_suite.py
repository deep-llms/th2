"""Eight independent evaluations, one per GPU, with matched saved controls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_stop_adaptation import read,write
from scripts.stop_eval_suite import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','previous','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();manifest=read(a.root/'evaluation_manifest.json')
    if len(manifest)!=8:raise ValueError('Expected six new fits and two existing controls')
    a.output.mkdir(parents=True,exist_ok=False);processes=[];logs=[]
    try:
        for gpu,(name,v) in enumerate(manifest.items()):
            log=(a.output/(name+'.log')).open('x');logs.append(log)
            argv=[sys.executable,str(Path(__file__).resolve().parents[1]/'eval/stop_v2_study.py'),
                '--checkpoint',v['checkpoint'],'--parent-recipe',str(a.parent/'prepared'/(v['arm']+'-s0.train.json')),
                '--previous',str(a.previous),'--arm',v['arm'],'--output',str(a.output/name)]
            if v['progress500']:argv.append('--progress500')
            processes.append(subprocess.Popen(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
                'OMP_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('STOP-v2 eval worker failed')
            time.sleep(1)
        if any(p.returncode for p in processes):raise RuntimeError('STOP-v2 eval failed')
        results={n:read(a.output/n/'result.json') for n in manifest}
        if any(r['status']!='ok' for r in results.values()):raise ValueError('Incomplete result')
        summary=summarize(results)
        for point in summary['points']:
            point['method']=manifest[point['branch']]['method']
            point['calibration_feasible']=results[point['branch']]['selection']['feasible']
        write(a.output/'summary.json',dict(**summary,results=results))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


if __name__=='__main__':main()

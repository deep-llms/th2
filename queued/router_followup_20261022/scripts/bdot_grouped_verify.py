"""Bounded grouped-GEMM numerical, eight-rank resume and full-update benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accelerate-config',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args()
    if a.repeats < 1:
        p.error('--repeats must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',checks={},benchmarks={})
    def save():
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,argv,gpus='0'):
        print('GROUPED_START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus},stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=1200)
    try:
        run('numerics',['tests/bdot/grouped_gpu_probe.py','--output',a.output/'numerics.json'])
        report['checks']['numerics']=json.loads((a.output/'numerics.json').read_text());save()
        for label,flags in [('no_reuse',['--no-reuse']),('soft',['--soft']),('home',['--home-only'])]:
            path=a.output/('ddp_'+label)
            run('ddp_'+label,['-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                'tests/bdot/ddp_probe.py',path,'--device','cuda','--bf16','--expert-backend','grouped',
                '--checkpointing',*flags],gpus='0,1')
            report['checks']['ddp_'+label]=[json.loads((path/f'rank{r}.json').read_text()) for r in range(2)];save()
        for curriculum in ('A','B'):
            path=a.output/('resume_'+curriculum)
            # Restart 10→20 across A's widening at 12 and B's soft→hard at 18.
            run('resume_'+curriculum,['tests/bdot/resume_gpu_probe.py','suite','--output',path,
                '--accelerate-config',a.accelerate_config,'--steps','120','--split','60',
                '--soft-split','10','--expert-backend','grouped','--curriculum',curriculum],
                gpus='0,1,2,3,4,5,6,7')
            report['checks']['resume_'+curriculum]=json.loads((path/'summary.json').read_text());save()
        for repeat in range(a.repeats):
            for label,config,progress,prior in [('native','dense_native',1.,None),('D','D',1.,None),
                                              ('hard','P1-B',1.,0.),('soft','P1-B',.01,0.)]:
                implementations = ['reference'] if label=='native' else ['reference','grouped']
                # Reverse the pair order on alternating repeats to reduce order bias.
                if repeat % 2:
                    implementations.reverse()
                for implementation in implementations:
                    name=f'{label}_{implementation}_r{repeat+1}';path=a.output/(name+'.json')
                    command=['-m','accelerate.commands.launch','--config_file',a.accelerate_config,
                        '--main_process_port','0','scripts/benchmark_bdot.py',
                        '--config',ROOT/'configs/bdot'/f'{config}.json','--output',path,'--device','cuda',
                        '--batch-size','16','--accumulation','4','--length','1024','--document-length','128',
                        '--warmup','5','--steps','50','--bf16','--attention-backend','fa4',
                        '--progress',str(progress),'--expert-backend',implementation]
                    if prior is not None:command+=['--router-prior',str(prior)]
                    run(name,command,gpus='0,1,2,3,4,5,6,7')
                    report['benchmarks'][name]=json.loads(path.read_text());save()
        report['comparison'] = {}
        for label in ('D','hard','soft'):
            times = {backend: [report['benchmarks'][f'{label}_{backend}_r{i+1}']['seconds_per_update']
                               for i in range(a.repeats)] for backend in ('reference','grouped')}
            report['comparison'][label] = dict(
                seconds_per_update=times,
                grouped_throughput_ratio=statistics.mean(times['reference'])/statistics.mean(times['grouped']),
                grouped_faster_in_every_repeat=max(times['grouped']) < min(times['reference']))
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=repr(error));save();raise
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'bdot/model.py',ROOT/'bdot/grouped.py',ROOT/'bdot/trainer.py']}
    save();print('GROUPED_VERIFICATION_PASSED',flush=True)


if __name__=='__main__':main()

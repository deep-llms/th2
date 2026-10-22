"""Numerical, eight-rank resume and complete-update timing gates before research."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.make_bdot_pilot import MLP_RUNS


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','reference','pilot','prepared','accelerate-config'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from scripts.gpu_status import require_free
    from bdot.routing import RoutingConfig
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',checks={},throughput={})
    def save():
        (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    def run(name,argv,gpus='0',timeout=1800):
        require_free(list(range(8)))
        print('MLP_CHECK_START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:
            report.update(status='failed',failed_check=name);save()
            raise RuntimeError('Check failed: '+name)
        require_free(list(range(8)))
        report['checks'][name]='passed';save()
        print('MLP_CHECK_OK '+name,flush=True)
    def launch(script,*args):
        return ['-m','accelerate.commands.launch','--config_file',a.accelerate_config,
                '--main_process_port','0',script,*args]
    for arm,_ in MLP_RUNS:
        cfg=read(a.prepared/(arm+'-s0.model.json'))
        opt=RoutingConfig(**cfg['bdot'])
        assert opt.route_scope=='mlp' and opt.reuse and opt.curriculum=='A'
        assert not opt.placement_film and opt.r==1
        assert opt.expert_backend=='grouped'
    for name,argv in [
        ('old_parity',['tests/bdot/dispatch_probe.py','--reference',a.reference,'--device','cuda']),
        ('mlp_numerics',['tests/bdot/partial_gpu_probe.py'])]:
        path=a.output/(name+'.json')
        run(name,[*argv,'--output',path])
        assert read(path)['status']=='passed'
    for arm in ('Partial-25','MLP-only-random'):
        name='resume_'+arm
        run(name,['tests/bdot/resume_gpu_probe.py','suite','--output',a.output/name,
            '--accelerate-config',a.accelerate_config,'--steps','60','--schedule-steps','100',
            '--split','40','--soft-split','10','--curriculum','A','--expert-backend','grouped',
            '--variant-config',a.prepared/(arm+'-s0.model.json')],gpus='0,1,2,3,4,5,6,7')
        assert read(a.output/name/'summary.json')['status']=='passed'
    for arm in ['D',*[name for name,_ in MLP_RUNS]]:
        name='T8_'+arm;path=a.output/(name+'.json')
        cfg=ROOT/'configs/bdot/D.json' if arm=='D' else a.prepared/(arm+'-s0.model.json')
        argv=launch('scripts/benchmark_bdot.py','--config',cfg,'--output',path,'--device','cuda',
            '--batch-size','16','--accumulation','4','--length','1024','--warmup','5','--steps','20',
            '--bf16','--attention-backend','fa4','--expert-backend','grouped','--progress','1')
        # Exercise actual multi-expert dispatch even before routers have trained.
        if arm!='D' and not arm.endswith('-random'):argv+=['--router-prior','0']
        run(name,argv,gpus='0,1,2,3,4,5,6,7')
        result=read(path)
        assert result['status']=='completed' and result['world_size']==8
        assert all(v is None or not isinstance(v,(int,float)) or math.isfinite(v)
                   for v in result['routing_metrics'].values())
        seconds=result['seconds_per_update'];assert math.isfinite(seconds) and seconds>0
        report['throughput'][arm]=dict(seconds_per_update=seconds,
            ratio_to_D=report['throughput'].get('D',dict(seconds_per_update=seconds))['seconds_per_update']/seconds)
        save()
    assert min(v['ratio_to_D'] for v in report['throughput'].values())>=.6,'Throughput below 60% of D'
    previous=read(a.pilot/'prepared/D.train.json')
    checkpoint=a.pilot/'prepared/runs/D/checkpoint-4768'
    path=a.output/'carry_D/result.json'
    run('carry_D',launch('eval/packed_ppl.py','--checkpoint',checkpoint,
        '--eval_data_dir',previous['eval_data_dir'],'--tokenizer_name',previous['tokenizer_name'],
        '--output_file',path,'--attention_backend','fa4','--precision','bf16','--batch_size','16'),
        gpus='0,1,2,3,4,5,6,7')
    result=read(path);original=read(checkpoint.parent/'final_eval.json')
    assert result['status']=='ok' and abs(result['loss']-original['eval_loss'])<1e-5
    assert (result['num_tokens'],result['num_sequences'])==(original['eval_num_tokens'],original['eval_num_sequences'])
    report['baseline']=dict(loss=result['loss'],num_tokens=result['num_tokens'],num_sequences=result['num_sequences'])
    report['status']='passed';save()
    print('MLP_ACCEPTANCE_PASSED '+json.dumps(report),flush=True)


if __name__=='__main__':main()

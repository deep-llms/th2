"""Bounded v3 GPU checks and v2 checkpoint carry-over; caller owns GPUs."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(Path(path).read_text())


def verify(output, reference, pilot, no_reuse, random_run, accelerate):
    from scripts.gpu_status import require_free
    from scripts.train_then_burn import digest
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',checks={},carryover={},throughput={})
    def save():
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,argv,gpus='0',timeout=1800):
        require_free(list(range(8)))
        print('V3_CHECK_START '+name,flush=True)
        with (output/(name+'.log')).open('x') as log:
            process=subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus},stdout=log,stderr=subprocess.STDOUT,
                timeout=timeout)
        if process.returncode:
            report.update(status='failed',failed_check=name);save()
            raise RuntimeError(name+' failed; see '+str(output/(name+'.log')))
        require_free(list(range(8)))
        print('V3_CHECK_OK '+name,flush=True)
    def launch(script,*args):
        return ['-m','accelerate.commands.launch','--config_file',accelerate,
                '--main_process_port','0',script,*args]
    previous=read(pilot/'prepared/P1-A.train.json')
    old_recipe=read(pilot/'prepared/runs/P1-A/training_recipe.json')
    for relative in ('bdot/model.py','bdot/routing.py','bdot/grouped.py','bdot/trainer.py'):
        if digest(reference/relative)!=old_recipe['code_sha256'][relative]:
            raise ValueError('Frozen v2 reference changed: '+relative)
    for name,argv in [
        ('T1',['tests/bdot/gpu_probe.py','--full-native','--native-only','--reference-math']),
        ('v2_parity',['tests/bdot/dispatch_probe.py','--reference',reference,'--device','cuda']),
        ('v3_numerics',['tests/bdot/grouped_gpu_probe.py','--v3'])]:
        path=output/(name+'.json');run(name,[*argv,'--output',path])
        result=read(path)
        if result['status']!='passed':raise ValueError(name+' receipt failed')
        report['checks'][name]=dict(status='passed',path=str(path));save()
    for arm in ('V3','V5'):
        name='resume_'+arm;path=output/name
        run(name,['tests/bdot/resume_gpu_probe.py','suite','--output',path,
            '--accelerate-config',accelerate,'--steps','60','--schedule-steps','100',
            '--split','40','--soft-split','10','--curriculum','A','--expert-backend','grouped',
            '--variant-config',ROOT/'configs/bdot'/f'{arm}.json'],gpus='0,1,2,3,4,5,6,7')
        result=read(path/'summary.json')
        if result['status']!='passed':raise ValueError(name+' failed')
        report['checks'][name]=dict(status='passed',path=str(path/'summary.json'));save()
    # Measure every distinct option set with the same full-width update layout.
    for arm in ('dense_native','D','V3','P1-A-prime','V1','V2','V4','C2','C3','V5','V6','V7'):
        name='T8_'+arm;path=output/(name+'.json')
        run(name,launch('scripts/benchmark_bdot.py','--config',ROOT/'configs/bdot'/f'{arm}.json',
            '--output',path,'--device','cuda','--batch-size','16','--accumulation','4',
            '--length','1024','--warmup','5','--steps','20','--bf16',
            '--attention-backend','fa4','--progress','1',
            '--expert-backend','reference' if arm=='dense_native' else 'grouped'),
            gpus='0,1,2,3,4,5,6,7')
        result=read(path)
        if result['status']!='completed':raise ValueError(name+' incomplete')
        report['throughput'][arm]=dict(seconds_per_update=result['seconds_per_update'],
            tokens_per_second=result['tokens_per_second'])
        report['throughput'][arm]['native_ratio']=(result['tokens_per_second']/
            report['throughput']['dense_native']['tokens_per_second'])
        save()
    if report['throughput']['V3']['native_ratio']<.6:
        report.update(status='failed',failed_check='T8_V3_below_60_percent');save()
        raise ValueError('V3 throughput fails prelaunch gate')
    for arm,run_root in [('D',pilot),('P1-A',pilot),('N-A',no_reuse),('R',random_run)]:
        checkpoint=run_root/'prepared/runs'/arm/'checkpoint-4768'
        original=read(checkpoint.parent/'final_eval.json')
        for version in ('v2','v3'):
            name=f'carry_{arm}_{version}';path=output/name/'result.json'
            argv=['--checkpoint',checkpoint,'--eval_data_dir',previous['eval_data_dir'],
                '--tokenizer_name',previous['tokenizer_name'],'--output_file',path,
                '--attention_backend','fa4','--precision','bf16','--batch_size','16',
                '--route_study','true','--route_sequences','4096','--route_artifact_sequences','256',
                '--routing_seed','0']
            run(name,launch(ROOT/'scripts/bdot_v3_verify.py','eval-worker',
                '--reference',reference if version=='v2' else 'current','--',*argv),
                gpus='0,1,2,3,4,5,6,7')
        a=read(output/f'carry_{arm}_v2/result.json');b=read(output/f'carry_{arm}_v3/result.json')
        from eval.route_study import compare_route_samples
        routes=compare_route_samples(output/f'carry_{arm}_v2/routes_sample.npz',
                                    output/f'carry_{arm}_v3/routes_sample.npz')
        if (abs(a['loss']-b['loss'])>1e-3 or abs(b['loss']-original['eval_loss'])>1e-3
                or routes['agreement']!=1. or a['num_tokens']!=original['eval_num_tokens']
                or b['num_tokens']!=original['eval_num_tokens']):
            raise ValueError('Carry-over failed: '+arm)
        report['carryover'][arm]=dict(status='passed',v2_loss=a['loss'],v3_loss=b['loss'],
            original_loss=original['eval_loss'],route_agreement=routes['agreement']);save()
    report['status']='passed';save()
    print('V3_VERIFICATION_PASSED '+json.dumps(report),flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='eval-worker':
        # Run the unchanged evaluator against a complete isolated old package.
        source=sys.argv[sys.argv.index('--reference')+1]
        if source!='current':
            import bdot
            bdot.__path__=[str(Path(source)/'bdot')]
        args=sys.argv[sys.argv.index('--')+1:]
        sys.argv=[str(ROOT/'eval/packed_ppl.py'),*args]
        runpy.run_path(sys.argv[0],run_name='__main__');return
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','reference','pilot','no-reuse','random-run','accelerate-config'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    verify(a.output,a.reference,a.pilot,a.no_reuse,a.random_run,a.accelerate_config)


if __name__=='__main__':main()

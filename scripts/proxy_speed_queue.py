"""Build the explicitly authorized CUDA gates, 25-step smokes and timing study."""
import argparse
import copy
import hashlib
from pathlib import Path

from deep_kv.__main__ import jobs
from run_experiments import load_jobs
from scripts.proxy_speed_validation import NEW_ARMS, BENCH_ARMS, ROOT, read, write


def make(root, recipe_path, rows, diagnostic=False, continuation=None):
    root=Path(root)
    recipe=read(recipe_path)
    recipe.update(attention_backend='fa4',checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,
                  eval_rows=32,monitor_rows=32,save_steps=24,save_total_limit=2,logging_steps=10)
    for k,v in dict(block_size=2048,per_device_train_batch_size=16,gradient_accumulation_steps=4,
                    max_steps=28600,warmup_steps=1430,isolate_documents=True,seed=42,data_seed=42).items():
        assert recipe[k]==v,(k,recipe[k])
    saved=root/'validation-recipe.json';write(saved,recipe)
    if diagnostic:
        items=[dict(name='repeatability',gpus=[0],timeout_seconds=3600,
            argv=['{python}','-u','-m','scripts.proxy_speed_validation','diagnose','--arm','P7-simple-short',
                  '--backends','fa4','fa4_deterministic',
                  '--recipe',str(saved),'--rows',str(rows),'--output','{run_dir}/repeatability.json'],
            required_outputs=[dict(path='repeatability.json',json_equals={'status':'measured'})])]
        path=root/'jobs.json';write(path,dict(jobs=items));load_jobs(path)
        return items
    items=[]
    def cpu(name,mode,arguments,result):
        items.append(dict(name=name,argv=['{python}','-u','-m','scripts.proxy_speed_validation',mode,*arguments,
                      '--output','{run_dir}/'+result],required_outputs=[dict(path=result,json_equals={'status':'passed'})]))
    cpu('numerics-all','gate-all',['--recipe',str(saved),'--rows',str(rows)],'numerics/summary.json')
    items[-1]['gpus']=list(range(8))
    items[-1]['timeout_seconds']=3600
    items[-1]['required_outputs'] += [dict(path='numerics/'+arm+'.json',json_equals={'status':'passed'})
                                    for arm in (*NEW_ARMS,'P6-iso','P7-simple','P7')]

    def train(arm,namespace,impl='optimized',profile=False):
        item=next(x for x in jobs(saved,stop_after=25,arms=[arm],seeds=[42])['jobs'] if 'gpus' in x)
        item=copy.deepcopy(item);item['name']=namespace.replace('/','-')+'-'+arm
        item['argv']=[x.replace('{run_dir}','{run_dir}/'+namespace) for x in item['argv']]
        for out in item['required_outputs']:out['path']=namespace+'/'+out['path']
        pos=item['argv'].index(str(ROOT/'train.py'))
        item['argv'][pos:pos+1]=['--module','scripts.proxy_speed_validation','train','--implementation',impl,
                              *(['--profile'] if profile else ['--audit-update25','--deterministic-fa4']),'--']
        item['timeout_seconds']=7200
        items.append(item)
    def validate(namespace,arms):
        dest=namespace+'/validated.json'
        items.append(dict(name='validate-'+namespace.replace('/','-'),
            argv=['{python}','-m','scripts.check_fa4_proxy','validate','--run-dir','{run_dir}/'+namespace,
                  '--steps','25','--arms',*arms,'--attention-backend','fa4','--output','{run_dir}/'+dest],
            required_outputs=[dict(path=dest,json_equals={'status':'passed'})]))
    for arm in NEW_ARMS:
        train(arm,'smoke')
    validate('smoke',list(NEW_ARMS))
    for arm in NEW_ARMS:
        source='{run_dir}/smoke/seed-42/'+arm
        destination='{run_dir}/resume/seed-42/'+arm
        cpu('copy-resume-'+arm,'copy-resume',['--source',source,'--destination',destination],'resume-ready/'+arm+'.json')
        train(arm,'resume')
        cpu('check-resume-'+arm,'check-resume',['--source',source,'--destination',destination],'resume-checks/'+arm+'.json')
    validate('resume',list(NEW_ARMS))
    for i,arm in enumerate(BENCH_ARMS):
        # Alternate pair order to reduce systematic warm-cache/order bias.
        for impl in (('previous','optimized') if i%2==0 else ('optimized','previous')):
            train(arm,'benchmark/'+impl,impl,profile=True)
    for impl in ('previous','optimized'):validate('benchmark/'+impl,list(BENCH_ARMS))
    cpu('summarize','summary',['--run-dir','{run_dir}'],'summary.json')
    if continuation is not None:
        source=Path(continuation).resolve(strict=True)
        previous=read(source/'run.json')
        completed=['numerics-all',*['smoke-'+arm for arm in NEW_ARMS]]
        assert previous['status']=='failed'
        assert [j['name'] for j in previous['jobs']]==completed+['validate-smoke']
        assert all(j['status']=='ok' for j in previous['jobs'][:-1])
        assert previous['jobs'][-1]['status']=='failed'
        assert read(source.parents[1]/'validation-recipe.json')==recipe
        before=read(source.parents[1]/'preflight.json')['source_sha256']
        for name in ('deep_kv/proxy.py','deep_kv/proxy_memory.py','deep_kv/proxy_training.py','train.py'):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==before[name],name
        artifacts=[]
        for job in previous['jobs'][:-1]:
            for artifact in job['artifacts']:
                path=(source/artifact['path']).resolve(strict=True)
                assert path.is_relative_to(source)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']
                artifacts.append(artifact)
        assert [j['name'] for j in items[:7]]==completed
        items=items[7:]
        # Completed training outputs remain read-only in their original root.
        # All new validation reports, resume copies and benchmarks are fresh.
        for job in items:
            if job['name']=='validate-smoke':
                mapping={'{run_dir}/smoke/validated.json':'{run_dir}/smoke-validated.json',
                         '{run_dir}/smoke':str(source/'smoke')}
                job['argv']=[mapping.get(v,v) for v in job['argv']]
                job['required_outputs'][0]['path']='smoke-validated.json'
            elif job['name'].startswith(('copy-resume-','check-resume-')):
                job['argv']=[v.replace('{run_dir}/smoke',str(source/'smoke')) for v in job['argv']]
        write(root/'continuation.json',dict(source=str(source),artifacts=artifacts,
              run_sha256=hashlib.sha256((source/'run.json').read_bytes()).hexdigest(),
              note='Reuse verified numerical/smoke outputs; rerun corrected validator, then resume/timing stages.'))
    path=root/'jobs.json';write(path,dict(jobs=items));load_jobs(path)
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','recipe','rows'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();items=make(args.root,args.recipe,args.rows)
    print('VALIDATION_STAGES',len(items),flush=True)


if __name__=='__main__':main()

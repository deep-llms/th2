"""Generate a fresh A/P6-iso 10,000-step comparison queue."""
import argparse
import json
from pathlib import Path

from deep_kv.__main__ import jobs
from run_experiments import load_jobs

ARMS=('A','P6-iso')
SEED=1042
STEPS=10000


def make(recipe_path, output, seed=SEED):
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    recipe=json.loads(Path(recipe_path).read_text())
    expected=dict(max_steps=28600,warmup_steps=1430,save_steps=250,
                  save_total_limit=0,block_size=2048,isolate_documents=True,
                  attention_backend='fa4',per_device_train_batch_size=16,
                  gradient_accumulation_steps=4,checkpoint_layers=False,
                  checkpoint_lm=False,checkpoint_aux=False)
    for key,value in expected.items():
        if recipe.get(key)!=value:
            raise ValueError(f'Unexpected recipe {key}: {recipe.get(key)!r}')
    generated=jobs(recipe_path,stop_after=STEPS,arms=ARMS,seeds=[seed])['jobs']
    items=[]
    for job in generated:
        if job['name']=='compare-seeds':
            continue  # One seed and one within-seed report; no cross-seed report.
        if 'gpus' in job:
            arm=job['argv'][job['argv'].index('--arm')+1]
            if arm not in ARMS or job['gpus']!=list(range(8)):
                raise ValueError('Unexpected generated training job')
            argv=job['argv']
            if '--proxy_module_seed' in argv:
                raise ValueError('Recipe must not specify proxy module seed')
            if arm!='A':
                argv+=['--proxy_module_seed',str(seed+1)]
            job['argv']=['env',f'PYTHONHASHSEED={seed}',*argv]
            items.append(job)
            name=f'seed-{seed}-validate-{arm}'
            items.append(dict(name=name,
                              argv=['{python}','-m','scripts.check_fa4_proxy','validate',
                                    '--run-dir','{run_dir}','--steps',str(STEPS),
                                    '--seed',str(seed),'--arms',arm,
                                    '--attention-backend','fa4',
                                    '--output','{run_dir}/'+name+'.json'],
                              required_outputs=[dict(path=name+'.json',
                                                     json_equals={'status':'passed'})]))
        else:
            items.append(job)
    output=Path(output)
    with output.open('x') as handle:
        json.dump({'jobs':items},handle,indent=2)
    load_jobs(output)
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=SEED)
    a=p.parse_args()
    print('STAGES',len(make(a.recipe,a.output,a.seed)))


if __name__=='__main__':
    main()

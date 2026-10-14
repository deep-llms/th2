"""Disposable P7-targetgrad timing study using the existing Trainer and supervisor."""
import argparse
import copy
import json
from pathlib import Path
import statistics


def read(path):return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def predecessor_ready(root, guard=Path('/mnt/local/_gpu_guard/DISABLED')):
    root=Path(root)/'supervised'
    state,run,burn=(read(root/p) for p in ('supervisor.json','run/run.json','burn-verified.json'))
    if (state.get('training_status')!='ok' or state.get('training_returncode')!=0
            or not state.get('finished_at') or run.get('status')!='ok'
            or not run.get('jobs') or any(j.get('status')!='ok' for j in run['jobs'])
            or not (root/'run/complete.json').is_file() or guard.exists()
            or not burn.get('collective_progress_verified')
            or not state.get('burn',{}).get('collective_progress_verified')
            or burn['session']!=state['burn']['session']):
        raise ValueError('Predecessor and verified burn handoff must be complete')


def make_jobs(root, recipe_path):
    from deep_kv.__main__ import jobs
    recipe=read(recipe_path)
    for key,value in dict(num_hidden_layers=12,block_size=2048,per_device_train_batch_size=16,
        gradient_accumulation_steps=4,attention_backend='fa4',checkpoint_layers=False,
        checkpoint_aux=False,checkpoint_lm=False,isolate_documents=True,seed=42,data_seed=42).items():
        if recipe[key]!=value:raise ValueError(('Unexpected recipe',key,recipe[key]))
    recipe.update(eval_rows=32,monitor_rows=32,save_steps=1000,save_total_limit=0)
    root=Path(root);path=root/'recipe.json';write(path,recipe)
    items=[]
    for arm in ('A','P7-simple-targetgrad'):
        job=copy.deepcopy(next(j for j in jobs(path,stop_after=50,arms=[arm],seeds=[42])['jobs'] if 'gpus' in j))
        job['name']=arm+'-profile';job['timeout_seconds']=1800
        pos=next(i for i,a in enumerate(job['argv']) if a.endswith('/train.py'))
        job['argv'][pos:pos+1]=['--module','scripts.profile_proxy_training']
        for rank in range(8):
            job['required_outputs'].append(dict(path=f'seed-42/{arm}/step-profile-rank{rank}.json',json_equals={'status':'profiled'}))
        job['required_outputs'].append(dict(path=f'seed-42/{arm}/component-profile.json',json_equals={'status':'profiled'}))
        items.append(job)
    items.append(dict(name='summarize',argv=['{python}','-m','scripts.p7_speed_study','summarize',
        '--root','{run_dir}'],required_outputs=[dict(path='summary.json',json_equals={'status':'passed'})]))
    write(root/'jobs.json',dict(jobs=items))
    return items


def prepare(args):
    import importlib.metadata as md
    import inspect as inspect_source
    import shutil
    import socket
    import subprocess
    from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
    from scripts.verified_gpu_reclaim import inspect,process
    from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH,digest
    from run_experiments import load_jobs
    root=Path(args.root)
    if socket.gethostname()!=args.host:raise ValueError('Wrong host')
    predecessor_ready(args.predecessor)
    expected=dict(torch='2.14.1',transformers='5.9.0',accelerate='1.13.0',datasets='4.8.5',
        **{'flash-attn-4':'4.0.0b33','nvidia-cutlass-dsl':'4.8.0'})
    versions={k:md.version(k) for k in expected}
    if any(versions[k].split('+')[0]!=v for k,v in expected.items()):raise ValueError(versions)
    recipe=read('proxy_heads.12layer.b200.json')
    for k in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
        if not Path(recipe[k]).exists():raise ValueError('Missing local '+k)
    if shutil.disk_usage(root).free<100*2**30:raise ValueError('Need 100GiB for disposable checkpoints')
    from flash_attn.cute import interface
    source=inspect_source.getsource(interface)
    (root/'fa4-interface.py').write_text(source)
    # Audit the actual pinned build, not GitHub main or an older wheel.
    api=dict(signature=str(inspect_source.signature(interface.flash_attn_varlen_func)),
        backward=inspect_source.getsource(interface.FlashAttnVarlenFunc.backward),
        source_sha256=digest(root/'fa4-interface.py'))
    write(root/'fa4-api.json',api)
    print('PINNED_FA4_API',json.dumps(api),flush=True)
    source=Path('resources/accelerate_config.yaml');destination=Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
    shutil.copy2(source,destination);cfg=load_config_from_file(str(destination)).to_dict()
    if source.read_bytes()!=destination.read_bytes() or cfg['num_processes']!=8 or cfg['mixed_precision']!='bf16':
        raise ValueError('Accelerate mismatch')
    subprocess.run(['accelerate','env'],check=True)
    make_jobs(root,'proxy_heads.12layer.b200.json');load_jobs(root/'jobs.json')
    status=inspect()
    if (status['host']!=args.host or status['guard_disabled'] or len(status['gpus'])!=8
            or any('B200' not in g['name'] for g in status['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH):raise ValueError('GPU ownership/state changed')
    approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
    if any(not (approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved))
           for p in status['workers']):raise ValueError('Unknown GPU workload; no signals sent')
    write(root/'inspection.json',status)
    write(root/'preflight.json',dict(status='passed',versions=versions,
        accelerate_cache=str(destination),accelerate_sha256=digest(destination),predecessor=args.predecessor))
    print('P7_SPEED_PREFLIGHT_PASSED',flush=True)


def summarize(args):
    root=Path(args.root);results={}
    for arm in ('A','P7-simple-targetgrad'):
        folder=root/'seed-42'/arm;result=read(folder/'result.json')
        if result['status']!='stopped' or result['global_step']!=50:raise ValueError('Incomplete timing run')
        ranks=[read(folder/f'step-profile-rank{i}.json') for i in range(8)]
        if any([s['step'] for s in r['steps']]!=list(range(1,51)) for r in ranks):raise ValueError('Incomplete rank timing')
        # Warm-up/profile excluded. Max rank start-to-start includes logging and
        # data delivery; do not quote isolated kernel sums as whole-step speed.
        values=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(19,49)]
        results[arm]=dict(median_seconds=statistics.median(values),mean_seconds=statistics.mean(values),
            step_range=[20,49],intervals=values,peak_gib=max(r['peak_gib'] for r in ranks),
            components=read(folder/'component-profile.json')['kernel_milliseconds'])
    report=dict(status='passed',results=results,
        overhead_percent=100*(results['P7-simple-targetgrad']['median_seconds']/results['A']['median_seconds']-1))
    write(root/'summary.json',report);print('P7_SPEED_SUMMARY',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','summarize'))
    p.add_argument('--root',required=True);p.add_argument('--host');p.add_argument('--predecessor')
    a=p.parse_args();globals()[a.mode](a)

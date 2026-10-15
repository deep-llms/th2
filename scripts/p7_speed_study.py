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


def await_previous(args):
    import time
    deadline=time.monotonic()+600
    while True:
        try:
            predecessor_ready(args.root)
            return
        except (FileNotFoundError, ValueError):
            path=Path(args.root)/'supervised/supervisor.json'
            state=read(path) if path.exists() else {}
            if state.get('finished_at') and state.get('training_status')!='ok':
                raise ValueError('Predecessor failed; no integration launch')
            if time.monotonic()>=deadline:raise TimeoutError('No verified predecessor burn handoff in ten minutes')
            print('WAITING_FOR_VERIFIED_PREDECESSOR_BURNS',flush=True)
            time.sleep(10)


def profile_predecessor_ready(root, receipt):
    """Accept only the documented summary-only failure, preserving its history."""
    import hashlib
    root=Path(root)
    state,run,burn=(read(root/'supervised'/p) for p in ('supervisor.json','run/run.json','burn-verified.json'))
    if state.get('training_status')=='ok':
        predecessor_ready(root)
    else:
        jobs=run.get('jobs',[])
        if (root.name!='q359-p7-speed-profile-20261015-a01'
                or state.get('training_status')!='failed' or not state.get('finished_at')
                or run.get('status')!='failed' or run.get('error')!='summarize exited with code 1'
                or [j['name'] for j in jobs]!=['A-profile','P7-simple-targetgrad-profile','summarize']
                or any(j.get('status')!='ok' or j.get('returncode')!=0 for j in jobs[:2])
                or jobs[2].get('status')!='failed' or jobs[2].get('returncode')!=1
                or not burn.get('collective_progress_verified')
                or not state.get('burn',{}).get('collective_progress_verified')
                or burn['session']!=state['burn']['session']
                or Path('/mnt/local/_gpu_guard/DISABLED').exists()):
            raise ValueError('Not the verified profile-summary-only failure')
        for job in jobs[:2]:
            for artifact in job['artifacts']:
                if hashlib.sha256((root/'supervised/run'/artifact['path']).read_bytes()).hexdigest()!=artifact['sha256']:
                    raise ValueError('Completed profiling artifact changed')
    # Recompute the corrected summary into the NEW study root. Do not erase
    # the original failure or rewrite its supervisor/run/completion records.
    report=profile_report(root/'supervised/run')
    report['source_summary_failed']=state['training_status']=='failed'
    write(receipt,report)
    print('VERIFIED_PROFILE_SUMMARY',json.dumps(report),flush=True)


def make_jobs(root, recipe_path, trial=False, integration=False, refinement=False, entries_integration=False):
    from deep_kv.__main__ import jobs
    recipe=read(recipe_path)
    for key,value in dict(num_hidden_layers=12,block_size=2048,per_device_train_batch_size=16,
        gradient_accumulation_steps=4,attention_backend='fa4',checkpoint_layers=False,
        checkpoint_aux=False,checkpoint_lm=False,isolate_documents=True,seed=42,data_seed=42).items():
        if recipe[key]!=value:raise ValueError(('Unexpected recipe',key,recipe[key]))
    recipe.update(eval_rows=32,monitor_rows=32,save_steps=1000,save_total_limit=0)
    root=Path(root);path=root/'recipe.json';write(path,recipe)
    if trial or refinement:
        checkpoint='/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01/supervised/run/seed-42/P7-simple-targetgrad/checkpoint-2500'
        rows='/mnt/local/_outputs/deep-llms_th2/q359-p7-teacher-probe-20261015-a01/rows.json'
        items=[dict(name='numerical-gate',gpus=list(range(8)),timeout_seconds=1800,
            argv=['{python}','-m','accelerate.commands.launch','--config_file',
                str(Path('resources/accelerate_config.yaml').resolve()),'--module','scripts.p7_speed_trial','gate',
                '--checkpoint',checkpoint,'--checkpoint-sha256','be0de11aba95411521a521c59c21b19777a36c92d418ef9647d4baa6a3845b45',
                '--rows',rows,'--rows-sha256','00cd77dc6923216b065855bdc40cf6105afa7a70d305452aa0a92a4c7e728bcc',
                '--output','{run_dir}/gate'],
            required_outputs=[dict(path='gate/summary.json',json_equals={'status':'passed'})]),
            dict(name='accepted-candidate-timings',timeout_seconds=5400,
                argv=['{python}','-m','scripts.p7_speed_study','benchmark','--root','{run_dir}',
                      '--recipe',str(path)],
                required_outputs=[dict(path='timing-summary.json',json_equals={'status':'passed'})])]
        if refinement:
            items[0]['argv'].append('--refinement')
            items[1]['argv'].append('--refinement')
        write(root/'jobs.json',dict(jobs=items));return items
    items=[]
    for arm in ('A','P7-simple-targetgrad'):
        job=copy.deepcopy(next(j for j in jobs(path,stop_after=50,arms=[arm],seeds=[42])['jobs'] if 'gpus' in j))
        job['name']=arm+'-profile';job['timeout_seconds']=1800
        pos=next(i for i,a in enumerate(job['argv']) if a.endswith('/train.py'))
        job['argv'][pos:pos+1]=(['--module','scripts.p7_speed_trial','train','--implementation','configured','--']
                              if integration else ['--module','scripts.profile_proxy_training'])
        if integration and arm=='P7-simple-targetgrad':
            job['argv']+=['--proxy_compile_auxiliary','true','--proxy_lse_attention','true']
            if entries_integration:job['argv']+=['--proxy_compile_entries','true']
        for rank in range(8):
            job['required_outputs'].append(dict(path=f'seed-42/{arm}/step-profile-rank{rank}.json',json_equals={'status':'profiled'}))
        job['required_outputs'].append(dict(path=f'seed-42/{arm}/component-profile.json',json_equals={'status':'profiled'}))
        items.append(job)
    items.append(dict(name='summarize',argv=['{python}','-m','scripts.p7_speed_study','summarize',
        '--root','{run_dir}'],required_outputs=[dict(path='summary.json',json_equals={'status':'passed'})]))
    if entries_integration:
        # Reuse the trained-checkpoint gate, now comparing the accumulated
        # optimizations directly to the ORIGINAL implementation as well.
        gate=make_jobs(root/'original-gate-config',recipe_path,refinement=True)[0]
        gate['argv']+=['--original-reference','--require-candidate','entries']
        items.insert(0,gate)
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
    if args.trial:profile_predecessor_ready(args.predecessor,root/'profile-summary.json')
    else:predecessor_ready(args.predecessor)
    if args.integration:
        timing=read(Path(args.predecessor)/'supervised/run/timing-summary.json')
        if timing['status']!='passed' or timing['results']['entries' if args.entries_integration else 'both']['speedup_percent']<=1:
            raise ValueError('Integrated candidate must have a verified measured improvement')
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
    make_jobs(root,'proxy_heads.12layer.b200.json',args.trial,args.integration,args.refinement,args.entries_integration);load_jobs(root/'jobs.json')
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


def profile_report(root):
    root=Path(root);results={};data_reference=None
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
        if (folder/'speed-trial-rank0.json').exists():
            audits=[read(folder/f'speed-trial-rank{rank}.json') for rank in range(8)]
            data=[a['data_hashes'] for a in audits]
            if any(len(hashes)!=8 for hashes in data):raise ValueError('Incomplete data-order audit')
            if data_reference is None:data_reference=data
            elif data!=data_reference:raise ValueError('Training sample/order mismatch')
            if any(a['graphs']['20']!=a['graphs']['49'] for a in audits):raise ValueError('Recompilation in measured interval')
            results[arm].update(data_order_verified=True,graph_counts=[a['graphs'] for a in audits])
    return dict(status='passed',results=results,
        overhead_percent=100*(results['P7-simple-targetgrad']['median_seconds']/results['A']['median_seconds']-1))


def summarize(args):
    root=Path(args.root);report=profile_report(root)
    write(root/'summary.json',report);print('P7_SPEED_SUMMARY',json.dumps(report),flush=True)


def benchmark(args):
    import subprocess
    import sys
    from deep_kv.__main__ import jobs
    from run_experiments import load_jobs
    root=Path(args.root);gate=read(root/'gate/summary.json')
    if gate['status']!='passed' or not gate['checkpoint_unchanged']:
        raise ValueError('Numerical acceptance required before timing')
    reference_mode='both' if args.refinement else 'eager'
    cases=[('reference-before',reference_mode)]+[(mode,mode) for mode in gate['accepted']]+[('reference-after',reference_mode)]
    items=[]
    for label,mode in cases:
        arm='P7-simple-targetgrad'
        job=copy.deepcopy(next(j for j in jobs(args.recipe,stop_after=50,arms=[arm],seeds=[42])['jobs'] if 'gpus' in j))
        pos=next(i for i,a in enumerate(job['argv']) if a.endswith('/train.py'))
        job['argv'][pos:pos+1]=['--module','scripts.p7_speed_trial','train','--implementation',mode,'--']
        job['argv']=[v.replace('{run_dir}','{run_dir}/'+label) for v in job['argv']]
        for output in job['required_outputs']:output['path']=label+'/'+output['path']
        job['name']=label;job['timeout_seconds']=1200
        for rank in range(8):
            job['required_outputs'].append(dict(path=f'{label}/seed-42/{arm}/step-profile-rank{rank}.json',json_equals={'status':'profiled'}))
        items.append(job)
    path=root/'timing-jobs.json';write(path,dict(jobs=items));load_jobs(path)
    subprocess.run([sys.executable,'run_experiments.py','--config',str(path),'--run-dir',str(root/'timings')],check=True)
    results={}; data_reference=None
    for label,mode in cases:
        folder=root/'timings'/label/'seed-42'/'P7-simple-targetgrad'
        result=read(folder/'result.json')
        if result['status']!='stopped' or result['global_step']!=50:raise ValueError('Incomplete training')
        ranks=[read(folder/f'step-profile-rank{rank}.json') for rank in range(8)]
        audits=[read(folder/f'speed-trial-rank{rank}.json') for rank in range(8)]
        data=[a['data_hashes'] for a in audits]
        if any(len(hashes)!=8 for hashes in data):raise ValueError('Incomplete data-order audit')
        if data_reference is None:data_reference=data
        elif data!=data_reference:raise ValueError('Different training samples/order')
        if any(a['graphs']['20']!=a['graphs']['49'] for a in audits):raise ValueError('Compilation inside measured interval')
        if any([s['step'] for s in r['steps']]!=list(range(1,51)) for r in ranks):raise ValueError('Incomplete timing')
        values=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(19,49)]
        results[label]=dict(mode=mode,median_seconds=statistics.median(values),mean_seconds=statistics.mean(values),
            intervals=values,peak_gib=max(r['peak_gib'] for r in ranks),evaluation=result['evaluation'],
            data_order_verified=True,graph_counts=[a['graphs'] for a in audits])
    reference=statistics.mean(results[label]['median_seconds'] for label in ('reference-before','reference-after'))
    for label in gate['accepted']:results[label]['speedup_percent']=100*(reference/results[label]['median_seconds']-1)
    write(root/'timing-summary.json',dict(status='passed',results=results,reference_seconds=reference,
        scope='Disposable 50-step eight-GPU Trainer trials; steps20-49; not a long training equivalence claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','summarize','benchmark','await_previous'))
    p.add_argument('--root',required=True);p.add_argument('--host');p.add_argument('--predecessor')
    p.add_argument('--trial',action='store_true');p.add_argument('--recipe')
    p.add_argument('--integration',action='store_true')
    p.add_argument('--refinement',action='store_true')
    p.add_argument('--entries-integration',action='store_true')
    a=p.parse_args()
    if a.entries_integration and not a.integration:p.error('--entries-integration requires --integration')
    globals()[a.mode](a)

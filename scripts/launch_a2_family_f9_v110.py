"""Admit the closed paper list after the recorded, intentional v1.9 drain."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from eval.a2_family_f9 import read
from eval.a2_family_f9_v110 import SPEC_SHA,EVALUATOR_COMMIT,write,select
from scripts.a2_family_f9_run import locked
from scripts.a2_family_f9_prepare import hash_entry
from scripts.a2_quant_io import lock
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import BURN_HASH,GUARD_HASH,approved_launcher
from efficient_llm.diagnostic_data import sha256
from run_experiments import write_json,load_jobs


def predecessor(root,host):
    root=Path(root);script=root/'source/resources/llm_pretrain_burn.py'
    drain=read(root/'scope_v110_drain/native_handoff_requested.json')
    sv=read(root/'supervised/supervisor.json');burn=read(root/'supervised/burn-verified.json')
    if not drain['drained'] or drain['remaining_children']:raise ValueError('Current children not drained')
    if not sv.get('finished_at') or sv.get('handoff_error') or not burn['collective_progress_verified']:
        raise ValueError('Intentional drain native handoff incomplete')
    if len(sv['free_after_training'])!=8 or any(g['pids'] for g in sv['free_after_training']):raise ValueError('Native release incomplete')
    if sha256(script)!=BURN_HASH or sha256('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:raise ValueError('Native code changed')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('GPU guard owned')
    fresh=inspect()
    if fresh['host']!=host or len(fresh['gpus'])!=8:raise ValueError('Wrong host/GPUs')
    for g in fresh['gpus']:
        if not g['pids']:raise ValueError('Expected verified native burn')
        for pid in g['pids']:
            if not (approved_launcher(pid,{str(script):BURN_HASH}) or approved_launcher(process(pid)['ppid'],{str(script):BURN_HASH})):
                raise ValueError('Unknown GPU owner; no signals sent')
    return fresh


def wait_handoff(root,host):
    root=Path(root);deadline=time.monotonic()+7200
    print('F9_V110_WAIT_NATIVE_HANDOFF',root,flush=True)
    while time.monotonic()<deadline:
        request=root/'scope_v110_drain/native_handoff_requested.json'
        if request.is_file():
            if not read(request)['drained']:raise ValueError('Old scorer drain did not complete')
            supervisor=root/'supervised/supervisor.json'
            if supervisor.is_file() and read(supervisor).get('finished_at'):return predecessor(root,host)
        time.sleep(5)
    raise TimeoutError('Native handoff not verified; no new GPU job launched')


def job_manifest(config,root):
    return dict(jobs=[dict(name='family-F9-v110-closed-list',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f9_v110_run','--config',str(config),'--receipt','{run_dir}/f9-complete.json'],
        required_outputs=[dict(path='f9-complete.json',json_equals=dict(execution_complete=True))]),
        dict(name='family-F9-v110-backup',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f9_backup','--root',str(root),'--spec-sha256',SPEC_SHA,'--receipt','{run_dir}/f9-backup.json'],
        required_outputs=[dict(path='f9-backup.json',json_equals=dict(verified=True))])])


def launch(plan,output,session,commit):
    cfg=read(plan);source=Path(__file__).resolve().parents[1];out=Path(output);old=Path(cfg['prior_root'])
    if cfg['spec_sha256']!=SPEC_SHA or not re.fullmatch('[0-9a-f]{40}',commit) or out.exists():raise ValueError('Invalid scope/commit or duplicate root')
    if socket.gethostname()!=cfg['host']:raise ValueError('Wrong node')
    for name in (session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Session exists')
    proof=read(source/cfg['cpu_proof'])
    if not proof['launch_ready']:raise ValueError('CPU proof not ready')
    for p,h in proof['files'].items():
        if sha256(source/p)!=h:raise ValueError('Tested source changed: '+p)
    stopped=read(old/'scope_v110_drain/dispatch_stopped.json')
    if stopped['spec_sha256']!=SPEC_SHA:raise ValueError('No authorized old dispatch stop')
    manifest=locked(cfg['evaluator_source_manifest'])
    for p,h in manifest['files'].items():
        if sha256(old/'source'/p)!=h:raise ValueError('Numerical evaluator source changed')
    attestation=locked(cfg['evaluator_attestation'])
    prior=old/'results/a2_family_20261107/downstream'
    oldcfg=read(prior/'config_admitted_a02.json')
    if oldcfg['code_commit']!=EVALUATOR_COMMIT:raise ValueError('Wrong evaluator commit')
    ready=locked(cfg['old_ready'])
    if not ready['cpu_ready']:raise ValueError('Old preparation incomplete')
    for item in oldcfg['locked_inputs']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Locked data changed: '+item['path'])
    from efficient_llm.a2.data import PackedRows
    PackedRows(oldcfg['opensci_cx_root'],'final',verify=True)
    original=locked(oldcfg['inventory']);groups,excluded=select(original['models']);entries=sum(groups,[])
    if [len(g) for g in groups]!=[13,12,9,12,9]:raise ValueError('Closed inventory differs')
    if [e['model_id'] for e in entries]!=cfg['selected_ids']:raise ValueError('Reviewed endpoint list differs')
    out.mkdir();frozen=freeze_source(source,out,commit,extra_files=cfg['extra_source_files'])
    write(out/'source-manifest.json',read(out/'source-manifest.json'))
    result=out/'results/a2_family_20261107/downstream';prep=result/'preparation';prep.mkdir(parents=True)
    # Fresh physical hash check; all selected bytes must match the loaded FT12 inventory.
    hash_failures=[];usable=[]
    def check(e):
        try:
            actual=hash_entry(e)
            if actual!=e:raise ValueError('Inventory identity/actual files changed')
            return e,None
        except Exception as error:return None,dict(model=e['model_id'],error=repr(error))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for e,error in pool.map(check,entries):
            if error:hash_failures.append(error)
            else:usable.append(e)
    # Keep full closed list in manifest; unusable rows get explicit skip flags.
    inventory=write(prep/'FT12_inventory.json',dict(models=entries,missing=[],hash_failures=hash_failures,
        excluded_prior_ids=excluded,source_inventory=oldcfg['inventory'],groups=[len(g) for g in groups]))
    loads=read(prior/'FT12_loads.json')
    if not loads['all_load']:raise ValueError('FT12 loads incomplete')
    for e in usable:locked(loads['loaded'][e['model_id']])
    spots=[]
    for group in groups:
        candidates=[prior/'FT12_loads'/e['model_id']/'FT12_spot.json' for e in group]
        found=[lock(p) for p in candidates if p.is_file() and read(p).get('passed')]
        spots.append(found)
    batches={};flags=[]
    for family in ('opensci','opensci13','smollm3'):
        p=prior/('FT11_'+family+'.json')
        if not p.is_file():continue
        v=read(p);batches[family]=v['batch_size'];write(prep/p.name,dict(origin=lock(p),result=v))
    for e in entries:
        if e['batch_family'] not in batches:raise ValueError('Missing family FT11 proof: '+e['batch_family'])
    flags.append(dict(test='intentional_v19_scope_change',detail=lock(old/'scope_v110_drain/dispatch_stopped.json')))
    flags.extend(read(prior/'flags.json')['flags']);flags.extend(dict(test='FT12_hash:'+e['model'],detail=e) for e in hash_failures)
    # Preserve exact raw preparations and all admission evidence in portable backup.
    import shutil
    for name in ('FT12_loads.json','FT13.json','flags.json'):
        shutil.copyfile(prior/name,prep/('v19_'+name))
    for p in (old/'SOURCE_COMMIT_ATTESTATION.json',old/'ADMISSION_CORRECTIONS.json',old/'scope_v110_drain/native_handoff_requested.json'):
        if p.is_file():shutil.copyfile(p,prep/p.name)
    benchmarks=oldcfg['benchmarks']
    for family,common in benchmarks.items():
        prepared=locked(common['prepared'])
        for name,h in prepared['protocol']['tokenizer_files'].items():
            if sha256(Path(common['tokenizer'])/name)!=h:raise ValueError('Actual tokenizer changed')
        for key in ('prepared','input_spec','manifest','tokenizer_manifest'):
            if key not in common:continue
            locked(common[key]);dest=prep/(family+'_'+key+'.json');shutil.copyfile(common[key]['path'],dest)
            common[key]=lock(dest)
    run=dict(oldcfg,spec_sha256=SPEC_SHA,code_commit=commit,evaluator_commit=EVALUATOR_COMMIT,
        source_manifest=lock(out/'source-manifest.json'),evaluator_source_manifest=cfg['evaluator_source_manifest'],
        evaluator_source=str(old/'source'),prior_root=str(prior),downstream_root=str(result),inventory=inventory,
        benchmarks=benchmarks,batches=batches,retained_flags=flags,old_ft13=lock(prior/'FT13.json'),
        unusable_ids=[x['model'] for x in hash_failures],spot_groups=spots,
        selected_ids=cfg['selected_ids'])
    write(prep/'FT12_v110.json',dict(selected=55,hash_verified=len(usable),loads_reused=len(usable),
        group_spots=spots,hash_failures=hash_failures,prior_scope_missing_excluded=True))
    config=write(result/'config.json',run)
    path=out/'jobs.json';write_json(path,job_manifest(config['path'],result));load_jobs(path)
    fresh=wait_handoff(old,cfg['host']);inspection=write(out/'inspection.json',fresh)
    shutil.copyfile(old/'scope_v110_drain/native_handoff_requested.json',prep/'native_handoff_requested.json')
    command=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(path),'--output',str(out/'supervised'),
        '--inspection',inspection['path'],'--host',cfg['host'],'--burn-session',session+'_burn',
        '--predecessor-burn-script',str(old/'source/resources/llm_pretrain_burn.py')]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(frozen),'OMP_NUM_THREADS':'1','A2_CODE_COMMIT':commit,
        'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline'}
    shell='exec '+shlex.join(command)+' >'+shlex.quote(str(out/'supervisor.log'))+' 2>&1'
    write(out/'launch.json',dict(config=config,jobs=lock(path),session=session,source_manifest=run['source_manifest'],
        scope='v1.10 closed55 only; same numerical evaluator; no training; no interim'))
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    print('F9_V110_NATIVE_QUEUE_SUBMITTED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','output','session','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args();launch(a.plan,a.output,a.session,a.code_commit)

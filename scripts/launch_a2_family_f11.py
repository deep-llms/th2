"""Immutable CPU preparation and automatic native handoff AFTER F10 report."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from eval.a2_family_f11 import read,write,SPEC_SHA
from scripts.a2_quant_io import lock
from scripts.a2_family_f9_run import locked
from scripts.launch_a2_downstream import freeze_source
from scripts.launch_a2_family_f10 import predecessor
from scripts.a2_family_f11_queue import descriptors
from efficient_llm.diagnostic_data import sha256
from run_experiments import write_json,load_jobs

def verify_source(source,manifest):
    for name,h in locked(manifest)['files'].items():
        if sha256(Path(source)/name)!=h:raise ValueError('Immutable source changed: '+name)

def prepare(plan,qroot):
    root=Path(qroot);froot=Path(plan['f10_root']);fcfg=read(froot/'config.json')
    if fcfg['code_commit']!=plan['f10_code_commit'] or fcfg['output']!=str(froot/'results/a2_family_20261107/f10'):
        raise ValueError('Wrong F10 lineage')
    verify_source(froot/'source',fcfg['source_manifest'])
    cfg={**plan,'f10':fcfg,'f10_config_lock':lock(froot/'config.json')}
    entries=sum(fcfg['groups'],[])
    if len(entries)!=42 or len({e['model_id'] for e in entries})!=42:raise ValueError('Wrong F11 calibration scope')
    if any(e['family']!='opensci' or e['arm'] not in ('P300','P1T','S72','S188') for e in entries):raise ValueError('Unregistered model')
    if [sum(e['role']==r for e in entries) for r in ('parent','oneshot','student')]!=[8,8,26]:raise ValueError('Wrong F11 role counts')
    checked={}
    def check(path,h):
        if str(path) in checked:
            if checked[str(path)]!=h:raise ValueError('Conflicting input locks')
            return
        if sha256(path)!=h:raise ValueError('Input changed: '+str(path))
        checked[str(path)]=h
    for e in entries:
        for path,h in e['file_hashes'].items():check(path,h)
    assets=locked(plan['assets'])
    from efficient_llm.a2.protocol import parent_ref
    for arm in ('P300','P1T'):
        e=assets['dclm/'+arm]
        if any(e[k]!=v for k,v in parent_ref('dclm',arm).items()):raise ValueError('Parent revision changed')
        for directory,key in [('weights_dir','files'),('code_dir','code_files')]:
            for name,h in e[key].items():check(Path(e[directory])/name,h)
        parent=next(e for e in entries if e['role']=='parent' and e['dataset']=='dclm' and e['arm']==arm)
        if e['files']['model.safetensors']!=parent['file_hashes'][str(Path(parent['path'])/'model.safetensors')]:raise ValueError('Training/evaluation parent differ')
    stream=plan['heal_streams']['dclm']['2030'];healm=locked(dict(path=str(Path(stream['path'])/'manifest.json'),sha256=stream['sha256']))
    cx=locked(plan['data_manifest'])
    for base,m,splits in [(Path(plan['data_root']),cx,('tuning','final')),(Path(stream['path']),healm,('heal',))]:
        if m['status']!='complete':raise ValueError('Incomplete data')
        for split in splits:
            v=m['streams'][split]
            if v['length']!=4096 or v['tokens']!=v['rows']*4096 or (split=='heal' and v['rows']<3051*64):raise ValueError('Wrong data dimensions')
            for suffix,key,scale in [('bin','token_sha256',4),('starts','starts_sha256',1)]:
                p=base/(split+'.'+suffix)
                if p.stat().st_size!=v['tokens']*scale:raise ValueError('Wrong data size')
                check(p,v[key])
        for name,key in [('document_order.bin','document_order_sha256'),('documents.bin','documents_sha256')]:
            if key in m and (base/name).exists():check(base/name,m[key])
    if any(Path(e['cx_root']).resolve()!=Path(plan['data_root']).resolve() for e in entries):raise ValueError('Different CulturaX data roots')
    for key in ('prepared','FT16','f9_manifest'):locked(fcfg[key])
    for common in fcfg['benchmarks'].values():
        for key in ('prepared','manifest','input_spec','tokenizer_manifest'):
            if key in common:locked(common[key])
    cfg['entries']=entries;cfg['runs']={r['run_id']:r for r in descriptors()};refs={}
    for e in entries:
        if e['role']=='parent' and e['dataset']=='dclm' and e['arm'] in ('P300','P1T'):
            refs['dclm/'+e['arm']+'/parent']=write(root/'references'/(e['arm']+'.json'),dict(loss=e['final_nll'],source_entry=e))
    cfg['reference_nll']=refs
    proof=write(root/'actual_inputs.json',dict(verified=True,files=checked,models=42,heal_rows=3051*64,stream_manifest=dict(path=str(Path(stream['path'])/'manifest.json'),sha256=stream['sha256'])))
    cfg['preparation_proof']=proof;write(root/'prepared_config.json',cfg)
    write(root/'ready.json',dict(cpu_ready=True,config=lock(root/'prepared_config.json'),actual_inputs=proof,waiting_for='F10 report/backup/native handoff'))
    return cfg

def predecessor_ready(cfg):
    """No GPU mutation here. Never equate an absent process with completion."""
    root=Path(cfg['f10_root']);result=Path(cfg['f10']['output'])
    svpath=root/'supervised/supervisor.json'
    if not svpath.exists() or not read(svpath).get('finished_at'):return False
    sv=read(svpath)
    if sv.get('handoff_error'):raise ValueError('F10 native handoff failed')
    if not (root/'supervised/burn-verified.json').is_file():return False
    for p in (result/'complete.json',result/'report_complete.json',result/'backup/verified.json',root/'supervised/burn-verified.json'):
        if not p.is_file():raise ValueError('F10 terminal lacks required receipt: '+str(p))
    complete=read(result/'complete.json');report=read(result/'report_complete.json');backup=read(result/'backup/verified.json')
    if not complete.get('report_complete') or not report.get('complete') or not backup.get('verified'):
        raise ValueError('F10 report/backup incomplete; preserve outputs and report block')
    if [complete.get(k) for k in ('nine_models','new_models','generation_models')]!=[11,42,34]:raise ValueError('F10 coverage incomplete')
    report_path=result/'REPORT_FULL_v1.11.md';expected=backup['files'].get(report_path.name)
    if not expected or sha256(report_path)!=expected['sha256']:raise ValueError('F10 report hash differs')
    archive=result/'backup/family_f10_scientific.tar.gz'
    if sha256(archive)!=backup['archive_sha256']:raise ValueError('F10 archive hash differs')
    if not read(root/'supervised/burn-verified.json').get('collective_progress_verified'):return False
    return True

def job_manifest(config):
    return dict(jobs=[dict(name='family-F11-v113-LM-then-calibration',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f11_queue','--config',str(config),'--receipt','{run_dir}/f11-complete.json'],
        required_outputs=[dict(path='f11-complete.json',json_equals=dict(execution_complete=True))])])

def launch(cfg,source,commit):
    out=Path(cfg['output_root']);session=cfg['session'];source=Path(source)
    if out.exists() or socket.gethostname()!=cfg['host']:raise ValueError('Wrong host or duplicate F11 root')
    if version('transformers')!='4.57.1' or version('torch').split('+')[0]!='2.14.1':raise ValueError('Numerical runtime changed')
    for s in (session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',s],capture_output=True).returncode==0:raise ValueError('Existing F11 session')
    tested=locked(cfg['cpu_proof'])
    for name,h in tested['files'].items():
        if sha256(source/name)!=h:raise ValueError('Untested code change')
    locked(cfg['f10_config_lock']);proof=locked(cfg['preparation_proof'])
    # Stat identity is only a fast change detector. Workers rehash model bytes;
    # the training stream has already been hashed in CPU preparation and is
    # rehashed at this final admission as well.
    for path,h in proof['files'].items():
        if sha256(path)!=h:raise ValueError('Actual input changed before F11')
    predecessor(cfg['f10_root'],cfg['host'])
    out.mkdir();frozen=freeze_source(source,out,commit,extra_files=['docs/A2_family_spec_v1.13.md','resources/a2_family_f11_plan_20261111.json',*tested['files']])
    cfg=dict(cfg,code_commit=commit,source_manifest=lock(out/'source-manifest.json'),results_root=str(out/'results/a2_family_20261107/f11'),
             previous_report=lock(Path(cfg['f10']['output'])/'REPORT_FULL_v1.11.md'),generation_batch=read(Path(cfg['f10']['output'])/'FT15.json')['batch_size'])
    config=write(out/'config.json',cfg);jobs=out/'jobs.json';write_json(jobs,job_manifest(config['path']));load_jobs(jobs)
    fresh=write(out/'inspection.json',predecessor(cfg['f10_root'],cfg['host']))
    cmd=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobs),'--output',str(out/'supervised'),
         '--inspection',fresh['path'],'--host',cfg['host'],'--burn-session',session+'_burn',
         '--predecessor-burn-script',str(Path(cfg['f10_root'])/'source/resources/llm_pretrain_burn.py')]
    shell=environment_shell(cmd,frozen,commit,out/'supervisor.log')
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write(out/'launch.json',dict(config=config,session=session,predecessor=cfg['f10_root']))
    print('F11_NATIVE_STARTED',out,flush=True)

def environment_shell(cmd,source,commit,log):
    values={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(source),'A2_CODE_COMMIT':commit,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline','PYTHONDONTWRITEBYTECODE':'1'}
    return 'exec env '+shlex.join([k+'='+v for k,v in values.items()]+cmd)+' >'+shlex.quote(str(log))+' 2>&1'

def install(plan,commit,proof_path):
    cfg=read(plan);source=Path(__file__).resolve().parents[1];root=Path(cfg['queue_root']);proof=read(proof_path)
    if cfg['spec_sha256']!=SPEC_SHA or proof.get('spec_sha256')!=SPEC_SHA or not proof.get('launch_ready'):raise ValueError('Wrong spec/CPU proof')
    if socket.gethostname()!=cfg['host'] or root.exists() or Path(cfg['output_root']).exists():raise ValueError('Wrong host or duplicate root')
    session=cfg['queue_session']
    if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise ValueError('Existing admission queue')
    for name,h in proof['files'].items():
        if sha256(source/name)!=h:raise ValueError('Changed tested source '+name)
    root.mkdir();frozen=freeze_source(source,root,commit,extra_files=['docs/A2_family_spec_v1.13.md','resources/a2_family_f11_plan_20261111.json',str(Path(proof_path).relative_to(source)) if Path(proof_path).is_absolute() else str(proof_path),*proof['files']])
    cfg['cpu_proof']=lock(frozen/Path(proof_path).resolve().relative_to(source))
    write(root/'plan.json',cfg)
    command=[sys.executable,'-u','-m','scripts.launch_a2_family_f11','--mode','wait','--plan',str(root/'plan.json'),'--code-commit',commit]
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),environment_shell(command,frozen,commit,root/'queue.log')],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write(root/'installed.json',dict(installed=True,queue_session=session,source_manifest=lock(root/'source-manifest.json'),plan=lock(root/'plan.json')))
    print('F11_CPU_QUEUE_INSTALLED',root,flush=True)

def wait(plan,commit):
    plan=read(plan);root=Path(plan['queue_root']);source=Path(__file__).resolve().parents[1]
    try:
        cfg=prepare(plan,root)
        while not predecessor_ready(cfg):
            write(root/'status.json',dict(state='ready_waiting_F10_report_backup_native',gpu_started=False));time.sleep(20)
        launch(cfg,source,commit)
        write(root/'status.json',dict(state='F11_launched',gpu_started=True,output=cfg['output_root']))
    except Exception as error:
        write(root/'status.json',dict(state='blocked',error=repr(error),gpu_started=Path(plan['output_root']).exists()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['install','wait'],required=True);p.add_argument('--plan',required=True);p.add_argument('--code-commit',required=True);p.add_argument('--cpu-proof')
    a=p.parse_args()
    if a.mode=='install':install(a.plan,a.code_commit,a.cpu_proof)
    else:wait(a.plan,a.code_commit)

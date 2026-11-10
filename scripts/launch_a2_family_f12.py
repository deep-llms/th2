"""Immutable CPU preparation and automatic native handoff AFTER F11 report."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from eval.a2_family_f12 import read,write,SPEC_SHA
from scripts.a2_quant_io import lock
from scripts.a2_family_f9_run import locked
from scripts.launch_a2_downstream import freeze_source
from scripts.launch_a2_family_f10 import predecessor
from scripts.a2_family_f12_queue import descriptors
from efficient_llm.diagnostic_data import sha256
from run_experiments import write_json,load_jobs

def verify_source(source,manifest):
    for name,h in locked(manifest)['files'].items():
        if sha256(Path(source)/name)!=h:raise ValueError('Immutable source changed: '+name)

def prepare(plan,qroot):
    root=Path(qroot);prior=locked(plan['f11_prepared'])
    if prior['output_root']!=plan['f11_root'] or prior['f10_root']!=plan['f10_root']:
        raise ValueError('Wrong F11 lineage')
    q11=Path(plan['f11_prepared']['path']).parent
    manifest=lock(q11/'source-manifest.json');verify_source(q11/'source',manifest)
    if read(manifest['path'])['code_commit']!=plan['f11_code_commit']:
        raise ValueError('Wrong frozen F11 source')
    # Reuse the tested F11 admission for exact parent revisions and all actual
    # input bytes, then require the longer F12 stream coverage explicitly.
    from scripts.launch_a2_family_f11 import prepare as prepare_inputs
    inherited=prepare_inputs(plan,root)
    cfg={**inherited,**plan}
    stream=plan['heal_streams']['dclm']['2030']
    manifest=locked(dict(path=str(Path(stream['path'])/'manifest.json'),sha256=stream['sha256']))
    if manifest['streams']['heal']['rows']<12207*64:raise ValueError('F12 stream too short')
    cfg['reference_13']=locked(plan['reference_13_lock'])
    inventory=locked(plan['inventory'])
    if not inventory.get('FT19',{}).get('passed'):raise ValueError('Inventory metadata not verified')
    raw=inventory['metadata_archive'];archive=Path(__file__).resolve().parents[1]/raw['path']
    if sha256(archive)!=raw['sha256']:raise ValueError('Inventory raw metadata changed')
    if cfg['reference_13']['nonembedding_parameters']!=1205424128:raise ValueError('F0 reference architecture changed')
    cfg['runs']={r['run_id']:r for r in descriptors()}
    proof=read(cfg['preparation_proof']['path']);proof.update(heal_rows=12207*64,f11_prepared=plan['f11_prepared'],f11_source_verified=True)
    cfg['preparation_proof']=write(root/'actual_inputs.json',proof)
    write(root/'prepared_config.json',cfg)
    write(root/'ready.json',dict(cpu_ready=True,config=lock(root/'prepared_config.json'),actual_inputs=cfg['preparation_proof'],waiting_for='F11 report/backup/native handoff'))
    return cfg

def predecessor_ready(cfg):
    """No GPU mutation here. Never equate an absent process with completion."""
    root=Path(cfg['f11_root']);result=root/'results/a2_family_20261107/f11'
    svpath=root/'supervised/supervisor.json'
    if not svpath.exists() or not read(svpath).get('finished_at'):return False
    sv=read(svpath)
    if sv.get('handoff_error'):raise ValueError('F11 native handoff failed')
    if not (root/'supervised/burn-verified.json').is_file():return False
    for p in (result/'complete.json',result/'report_complete.json',result/'backup/verified.json',root/'supervised/burn-verified.json'):
        if not p.is_file():raise ValueError('F11 terminal lacks required receipt: '+str(p))
    complete=read(result/'complete.json');report=read(result/'report_complete.json');backup=read(result/'backup/verified.json')
    if not complete.get('report_complete') or not report.get('complete') or not backup.get('verified'):
        raise ValueError('F11 report/backup incomplete; preserve outputs and report block')
    if not complete.get('execution_complete') or not report.get('downstream_replayed') or report.get('calibration_replayed')!=42 or not report.get('LM_replayed'):raise ValueError('F11 coverage incomplete')
    report_path=result/'REPORT_FULL_v1.13.md';expected=backup['files'].get(report_path.name)
    if not expected or sha256(report_path)!=expected['sha256']:raise ValueError('F11 report hash differs')
    archive=result/'backup/family_f11_scientific.tar.gz'
    if sha256(archive)!=backup['archive_sha256']:raise ValueError('F11 archive hash differs')
    if not read(root/'supervised/burn-verified.json').get('collective_progress_verified'):return False
    return True

def job_manifest(config):
    return dict(jobs=[dict(name='family-F12-v116-width-budget-curve',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f12_queue','--config',str(config),'--receipt','{run_dir}/f12-complete.json'],
        required_outputs=[dict(path='f12-complete.json',json_equals=dict(execution_complete=True))])])

def launch(cfg,source,commit):
    out=Path(cfg['output_root']);session=cfg['session'];source=Path(source)
    if out.exists() or socket.gethostname()!=cfg['host']:raise ValueError('Wrong host or duplicate F12 root')
    if version('transformers')!='4.57.1' or version('torch').split('+')[0]!='2.14.1':raise ValueError('Numerical runtime changed')
    for s in (session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',s],capture_output=True).returncode==0:raise ValueError('Existing F12 session')
    tested=locked(cfg['cpu_proof'])
    for name,h in tested['files'].items():
        if sha256(source/name)!=h:raise ValueError('Untested code change')
    locked(cfg['f10_config_lock']);proof=locked(cfg['preparation_proof'])
    # Stat identity is only a fast change detector. Workers rehash model bytes;
    # the training stream has already been hashed in CPU preparation and is
    # rehashed at this final admission as well.
    for path,h in proof['files'].items():
        if sha256(path)!=h:raise ValueError('Actual input changed before F12')
    predecessor(cfg['f11_root'],cfg['host'])
    out.mkdir();frozen=freeze_source(source,out,commit,extra_files=['docs/A2_family_spec_v1.16.md','resources/a2_family_f12_plan_20261111.json',*tested['files']])
    cfg=dict(cfg,code_commit=commit,source_manifest=lock(out/'source-manifest.json'),results_root=str(out/'results/a2_family_20261107/f12'),
             previous_report=lock(Path(cfg['f11_root'])/'results/a2_family_20261107/f11/REPORT_FULL_v1.13.md'),generation_batch=read(Path(cfg['f10']['output'])/'FT15.json')['batch_size'])
    config=write(out/'config.json',cfg);jobs=out/'jobs.json';write_json(jobs,job_manifest(config['path']));load_jobs(jobs)
    fresh=write(out/'inspection.json',predecessor(cfg['f11_root'],cfg['host']))
    cmd=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobs),'--output',str(out/'supervised'),
         '--inspection',fresh['path'],'--host',cfg['host'],'--burn-session',session+'_burn',
         '--predecessor-burn-script',str(Path(cfg['f11_root'])/'source/resources/llm_pretrain_burn.py')]
    shell=environment_shell(cmd,frozen,commit,out/'supervisor.log')
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write(out/'launch.json',dict(config=config,session=session,predecessor=cfg['f11_root']))
    print('F12_NATIVE_STARTED',out,flush=True)

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
    root.mkdir();frozen=freeze_source(source,root,commit,extra_files=['docs/A2_family_spec_v1.16.md','resources/a2_family_f12_plan_20261111.json',str(Path(proof_path).relative_to(source)) if Path(proof_path).is_absolute() else str(proof_path),*proof['files']])
    cfg['cpu_proof']=lock(frozen/Path(proof_path).resolve().relative_to(source))
    for key in ('inventory','reference_13_lock'):
        name=Path(cfg[key]['path']).name;target=frozen/'resources'/name
        if sha256(target)!=cfg[key]['sha256']:raise ValueError('Frozen metadata changed')
        cfg[key]=lock(target)
    write(root/'plan.json',cfg)
    command=[sys.executable,'-u','-m','scripts.launch_a2_family_f12','--mode','wait','--plan',str(root/'plan.json'),'--code-commit',commit]
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),environment_shell(command,frozen,commit,root/'queue.log')],check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    write(root/'installed.json',dict(installed=True,queue_session=session,source_manifest=lock(root/'source-manifest.json'),plan=lock(root/'plan.json')))
    print('F12_CPU_QUEUE_INSTALLED',root,flush=True)

def wait(plan,commit):
    plan=read(plan);root=Path(plan['queue_root']);source=Path(__file__).resolve().parents[1]
    try:
        cfg=prepare(plan,root)
        while not predecessor_ready(cfg):
            write(root/'status.json',dict(state='ready_waiting_F11_report_backup_native',gpu_started=False));time.sleep(20)
        launch(cfg,source,commit)
        write(root/'status.json',dict(state='F12_launched',gpu_started=True,output=cfg['output_root']))
    except Exception as error:
        write(root/'status.json',dict(state='blocked',error=repr(error),gpu_started=Path(plan['output_root']).exists()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['install','wait'],required=True);p.add_argument('--plan',required=True);p.add_argument('--code-commit',required=True);p.add_argument('--cpu-proof')
    a=p.parse_args()
    if a.mode=='install':install(a.plan,a.code_commit,a.cpu_proof)
    else:wait(a.plan,a.code_commit)

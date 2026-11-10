"""Launch registered F10 only after CPU admission and a fresh native handoff."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from eval.a2_family_f10 import SPEC_SHA,write
from eval.a2_family_f9 import read
from scripts.a2_family_f9_run import locked
from scripts.a2_quant_io import lock
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import BURN_HASH,GUARD_HASH,approved_launcher
from efficient_llm.diagnostic_data import sha256
from run_experiments import write_json,load_jobs


def job_manifest(config):
    return dict(jobs=[dict(name='family-F10-v111-evaluation-only',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-u','-m','scripts.a2_family_f10_run','--config',str(config),'--receipt','{run_dir}/f10-complete.json'],
        required_outputs=[dict(path='f10-complete.json',json_equals=dict(execution_complete=True))])])


def predecessor(root,host):
    root=Path(root);script=root/'source/resources/llm_pretrain_burn.py'
    sv=read(root/'supervised/supervisor.json');burn=read(root/'supervised/burn-verified.json')
    if not sv.get('finished_at') or sv.get('handoff_error') or not burn['collective_progress_verified']:raise ValueError('Native predecessor handoff incomplete')
    if len(sv['free_after_training'])!=8 or any(g['pids'] for g in sv['free_after_training']):raise ValueError('Native release incomplete')
    if sha256(script)!=BURN_HASH or sha256('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:raise ValueError('Native guard/burn differs')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():raise ValueError('Another supervisor owns guard')
    fresh=inspect()
    if fresh['host']!=host or len(fresh['gpus'])!=8:raise ValueError('Wrong host/GPUs')
    for g in fresh['gpus']:
        if not g['pids']:raise ValueError('Expected predecessor burn')
        for pid in g['pids']:
            if not (approved_launcher(pid,{str(script):BURN_HASH}) or approved_launcher(process(pid)['ppid'],{str(script):BURN_HASH})):
                raise ValueError('Unknown GPU owner; no signals sent')
    return fresh


def launch(preparation,output,session,commit,proof):
    prep=Path(preparation);out=Path(output);source=Path(__file__).resolve().parents[1]
    ready=read(prep/'ready.json');cfg=locked(ready['config'])
    if not ready['cpu_ready'] or ready['new_models']!=42 or ready['oneshots']!=11 or cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Preparation incomplete')
    if socket.gethostname()!=cfg['host'] or out.exists() or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Wrong host, code or duplicate output')
    if version('transformers')!='4.57.1' or version('torch').split('+')[0]!='2.14.1':raise ValueError('Changed numerical runtime')
    tested=read(proof)
    if not tested['launch_ready'] or tested['spec_sha256']!=SPEC_SHA:raise ValueError('Missing CPU proof')
    for name,h in tested['files'].items():
        if sha256(source/name)!=h:raise ValueError('Untested source change: '+name)
    for name in (session,session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    for key in ('prepared','FT16','input_manifest','f9_manifest','f10a'):locked(cfg[key])
    record=locked(cfg['FT16'])
    if record['counts']!={'mmlu':14042,'triviaqa':17944} or not record['recorded_before_scoring']:raise ValueError('FT16 coverage differs')
    for path,h in record['input_hashes'].items():
        if sha256(path)!=h:raise ValueError('Actual task data changed')
    prior=locked(cfg['f9_config'])
    for item in prior['locked_inputs']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Actual old data changed')
    from efficient_llm.a2.data import PackedRows
    PackedRows(prior['opensci_cx_root'],'final',verify=True)
    for common in cfg['benchmarks'].values():
        for key in ('prepared','input_spec','manifest','tokenizer_manifest'):
            if key in common:locked(common[key])
        for name,h in read(common['prepared']['path'])['protocol']['tokenizer_files'].items():
            if sha256(Path(common['tokenizer'])/name)!=h:raise ValueError('Actual tokenizer changed')
    # CPU admission freshly hashed every model; workers rehash each full model
    # immediately before loading it. Require that admission has the exact set.
    if ready['model_hashes']!=37 or len(set(cfg['fresh_model_hashes']))!=37:raise ValueError('Incomplete model admission')
    predecessor(cfg['f9_root'],cfg['host'])
    out.mkdir();frozen=freeze_source(source,out,commit,extra_files=['docs/A2_family_spec_v1.11.md',
        'resources/a2_family_f10_plan_20261111.json','resources/a2_family_f10_inputs_20261111.json',str(Path(proof).resolve().relative_to(source))])
    write(out/'source-manifest.json',read(out/'source-manifest.json'))
    cfg.update(source_manifest=lock(out/'source-manifest.json'),code_commit=commit,
        output=str(out/'results/a2_family_20261107/f10'),preparation_ready=lock(prep/'ready.json'),cpu_proof=lock(proof))
    config=write(out/'config.json',cfg);jobs=out/'jobs.json';write_json(jobs,job_manifest(config['path']));load_jobs(jobs)
    fresh=write(out/'inspection.json',predecessor(cfg['f9_root'],cfg['host']))
    cmd=[sys.executable,'-u','-m','scripts.train_then_burn','--config',str(jobs),'--output',str(out/'supervised'),
         '--inspection',fresh['path'],'--host',cfg['host'],'--burn-session',session+'_burn',
         '--predecessor-burn-script',str(Path(cfg['f9_root'])/'source/resources/llm_pretrain_burn.py')]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(frozen),'A2_CODE_COMMIT':commit,
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline'}
    shell='exec '+shlex.join(cmd)+' >'+shlex.quote(str(out/'supervisor.log'))+' 2>&1'
    write(out/'launch.json',dict(config=config,jobs=lock(jobs),session=session,scope='F10 only; FT14/FT15/11oneshots/42newtask/analysis/backup; native burn after'))
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
    print('F10_NATIVE_QUEUE_SUBMITTED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('preparation','output','session','code-commit','cpu-proof'):p.add_argument('--'+k,required=True)
    a=p.parse_args();launch(a.preparation,a.output,a.session,a.code_commit,a.cpu_proof)

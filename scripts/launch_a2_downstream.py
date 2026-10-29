"""Supervised fixed A2 downstream screen after a reviewed terminal handoff.

Uses the existing native supervisor unchanged. No training, installations or
network access; all checkpoint and benchmark inputs must already be local.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

from efficient_llm.kv_long import verify_handoff
from eval.a2_plan import prepare_identities
from eval.a2_run import lock, write, validate_plan
from eval.downstream_suite import COUNTS
from scripts.a2_remaining import read_locked
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest, BURN_HASH, GUARD_HASH
from run_experiments import load_jobs


def job_config(plan_lock, timeout_seconds):
    return dict(jobs=[dict(name='a2-downstream',gpus=list(range(8)),timeout_seconds=timeout_seconds+60,
        argv=['{python}','-m','eval.a2_run','--plan',plan_lock['path'],'--plan-sha256',plan_lock['sha256'],
            '--timeout-seconds',str(timeout_seconds),'--output','{run_dir}/downstream'],
        required_outputs=[dict(path='downstream/complete.json',json_equals=dict(status='ok',models=14)),
                          dict(path='downstream/results.json',json_equals=dict(status='ok',diagnostic=False))])])


def freeze_source(source, output, commit, extra_files=()):
    frozen=output/'source';files=list(source.glob('*.py'))
    for directory in ('efficient_llm','eval','scripts'):
        files.extend((source/directory).rglob('*.py'))
    files.extend(source/name for name in ('resources/accelerate_b200.yaml','resources/llm_pretrain_burn.py',
        'resources/a2_inputs_20261028.json','resources/downstream_english_20261023.json',
        'resources/a2_early_grid_reuse.json','resources/a2_pair_reuse.json','docs/A2_pilot_spec.md'))
    files.extend(source/name for name in extra_files)
    identities={}
    for file in files:
        relative=file.relative_to(source);target=frozen/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
        identities[str(relative)]=digest(target)
    write(output/'source-manifest.json',dict(code_commit=commit,files=identities))
    return frozen


def remaining_budget(spent,elapsed,requested):
    if not all(isinstance(v,(float,int)) and v>=0 for v in (spent,elapsed,requested)) or not 0<requested<=10800:
        raise ValueError('Invalid bounded downstream budget')
    # Leave ten minutes for artifact checks and communicating-burn restoration.
    if spent+elapsed+requested+660>18*3600:raise ValueError('Phase hard cap would be exceeded')
    return dict(prior_seconds=spent,preparation_seconds=elapsed,timeout_seconds=requested,
        phase_hard_seconds=18*3600,report_due=spent+elapsed+requested>=15*3600)


def verify_failed_eval_handoff(output_parent,burn_script,reviewed_lock):
    reviewed=read_locked(reviewed_lock);root=burn_script.parents[2]
    if (reviewed.get('scope')!='a2_batch8_numerical_gate_failure_before_scoring'
            or root.parent.resolve()!=output_parent.resolve()
            or burn_script.resolve()!=(root/'source/resources/llm_pretrain_burn.py').resolve()):
        raise ValueError('Wrong reviewed failed-evaluation predecessor')
    paths=dict(run='supervised/run/run.json',supervisor='supervised/supervisor.json',
        burn='supervised/burn-verified.json',gate='supervised/run/downstream/dclm-P300-original-check.json')
    records={}
    for name,rel in paths.items():
        if Path(reviewed[name]['path']).resolve()!=(root/rel).resolve():raise ValueError('Wrong failure receipt location')
        records[name]=read_locked(reviewed[name])
    run,sv,burn,gate=(records[k] for k in ('run','supervisor','burn','gate'))
    if (run['status']!='failed' or len(run['jobs'])!=1 or run['jobs'][0]['name']!='a2-downstream'
            or run['jobs'][0]['returncode']!=1 or sv['training_status']!='failed' or not sv.get('finished_at')
            or len(sv['free_after_training'])!=8 or any(g['pids'] for g in sv['free_after_training'])
            or not burn['collective_progress_verified'] or gate['passed'] is not False
            or gate['production_batch_size']!=8 or gate['manual_max_nats_per_token']>2e-5
            or gate['batch_max_nats_per_token']>.01 or gate['production_max_nats_per_token']<=.01
            or gate['greedy_reference_mismatches'] or gate['greedy_batch_mismatches']
            or list((root/'supervised/run/downstream/scores').glob('*/scores.json'))):
        raise ValueError('Failed numerical check or successful cleanup differs from reviewed evidence')
    return reviewed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','history','assets','sets'):
        p.add_argument('--'+name,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('output','predecessor-burn-script','eval-python','tokenizer','dataset-root'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('host','session','code-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--spent-seconds',type=float,required=True)
    p.add_argument('--failed-handoff');p.add_argument('--failed-handoff-sha256')
    p.add_argument('--timeout-seconds',type=float,default=10800)
    a=p.parse_args();started=time.monotonic();source=Path(__file__).resolve().parents[1]
    remaining_budget(a.spent_seconds,0,a.timeout_seconds)
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host or existing output')
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):raise ValueError('Pinned source required')
    for name in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    for name,expected in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
        if version(name).split('+')[0]!=expected:raise ValueError('Unexpected native runtime')
    if not a.eval_python.is_file():raise ValueError('Existing CPU eval runtime required')
    locks=[dict(path=getattr(a,k),sha256=getattr(a,k+'_sha256')) for k in ('stage','history','assets','sets')]
    stage=read_locked(locks[0]);prior=a.predecessor_burn_script.parents[2]
    required={'supervisor':'supervised/supervisor.json','run_complete':'supervised/run/complete.json',
              'burn':'supervised/burn-verified.json','complete':'supervised/run/remaining/heals/complete.json'}
    failed=None
    if a.failed_handoff:
        if not a.failed_handoff_sha256:raise ValueError('Pinned failure receipt required')
        failed=verify_failed_eval_handoff(a.output.parent,a.predecessor_burn_script,
            dict(path=a.failed_handoff,sha256=a.failed_handoff_sha256))
        prior=Path(stage['supervisor']['path']).parents[1]
        verify_handoff(a.output.parent,prior/'source/resources/llm_pretrain_burn.py',digest)
    else:
        verify_handoff(a.output.parent,a.predecessor_burn_script,digest)
    if any(Path(stage[k]['path']).resolve()!=(prior/rel).resolve() for k,rel in required.items()):
        raise ValueError('Reviewed training stage receipt locations differ')
    if (digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH
            or Path('/mnt/local/_gpu_guard/DISABLED').exists()):raise ValueError('Unknown burn/guard owner')
    a.output.mkdir();frozen=freeze_source(source,a.output,a.code_commit)
    identities=prepare_identities(*locks,a.output/'identities')
    plan=dict(schema=1,identities=identities['identities'],tasks=list(COUNTS),gpus=list(range(8)),
        max_length=4096,batch_size=2,native_python=sys.executable,eval_python=str(a.eval_python.resolve()),
        tokenizer=str(a.tokenizer.resolve()),dataset_root=str(a.dataset_root.resolve()),
        input_spec=lock(frozen/'resources/a2_inputs_20261028.json'),
        manifest=lock(frozen/'resources/downstream_english_20261023.json'))
    if failed:plan['prepared']=failed['prepared']
    validate_plan(plan);plan_lock=write(a.output/'plan.json',plan)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_MODE':'offline',
         'OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0','TOKENIZERS_PARALLELISM':'false',
         'PYTHONPATH':str(frozen),'A2_CODE_COMMIT':a.code_commit}
    from accelerate.commands.config.config_args import default_config_file,load_config_from_file
    cache=Path(default_config_file);cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():shutil.copyfile(cache,a.output/'previous-accelerate.yaml')
    config=frozen/'resources/accelerate_b200.yaml';shutil.copyfile(config,cache)
    actual=load_config_from_file(str(cache))
    if cache.read_bytes()!=config.read_bytes() or actual.num_processes!=8 or actual.mixed_precision!='bf16' or actual.use_cpu:
        raise ValueError('Accelerate mismatch')
    verification=subprocess.run([sys.executable,'-m','accelerate.commands.env'],check=True,capture_output=True,text=True,
        env={**env,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    (a.output/'accelerate-env.log').write_text(verification.stdout+verification.stderr)
    print(verification.stdout,flush=True)
    write(a.output/'phase-budget.json',remaining_budget(a.spent_seconds,time.monotonic()-started,a.timeout_seconds))
    write(a.output/'inspection.json',inspect())
    jobs=a.output/'jobs.json';write(jobs,job_config(plan_lock,a.timeout_seconds));load_jobs(jobs)
    argv=['env',*[k+'='+env[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE',
        'TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')],sys.executable,'-u','-m','scripts.train_then_burn',
        '--config',str(jobs),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
        '--host',a.host,'--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('A2_DOWNSTREAM_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()

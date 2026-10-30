"""Launch one authorized six-model FWE screen after both training parts are terminal.

CPU admission only until the unchanged native supervisor verifies exact burn
identities and free GPUs. The running training source is never modified.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

from efficient_llm.kv_long import verify_handoff
from efficient_llm.a2_stage1b.protocol import SPEC_SHA, REPLICATION_SPEC_SHA
from eval.a2_run import lock, write
from eval.a2_stage1b_final_plan import prepare_plan, validate_policy, validate_time_policy, admit_final_regression
from scripts.launch_a2_stage1b_downstream import bind_predecessor
from eval.downstream_suite import COUNTS
from scripts.a2_remaining import read_locked
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest, BURN_HASH, GUARD_HASH
from run_experiments import load_jobs


def budget(spent, elapsed, expected, time_policy):
    validate_time_policy(time_policy)
    if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (spent,elapsed,expected)) or expected<=0:
        raise ValueError('Invalid report-only time estimate')
    total=spent+elapsed+expected+660
    return dict(prior_seconds=spent,preparation_seconds=elapsed,expected_seconds=expected,
        cleanup_estimate_seconds=660,projected_node_hours=total/3600,report_due=total>20*3600,
        former_24h_exceeded=total>24*3600,hard_node_hours=None,hard_deadline=None,
        timeout_seconds=None,time_policy=time_policy)


def job_config(plan_lock):
    return dict(jobs=[dict(name='stage1b-downstream-final',gpus=list(range(8)),timeout_seconds=None,
        argv=['{python}','-m','eval.a2_stage1b_final_run','--plan',plan_lock['path'],
              '--plan-sha256',plan_lock['sha256'],
              '--output','{run_dir}/downstream'],required_outputs=[
            dict(path='downstream/complete.json',json_equals=dict(status='ok',models=6)),
            dict(path='downstream/results.json',json_equals=dict(status='ok',diagnostic=False)),
            dict(path='downstream/batch_decision.json',json_equals=dict(status='compared',requests=78259)),
            dict(path='downstream/paired_accuracy.json',json_equals=dict(resamples=10000,seed=0))])])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('part1','part2','assets','regression','policy','failed-gate','prepared','time-policy'):
        p.add_argument('--'+name,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('output','predecessor-burn-script','eval-python','tokenizer','dataset-root'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('host','session','code-commit'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--spent-seconds',type=float,required=True)
    p.add_argument('--expected-seconds',type=float,default=7200)
    p.add_argument('--over20h-reported',action='store_true')
    a=p.parse_args();started=time.monotonic();source=Path(__file__).resolve().parents[1]
    def remaining_budget():
        receipt=budget(a.spent_seconds,time.monotonic()-started,a.expected_seconds,dict(path=a.time_policy,sha256=a.time_policy_sha256))
        if receipt['report_due'] and not a.over20h_reported:
            raise ValueError('Send the over20h Stage1 estimate before launch')
        return receipt
    remaining_budget()
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host or existing output')
    if digest(source/'docs/A2_stage1_spec.md')!=SPEC_SHA or digest(source/'docs/A2_stage1b_fwe_spec.md')!=REPLICATION_SPEC_SHA:raise ValueError('Stage1b specification changed')
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):raise ValueError('Pinned source required')
    for name in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    for name,expected in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
        if version(name).split('+')[0]!=expected:raise ValueError('Unexpected native runtime')
    if not a.eval_python.is_file():raise ValueError('Existing CPU eval runtime required')
    locks={name:dict(path=getattr(a,name),sha256=getattr(a,name+'_sha256'))
           for name in ('part1','part2','assets','regression','policy','failed_gate','prepared','time_policy')}
    validate_policy(locks['policy']);validate_time_policy(locks['time_policy'])
    if read_locked(locks['part1'])['phase']!='part1' or read_locked(locks['part2'])['phase']!='part2':
        raise ValueError('Actual terminal receipts from both training parts required')
    regression_admission=admit_final_regression(locks['regression'],source,locks['policy'])
    # Only the final training part can be the immediate GPU predecessor.
    bind_predecessor(locks['part2'],locks['assets'],a.output.parent,a.predecessor_burn_script)
    if (digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH
            or Path('/mnt/local/_gpu_guard/DISABLED').exists()):raise ValueError('Unknown burn/guard owner')
    a.output.mkdir();frozen=freeze_source(source,a.output,a.code_commit,
        extra_files=('docs/A2_stage1_spec.md','docs/A2_stage1b_fwe_spec.md','resources/a2_stage1b_continuation_20261030.json','resources/a2_stage1b_time_policy_20261030.json'))
    # Recheck copied evaluation files against the passing regression and policy.
    if admit_final_regression(locks['regression'],frozen,locks['policy'])!=regression_admission:
        raise ValueError('Frozen evaluation source differs from admission')
    write(a.output/'regression-admission.json',regression_admission)
    common=dict(tasks=list(COUNTS),gpus=list(range(8)),max_length=4096,batch_size=2,
        native_python=sys.executable,eval_python=str(a.eval_python.resolve()),tokenizer=str(a.tokenizer.resolve()),
        dataset_root=str(a.dataset_root.resolve()),input_spec=lock(frozen/'resources/a2_inputs_20261028.json'),
        manifest=lock(frozen/'resources/downstream_english_20261023.json'))
    plan=prepare_plan(locks['part1'],locks['part2'],locks['assets'],a.output/'identities',common=common,
        policy=locks['policy'],time_policy=locks['time_policy'],regression=locks['regression'],failed_gate=locks['failed_gate'],prepared=locks['prepared'])
    plan_lock=write(a.output/'plan.json',plan)
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
    write(a.output/'stage-budget.json',remaining_budget());write(a.output/'inspection.json',inspect())
    jobs=a.output/'jobs.json';write(jobs,job_config(plan_lock));load_jobs(jobs)
    argv=['env',*[k+'='+env[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE',
        'TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')],sys.executable,'-u','-m','scripts.train_then_burn',
        '--config',str(jobs),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
        '--host',a.host,'--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('STAGE1B_FINAL_DOWNSTREAM_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()

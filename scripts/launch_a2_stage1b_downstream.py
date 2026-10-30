"""Launch the authorized registered four/two-checkpoint Stage1b screen after terminal training.

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
from eval.a2_stage1b_plan import prepare_identities, validate_terminal_bundle
from eval.a2_stage1b_run import validate_plan
from eval.a2_stage1b_regression import admit_regression
from eval.downstream_suite import COUNTS
from scripts.a2_remaining import read_locked
from scripts.launch_a2_downstream import freeze_source
from scripts.gpu_process import inspect
from scripts.train_then_burn import digest, BURN_HASH, GUARD_HASH
from run_experiments import load_jobs


def budget(spent, elapsed, requested, deadline, now=None):
    if (not all(type(v) in (int,float) and math.isfinite(v) and v >= 0
                for v in (spent,elapsed,requested)) or not 0 < requested <= 10800):
        raise ValueError('Invalid Stage1 evaluation budget')
    end = datetime.fromisoformat(deadline)
    if end.tzinfo is None:
        raise ValueError('Explicit timezone required for continuation deadline')
    now = now or datetime.now(timezone.utc)
    total = spent+elapsed+requested+660
    if total > 24*3600 or (end-now).total_seconds() < requested+660:
        raise ValueError('Stage1 cap or continuation deadline would be exceeded')
    return dict(prior_seconds=spent,preparation_seconds=elapsed,timeout_seconds=requested,
                cleanup_reserve_seconds=660,stage_hard_seconds=24*3600,
                report_due=total>20*3600,deadline=end.isoformat())


def job_config(plan_lock, timeout, phase):
    if phase not in ("part1","part2"):raise ValueError("Unknown FWE downstream part")
    return dict(jobs=[dict(name='stage1b-downstream-'+phase,gpus=list(range(8)),timeout_seconds=timeout+60,
        argv=['{python}','-m','eval.a2_stage1b_run','--plan',plan_lock['path'],
              '--plan-sha256',plan_lock['sha256'],'--timeout-seconds',str(timeout),
              '--output','{run_dir}/downstream'],required_outputs=[
            dict(path='downstream/complete.json',json_equals=dict(status='ok',models=4 if phase=="part1" else 2)),
            dict(path='downstream/results.json',json_equals=dict(status='ok',diagnostic=False)),
            dict(path='downstream/paired_accuracy.json',json_equals=dict(resamples=10000,seed=0))])])


def bind_predecessor(stage_lock, assets_lock, output_parent, burn_script):
    stage,_,_,_=validate_terminal_bundle(stage_lock,assets_lock)
    prior=burn_script.resolve().parents[2]
    if prior.parent != output_parent.resolve() or burn_script.resolve()!=prior/'source/resources/llm_pretrain_burn.py':
        raise ValueError('Unexpected Stage1 predecessor')
    fixed={'supervisor':'supervised/supervisor.json','run_complete':'supervised/run/complete.json',
           'burn':'supervised/burn-verified.json'}
    for key,name in [('results','runs.json'),('choice','lr_choice.json'),('tests','tests.json'),
                     ('decision','decision_stage1_fwe.json'),('complete','complete.json')]:
        fixed[key]='supervised/run/results/a2_stage1_fwe/'+name
    if any(Path(stage[key]['path']).resolve()!=prior/rel for key,rel in fixed.items()):
        raise ValueError('Stage1 receipts belong to a different predecessor')
    verify_handoff(output_parent,burn_script,digest)
    return prior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','assets','regression'):
        p.add_argument('--'+name,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('output','predecessor-burn-script','eval-python','tokenizer','dataset-root'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('host','session','code-commit','deadline'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=('part1','part2'),required=True)
    p.add_argument('--spent-seconds',type=float,required=True)
    p.add_argument('--timeout-seconds',type=float,default=7200)
    p.add_argument('--over20h-reported',action='store_true')
    p.add_argument('--prepared');p.add_argument('--prepared-sha256')
    a=p.parse_args();started=time.monotonic();source=Path(__file__).resolve().parents[1]
    def remaining_budget():
        receipt=budget(a.spent_seconds,time.monotonic()-started,a.timeout_seconds,a.deadline)
        if receipt['report_due'] and not a.over20h_reported:
            raise ValueError('Send the over20h Stage1 estimate before launch')
        return receipt
    remaining_budget()
    if socket.gethostname()!=a.host or a.output.exists():raise ValueError('Wrong host or existing output')
    if digest(source/'docs/A2_stage1_spec.md')!=SPEC_SHA or digest(source/'docs/A2_stage1b_fwe_spec.md')!=REPLICATION_SPEC_SHA:raise ValueError('Stage1b specification changed')
    if len(a.code_commit)!=40 or any(c not in '0123456789abcdef' for c in a.code_commit):raise ValueError('Pinned source required')
    if bool(a.prepared)!=bool(a.prepared_sha256):raise ValueError('Reused requests require a SHA lock')
    if a.phase=='part2' and not a.prepared:raise ValueError('Reuse Part1 canonical requests for Part2')
    for name in (a.session,a.session+'_burn'):
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session')
    for name,expected in [('torch','2.14.1'),('transformers','4.57.1'),('accelerate','1.13.0')]:
        if version(name).split('+')[0]!=expected:raise ValueError('Unexpected native runtime')
    if not a.eval_python.is_file():raise ValueError('Existing CPU eval runtime required')
    stage_lock=dict(path=a.stage,sha256=a.stage_sha256);assets_lock=dict(path=a.assets,sha256=a.assets_sha256)
    if read_locked(stage_lock).get('phase')!=a.phase:raise ValueError('Training/eval phase mismatch')
    regression_lock=dict(path=a.regression,sha256=a.regression_sha256)
    rb,rv,_=admit_regression(regression_lock,source)
    if a.phase=='part1':
        if rv['training_predecessor']!=dict(stage=stage_lock,assets=assets_lock):
            raise ValueError('Regression followed a different training run')
        prior=a.predecessor_burn_script.resolve().parents[2]
        if (prior.parent!=a.output.parent.resolve()
                or a.predecessor_burn_script.resolve()!=prior/'source/resources/llm_pretrain_burn.py'
                or Path(rb['supervisor']['path']).resolve()!=prior/'supervised/supervisor.json'
                or Path(rb['burn']['path']).resolve()!=prior/'supervised/burn-verified.json'):
            raise ValueError('Wrong regression predecessor burn')
        verify_handoff(a.output.parent,a.predecessor_burn_script,digest)
    else:
        bind_predecessor(stage_lock,assets_lock,a.output.parent,a.predecessor_burn_script)
    if (digest(a.predecessor_burn_script)!=BURN_HASH or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH
            or Path('/mnt/local/_gpu_guard/DISABLED').exists()):raise ValueError('Unknown burn/guard owner')
    a.output.mkdir();frozen=freeze_source(source,a.output,a.code_commit,extra_files=('docs/A2_stage1_spec.md','docs/A2_stage1b_fwe_spec.md'))
    identities=prepare_identities(stage_lock,assets_lock,a.output/'identities')
    plan=dict(schema=2,phase=a.phase,identities=identities['identities'],identity_bundle=lock(a.output/'identities/identities.json'),
        tasks=list(COUNTS),gpus=list(range(8)),max_length=4096,batch_size=2,
        native_python=sys.executable,eval_python=str(a.eval_python.resolve()),tokenizer=str(a.tokenizer.resolve()),
        dataset_root=str(a.dataset_root.resolve()),input_spec=lock(frozen/'resources/a2_inputs_20261028.json'),
        manifest=lock(frozen/'resources/downstream_english_20261023.json'))
    if a.prepared:
        plan['prepared']=dict(path=a.prepared,sha256=a.prepared_sha256);read_locked(plan['prepared'])
    validate_plan(plan);plan['regression']=regression_lock;plan_lock=write(a.output/'plan.json',plan)
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
    jobs=a.output/'jobs.json';write(jobs,job_config(plan_lock,a.timeout_seconds,a.phase));load_jobs(jobs)
    argv=['env',*[k+'='+env[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','WANDB_MODE','OMP_NUM_THREADS','NCCL_NVLS_ENABLE',
        'TOKENIZERS_PARALLELISM','PYTHONPATH','A2_CODE_COMMIT')],sys.executable,'-u','-m','scripts.train_then_burn',
        '--config',str(jobs),'--output',str(a.output/'supervised'),'--inspection',str(a.output/'inspection.json'),
        '--host',a.host,'--burn-session',a.session+'_burn','--predecessor-burn-script',str(a.predecessor_burn_script)]
    command='exec '+shlex.join(argv)+' >'+shlex.quote(str(a.output/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(frozen),command],env=env,check=True)
    subprocess.run(['tmux','set-option','-w','-t',a.session,'remain-on-exit','on'],check=True)
    print('STAGE1B_DOWNSTREAM_SUBMITTED',a.output,flush=True)


if __name__=='__main__':main()

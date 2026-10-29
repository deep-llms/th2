"""Run the fixed A2 downstream screen inside an already-owned GPU allocation.

The unchanged outer supervisor must verify training completion, acquire GPUs,
copy/verify Accelerate configuration, and restore burns. This coordinator never
acquires or stops a GPU workload and never changes an environment.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from efficient_llm.diagnostic_data import sha256
from eval.a2_checkpoint import verify_identity
from eval.a2_request_bridge import digest
from eval.a2_suite import scoring_jobs, execute_scoring, collect_files
from eval.downstream_suite import COUNTS
from scripts.a2_remaining import read_locked


def lock(path):
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with path.open('x') as stream:stream.write(encoded)
    return lock(path)


def validate_plan(plan):
    expected={}
    for ds in ('dclm','fwe','nem'):
        for parent in ('P300','P1T'):
            expected[f'{ds}-{parent}-original']=(ds,parent,'parent',None)
            for budget in (('0.2B','0.8B') if ds=='nem' else ('0.2B',)):
                expected[f'{ds}-{parent}-S1-{budget}']=(ds,parent,'healed',budget)
    if (plan.get('schema')!=1 or set(plan['identities'])!=set(expected)
            or plan.get('tasks')!=list(COUNTS) or plan.get('max_length')!=4096
            or plan.get('gpus')!=list(range(8)) or plan.get('batch_size',8) not in (1,2,8)):
        raise ValueError('Require the fixed fourteen-model nine-task screen')
    identities={}
    for name,item in plan['identities'].items():
        identity=read_locked(item);verify_identity(identity)
        ds,parent,kind,budget=expected[name]
        if (identity['dataset'],identity['parent'],identity['kind'])!=(ds,parent,kind):
            raise ValueError('Checkpoint label does not match identity')
        if kind=='healed':
            row=read_locked(identity['training_receipts']['result'])
            if (row['set'],row['budget'],row['teacher'])!=('S1',budget,parent):
                raise ValueError('Wrong preregistered checkpoint')
        identities[name]=identity
    for name in ('native_python','eval_python','tokenizer','dataset_root'):
        if not Path(plan[name]).is_absolute():raise ValueError('Explicit absolute runtime/input paths required')
    for name in ('manifest','input_spec'):read_locked(plan[name])
    return identities


def run_screen(plan, output, *, timeout_seconds):
    return _run_screen(plan, output, timeout_seconds=timeout_seconds,
        validator=validate_plan, gate_names=('dclm-P300-original','dclm-P300-S1-0.2B'))


def _run_screen(plan, output, *, timeout_seconds, validator, gate_names, finalize=None):
    """Shared execution; the screen-specific validator fixes model identities."""
    if not math.isfinite(timeout_seconds) or timeout_seconds<=0:raise ValueError('Finite budget required')
    deadline=time.monotonic()+timeout_seconds
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('Downstream screen budget exhausted')
        return value
    identities=validator(plan);root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',plan)
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
         'TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'offline','TOKENIZERS_PARALLELISM':'false'}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
    def command(argv, logname, gpu=None):
        child={**env,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)}
        with (root/logname).open('x') as stream:
            subprocess.run(argv,env=child,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=remaining())
    bridge=[plan['eval_python'],'-m','eval.a2_requests']
    common=['--tokenizer',plan['tokenizer'],'--input-spec',plan['input_spec']['path'],
        '--dataset-root',plan['dataset_root'],'--manifest',plan['manifest']['path'],
        '--max-length','4096','--tasks',*plan['tasks']]
    prepared_path=root/'prepared.json'
    if plan.get('prepared'):
        reused=read_locked(plan['prepared']);protocol=reused['protocol']
        if (protocol['tasks']!=plan['tasks'] or protocol['max_length']!=4096 or protocol['limit'] is not None
                or protocol['seed']!=42 or protocol['num_fewshot']!=0
                or protocol['manifest_sha256']!=plan['manifest']['sha256']
                or protocol['tokenizer_files']!=read_locked(plan['input_spec'])['tokenizer_provenance']['gpt_neox']['files']
                or reused['request_sha256']!=digest(reused['requests'])
                or reused['protocol_sha256']!=digest(protocol)):
            raise ValueError('Reused canonical request protocol changed')
        with prepared_path.open('xb') as stream:stream.write(Path(plan['prepared']['path']).read_bytes())
    else:
        command([*bridge,'prepare',*common,'--output',str(prepared_path)],'prepare.log')
    prepared_lock=lock(prepared_path);prepared=read_locked(prepared_lock)
    # Both depth/load paths are checked on actual weights, once, before scoring.
    for name in gate_names:
        item=plan['identities'][name];target=root/(name+'-check.json')
        command([plan['native_python'],'-m','eval.a2_check','--identity',item['path'],
            '--identity-sha256',item['sha256'],'--prepared',prepared_lock['path'],
            '--prepared-sha256',prepared_lock['sha256'],'--batch-size',str(plan.get('batch_size',8)),'--output',str(target)],name+'-check.log',0)
        check=json.loads(target.read_text())
        if (check.get('passed') is not True or check.get('production_batch_size')!=plan.get('batch_size',8)
                or check['checkpoint_identity_sha256']!=digest(identities[name])
                or check['prepared_sha256']!=digest(prepared)):
            raise ValueError('Actual-weight numerical gate failed or mismatched')
    jobs=scoring_jobs(plan['identities'],prepared_lock,root/'scores',plan['native_python'],plan['gpus'],batch_size=plan.get('batch_size',8))
    execute_scoring(jobs,timeout_seconds=remaining())
    scores={j['name']:lock(j['output']) for j in jobs};replays={}
    for name,score in scores.items():
        target=root/'replays'/(name+'.json')
        command([*bridge,'replay',*common,'--prepared',prepared_lock['path'],
            '--scored',score['path'],'--output',str(target)],name+'-replay.log')
        replays[name]=lock(target)
    # Recheck the original locks after all children finish, then bind every result.
    for name,item in plan['identities'].items():
        if read_locked(item)!=identities[name]:raise ValueError('Identity changed during screen')
    for name in ('manifest','input_spec'):read_locked(plan[name])
    result=collect_files(identities,prepared_lock,scores,replays)
    write(root/'results.json',result)
    extra={} if finalize is None else finalize(plan,root,result,replays)
    write(root/'complete.json',dict(status='ok',models=len(identities),
        prepared=prepared_lock,scores=scores,replays=replays,results=lock(root/'results.json'),
        elapsed_seconds=timeout_seconds-remaining(),additional_results=extra))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--timeout-seconds',type=float,required=True);a=p.parse_args()
    run_screen(read_locked(dict(path=a.plan,sha256=a.plan_sha256)),a.output,timeout_seconds=a.timeout_seconds)


if __name__=='__main__':main()

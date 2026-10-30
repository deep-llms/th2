"""One six-model FWE screen after all registered heals, under native ownership."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from eval.a2_run import lock,write
from eval.a2_request_bridge import digest
from eval.a2_suite import scoring_jobs,execute_scoring,collect_files
from eval.a2_stage1b_final_plan import validate_plan
from eval.a2_stage1b_stats import paired_report
from scripts.a2_remaining import read_locked


def execute_final_scoring(jobs,*,timeout_seconds):
    if timeout_seconds is not None:
        return execute_scoring(jobs,timeout_seconds=timeout_seconds)
    # Same native scorer argv/one worker per allocated lane. Latest user policy
    # removes elapsed-time termination; exceptions still fail this screen honestly.
    from concurrent.futures import ThreadPoolExecutor
    lanes={}
    for job in jobs:lanes.setdefault(job['gpu'],[]).append(job)
    if len(lanes)>8:raise ValueError('Too many allocated lanes')
    def lane(items):
        for job in items:
            target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('Score output exists')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu']),'CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
            with target.with_suffix('.log').open('x') as log:
                subprocess.run(job['argv'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not target.is_file():raise ValueError('Native scorer produced no artifact')
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,items) for items in lanes.values()]
        for future in futures:future.result()


def run_screen(plan,output,*,timeout_seconds=None):
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds<=0):
        raise ValueError('Invalid optional test timeout')
    started=time.monotonic();deadline=None if timeout_seconds is None else started+timeout_seconds
    def remaining():
        if deadline is None:return None
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('Final downstream budget exhausted')
        return value
    identities=validate_plan(plan);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',plan)
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'WANDB_MODE':'offline','TOKENIZERS_PARALLELISM':'false','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(k,None)
    def command(argv,logname,gpu=None):
        with (root/logname).open('x') as log:
            subprocess.run(argv,env={**env,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining())
    # Byte-for-byte reuse of the original full Stage1 request artifact.
    read_locked(plan['prepared']);prepared_path=root/'prepared.json'
    with prepared_path.open('xb') as f:f.write(Path(plan['prepared']['path']).read_bytes())
    prepared_lock=lock(prepared_path);prepared=read_locked(prepared_lock)
    if prepared_lock['sha256']!=plan['prepared']['sha256']:raise ValueError('Prepared copy changed')
    gates={}
    for name,item in plan['identities'].items():
        target=root/(name+'-check.json')
        command([plan['native_python'],'-m','eval.a2_stage1b_check','--identity',item['path'],
            '--identity-sha256',item['sha256'],'--prepared',prepared_lock['path'],
            '--prepared-sha256',prepared_lock['sha256'],'--output',str(target)],name+'-check.log',0)
        gate=json.loads(target.read_text())
        if (gate.get('passed') is not True or gate['production_batch_size']!=2
                or gate['checkpoint_identity_sha256']!=digest(identities[name])
                or gate['prepared_sha256']!=digest(prepared)):
            raise ValueError('Actual-weight revised gate failed or mismatched')
        gates[name]=lock(target)
    parent=plan['identities']['P300-parent'];jobs=[]
    predictions={}
    for batch in (1,2):
        target=root/'batch_comparison'/str(batch)/'scores.json';pred=target.with_name('predictions.json')
        jobs.append(dict(name=str(batch),gpu=batch-1,output=str(target),argv=[plan['native_python'],
            '-m','eval.a2_stage1b_parent_score','--identity',parent['path'],'--identity-sha256',parent['sha256'],
            '--prepared',prepared_lock['path'],'--prepared-sha256',prepared_lock['sha256'],
            '--batch-size',str(batch),'--output',str(target),'--predictions-output',str(pred)]))
        predictions[str(batch)]=pred
    execute_final_scoring(jobs,timeout_seconds=remaining())
    parent_scores={j['name']:lock(j['output']) for j in jobs}
    predictions={k:lock(v) for k,v in predictions.items()}
    bridge=[plan['eval_python'],'-m','eval.a2_requests']
    common=['--tokenizer',plan['tokenizer'],'--input-spec',plan['input_spec']['path'],
        '--dataset-root',plan['dataset_root'],'--manifest',plan['manifest']['path'],
        '--max-length','4096','--tasks',*plan['tasks']]
    def replay(name,score):
        target=root/'replays'/(name+'.json')
        command([*bridge,'replay',*common,'--prepared',prepared_lock['path'],
            '--scored',score['path'],'--output',str(target)],name+'-replay.log')
        return lock(target)
    parent_replays={k:replay('P300-parent-batch'+k,v) for k,v in parent_scores.items()}
    comparison_inputs=write(root/'batch_comparison_inputs.json',dict(identity=parent,prepared=prepared_lock,
        scores=parent_scores,replays=parent_replays,predictions=predictions))
    decision_path=root/'batch_decision.json'
    command([plan['eval_python'],'-m','eval.a2_stage1b_batch_report','--inputs',comparison_inputs['path'],
        '--inputs-sha256',comparison_inputs['sha256'],'--dataset-root',plan['dataset_root'],
        '--manifest',plan['manifest']['path'],'--output',str(decision_path)],'batch-comparison.log')
    decision_lock=lock(decision_path);decision=read_locked(decision_lock)
    batch=decision['production_batch_size']
    if (decision['status']!='compared' or decision['requests']!=78259 or batch not in (1,2)
            or decision.get('hard_batch_passed') is not True
            or decision['inputs']!=read_locked(comparison_inputs)
            or (batch==2)!=decision['batch2_accepted']):raise ValueError('Incomplete batch decision')
    # No other production scoring until the complete P300 decision exists.
    other={k:v for k,v in plan['identities'].items() if k!='P300-parent'}
    jobs=scoring_jobs(other,prepared_lock,root/'scores',plan['native_python'],plan['gpus'],batch_size=batch)
    execute_final_scoring(jobs,timeout_seconds=remaining())
    scores={'P300-parent':parent_scores[str(batch)],**{j['name']:lock(j['output']) for j in jobs}}
    replays={'P300-parent':parent_replays[str(batch)]}
    replays.update({name:replay(name,scores[name]) for name in other})
    if validate_plan(plan)!=identities:raise ValueError('Identity changed during screen')
    if any(read_locked(v)['batch_size']!=batch for v in scores.values()):raise ValueError('Mixed production batches')
    result=collect_files(identities,prepared_lock,scores,replays)
    result.update(production_batch_size=batch,batch_decision=decision_lock,revised_gates=gates,
                  policy=plan['policy'],failed_gate=plan['failed_gate'])
    result_lock=write(root/'results.json',result)
    paired=write(root/'paired_accuracy.json',paired_report({k:read_locked(v) for k,v in replays.items()},'final'))
    write(root/'complete.json',dict(status='ok',models=6,prepared=prepared_lock,scores=scores,replays=replays,
        results=result_lock,production_batch_size=batch,batch_decision=decision_lock,revised_gates=gates,
        elapsed_seconds=time.monotonic()-started,time_policy=plan['time_policy'],additional_results=dict(paired_accuracy=paired)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','plan-sha256','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args()
    run_screen(read_locked(dict(path=a.plan,sha256=a.plan_sha256)),a.output)


if __name__=='__main__':main()

"""Continue the registered FWE screen, reporting failed checks without aborting.

The native supervisor owns GPU acquisition/cleanup. Original scorers and numeric
kernels stay frozen. An unusable artifact is flagged and excluded, never invented;
independent models and subsequent possible checks continue without user input.
"""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from eval.a2_run import lock, write
from eval.a2_request_bridge import digest
from eval.a2_suite import scoring_jobs, collect_files
from eval.a2_stage1b_final_plan import validate_plan
from eval.a2_stage1b_stats import paired_report
from scripts.a2_remaining import read_locked


class Journal:
    def __init__(self, root):
        self.root=Path(root);self.events=[];self.mutex=threading.Lock()

    def record(self, name, passed, note, diagnostic=False):
        event=dict(check=name,passed=bool(passed),diagnostic=diagnostic,note=note)
        with self.mutex:
            self.events.append(event)
            with (self.root/'checks.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print('REPORTED_CHECK',json.dumps(event),flush=True)
        return passed

    def attempt(self, name, fn):
        try:
            value=fn()
            self.record(name,True,'Completed')
            return value
        except Exception as error:
            self.record(name,False,repr(error)+'; continue remaining registered work')
            return None


def require(condition, message):
    if not condition:raise ValueError(message)


def run(plan, output):
    started=time.monotonic();root=Path(output);root.mkdir(parents=True,exist_ok=False)
    journal=Journal(root);write(root/'plan.json',plan)
    journal.attempt('plan_validation',lambda:validate_plan(plan))
    reuse=read_locked(plan['continuation']);policy=read_locked(reuse['policy'])
    decision=read_locked(reuse['decision'])
    # Preserve original false hard_batch_passed value and the original decision SHA.
    write(root/'batch_decision.json',decision)
    write(root/'option2.json',dict(policy=reuse['policy'],prior_decision=reuse['decision'],
        production_batch_size=2,full_set_ll_diagnostic=True,
        max_nats_per_token=decision['full_max_nats_per_token'],worst_request_index=71485,
        token_mismatch_rate=decision['full_greedy_token_mismatch_rate']))
    journal.record('full_set_LL',decision['hard_batch_passed'],
        'Diagnostic only under Option 2; max='+str(decision['full_max_nats_per_token'])+
        '; token mismatch rate='+str(decision['full_greedy_token_mismatch_rate']),diagnostic=True)
    journal.record('accuracy_rule',decision['batch2_accepted'],'User selected batch 2')
    prepared_lock=plan['prepared'];prepared=read_locked(prepared_lock)
    ids={}
    for name,item in plan['identities'].items():
        value=journal.attempt('identity:'+name,lambda item=item:read_locked(item))
        if value is not None:ids[name]=value
    gates={}
    from scripts.verify_a2_stage1b_final import verify_gate
    for name,item in reuse['gates'].items():
        journal.attempt('representative_gate:'+name,lambda name=name,item=item:
            verify_gate(read_locked(item),ids[name],prepared))
        gates[name]=item
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'WANDB_MODE':'offline','TOKENIZERS_PARALLELISM':'false','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
    def command(argv,log,gpu=None):
        with (root/log).open('x') as stream:
            result=subprocess.run(argv,env={**env,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu)},
                stdout=stream,stderr=subprocess.STDOUT,check=False)
        journal.record('process:'+log,result.returncode==0,'returncode='+str(result.returncode))
        return result.returncode
    scores={};replays={};independent={}
    parent_score=decision['inputs']['scores']['2'];parent_replay=decision['inputs']['replays']['2']
    def parent_reuse():
        value=read_locked(parent_score)
        require(value['batch_size']==2 and value['checkpoint_identity_sha256']==digest(ids['P300-parent'])
            and value['prepared_sha256']==digest(prepared),'Parent reuse identity/batch mismatch')
        collect_files({'P300-parent':ids['P300-parent']},prepared_lock,
            {'P300-parent':parent_score},{'P300-parent':parent_replay})
        scores['P300-parent']=parent_score;replays['P300-parent']=parent_replay
    journal.attempt('reuse_P300_parent_batch2',parent_reuse)
    other={n:v for n,v in plan['identities'].items() if n!='P300-parent'}
    def score_one(index,name,item):
        def work():
            job=scoring_jobs({name:item},prepared_lock,root/'scores',plan['native_python'],plan['gpus'],batch_size=2)[0]
            # A distinct allocated GPU for each of the five registered models.
            command(job['argv'],name+'-score.log',plan['gpus'][index])
            target=lock(job['output']);value=read_locked(target)
            require(value['batch_size']==2 and value['checkpoint_identity_sha256']==digest(ids[name])
                and value['prepared_sha256']==digest(prepared) and len(value['scores'])==78259,
                'Incomplete or mismatched production artifact')
            return name,target
        return journal.attempt('score:'+name,work)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(score_one,i,n,v) for i,(n,v) in enumerate(other.items())]
        for future in futures:
            value=future.result()
            if value is not None:scores[value[0]]=value[1]
    common=['--tokenizer',plan['tokenizer'],'--input-spec',plan['input_spec']['path'],
        '--dataset-root',plan['dataset_root'],'--manifest',plan['manifest']['path'],
        '--max-length','4096','--tasks',*plan['tasks']]
    def replay(name,score,folder):
        target=root/folder/(name+'.json')
        command([plan['eval_python'],'-m','eval.a2_requests','replay',*common,
            '--prepared',prepared_lock['path'],'--scored',score['path'],'--output',str(target)],name+'-'+folder+'.log')
        item=lock(target)
        collect_files({name:ids[name]},prepared_lock,{name:score},{name:item})
        return item
    for name,score in scores.items():
        if name not in replays:
            item=journal.attempt('replay:'+name,lambda name=name,score=score:replay(name,score,'replays'))
            if item is not None:replays[name]=item
    eligible={n:ids[n] for n in scores if n in ids and n in replays}
    result=journal.attempt('collect_results',lambda:collect_files(eligible,prepared_lock,
        {n:scores[n] for n in eligible},{n:replays[n] for n in eligible})) if eligible else None
    if result is None:result=dict(status='incomplete',diagnostic=True,scores={})
    all_models=set(eligible)==set(plan['identities'])
    result.update(production_batch_size=2,batch_decision=reuse['decision'],revised_gates=gates,
        continuation=plan['continuation'],all_registered_models=all_models)
    result_lock=write(root/'results.json',result)
    pairs=journal.attempt('paired_bootstrap',lambda:paired_report({n:read_locked(v) for n,v in replays.items()},'final'))
    paired_lock=write(root/'paired_accuracy.json',pairs if pairs is not None else
        dict(status='unavailable',reason='See checks.jsonl; no fabricated CI'))
    # Fresh harness replay of raw scores, including the reused P300 parent.
    # Every model is attempted independently even if another replay fails.
    for name,score in scores.items():
        item=journal.attempt('independent_replay:'+name,lambda name=name,score=score:replay(name,score,'independent_replays'))
        if item is not None:independent[name]=item
    verification_inputs=write(root/'verification_inputs.json',dict(identities=plan['identities'],prepared=prepared_lock,
        scores=scores,replays=replays,independent_replays=independent,results=result_lock,paired=paired_lock,
        decision=reuse['decision']))
    verification_path=root/'independent_verification.json'
    journal.attempt('independent_verification_process',lambda:command([plan['native_python'],'-m','scripts.verify_a2_stage1b_option2','--inputs',verification_inputs['path'],
        '--inputs-sha256',verification_inputs['sha256'],'--output',str(verification_path)],'independent-verification.log'))
    verification=journal.attempt('independent_verification_artifact',lambda:read_locked(lock(verification_path)))
    verified=bool(verification and verification['passed'])
    journal.record('independent_verification',verified,'See independent_verification.json')
    scientific_complete=all_models and pairs is not None and verified
    checks=write(root/'checks.json',journal.events)
    done=dict(status='ok' if scientific_complete else 'completed_with_issues',execution_complete=True,
        scientific_complete=scientific_complete,models=len(eligible),expected_models=6,prepared=prepared_lock,
        scores=scores,replays=replays,independent_replays=independent,results=result_lock,production_batch_size=2,
        batch_decision=reuse['decision'],revised_gates=gates,checks=checks,continuation=plan['continuation'],
        time_policy=plan['time_policy'],elapsed_seconds=time.monotonic()-started,
        additional_results=dict(paired_accuracy=paired_lock),
        independent_verification=lock(verification_path) if verification_path.exists() else None)
    write(root/'complete.json',done)
    report=['# FWE Option 2 downstream results','',
        'Execution completed; scientific_complete='+str(scientific_complete)+'.',
        'Batch 2. P300-parent scores reused; full-set LL is diagnostic only.',
        'Full max 0.10355663299560547 nats/token at request 71485; token mismatch 0.09372587936927997%.','',
        '## Results','', '```json',json.dumps(result,indent=2),'```','',
        '## Paired bootstrap','', '```json',json.dumps(pairs,indent=2),'```','',
        '## Checks and remaining issues','']
    report.extend('- '+e['check']+': '+('PASS' if e['passed'] else 'FLAG')+'; '+e['note'] for e in journal.events)
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    return done

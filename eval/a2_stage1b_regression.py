"""One original DCLM parent regression through the shared candidate scoring engine.

A mismatch is a scientific gate result, not a crashed job. The native supervisor
still restores its burn; FWE downstream must inspect regression.passed separately.
"""
from pathlib import Path
import math
from eval.a2_run import _run_screen, lock, write
from eval.a2_checkpoint import verify_identity
from eval.a2_suite import collect_files
from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS, PRIMARY
from scripts.a2_remaining import read_locked

NAME = 'P300-parent'


def reference_inputs(reference_lock):
    ref = read_locked(reference_lock)
    source = read_locked(ref['source_manifest'])
    if source['code_commit'] != '13686edd9cb8d7914b0f2c44c3c362c435a808f6':
        raise ValueError('Wrong original Stage1 downstream source')
    old_plan = read_locked(ref['plan'])
    done = read_locked(ref['complete'])
    if done.get('status') != 'ok' or done.get('models') != 6:
        raise ValueError('Original Stage1 screen incomplete')
    identity_lock = old_plan['identities'][NAME]
    identity = read_locked(identity_lock)
    verify_identity(identity)
    if (identity['dataset'], identity['parent'], identity['kind']) != ('dclm','P300','parent'):
        raise ValueError('Regression is restricted to original DCLM P300 parent')
    prepared = read_locked(done['prepared'])
    if len(prepared['requests']) != 78259 or old_plan['batch_size'] != 2:
        raise ValueError('Wrong original protocol')
    # Validate all score/replay links and full task coverage using the common collector.
    original = collect_files({NAME: identity}, done['prepared'],
        {NAME: done['scores'][NAME]}, {NAME: done['replays'][NAME]})
    return ref, old_plan, done, identity_lock, identity, original


def make_plan(reference_lock, *, native_python, eval_python, tokenizer, dataset_root,
              input_spec, manifest):
    _, old, done, identity_lock, _, _ = reference_inputs(reference_lock)
    if input_spec['sha256'] != old['input_spec']['sha256'] or manifest['sha256'] != old['manifest']['sha256']:
        raise ValueError('Changed original task/tokenizer assets')
    return dict(schema=1, purpose='stage1b_dclm_parent_regression', reference=reference_lock,
        identities={NAME: identity_lock}, prepared=done['prepared'], tasks=list(COUNTS),
        max_length=4096, batch_size=2, gpus=list(range(8)), native_python=native_python,
        eval_python=eval_python,tokenizer=tokenizer,dataset_root=dataset_root,
        input_spec=input_spec,manifest=manifest)


def validate_plan(plan):
    _, old, done, identity_lock, identity, _ = reference_inputs(plan['reference'])
    if (plan.get('purpose') != 'stage1b_dclm_parent_regression'
            or plan['identities'] != {NAME: identity_lock} or plan['prepared'] != done['prepared']
            or plan['tasks'] != list(COUNTS) or plan['batch_size'] != 2
            or plan['max_length'] != 4096 or plan['gpus'] != list(range(8))):
        raise ValueError('Regression plan changed')
    for key in ('input_spec','manifest'):
        read_locked(plan[key])
        if plan[key]['sha256'] != old[key]['sha256']:raise ValueError('Changed benchmark inputs')
    for key in ('native_python','eval_python','tokenizer','dataset_root'):
        if not Path(plan[key]).is_absolute():raise ValueError('Absolute local inputs required')
    return {NAME: identity}


def compare(old_scores, new_scores, old_results, new_results):
    for key in ('request_sha256','prepared_sha256','checkpoint_identity_sha256',
                'likelihood_requests','batch_size','precision','max_length'):
        if old_scores[key] != new_scores[key]:raise ValueError('Unaligned score metadata: '+key)
    a,b=old_scores['scores'],new_scores['scores']
    if not a or len(a)!=len(b) or len(a)!=old_scores['likelihood_requests']:
        raise ValueError('Incomplete score coverage')
    differences=[]
    for x,y in zip(a,b):
        if not all(math.isfinite(v) for v in (x[0],y[0])):raise ValueError('Nonfinite likelihood')
        differences.append(y[0]-x[0])
    index=max(range(len(a)),key=lambda i:abs(differences[i]))
    tasks={}
    for task in COUNTS:
        x,y=old_results['scores'][NAME][task],new_results['scores'][NAME][task]
        if (x['primary_metric']!=PRIMARY[task] or y['primary_metric']!=PRIMARY[task]
                or x['examples']!=COUNTS[task] or y['examples']!=COUNTS[task]
                or x['prompt_sha256']!=y['prompt_sha256']):raise ValueError('Unpaired task: '+task)
        tasks[task]=dict(metric=PRIMARY[task],old=x['accuracy'],new=y['accuracy'],
            difference=y['accuracy']-x['accuracy'],identical=x['accuracy']==y['accuracy'],
            accuracy_fields={k:dict(old=x['metrics'][k],new=y['metrics'][k])
                for k in ('acc,none','acc_norm,none') if k in x['metrics']})
    passed=all(t['identical'] for t in tasks.values())
    return dict(status='compared',passed=passed,tasks=tasks,requests=len(a),
        max_abs_loglikelihood_difference=abs(differences[index]),
        max_difference_index=index,signed_difference_at_max=differences[index],
        greedy_mismatches=sum(x[1]!=y[1] for x,y in zip(a,b)),
        failure_action=None if passed else 'hold_fwe_downstream_send_interim_continue_part2_heals')


def finalize(plan, root, result, replay_locks):
    validate_plan(plan)
    ref, _, done, _, _, old_result=reference_inputs(plan['reference'])
    old_scores=read_locked(done['scores'][NAME])
    new_lock=lock(root/'scores'/NAME/'scores.json');new_scores=read_locked(new_lock)
    report=compare(old_scores,new_scores,old_result,result)
    prepared=read_locked(plan['prepared']);request=prepared['requests'][report['max_difference_index']]
    # Match the canonical text pair to harness samples without exposing dataset text.
    replay=read_locked(replay_locks[NAME]);matches=[]
    for task,samples in replay['samples'].items():
        for sample in samples:
            for choice,args in enumerate(sample['arguments']):
                if list(args)==request['text']:matches.append(dict(task=task,doc_id=sample['doc_id'],choice=choice))
    report.update(max_request_sha256=digest(request),max_request_locations=matches,
        source_reference=ref,original_scores=done['scores'][NAME],candidate_scores=new_lock,
        candidate_replay=replay_locks[NAME],prepared=plan['prepared'],
        original_packages=old_scores['packages'],candidate_packages=new_scores['packages'])
    return dict(regression=write(root/'regression.json',report))


def main():
    import argparse
    p=argparse.ArgumentParser()
    for key in ('plan','plan-sha256','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--timeout-seconds',type=float,required=True);a=p.parse_args()
    _run_screen(read_locked(dict(path=a.plan,sha256=a.plan_sha256)),a.output,
        timeout_seconds=a.timeout_seconds,validator=validate_plan,gate_names=(NAME,),finalize=finalize)



def admit_regression(bundle_lock, source):
    """Require an actual successful regression and its native terminal handoff.

    Candidate evaluation/model files must be identical to the re-scored source;
    later launcher/report changes cannot silently change the scoring path.
    """
    from efficient_llm.diagnostic_data import sha256
    bundle=read_locked(bundle_lock)
    v={k:read_locked(bundle[k]) for k in ('source_manifest','complete','supervisor','run_complete','burn','training_predecessor')}
    done=v['complete'];sv=v['supervisor'];terminal=v['run_complete']
    if (done.get('status')!='ok' or done.get('models')!=1
            or sv.get('training_status')!='ok' or sv.get('training_returncode')!=0
            or not v['burn'].get('collective_progress_verified')
            or terminal.get('status')!='ok' or len(terminal.get('jobs',[]))!=1):
        raise ValueError('Regression native handoff incomplete')
    j=terminal['jobs'][0]
    if j.get('name')!='stage1b-regression' or j.get('returncode')!=0 or j.get('status')!='ok':
        raise ValueError('Wrong regression job')
    base=Path(bundle['run_complete']['path']).resolve().parent
    target=Path(done['additional_results']['regression']['path']).resolve()
    found=set()
    for art in j['artifacts']:
        p=(base/art['path']).resolve()
        if not p.is_relative_to(base) or sha256(p)!=art['sha256']:raise ValueError('Changed regression terminal artifact')
        found.add(p)
    if target not in found or Path(bundle['complete']['path']).resolve() not in found:
        raise ValueError('Unbound regression output')
    report=read_locked(done['additional_results']['regression'])
    if report.get('status')!='compared' or report.get('passed') is not True:
        raise ValueError('Regression did not pass; FWE downstream is held')
    files=v['source_manifest']['files']
    checked={k:h for k,h in files.items() if k.startswith(('eval/','efficient_llm/'))}
    actual={str(p.relative_to(source)) for folder in ('eval','efficient_llm') for p in (source/folder).rglob('*.py')}
    if actual!=set(checked) or any(sha256(source/k)!=h for k,h in checked.items()):
        raise ValueError('Candidate scoring/model code differs from regression')
    return bundle,v,report


if __name__=='__main__':main()

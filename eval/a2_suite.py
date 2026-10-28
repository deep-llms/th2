"""Matched A2 downstream job planning and result collection.

No GPU ownership or launch happens on import. The outer supervised coordinator
must validate terminal training handoff and actual-weight numerical gates first.
Each checkpoint scores the entire shared request set in one model load; workers
can run independently across the eight allocated GPUs.
"""
import math
from pathlib import Path

from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS, PRIMARY
from efficient_llm.diagnostic_data import sha256


def scoring_jobs(identities, prepared, output, native_python, gpus):
    """Plan pinned native scoring jobs; caller schedules one job per free slot."""
    if (len(gpus)!=8 or len(set(gpus))!=8 or any(type(g) is not int or g<0 for g in gpus)
            or not identities or not Path(native_python).is_absolute()):
        raise ValueError('Explicit native runtime and eight unique physical GPUs required')
    if any(not name or not name.replace('-','').replace('_','').isalnum() for name in identities):
        raise ValueError('Invalid checkpoint label')
    root=Path(output)
    jobs=[]
    for index,(name,identity) in enumerate(identities.items()):
        for item in (identity,prepared):
            if sha256(item['path'])!=item['sha256']:raise ValueError('Pinned scoring input changed')
        target=root/name/'scores.json'
        if target.exists():raise ValueError('Refuse score overwrite')
        jobs.append(dict(name=name,gpu=gpus[index%8],output=str(target),argv=[native_python,
            '-m','eval.a2_score','--identity',identity['path'],'--identity-sha256',identity['sha256'],
            '--prepared',prepared['path'],'--prepared-sha256',prepared['sha256'],
            '--output',str(target),'--device','cuda:0','--batch-size','8']))
    return jobs


def collect(identities, prepared, scores, replays):
    """Require every model, task, sample, prompt and native-score identity."""
    protocol=prepared['protocol'];tasks=protocol['tasks'];limit=protocol['limit']
    if (set(tasks)!=set(COUNTS) or len(tasks)!=len(COUNTS) or protocol['num_fewshot']!=0
            or protocol['seed']!=42 or set(identities)!=set(scores) or set(scores)!=set(replays)
            or prepared['request_sha256']!=digest(prepared['requests'])
            or prepared['protocol_sha256']!=digest(protocol)):
        raise ValueError('Incomplete fixed nine-task comparison')
    reference_prompts={};summary={};native_runtime=None
    for name,identity in identities.items():
        score=scores[name];result=replays[name];metadata=result['a2_metadata']
        if (score['checkpoint_identity']!=identity or score['checkpoint_identity_sha256']!=digest(identity)
                or score['prepared_sha256']!=digest(prepared)
                or score['request_sha256']!=prepared['request_sha256']
                or score['likelihood_requests']!=len(prepared['requests'])
                or len(score['scores'])!=len(prepared['requests'])
                or score['max_length']!=protocol['max_length']
                or score['precision']!='fp32_parameters_bf16_cuda_autocast'
                or metadata['protocol']!=protocol or metadata['prepared_sha256']!=digest(prepared)
                or metadata['request_sha256']!=prepared['request_sha256']
                or set(result['results'])!=set(tasks)):
            raise ValueError('Mismatched model/request/precision in collected score')
        # Replay receipt must bind the exact serialized scores loaded by the caller.
        # The coordinator checks scored_file_sha256 against the actual file bytes.
        if native_runtime is None:native_runtime=score['packages']
        if score['packages']!=native_runtime:raise ValueError('Different native scoring runtimes')
        for value,greedy in score['scores']:
            if (type(value) not in (float,int) or not math.isfinite(value) or value>0
                    or type(greedy) is not bool):raise ValueError('Invalid token likelihood')
        rows={}
        for task in tasks:
            samples=result['samples'][task];n=COUNTS[task] if limit is None else min(limit,COUNTS[task])
            if (len(samples)!=n or len({r['doc_id'] for r in samples})!=n
                    or result['n-samples'][task]!=dict(original=COUNTS[task],effective=n)
                    or result['n-shot'][task]!=0):raise ValueError('Incomplete benchmark examples')
            prompts=digest(sorted((r['doc_id'],digest({k:r[k] for k in ('doc','arguments','target')})) for r in samples))
            if reference_prompts.setdefault(task,prompts)!=prompts:raise ValueError('Unpaired benchmark prompts')
            value=float(result['results'][task][PRIMARY[task]])
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid task accuracy')
            rows[task]=dict(accuracy=value,primary_metric=PRIMARY[task],examples=n,
                            metrics=result['results'][task],prompt_sha256=prompts)
        rows['multiple_choice_average']=sum(rows[t]['accuracy'] for t in tasks if t!='lambada_openai')/8
        summary[name]=rows
    return dict(status='ok',diagnostic=limit is not None,scores=summary,protocol=protocol,
                native_packages=native_runtime,checkpoint_hashes={k:digest(v) for k,v in identities.items()},
                prepared_sha256=digest(prepared),paired_prompt_hashes=reference_prompts)


def collect_files(identities, prepared_lock, score_locks, replay_locks):
    from scripts.a2_remaining import read_locked
    if set(score_locks)!=set(replay_locks):raise ValueError('Missing replay/score artifact')
    scores={k:read_locked(v) for k,v in score_locks.items()}
    replays={k:read_locked(v) for k,v in replay_locks.items()}
    for name,replay in replays.items():
        if replay['a2_metadata']['scored_file_sha256']!=score_locks[name]['sha256']:
            raise ValueError('Replay used a different native score artifact')
    return collect(identities,read_locked(prepared_lock),scores,replays)


def execute_scoring(jobs, *, timeout_seconds):
    """Outer supervisor owns the allocation; serialize jobs assigned to each GPU."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    import subprocess
    if not math.isfinite(timeout_seconds) or timeout_seconds<=0:raise ValueError('Finite timeout required')
    lanes={}
    for job in jobs:lanes.setdefault(job['gpu'],[]).append(job)
    if len(lanes)>8:raise ValueError('Too many GPU lanes')
    def lane(items):
        for job in items:
            target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('Score output exists')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu']),'CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
            with target.with_suffix('.log').open('x') as stream:
                subprocess.run(job['argv'],env=env,stdout=stream,stderr=subprocess.STDOUT,
                               check=True,timeout=timeout_seconds)
            if not target.is_file():raise ValueError('Native scorer produced no artifact')
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(lane,items) for items in lanes.values()]
        for future in futures:future.result()


def select_initial_screen(verified_rows):
    """Fixed supplementary screen: six parents, six S1 main fits, two NEM long.

    Caller must validate original receipts first. No best-on-benchmark selection;
    S2 and T14 are not substitutes for these prespecified checkpoints.
    """
    result={}
    for dataset in ('dclm','fwe','nem'):
        for parent in ('P300','P1T'):
            result[f'{dataset}-{parent}-original']=dict(kind='parent',dataset=dataset,parent=parent)
            for budget in (('0.2B','0.8B') if dataset=='nem' else ('0.2B',)):
                eligible=[row for row in verified_rows if
                    (row['dataset'],row['parent'],row['teacher'],row['set'],row['budget'])==
                    (dataset,parent,parent,'S1',budget) and row['status']=='complete'
                    and row.get('diverged') is False and row.get('suffix','') in ('','-rerun')]
                if len(eligible)!=1:raise ValueError('Missing or ambiguous fixed downstream checkpoint')
                row=eligible[0]
                result[f'{dataset}-{parent}-S1-{budget}']=dict(kind='healed',dataset=dataset,
                    parent=parent,run_id=row['run_id'])
    return result

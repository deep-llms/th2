"""CPU-only full-request batch decision using the pinned canonical task objects."""
import argparse
import os
from pathlib import Path

from eval.a2_run import write
from eval.a2_stage1b_batch_compare import (decide_batch,ll_differences,
    prediction_flips,lambada_token_prediction_flips)
from eval.a2_suite import collect_files
from eval.downstream_suite import COUNTS,PRIMARY
from scripts.a2_remaining import read_locked


def compare(inputs,choices):
    """Consume SHA-locked original artifacts, never recompute production scores."""
    identity=read_locked(inputs['identity']);prepared=read_locked(inputs['prepared'])
    scores={k:read_locked(inputs['scores'][k]) for k in ('1','2')}
    replays={k:read_locked(inputs['replays'][k]) for k in ('1','2')}
    tokens={k:read_locked(inputs['predictions'][k]) for k in ('1','2')}
    if (identity['dataset'],identity['parent'],identity['kind'])!=('fwe','P300','parent'):
        raise ValueError('Batch decision requires the FWE P300 parent')
    summary=collect_files({k:identity for k in ('1','2')},inputs['prepared'],inputs['scores'],inputs['replays'])
    if summary['diagnostic']:raise ValueError('Deciding comparison requires full benchmark coverage')
    for k in ('1','2'):
        for key in ('prepared_sha256','request_sha256','checkpoint_identity_sha256','batch_size'):
            if tokens[k][key]!=scores[k][key]:raise ValueError('Argmax/score identity mismatch')
        if len(tokens[k]['predictions'])!=len(prepared['requests']):raise ValueError('Incomplete argmax evidence')
        for pred,request,score in zip(tokens[k]['predictions'],prepared['requests'],scores[k]['scores']):
            if len(pred)!=len(request['continuation']) or (pred==request['continuation'])!=score[1]:
                raise ValueError('Captured argmax does not match the scored request')
    result=decide_batch(*({t:summary['scores'][k][t]['accuracy'] for t in COUNTS} for k in ('1','2')))
    flips={}
    for task in COUNTS:
        a,b=(replays[k]['samples'][task] for k in ('1','2'))
        if task=='lambada_openai':
            flips[task]=lambada_token_prediction_flips(a,prepared,tokens['1'],tokens['2'])
            flips[task]['correctness_flips']=prediction_flips(a,b,None,lambada=True)
        else:
            flips[task]=prediction_flips(a,b,choices[task],normalized=PRIMARY[task]=='acc_norm,none')
    likelihood=ll_differences(prepared,scores['1'],scores['2'])
    total=sum(len(x) for x in tokens['1']['predictions'])
    mismatch=sum(a!=b for x,y in zip(tokens['1']['predictions'],tokens['2']['predictions']) for a,b in zip(x,y))
    max_per_token=max(abs(r['nats_per_token']) for r in likelihood['rows'])
    result.update(status='compared',requests=78259,prediction_flips=flips,inputs=inputs,likelihood=likelihood,
        full_compared_tokens=total,full_greedy_token_mismatches=mismatch,
        full_greedy_token_mismatch_rate=mismatch/total,full_max_nats_per_token=max_per_token,
        hard_batch_passed=max_per_token<=.1 and mismatch/total<=.001)
    return result


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','inputs-sha256','dataset-root','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();inputs=read_locked(dict(path=a.inputs,sha256=a.inputs_sha256))
    if Path(a.output).exists():raise ValueError('Output exists')
    from lm_eval.tasks import get_task_dict
    from eval.benchmark_data import local_dataset_paths,task_configs,benchmark_task_manager
    paths=local_dataset_paths(a.dataset_root,a.manifest)
    tasks=get_task_dict(task_configs(list(COUNTS),paths),task_manager=benchmark_task_manager())
    if set(tasks)!=set(COUNTS):raise ValueError('Canonical task selection changed')
    result=compare(inputs,{t:tasks[t].doc_to_choice for t in COUNTS if t!='lambada_openai'})
    rows=result['likelihood'].pop('rows')
    result['likelihood']['per_request']=write(Path(a.output).with_name('per_request_ll_differences.json'),rows)
    write(a.output,result)
    if not result['hard_batch_passed']:
        raise RuntimeError('Full P300 comparison exceeds user hard numerical limits; diagnostic retained')


if __name__=='__main__':main()

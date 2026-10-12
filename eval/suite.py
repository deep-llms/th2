"""Sequential expanded English evaluation; no GPU process management or downloads.

This entry point is suitable as a job in the existing experiment runner. Every
subprocess must succeed and pass artifact validation before the next starts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from eval.benchmarks import resolve_tasks, local_dataset_paths, task_configs
from eval.reporting import benchmark_averages

PROTOCOLS = {
    'zero': (0, ['xcsr','truthfulqa','belebele','lambada','blimp','sciq']),
    'five': (5, ['mmlu','openbookqa','social_iqa','boolq','piqa','arc_easy','winogrande',
                 'xnli','paws-x','xstorycloze']),
    'ten': (10, ['hellaswag']),
    'twentyfive': (25, ['arc_challenge']),
    'math': (5, ['gsm8k']),
    'continuation': (5, ['mmlu_continuation']),
    'paloma': (0, ['paloma']),
}


def selected_protocols(groups=None, new_only=False):
    groups=list(PROTOCOLS) if groups is None else groups
    if not groups or len(set(groups))!=len(groups) or any(g not in PROTOCOLS for g in groups):
        raise ValueError('Select distinct supported protocol groups')
    previous={'xnli','xstorycloze','paws-x','piqa','arc_easy','winogrande','hellaswag','arc_challenge'}
    chosen={g:(shots,[t for t in tasks if not new_only or t not in previous])
            for g,(shots,tasks) in PROTOCOLS.items() if g in groups}
    chosen={g:v for g,v in chosen.items() if v[1]}
    if not chosen:raise ValueError('No new tasks in selected groups')
    return chosen


def commands(checkpoints, root, dataset_root, manifest, backend='fa4', limit=None, protocols=None):
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError('Provide distinct checkpoints')
    jobs=[]
    for i, checkpoint in enumerate(checkpoints):
        for group,(shots,tasks) in (PROTOCOLS if protocols is None else protocols).items():
            dest=Path(root)/f'{i:02d}'/group
            argv=[sys.executable,'-m','eval.eval_checkpoint','--checkpoint',str(checkpoint),
                '--output-dir',str(dest),'--bench-only','--english-only','--tasks',*tasks,
                '--num-fewshot',str(shots),'--seed','42','--batch-size','16','--bf16',
                '--attention-backend',backend,'--dataset-root',str(dataset_root),
                '--dataset-manifest',str(manifest)]
            if limit is not None:argv+=['--limit',str(limit)]
            jobs.append(dict(checkpoint=str(checkpoint),group=group,output=str(dest),argv=argv))
    return jobs


def collect(root, checkpoints, limit=None, protocols=None):
    root=Path(root); result={}; identity={}
    protocols=PROTOCOLS if protocols is None else protocols
    for i,checkpoint in enumerate(checkpoints):
        scores={}; source_hash=None; counts=Counter(); phenomena={}
        for group,(shots,tasks) in protocols.items():
            folder=root/f'{i:02d}'/group
            metadata=json.loads((folder/'eval_metadata.json').read_text())
            if metadata['status']!='completed' or metadata['arguments']['num_fewshot']!=shots:
                raise ValueError('Incomplete/wrong-shot benchmark output')
            if metadata['arguments']['seed']!=42 or metadata['arguments']['limit']!=limit:
                raise ValueError('Evaluation seed/limit differs')
            for key in ('tokenizer_sha256','dataset_manifest_sha256','attention_backend'):
                value=metadata.get(key)
                if value is None:raise ValueError('Missing evaluation identity: '+key)
                identity.setdefault(('metadata',key),value)
                if identity['metadata',key]!=value:raise ValueError('Evaluation identity differs: '+key)
            if Path(metadata['checkpoint']).resolve()!=Path(checkpoint).resolve():
                raise ValueError('Wrong source checkpoint')
            current=metadata.get('checkpoint_sha256')
            if source_hash is None:source_hash=current
            if current!=source_hash:raise ValueError('Source weights changed between task groups')
            values=json.loads((folder/'eval_benchmarks.json').read_text())
            full=json.loads((folder/'eval_benchmarks_full.json').read_text())
            expected=set(resolve_tasks(tasks,True))
            if set(values)!=expected:raise ValueError('Incomplete task group')
            if set(scores)&expected:raise ValueError('Duplicate benchmark result')
            samples=[json.loads(line) for line in (folder/'eval_samples.jsonl').read_text().splitlines()]
            seen=set();documents=set()
            for sample in samples:
                key=(sample['task'],sample['doc_id'])
                record=(*key,sample.get('filter','none'))
                if record in seen:raise ValueError('Duplicate scored example within a filter')
                seen.add(record)
                if key not in documents:counts[sample['task']]+=1
                documents.add(key)
                if sample['task'].startswith('blimp_'):
                    phenomenon=sample['doc'].get('linguistics_term')
                    if phenomenon:
                        phenomena.setdefault(phenomenon,set()).add(sample['task'])
                payload=json.dumps({k:sample[k] for k in ('doc','arguments','target')},sort_keys=True)
                digest=hashlib.sha256(payload.encode()).hexdigest()
                identity.setdefault(key,digest)
                if identity[key]!=digest:raise ValueError('Examples/prompts differ between checkpoints')
            if {k[0] for k in seen}!=expected:raise ValueError('Missing scored task samples')
            for task in expected:
                doc_ids={k[1] for k in documents if k[0]==task}
                observed=len(doc_ids)
                filters={k.rsplit(',',1)[1] for k in values[task] if ',' in k}
                if not filters or {k[2] for k in seen if k[0]==task}!=filters:
                    raise ValueError('Scored filters differ from reported metrics')
                if any({k[1] for k in seen if k[0]==task and k[2]==f}!=doc_ids for f in filters):
                    raise ValueError('Incomplete benchmark filter sample count')
                count=full['n-samples'][task]
                if observed!=count['effective'] or limit is None and count['effective']!=count['original']:
                    raise ValueError('Incomplete benchmark sample count')
                if full['n-shot'][task]!=shots:
                    raise ValueError('Actual task shot count differs from protocol')
            reference=identity.setdefault(('sample_set',group),seen)
            if reference!=seen:raise ValueError('Example sets differ between checkpoints')
            scores.update(values)
        result[str(checkpoint)]=dict(scores=scores,summary=benchmark_averages(scores,counts),checkpoint_sha256=source_hash,
            sample_counts=dict(counts),blimp_phenomena={k:sum(scores[t]['acc,none'] for t in v)/len(v) for k,v in phenomena.items()})
    return dict(status='passed',diagnostic=limit is not None,protocols=protocols,checkpoints=result)


def execute(jobs, workers=1):
    """Independent checkpoint/group jobs on the GPUs already allocated to us."""
    if workers<1:raise ValueError('Positive worker count required')
    if workers==1:
        visible=[os.environ.get('CUDA_VISIBLE_DEVICES','0')]
    else:
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if len(visible)<workers or any(not g or g=='-1' for g in visible) or len(set(visible))!=len(visible):
            raise ValueError('Parallel evaluation needs explicit distinct allocated GPUs')
    pending=queue.Queue();stop=threading.Event()
    for job in jobs:pending.put(job)
    def worker(gpu):
        while not stop.is_set():
            try:job=pending.get_nowait()
            except queue.Empty:return
            log=Path(job['output']).with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
            try:
                with log.open('x') as stream:
                    subprocess.run(job['argv'],check=True,stdout=stream,stderr=subprocess.STDOUT,
                                   env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu})
            except BaseException:
                stop.set();raise
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,gpu) for gpu in visible[:workers]]
        for future in futures:future.result()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--dataset-root',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--attention-backend',choices=['sdpa','fa4'],default='fa4')
    p.add_argument('--limit',type=int);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--groups',nargs='+',choices=list(PROTOCOLS))
    p.add_argument('--new-only',action='store_true',help='Exclude the eight previously completed few-shot tasks')
    p.add_argument('--workers',type=int,default=1,help='Independent jobs on explicitly allocated GPUs')
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('Limit must be positive')
    root=Path(a.output)
    if root.exists():p.error('Output must be fresh')
    protocols=selected_protocols(a.groups,a.new_only)
    mapping=local_dataset_paths(a.dataset_root,a.manifest)
    for _,tasks in protocols.values():task_configs(resolve_tasks(tasks,True),mapping)
    jobs=commands(a.checkpoints,root,a.dataset_root,a.manifest,a.attention_backend,a.limit,protocols)
    root.mkdir(parents=True)
    (root/'plan.json').write_text(json.dumps(jobs,indent=2))
    if a.plan_only:return
    execute(jobs,a.workers)
    (root/'summary.json').write_text(json.dumps(collect(root,a.checkpoints,a.limit,protocols),indent=2,allow_nan=False))


if __name__=='__main__':main()

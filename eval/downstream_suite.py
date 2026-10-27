"""Matched offline zero-shot screen, eight independent single-GPU workers.

Uses the task grouping/worker pattern from deep2shallow eval/suite.py. The
outer existing runner owns GPUs, timeouts, and automatic burn restoration.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

GROUPS = {
    'commonsense': ['hellaswag'],
    'science': ['arc_easy', 'arc_challenge', 'openbookqa', 'sciq'],
    'reading': ['boolq', 'piqa'],
    'language': ['winogrande', 'lambada_openai'],
}
COUNTS = dict(hellaswag=10042, arc_easy=2376, arc_challenge=1172, piqa=1838,
              winogrande=1267, boolq=3270, openbookqa=500, sciq=1000, lambada_openai=5153)
PRIMARY = {t: ('acc_norm,none' if t in ('hellaswag','arc_easy','arc_challenge','piqa','openbookqa')
               else 'acc,none') for t in COUNTS}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def collect(root, checkpoints, limit=None, expected_weights=None):
    scores, identities, metadata = {}, {}, {}
    for arm, checkpoint in checkpoints.items():
        scores[arm] = {}; metadata[arm] = {}
        weight_hash = None
        for group, tasks in GROUPS.items():
            r = json.loads((Path(root)/arm/(group+'.json')).read_text())
            m = r['kv_metadata']; a = m['arguments']
            if (m['status'] != 'ok' or m['global_step'] != 20752 or
                    m['checkpoint'] != str(Path(checkpoint).resolve()) or
                    a['limit'] != limit or a['num_fewshot'] != 0 or a['seed'] != 42 or
                    a['precision'] != 'bf16' or a['attention_backend'] != 'sdpa' or
                    a['max_length'] != 1024 or m['weight_dtype'] != 'torch.bfloat16' or
                    set(r['results']) != set(tasks)):
                raise ValueError('Incomplete or mismatched evaluation protocol: '+arm+'/'+group)
            for key in ('manifest_sha256','tokenizer_sha256','weight_dtype','packages'):
                identity = identities.setdefault(key, m[key])
                if identity != m[key]:raise ValueError('Evaluation runtime/input differs: '+key)
            if weight_hash is None:weight_hash=m['weights_sha256']
            if weight_hash != m['weights_sha256']:raise ValueError('Checkpoint weights changed across workers')
            if expected_weights is not None and weight_hash != expected_weights[arm]:
                raise ValueError('Checkpoint weights differ from preflight')
            metadata[arm][group] = m
            for task in tasks:
                rows = r['samples'][task]; n = COUNTS[task] if limit is None else min(limit,COUNTS[task])
                if (len(rows) != n or len({x['doc_id'] for x in rows}) != n or
                        r['n-samples'][task] != dict(original=COUNTS[task], effective=n) or
                        r['n-shot'][task] != 0):
                    raise ValueError('Incomplete/wrong benchmark population: '+task)
                prompts = sorted((x['doc_id'],digest({k:x[k] for k in ('doc','arguments','target')})) for x in rows)
                reference = identities.setdefault(task, digest(prompts))
                if reference != digest(prompts):raise ValueError('Unpaired examples/prompts: '+task)
                value = float(r['results'][task][PRIMARY[task]])
                if not math.isfinite(value) or not 0 <= value <= 1:raise ValueError('Invalid accuracy')
                scores[arm][task] = dict(primary_metric=PRIMARY[task], accuracy=value,
                    examples=n, metrics=r['results'][task], prompt_sha256=reference)
        # One vote per multiple-choice task. LAMBADA accuracy/PPL stay separate.
        scores[arm]['multiple_choice_average'] = sum(scores[arm][t]['accuracy'] for t in COUNTS if t!='lambada_openai')/8
    return dict(status='ok', diagnostic=limit is not None, protocol='zero-shot, seed 42, context 1024',
        scores=scores, metadata=metadata, paired_prompt_hashes={t:identities[t] for t in COUNTS})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', required=True, help='JSON mapping arm names to checkpoints')
    p.add_argument('--output', required=True);p.add_argument('--dataset-root', required=True)
    p.add_argument('--manifest', required=True);p.add_argument('--limit', type=int)
    p.add_argument('--tokenizer', required=True)
    a=p.parse_args();root=Path(a.output)
    checkpoints=json.loads(Path(a.checkpoints).read_text())
    if len(checkpoints)!=2 or any(not k.replace('-','').isalnum() for k in checkpoints):
        p.error('Expect exactly two checkpoint names suitable as directory names')
    if a.limit is not None and a.limit<1:p.error('Limit must be positive')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=8 or len(set(devices))!=8 or any(not d.isdigit() for d in devices):
        p.error('Require eight explicitly allocated, distinct GPU IDs')
    root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for arm,checkpoint in checkpoints.items():
        for group,tasks in GROUPS.items():
            output=root/arm/(group+'.json')
            argv=[sys.executable,'-m','eval.kv_harness','--checkpoint',checkpoint,
                '--tokenizer',a.tokenizer,
                '--output',str(output),'--dataset-root',a.dataset_root,'--manifest',a.manifest,
                '--precision','bf16','--batch-size','8',
                '--device','cuda:0','--max-length','1024','--num-fewshot','0','--seed','42',
                '--tasks',*tasks]
            if a.limit is not None:argv+=['--limit',str(a.limit)]
            jobs.append(dict(arm=arm,group=group,argv=argv,output=str(output)))
    (root/'plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    def execute(job, gpu):
        log=Path(job['output']).with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
        # Ordinary subprocesses stay inside the outer runner-owned session.
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
        with log.open('x') as stream:
            subprocess.run(job['argv'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(execute,job,gpu) for job,gpu in zip(jobs,devices)]
        for f in futures:f.result()
    identity=json.loads(Path(a.checkpoints).with_name('input-identity.json').read_text())
    if identity['checkpoints'] != checkpoints:raise ValueError('Checkpoint plan changed')
    result=collect(root,checkpoints,a.limit,identity['weights_sha256'])
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DOWNSTREAM_SUITE_OK '+str(root/'summary.json'),flush=True)


if __name__=='__main__':main()

"""Strict adapter for the project's lm-eval 0.4.10 result/sample files.

Old few-shot files can be imported explicitly. They retain their original
provenance; importing does not claim that they were generated prospectively.
"""
import json
import math
from pathlib import Path

import numpy as np

from . import protocol


def load_json(path):
    def reject(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def arguments(sample):
    args = sample['arguments']
    if isinstance(args, dict):
        args = [args[k] for k in sorted(args, key=lambda k: int(k.rsplit('_', 1)[1]))]
        args = [[a[f'arg_{j}'] for j in range(len(a))] for a in args]
    if not isinstance(args, list) or not args:
        raise ValueError('Missing request arguments')
    return args


def close(a, b):
    a, b = float(a), float(b)
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12):
        raise ValueError(f'Logged metric disagrees with recomputation: {a}, {b}')


def parse(sample, config, rule):
    args = arguments(sample)
    for key in ('doc_hash','prompt_hash','target_hash'):
        if not isinstance(sample.get(key), str) or not sample[key]:
            raise ValueError('Missing sample identity: ' + key)
    identity = {k:sample[k] for k in ('doc_hash','prompt_hash','target_hash','doc','target')}
    identity['arguments'] = args
    response = sample['filtered_resps']
    if rule == 'rolling':
        if len(response) != 1 or len(args) != 1 or len(args[0]) != 1:
            raise ValueError('Expected one rolling request')
        ll = float(response[0]); text = sample['target']
        nb = len(text.encode('utf-8'))
        if not math.isfinite(ll) or not nb or args[0][0] != text:
            raise ValueError('Invalid rolling likelihood/text/byte denominator')
        close(sample['bits_per_byte'][0], ll)
        close(sample['bits_per_byte'][1], nb)
        return dict(identity=identity,ll=ll,bytes=nb,metrics={})
    if len(args) != len(response) or any(len(a)!=2 or not all(isinstance(x,str) for x in a) for a in args):
        raise ValueError('Likelihood requests/responses differ')
    if any(len(r)!=2 or type(r[1]) is not bool for r in response):
        raise ValueError('Malformed likelihood or greedy-match response')
    ll = np.asarray([r[0] for r in response],dtype=np.float64)
    if not np.isfinite(ll).all(): raise ValueError('Non-finite log likelihood')
    if rule == 'll':
        if len(ll)!=1: raise ValueError('Expected a single continuation')
        n = len(args[0][1].strip())
        if not n: raise ValueError('Empty answer')
        acc = float(response[0][1]); close(sample['acc'],acc)
        return dict(identity=identity,ll=float(ll[0]),chance=0.,metrics=dict(M1=float(ll[0]/n),ACC=acc))
    delim = config.get('target_delimiter')
    delim = ' ' if delim is None else delim
    if any(not a[1].startswith(delim) for a in args): raise ValueError('Missing target delimiter')
    n = np.asarray([len(a[1])-len(delim) for a in args],dtype=np.float64)
    if len(ll)<2 or np.any(n<=0): raise ValueError('Empty choice or fewer than two choices')
    multi = config['task'] == 'winogrande'
    if multi:
        if rule!='raw' or len({a[1] for a in args})!=1 or args[0][1]!=delim+sample['target']:
            raise ValueError('Invalid shared-continuation task')
        gold = {'1':0,'2':1}[sample['doc']['answer']]
    else:
        target = sample['target']
        gold = int(target)
        if str(gold)!=str(target): raise ValueError('Non-integer gold index')
    if not 0<=gold<len(ll): raise ValueError('Invalid gold index')
    identity['gold'] = gold
    raw = float(np.argmax(ll)==gold); norm = float(np.argmax(ll/n)==gold)
    close(sample['acc'], raw)
    if 'acc_norm' in sample: close(sample['acc_norm'],norm)
    if rule=='char' and 'acc_norm' not in sample: raise ValueError('Missing required acc_norm')
    scores = ll/n if rule=='char' else ll
    prob = np.exp(scores-scores.max()); prob /= prob.sum()
    metrics = dict(M1=float(ll[gold]/n[gold]),M2=float(prob[gold]),
                   M3=float(scores[gold]-np.delete(scores,gold).max()),ACC=norm if rule=='char' else raw)
    if not all(math.isfinite(x) for x in metrics.values()): raise ValueError('Non-finite metric')
    return dict(identity=identity,ll=ll.tolist(),chance=1/len(ll),raw_acc=raw,norm_acc=norm,metrics=metrics)


def story_clusters(texts):
    """Connected components of exact string-prefix related stories.

    A trie traversal is unnecessary: lexicographic order makes each prefix's
    descendants contiguous. Equality also connects repeated questions.
    """
    if any(not isinstance(t,str) or not t for t in texts): raise ValueError('Empty bAbI story')
    ids = [None]*len(texts); root = None
    for i in sorted(range(len(texts)),key=lambda j:texts[j]):
        if root is None or not texts[i].startswith(root): root = texts[i]
        ids[i] = protocol.digest(root)
    return ids


def validate(results, samples, task, shots, selection=None):
    expected = set(protocol.inventory(task))
    if set(samples)!=expected: raise ValueError('Incomplete or unexpected subtask samples')
    for field in ('results','configs','n-samples','n-shot'):
        if not expected.issubset(results.get(field,{})): raise ValueError('Missing expected subtasks in '+field)
    if selection is not None and set(selection)!=expected: raise ValueError('Incomplete frozen selection')
    rule = 'rolling' if task.startswith('paloma_') else protocol.RULES[task]
    records = {}; sizes = {}
    for sub in sorted(expected):
        rows = samples[sub]; count = results['n-samples'][sub]
        original,effective = count['original'],count['effective']
        if type(original) is not int or type(effective) is not int or original<1:
            raise ValueError('Invalid n-samples counts')
        if effective!=len(rows) or (selection is None and original!=effective):
            raise ValueError('Incomplete sample counts')
        if selection is not None and (set(selection[sub])!={s['doc_id'] for s in rows} or len(selection[sub])!=len(rows)):
            raise ValueError('Samples differ from frozen selection')
        if results['n-shot'][sub]!=shots: raise ValueError('Wrong few-shot protocol')
        config = results['configs'][sub]
        if config.get('task')!=sub:raise ValueError('Mismatched subtask config')
        parsed = []
        for row in rows:
            if row.get('filter')!='none': raise ValueError('Unsupported sample filter')
            if type(row['doc_id']) is not int or not 0<=row['doc_id']<original:
                raise ValueError('Invalid document ID')
            key = (sub,row['doc_id'])
            if key in records: raise ValueError('Duplicate example')
            records[key] = parse(row,config,rule); parsed.append(records[key])
        summary = results['results'][sub]
        if rule=='rolling':
            close(-sum(r['ll'] for r in parsed)/sum(r['bytes'] for r in parsed)/math.log(2),summary['bits_per_byte,none'])
        else:
            close(np.mean([r['metrics']['ACC'] if rule=='ll' else r['raw_acc'] for r in parsed]),summary['acc,none'])
            if rule=='ll': close(math.exp(-np.mean([r['ll'] for r in parsed])),summary['perplexity,none'])
            elif 'acc_norm,none' in summary: close(np.mean([r['norm_acc'] for r in parsed]),summary['acc_norm,none'])
        sizes[sub] = original
    ids = sorted(records); ordered = [records[i] for i in ids]
    metrics = {k:np.asarray([r['metrics'][k] for r in ordered]) for k in ordered[0]['metrics']}
    clusters = story_clusters([r['identity']['doc']['input'] for r in ordered]) if task in protocol.PROBES else None
    return dict(task=task,ids=ids,records=ordered,metrics=metrics,clusters=clusters,sizes=sizes,
                chance=float(np.mean([r.get('chance',0.) for r in ordered])),rule=rule)


def read(directory, task, shots, frozen, legacy=False, selection=None):
    """Read a dedicated task directory or a historical combined shot group."""
    directory = Path(directory)
    meta = load_json(directory/'eval_metadata.json')
    if meta.get('status')!='completed' or meta.get('packages',{}).get('lm_eval')!='0.4.10':
        raise ValueError('Require completed lm_eval 0.4.10 artifact')
    for key in ('checkpoint_sha256','tokenizer_sha256','dataset_manifest_sha256'):
        if not meta.get(key): raise ValueError('Missing provenance: '+key)
    if not legacy and meta.get('protocol_sha256')!=frozen['protocol_sha256']:
        raise ValueError('Result protocol hash differs')
    if not legacy and meta.get('selection')!=selection:
        raise ValueError('Result selection differs')
    if not legacy:
        if (meta.get('task')!=task or meta.get('task_config_sha256')!=protocol.digest(protocol.configs(task)) or meta.get('seed')!=42
            or meta.get('num_fewshot')!=shots or meta.get('softmax_dtype')!='float32'
            or meta.get('prefix_token_id') is None or meta.get('batch_size',0)<1):
            raise ValueError('Paired evaluation configuration differs')
    elif meta.get('arguments',{}).get('seed')!=42 or meta.get('arguments',{}).get('num_fewshot')!=shots:
        raise ValueError('Historical evaluation seeds/shots differ')
    results = load_json(directory/'eval_benchmarks_full.json')
    samples = {name:[] for name in protocol.inventory(task)}
    with (directory/'eval_samples.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line,parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
            if row['task'] in samples: samples[row['task']].append(row)
            elif not legacy: raise ValueError('Unexpected subtask samples')
    result = validate(results,samples,task,shots,selection)
    result['metadata'] = meta
    result['config_hashes'] = {name:protocol.digest(results['configs'][name]) for name in samples}
    return result


def align(a,b):
    if a['ids']!=b['ids'] or a['sizes']!=b['sizes'] or a['rule']!=b['rule']:
        raise ValueError('Different example sets or scoring rules')
    if any(x['identity']!=y['identity'] for x,y in zip(a['records'],b['records'])):
        raise ValueError('Paired documents, prompts, targets, gold indices or full arguments differ')
    if a['clusters']!=b['clusters']: raise ValueError('Different story clusters')
    if 'metadata' in a and 'metadata' in b:
        for key in ('tokenizer_sha256','dataset_manifest_sha256'):
            if a['metadata'][key]!=b['metadata'][key]: raise ValueError('Paired provenance differs: '+key)
        if a['metadata'].get('packages')!=b['metadata'].get('packages'):
            raise ValueError('Paired library versions differ')
        if a['metadata'].get('prefix_token_id')!=b['metadata'].get('prefix_token_id'):
            raise ValueError('Different empty-context prefix tokens')
    if a.get('config_hashes')!=b.get('config_hashes'): raise ValueError('Paired task configs differ')
    with np.errstate(over='ignore',invalid='ignore'):
        differences={m:b['metrics'][m]-a['metrics'][m] for m in a['metrics']}
    if any(not np.isfinite(d).all() for d in differences.values()):
        raise ValueError('Non-finite paired metric difference')
    return differences

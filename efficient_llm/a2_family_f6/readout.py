"""F6 array replay, overlap and paired statistics without model execution."""
from pathlib import Path
import json
import numpy as np
from scripts.a2_family_report import pair_scores, arrays
from scripts.a2_family_v14_gate import sha
from .protocol import reading, joint_reading
from .common import read, write


def normalized(score):
    # Native A2 OpenSci evaluates fixed-length packed rows, 4095 targets each.
    n = len(score['per_sequence_nll'])
    if score['sequences'] != n or score['tokens'] != n*4095:
        raise ValueError('Not the registered OpenSci final-set estimator')
    value = dict(score, per_sequence_tokens=[4095]*n)
    arrays(value)
    return value


def adapted(result):
    return dict(baseline={'final': normalized(result['baseline'])},
                pre={'final': normalized(result['pre']['final'])},
                scores={'final': normalized(result['scores']['final'])})


def exposure(result, audit):
    records = result['raw_order_files']
    if len(records) != 8 or len({r['path'] for r in records}) != 8:
        raise ValueError('Eight distinct rank traces required')
    seen = set(); count = 0
    for item in records:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Actual sample trace changed')
        with Path(item['path']).open() as f:
            for line in f:
                ids = json.loads(line)['sample_ids']; seen.update(ids); count += len(ids)
    expected = result['registered_run']['steps']*64
    if seen != set(range(expected)) or count != expected:
        raise ValueError('Actual prefix not consumed once in full')
    hits = []
    for doc in audit['prefix_coverage']['2030']['hits']:
        start, stop = doc['token_start'], doc['token_stop']
        consumed = sum(max(0, min(stop, (row+1)*4096)-max(start, row*4096))
                       for row in range(start//4096, (stop-1)//4096+1) if row in seen)
        hits.append(dict(doc, actual_consumed_tokens=consumed))
    return dict(status='seen' if any(d['actual_consumed_tokens'] for d in hits) else 'not seen',
                actual_rows=count, documents=hits)


def analyze(results, audit, historical, final=False):
    groups = {}; rows = []; bad = []; pairs = []
    for result in results:
        item = result['registered_run']; ds = item['dataset']
        try:
            value = adapted(result)
            par, c = arrays(value['baseline']['final'])
            pre, _ = arrays(value['pre']['final']); post, _ = arrays(value['scores']['final'])
            drop = audit['exclude_final_sequence_ids'] if ds == 'fwe' else []
            keep = np.ones(len(c), dtype=bool); keep[drop] = False
            avg = lambda x: float(x@c/c.sum())
            reduced = lambda x: float(x[keep]@c[keep]/c[keep].sum())
            exp = exposure(result, audit) if ds == 'fwe' else dict(status='not applicable; FWE overlap only')
            rows.append(dict(**item, parent=avg(par), NLL=avg(post), delta_pre=avg(pre-par),
                             delta_post=avg(post-par), reduced_NLL=reduced(post),
                             reduced_delta_pre=reduced(pre-par), reduced_delta_post=reduced(post-par),
                             dropped_sequences=len(drop), dropped_prediction_tokens=int(c[~keep].sum()),
                             exposure=exp, flags=result.get('flags', [])))
            groups.setdefault((ds, item['budget']), {})[item['arm']] = value
        except Exception as error:
            bad.append(dict(run_id=item['run_id'], error=repr(error)))
    for (ds, budget), arms in sorted(groups.items()):
        if set(arms) != {'S72', 'S188'}:
            continue
        drop = audit['exclude_final_sequence_ids'] if ds == 'fwe' else []
        try:
            full = pair_scores(arms['S72'], arms['S188'], repeats=10000 if final else 0)
            reduced = pair_scores(arms['S72'], arms['S188'], drop, repeats=10000 if final else 0)
        except Exception as error:
            bad.append(dict(dataset=ds,budget=budget,error=repr(error)))
            continue
        for v in (full, reduced):
            if 'bootstrap' in v:
                v['bootstrap']['unit'] = 'paired fixed-length packed sequences (native A2 S estimator)'
        pairs.append(dict(dataset=ds, budget=budget, full=full, reduced=reduced,
                          difference=reduced['E']-full['E']))
    fwe = [p for p in pairs if p['dataset']=='fwe']
    immaterial = all(abs(p['difference'])<.001 for p in fwe) if fwe else None
    primary = 'full' if immaterial else 'reduced'
    e02 = {ds: historical[ds]['E_full'] for ds in ('dclm','fwe')}
    # Historical full-set values are the preregistered .2B baseline. Retain the
    # matching historical reduced arrays when contamination changes the basis.
    if primary == 'reduced':
        e02['fwe'] = historical['fwe']['E_reduced']
    values = {(p['dataset'],p['budget']):p['reduced' if p['dataset']=='fwe' and primary=='reduced' else 'full']['E'] for p in pairs}
    fe6 = {ds: reading(e02[ds],values[ds,'0.8B']) for ds in ('dclm','fwe') if (ds,'0.8B') in values}
    extended = {ds: reading(values[ds,'0.8B'],values[ds,'3.2B']) for ds in ('dclm','fwe') if (ds,'0.8B') in values and (ds,'3.2B') in values}
    for pair in pairs:
        for basis in ('full','reduced'):
            short = next(r for r in rows if (r['dataset'],r['arm'],r['budget'])==(pair['dataset'],'S72',pair['budget']))
            delta = short['delta_post' if basis=='full' else 'reduced_delta_post']
            pair[basis]['E_over_short_delta'] = pair[basis]['E']/delta if abs(delta)>1e-15 else None
    return dict(rows=rows,pairs=pairs,missing_or_invalid=bad,FE6=fe6,FE6_extended=extended,
                E02=e02,overlap=dict(immaterial=immaterial,primary_FWE=primary),
                interval_caveat='OpenSci native final set is packed, not one document per row; paired sequence CI preserves A2 S and does not claim document-level independence.')

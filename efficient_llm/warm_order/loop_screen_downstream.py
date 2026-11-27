"""Zero-shot downstream scoring of loop-screen checkpoints on the pinned lm-eval requests.

Self-contained on purpose (the frozen node source may lack the older harness
bridge). prepare_rows/score semantics copy efficient_llm/warm_order/downstream.py:
common-context left truncation to 512 inputs, FP32 with autocast disabled,
continuation log-likelihood summed over continuation tokens, greedy flag for
LAMBADA. Item metrics follow downstream_report.compact: HellaSwag/PIQA acc_norm
(length in characters), ARC-Easy/LAMBADA acc. tests/test_loop_screen_downstream.py
checks equivalence with the original scorer on fixtures.
"""
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

TASKS = ('hellaswag', 'piqa', 'arc_easy', 'lambada_openai')
COUNTS = dict(hellaswag=10042, piqa=1838, arc_easy=2376, lambada_openai=5153)
PRIMARY = dict(hellaswag='acc_norm', piqa='acc_norm', arc_easy='acc', lambada_openai='acc')


def digest(value):
    # Identical to eval.a2_request_bridge.digest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prepare_rows(requests, length=512):
    longest = {}
    for r in requests:
        ctx = tuple(r['context'] or [0])
        target = r['continuation']
        if not target or len(target) > length:
            raise ValueError('Invalid continuation length')
        longest[ctx] = max(longest.get(ctx, 0), len(target))
    rows = []
    for i, r in enumerate(requests):
        ctx = tuple(r['context'] or [0])
        target = list(r['continuation'])
        kept = list(ctx[-(length + 1 - longest[ctx]):])
        rows.append(dict(index=i, context=kept, tokens=(kept + target)[:-1], target=target,
                         truncated=len(kept) < len(ctx)))
    return rows


@torch.no_grad()
def score_requests(model, path, requests, device, batch_size=16):
    """Return [[loglik, greedy_exact], ...] in request order. Caller sets runtime scopes."""
    model.eval()
    rows = sorted(prepare_rows(requests, model.c.length), key=lambda r: -len(r['tokens']))
    unembed = model.output_weight() if hasattr(model, 'output_weight') else model.embed.weight  # f19 Ouro: untied head
    scores = [None] * len(rows)
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        ids = torch.zeros((len(part), len(part[0]['tokens'])), device=device, dtype=torch.long)
        for i, r in enumerate(part):
            ids[i, :len(r['tokens'])] = torch.tensor(r['tokens'], device=device)
        with torch.autocast(torch.device(device).type, enabled=False):
            h = model.readout_hidden(model.hidden(ids, path))
        for i, r in enumerate(part):
            a = len(r['context']) - 1
            b = a + len(r['target'])
            with torch.autocast(torch.device(device).type, enabled=False):
                logits = F.linear(h[i, a:b], unembed).float()
            target = torch.tensor(r['target'], device=device)
            value = logits.log_softmax(-1).gather(1, target[:, None]).sum()
            if not torch.isfinite(value):
                raise ValueError('Nonfinite score')
            scores[r['index']] = [float(value), bool(torch.equal(logits.argmax(-1), target))]
    meta = dict(truncated_requests=sum(r['truncated'] for r in rows),
                short_prefix_requests=sum(len(r['context']) < model.c.prefix for r in rows))
    return scores, meta


def choices_of(task, doc):
    if task == 'hellaswag':
        return doc['choices']
    if task == 'piqa':
        return [doc['sol1'], doc['sol2']]
    return doc['choices']['text']


def items(raw, prepared, scores, expected_counts=COUNTS):
    """Per-item correctness from OUR scores, using the harness sample structure (docs, args, gold)."""
    if raw['a2_metadata']['prepared_sha256'] != digest(prepared):
        raise ValueError('Request identity')
    # Duplicate request texts can score slightly differently for token-routed (DOT) models: batch-shape float
    # noise can flip a near-tied argmax route. Use the first occurrence in request order (deterministic) and
    # count/measure the disagreement instead of failing (dot_sandwich a01 downstream failure).
    lookup, spread = {}, {}
    for r, s in zip(prepared['requests'], scores, strict=True):
        key, value = tuple(r['text']), (float(s[0]), bool(s[1]))
        if key in lookup:
            spread[key] = max(spread.get(key, 0.), abs(value[0] - lookup[key][0]), float(value[1] != lookup[key][1]))
        else:
            lookup[key] = value
    result = {}
    for task in TASKS:
        ll_rows, gold_rows, chars_rows, correct, ids = [], [], [], [], []
        ambiguous, worst = 0, 0.
        for r in sorted(raw['samples'][task], key=lambda r: r['doc_id']):
            values = []
            for arg in r['arguments']:
                values.append(lookup[tuple(arg)])
                if spread.get(tuple(arg), 0.) > 0:
                    ambiguous += 1
                    worst = max(worst, spread[tuple(arg)])
            ll = np.array([v for v, _ in values], dtype=np.float64)
            if not np.isfinite(ll).all() or (ll > 0).any():
                raise ValueError('Invalid log likelihood')
            if task == 'lambada_openai':
                if len(values) != 1:
                    raise ValueError('LAMBADA request count')
                ok, gold, chars = int(values[0][1]), 0, [1]
            else:
                chars = [len(c) for c in choices_of(task, r['doc'])]
                gold = int(r['target'])
                if min(chars) <= 0 or len(chars) != len(ll) or not 0 <= gold < len(ll):
                    raise ValueError('Choice shape')
                vals = ll / np.array(chars) if PRIMARY[task] == 'acc_norm' else ll
                ok = int(vals.argmax() == gold)
            ids.append(r['doc_id'])
            ll_rows.append(ll)
            gold_rows.append(gold)
            chars_rows.append(chars)
            correct.append(ok)
        if len(ids) != expected_counts[task] or len(set(ids)) != len(ids):
            raise ValueError('Incomplete item population: '+task)
        width = max(len(x) for x in ll_rows)
        ll_mat = np.zeros((len(ids), width))
        chars_mat = np.zeros((len(ids), width), dtype=np.int32)
        for i, (x, c) in enumerate(zip(ll_rows, chars_rows)):
            ll_mat[i, :len(x)] = x
            chars_mat[i, :len(c)] = c
        result[task] = dict(metric=PRIMARY[task], accuracy=float(np.mean(correct)), ambiguous=ambiguous, ambiguous_max=worst,
                            id=np.array(ids),
                            correct=np.array(correct, dtype=np.int8), gold=np.array(gold_rows, dtype=np.int16),
                            ll=ll_mat, chars=chars_mat, count=np.array([len(x) for x in ll_rows], dtype=np.int16))
    return result


def save_items(result, path):
    arrays = {f'{t}_{k}': v for t, x in result.items() for k, v in x.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)
    return {t: dict(metric=x['metric'], accuracy=x['accuracy'], items=len(x['id']),
                    ambiguous_duplicates=x.get('ambiguous', 0), ambiguous_max=x.get('ambiguous_max', 0.))
            for t, x in result.items()}


def paired_accuracy(a, b, draws=2000, seed=901):
    """Paired item bootstrap of accuracy difference a-b (same convention as downstream_report.paired)."""
    delta = a.astype(int) - b.astype(int)
    count = np.array([(delta == k).sum() for k in (-1, 0, 1)])
    rng = np.random.default_rng(seed)
    d = rng.multinomial(len(delta), count / len(delta), size=draws)
    values = (d[:, 2] - d[:, 0]) / len(delta)
    return dict(delta=float(delta.mean()), lo=float(np.quantile(values, .025)), hi=float(np.quantile(values, .975)),
                wins=int(count[2]), losses=int(count[0]), items=len(delta))

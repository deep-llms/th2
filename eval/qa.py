"""SQuAD v2 transfer for causal custom backbones, with development-only calibration.

Question precedes context. EOS after each context window is the no-answer
position, so its causal hidden state sees the whole window. No bidirectional
attention or proxy cache is introduced.
"""
from collections import Counter
import hashlib
import re
import string

import numpy as np
import torch
from torch.nn import functional as F
from datasets import Dataset, DatasetDict, load_dataset

from eval.finetune import PairClassifier, PairCollator
from deep_kv.model import Context
from eval.benchmarks import local_dataset_paths
from eval.models import file_hash


class SpanClassifier(PairClassifier):
    def __init__(self, backbone, seed):
        super().__init__(backbone, 2, seed)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        valid = attention_mask.bool()
        lengths = valid.sum(-1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        if not torch.equal(valid, positions < lengths[:, None]) or not bool((lengths >= 2).all()):
            raise ValueError('Span inputs require contiguous right padding')
        context = Context(input_ids, torch.ones_like(valid),
                          torch.where(valid, positions, positions-lengths[:, None]), (~valid).long())
        logits = self.score(self.wrapped.hidden_states(context)[0]).float()
        logits = logits.masked_fill(~valid[:, :, None], torch.finfo(logits.dtype).min)
        result = dict(logits=logits)
        if start_positions is not None:
            for labels in (start_positions, end_positions):
                if labels is None or not bool(((labels >= 0) & (labels < lengths)).all()):
                    raise ValueError('Span targets outside valid input')
            result['loss'] = (F.cross_entropy(logits[:, :, 0], start_positions) +
                              F.cross_entropy(logits[:, :, 1], end_positions)) / 2
        return result


class SpanCollator(PairCollator):
    def __call__(self, rows):
        batch = super().__call__([dict(input_ids=r['input_ids'], labels=0) for r in rows])
        batch.pop('labels')
        for key in ('start_positions', 'end_positions'):
            batch[key] = torch.tensor([r[key] for r in rows], dtype=torch.long)
        return batch


def grouped_splits(raw):
    def key(row): return hashlib.sha256(' '.join(row['context'].split()).encode()).hexdigest()
    heldout = {key(r) for r in raw['validation']}
    keys = [key(r) for r in raw['train']]
    groups = sorted(set(keys)-heldout, key=lambda k: hashlib.sha256(('split-42:'+k).encode()).hexdigest())
    if len(groups) < 10: raise ValueError('Too few SQuAD context groups for development split')
    dev = set(groups[:max(1, len(groups)//10)])
    return DatasetDict(train=raw['train'].select([i for i,k in enumerate(keys) if k not in heldout|dev]),
        validation=raw['train'].select([i for i,k in enumerate(keys) if k in dev]), test=raw['validation'])


def features(examples, tokenizer, max_length, stride=128):
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError('SQuAD requires a fast tokenizer with EOS')
    if max_length < 16: raise ValueError('SQuAD context too short')
    rows, metadata = [], []
    for example in examples:
        # Bound the question first, leaving useful capacity for sliding windows.
        question_ids = tokenizer.encode('Question: '+example['question'].strip(), add_special_tokens=False)
        question = tokenizer.decode(question_ids[:min(64, max_length//4)], skip_special_tokens=False)+'\nContext: '
        prefix_size = len(tokenizer.encode(question, add_special_tokens=False))
        capacity = max_length - prefix_size - 1
        if capacity < 4: raise ValueError('Question leaves insufficient context capacity')
        tokenized = tokenizer(question, example['context'], add_special_tokens=False,
            truncation='only_second', max_length=max_length-1, stride=min(stride, capacity//2),
            return_overflowing_tokens=True, return_offsets_mapping=True)
        answers = example['answers']
        for start, text in zip(answers['answer_start'], answers['text']):
            if example['context'][start:start+len(text)] != text:
                raise ValueError('SQuAD answer offsets do not match context')
        for w, ids in enumerate(tokenized['input_ids']):
            offsets = [list(o) if seq == 1 and o[1] > o[0] else None
                       for o,seq in zip(tokenized['offset_mapping'][w], tokenized.sequence_ids(w))]
            null = len(ids)
            start_pos = end_pos = null
            context_tokens = [i for i,o in enumerate(offsets) if o is not None]
            if answers['text'] and context_tokens:
                a = answers['answer_start'][0]; b = a + len(answers['text'][0])
                if offsets[context_tokens[0]][0] <= a and offsets[context_tokens[-1]][1] >= b:
                    starts = [i for i in context_tokens if offsets[i][0] <= a < offsets[i][1]]
                    ends = [i for i in context_tokens if offsets[i][0] < b <= offsets[i][1]]
                    if starts and ends: start_pos, end_pos = starts[-1], ends[0]
            rows.append(dict(input_ids=ids+[tokenizer.eos_token_id], start_positions=start_pos, end_positions=end_pos))
            metadata.append(dict(id=example['id'], offsets=offsets+[None], null=null))
    return Dataset.from_list(rows), metadata


def encode_windows(batch, tokenizer, max_length):
    examples = [dict(zip(batch, values)) for values in zip(*batch.values())]
    encoded, metadata = features(examples, tokenizer, max_length)
    result = encoded.to_dict()
    result.update(example_id=[m['id'] for m in metadata], offsets=[m['offsets'] for m in metadata],
                  null=[m['null'] for m in metadata])
    return result


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_scores(prediction, gold):
    gold = [g for g in gold if normalize(g)] or ['']
    pred = normalize(prediction)
    exact, f1 = [], []
    for text in gold:
        target = normalize(text)
        exact.append(float(pred == target))
        p, t = pred.split(), target.split()
        if not p or not t:
            f1.append(float(p == t)); continue
        common = sum((Counter(p) & Counter(t)).values())
        f1.append(2*common/(len(p)+len(t)))
    return max(exact), max(f1)


def decode_spans(logits, metadata, examples, max_answer_length=30):
    if len(logits) != len(metadata): raise ValueError('Span feature count mismatch')
    examples = {r['id']: r for r in examples}
    best = {key: (-float('inf'), '') for key in examples}
    nulls = {key: float('inf') for key in examples}
    for scores, feature in zip(logits, metadata):
        n = len(feature['offsets'])
        if scores.shape[0] < n or not np.isfinite(scores[:n]).all():
            raise ValueError('Invalid span logits')
        key = feature['id']; null = feature['null']
        nulls[key] = min(nulls[key], float(scores[null].sum()))
        offsets = feature['offsets']
        # Exhaustive bounded-span search: no top-k approximation can omit the winner.
        for start, offset in enumerate(offsets):
            if offset is None: continue
            for end in range(start, min(n, start+max_answer_length)):
                if offsets[end] is None: break
                score = float(scores[start, 0] + scores[end, 1])
                if score > best[key][0]:
                    best[key] = (score, examples[key]['context'][offset[0]:offsets[end][1]])
    if any(not np.isfinite(best[k][0]) or not np.isfinite(nulls[k]) for k in examples):
        raise ValueError('Example lacks a valid context span')
    return {k: dict(text=best[k][1], difference=nulls[k]-best[k][0]) for k in examples}


def score_answers(decoded, examples, threshold):
    values = [answer_scores('' if decoded[r['id']]['difference'] > threshold else decoded[r['id']]['text'],
                            r['answers']['text']) for r in examples]
    if not values: raise ValueError('Empty SQuAD evaluation')
    return dict(exact_match=float(np.mean([v[0] for v in values])), f1=float(np.mean([v[1] for v in values])))


def select_threshold(decoded, examples):
    # Sweep tied differences together. O(N log N), avoiding quadratic rescoring.
    records = []
    total = 0.
    for row in examples:
        value = decoded[row['id']]
        empty = answer_scores('', row['answers']['text'])[1]
        nonempty = answer_scores(value['text'], row['answers']['text'])[1]
        total += empty
        records.append((value['difference'], nonempty-empty))
    records.sort()
    threshold = float(np.nextafter(records[0][0], -np.inf)); best = total
    i = 0
    while i < len(records):
        d = records[i][0]
        while i < len(records) and records[i][0] == d:
            total += records[i][1]; i += 1
        if total > best:
            best, threshold = total, float(d)
    return threshold


class QAMetrics:
    def __init__(self, metadata, examples, threshold=None):
        self.metadata, self.examples = metadata, list(examples)
        self.fixed_threshold = threshold
        self.threshold = threshold
        self.decoded = None

    def __call__(self, prediction):
        self.decoded = decode_spans(np.asarray(prediction.predictions), self.metadata, self.examples)
        self.threshold = (select_threshold(self.decoded, self.examples) if self.fixed_threshold is None else self.fixed_threshold)
        return score_answers(self.decoded, self.examples, self.threshold)


def prepare_data(task, tokenizer, args):
    mapping = local_dataset_paths(task.dataset_root, task.dataset_manifest)['squad_v2']
    with args.main_process_first(desc='SQuAD split and sliding windows'):
        raw = load_dataset(mapping['dataset_path'], **mapping['dataset_kwargs'])
        split = grouped_splits(raw)
        encoded, metadata, examples = {}, {}, {}
        for name in (('test',) if task.evaluate_run else ('train', 'validation')):
            source = 'validation' if task.smoke and task.evaluate_run else name
            part = split[source]
            if task.smoke: part = part.select(range(min(len(part), 64 if name == 'train' else 16)))
            if len(set(part['id'])) != len(part): raise ValueError('Duplicate SQuAD IDs')
            windows = part.map(encode_windows, batched=True, batch_size=128,
                num_proc=task.preprocessing_num_workers, remove_columns=part.column_names,
                fn_kwargs=dict(tokenizer=tokenizer, max_length=task.max_length),
                desc='SQuAD windows/'+name)
            metadata[name] = [dict(id=r['example_id'], offsets=r['offsets'], null=r['null'])
                              for r in windows.select_columns(['example_id', 'offsets', 'null'])]
            encoded[name] = windows.remove_columns(['example_id','offsets','null'])
            examples[name] = part
    def digest(part):
        import json
        return hashlib.sha256('\n'.join(json.dumps(r, sort_keys=True) for r in part).encode()).hexdigest()
    info = dict(task='squad_v2', rows={k:len(v) for k,v in encoded.items()},
        example_rows={k:len(v) for k,v in examples.items()},
        train_order_sha256=digest(split['train']), split_order_sha256={k:digest(v) for k,v in examples.items()},
        fingerprints={k:v._fingerprint for k,v in encoded.items()},
        dataset_manifest_sha256=file_hash(task.dataset_manifest),
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        split_policy='context-grouped train-derived 10% development; public validation final holdout',
        null_position='EOS after context window', max_answer_length=30)
    encoded['_qa'] = (metadata, examples)
    return encoded, info

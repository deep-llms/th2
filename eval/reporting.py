"""Explicit benchmark-level aggregation; never count each subtask as a benchmark."""
import math
import statistics


def benchmark_averages(results, sample_counts=None):
    grouped = {}
    paloma = {}
    for task, metrics in results.items():
        if task.startswith('paloma_'):
            values = {}
            for name in ('word_perplexity', 'byte_perplexity', 'bits_per_byte'):
                value = float(metrics[name+',none'])
                if not math.isfinite(value) or value < (0 if name == 'bits_per_byte' else 1):
                    raise ValueError('Invalid Paloma metric: '+task+'/'+name)
                values[name] = value
            paloma[task] = values
            continue
        if task.startswith('mmlu_'):
            group = 'mmlu_continuation' if task.endswith('_continuation') else 'mmlu'
            metric = 'acc,none'
        elif task.startswith('blimp_'):
            group, metric = 'blimp', 'acc,none'
        elif task.startswith('truthfulqa_mc'):
            group, metric = 'truthfulqa', 'acc,none'
        elif task in ('xcsqa_en', 'xcodah_en'):
            group, metric = 'xcsr', 'acc,none'
        elif task == 'gsm8k':
            group, metric = task, 'exact_match,strict-match'
        else:
            group = task
            metric = 'acc_norm,none' if task in ('arc_easy','arc_challenge','hellaswag','piqa','openbookqa') else 'acc,none'
        if metric not in metrics: raise ValueError(f'Missing primary metric {task}/{metric}')
        value = float(metrics[metric])
        if not math.isfinite(value): raise ValueError('Nonfinite benchmark score')
        grouped.setdefault(group, []).append((task,value))
    for name, count in [('mmlu',57),('mmlu_continuation',57),('blimp',67),('truthfulqa',2),('xcsr',2)]:
        if name in grouped and len(grouped[name]) != count:
            raise ValueError(f'Incomplete {name}; do not label a subset as the full benchmark')
    means = {name:statistics.mean(v for _,v in rows) for name,rows in grouped.items()}
    for name in ('mmlu', 'mmlu_continuation'):
        if name in grouped:
            if sample_counts is None or any(sample_counts.get(t,0)<=0 for t,_ in grouped[name]):
                raise ValueError('MMLU requires per-subject scored example counts')
            means[name] = sum(v*sample_counts[t] for t,v in grouped[name])/sum(sample_counts[t] for t,_ in grouped[name])
    categories = {'language_diagnostics': ('blimp','lambada_openai'), 'generation': ('gsm8k',),
                  'continuation_diagnostics': ('mmlu_continuation',)}
    groups = {}
    for category, names in categories.items():
        values = {k:v for k,v in means.items() if k in names}
        if values: groups[category] = dict(scores=values, average=statistics.mean(values.values()))
    values = {k:v for k,v in means.items() if k not in sum(categories.values(), ())}
    if values: groups['multiple_choice'] = dict(scores=values, average=statistics.mean(values.values()))
    if paloma:
        groups['corpus_likelihood'] = dict(scores=paloma, higher_is_better=False,
            average={metric:statistics.mean(row[metric] for row in paloma.values()) for metric in next(iter(paloma.values()))},
            aggregation='descriptive unweighted mean across selected sources, not pooled corpus perplexity',
            source_count=len(paloma))
    return dict(groups=groups, aggregation='unweighted benchmark means; MMLU variants separately weighted by scored examples; BLiMP macro-subset; TruthfulQA mean MC1/MC2; XCSR mean of its two tasks; accuracy scores on 0–1 scale; Paloma likelihood metrics separate')


def glue_average(scores):
    required = {'glue_'+s for s in ('cola','sst2','mrpc','qqp','stsb','mnli_matched','mnli_mismatched','qnli','rte','wnli')}
    if set(scores) != required: raise ValueError('Full GLUE local aggregate requires all nine tasks, both MNLI splits')
    values = {k:v for k,v in scores.items() if not k.startswith('glue_mnli_')}
    values['glue_mnli'] = (scores['glue_mnli_matched']+scores['glue_mnli_mismatched'])/2
    if any(not math.isfinite(v) for v in values.values()): raise ValueError('Nonfinite GLUE score')
    return statistics.mean(values.values())

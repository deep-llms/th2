"""Fixed sharing placement controls; use the unchanged shared-attention decoder."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from efficient_llm.kv_mlp import validate_map
from efficient_llm.diagnostic import paired_summary, write


LAYOUTS = ('full50', 'uniform25', 'upper25')


def layout_sources(layers, layout):
    if layers < 4 or layers % 4 or layout not in LAYOUTS:
        raise ValueError('Placement controls require a multiple of four layers and a known layout')
    consumers = (range(1, layers, 2) if layout == 'full50' else
                 range(3, layers, 4) if layout == 'uniform25' else
                 range(layers//2+1, layers, 2))
    result = list(range(layers))
    for i in consumers:
        result[i] = i-1
    validate_map(result, layers)
    return result


def apply_layout(model, layout):
    n = model.config.num_hidden_layers
    if (model.config.kv_sources != list(range(n))
            or model.config.mlp_widths != [model.config.intermediate_size]*n):
        raise ValueError('Placement conversion requires an unpruned dense parent')
    sources = layout_sources(n, layout)
    for i, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        attention.source_idx = sources[i]
        if sources[i] != i:
            attention.k_proj = attention.v_proj = attention.k_norm = None
    model.config.kv_sources = sources
    return model


def train_affected_qkv(model):
    """Freeze everything except Q/K/V in sharing producer/consumer pairs."""
    sources = model.config.kv_sources
    validate_map(sources, model.config.num_hidden_layers)
    affected = {j for i, source in enumerate(sources) if source != i for j in (source, i)}
    if not affected:
        raise ValueError('QKV adaptation requires a shared model')
    model.requires_grad_(False)
    for i in sorted(affected):
        attention = model.model.layers[i].self_attn
        for name in ('q_proj', 'k_proj', 'v_proj'):
            projection = getattr(attention, name)
            if projection is not None:
                projection.requires_grad_(True)
    return sorted(affected)


def parameter_fingerprint(model, *, trainable):
    """Exact stored parameter bytes, including names/shapes; never alter weights."""
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
            digest.update(parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def summarize(runs, baseline, output):
    a = json.loads(baseline.read_text())
    if a['status'] != 'passed' or a['arm'] != 'A' or a['smoke'] or a['phase'] != 'new_recovery':
        raise ValueError('Require completed matched dense recovery')
    reports = {layout: json.loads((runs/layout/'result.json').read_text()) for layout in LAYOUTS}
    recipe_keys = ['learning_rate', 'lr_scheduler_type', 'warmup_steps', 'weight_decay',
                   'max_grad_norm', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                   'seed', 'data_seed', 'bf16']
    comparisons = {}
    for layout, b in reports.items():
        if (b['status'] != 'passed' or b['smoke'] or b['phase'] != 'placement_conversion'
                or b['sharing_layout'] != layout):
            raise ValueError('Incomplete placement: '+layout)
        for key in ['steps', 'world_size', 'input_tokens', 'sample_order_sha256_per_rank',
                    'probe_sha256', 'parent_steps', 'parent_result_sha256', 'initial_model_files']:
            if a[key] != b[key]:raise ValueError('Unmatched placement: '+key)
        if any(a['training_arguments'][k] != b['training_arguments'][k] for k in recipe_keys):
            raise ValueError('Unmatched recovery recipe')
        expected = layout_sources(len(a['kv_sources']), layout)
        if b['kv_sources'] != expected or b['mlp_widths'] != a['mlp_widths']:
            raise ValueError('Unexpected placement architecture')
        x, y = a['inference_validation'], b['inference_validation']
        paired = paired_summary(y['per_document_nll'], x['per_document_nll'])
        comparisons[layout] = dict(ppl=y['ppl'], ppl_ratio=y['ppl']/x['ppl'], paired=paired,
            cache_fraction=len(set(expected))/len(expected), training_seconds=b['training_seconds'],
            within_five_percent=paired['paired_document_bootstrap_95ci'][1] <= math.log(1.05))
    upper = reports['upper25']['inference_validation']
    uniform = reports['uniform25']['inference_validation']
    write(output, dict(status='passed', dense_ppl=a['inference_validation']['ppl'],
        comparisons=comparisons, upper_minus_uniform=paired_summary(
            upper['per_document_nll'], uniform['per_document_nll']),
        limitations=['One seed and reused 256-document development probe.',
                     'Cache fractions are structural; serving speed and long-context quality unmeasured.']))


def summarize_qkv(result, control, baseline, output):
    q, c, a = (json.loads(p.read_text()) for p in (result, control, baseline))
    if (q.get('trainable_scope') != 'affected_qkv' or c.get('trainable_scope', 'all') != 'all'
            or q['sharing_layout'] != 'uniform25' or c['sharing_layout'] != 'uniform25'
            or a['arm'] != 'A' or a['phase'] != 'new_recovery'):
        raise ValueError('Require QKV-only uniform25, all-parameter control and dense baseline')
    for r in (q, c, a):
        if r['status'] != 'passed' or r['smoke']:
            raise ValueError('Incomplete adaptation comparison')
        for key in ['steps', 'world_size', 'input_tokens', 'sample_order_sha256_per_rank',
                    'probe_sha256', 'parent_steps', 'parent_result_sha256', 'initial_model_files']:
            if r[key] != a[key]:raise ValueError('Unmatched adaptation: '+key)
        for key in ['learning_rate', 'lr_scheduler_type', 'warmup_steps', 'weight_decay',
                    'max_grad_norm', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                    'seed', 'data_seed', 'bf16']:
            if r['training_arguments'][key] != a['training_arguments'][key]:
                raise ValueError('Unmatched adaptation recipe: '+key)
    for key in ('kv_sources', 'mlp_widths', 'parameters', 'phase'):
        if q[key] != c[key]:raise ValueError('Changed architecture/phase: '+key)
    initial = [r['validation']['0']['per_document_nll'] for r in (q, c)]
    if max(abs(x-y) for x,y in zip(*initial, strict=True)) > 1e-5:
        raise ValueError('Conversion initial losses differ')
    proof = q['adaptation']
    if (proof['frozen_before'] != proof['frozen_after']
            or proof['trainable_before'] == proof['trainable_after']):
        raise ValueError('Invalid frozen/trainable weight proof')
    comparisons = {}
    x = q['inference_validation']
    for name, ref in [('dense', a), ('all_parameters', c)]:
        y = ref['inference_validation']
        comparisons[name] = dict(ppl_ratio=x['ppl']/y['ppl'], paired=paired_summary(
            x['per_document_nll'], y['per_document_nll']))
    write(output, dict(status='passed', ppl=x['ppl'], comparisons=comparisons,
        within_five_percent=comparisons['dense']['paired']['paired_document_bootstrap_95ci'][1] <= math.log(1.05),
        trainable_parameters=proof['trainable_parameters'], training_seconds=q['training_seconds'],
        frozen_weights_unchanged=True,
        limitations=['One seed, reused development probe; not an independent confirmation.',
                     'Only trainable parameter scope changes; not a reproduction of SwiftKV.']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'baseline', 'output'):p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--control', type=Path)
    a = p.parse_args()
    if a.output.exists():raise ValueError('Require fresh summary output')
    if a.control:
        summarize_qkv(a.runs/'qkv/result.json', a.control, a.baseline, a.output)
    else:
        summarize(a.runs, a.baseline, a.output)


if __name__ == '__main__':main()

"""Fixed sharing placement controls; use the unchanged shared-attention decoder."""
import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'baseline', 'output'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():raise ValueError('Require fresh summary output')
    summarize(a.runs, a.baseline, a.output)


if __name__ == '__main__':main()

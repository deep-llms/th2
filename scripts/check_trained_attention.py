"""Same-weight backward checks on disposable attention benchmark checkpoints.

No optimizer updates. Each worker compares both backends on one fixed batch;
small batches also use math SDPA in FP32 as an independent precision reference.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import load_model
from scripts.benchmark_document_training import capture, compare, new_model, write

MODES = ('sdpa_isolated', 'fa4_isolated')
CASES = (('data', 2), ('eval', 2), ('data', 16), ('eval', 16))


def fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def acceptable(row):
    return (row['targets'][0] == row['targets'][1] and row['loss_abs_diff'] < .01
            and row['gradient_relative_l2'] < .03
            and row['hidden']['relative_l2'] < .02
            and row['logits']['relative_l2'] < .02)


def worker(args):
    from datasets import load_from_disk
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite a diagnostic receipt')
    mode = MODES[args.index // len(CASES)]
    split, batch = CASES[args.index % len(CASES)]
    checkpoint = source/mode/'final_model/model.safetensors'
    config = json.loads((source/'data.json').read_text())['recipe']['config_name']
    model = new_model(config, 'cpu')
    # load_model handles tied embeddings and requires exact checkpoint keys.
    load_model(model, str(checkpoint), strict=True, device='cpu')
    initial = fingerprint(model)
    model.to('cuda')
    rows = list(load_from_disk(str(source/split)).select(range(batch)))
    assert all(len(row['input_ids']) == 2048 for row in rows)
    assert any(len(set(row['segments'])) > 1 for row in rows)
    assert model.backbone.config.attention_dropout == 0
    digest = hashlib.sha256()
    for key in ('input_ids', 'segments'):
        digest.update(torch.tensor([r[key] for r in rows], dtype=torch.long).numpy().tobytes())
    result = dict(checkpoint=str(checkpoint), parameter_sha256=initial, split=split,
                  batch=batch, input_sha256=digest.hexdigest(), index=args.index, comparisons={})
    def take(backend, fp32=False):
        print('CAPTURE', args.index, backend, 'fp32' if fp32 else 'bf16', flush=True)
        with sdpa_kernel(SDPBackend.MATH) if fp32 else nullcontext():
            return capture(model, rows, backend, bf16=not fp32)
    dense = take('sdpa_isolated')
    flash = take('fa4_isolated')
    result['comparisons']['fa4_vs_dense'] = compare(dense, flash, False)
    repeat = take('sdpa_isolated')
    result['comparisons']['dense_repeat'] = compare(dense, repeat, False)
    del repeat
    repeat = take('fa4_isolated')
    result['comparisons']['fa4_repeat'] = compare(flash, repeat, False)
    del repeat
    # Full FP32 at micro16 has not been capacity-verified. Use micro2 only.
    if batch == 2:
        reference = take('sdpa_isolated', fp32=True)
        result['comparisons']['dense_vs_fp32_math'] = compare(reference, dense, False)
        result['comparisons']['fa4_vs_fp32_math'] = compare(reference, flash, False)
        del reference
    del dense, flash
    result['parameters_unchanged'] = fingerprint(model) == initial
    result['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    result['backend_gate_passed'] = acceptable(result['comparisons']['fa4_vs_dense'])
    result['reference_gates_passed'] = (all(acceptable(result['comparisons'][key])
        for key in ('dense_vs_fp32_math', 'fa4_vs_fp32_math')) if batch == 2 else None)
    result['status'] = 'completed'
    write(output, result)
    assert result['parameters_unchanged'], 'Diagnostic modified checkpoint parameters'
    print('SAME_WEIGHT_RESULT', json.dumps(result), flush=True)


def summarize(args):
    root = Path(args.output)
    rows = [json.loads((root/f'case-{i}.json').read_text()) for i in range(8)]
    for i, row in enumerate(rows):
        assert row['index'] == i and row['status'] == 'completed' and row['parameters_unchanged']
        assert row['split'] == CASES[i % 4][0] and row['batch'] == CASES[i % 4][1]
        assert row['checkpoint'] == str(Path(args.source)/MODES[i//4]/'final_model/model.safetensors')
        assert row['input_sha256'] == rows[(i+4)%8]['input_sha256']
        assert row['parameter_sha256'] == rows[i//4*4]['parameter_sha256']
    passed = all(r['backend_gate_passed'] for r in rows)
    # FP32 comparisons are reported separately; never hide reference failures.
    reference_passed = all(r['reference_gates_passed'] for r in rows if r['batch'] == 2)
    result = dict(status='passed' if passed and reference_passed else 'failed',
                  backend_gates_passed=passed, fp32_reference_gates_passed=reference_passed,
                  tolerances=dict(loss_absolute=.01, gradient_relative_l2=.03, output_relative_l2=.02),
                  cases=rows)
    write(root/'summary.json', result)
    print('TRAINED_ATTENTION_SUMMARY', json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('Same-weight numerical gate failed; complete evidence saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'summarize'])
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', type=int, choices=range(8))
    args = parser.parse_args()
    if args.command == 'worker' and args.index is None:
        parser.error('worker requires --index')
    {'worker': worker, 'summarize': summarize}[args.command](args)

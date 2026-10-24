"""CUDA BF16 loss/gradient/update checks for optional dual-exit optimizations.

Use an allocated visible GPU. Does not manage GPUs or modify experiment files.
The baseline clone repeat measures backend noise separately from optimization.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import Qwen3Config
from bdot.model import BDOTForCausalLM
from packed_training import PackedCollator, model_inputs


def run(config, inputs, backend, *, shared=False, native=False):
    torch.manual_seed(73)
    c = Qwen3Config(**copy.deepcopy(config))
    c._attn_implementation = backend
    c.bdot_exit_training.update(shared_prefix=shared, ce_backend='native' if native else 'chunked')
    m = BDOTForCausalLM(c).cuda().train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-5, betas=(.9, .95), weight_decay=.1)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = m(**inputs)
    output.loss.backward()
    grads = {name: p.grad.detach().cpu() for name, p in m.named_parameters() if p.grad is not None}
    loss = float(output.loss.detach())
    logits = output.logits.detach().cpu()
    norm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    weights = {name: p.detach().cpu() for name, p in m.named_parameters()}
    m.finish_update()
    state = {name: getattr(m, name).detach().cpu() for name in
             ('routing_counter', 'completed_steps', 'training_totals', 'exit_training_totals', 'balance_bias')}
    del output, m, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return dict(loss=loss, logits=logits, grads=grads, weights=weights, state=state, grad_norm=norm)


def compare(ref, candidate):
    if ref['grads'].keys() != candidate['grads'].keys():
        raise AssertionError('Trainable gradient coverage changed')
    error = signal = dot = norm2 = 0.
    max_gradient = max_weight = 0.
    for name, x in ref['grads'].items():
        y = candidate['grads'][name]
        diff = y - x
        error += float(diff.square().sum()); signal += float(x.square().sum())
        dot += float((x*y).sum()); norm2 += float(y.square().sum())
        max_gradient = max(max_gradient, float(diff.abs().max()))
    for name, x in ref['weights'].items():
        max_weight = max(max_weight, float((candidate['weights'][name]-x).abs().max()))
    relative = math.sqrt(error / max(signal, 1e-30))
    cosine = dot / math.sqrt(max(signal * norm2, 1e-60))
    logits_max = float((ref['logits'].float()-candidate['logits'].float()).abs().max())
    loss_gap = abs(ref['loss'] - candidate['loss'])
    state_error = {name: float((x-candidate['state'][name]).abs().max()) for name, x in ref['state'].items()}
    # Loss/gradient accumulation changes can perturb BF16 rounding; record it,
    # do not call mathematical equivalence bitwise replay equivalence.
    passed = (loss_gap <= 1e-4 and logits_max == 0 and relative < .01 and cosine > .9999
              and max_weight <= 1e-4 and state_error['routing_counter'] == 0
              and state_error['completed_steps'] == 0 and state_error['balance_bias'] == 0)
    return dict(passed=passed, loss_abs=loss_gap, full_logits_max_abs=logits_max,
                gradient_relative_l2=relative, gradient_cosine=cosine,
                gradient_max_abs=max_gradient, weight_max_abs_after_adamw=max_weight,
                state_max_abs=state_error, loss=candidate['loss'], grad_norm=candidate['grad_norm'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attention-backend', choices=('sdpa', 'fa4'), default='sdpa')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--length', type=int, default=128)
    a = p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or a.output.exists():
        p.error('Allocate a visible GPU and choose a fresh receipt')
    if a.length < 4 or a.batch_size < 1:
        p.error('Invalid test dimensions')
    torch.set_num_threads(1)
    config = json.loads(a.config.read_text())
    if config.get('bdot_exit_training', {}).get('kl_weight') != 0:
        p.error('Use a DualCE config with zero KL')
    backend = 'sdpa'
    if a.attention_backend == 'fa4':
        from packed_fa4 import register_backend
        backend = register_backend()
    gen = torch.Generator().manual_seed(19)
    ids = torch.randint(config['vocab_size'], (a.batch_size, a.length), generator=gen)
    rows = [dict(input_ids=x.tolist(), labels=x.tolist(), attention_mask=[1]*a.length,
                 segments=(torch.arange(a.length)//(a.length//2)).tolist()) for x in ids]
    inputs = model_inputs({k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in
                           PackedCollator(attention_backend=a.attention_backend)(rows).items()})
    ref = run(config, inputs, backend)
    results = {}
    for name, flags in [('reference_repeat', {}), ('native_ce', dict(native=True)),
                        ('shared_prefix_native_ce', dict(native=True, shared=True))]:
        result = run(config, inputs, backend, **flags)
        results[name] = compare(ref, result)
        del result
        print(name, json.dumps(results[name]), flush=True)
    receipt = dict(status='passed' if all(x['passed'] for x in results.values()) else 'failed',
                   torch=torch.__version__, gpu=torch.cuda.get_device_name(0), backend=a.attention_backend,
                   config=config, batch_size=a.batch_size, length=a.length, comparisons=results,
                   source_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'bdot/exit_training.py').read_bytes()).hexdigest(),
                   scope='Full model/vocabulary, BF16 one-update CUDA numerical test; not convergence or DDP validation.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(receipt, f, indent=2)
    if receipt['status'] != 'passed':
        raise SystemExit('Numerical gates failed; inspect receipt')


if __name__ == '__main__':
    main()

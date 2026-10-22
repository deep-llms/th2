"""Compare dispatch optimizations to a frozen pre-change package, CPU or CUDA.

The reference directory is supplied by the caller, never downloaded. CUDA use
requires an explicitly allocated device. Check routes, unused gradients, loss,
AdamW with existing moments, routing state and lazy diagnostics.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--partial', action='store_true', help='Include frozen partial-MLP arms')
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output path required')
    if a.device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Explicit GPU allocation required')
    import torch
    from transformers import Qwen3Config
    from bdot.model import BDOTForCausalLM
    from bdot.routing import RoutingConfig
    from packed_training import PackedCollator, model_inputs
    # Isolate relative imports too: old model + new grouped/routing helpers
    # would not be a genuine pre-change reference.
    import importlib
    from dataclasses import fields
    name = '_bdot_frozen_reference'
    spec = importlib.util.spec_from_file_location(name, a.reference/'bdot/__init__.py',
        submodule_search_locations=[str(a.reference/'bdot')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    reference = importlib.import_module(name+'.model')
    old_fields = {f.name for f in fields(importlib.import_module(name+'.routing').RoutingConfig)}
    torch.set_num_threads(1)
    torch.manual_seed(31)
    variants = [
        dict(routing='fixed', curriculum='none', r=0),
        dict(curriculum='A', progress=0.),
        dict(curriculum='A', progress=.2, reuse=False),
        dict(curriculum='B', progress=.01),
        dict(curriculum='B', progress=.01, reuse=False),
        dict(curriculum='B', progress=.7),
        dict(curriculum='none', gate='scale'),
        dict(routing='random', curriculum='none'),
    ]
    if a.partial:
        variants += [dict(route_scope='mlp',routed_mlp_fraction=f,curriculum='A',routing=route)
                     for f in (.25,.5,1.) for route in ('learned','random')]
    records = []
    for backend in (('sdpa', 'fa4') if a.device == 'cuda' else ('sdpa',)):
        for expert_backend in (('reference', 'grouped') if a.device == 'cuda' else ('reference',)):
            for i, options in enumerate(variants):
                options = dict(options)
                progress = options.pop('progress', 1.)
                gpu = a.device == 'cuda'
                config = Qwen3Config(
                    vocab_size=256, hidden_size=256 if gpu else 32,
                    intermediate_size=768 if gpu else 64, num_hidden_layers=6,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=128 if gpu else 8,
                    attention_dropout=0., tie_word_embeddings=True, rms_norm_eps=1e-6,
                    max_position_embeddings=128, use_cache=False)
                config._attn_implementation = 'sdpa'
                if backend == 'fa4':
                    from packed_fa4 import register_backend
                    config._attn_implementation = register_backend()
                config.bdot = RoutingConfig(n_stem=1, n_coda=1, S=4, default_prior=0.,
                                            expert_backend=expert_backend, **options).to_dict()
                old_config = copy.deepcopy(config)
                old_config.bdot = {k:v for k,v in config.bdot.items() if k in old_fields}
                old = reference.BDOTForCausalLM(old_config).to(a.device)
                new = BDOTForCausalLM(copy.deepcopy(config)).to(a.device)
                new.load_state_dict(old.state_dict())
                old.progress = new.progress = progress
                old.capture_routes = new.capture_routes = True
                # Alternate checkpointing, including soft, home, and quota cases.
                if i % 2:
                    for model in (old, new):
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={'use_reentrant': False})
                ids = torch.randint(2, 256, (2, 64)).tolist()
                rows = [dict(input_ids=x, labels=x.copy(), attention_mask=[1]*64,
                             segments=[0]*17+[1]*47) for x in ids]
                inputs = model_inputs({k: v.to(a.device) if isinstance(v, torch.Tensor) else v
                    for k, v in PackedCollator(attention_backend=backend)(rows).items()})
                optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.1)
                              for model in (old, new)]
                for model, optimizer in zip((old, new), optimizers):
                    for parameter in model.parameters():
                        optimizer.state[parameter] = dict(step=torch.tensor(5.),
                            exp_avg=torch.full_like(parameter, .001),
                            exp_avg_sq=torch.full_like(parameter, .0001))
                errors = []
                for update in range(3):
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(a.device, dtype=torch.bfloat16, enabled=gpu):
                        x = old(**inputs)
                        y = new(**inputs)
                    torch.testing.assert_close(old.last_routes, new.last_routes, atol=0, rtol=0)
                    try:
                        torch.testing.assert_close(x.logits, y.logits, atol=.002 if gpu else 1e-6,
                                                   rtol=.002 if gpu else 1e-5)
                    except AssertionError:
                        print('DISPATCH_FAILURE_CASE', backend, expert_backend, options, update, flush=True)
                        seen, hooks = {}, []
                        for label, model in [('old', old), ('new', new)]:
                            for key, module in model.named_modules():
                                if (key.startswith('model.bdot_experts.')
                                        or key in [f'model.layers.{i}' for i in range(6)]):
                                    def capture(module, args, result, label=label, key=key):
                                        if isinstance(result, torch.Tensor):
                                            seen[label,key] = (args[0].detach().cpu(), result.detach().cpu(),
                                                               args[0].stride(), result.stride())
                                    hooks.append(module.register_forward_hook(capture))
                        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16, enabled=gpu):
                            repeated = old(**inputs).logits
                            new(**inputs)
                        print('OLD_REPEAT_MAX_ABS', float((repeated-x.logits).abs().max()), flush=True)
                        for label, key in seen:
                            if label == 'old' and ('new', key) in seen:
                                oi, oo, ois, oos = seen[label,key]
                                ni, no, nis, nos = seen['new',key]
                                print('ACTIVATION', key, str(oi.dtype), str(ni.dtype), str(oo.dtype),
                                      str(no.dtype), float((oi.flatten()-ni.flatten()).abs().max()),
                                      float((oo.flatten()-no.flatten()).abs().max()),
                                      ois, nis, oos, nos, flush=True)
                        for hook in hooks:
                            hook.remove()
                        raise
                    x.loss.backward()
                    y.loss.backward()
                    numerator = denominator = 0.
                    for (key, op), (_, np) in zip(old.named_parameters(), new.named_parameters()):
                        assert (op.grad is None) == (np.grad is None), key
                        if op.grad is not None:
                            assert torch.isfinite(np.grad).all(), key
                            numerator += float((op.grad.float()-np.grad.float()).square().sum())
                            denominator += float(op.grad.float().square().sum())
                    error = (numerator/max(denominator, 1e-30))**.5
                    assert error < (.01 if gpu else 1e-5), error
                    errors.append(error)
                    for model, optimizer in zip((old, new), optimizers):
                        optimizer.step()
                        model.finish_update()
                    step_delta = max(float((op-np).detach().abs().max())
                                     for op,np in zip(old.parameters(),new.parameters()))
                    print('DISPATCH_UPDATE '+json.dumps(dict(backend=backend, expert_backend=expert_backend,
                        variant=i, update=update, gradient_relative=error, parameter_max_abs=step_delta)), flush=True)
                    torch.testing.assert_close(old.balance_bias, new.balance_bias, atol=0, rtol=0)
                    torch.testing.assert_close(old.training_totals, new.training_totals,
                                               atol=.01 if gpu else 1e-4, rtol=1e-5)
                    # Reading diagnostics must not change any state; repeated reads
                    # and reads delayed across forward calls are supported.
                    assert new.last_update_metrics == new.last_update_metrics
                    for key, value in old.last_update_metrics.items():
                        other = new.last_update_metrics[key]
                        if value is None:
                            assert other is None
                        else:
                            assert abs(value-other) <= 1e-4*max(1., abs(value)), (key,value,other)
                delta = max(float((op-np).detach().abs().max())
                            for op,np in zip(old.parameters(),new.parameters()))
                assert delta < (1e-5 if gpu else 1e-6), delta
                records.append(dict(backend=backend, expert_backend=expert_backend, options=options,
                    progress=progress, checkpointing=bool(i % 2), gradient_relative=errors,
                    max_parameter_difference=delta))
                print('DISPATCH_CASE '+json.dumps(records[-1]), flush=True)
                del old, new, optimizers, x, y
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(status='passed', cases=records), indent=2)+'\n')


if __name__ == '__main__':
    main()

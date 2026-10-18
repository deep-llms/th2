"""Real CUDA numerical checks for B-DOT. No CPU attention substitute.

Set CUDA_VISIBLE_DEVICES to one authorized free GPU. Tiny dimensions retain
12 layers, S=8, head_dim=128 and GQA. --full-native adds spec-size FP32 T1.
"""
import argparse
import copy
from contextlib import contextmanager, nullcontext
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import Qwen3Config
from bdot.model import BDOTForCausalLM, RoutedStep
from bdot.routing import RoutingConfig
from packed_training import PackedCollator, model_inputs


def make_model(**options):
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=768,
        num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, attention_dropout=0., use_cache=False, max_position_embeddings=256,
        tie_word_embeddings=True, rms_norm_eps=1e-6)
    cfg._attn_implementation = 'sdpa'
    cfg.bdot = RoutingConfig(**options).to_dict()
    return BDOTForCausalLM(cfg).cuda().train()


def native_pairs(model, ref):
    yield 'embedding', model.model.embed_tokens.weight, ref.model.embed_tokens.weight
    yield 'final_norm', model.model.norm.weight, ref.model.norm.weight
    for i, layer in enumerate(model.model.layers):
        other = ref.model.layers[i]
        if not isinstance(layer, RoutedStep):
            for (name, p), (_, q) in zip(layer.named_parameters(), other.named_parameters()):
                yield f'{i}.{name}', p, q
        else:
            expert = model.model.bdot_experts[layer.step]
            for name in ('q_proj', 'q_norm', 'o_proj'):
                yield f'{i}.{name}', getattr(expert, name).weight, getattr(other.self_attn, name).weight
            for name in ('k_proj', 'k_norm', 'v_proj'):
                yield f'{i}.{name}', getattr(layer, name).weight, getattr(other.self_attn, name).weight
            yield f'{i}.input_norm', layer.input_layernorm.weight, other.input_layernorm.weight
            yield f'{i}.mlp_norm', expert.mlp_norm.weight, other.post_attention_layernorm.weight
            for (name, p), (_, q) in zip(expert.mlp.named_parameters(), other.mlp.named_parameters()):
                yield f'{i}.{name}', p, q


def differences(pairs, grad=False):
    sums, maxima, names = [], [], []
    for name, p, q in pairs:
        p, q = (p.grad, q.grad) if grad else (p, q)
        if p is None or q is None:
            if p is not None or q is not None:
                raise AssertionError(f'Different unused parameter: {name}')
            continue
        p, q = p.detach().float(), q.detach().float()
        diff = p-q
        sums.append(torch.stack((diff.square().sum(), q.square().sum())))
        maxima.append(diff.abs().max())
        names.append(name)
    totals = torch.stack(sums).sum(0)
    max_values = torch.stack(maxima)
    return dict(relative_l2=float((totals[0]/totals[1].clamp_min(1e-24)).sqrt()),
                max_abs=float(max_values.max()), worst_parameter=names[int(max_values.argmax())],
                finite=bool(torch.isfinite(totals).all() & torch.isfinite(max_values).all()))


def inputs(rows, backend):
    return model_inputs({k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in PackedCollator(attention_backend=backend)(rows).items()})


@contextmanager
def capture_choices(model):
    """Record both soft branches; primary paths alone miss top-two changes."""
    choices, originals = [], []
    for step in getattr(model, 'routed_steps', []):
        original = step.select
        originals.append((step, original))
        def tracked(h, context, original=original):
            selected, weights = original(h, context)
            choices.append(selected.detach().clone())
            return selected, weights
        step.select = tracked
    try:
        yield choices
    finally:
        for step, original in originals:
            step.select = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--fa4', action='store_true')
    p.add_argument('--full-native', action='store_true')
    p.add_argument('--native-only', action='store_true')
    p.add_argument('--reference-math', action='store_true', help='FP32 T1 uses deterministic SDPA math reference')
    p.add_argument('--native-self-control', action='store_true', help='Compare native Qwen3 against an identical native copy')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        p.error('Choose a fresh receipt')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
        p.error('Explicitly allocate exactly one CUDA GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.fa4:
        from packed_fa4 import register_backend
        register_backend()
    generator = torch.Generator().manual_seed(9)
    rows = []
    for boundary in (37, 81):
        ids = torch.randint(0, 512, (128,), generator=generator).tolist()
        rows.append(dict(input_ids=ids, labels=ids, attention_mask=[1]*128,
                         segments=[0]*boundary + [1]*(128-boundary)))
    result = dict(status='running', torch=torch.__version__, cuda=torch.version.cuda,
                  transformers=version('transformers'), gpu=torch.cuda.get_device_name(),
                  fa4=version('flash-attn-4') if a.fa4 else None, checks=[])
    out.parent.mkdir(parents=True, exist_ok=True)
    def record(value):
        result['checks'].append(value)
        out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(value), flush=True)
    def named_pairs(x, y):
        right = dict(y.named_parameters())
        return [(n, p, right[n]) for n, p in x.named_parameters()]
    def compare(x, y, backend, bf16, pairs, name, exact=False, atol=1e-4):
        with capture_choices(x) as xchoices, capture_choices(y) as ychoices, torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            first = x(**inputs(rows, 'sdpa'))
            second = y(**inputs(rows, backend))
        first.loss.backward()
        second.loss.backward()
        logits = differences([('logits', first.logits, second.logits)])
        gradients = differences(pairs, grad=True)
        rec = dict(name=name, loss_abs=float((first.loss-second.loss).abs().detach()),
                   logits=logits, gradients=gradients, bf16=bf16, backend=backend)
        if getattr(x, 'capture_routes', False):
            rec['route_disagreement_fraction'] = float((x.last_routes != y.last_routes).float().mean())
            rec['all_branch_disagreements'] = sum(int((a != b).sum()) for a,b in zip(xchoices,ychoices))
        record(rec)
        if rec.get('all_branch_disagreements', 0):
            raise AssertionError('Different discrete paths (including secondary soft choices); '
                                 'this is not an identical-path gradient comparison: '+str(rec))
        assert logits['finite'] and gradients['finite'], rec
        if exact:
            assert max(logits['max_abs'], gradients['max_abs']) <= atol, rec
        else:
            assert logits['relative_l2'] <= .02 and gradients['relative_l2'] <= .05, rec
        return first, second, rec

    # T1: compare mapped native parameters, gradients and one AdamW update.
    for bf16 in (False, True):
        x = make_model(routing='fixed', curriculum='none', step_film=False)
        if a.full_native and not bf16:
            del x
            config = json.loads((Path(__file__).resolve().parents[2]/'configs/bdot/D.json').read_text())
            config['bdot']['step_film'] = False
            x = BDOTForCausalLM(Qwen3Config(**config)).cuda().train()
        y = x.native_reference().train()
        if a.native_self_control:
            x = copy.deepcopy(y)
        pairs = named_pairs(x,y) if a.native_self_control else list(native_pairs(x, y))
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel(SDPBackend.MATH) if a.reference_math and not bf16 else nullcontext()
        with ctx:
            first, second, rec = compare(x, y, 'sdpa', bf16, pairs, 'fixed_native', exact=True)
        for model in (x, y):
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.1)
            opt.step()
            del opt
        update = differences(pairs)
        record(dict(name='fixed_native_adamw', bf16=bf16, parameters=update))
        assert update['finite'] and update['max_abs'] < 1e-4
        del x, y, pairs, first, second
        torch.cuda.empty_cache()

    if a.native_only:
        result.update(status='passed', settings=vars(a))
        out.write_text(json.dumps(result, indent=2)+'\n')
        return

    for reuse in (True, False):
        for progress in (.01, .7):
            for backend in (['sdpa', 'fa4'] if a.fa4 else ['sdpa']):
                x = make_model(reuse=reuse, default_prior=0., curriculum='B')
                y = copy.deepcopy(x)
                x.progress = y.progress = progress
                x.capture_routes = y.capture_routes = True
                if backend == 'fa4':
                    y.set_attn_implementation('template_fa4')
                y.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                # Controlled weights keep routes stable across small BF16 kernel
                # differences; report all natural-route disagreements explicitly.
                first, second, rec = compare(x, y, backend, True, named_pairs(x, y),
                    f'reuse={reuse},progress={progress},checkpointed_{backend}', exact=backend=='sdpa')
                torch.testing.assert_close(x.load_counts, y.load_counts, atol=0, rtol=0)
                if rec['route_disagreement_fraction']:
                    raise AssertionError('Route decisions differ; do not interpret as an identical-path comparison')
                # Dynamic routing changes expert GEMM group sizes when another
                # document changes; BF16 can then round differently. Report the
                # natural-path difference, and freeze routes to test mask leakage
                # independently of that group-size effect.
                y.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    before = y(**inputs(rows, backend)).logits
                    frozen_routes = y.last_routes.clone()
                    changed = copy.deepcopy(rows)
                    changed[0]['input_ids'][:37] = [(v+13)%512 for v in rows[0]['input_ids'][:37]]
                    after = y(**inputs(changed, backend)).logits
                    record(dict(name='natural_route_bf16_grouping_effect', backend=backend,
                        reuse=reuse, max_abs=float((before[0,37:]-after[0,37:]).abs().max()),
                        route_changes=int((frozen_routes[:,0,37:] != y.last_routes[:,0,37:]).sum())))
                    after = y(**inputs(changed, backend), bdot_forced_routes=frozen_routes).logits
                    torch.testing.assert_close(before[0,37:], after[0,37:], atol=0, rtol=0)
                    torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
                    changed = copy.deepcopy(rows)
                    changed[0]['input_ids'][20:37] = [(v+11)%512 for v in rows[0]['input_ids'][20:37]]
                    after = y(**inputs(changed, backend), bdot_forced_routes=frozen_routes).logits
                    torch.testing.assert_close(before[0,:20], after[0,:20], atol=0, rtol=0)
                # A mask leak also produces a nonzero input gradient across the
                # boundary. Check directly, without relying on roundoff tolerance.
                y.zero_grad(set_to_none=True)
                batch = inputs(rows, backend)
                embedded = y.model.embed_tokens(batch.pop('input_ids')).detach().requires_grad_()
                batch.pop('labels')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = y(inputs_embeds=embedded, **batch).logits
                    logits[0,37:].float().square().sum().backward()
                assert torch.count_nonzero(embedded.grad[0,:37]) == 0
                assert torch.count_nonzero(embedded.grad[1]) == 0
                record(dict(name='causality_and_isolation', backend=backend, reuse=reuse,
                            status='passed', fixed_route_bitwise=True, cross_document_gradient=0))
                del x, y, first, second, before, after
                torch.cuda.empty_cache()
    result['status'] = 'passed'
    result['settings'] = vars(a)
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

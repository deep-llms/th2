"""Evaluation-only, single-site intervention toward the frozen consumer teacher.

The teacher comes from the unmodified trajectory. This is deliberately NOT a
self-consistent recurrent rollout or a bound on an achievable trained model.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch

ARMS = ('P7-simple', 'P7-simple-lm', 'P7-simple-deep', 'P7-simple-consumer')


@torch.no_grad()
def capture(model, context):
    if model.training or model.arm not in ARMS or not model.proxy_enabled:
        raise ValueError('Expected an enabled, evaluated supported P7 checkpoint')
    sites, windows, targets = {}, {}, {}
    attention, block = model.memory_attention, model.block

    def record_attention(layer, u, head, prediction, mask, rotary, plan):
        output = attention(layer, u, head, prediction, mask, rotary, plan)
        sites[layer.self_attn.layer_idx+1] = dict(
            args=(layer, u, head, prediction, mask, rotary, plan), student=output)
        return output

    def record_block(index, *args, **kwargs):
        result = block(index, *args, **kwargs)
        layer = index+1
        # Same FP32 sum and order as training, before the site's deep adapter.
        for site in model.layers:
            if site <= layer < site+model.settings.lookahead:
                value = result[1].float()
                windows[site] = windows[site]+value if site in windows else value
                if layer == site+model.settings.lookahead-1:
                    targets[site] = model.normalize_target(windows.pop(site), site)
        return result

    with patch.object(model, 'memory_attention', record_attention), patch.object(model, 'block', record_block):
        hidden = model.hidden_states(context)[0]
    if windows or set(targets) != set(model.layers) or set(sites) != set(targets):
        raise RuntimeError('Incomplete teacher capture')
    for site, entry in sites.items():
        layer, u, head, pred, mask, rotary, plan = entry['args']
        teacher = attention(layer, u, head, targets[site].to(pred.dtype), mask, rotary, plan)
        if not torch.equal(teacher[:, :, :-4], entry['student'][:, :, :-4]):
            raise RuntimeError('Teacher replacement changed the native heads')
        entry.update(teacher=teacher[:, :, -4:], target=targets[site])
    return hidden, sites


@contextmanager
def intervention(model, sites, site, alpha):
    """Change exactly one site's last four head outputs; recompute downstream."""
    if model.training or torch.is_grad_enabled() or site not in sites:
        raise ValueError('Evaluation under no_grad with a captured site is required')
    original = model.memory_attention
    calls = []

    def replace(layer, u, head, prediction, mask, rotary, plan):
        actual = original(layer, u, head, prediction, mask, rotary, plan)
        if layer.self_attn.layer_idx+1 != site:
            return actual
        calls.append(site)
        saved = sites[site]
        if not torch.equal(u, saved['args'][1]) or not torch.equal(actual, saved['student']):
            raise RuntimeError('The intervention source changed before the selected site')
        if alpha == 0:
            return actual
        selected = (actual[:, :, -4:].float() + alpha *
                    (saved['teacher'].float()-actual[:, :, -4:].float())).to(actual.dtype)
        return torch.cat((actual[:, :, :-4], selected), dim=2)

    with patch.object(model, 'memory_attention', replace):
        yield
    if calls != [site]:
        raise RuntimeError('The selected site was not visited exactly once')


@torch.no_grad()
def validate_capture(model, sites, context):
    """Match the actual training consumer objective, including BF16 target cast."""
    from deep_kv.proxy_memory import memory_consumer_loss
    errors = {}
    for site, entry in sites.items():
        layer, u, head, pred, mask, rotary, plan = entry['args']
        teacher = entry['teacher'].float()
        student = entry['student'][:, :, -4:].float()
        expected = (((student-teacher).square().mean((-1,-2)) /
                     (teacher.square().mean((-1,-2))+1e-6))*context.valid).sum(-1)[:,None]
        actual = memory_consumer_loss(layer, head, u, pred, entry['target'], mask,
            rotary, plan, context.valid, model.attention_backend, getattr(model, 'fa4_kernel', None))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        errors[site] = float((actual-expected).abs().max())
    return errors

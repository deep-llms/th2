import copy
from collections import Counter

import unittest
import torch
from torch.nn import functional as F

from efficient_llm.warm_order.embedding_anchor import (
    EmbeddingAnchor, TRAIN_PATHS, HELD_OUT_PATHS, exposure,
)
from efficient_llm.warm_order.model import Config


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    m = EmbeddingAnchor(Config(d=16, ff=32, heads=2, kv_heads=1,
                              head_dim=8, vocab=31, prefix=3, length=9))
    return m, torch.randint(0, 31, (2, 9))


def reference_block(block, h, pos, seg):
    h = h + block.self_attn(block.input_layernorm(h), pos, seg)
    mlp = block.mlp
    z = block.post_attention_layernorm(h)
    return h + mlp.down_proj(F.silu(mlp.gate_proj(z)) * mlp.up_proj(z))


def test_path_and_exposure_contract():
    assert len(TRAIN_PATHS) == len(HELD_OUT_PATHS) == 12
    assert not set(TRAIN_PATHS) & set(HELD_OUT_PATHS)
    for paths in [TRAIN_PATHS, HELD_OUT_PATHS]:
        assert set(Counter((k, b) for p in paths for k, b in enumerate(p)).values()) == {3}
        edges = Counter(edge for p in paths for edge in zip(p, p[1:]))
        assert len(edges) == 12 and set(edges.values()) == {3}
    assert set(Counter(exposure(s, 'A').path for s in range(768)).values()) == {192}
    pairs = Counter((exposure(s, 'A').path[0], exposure(s, 'A').donor) for s in range(768, 1152))
    assert len(pairs) == 16 and set(pairs.values()) == {24}
    for s in range(1920):
        a, c = exposure(s, 'A'), exposure(s, 'C')
        assert (a.row_start, a.row_stop, a.path, a.donor) == (c.row_start, c.row_stop, c.path, c.donor)
        assert a.row_start == s * 32 and a.row_stop == (s + 1) * 32
    assert exposure(1919, 'A').row_stop == 61440
    assert exposure(768, 'A').alpha == .05
    assert exposure(1151, 'A').alpha == .5
    with unittest.TestCase().assertRaises(ValueError): exposure(1920, 'A')


def test_forward_and_skip_reference(fixture):
    m, x = fixture
    assert not hasattr(m, 'stem') and not hasattr(m, 'coda')
    assert len({b.self_attn.q_proj.weight.data_ptr() for b in m.blocks}) == 4
    p, seg = torch.arange(9).expand_as(x), torch.zeros_like(x)
    for alpha in (1., .5):
        for skip in (None, 2):
            e = m.embed(x); h = e
            for slot, i in enumerate((0, 2, 1, 3)):
                if slot: h = (1 - alpha) * e + alpha * h
                if i != skip: h = reference_block(m.blocks[i], h, p, seg)
            torch.testing.assert_close(m.hidden(x, (0, 2, 1, 3), alpha=alpha, skip=skip), h, rtol=0, atol=0)


def test_causality_and_suffix_loss(fixture):
    m, x = fixture
    y = x.clone(); y[:, 5:] = (y[:, 5:] + 1) % 31
    h = m.hidden(x, (2, 1, 3, 0), alpha=.5)
    changed = m.hidden(y, (2, 1, 3, 0), alpha=.5)
    torch.testing.assert_close(h[:, :5], changed[:, :5], rtol=0, atol=0)
    z = m.head_output_norm(m.head_ffn(m.head_input_norm(h)))
    logits = F.linear(z[:, 2:-1], m.embed.weight)
    expected = F.cross_entropy(logits.flatten(0, 1), x[:, 3:].flatten(), reduction='none').reshape(2, 6).mean(1)
    torch.testing.assert_close(m.nll(x, (2, 1, 3, 0), alpha=.5), expected)


def test_detached_donor_and_self_gradients(fixture, donor):
    m, x = fixture; ref = copy.deepcopy(m)
    loss = m.nll(x, (0,), alpha=.3, donor=donor).sum(); loss.backward()
    e = ref.embed(x); pos = torch.arange(9).expand_as(x); seg = torch.zeros_like(x)
    d = reference_block(ref.blocks[donor], e, pos, seg).detach()
    h = reference_block(ref.blocks[0], .7 * e + .3 * d, pos, seg)
    logits = F.linear(ref.readout_hidden(h)[:, 2:-1], ref.embed.weight)
    F.cross_entropy(logits.flatten(0, 1), x[:, 3:].flatten(), reduction='sum').div(6).backward()
    for (name, p), (other_name, q) in zip(m.named_parameters(), ref.named_parameters()):
        assert name == other_name and (p.grad is None) == (q.grad is None)
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad)
    assert all(p.grad is None for b in m.blocks[1:] for p in b.parameters())


def test_inactive_adam_state_does_not_advance(fixture):
    m, x = fixture
    opt = torch.optim.AdamW(m.parameters(), lr=.0006, betas=(.9, .95), weight_decay=0)
    opt.zero_grad(set_to_none=True)
    m.nll(x, (1,)).mean().backward(); opt.step()
    before = {n: p.detach().clone() for n, p in m.blocks[1].named_parameters()}
    before_steps = {n: opt.state[p]['step'].clone() for n, p in m.blocks[1].named_parameters()}
    opt.zero_grad(set_to_none=True)
    m.nll(x, (0,), donor=1, alpha=.5).mean().backward(); opt.step()
    for n, p in m.blocks[1].named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        torch.testing.assert_close(opt.state[p]['step'], before_steps[n], rtol=0, atol=0)
    assert opt.state[m.embed.weight]['step'].item() == 2


class EmbeddingAnchorTests(unittest.TestCase):
    def test_schedules(self):
        test_path_and_exposure_contract()

    def test_reference(self):
        test_forward_and_skip_reference(fixture())

    def test_causal_loss(self):
        test_causality_and_suffix_loss(fixture())

    def test_self_donor(self):
        test_detached_donor_and_self_gradients(fixture(), 0)

    def test_other_donor(self):
        test_detached_donor_and_self_gradients(fixture(), 1)

    def test_adam(self):
        test_inactive_adam_state_does_not_advance(fixture())


if __name__ == '__main__':
    unittest.main()

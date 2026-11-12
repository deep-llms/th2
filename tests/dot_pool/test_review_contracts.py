"""Independent numerical contracts added during the post-run v1.2 review."""
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from efficient_llm.dot_pool.data import e2_batch
from efficient_llm.dot_pool.model import Experts, MLP, Model
from tests.dot_pool.test_core import tiny


class ReviewContracts(unittest.TestCase):
    def test_e1_random_routes_match_each_split_evaluation_draw(self):
        from efficient_llm.dot_pool.data import e1_batch
        from efficient_llm.dot_pool.train import seed
        from scripts.dot_pool_evaluate import routes
        torch.manual_seed(49)
        m = Model(tiny('SR')).eval()
        rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8]] * 17)
        expected = []
        seed(9182)
        m.capture = True
        with torch.no_grad():
            for start in range(0, len(rows), 32):
                inp = e1_batch(rows[start:start+32], 'cpu')
                m(**{k: v for k, v in inp.items() if k != 'labels'})
                expected.append(torch.stack([r['ids'] for r in m.routes], 2)[:, 2].numpy())
        expected = np.concatenate(expected)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ['id_dev', 'ood_dev', 'id_test', 'ood_test', 'ood_bridge', 'ood_dev_bridge']:
                np.save(root / (name + '.npy'), rows)
            routes(root, {'data': str(root), 'study': 'E1'}, m)
            with np.load(root / 'routes_sample.npz') as actual:
                for name in ['id_dev', 'ood_dev', 'id_test', 'ood_test']:
                    np.testing.assert_array_equal(actual[name + '_ids'], expected)

    def test_two_half_experts_reproduce_dense_ffn(self):
        torch.manual_seed(46)
        c = tiny('SO')
        dense = MLP(c.d, 2 * c.expert_ff)
        experts = Experts(2, c)
        with torch.no_grad():
            for i in range(2):
                sl = slice(i*c.expert_ff, (i+1)*c.expert_ff)
                experts.gate_up[i].copy_(torch.cat([
                    dense.gate_proj.weight[sl].T, dense.up_proj.weight[sl].T], -1))
                experts.down[i].copy_(dense.down_proj.weight[:, sl].T)
        x = torch.randn(7, c.d)
        ids = torch.tensor([[0, 1]]).expand(7, 2)
        actual = 2 * (experts(x, ids) * .5).sum(1)
        torch.testing.assert_close(actual, dense(x), atol=1e-7, rtol=1e-5)

    def test_sequence_balance_matches_explicit_spec(self):
        for arm in ['SO', 'DOT', 'PL']:
            torch.manual_seed(47)
            m = Model(replace(tiny(arm), study='E2', label_rate=0))
            m.capture = True
            observed = []
            original = m._router_logits
            def capture(s, x, previous):
                logits = original(s, x, previous)
                observed.append(logits)
                return logits
            m._router_logits = capture
            inp = e2_batch([[1, 2, 0, 3, 4], [5, 0, 6, 7, 8]], 'cpu')
            inp['segments'][1, -2:] = -1
            inp['labels'][1, -2:] = -100
            result = m(**inp)
            values = []
            for b in range(2):
                valid = inp['segments'][b] >= 0
                scopes = [list(range(m.c.middle))] if arm != 'PL' else [[s] for s in range(m.c.middle)]
                for scope in scopes:
                    n = observed[0].shape[-1]
                    counts = torch.zeros(n)
                    probs = []
                    for s in scope:
                        for t in valid.nonzero().flatten():
                            sig = observed[s][b, t].sigmoid()
                            probs.append(sig / sig.sum())
                            for expert in m.routes[s]['ids'][b, t]:
                                counts[expert] += 1
                    frequency = n * counts / (m.c.k * len(probs))
                    values.append((frequency * torch.stack(probs).mean(0)).sum())
            reference = m.c.balance_alpha * torch.stack(values).mean()
            torch.testing.assert_close(result['balance_loss'], reference)
            grads = torch.autograd.grad(result['balance_loss'], m.router, retain_graph=True)[0]
            reference_grads = torch.autograd.grad(reference, m.router)[0]
            torch.testing.assert_close(grads, reference_grads, atol=1e-9, rtol=1e-5)

    def test_taylor_scores_and_update_invariant_to_microbatch_split(self):
        torch.manual_seed(48)
        c = replace(tiny(), study='E2', label_rate=1)
        model = Model(c)
        with torch.no_grad():
            model.path.normal_(std=.07)
            model.depth.normal_(std=.03)
        models = [model, copy.deepcopy(model)]
        inp = e2_batch([[1, 2, 0, 3, 4], [5, 0, 6, 7, 8],
                        [9, 10, 11, 0, 12], [13, 14, 15, 16, 17]], 'cpu')
        count = int((inp['labels'][:, 1:] != -100).sum())
        for m, slices in zip(models, [[slice(None)], [slice(0, 1), slice(1, 4)]]):
            m.gradient_factor = count
            for sl in slices:
                data = {k: v[sl] for k, v in inp.items()}
                out = m(**data)
                loss = out['nll_sum'] / count + out['balance_loss'] * len(data['tokens']) / 4
                loss.backward()
        for s in range(c.middle):
            records = [[r for r in m.labels if r['s'] == s] for m in models]
            for key in ['ids', 'c', 'q']:
                a, b = [torch.cat([r[key] for r in rs]) for rs in records]
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
            a, b = [torch.cat([r['c'] for r in rs]) for rs in records]
            self.assertTrue(torch.equal(a.argmax(-1), b.argmax(-1)))
        for (_, a), (_, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()

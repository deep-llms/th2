import copy
import unittest
import torch
from efficient_llm.warm_order.model import Model, Config
from efficient_llm.warm_order.multi_exit import objective, block_tokens


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.model = Model(Config(d=8, heads=1, kv_heads=1, head_dim=8,
                                  ff=16, vocab=19, length=8, prefix=2))
        self.tokens = torch.randint(19, (3, 8))

    def test_final_only_gradient_parity(self):
        for path in ([0, 1, 2, 3], [3, 2, 1, 0]):
            a = copy.deepcopy(self.model)
            b = copy.deepcopy(self.model)
            loss, _, _ = objective(a, self.tokens, path, 0.)
            reference = b.nll(self.tokens, path).mean()
            torch.testing.assert_close(loss, reference, atol=0, rtol=0)
            loss.backward()
            reference.backward()
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa.grad, pb.grad, atol=2e-7, rtol=2e-6)
        self.assertEqual(block_tokens(self.tokens), 18 * 3 * 8)

    def test_independent_exit_gradients_and_causality(self):
        a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
        path = [2, 3, 0, 1]
        loss, _, _ = objective(a, self.tokens, path, .2)
        loss.backward()
        # Accumulate each shared-module exit separately, independently of stack/mean.
        for depth in range(1, 5):
            coefficient = 1. if depth == 4 else .2 / 3
            (coefficient * b.nll(self.tokens, path[:depth]).mean()).backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-7, rtol=2e-6)
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 19
        with torch.no_grad():
            for depth in range(1, 5):
                torch.testing.assert_close(a.hidden(self.tokens, path[:depth])[:, :4],
                                           a.hidden(changed, path[:depth])[:, :4], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            objective(a, self.tokens, [0, 0, 1, 2], .2)


if __name__ == '__main__':
    unittest.main()

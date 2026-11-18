import copy
import unittest
import torch

from efficient_llm.warm_order.local_policy import (
    LOCAL_ORDERS, LocalRouter, policy_probabilities, sample_actions,
    policy_gradient_loss, full_information_loss, routed_nll,
)
from efficient_llm.warm_order.model import Config, Model


class LocalPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(917)

    def test_enumerated_pg_gradient_matches_expected_cost(self):
        # Exact expectation over all actions, not a noisy Monte Carlo check.
        costs = torch.tensor([[3.1, 2.4, 3.5, 3.0], [4.1, 4.8, 4.0, 3.8]], dtype=torch.float64)
        for epsilon in (0., .1, .4, 1.):
            logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
            probs = policy_probabilities(logits, epsilon)
            exact = torch.autograd.grad((probs * costs).sum(-1).mean(), logits, retain_graph=True)[0]
            expected = torch.zeros_like(logits)
            for action in range(4):
                # Weight each sample by its sampling probability; baseline may
                # vary by context but not by the chosen action.
                for row in range(2):
                    loss = policy_gradient_loss(logits[row:row+1], torch.tensor([action]),
                                                costs[row:row+1, action], costs[row:row+1, 0], epsilon)
                    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
                    expected += probs[row, action].detach() * grad / 2
            torch.testing.assert_close(expected, exact, atol=1e-12, rtol=1e-12)

    def test_rng_restore_reproduces_actual_mixture_draw(self):
        logits = torch.tensor([[5., 0., -1., 1.]]).repeat(100, 1)
        p = policy_probabilities(logits, .3)
        gen = torch.Generator().manual_seed(4)
        saved = gen.get_state()
        a = sample_actions(p, gen)
        gen.set_state(saved)
        self.assertTrue(torch.equal(a, sample_actions(p, gen)))
        self.assertGreater(int((a != 0).sum()), 0)

    def test_costs_and_features_are_detached(self):
        router = LocalRouter(8)
        features = torch.randn(3, 8, requires_grad=True)
        costs = torch.randn(3, 4, requires_grad=True)
        loss = full_information_loss(router(features), costs)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(costs.grad)
        self.assertGreater(float(router.net[-1].bias.grad.abs().sum()), 0)
        logits = torch.randn(3, 4, requires_grad=True)
        policy_gradient_loss(logits, torch.tensor([1, 2, 3]), costs[:, 1], costs[:, 0], .1).backward()
        self.assertIsNone(costs.grad)

    def test_grouped_body_gradients_and_prefix_causality(self):
        model = Model(Config(d=16, heads=2, kv_heads=1, head_dim=8, ff=32,
                             vocab=31, prefix=3, length=9))
        reference = copy.deepcopy(model)
        inputs = torch.randint(31, (5, 9))
        actions = torch.tensor([0, 1, 1, 2, 3])
        grouped = routed_nll(model, inputs, actions)
        single = torch.cat([reference.nll(inputs[i:i+1], LOCAL_ORDERS[a]) for i, a in enumerate(actions)])
        torch.testing.assert_close(grouped, single, atol=1e-6, rtol=1e-6)
        grouped.mean().backward(); single.mean().backward()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
        changed = inputs.clone(); changed[:, 3:] = torch.randint(31, (5, 6))
        router = LocalRouter(16)
        torch.testing.assert_close(router(model.features(inputs)), router(model.features(changed)), atol=0, rtol=0)

    def test_fail_closed_and_prior(self):
        router = LocalRouter(8)
        torch.testing.assert_close(router(torch.randn(2, 8)).softmax(-1),
                                   torch.tensor([[.85, .05, .05, .05]]).expand(2, -1))
        with self.assertRaises(ValueError):
            policy_probabilities(torch.zeros(2, 4), 1.1)
        with self.assertRaises(ValueError):
            full_information_loss(torch.zeros(2, 4), torch.zeros(2, 4), 0)


if __name__ == '__main__':
    unittest.main()

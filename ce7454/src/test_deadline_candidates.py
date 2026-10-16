"""Offline CPU checks for the optional candidate architectures."""
import copy
import json
import unittest

import torch
from torch.nn import functional as F

from deadline_candidates import MultiDepthStats, spatial_stats
from models import build_model, count_trainable


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_statistics_and_flat_gradient(self):
        for x in (torch.randn(2, 4, 7, 9), torch.ones(2, 4, 7, 9)):
            x.requires_grad_(True)
            mean = x.mean((-2, -1))
            var = (x - mean[..., None, None]).square().mean((-2, -1))
            reference = torch.cat((mean, (var + 1e-6).sqrt()), 1)
            torch.testing.assert_close(spatial_stats(x), reference)
            spatial_stats(x).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_extension_identity(self):
        torch.manual_seed(42)
        base = build_model('grip_nodown_ibna').eval()
        extended = MultiDepthStats(copy.deepcopy(base)).eval()
        x = torch.randn(2, 3, 33, 35)
        with torch.no_grad():
            torch.testing.assert_close(base(x), extended(x), rtol=1e-6, atol=1e-6)

    def test_sr_common_backbone_initialization(self):
        torch.manual_seed(42)
        mean = build_model('srresnet').state_dict()
        torch.manual_seed(42)
        stats = build_model('srresnet_stats').state_dict()
        for name, value in mean.items():
            if not name.startswith('fc.'):
                torch.testing.assert_close(value, stats[name], rtol=0, atol=0)

    def test_models_gradients_reload_and_resolution(self):
        rows = []
        for name in ('srresnet', 'srresnet_stats', 'srresnet_wide_stats', 'grip_nodown_stats', 'grip_nodown_ibna_stats'):
            with self.subTest(name=name):
                torch.manual_seed(42)
                model = build_model(name)
                self.assertLessEqual(count_trainable(model), 12300000)
                x = torch.randn(2, 3, 32, 34)
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x), torch.tensor([0, 2]))
                    loss.backward()
                    for n, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, n)
                        self.assertTrue(torch.isfinite(p.grad).all(), n)
                    optimizer.step()
                model.eval()
                restored = build_model(name).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
                    # Full inference size; no dense head fixed to the training crop.
                    logits = model(torch.zeros(1, 3, 512, 512))
                self.assertEqual(tuple(logits.shape), (1, 3))
                self.assertTrue(torch.isfinite(logits).all())
                rows.append({'model': name, 'parameters': count_trainable(model), 'loss': loss.item()})
        print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    unittest.main()

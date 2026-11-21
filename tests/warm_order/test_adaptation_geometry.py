import unittest
import torch

from efficient_llm.warm_order.adaptation_geometry import GeometryMoments, aligned_observations
from efficient_llm.warm_order.model import Model, Config


class AdaptationGeometry(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(125)

    def test_streaming_matches_centered_matrix_and_raw_metrics(self):
        x = torch.randn(17, 5, dtype=torch.float64)+1000
        y = x*.8 + torch.randn_like(x)*.2
        a, b = GeometryMoments(5), GeometryMoments(5)
        a.update(x, y)
        for start, stop in [(0, 1), (1, 7), (7, 17)]: b.update(x[start:stop], y[start:stop])
        r, s = a.report(), b.report()
        xc, yc = x-x.mean(0), y-y.mean(0)
        cka = float((xc.T@yc).square().sum()/((xc.T@xc).norm()*(yc.T@yc).norm()))
        self.assertAlmostEqual(r['centered_linear_cka'], cka, places=12)
        self.assertAlmostEqual(s['centered_linear_cka'], cka, places=11)
        self.assertAlmostEqual(r['mean_squared_distance'], float((x-y).square().sum(1).mean()), places=10)
        self.assertAlmostEqual(s['mean_squared_distance'], r['mean_squared_distance'], places=10)
        torch.testing.assert_close(a.xx, b.xx, atol=1e-10, rtol=1e-10)

    def test_cka_invariance_is_not_coordinate_identity_and_degenerate_is_explicit(self):
        x = torch.randn(15, 4, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
        m = GeometryMoments(4); m.update(x, 3*x@q+2)
        r = m.report()
        self.assertAlmostEqual(r['centered_linear_cka'], 1., places=12)
        self.assertGreater(r['mean_squared_distance'], 0.)
        m = GeometryMoments(4); m.update(torch.zeros(2, 4), torch.zeros(2, 4))
        self.assertIsNone(m.report()['centered_linear_cka'])
        self.assertIsNone(m.report()['mean_cosine'])

    def test_alignment_labels_fixed_positions_and_no_mutation(self):
        m = Model(Config(d=8, heads=2, kv_heads=1, head_dim=4, ff=16,
                         vocab=19, prefix=2, length=8)).eval().requires_grad_(False)
        tokens = torch.randint(0, 19, (2, 8)); rng = torch.get_rng_state().clone()
        samples = aligned_observations(m, m, tokens, (1, 6))
        self.assertEqual(samples['slot/2/input']['left_block'], 'middle.1')
        self.assertEqual(samples['slot/2/input']['right_block'], 'middle.2')
        self.assertEqual(samples['block/2/input']['right_block'], 'middle.1')
        self.assertEqual(samples['block/2/input']['right_slot'], 3)
        self.assertEqual(samples['slot/2/input']['x'].shape, (4, 8))
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValueError): aligned_observations(m, m, tokens, (1, 1))
        with self.assertRaises(ValueError): GeometryMoments(8).update(torch.ones(2, 8), torch.full((2, 8), float('nan')))


if __name__ == '__main__': unittest.main()

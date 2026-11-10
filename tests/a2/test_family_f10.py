import unittest
import numpy as np
from eval.a2_family_f10 import holm, test_statistic


class F10Tests(unittest.TestCase):
    def test_holm_unsorted_and_monotonic(self):
        np.testing.assert_allclose(holm([.04, .001, .02, .5]), [.08, .004, .06, .5])
        np.testing.assert_allclose(holm([.03, .031, .032]), [.09, .09, .09])

    def test_degenerate_and_nonfinite(self):
        self.assertEqual(test_statistic(0., np.zeros(10))['p_normal'], 1.)
        self.assertEqual(test_statistic(1., np.ones(10))['p_normal'], 0.)
        with self.assertRaises(ValueError): holm([float('nan')])

    def test_sign_symmetric_p_values(self):
        x=np.array([.2,.4,.5,.7,.9])
        a=test_statistic(.5,x); b=test_statistic(-.5,-x)
        self.assertEqual(a['p_normal'],b['p_normal'])
        self.assertEqual(a['p_bootstrap_two_sided'],b['p_bootstrap_two_sided'])
        self.assertAlmostEqual(a['bootstrap_SE_pp'],np.std(x,ddof=1))

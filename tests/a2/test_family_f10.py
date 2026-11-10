import unittest
import numpy as np
from eval.a2_family_f10 import holm, test_statistic
from eval.a2_family_f10_tasks import answer_metrics, likelihood_metrics


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

    def test_alias_normalization_and_f1(self):
        self.assertEqual(answer_metrics('The New_York!', ['NYC','New York']),dict(exact_match=1.,token_f1=1.))
        self.assertEqual(answer_metrics('red dog',['a red cat'])['token_f1'],.5)
        self.assertEqual(answer_metrics('',['Paris'])['token_f1'],0.)

    def test_conf_is_not_choice_probability(self):
        rows=[dict(doc_id='0',requests=[0,1],character_lengths=[2,10],gold=1)]
        r=likelihood_metrics(rows,[[-1.,True],[-2.,False]],'mmlu')[0]
        self.assertEqual(r['conf'],-.2)
        self.assertEqual(r['correctness'],1.)
        self.assertEqual(r['acc'],0.)
        self.assertGreater(r['p_correct'],.5)

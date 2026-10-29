import unittest
import numpy as np
from eval.a2_stage1_stats import MODELS, paired_accuracy_intervals

class Stage1StatsTest(unittest.TestCase):
    def fixture(self):
        return {t:{m:[0,1,0,1] for m in MODELS}
                for t in [*[f'mc{i}' for i in range(8)],'lambada_openai']}

    def test_pairing_cancels_identical_models_and_is_deterministic(self):
        tasks=self.fixture()
        a=paired_accuracy_intervals(tasks)
        self.assertEqual(a,paired_accuracy_intervals(dict(reversed(list(tasks.items())))))
        for key in ('mc_mean','lambada_accuracy'):
            for value in a[key].values():self.assertEqual(value,dict(estimate=0.,ci95=[0.,0.]))

    def test_direction_and_equal_task_weight(self):
        tasks=self.fixture()
        for t,r in tasks.items():
            n=100 if t=='mc0' else 4
            for m in MODELS:r[m]=[1]*n
            r['P1T-student']=[0]*n
        a=paired_accuracy_intervals(tasks)
        self.assertEqual(a['mc_mean']['P1T_drop'],dict(estimate=1.,ci95=[1.,1.]))
        self.assertEqual(a['mc_mean']['P1T_minus_P300_drop']['estimate'],1.)
        tasks['mc0']['P300-student']=[0]*100
        b=paired_accuracy_intervals(tasks)
        self.assertEqual(b['mc_mean']['P300_drop']['estimate'],.125)
        self.assertEqual(b['mc_mean']['X72_minus_P300_accuracy']['estimate'],.125)

    def test_empirical_pair_bootstrap_has_expected_variance(self):
        tasks=self.fixture()
        for r in tasks.values():
            for m in MODELS:r[m]=[1]*100
            r['P300-student']=[0]*50+[1]*50
        a=paired_accuracy_intervals(tasks)
        interval=a['lambada_accuracy']['P300_drop']['ci95']
        self.assertAlmostEqual(interval[0],.40,delta=.02)
        self.assertAlmostEqual(interval[1],.60,delta=.02)
        self.assertEqual(a['mc_mean']['P300_drop']['estimate'],.5)
        self.assertLess(a['mc_mean']['P300_drop']['ci95'][1]-.5,.05)

    def test_rejects_missing_nonbinary_and_unpaired_lengths(self):
        for defect in ('model','nan','length','task'):
            t=self.fixture()
            if defect=='model':del t['mc0'][MODELS[0]]
            elif defect=='nan':t['mc0'][MODELS[0]][0]=np.nan
            elif defect=='length':t['mc0'][MODELS[0]].append(1)
            else:del t['mc0']
            with self.assertRaises(ValueError):paired_accuracy_intervals(t)

if __name__=='__main__':unittest.main()

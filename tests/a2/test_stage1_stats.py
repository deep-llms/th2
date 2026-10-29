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


class Stage1ReplayPairingTest(unittest.TestCase):
    def fixture(self):
        from eval.downstream_suite import COUNTS, PRIMARY
        import copy
        protocol=dict(tasks=list(COUNTS),num_fewshot=0,seed=42,max_length=4096,limit=None)
        base=dict(a2_metadata=dict(protocol=protocol,prepared_sha256='prepared',request_sha256='requests'),
                  samples={},results={},**{'n-shot':{},'n-samples':{}})
        for task,n in COUNTS.items():
            key=PRIMARY[task]
            base['samples'][task]=[dict(doc_id=i,doc={'id':i},arguments=['same'],target=0,
                                     **{key.split(',')[0]:i%2}) for i in range(n)]
            base['results'][task]={key:sum(i%2 for i in range(n))/n}
            base['n-shot'][task]=0;base['n-samples'][task]=dict(original=n,effective=n)
        return {name:copy.deepcopy(base) for name in MODELS}

    def test_aligns_by_id_and_rejects_prompt_score_and_protocol_tampering(self):
        from eval.a2_stage1_stats import aligned_accuracy_arrays
        r=self.fixture();r[MODELS[1]]['samples']['hellaswag'].reverse()
        arrays=aligned_accuracy_arrays(r)
        np.testing.assert_array_equal(arrays['hellaswag'][MODELS[0]],arrays['hellaswag'][MODELS[1]])
        row=r[MODELS[1]]['samples']['hellaswag'][0]
        row['target']=1
        with self.assertRaisesRegex(ValueError,'pairing'):aligned_accuracy_arrays(r)
        row['target']=0;row['acc_norm']=1-row['acc_norm']
        with self.assertRaisesRegex(ValueError,'aggregate'):aligned_accuracy_arrays(r)
        row['acc_norm']=1-row['acc_norm'];r[MODELS[1]]['a2_metadata']['protocol']['seed']=43
        with self.assertRaisesRegex(ValueError,'protocol'):aligned_accuracy_arrays(r)

if __name__=='__main__':unittest.main()

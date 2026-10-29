import copy
import unittest
from eval.a2_stage1_stats import paired_accuracy_intervals, aligned_accuracy_arrays
from eval.a2_stage1b_stats import PART_MODELS, PART_CONTRASTS, paired_report
from tests.a2 import test_stage1_stats as original


class Stage1bStatsTests(unittest.TestCase):
    def test_registered_four_and_two_model_contrasts(self):
        for part in ('part1','part2'):
            models=PART_MODELS[part]
            tasks={task:{m:[0,1,0,1] for m in models}
                   for task in [*[f'mc{i}' for i in range(8)],'lambada_openai']}
            report=paired_accuracy_intervals(tasks,models=models,contrasts=PART_CONTRASTS[part])
            for value in report['mc_mean'].values():
                self.assertEqual(value,dict(estimate=0.,ci95=[0.,0.]))
            student='P1T-student' if part=='part1' else 'X72-student'
            for records in tasks.values():
                for name in models:records[name]=[1]*4
                records[student]=[0]*4
            report=paired_accuracy_intervals(tasks,models=models,contrasts=PART_CONTRASTS[part])
            name='P1T_minus_P300_drop' if part=='part1' else 'X72_drop_from_S72'
            self.assertEqual(report['lambada_accuracy'][name],dict(estimate=1.,ci95=[1.,1.]))

    def test_subset_replay_pairing_and_final_original_equivalence(self):
        replays=original.Stage1ReplayPairingTest().fixture()
        subset={m:replays[m] for m in PART_MODELS['part1']}
        self.assertEqual(paired_report(subset,'part1')['models'],list(PART_MODELS['part1']))
        bad=copy.deepcopy(subset);bad['P1T-student']['samples']['hellaswag'][0]['target']=9
        with self.assertRaisesRegex(ValueError,'pairing'):paired_report(bad,'part1')
        with self.assertRaises(ValueError):paired_report(replays,'part1')
        final=paired_report(replays,'final')
        expected=paired_accuracy_intervals(aligned_accuracy_arrays(replays))
        for k,v in expected.items():self.assertEqual(final[k],v)

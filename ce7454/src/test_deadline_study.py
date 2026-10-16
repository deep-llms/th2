import math
from pathlib import Path
import unittest

from deadline_study import ARMS, select_budget, training_command


class StudyTests(unittest.TestCase):
    def test_scratch_then_own_checkpoint_only(self):
        for arm in ARMS:
            base = Path('/fresh') / (arm + '_base')
            cmd = training_command(arm, Path('/cache'), base, 450)
            self.assertNotIn('--init-from', cmd)
            self.assertNotIn('--kd-teacher', cmd)
            self.assertEqual(cmd[cmd.index('--clip-grad') + 1], '0')
            self.assertEqual(cmd[cmd.index('--data-seed') + 1], '42')
            final = training_command(arm, Path('/cache'), Path('/fresh/final'), 40,
                                     final=True, parent=base / 'last.pt')
            self.assertEqual(final[final.index('--init-from') + 1], str(base / 'last.pt'))
            self.assertEqual(final[final.index('--batch-size') + 1], '16')
        with self.assertRaises(ValueError):
            training_command(ARMS[0], Path('/cache'), Path('/output'), 40, final=True)

    def test_common_budget_and_deadline(self):
        reports = {arm: {crop: {'mean_end_to_end_step_seconds_after_warmup': .1,
                               'full512_eval_batch32_seconds': .05}
                         for crop in (256, 512)} for arm in ARMS}
        long = select_budget(reports, 100000)
        self.assertEqual(long['base_epochs'], 450)
        medium = select_budget(reports, long['estimated_seconds_with_margin'] - 1)
        self.assertEqual(medium['base_epochs'], 300)
        with self.assertRaises(RuntimeError):
            select_budget(reports, 100)
        reports[ARMS[0]][256]['mean_end_to_end_step_seconds_after_warmup'] = math.nan
        with self.assertRaises(ValueError):
            select_budget(reports, 100000)


if __name__ == '__main__':
    unittest.main()

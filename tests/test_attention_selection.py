"""Selection uses complete eligible results and honors the requested tie preference."""
import unittest
from scripts.compare_attention_envs import choose


def candidate(name, native, routed, eligible=True):
    return dict(name=name,eligible=eligible,benchmarks=[
        dict(model=model,seconds_per_update=t) for model,values in
        [('native',native),('routed',routed)] for t in values])


class SelectionTests(unittest.TestCase):
    def test_ties_keep_fa4(self):
        self.assertEqual(choose([candidate('fa4',[1,1.01],[2,2.01]),
            candidate('stock',[.98,.99],[2,2.01]), candidate('fixed',[.97,.98],[2,2])]),'fa4')

    def test_clear_gain_selects_stock_and_failure_is_excluded(self):
        self.assertEqual(choose([candidate('fa4',[1,1.01],[2,2.01]),
            candidate('stock',[.90,.91],[2,2.01]),
            candidate('fixed',[.7,.7],[1,1],False)]),'stock')

    def test_routed_regression_or_overlapping_repeats_cannot_win(self):
        self.assertEqual(choose([candidate('fa4',[1,1.01],[2,2]),
            candidate('stock',[.90,.91],[2.2,2.2]),
            candidate('fixed',[.8,1.02],[2,2])]),'fa4')

    def test_no_valid_candidate_fails(self):
        with self.assertRaises(RuntimeError): choose([candidate('fa4',[],[],False)])


if __name__=='__main__':unittest.main()

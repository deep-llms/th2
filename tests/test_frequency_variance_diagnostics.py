"""CPU A3 aggregation/bin tests and mock-only five-GPU dispatch checks."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import frequency_variance_diagnostics as a3


class GridTests(unittest.TestCase):
    def test_quintiles_match_coarsened_old_deciles_and_preserve_ties(self):
        from ccm.evaluation import bins
        values = np.array([0]*20+[1]*12+[2]*30+list(range(3, 41)))
        q, edges = a3.quintiles(values)
        np.testing.assert_array_equal(q, bins(values)//2)
        for v in np.unique(values):
            self.assertEqual(len(set(q[values==v])), 1)
        self.assertEqual(len(edges), 4)
        with self.assertRaises(ValueError):
            a3.quintiles([0, np.nan])

    def test_add_row_duplicate_hits_and_masked_misses(self):
        sums, counts = np.zeros(4), np.zeros(4, dtype=np.int64)
        cells = np.array([0, 7, 7, 24])
        slots = np.array([-1, 0, 1, 1, 2, 3, 2])
        loss = np.array([np.nan, 1., 2., 3., 4., 5., np.nan])
        hit = np.array([False, True, True, True, True, True, False])
        s, n = a3.add_row(sums, counts, cells, slots, loss, hit)
        np.testing.assert_array_equal(sums, [1, 5, 4, 5])
        np.testing.assert_array_equal(counts, [1, 2, 1, 1])
        self.assertEqual(s[7], 9)
        self.assertEqual(n[7], 3)
        self.assertEqual(n.sum(), 5)
        self.assertEqual(s.sum(), 15)
        self.assertIsNone(a3.means(s, n)[1])
        self.assertEqual(a3.means(s, n)[7], 3)
        # No-hit row must not touch the dummy/final vocabulary slot.
        before=sums.copy()
        ss, nn=a3.add_row(sums, counts, cells, np.array([-1]), np.array([np.nan]), np.array([False]))
        np.testing.assert_array_equal(sums,before)
        self.assertEqual(ss.sum()+nn.sum(),0)

    def test_joint_cell_ids_and_marginals(self):
        f = np.repeat(np.arange(5),5); v = np.tile(np.arange(5),5)
        cell=f*5+v
        np.testing.assert_array_equal(cell,np.arange(25))
        s,n=a3.add_row(np.zeros(25),np.zeros(25,dtype=np.int64),cell,np.arange(25),
                       np.arange(1,26,dtype=float),np.ones(25,dtype=bool))
        np.testing.assert_array_equal(s.reshape(5,5).sum(1),[15,40,65,90,115])
        np.testing.assert_array_equal(s.reshape(5,5).sum(0),[55,60,65,70,75])
        self.assertEqual(n.sum(),25)

    def test_five_workers_reaped_on_failure_and_correct_gpu_mapping(self):
        m=importlib.import_module('pilot_a3_diagnostics')
        children=[Mock() for _ in range(5)]
        for c,code in zip(children,[1,0,0,0,0]):c.wait.return_value=code
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'OUT',Path(tmp)), \
                patch.object(m,'require_free'), patch.object(m.subprocess,'Popen',side_effect=children) as launch:
            with self.assertRaises(RuntimeError):m.panel(False)
            self.assertEqual([c.kwargs['env']['CUDA_VISIBLE_DEVICES'] for c in launch.call_args_list],list('01234'))
            for c in children:
                c.wait.assert_called_once_with();c.kill.assert_not_called();c.terminate.assert_not_called()

    def test_midlaunch_failure_still_reaps_owned_children(self):
        m=importlib.import_module('pilot_a3_diagnostics'); child=Mock();child.wait.return_value=0
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'OUT',Path(tmp)),patch.object(m,'require_free'), \
             patch.object(m.subprocess,'Popen',side_effect=[child,OSError('synthetic')]):
            with self.assertRaises(OSError):m.panel(True)
            child.wait.assert_called_once_with()


if __name__=='__main__':unittest.main()

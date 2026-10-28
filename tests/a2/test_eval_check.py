import copy
import unittest
from unittest.mock import patch
from eval.a2_check import select_requests,check_model
from eval.a2_request_bridge import digest
from eval.a2_likelihood import score_requests
from test_model import tiny_model


class EvalCheckTests(unittest.TestCase):
    def rows(self):
        return [dict(context=[2,3],continuation=[4,5]),dict(context=[],continuation=[6]),
                dict(context=[2]*40,continuation=[3,4,5]),dict(context=[0],continuation=[4]*16)]

    def test_manual_label_mask_and_batching_with_truncation(self):
        result=check_model(tiny_model(),self.rows(),max_length=16)
        self.assertTrue(result['passed']);self.assertEqual(result['truncated_requests'],1)
        self.assertLess(result['manual_max_nats_per_token'],2e-5)

    def test_gate_rejects_wrong_likelihood(self):
        real=score_requests
        def wrong(*args,**kwargs):
            result=real(*args,**kwargs);result['scores'][0]=(result['scores'][0][0]+.2,False)
            return result
        with patch('eval.a2_check.score_requests',side_effect=wrong):
            self.assertFalse(check_model(tiny_model(),self.rows(),max_length=16)['passed'])

    def test_fixed_selection_and_request_hash(self):
        rows=self.rows();prepared=dict(requests=rows,request_sha256=digest(rows))
        indices,selected=select_requests(prepared)
        self.assertEqual(len(indices),len(set(indices)));self.assertEqual(selected,[rows[i] for i in indices])
        bad=copy.deepcopy(prepared);bad['requests'][0]['context'].append(5)
        with self.assertRaises(ValueError):select_requests(bad)


if __name__=='__main__':unittest.main()

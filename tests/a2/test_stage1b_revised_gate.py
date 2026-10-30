import unittest
from eval.a2_stage1b_check import gate_decision


class RevisedGateTests(unittest.TestCase):
    def test_old_failure_is_reported_and_accepted(self):
        r=gate_decision(0,.011164188385009766,0,1000)
        self.assertTrue(r['passed']);self.assertTrue(r['old_batch_limit_exceeded'])

    def test_inclusive_limits_and_token_denominator(self):
        self.assertTrue(gate_decision(2e-5,.1,1,1000)['passed'])
        for args in [(2.1e-5,0,0,1000),(0,.100001,0,1000),(0,0,2,1000)]:
            self.assertFalse(gate_decision(*args)['passed'])
        for args in [(float('nan'),0,0,1),(0,0,0,0),(0,0,2,1)]:
            with self.assertRaises(ValueError):gate_decision(*args)

    def test_batch_choice_uses_mean_and_lambada_not_each_mc_task(self):
        from eval.a2_stage1b_batch_compare import decide_batch
        from eval.downstream_suite import COUNTS
        a={k:.5 for k in COUNTS};b=dict(a)
        b['hellaswag']+=.004;b['piqa']-=.004
        self.assertTrue(decide_batch(a,b)['batch2_accepted'])
        b['lambada_openai']+=.001
        self.assertTrue(decide_batch(a,b)['batch2_accepted'])
        b['lambada_openai']+=.000001
        self.assertEqual(decide_batch(a,b)['production_batch_size'],1)

    def test_padding_pairs_reproduce_sorted_scoring_and_truncation(self):
        from eval.a2_stage1b_batch_compare import padding_pairs
        requests=[dict(context=[1]*n,continuation=[2]) for n in (3,7,5,7,5000)]
        p=padding_pairs(requests)
        self.assertEqual(p[4]['pair_indices'],[4,1])
        self.assertEqual(p[1]['request_padding_tokens'],4096-7)
        self.assertEqual(p[3]['pair_indices'],[3,2])
        self.assertTrue(p[3]['pair_has_padding'])
        self.assertFalse(p[0]['pair_has_padding'])

    def test_prediction_flips_are_not_correctness_flips(self):
        from eval.a2_stage1b_batch_compare import prediction_flips
        a=[dict(doc_id=0,doc={},arguments=['a','b','c'],target='c',filtered_resps=[[-1,False],[-2,False],[-3,False]])]
        b=[dict(a[0],filtered_resps=[[-2,False],[-1,False],[-3,False]])]
        r=prediction_flips(a,b,lambda d:['a','b','c'])
        self.assertEqual(r['flips'],1)  # Both wrong, different predictions.

    def test_actual_tiny_native_model_token_diagnostics(self):
        from eval.a2_stage1b_check import check_model
        from test_model import tiny_model
        rows=[dict(context=[2,3],continuation=[4,5]),dict(context=[],continuation=[6]),
              dict(context=[2]*40,continuation=[3,4,5]),dict(context=[0],continuation=[4]*16)]
        result=check_model(tiny_model(),rows,16)
        self.assertTrue(result['passed'])
        self.assertEqual(result['compared_tokens'],22)
        self.assertEqual(result['greedy_token_mismatches'],0)
        self.assertEqual([len(x) for x in result['batch1_token_predictions']],[2,1,3,16])

    def test_readonly_hook_preserves_production_scores(self):
        from eval.a2_stage1b_parent_score import capture_scores
        from eval.a2_likelihood import score_requests
        from test_model import tiny_model
        model=tiny_model()
        requests=[dict(context=[2]*n,continuation=[3,4]) for n in (2,8,3,40,1)]
        for batch in (1,2):
            expected=score_requests(model,requests,batch_size=batch,max_length=16)
            actual,predictions=capture_scores(model,requests,batch_size=batch,length=16)
            self.assertEqual(expected,actual)
            self.assertEqual(len(predictions),len(requests))
            for r,p,s in zip(requests,predictions,actual['scores']):
                self.assertEqual(p==r['continuation'],s[1])
            self.assertFalse(model._forward_hooks)

    def test_lambada_two_wrong_predictions_still_flip(self):
        from eval.a2_stage1b_batch_compare import lambada_token_prediction_flips
        from eval.a2_request_bridge import digest
        prepared=dict(requests=[dict(text=['context','target'])])
        a=dict(batch_size=1,prepared_sha256=digest(prepared),request_sha256='r',
               checkpoint_identity_sha256='c',predictions=[[7]])
        b=dict(a,batch_size=2,predictions=[[8]])
        rows=[dict(doc_id=1,arguments=[['context','target']])]
        self.assertEqual(lambada_token_prediction_flips(rows,prepared,a,b)['flips'],1)

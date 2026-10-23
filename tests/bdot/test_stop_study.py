import unittest
import torch
from datasets import Dataset
from test_stopping import stopped
from eval.stop_study import fit_head, score, random_probability


class StopStudyTests(unittest.TestCase):
    def test_head_fit_changes_only_head_and_policy_scores_real_mixed_exits(self):
        m=stopped(True,0.).eval()
        rows=[dict(input_ids=[1,2,3,4,5,6],labels=[1,2,3,4,5,6],
                   attention_mask=[1]*6,segments=[0,0,0,1,1,1]) for _ in range(4)]
        data=Dataset.from_list(rows)
        before={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('stop_controller.')}
        result=fit_head(m,data,'sdpa',size=2,samples_per_row=4,epochs=3)
        self.assertEqual(result['examples'],32)
        self.assertLessEqual(result['bce_by_epoch'][-1],result['bce_by_epoch'][0])
        for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0,msg=k)
        self.assertFalse(any(p.requires_grad for p in m.parameters()))
        full=score(m,data,'sdpa',dict(policy='none'),size=2)
        early=score(m,data,'sdpa',dict(policy='learned',threshold=0.),size=2)
        self.assertEqual(full['num_tokens'],16)
        self.assertEqual(full['mean_middle_steps'],4)
        self.assertEqual(early['mean_middle_steps'],2)
        self.assertEqual(early['num_tokens'],full['num_tokens'])

    def test_constant_probability_matches_expected_compute(self):
        for mean in (2,3,5,7,8):
            p=random_probability(mean)
            self.assertAlmostEqual(2+sum((1-p)**i for i in range(1,7)),mean,places=10)


if __name__=='__main__':unittest.main()

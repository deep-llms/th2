import copy
import unittest

from test_attention_shift_eval import ShiftEvaluation
from efficient_llm.warm_order.attention_shift_node import prepare_seed, run_prepared
from efficient_llm.warm_order.attention_shift_train import admission_check, source_hashes, read
from efficient_llm.warm_order.adaptation_runtime import training_policy
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.data import sha


class ShiftNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ShiftEvaluation.setUpClass();cls.f=ShiftEvaluation
    @classmethod
    def tearDownClass(cls):ShiftEvaluation.tearDownClass()

    def test_prepared_parity_resume_terminal_ten_endpoints(self):
        f=self.f;root=f.f.root/'prepared-shift';proof=prepare_seed(f.configs,f.reference,root)
        self.assertEqual(proof['train_updates'],0);self.assertEqual(proof['reference_cells'],8)
        self.assertEqual(run_prepared(root,proof['config_sha256'],device='cpu')['cases'],10)
        for b in ('C','S'):
            gate=read(root/'work'/b/'gate.json');self.assertTrue(gate['exact_resume'])
            self.assertEqual(read(root/'work'/b/'fit/complete.json')['step'],4)
        with self.assertRaisesRegex(ValueError,'Existing work'):
            run_prepared(root,proof['config_sha256'],device='cpu')
        with self.assertRaisesRegex(ValueError,'Fixtures'):
            run_prepared(root,proof['config_sha256'],device='cuda')

    def test_admission_binds_config_source_policy_and_step(self):
        c=copy.deepcopy(self.f.c);c['fixture_only']=False
        p=self.f.f.root/'admission.json'
        r=dict(config=c,source=source_hashes(),pass_=True,policy=training_policy('cuda'),fixture_only=False,
               kind='numerical')
        def put():
            write_json(p,r);return dict(path=str(p),sha256=sha(p))
        admission_check(c,put(),40,True)
        with self.assertRaisesRegex(ValueError,'bounded'):admission_check(c,put(),41,True)
        with self.assertRaisesRegex(ValueError,'requires'):admission_check(c,put(),None,False)
        r.update(kind='resume',step=40,exact_resume=True);admission_check(c,put(),None,False)
        r['config']=dict(c,branch='S')
        with self.assertRaisesRegex(ValueError,'source/config'):admission_check(c,put(),None,False)


if __name__=='__main__':unittest.main()

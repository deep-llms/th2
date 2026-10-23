import importlib.util
import tempfile
from pathlib import Path
import unittest
import torch
from test_model import model
from test_data_and_trainer import tokenizer_assets


class HarnessTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('lm_eval'),'optional lm-eval environment not installed')
    def test_likelihood_matches_native_qwen(self):
        from lm_eval.api.instance import Instance
        from lm_eval.models.huggingface import HFLM
        from eval.bdot_harness import make_harness
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);tok=tokenizer_assets(p)
            m=model(routing='fixed',curriculum='none',step_film=False).eval()
            ref=m.native_reference().eval();m.save_pretrained(p)
            custom=make_harness(p,device='cpu',batch_size=2,max_length=16)
            native=HFLM(pretrained=ref,tokenizer=tok,backend='causal',device='cpu',batch_size=2,max_length=16)
            requests=[Instance(request_type='loglikelihood',doc={},arguments=(ctx,cont),idx=i)
                      for i,(ctx,cont) in enumerate([('a b',' c d'),('a',' b'),('',' a b')])]
            actual=custom.loglikelihood(requests);expected=native.loglikelihood(requests)
            for (a,ag),(b,bg) in zip(actual,expected):
                self.assertAlmostEqual(a,b,places=6);self.assertEqual(ag,bg)

    @unittest.skipUnless(importlib.util.find_spec('lm_eval'),'optional lm-eval environment not installed')
    def test_routed_reload_padding_and_state(self):
        from eval.bdot_harness import make_harness
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);tokenizer_assets(p)
            m=model(routing='learned',curriculum='A',route_scope='mlp',routed_mlp_fraction=.25,
                    router_metric='cosine').eval()
            m.completed_steps.fill_(10000);m.balance_bias.fill_(.1);m.save_pretrained(p)
            h=make_harness(p,batch_size=2,max_length=16)
            ids=torch.tensor([[2,3,4,5],[6,7,8,9]])
            with torch.no_grad():expected=m(input_ids=ids,use_cache=False).logits
            torch.testing.assert_close(h._model_call(ids),expected,atol=1e-6,rtol=1e-6)
            padded=torch.cat((ids,torch.zeros(2,3,dtype=torch.long)),dim=1)
            torch.testing.assert_close(h._model_call(padded)[:,:4],expected,atol=1e-6,rtol=1e-6)
            for name in ('balance_bias','completed_steps','routing_counter','training_totals'):
                torch.testing.assert_close(getattr(h.model,name),getattr(m,name),atol=0,rtol=0)
            self.assertEqual(h.model.evaluation_policy,'target')
            with self.assertRaises(ValueError):h.generate_until([])

    def test_refuse_invalid_backend_and_context(self):
        from eval.bdot_harness import make_harness
        with self.assertRaises(ValueError):make_harness('/missing',attention_backend='fa4')
        with self.assertRaises(ValueError):make_harness('/missing',max_length=1)

if __name__=='__main__':unittest.main()

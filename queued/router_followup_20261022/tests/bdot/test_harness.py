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

if __name__=='__main__':unittest.main()

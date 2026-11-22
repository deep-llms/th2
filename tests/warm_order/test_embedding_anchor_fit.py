import unittest
import json
from pathlib import Path
import test_embedding_anchor_device_prepare as fixtures
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_qualify import qualify_seed
from efficient_llm.warm_order.embedding_anchor_fit import run_seed
from efficient_llm.warm_order.embedding_anchor_eval import replay
from efficient_llm.warm_order.embedding_anchor_fit_finish import validate_seed

class FitTests(unittest.TestCase):
    setUp=fixtures.PreparedDeviceTests.setUp
    tearDown=fixtures.PreparedDeviceTests.tearDown
    historical_fixture=fixtures.PreparedDeviceTests.historical_fixture

    def test_gated_scratch_to_paired_terminal_and_replay(self):
        root,h=self.historical_fixture();base=root.parent
        q=self.root/'qualification';qualify_seed(base,sha(base/'prepared.json'),0,q,device='cpu',fixture_steps=2)
        args=(self.path,sha(self.path),base,sha(base/'prepared.json'),q/'complete.json',sha(q/'complete.json'),0)
        result=run_seed(*args,self.root/'fit',device='cpu')
        assert result['registration']['physical_fit_updates']==3072
        assert set(result['runs'])=={'A','C'}
        plan=dict(prepared_sha256=args[3],pilot_plan_sha256=args[1],gates={'0':{'sha256':args[5]}},seed_configs={'0':h})
        _,states=validate_seed(self.root/'fit',plan,0);assert len(states)==10
        assert replay(self.root/'fit/evaluation')['rows']==512
        assert len(list((self.root/'fit').rglob('*.pt')))==10
        with self.assertRaises(ValueError):run_seed(*args,self.root/'fit',device='cpu')
        bad=list(args);bad[5]='0'*64
        with self.assertRaisesRegex(ValueError,'receipt'):run_seed(*bad,self.root/'bad',device='cpu')

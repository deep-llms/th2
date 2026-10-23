import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from test_model import model
from bdot.model import BDOTForCausalLM
from scripts.prepare_stop_adaptation import initialize


class StopAdaptationTests(unittest.TestCase):
    def test_exact_backbone_and_deliberate_reset(self):
        m=model(route_scope='mlp',routed_mlp_fraction=.25,router_metric='cosine',curriculum='A')
        m.completed_steps.fill_(10000);m.routing_counter.fill_(40000)
        m.balance_bias.copy_(torch.tensor([.3,-.2,.1,.05]))
        m.training_totals.fill_(123.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'parent';m.save_pretrained(cp)
            (cp/'trainer_state.json').write_text(json.dumps(dict(global_step=10000)))
            initialize(cp,root/'new',.05)
            n=BDOTForCausalLM.from_pretrained(root/'new',attn_implementation='sdpa')
            for name,value in m.state_dict().items():
                if name in ('completed_steps','routing_counter','training_totals','logged_training_totals'):
                    self.assertEqual(n.state_dict()[name].count_nonzero(),0)
                else:torch.testing.assert_close(value,n.state_dict()[name],atol=0,rtol=0,msg=name)
            self.assertEqual(n.options.phase(0.,True),(1,False,0.))
            self.assertTrue(all(not p.requires_grad for p in n.stop_controller.parameters()))
            self.assertEqual(n.stop_controller.options.train_random_prob,.05)


if __name__=='__main__':unittest.main()

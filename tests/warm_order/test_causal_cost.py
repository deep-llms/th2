import unittest
import tempfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.window_cf import PATH
from efficient_llm.warm_order.causal_cost import Policy, CostHead, first_edit, cached_policy, decisions
from efficient_llm.warm_order.causal_cost_fit import cache, fit, predict, select, constant
from efficient_llm.warm_order.data import Rows


class CausalCostTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(73)
        c = Config(d=8,heads=2,kv_heads=1,head_dim=4,ff=16,vocab=17,length=9,prefix=2)
        self.m = Policy(c,'WL',4,2,0).eval()
        with torch.no_grad():
            self.m.router.weight.normal_(std=.2)
            self.m.film_gamma.normal_(std=.1)
        self.x = torch.randint(17,(3,9))

    def test_first_edit_uses_earliest_and_not_future(self):
        p = torch.zeros(2,3,4); factual = torch.zeros(2,3,dtype=torch.long)
        p[:,0,1] = -.01; p[:,2,2] = -2
        chosen, mask = first_edit(p,factual,.002)
        self.assertEqual(chosen.tolist(),[[1,0,0],[1,0,0]])
        self.assertTrue(torch.all(mask.sum(1)==1))
        p[:,1:] = -100
        torch.testing.assert_close(first_edit(p,factual,.002)[0][:,0],chosen[:,0])

    def test_noedit_identity_causality_and_single_depth(self):
        m,x=self.m,self.x; original=m.hidden(x).detach(); m.cost_head=CostHead(8,12).eval()
        m.policy_depth=None
        torch.testing.assert_close(original,m.hidden(x),rtol=0,atol=0)
        for d in range(4):
            m.policy_depth=d; m.threshold=0
            full=m.hidden(x).detach()
            self.assertTrue(torch.all(torch.cat(m.edit_masks,1).sum(1)<=1))
            for z in range(4):
                if z!=d:self.assertFalse(m.edit_masks[z].any())
            for cut in (1,2,4,8):
                changed=x.clone();changed[:,cut:]=(changed[:,cut:]+3)%17
                torch.testing.assert_close(full[:,:cut],m.hidden(changed)[:,:cut],rtol=0,atol=0)

    def test_cached_selection_matches_actual_intervened_forward(self):
        m,x=self.m,self.x; layout=decisions(9,4,2)
        with torch.no_grad():
            original=m.nll(x,PATH); factual=torch.cat(m.last_routes,1); costs=[]
            for d,j,a,b in layout:
                for k in range(4):
                    m.intervention=(d,j,k,'hold');costs.append(m.nll(x,PATH))
            m.intervention=None;costs=torch.stack(costs,1)
            # Static predictor forces a nontrivial expert; independent forward
            # must match the corresponding enumerated single-edit cost exactly.
            predicted=torch.randn(len(layout),4)*.02;m.constant_cost=predicted
            for d in range(4):
                got,action=cached_policy(predicted[None].expand(3,-1,-1),factual,costs,original,d,0,layout)
                m.policy_depth=d;m.threshold=0
                torch.testing.assert_close(m.nll(x,PATH),got,rtol=0,atol=0)
                self.assertTrue(torch.all(torch.cat(m.edit_masks,1).sum(1)<=1))

    def test_cache_head_resume_and_selected_policy_actual_forward(self):
        m,x=self.m,self.x
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stem=str(root/'data');x.numpy().astype('<u4').tofile(stem+'.bin');np.zeros(x.shape,dtype='u1').tofile(stem+'.starts')
            before={k:v.detach().clone() for k,v in m.state_dict().items()}
            a=cache(m,Rows(stem,9),np.arange(3),'cpu');c=asdict(m.c)
            full,_=fit(a,c,4,2,19,root/'full',epochs=3,hidden=12,batch=7)
            fit(a,c,4,2,19,root/'resume',epochs=1,hidden=12,batch=7)
            resumed,_=fit(a,c,4,2,19,root/'resume',epochs=3,resume=True,hidden=12,batch=7)
            for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
            changed=dict(a);changed['nll']=a['nll']+.1
            with self.assertRaises(ValueError):fit(changed,c,4,2,19,root/'resume',epochs=4,resume=True,hidden=12,batch=7)
            for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
            for kind in ('learned','constant'):
                prediction=predict(full,a,9,4,2) if kind=='learned' else constant(a)[None].expand(len(x),-1,-1)
                selection=select(prediction,a,9,4,2)['selected'];m.policy_depth=selection['depth'];m.threshold=selection['threshold']
                m.cost_head=full if kind=='learned' else None;m.constant_cost=constant(a) if kind=='constant' else None
                with torch.no_grad():actual=m.nll(x,PATH)
                expected,action=cached_policy(prediction,torch.as_tensor(a['routes']).long(),torch.as_tensor(a['costs']),torch.as_tensor(a['nll']),m.policy_depth,m.threshold,decisions(9,4,2))
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                self.assertAlmostEqual(float(actual.double().mean()),selection['nll'],places=9)


if __name__=='__main__':unittest.main()

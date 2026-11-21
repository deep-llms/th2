import copy
import unittest
import torch
from torch.nn import functional as F
from efficient_llm.warm_order.model import Model,Config
from efficient_llm.warm_order.adaptation_sublayer import sublayer_crossovers,score_sublayer,sublayer_effects,partition
from efficient_llm.warm_order.adaptation_context import context_crossovers

class SublayerCrossovers(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(8201)
        self.p=Model(Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=24,vocab=31,prefix=3,length=12)).eval().requires_grad_(False)
        self.a=copy.deepcopy(self.p)
        with torch.no_grad():
            for v in self.a.parameters():v.add_(torch.randn_like(v)*.003)
        self.x=torch.randint(0,31,(2,12))

    def test_independent_weight_grafts_native_context_and_nll(self):
        r=sublayer_crossovers(self.p,self.a,self.x);s=score_sublayer(self.p,self.a,self.x,r)
        groups=partition(self.p.middle[2]);states=[self.p.middle[2].state_dict(),self.a.middle[2].state_dict()]
        context=context_crossovers(self.p,self.a,self.x)
        for key,previous in [('00','000'),('11','010'),('S','0S0')]:
            torch.testing.assert_close(r['cells'][key],context['cells'][previous],atol=0,rtol=0)
        for key in ('00','01','10','11'):
            graft=copy.deepcopy(self.p)
            state={n:states[int(key[0 if n in groups['attention'] else 1])][n] for n in states[0]}
            graft.middle[2].load_state_dict(state)
            h=graft.hidden(self.x,(0,2,1,3))
            torch.testing.assert_close(h,r['cells'][key],atol=0,rtol=0)
            loss=F.cross_entropy(F.linear(h[:,2:-1],self.p.embed.weight).flatten(0,1),self.x[:,3:].flatten(),reduction='none').reshape(2,9)
            torch.testing.assert_close(s['cells'][key]['nll_sum'],loss.double().sum(1),atol=2e-6,rtol=1e-5)
            self.assertEqual(s['cells'][key]['token_count'].tolist(),[9,9])

    def test_fixed_parent_boundary_causality_and_immutability(self):
        state=[{k:v.clone() for k,v in m.state_dict().items()} for m in (self.p,self.a)];rng=torch.get_rng_state().clone()
        r=sublayer_crossovers(self.p,self.a,self.x)
        x=self.x.clone();x[:,7:]=(x[:,7:]+1)%31;t=sublayer_crossovers(self.p,self.a,x)
        for k in r['cells']:torch.testing.assert_close(r['cells'][k][:,:7],t['cells'][k][:,:7],atol=0,rtol=0)
        for m,ref in zip((self.p,self.a),state):
            for k,v in m.state_dict().items():torch.testing.assert_close(v,ref[k],atol=0,rtol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
        other=copy.deepcopy(self.a)
        with torch.no_grad():
            for name,v in other.named_parameters():
                if not name.startswith('middle.2.'):v.add_(10)
        changed=sublayer_crossovers(self.p,other,self.x)
        for k in r['cells']:torch.testing.assert_close(r['cells'][k],changed['cells'][k],atol=0,rtol=0)

    def test_identity_partition_and_interaction(self):
        same=sublayer_crossovers(self.p,self.p,self.x)
        for k in ('01','10','11'):torch.testing.assert_close(same['cells'][k],same['cells']['00'],atol=0,rtol=0)
        with torch.no_grad():
            for m in (self.p,self.a):
                m.middle[2].self_attn.o_proj.weight.zero_();m.middle[2].mlp.down_proj.weight.zero_()
        r=sublayer_crossovers(self.p,self.a,self.x)
        for v in r['cells'].values():torch.testing.assert_close(v,r['cells']['S'],atol=0,rtol=0)
        loss={k:torch.tensor([5.]) for k in ('00','01','10','11','S')};loss['11']=torch.tensor([3.])
        self.assertEqual(sublayer_effects(loss)['interaction'].item(),2.)
        self.a.middle[2].register_parameter('unexpected',torch.nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(ValueError,'partition'):partition(self.a.middle[2])

    def test_invalid_scope(self):
        for kwargs in ({'target':True},{'target':1},{'path':(0,1,2,3)}):
            with self.assertRaises(ValueError):sublayer_crossovers(self.p,self.a,self.x,**kwargs)
        self.a.train()
        with self.assertRaises(ValueError):sublayer_crossovers(self.p,self.a,self.x)
        with self.assertRaises(ValueError):sublayer_effects({'00':torch.tensor([1.])})

if __name__=='__main__':unittest.main()

import copy
import unittest
import torch
from torch.nn import functional as F
from efficient_llm.warm_order.model import Model,Config
from efficient_llm.warm_order.adaptation_factors import factors_crossovers,score_factors,factors_effects,partition,GROUPS,CELLS
from efficient_llm.warm_order.adaptation_sublayer import sublayer_crossovers

class FactorCrossovers(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(9101)
        self.p=Model(Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=24,vocab=31,prefix=3,length=12)).eval().requires_grad_(False)
        self.a=copy.deepcopy(self.p)
        with torch.no_grad():
            for v in self.a.parameters():v.add_(torch.randn_like(v)*.003)
        self.x=torch.randint(0,31,(2,12))

    def test_every_factor_graft_and_independent_nll(self):
        r=factors_crossovers(self.p,self.a,self.x);s=score_factors(self.p,self.a,self.x,r)
        groups=partition(self.p.middle[2]);states=[self.p.middle[2].state_dict(),self.a.middle[2].state_dict()]
        for key in CELLS:
            if key=='S':continue
            family,bits=key.split('/');graft=copy.deepcopy(self.p);state=copy.deepcopy(states[0])
            for bit,group in zip(bits,GROUPS[family]):
                for name in groups[family+'/'+group]:state[name]=states[int(bit)][name]
            graft.middle[2].load_state_dict(state)
            h=graft.hidden(self.x,(0,2,1,3))
            torch.testing.assert_close(h,r['cells'][key],atol=2e-6,rtol=1e-5)
            loss=F.cross_entropy(F.linear(h[:,2:-1],self.p.embed.weight).flatten(0,1),self.x[:,3:].flatten(),reduction='none').reshape(2,9)
            torch.testing.assert_close(s['cells'][key]['nll_sum'],loss.double().sum(1),atol=2e-6,rtol=1e-5)
        previous=sublayer_crossovers(self.p,self.a,self.x)
        for key,old in [('attn/000','00'),('ffn/0000','00'),('attn/111','10'),('ffn/1111','01'),('S','S')]:
            torch.testing.assert_close(r['cells'][key],previous['cells'][old],atol=2e-6,rtol=1e-5)

    def test_fixed_boundary_causality_no_mutation(self):
        states=[{k:v.clone() for k,v in m.state_dict().items()} for m in (self.p,self.a)];rng=torch.get_rng_state().clone()
        r=factors_crossovers(self.p,self.a,self.x);x=self.x.clone();x[:,7:]=(x[:,7:]+1)%31
        changed=factors_crossovers(self.p,self.a,x)
        for k in CELLS:torch.testing.assert_close(r['cells'][k][:,:7],changed['cells'][k][:,:7],atol=0,rtol=0)
        for m,state in zip((self.p,self.a),states):
            for k,v in m.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
        with torch.no_grad():
            for n,v in self.a.named_parameters():
                if not n.startswith('middle.2.'):v.add_(5.)
        changed=factors_crossovers(self.p,self.a,self.x)
        for k in CELLS:torch.testing.assert_close(r['cells'][k],changed['cells'][k],atol=0,rtol=0)

    def test_identical_partition_and_conditional_algebra(self):
        r=factors_crossovers(self.p,self.p,self.x)
        for k in CELLS:
            if k!='S':torch.testing.assert_close(r['cells'][k],r['cells']['attn/000'],atol=2e-6,rtol=1e-5)
        loss={k:torch.tensor([5.]) for k in CELLS};loss['attn/111']=torch.tensor([3.]);loss['ffn/1111']=torch.tensor([4.])
        effects=factors_effects(loss)
        self.assertEqual(effects['attn/N/other00'].item(),0)
        self.assertEqual(effects['attn/N/other11'].item(),2)
        self.assertEqual(effects['attn/nonadditivity'].item(),2)
        self.assertEqual(effects['ffn/nonadditivity'].item(),1)
        groups=partition(self.p.middle[2]);self.assertIn('self_attn.q_norm.weight',groups['attn/N'])
        self.assertNotIn('self_attn.q_norm.weight',groups['attn/P'])
        self.a.middle[2].register_parameter('unexpected',torch.nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(ValueError,'partition'):partition(self.a.middle[2])

    def test_invalid_scope(self):
        for kw in ({'target':True},{'target':1},{'path':(0,1,2,3)}):
            with self.assertRaises(ValueError):factors_crossovers(self.p,self.a,self.x,**kw)
        self.a.train()
        with self.assertRaises(ValueError):factors_crossovers(self.p,self.a,self.x)
        with self.assertRaises(ValueError):factors_effects({})

if __name__=='__main__':unittest.main()

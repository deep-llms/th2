import copy
import unittest

import torch
from torch.nn import functional as F

from efficient_llm.warm_order.model import Model, Config
from efficient_llm.warm_order.adaptation_analysis import SWAPPED
from efficient_llm.warm_order.adaptation_context import context_crossovers, score_context, context_effects


class ContextCrossovers(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7301)
        self.p = Model(Config(d=16, heads=2, kv_heads=1, head_dim=8,
                              ff=24, vocab=31, prefix=3, length=12)).eval().requires_grad_(False)
        self.a = copy.deepcopy(self.p)
        with torch.no_grad():
            for value in self.a.parameters():
                value.add_(torch.randn_like(value)*.003)
        self.x = torch.randint(0,31,(2,12))

    def test_native_endpoints_and_independent_hybrid_readout(self):
        result = context_crossovers(self.p,self.a,self.x)
        self.assertEqual(len(result['cells']),12)
        scored = score_context(self.p,self.a,self.x,result)
        models = (self.p,self.a)
        pos = torch.arange(12).expand_as(self.x);seg=torch.zeros_like(self.x)
        for key, hidden in result['cells'].items():
            i,b,r=key;prefix=models[int(i)];reader=models[int(r)]
            h=prefix.middle[0](prefix.stem(prefix.embed(self.x),pos,seg),pos,seg)
            if b!='S':h=models[int(b)].middle[2](h,pos,seg)
            for index in (1,3):h=reader.middle[index](h,pos,seg)
            h=reader.norm(reader.coda(h,pos,seg))
            torch.testing.assert_close(h,hidden,atol=0,rtol=0)
            nll=F.cross_entropy(F.linear(h[:,2:-1],reader.embed.weight).flatten(0,1),
                                self.x[:,3:].flatten(),reduction='none').reshape(2,9)
            torch.testing.assert_close(scored['cells'][key]['nll_sum'],nll.double().sum(1),atol=2e-6,rtol=1e-5)
            self.assertEqual(scored['cells'][key]['token_count'].tolist(),[9,9])
        for bit,model in enumerate(models):
            torch.testing.assert_close(result['cells'][str(bit)*3],model.hidden(self.x,SWAPPED),atol=0,rtol=0)
        self.assertGreater(float((result['cells']['010']-result['cells']['000']).abs().max()),0)
        effects=scored['effects_nll_sum']
        torch.testing.assert_close(effects['total'],effects['input/block0/reader0']+
                                   effects['block/input1/reader0']+effects['reader/input1/block1'],atol=1e-12,rtol=0)

    def test_future_causality_and_no_state_or_rng_mutation(self):
        states=[{k:v.clone() for k,v in m.state_dict().items()} for m in (self.p,self.a)]
        rng=torch.get_rng_state().clone();changed=self.x.clone();changed[:,7:]=(changed[:,7:]+1)%31
        a=context_crossovers(self.p,self.a,self.x);b=context_crossovers(self.p,self.a,changed)
        for key in a['cells']:
            torch.testing.assert_close(a['cells'][key][:,:7],b['cells'][key][:,:7],rtol=0,atol=0)
        for m,state in zip((self.p,self.a),states):
            for key,value in m.state_dict().items():torch.testing.assert_close(value,state[key],rtol=0,atol=0)
            self.assertTrue(all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules()))
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)

    def test_identity_block_skip_and_identical_models(self):
        for m in (self.p,self.a):
            for value in m.middle[2].parameters():value.zero_()
        result=context_crossovers(self.p,self.a,self.x)
        for i in (0,1):
            for r in (0,1):
                for b in (0,1):torch.testing.assert_close(result['cells'][f'{i}{b}{r}'],result['cells'][f'{i}S{r}'],rtol=0,atol=0)
        result=context_crossovers(self.p,self.p,self.x)
        for h in result['cells'].values():torch.testing.assert_close(h,result['cells']['000'],rtol=0,atol=0)

    def test_bad_context_and_nonadditive_effect(self):
        for kwargs in ({'target':True},{'path':(0,1,1,3)},{'target':4}):
            with self.assertRaises(ValueError):context_crossovers(self.p,self.a,self.x,**kwargs)
        self.a.train()
        with self.assertRaises(ValueError):context_crossovers(self.p,self.a,self.x)
        self.a.eval()
        loss={f'{i}{b}{r}':torch.tensor([5.],dtype=torch.float64)
              for i in (0,1) for b in (0,1,'S') for r in (0,1)}
        loss['110']=torch.tensor([3.],dtype=torch.float64)
        effects=context_effects(loss)
        self.assertEqual(effects['input/block0/reader0'].item(),0)
        self.assertEqual(effects['block/input0/reader0'].item(),0)
        self.assertEqual(effects['input_block_interaction/reader0'].item(),2)
        del loss['0S0']
        with self.assertRaises(ValueError):context_effects(loss)
        loss['0S0']=torch.tensor([5.],dtype=torch.float64);loss['000']=[5.]
        with self.assertRaises(ValueError):context_effects(loss)


if __name__=='__main__':unittest.main()

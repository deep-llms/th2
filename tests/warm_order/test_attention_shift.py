import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

import test_adaptation_train as fixture
from efficient_llm.warm_order.attention_shift import freeze, fingerprint, shifted, inject, PATHS
from efficient_llm.warm_order.attention_shift_train import train, selections, validate_optimizer, monitor
from efficient_llm.warm_order.adaptation_train import load_parent
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.model import Config, Model


def manual_hidden(model,x,bias,branch):
    pos=torch.arange(x.shape[1]).expand_as(x);seg=torch.zeros_like(x)
    h=model.stem(model.embed(x),pos,seg)
    for i in PATHS[branch]:
        block=model.middle[i]
        if i==2:
            u=block.self_attn(block.input_layernorm(h),pos,seg)+bias
            h=h+u;h=h+block.mlp(block.post_attention_layernorm(h))
        else:h=block(h,pos,seg)
    return model.norm(model.coda(h,pos,seg))


class AttentionShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AdaptationTrain.setUpClass();f=fixture.AdaptationTrain;cls.f=f
        np.save(f.root/'shift-cal.npy',np.array([196,197]))
        old=copy.deepcopy(f.config);old['calibration_indices']=str(f.root/'shift-cal.npy')
        old['expected_inputs'][old['calibration_indices']]=sha(old['calibration_indices'])
        p=f.root/'data-recipe.json';p.write_text(json.dumps(dict(config=old)))
        cls.c={k:old[k] for k in ('parent','parent_hashes','data','indices','calibration_indices','confirmation_indices','expected_inputs')}
        cls.c.update(schema=1,domain='dclm',target=2,branch='C',seed=0,lr=.001,lr_policy='constant',
                     betas=[.9,.95],weight_decay=0.,clip_grad_norm=1.,fixture_only=True,
                     steps=4,batch=2,threads=1,monitor_every=2,monitor_rows=2,snapshots=[0,1,4],
                     data_recipe=dict(path=str(p),sha256=sha(p)))

    @classmethod
    def tearDownClass(cls):fixture.AdaptationTrain.tearDownClass()

    def test_zero_identity_gradient_reference_and_causality(self):
        torch.manual_seed(8);m=freeze(Model(Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,prefix=2,length=8)))
        x=torch.randint(0,19,(2,8));before=fingerprint(m)
        for branch in PATHS:
            zero=torch.zeros(8,requires_grad=True)
            torch.testing.assert_close(shifted(m,x,zero,branch,hidden=True),m.hidden(x,PATHS[branch]),atol=0,rtol=0)
            bias=torch.nn.Parameter(torch.linspace(-.03,.04,8));h=shifted(m,x,bias,branch,hidden=True)
            independent=manual_hidden(m,x,bias,branch)
            torch.testing.assert_close(h,independent,atol=0,rtol=0)
            native_loss=shifted(m,x,bias,branch).mean()
            manual_loss=torch.nn.functional.cross_entropy(torch.nn.functional.linear(independent[:,1:-1],m.embed.weight).reshape(-1,19),x[:,2:].reshape(-1))
            g1=torch.autograd.grad(native_loss,bias)[0];g2=torch.autograd.grad(manual_loss,bias)[0]
            torch.testing.assert_close(g1,g2,atol=2e-7,rtol=1e-5);self.assertGreater(float(g1.norm()),0)
            changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%19
            torch.testing.assert_close(h[:,:5],shifted(m,changed,bias,branch,hidden=True)[:,:5],atol=0,rtol=0)
            self.assertFalse(m.middle[2].self_attn._forward_hooks)
        self.assertEqual(before,fingerprint(m));self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_hook_exception_cleanup_and_refuse_unfrozen_or_nested(self):
        m=Model(Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,prefix=2,length=8));b=torch.zeros(8)
        with self.assertRaisesRegex(ValueError,'frozen'):
            with inject(m,b):pass
        freeze(m)
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with inject(m,b):raise RuntimeError('sentinel')
        self.assertFalse(m.middle[2].self_attn._forward_hooks)
        with self.assertRaisesRegex(ValueError,'stacking'):
            with inject(m,b):
                with inject(m,b):pass
        self.assertFalse(m.middle[2].self_attn._forward_hooks)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            with inject(m,b):pass

    def test_exact_resume_both_paths_and_only_vector_changes(self):
        traces=[]
        for branch in PATHS:
            c=dict(self.c,branch=branch);whole=self.f.root/('shift-whole-'+branch);part=self.f.root/('shift-part-'+branch)
            m,b,o=train(c,whole);train(c,part,stop_after=2);train(c,part,resume=True)
            a=torch.load(whole/'fullstate.pt',weights_only=False);z=torch.load(part/'fullstate.pt',weights_only=False)
            fixture.same(a,z)
            self.assertNotIn('model',a);self.assertEqual(a['recipe']['parameter_names'],['attention_shift'])
            self.assertEqual(a['recipe']['trainable_parameters'],8);self.assertGreater(float(b.detach().norm()),0.)
            initial=torch.load(whole/'snapshots/000000/fullstate.pt',weights_only=False)
            self.assertEqual(float(initial['bias'].abs().sum()),0.);self.assertFalse(initial['optimizer']['state'])
            self.assertEqual(float(a['optimizer']['state'][0]['step']),4.)
            self.assertEqual((whole/'trace.jsonl').read_text(),(part/'trace.jsonl').read_text())
            self.assertEqual((whole/'validation.jsonl').read_text(),(part/'validation.jsonl').read_text())
            traces.append([json.loads(s) for s in (whole/'trace.jsonl').read_text().splitlines()])
            self.assertEqual(fingerprint(m),a['recipe']['parent_fingerprint'])
            self.assertFalse(m.middle[2].self_attn._forward_hooks)
        self.assertEqual([x['rows'] for x in traces[0]],[x['rows'] for x in traces[1]])
        self.assertEqual([x['lr'] for x in traces[0]],[x['lr'] for x in traces[1]])
        self.assertEqual(self.f.pins,{k:sha(self.f.root/'parent'/k) for k in self.f.pins})

    def test_first_step_independent_adam_and_monitor_rng(self):
        c=self.c;out=self.f.root/'shift-first';train(c,out,stop_after=1)
        actual=torch.load(out/'fullstate.pt',weights_only=False)
        m,_,_,_=load_parent(c['parent'],c['parent_hashes']);freeze(m);m.train()
        rows=Rows(c['data'],8);x=rows.batch(np.array([160,161]),'cpu');b=torch.nn.Parameter(torch.zeros(8))
        opt=torch.optim.AdamW([b],lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=0.)
        h=manual_hidden(m,x,b,'C');loss=torch.nn.functional.cross_entropy(torch.nn.functional.linear(h[:,1:-1],m.embed.weight).reshape(-1,19),x[:,2:].reshape(-1));loss.backward()
        torch.nn.utils.clip_grad_norm_([b],1.);opt.step()
        torch.testing.assert_close(actual['bias'],b.detach(),atol=1e-9,rtol=1e-6)
        state=torch.get_rng_state().clone();mode=m.training
        monitor(m,b,rows,np.array([196,197]),'cpu')
        torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0);self.assertEqual(mode,m.training)

    def test_binding_resume_counter_and_gpu_guards(self):
        c=copy.deepcopy(self.c);c['data_recipe']['sha256']='bad'
        with self.assertRaisesRegex(ValueError,'recipe hash'):selections(c,8)
        c=copy.deepcopy(self.c);c['expected_inputs'][c['indices']]='bad'
        with self.assertRaisesRegex(ValueError,'binding'):selections(c,8)
        c=copy.deepcopy(self.c);c['fixture_only']=False
        with self.assertRaisesRegex(ValueError,'budget'):train(c,self.f.root/'bad-production')
        with self.assertRaisesRegex(ValueError,'GPU correction admission'):train(self.c,self.f.root/'bad-cuda',device='cuda')
        out=self.f.root/'shift-guard';train(self.c,out,stop_after=2)
        with self.assertRaisesRegex(ValueError,'recipe'):train(dict(self.c,branch='S'),out,resume=True)
        s=torch.load(out/'fullstate.pt',weights_only=False);bad=copy.deepcopy(s['optimizer']);bad['state'][0]['step'].zero_()
        with self.assertRaisesRegex(ValueError,'counter'):validate_optimizer(bad,2,8)
        (out/'fullstate.pt').write_bytes((out/'fullstate.pt').read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'hash'):train(self.c,out,resume=True)


if __name__=='__main__':unittest.main()

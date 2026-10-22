"""Channel-split invariants, routing gradients, replay, loading and causality."""
import copy
import tempfile
import unittest
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
from test_model import model, batch
from test_grouped import enable_grouped, reference_mm
from bdot import grouped
from bdot.model import BDOTForCausalLM
from bdot.routing import RoutingConfig


def partial(fraction=.25, **kwargs):
    return model(route_scope='mlp', routed_mlp_fraction=fraction, curriculum='A', **kwargs)


class PartialMLPTests(unittest.TestCase):
    def test_six_job_pairs_and_recipe_contract(self):
        from scripts.bdot_mlp_pipeline import prepare
        from scripts.make_bdot_pilot import MLP_RUNS
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);pilot=tmp/'pilot'
            subprocess.run([sys.executable,str(root/'scripts/make_bdot_pilot.py'),
                '--output',str(pilot/'prepared'),'--tokenizer',str(tmp/'tok'),'--data',str(tmp/'data'),
                '--eval-data',str(tmp/'eval'),'--revision','a'*40,'--accelerate-config',str(tmp/'accelerate')],
                check=True,capture_output=True)
            path=prepare(tmp/'new',pilot)
            jobs=json.loads(path.read_text())['jobs']
            self.assertEqual(len(jobs),14)
            self.assertEqual([j['name'] for j in jobs[1:-1:2]],[name+'-s0' for name,_ in MLP_RUNS])
            self.assertEqual(jobs[0]['name'],'mlp_acceptance')
            self.assertEqual(jobs[-1]['name'],'mlp_report')
            for j in jobs[:-1]:self.assertEqual(j['gpus'],list(range(8)))
            for base in ('Partial-25','Partial-50','MLP-only'):
                configs=[json.loads((tmp/'new/prepared'/(name+'-s0.model.json')).read_text())
                         for name in (base,base+'-random')]
                for c in configs:RoutingConfig(**c['bdot'])
                configs[0]['bdot']['routing']='random'
                self.assertEqual(*configs)

    def test_home_split_preserves_initial_weights_logits_gradients_and_updates(self):
        for fraction in (.25, .5, 1.):
            with self.subTest(fraction=fraction):
                m=partial(fraction, step_film=False)
                original=model(curriculum='A',step_film=False)
                ref=m.native_reference()
                dense=original.native_reference()
                for (name,p),(_,q) in zip(ref.named_parameters(),dense.named_parameters()):
                    torch.testing.assert_close(p,q,atol=0,rtol=0,msg=name)
                self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in original.parameters()))
                for s,t in zip(m.routed_steps,original.routed_steps):
                    for p,q in zip(s.router.parameters(),t.router.parameters()):
                        torch.testing.assert_close(p,q,atol=0,rtol=0)
                a=m(**batch());b=ref(**batch())
                torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-5)
                a.loss.backward();b.loss.backward()
                # Compare the channel gradients against the unsplit dense MLP.
                for i,step in enumerate(m.routed_steps):
                    native=ref.model.layers[m.options.n_stem+i]
                    routed=m.model.bdot_experts[i].mlp
                    for name in ('gate_proj','up_proj','down_proj'):
                        pieces=[getattr(routed,name).weight.grad]
                        if step.core_mlp is not None:pieces.insert(0,getattr(step.core_mlp,name).weight.grad)
                        grad=torch.cat(pieces,dim=1 if name=='down_proj' else 0)
                        torch.testing.assert_close(grad,getattr(native.mlp,name).weight.grad,atol=1e-6,rtol=1e-4)
                    self.assertTrue(all(p.grad is None for p in step.router.parameters()))
                for net in (m,ref):torch.optim.AdamW(net.parameters(),lr=1e-3).step()
                merged=m.native_reference()
                for (name,p),(_,q) in zip(merged.named_parameters(),ref.named_parameters()):
                    torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-4,msg=name)

    def test_random_has_identical_curriculum_and_stays_home_initially(self):
        learned=partial();random=partial(routing='random')
        for progress in (0.,.0999,.1,.15,.2,1.):
            self.assertEqual(learned.options.phase(progress,True)[:2],random.options.phase(progress,True)[:2])
            random.train();random.progress=progress;random.capture_routes=True
            random(**batch())
            home=torch.arange(4)[:,None,None]
            self.assertTrue(((random.last_routes-home).abs() <= (0 if progress<.1 else 1)).all())
        with self.assertRaises(ValueError):RoutingConfig(routing='random',curriculum='A')
        with self.assertRaises(ValueError):RoutingConfig(route_scope='mlp',curriculum='B')

    def test_gate_gradient_is_routed_component_only_and_unselected_expert_unused(self):
        m=partial();step=m.routed_steps[1]
        x=torch.randn(2,3,16,requires_grad=True)
        gate=torch.ones(6,1,requires_grad=True)
        ids=torch.zeros(6,1,dtype=torch.long)
        result=step.mlp_update(x,ids,gate,False)
        expected=step.core_mlp(x)+m.model.bdot_experts[0].mlp(x)
        torch.testing.assert_close(result,expected)
        signal=torch.randn_like(result)
        result.backward(signal)
        expected_gate=(m.model.bdot_experts[0].mlp(x).detach()*signal).sum(-1).reshape(6,1)
        torch.testing.assert_close(gate.grad,expected_gate,atol=1e-6,rtol=1e-5)
        self.assertTrue(all(p.grad is None for p in m.model.bdot_experts[2].mlp.parameters()))

    def test_grouped_and_checkpointed_mixed_routes_match(self):
        for fraction in (.25,.5,1.):
            a=partial(fraction,default_prior=0.,explore_eps_max=.5,z_loss_coef=0.)
            b=enable_grouped(copy.deepcopy(a));a.progress=b.progress=.15
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            a.capture_routes=b.capture_routes=True
            with patch.object(grouped,'_grouped_mm',side_effect=reference_mm):
                x=a(**batch());y=b(**batch());x.loss.backward();y.loss.backward()
            torch.testing.assert_close(x.logits,y.logits,atol=1e-6,rtol=1e-5)
            self.assertTrue(torch.equal(a.last_routes,b.last_routes))
            self.assertTrue(torch.equal(a.load_counts,b.load_counts))
            for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                if p.grad is None:self.assertIsNone(q.grad,name)
                else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4,msg=name)
            for step in a.routed_steps:self.assertGreater(float(step.router.weight.grad.abs().sum()),0.)

    def test_loading_and_next_random_draw(self):
        for fraction in (.25,.5,1.):
            for routing in ('learned','random'):
                a=partial(fraction,routing=routing);a.progress=.15
                a(**batch()).loss.backward();torch.optim.AdamW(a.parameters(),lr=.001).step();a.finish_update()
                with tempfile.TemporaryDirectory() as tmp:
                    a.save_pretrained(tmp);b=BDOTForCausalLM.from_pretrained(tmp,attn_implementation='sdpa')
                for key,value in a.state_dict().items():
                    torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0,msg=key)
                a.train();b.train();b.progress=a.progress;a.capture_routes=b.capture_routes=True
                torch.testing.assert_close(a(**batch()).logits,b(**batch()).logits,atol=0,rtol=0)
                self.assertTrue(torch.equal(a.last_routes,b.last_routes))

    def test_causality_isolation_and_cached_logits(self):
        m=partial(default_prior=0.).eval();inputs=batch();out=m(**inputs).logits
        changed=copy.deepcopy(inputs);changed['input_ids'][0,2:]=17
        torch.testing.assert_close(out[0,:2],m(**changed).logits[0,:2],atol=1e-6,rtol=1e-5)
        changed=copy.deepcopy(inputs);changed['input_ids'][0,:3]=18
        torch.testing.assert_close(out[0,3:],m(**changed).logits[0,3:],atol=1e-6,rtol=1e-5)
        cache=None
        for t in range(6):
            step=m(input_ids=inputs['input_ids'][:,t:t+1],position_ids=inputs['position_ids'][:,t:t+1],
                attention_mask=inputs['attention_mask'][:,:,t:t+1,:t+1],use_cache=True,past_key_values=cache)
            cache=step.past_key_values
            torch.testing.assert_close(out[:,t],step.logits[:,0],atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()

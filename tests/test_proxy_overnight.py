"""P7 deep residuals: forward reference, routing, backends, resume and queue gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from deep_kv import P7_DEEP_ARMS, followup_metadata
from deep_kv.proxy import compute_budget
from deep_kv.proxy_memory import MemoryPlan
from tests.test_proxy_screen24 import model, config, fixture
from tests.test_proxy_extensions import activate
from tests.test_proxy_heads import batch, objective


def reference(m,ctx):
    h=m.backbone.model.embed_tokens(ctx.input_ids)
    rotary=m.backbone.model.rotary_emb(h,ctx.position_ids)
    plan=MemoryPlan(ctx.segments,sample_queries=False,convolution=False)
    pending={};predictions={};mlps={}
    for number,block in enumerate(m.backbone.model.layers,1):
        u=block.input_layernorm(h)
        if number in m.layers:
            head=m.heads[str(number)]
            p=F.linear(F.silu(F.linear(u.detach(),head.w1.weight)),head.w2.weight)
            predictions[number]=p
            pending[number+3]=(number,u.detach() if m.arm.endswith('-shallow') else p)
            attention=m.memory_attention(block,u,head,p,ctx.allowed(),rotary,plan)
        else:attention=m.attention(block,u,None,ctx.allowed(),rotary)
        residual=h+block.self_attn.o_proj(attention.reshape(*u.shape[:-1],-1))
        mlps[number]=block.mlp(block.post_attention_layernorm(residual));h=residual+mlps[number]
        if number in pending:
            site,x=pending.pop(number);ext=m.extensions[str(site)]
            normalized=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
            added=F.linear(F.silu(F.linear(normalized.to(x.dtype),ext.down.weight)),ext.up.weight)
            scale=(h.detach().float().square().mean(-1,keepdim=True)+1e-6).sqrt()
            h=h+(added*scale).to(h.dtype)
    loss=[]
    for site,p in predictions.items():
        raw=sum(mlps[i].detach().float() for i in range(site,site+4));j=m.mean_index[site];v=m.sigma2[j]
        target=((raw-m.mu[j])*torch.rsqrt(v.clamp_min(.01*v.median())+1e-6)).clamp(-10,10)
        loss.append(((1-F.cosine_similarity(p.float(),target,dim=-1,eps=1e-6))*ctx.valid).sum())
    h=m.backbone.model.norm(h)
    return h,m.lm_statistics(ctx,h)[0].sum(),torch.stack(loss).mean()


class OvernightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_parent_and_independent_active_reference(self):
        parent=model('P7-simple');expected=parent(batch())
        for arm in P7_DEEP_ARMS:
            m=model(arm)
            for n,p in parent.state_dict().items():torch.testing.assert_close(m.state_dict()[n],p,rtol=0,atol=0)
            for key in ('lm_sum','aux_sum'):torch.testing.assert_close(m(batch())[key],expected[key],rtol=0,atol=0)
            self.assertEqual(compute_budget(config(),m.settings,8)[arm]['extra_parameters'],
                sum(p.numel() for p in m.heads.parameters())+sum(p.numel() for p in m.extensions.parameters()))
            activate(m);ref=copy.deepcopy(m);ctx=batch();out=m(ctx);h,lm,aux=reference(ref,ctx)
            torch.testing.assert_close(out['lm_sum'],lm,rtol=1e-6,atol=2e-6)
            torch.testing.assert_close(out['aux_sum'],aux,rtol=1e-6,atol=2e-6)
            torch.testing.assert_close(m.hidden_states(ctx)[0],h,rtol=1e-6,atol=2e-6)
            objective(m,out).backward();(lm/out['lm_count']+.1*aux/out['aux_count']).backward()
            for (n,p),(_,q) in zip(m.named_parameters(),ref.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=4e-5,atol=3e-6,msg=n)

    def test_gradient_routes_causality_and_fa4(self):
        from tests.test_fa4_baseline import reference_kernel
        for arm in P7_DEEP_ARMS:
            m=model(arm);activate(m)
            for key in ('lm_sum','aux_sum'):
                m.zero_grad(set_to_none=True);m(batch())[key].backward()
                for h in m.heads.values():
                    learns=key=='aux_sum' or arm=='P7-simple-deep'
                    self.assertEqual(h.w1.weight.grad is not None and bool(h.w1.weight.grad.abs().sum()),learns)
                if key=='aux_sum':
                    self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
                    self.assertTrue(all(p.grad is None for p in m.extensions.parameters()))
            ctx=batch();hidden=m.hidden_states(ctx)[0]
            changed=copy.deepcopy(ctx);changed.input_ids[:,:3]=20
            torch.testing.assert_close(m.hidden_states(changed)[0][:,3:],hidden[:,3:],rtol=0,atol=0)
            changed=copy.deepcopy(ctx);changed.input_ids[:,6:]=20
            torch.testing.assert_close(m.hidden_states(changed)[0][:,:6],hidden[:,:6],rtol=0,atol=0)
            with m.without_proxy():torch.testing.assert_close(m.hidden_states(ctx)[0],model('A').hidden_states(ctx)[0],rtol=0,atol=0)
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                flash=model(arm,backend='fa4');flash.load_state_dict(m.state_dict())
            m.zero_grad(set_to_none=True)
            for x in (m,flash):objective(x,x(ctx)).backward()
            for (n,p),(_,q) in zip(m.named_parameters(),flash.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=4e-4,atol=3e-6,msg=n)
            for bf16 in (False,True):
                ref=model(arm);activate(ref);checked=model(arm,checkpoint=True);checked.load_state_dict(ref.state_dict())
                for x in (ref,checked):
                    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):objective(x,x(ctx)).backward()
                for (n,p),(_,q) in zip(ref.named_parameters(),checked.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,n)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=.03 if bf16 else 3e-5,atol=3e-6,msg=n)

    def test_weight_scaling_and_unaffected_lm(self):
        for arm in ('P6','P7-simple-lm','P6-iso-targetgrad'):
            lo=model(arm,lambda_max=.03);hi=model(arm,lambda_max=.3)
            torch.testing.assert_close(lo(batch())['lm_sum'],hi(batch())['lm_sum'],rtol=0,atol=0)
            for m in (lo,hi):(m.auxiliary_weight(250)*m(batch())['aux_sum']).backward()
            for (n,p),(_,q) in zip(lo.named_parameters(),hi.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:torch.testing.assert_close(q.grad,10*p.grad,rtol=5e-5,atol=2e-6,msg=n)

    def test_real_trainer_resume_reload_report(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        from eval.models import load_checkpoint
        from deep_kv.report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root)
            for arm in P7_DEEP_ARMS:
                args={**recipe,'arm':arm,'output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':1});invoke(root,args)
                invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                for n in resumed:torch.testing.assert_close(resumed[n],full[n],rtol=0,atol=0,msg=n)
                loaded,_,_=load_checkpoint(root/arm,device='cpu',attention_backend='sdpa')
                self.assertEqual(loaded.wrapped.arm,arm)
                c=json.loads((root/arm/'train_config.json').read_text())
                self.assertEqual(c['proxy_target']['extension'],followup_metadata(arm)['extension'])
            report(root,P7_DEEP_ARMS)

    def test_summary_reuses_only_matched_baseline(self):
        from scripts.proxy_overnight_queue import CASES,summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'A';base.mkdir()
            a=dict(arm='A',global_step=2500,status='complete',evaluation=dict(eval_lm_loss=3.,eval_target_tokens=100))
            c=dict(train_fingerprint='train',eval_fingerprint='eval',training=dict(seed=42,data_seed=42))
            (base/'result.json').write_text(json.dumps(a));(base/'train_config.json').write_text(json.dumps(c))
            for label,arm,weight in CASES:
                folder=root/label/'seed-42'/arm;folder.mkdir(parents=True)
                (folder/'result.json').write_text(json.dumps({**a,'arm':arm,'training_cost':{}}))
                (folder/'train_config.json').write_text(json.dumps({**c,'pilot':{'proxy_lambda_max':weight}}))
                (root/label/'validated.json').write_text(json.dumps({'status':'passed'}))
            bad={**c,'eval_fingerprint':'different'};(base/'train_config.json').write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):summarize(root,base)
            self.assertFalse((root/'summary.json').exists())
            (base/'train_config.json').write_text(json.dumps(c));summarize(root,base)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(len(summary['results']),len(CASES))
            self.assertTrue(all(r['lm_loss_delta_vs_A']==0 for r in summary['results']))

    def test_queue_and_predecessor_fail_closed(self):
        from scripts.proxy_overnight_queue import make,CASES,predecessor_ready
        from scripts.proxy_12layer_queue import ALL_SCREEN_ARMS
        from deep_kv import P6_CONSUMER_ARMS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);items=make('proxy_heads.12layer.b200.json',root/'jobs.json',root/'rows.json')
            self.assertEqual(len(items),4*len(CASES)+2)
            production=[j for j in items if j['name'].endswith('-train') and not j['name'].startswith('smoke-')]
            self.assertEqual(len(production),len(CASES))
            for (label,arm,weight),j in zip(CASES,production):
                self.assertEqual(j['gpus'],list(range(8)));a=j['argv']
                for flag,value in [('arm',arm),('proxy_lambda_max',str(weight)),('max_steps','2500'),('stop_after','2500'),('save_total_limit','0'),('proxy_module_seed','43')]:self.assertEqual(a[a.index('--'+flag)+1],value)
                self.assertNotIn('--resume_from_checkpoint',a)
                self.assertIn('/'+label+'/seed-42/'+arm,a[a.index('--output_dir')+1])
            outputs=[o['path'] for j in items for o in j['required_outputs']]
            self.assertEqual(len(outputs),len(set(outputs)))
            predecessor=root/'previous';s=predecessor/'supervised';s.mkdir(parents=True);guard=root/'guard'
            self.assertFalse(predecessor_ready(predecessor,guard))
            state={'training_status':'running'};(s/'supervisor.json').write_text(json.dumps(state))
            self.assertFalse(predecessor_ready(predecessor,guard))
            state.update(training_status='ok',training_returncode=0,finished_at='done',burn={'session':'burn','collective_progress_verified':True})
            (s/'supervisor.json').write_text(json.dumps(state));(s/'burn-verified.json').write_text(json.dumps(state['burn']))
            (s/'run').mkdir();run={'status':'ok','jobs':[dict(name=f'seed-42-{kind}-{arm}',status='ok') for arm in ALL_SCREEN_ARMS if arm not in P6_CONSUMER_ARMS for kind in ('arm','validate')]}
            (s/'run/run.json').write_text(json.dumps(run));guard.touch();self.assertFalse(predecessor_ready(predecessor,guard));guard.unlink()
            self.assertTrue(predecessor_ready(predecessor,guard))
            run['jobs'][-1]['status']='failed';(s/'run/run.json').write_text(json.dumps(run))
            with self.assertRaises(RuntimeError):predecessor_ready(predecessor,guard)


if __name__=='__main__':unittest.main()

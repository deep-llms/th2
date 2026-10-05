"""Revision-7 proxy contract: local CPU acceptance, no GPU workload or real-data training."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import TrainingArguments

from deep_kv import SCREEN_ARMS
from deep_kv.model import DeepKV
from deep_kv.proxy import EMSPlan, ProxyModel, ProxySettings, compute_budget, proxy_layers
from deep_kv.proxy_training import ProxyTrainer
from deep_kv.packing import isolated_data_collator
from tests.test_deep_kv import config


def cfg():
    c=config();c.num_hidden_layers=8;c.layer_types=["full_attention"]*8
    return c


def settings(**kwargs):
    return ProxySettings(groups=1,width=8,features=9,chunk_size=2,**kwargs)


def model(arm,checkpoint=False,**kwargs):
    return ProxyModel.from_scratch(cfg(),arm,consumer=2,deep_target=8,proxy_settings=settings(**kwargs),
        checkpoint_layers=checkpoint,checkpoint_lm=checkpoint,checkpoint_aux=checkpoint,lm_chunk=3)


def batch():
    return ProxyTrainer.context(isolated_data_collator([dict(input_ids=[3,4,31,8,9,10,11,31],
        labels=[3,4,31,8,9,10,11,31],attention_mask=[1]*8,segments=[0]*3+[1]*5)]))


def objective(m,out):
    return out['lm_sum']/out['lm_count']+.1*out['aux_sum']/out['aux_count'].clamp_min(1)


class ProxyHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_t1_zero_gate_equivalence_and_shared_initialization(self):
        base=DeepKV.from_scratch(cfg(),'A',consumer=2,deep_target=8,checkpoint_layers=False).eval();ctx=batch()
        reference=base(ctx)
        with torch.no_grad():
            for arm in SCREEN_ARMS:
                if arm.startswith('V'):continue
                current=model(arm).eval()
                for name,value in base.backbone.state_dict().items():
                    torch.testing.assert_close(current.backbone.state_dict()[name],value,rtol=0,atol=0)
                captures=[]
                handles=[m.backbone.model.norm.register_forward_hook(lambda module,args,out: captures.append(out.clone())) for m in (base,current)]
                base(ctx);current(ctx)
                for handle in handles:handle.remove()
                delta=(base.backbone.lm_head(captures[0])-current.backbone.lm_head(captures[1])).abs().max().item()
                self.assertLessEqual(delta,1e-5,(arm,delta))
                torch.testing.assert_close(current(ctx)['lm_sum'],reference['lm_sum'],rtol=0,atol=1e-5)

    def test_t2_gqa_replacement_mapping(self):
        for arm in ('P1-block','P3-block'):
            m=model(arm);ctx=batch();head=m.heads['2'];layer=m.backbone.model.layers[1]
            u=torch.randn(1,8,32)
            rotary=m.backbone.model.rotary_emb(u,ctx.position_ids)
            plan=EMSPlan(ctx.segments,m.gamma,2) if m.family=='P3' else None
            estimates=head.estimates(u,plan)
            native=m.attention(layer,u,head.inject(u,estimates),ctx.allowed(),rotary)
            with torch.no_grad():head.alpha.normal_(0,.2)
            proxy=m.attention(layer,u,head.inject(u,estimates),ctx.allowed(),rotary)
            # [batch, token, query-head, head-dim], nk=2, nq=4, final group replaced.
            torch.testing.assert_close(native[:,:,:2],proxy[:,:,:2],rtol=0,atol=0)
            self.assertGreater((native[:,:,2:]-proxy[:,:,2:]).abs().max().item(),1e-5)

    def test_t4_chunkwise_dense_and_sequential_all_boundaries(self):
        gamma=torch.tensor([.5,.9,.99]);t=256;c=64
        layouts=[[],[64],[23],[1,2,7,15,63,64,129], [1,65,193]]
        maximum=0.
        for starts in layouts:
            ids=torch.zeros(1,t,dtype=torch.long)
            for start in starts:ids[:,start:]+=1
            plan=EMSPlan(ids,gamma,c)
            # Positive inputs keep pointwise relative errors well conditioned.
            x=(torch.rand(1,t,7)+.1).requires_grad_()
            for scale,g in enumerate(gamma):
                actual=plan.apply(x,scale)
                index=torch.arange(t);delta=index[:,None]-index[None,:]
                mask=(delta>=0)[None] & (ids[:,:,None]==ids[:,None,:])
                matrix=(1-g)*torch.exp(delta.clamp_min(0)*torch.log(g))*mask
                dense=matrix@x
                state=torch.zeros_like(x[:,0]);rows=[]
                for j in range(t):
                    if j==0 or ids[0,j]!=ids[0,j-1]:state=torch.zeros_like(state)
                    state=g*state+(1-g)*x[:,j];rows.append(state)
                sequential=torch.stack(rows,1)
                for reference in (dense,sequential):
                    error=((actual-reference).abs()/reference.abs()).max().item()
                    maximum=max(maximum,error)
                    self.assertLessEqual(error,1e-5,(starts,float(g),error))
                grad=torch.randn_like(actual)
                a=torch.autograd.grad(actual,x,grad,retain_graph=True)[0]
                b=torch.autograd.grad(dense,x,grad,retain_graph=True)[0]
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        print('T4 maximum pointwise relative error',maximum)

    def test_t3_all_document_two_outputs_targets_and_gradients_isolated(self):
        for arm in ('P1-block','P1-flow','P3-block','P3-flow'):
            m=model(arm,checkpoint=True)
            with torch.no_grad():
                for head in m.heads.values():head.alpha.fill_(.2)
            ctx=batch();ctx.valid[:,:3]=True
            # Capture complete normalized states, and both estimator/target EMS calls.
            records=[]
            def capture(module,inputs,output):
                output.retain_grad();records.append(output)
            hook=m.backbone.model.embed_tokens.register_forward_hook(capture)
            # Only document two contributes to loss; masks still contain document one.
            ctx.labels=ctx.input_ids.clone();ctx.labels[:,:3]=-100
            targets=[]
            activations=[];scan_states=[]
            original_block=m.block;original_scan=EMSPlan.apply
            def capture_block(*args):
                result=original_block(*args);activations.append(result[0].detach()[:,3:].clone());return result
            def capture_scan(plan,*args):
                result=original_scan(plan,*args);scan_states.append(result.detach()[:,3:].clone());return result
            original=m.layer_cosines
            def doc2(layer,u,estimates,raw,plan,valid,auxiliary_grad):
                targets.append(raw.detach().clone())
                eligible=valid.clone();eligible[:,:3]=False
                return original(layer,u,estimates,raw,plan,eligible,auxiliary_grad)
            with patch.object(m,'layer_cosines',side_effect=doc2), \
                    patch.object(m,'block',side_effect=capture_block), patch.object(EMSPlan,'apply',capture_scan):
                out=m(ctx);loss=out['lm_sum']+.1*out['aux_sum']
                first_activations=activations.copy();first_scans=scan_states.copy()
                loss.backward()
                self.assertEqual(records[0].grad[:,:3].count_nonzero().item(),0,arm)
                first=[value[:,3:].clone() for value in targets];targets.clear()
                activations.clear();scan_states.clear()
                changed=copy.deepcopy(ctx);changed.input_ids[:,:2]=torch.tensor([15,16])
                m.zero_grad(set_to_none=True);other=m(changed)
                torch.testing.assert_close(out['lm_sum'],other['lm_sum'],rtol=0,atol=0)
                for a,b in zip(first,targets):torch.testing.assert_close(a,b[:,3:],rtol=0,atol=0)
                for before,after in ((first_activations,activations),(first_scans,scan_states)):
                    self.assertEqual(len(before),len(after))
                    for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
            hook.remove()

    def test_t5_t6_t7_gradients_detachment_and_lambda_zero(self):
        for arm in ('P1-block','P1-flow','P3-block','P3-flow'):
            m=model(arm)
            originals=[]
            original=m.layer_cosines
            def capture(layer,u,estimates,raw,*args):
                self.assertFalse(raw.requires_grad)
                originals.append(raw)
                return original(layer,u,estimates,raw,*args)
            with patch.object(m,'layer_cosines',side_effect=capture):out=m(batch())
            self.assertTrue(originals);self.assertTrue(torch.isfinite(out['aux_sum']))
            out['aux_sum'].backward()
            backbone=sum(float(p.grad.abs().sum()) for p in m.backbone.parameters() if p.grad is not None)
            if arm.endswith('block'):self.assertEqual(backbone,0)
            else:self.assertGreater(backbone,0)
            if arm.endswith('block'):
                self.assertTrue(all(h.alpha.grad is None or not h.alpha.grad.any() for h in m.heads.values()))
            m.zero_grad(set_to_none=True);out=m(batch());out['lm_sum'].backward()
            self.assertTrue(all(h.alpha.grad.abs().sum()>0 for h in m.heads.values()))
            control=model(arm[:2]+'-lambda0')
            diagnostic=control(batch(),auxiliary_grad=False)
            self.assertFalse(diagnostic['aux_sum'].requires_grad)
            self.assertEqual(control.auxiliary_weight(1000),0)
            control.zero_grad(set_to_none=True);diagnostic['lm_sum'].backward()
            for h in control.heads.values():self.assertGreater(h.alpha.grad.abs().sum(),0)

    def test_t8_compute_parameter_counts_and_gate_groups(self):
        from transformers import Qwen3Config
        config=Qwen3Config(hidden_size=1024,intermediate_size=3072,num_hidden_layers=28,
            num_attention_heads=16,num_key_value_heads=8,head_dim=128,vocab_size=151936)
        budget=compute_budget(config,ProxySettings())
        self.assertEqual(budget['P1']['widening'],72);self.assertEqual(budget['P3']['widening'],88)
        self.assertLess(budget['P1']['mismatch_fraction_of_total'],.002)
        self.assertLess(budget['P3']['mismatch_fraction_of_total'],.002)
        base=sum(p.numel() for p in model('A').parameters())
        for arm in ('P1-block','P3-block','V1','V3'):
            m=model(arm);cost=compute_budget(cfg(),settings())
            family=arm[:2] if arm.startswith('P') else 'P'+arm[1]
            expected=cost[family]['extra_parameters' if arm.startswith('P') else 'widened_extra_parameters']
            self.assertEqual(sum(p.numel() for p in m.parameters())-base,expected)
        with tempfile.TemporaryDirectory() as tmp:
            m=model('P3-block');trainer=ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[],weight_decay=.1))
            trainer.create_optimizer()
            decay={id(p):g['weight_decay'] for g in trainer.optimizer.param_groups for p in g['params']}
            for h in m.heads.values():
                self.assertEqual(decay[id(h.alpha)],0)
                self.assertEqual(decay[id(h.w1.weight)],.1)
                self.assertEqual(decay[id(h.w2.weight)],.1)
                self.assertTrue(all(decay[id(p.weight)]==.1 for p in h.projections))

    def test_checkpoint_replay_and_mu_update_are_separate(self):
        for arm in ('P1-block','P1-flow','P3-block','P3-flow'):
            a=model(arm);b=model(arm,checkpoint=True);b.load_state_dict(a.state_dict())
            for m in (a,b):
                for phase in ('mean','variance'):
                    out=m(batch(),compute_auxiliary_losses=False,collect_target_statistics=True,statistics_mode=phase)
                    m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'],initialize=phase)
                before=m.mu.clone();out=m(batch(),collect_target_statistics=True)
                objective(m,out).backward()
                torch.testing.assert_close(m.mu,before,rtol=0,atol=0)
                expected=before+.01*out['center_sums']/out['center_counts'][:,None]
                m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'])
                torch.testing.assert_close(m.mu,expected)
            for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                if p.grad is None:self.assertIsNone(q.grad,name)
                else:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6,msg=name)
            torch.testing.assert_close(a.mu,b.mu,rtol=0,atol=0)

    def test_hidden_states_uses_proxy_path_and_channel_standardization(self):
        for arm in ('P1-block','P3-block'):
            m=model(arm).eval();ctx=batch()
            with torch.no_grad():
                m.sigma2.fill_(4.)
                m.mu.fill_(.25)
                for head in m.heads.values():head.alpha.fill_(.2)
                hidden,_,_=m.hidden_states(ctx)
                lm_rows,_,_=m.lm_statistics(ctx,hidden)
                torch.testing.assert_close(lm_rows.sum(),m(ctx)['lm_sum'],rtol=0,atol=0)
                with m.without_proxy():native,_,_=m.hidden_states(ctx)
                self.assertGreater((hidden-native).abs().max().item(),1e-5)
                raw=torch.arange(32,dtype=torch.float32).view(1,1,32)
                expected=((raw-.25)/(4.+1e-6)**.5).clamp(-10,10)
                normalized=m.normalize_target(raw,m.mean_layers[0])
                torch.testing.assert_close(normalized,expected)
                self.assertFalse(normalized.requires_grad)

    def test_lookahead_windows_and_absolute_lambda_schedule(self):
        # Enough layers to exercise k=8, including its final complete target window.
        for k in (1,2,4,8):
            config=cfg();config.num_hidden_layers=12;config.layer_types=['full_attention']*12
            m=ProxyModel.from_scratch(config,'P1-block',consumer=2,deep_target=12,
                                      proxy_settings=settings(lookahead=k))
            self.assertEqual(m.layers,proxy_layers(config,'P1',k))
            observed={};original=m.block
            def capture(index,*args):
                value=original(index,*args);observed[index+1]=value[1].float().clone();return value
            with patch.object(m,'block',side_effect=capture):
                out=m(batch(),compute_auxiliary_losses=False,collect_target_statistics=True)
            for layer in m.layers:
                expected=sum(observed[i] for i in range(layer,layer+k)).sum((0,1))
                torch.testing.assert_close(out['center_sums'][m.mean_index[layer]],expected)
        m=model('P1-block',decay_start=1000,decay_end=2000)
        for step,expected in ((0,0),(125,.05),(250,.1),(1000,.1),(1500,.05),(2000,0),(2500,0)):
            self.assertAlmostEqual(m.auxiliary_weight(step),expected)

    def test_all_screen_arms_use_one_boolean_sdpa_call_per_layer(self):
        from deep_kv.sdpa_audit import SDPAAudit
        for arm in SCREEN_ARMS:
            m=model(arm);audit=SDPAAudit(m)
            with audit.observe():
                out=m(batch());loss=objective(m,out)
                loss.register_hook(audit.backward_marker);loss.backward()
            self.assertFalse(audit.result()['unattributed_calls'])
            self.assertFalse(audit.result()['unattributed_backward'])
            self.assertEqual(len(audit.calls),8)
            for call in audit.calls:
                self.assertEqual(call['mask']['dtype'],'torch.bool')
                self.assertEqual(call['mask']['shape'],[1,1,8,8])
                self.assertFalse(call['is_causal'])

    def test_widening_uses_actual_sequence_length(self):
        widths=[]
        for length in (8,2048):
            cost=compute_budget(cfg(),settings(),length)
            m=ProxyModel.from_scratch(cfg(),'V3',consumer=2,deep_target=8,
                proxy_settings=settings(),sequence_length=length)
            widths.append(m.backbone.config.intermediate_size)
            self.assertEqual(widths[-1]-cfg().intermediate_size,cost['P3']['widening'])
        self.assertNotEqual(*widths)
        for chunk,length in ((0,8),(3,8),(2,0)):
            invalid=settings();invalid.chunk_size=chunk
            with self.assertRaisesRegex(ValueError,'chunk_size'):
                compute_budget(cfg(),invalid,length)

    def test_full_depth_targets_losses_and_gradients_against_independent_reference(self):
        # Preserve the real 28-layer, 16Q/8KV, Q-width=2d layout at a small d.
        config=cfg();config.num_hidden_layers=28;config.layer_types=['full_attention']*28
        config.num_attention_heads=16;config.num_key_value_heads=8;config.head_dim=4
        rows=[dict(input_ids=list(range(3,19)),labels=list(range(3,19)),attention_mask=[1]*16,
                   segments=ids) for ids in ([0]*3+[1]*13,[0]+[1]*7+[2]*8)]
        ctx=ProxyTrainer.context(isolated_data_collator(rows))
        for arm,groups in (('P1-block',1),('P1-flow',2),('P3-block',4),('P3-flow',2)):
            opt=ProxySettings(groups=groups,width=8,features=9,chunk_size=4)
            m=ProxyModel.from_scratch(config,arm,consumer=2,deep_target=28,proxy_settings=opt,
                checkpoint_layers=False,checkpoint_aux=False,checkpoint_lm=False)
            with torch.no_grad():
                m.mu.normal_(0,.05);m.sigma2.uniform_(.01,.1)
                for head in m.heads.values():head.alpha.fill_(.15)
            blocks={};original=m.block
            def capture(index,*args):
                value=original(index,*args);blocks[index+1]=value;return value
            with patch.object(m,'block',side_effect=capture):
                actual=m(ctx,collect_target_statistics=True)
            def normalized(x,layer):
                centered=x.detach().float()-m.mu[m.mean_index[layer]]
                variance=m.sigma2[m.mean_index[layer]]
                return (centered/torch.sqrt(variance.clamp_min(.01*variance.median())+1e-6)).clamp(-10,10)
            def sequential(x,gamma):
                state=torch.zeros_like(x[:,0]);values=[]
                for token in range(x.shape[1]):
                    if token:
                        state=state*(ctx.segments[:,token]==ctx.segments[:,token-1])[:,None]
                    state=gamma*state+(1-gamma)*x[:,token];values.append(state)
                return torch.stack(values,1)
            losses=[]
            for layer in m.layers:
                head=m.heads[str(layer)];u=blocks[layer][2]
                if arm.endswith('block'):u=u.detach()
                prediction=F.linear(F.silu(F.linear(u,head.w1.weight)),head.w2.weight)
                if m.family=='P1':
                    raw=sum(blocks[i][1].float() for i in range(layer,layer+4))
                    targets=(normalized(raw,layer),);predictions=(prediction,)
                    torch.testing.assert_close(actual['center_sums'][m.mean_index[layer]],(raw-m.mu[m.mean_index[layer]]).sum((0,1)))
                else:
                    band=list(range(layer+2,min(layer+6,28)+1))
                    raw=torch.stack([normalized(blocks[i][0].detach().float()-blocks[layer-1][0].detach().float(),(layer,i)) for i in band]).mean(0)
                    targets=tuple(sequential(raw,g) for g in m.gamma)
                    predictions=tuple(F.linear(sequential(p,g),projection.weight)
                        for p,g,projection in zip(prediction.chunk(3,-1),m.gamma,head.projections))
                losses.extend((1-F.cosine_similarity(p,t.detach(),dim=-1,eps=1e-6)).mean()
                              for p,t in zip(predictions,targets))
            expected=torch.stack(losses).mean()
            torch.testing.assert_close(actual['aux_sum']/actual['aux_count'],expected,rtol=1e-5,atol=1e-6)
            if m.family=='P3':
                self.assertEqual(len(m.mean_layers),58)
                for proxy,deep in m.mean_layers:
                    raw=blocks[deep][0].detach().float()-blocks[proxy-1][0].detach().float()
                    shifted=raw-m.mu[m.mean_index[(proxy,deep)]]
                    torch.testing.assert_close(actual['center_sums'][m.mean_index[(proxy,deep)]],shifted.sum((0,1)))
                    torch.testing.assert_close(actual['center_squares'][m.mean_index[(proxy,deep)]],shifted.square().sum((0,1)))
            parameters=list(m.named_parameters())
            a=torch.autograd.grad(actual['aux_sum']/actual['aux_count'],[p for _,p in parameters],retain_graph=True,allow_unused=True)
            b=torch.autograd.grad(expected,[p for _,p in parameters],allow_unused=True)
            for (name,_),left,right in zip(parameters,a,b):
                if left is None:self.assertIsNone(right,(arm,name))
                else:torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6,msg=arm+'/'+name)



if __name__=='__main__':unittest.main()

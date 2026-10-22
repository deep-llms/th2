"""Independent dispatch, precision and mask checks added during code review."""
import copy
import unittest

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from bdot.model import BDOTForCausalLM
from test_model import model, batch, mapped_pairs


def dense_step_reference(step, h, choices, gates, mask, positions):
    """Compute every module on every token, then select; no sparse dispatch code."""
    b, t, d = h.shape
    x = step.input_layernorm(h)
    k = step.k_norm(step.k_proj(x).view(b,t,-1,step.head_dim)).transpose(1,2)
    v = step.v_proj(x).view(b,t,-1,step.head_dim).transpose(1,2)
    outputs = []
    for expert in step.experts:
        q = expert.q_norm(expert.q_proj(x).view(b,t,-1,step.head_dim)).transpose(1,2)
        qrot, krot = apply_rotary_pos_emb(q,k,*positions)
        # Explicit GQA scores also provide an independent attention reference.
        scores = qrot @ krot.repeat_interleave(step.num_key_value_groups,dim=1).transpose(-1,-2)
        scores = (scores*step.scaling).masked_fill(~mask,-torch.inf)
        attention = scores.softmax(-1) @ v.repeat_interleave(step.num_key_value_groups,dim=1)
        h1 = h + expert.o_proj(attention.transpose(1,2).reshape(b,t,-1))
        u = expert.mlp_norm(h1)
        if step.options.step_film:
            u = u*(1+step.film_gain)+step.film_bias
        outputs.append(h1+expert.mlp(u))
    all_outputs = torch.stack(outputs,dim=2).reshape(b*t,len(outputs),d)
    selected = all_outputs.gather(1,choices[:,:,None].expand(-1,-1,d))
    deltas = selected-h.reshape(b*t,1,d)
    return h+(gates[:,:,None]*deltas).sum(1).reshape(b,t,d)


class ReviewTests(unittest.TestCase):
    def test_fa4_dispatch_plumbing_against_documentwise_reference(self):
        # This checks only adapter wiring. Actual FA4 CUDA numerics are still
        # a separate launch gate; no CPU kernel fallback is shipped.
        from unittest.mock import patch
        from packed_training import PackedCollator,model_inputs
        import packed_fa4
        def kernel(q,k,v,*,cu_seqlens_q,cu_seqlens_k,causal,softmax_scale,**kwargs):
            self.assertTrue(torch.equal(cu_seqlens_q,cu_seqlens_k))
            pieces=[]
            for start,end in zip(cu_seqlens_q[:-1].tolist(),cu_seqlens_q[1:].tolist()):
                out=F.scaled_dot_product_attention(q[start:end].transpose(0,1)[None],
                    k[start:end].transpose(0,1)[None],v[start:end].transpose(0,1)[None],
                    is_causal=causal,enable_gqa=True,scale=softmax_scale)
                pieces.append(out[0].transpose(0,1))
            return torch.cat(pieces),None
        rows=[dict(input_ids=[1,2,3,4,5,6],labels=[1,2,3,4,5,6],
                   attention_mask=[1]*6,segments=[0,0,1,1,1,2])]
        with patch.object(packed_fa4,'validate_kernel_inputs'),patch.object(packed_fa4,'load_kernel',return_value=kernel):
            packed_fa4.register_backend()
            for reuse in (True,False):
                for progress in (.01,.2):
                    a=model(reuse=reuse,curriculum='B',default_prior=0.)
                    b=copy.deepcopy(a);a.progress=b.progress=progress
                    b.set_attn_implementation(packed_fa4.BACKEND)
                    b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                    x=a(**model_inputs(PackedCollator()(rows)))
                    y=b(**model_inputs(PackedCollator(attention_backend='fa4')(rows)))
                    torch.testing.assert_close(x.logits,y.logits,atol=3e-7,rtol=3e-5)
                    x.loss.backward();y.loss.backward()
                    for p,q in zip(a.parameters(),b.parameters()):
                        if p.grad is None:self.assertIsNone(q.grad)
                        else:torch.testing.assert_close(p.grad,q.grad,atol=3e-7,rtol=3e-5)

    def test_sparse_dispatch_matches_independent_dense_outputs_and_gradients(self):
        for branches in (1,2):
            with self.subTest(branches=branches):
                a=model(r=3);b=copy.deepcopy(a)
                sa,sb=a.routed_steps[1],b.routed_steps[1]
                inputs=batch(); h=torch.randn(2,6,16,requires_grad=True)
                rh=h.detach().clone().requires_grad_()
                positions=a.model.rotary_emb(h,inputs['position_ids'])
                # Different choices within the same document/batch.
                choices=torch.stack([(torch.arange(12)+j)%4 for j in range(branches)],1)
                scores=torch.randn(12,branches,requires_grad=True)
                rs=scores.detach().clone().requires_grad_()
                gates=(scores-scores.detach()).exp() if branches==1 else scores.softmax(-1)
                rgates=(rs-rs.detach()).exp() if branches==1 else rs.softmax(-1)
                actual=sa.execute(h,choices,gates,inputs['attention_mask'],positions,None,{})
                expected=dense_step_reference(sb,rh,choices,rgates,inputs['attention_mask'],positions)
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                actual.square().sum().backward();expected.square().sum().backward()
                torch.testing.assert_close(h.grad,rh.grad,atol=2e-6,rtol=1e-5)
                # Mixing full residual states versus deltas cancels the common
                # residual algebraically; fp32 roundoff remains in gate grads.
                torch.testing.assert_close(scores.grad,rs.grad,atol=1e-5,rtol=1e-4)
                for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                    if p.grad is None:self.assertIsNone(q.grad,name)
                    else:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-5,msg=name)

    def test_bf16_autocast_native_and_checkpointed_routing(self):
        a=model(routing='fixed',curriculum='none',step_film=False);b=a.native_reference()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            x=a(**batch());y=b(**batch())
        torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
        x.loss.backward();y.loss.backward()
        for p,q in mapped_pairs(a,b):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=.02)
        for reuse in (True,False):
            for progress in (.01,.2):
                a=model(reuse=reuse,curriculum='B',default_prior=0.)
                b=copy.deepcopy(a);a.progress=b.progress=progress
                b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                seen=[]
                hooks=[step.router.register_forward_hook(lambda m,i,o:seen.append(o.dtype)) for step in b.routed_steps]
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    x=a(**batch());y=b(**batch())
                # The last no-reuse choice is forced and now bypasses its
                # router. Every router that does execute must still be FP32.
                self.assertEqual(seen,[torch.float32]*(4 if reuse else 3))
                torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
                x.loss.backward();y.loss.backward()
                for p,q in zip(a.parameters(),b.parameters()):
                    if p.grad is None:self.assertIsNone(q.grad)
                    else:
                        self.assertTrue(torch.isfinite(p.grad).all())
                        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
                torch.testing.assert_close(a.load_counts,b.load_counts,atol=0,rtol=0)
                for hook in hooks:hook.remove()

    def test_ten_thousand_random_paths_per_mask(self):
        for reuse in (True,False):
            for reach in (0,1,2,7):
                with self.subTest(reuse=reuse,reach=reach):
                    c=copy.deepcopy(model().config)
                    c.num_hidden_layers=10;c.layer_types=['full_attention']*10
                    c.bdot.update(S=8,reuse=reuse,r=reach,routing='random',curriculum='none')
                    m=BDOTForCausalLM(c).eval();m.capture_routes=True
                    with torch.no_grad():m(input_ids=torch.ones(10000,1,dtype=torch.long),use_cache=False)
                    routes=m.last_routes[:,:,0]
                    self.assertTrue(((routes-torch.arange(8)[:,None]).abs()<=reach).all())
                    if not reuse:
                        self.assertTrue((routes.sort(0).values==torch.arange(8)[:,None]).all())


if __name__=='__main__':unittest.main()

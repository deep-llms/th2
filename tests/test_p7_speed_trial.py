"""Independent attention oracle and live-target tests, not a CUDA speed claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.p7_speed_trial import attention_lse, merge_attention, implementation
from deep_kv.proxy_memory import flash_joint_attention
from tests.test_proxy_memory import model, objective
from tests.test_proxy_heads import batch


def reference_kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, causal, softmax_scale, return_lse=False):
    assert causal and torch.equal(cu_seqlens_q, cu_seqlens_k)
    outputs, lses = [], []
    for start, end in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
        query = q[start:end].transpose(0, 1)
        keys = k[start:end].transpose(0, 1).repeat_interleave(q.shape[1]//k.shape[1], 0)
        values = v[start:end].transpose(0, 1).repeat_interleave(q.shape[1]//v.shape[1], 0)
        score = (query @ keys.transpose(-1, -2))*softmax_scale
        score = score.masked_fill(torch.ones(end-start, end-start, dtype=torch.bool).triu(1), -torch.inf)
        outputs.append((score.softmax(-1) @ values).transpose(0, 1))
        lses.append(score.logsumexp(-1))
    return torch.cat(outputs), torch.cat(lses, dim=1) if return_lse else None


class P7SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        # Tests mix ordinary single-backward Trainer runs with retain_graph
        # gradient-route probes. Do not reuse AOT donated-buffer graphs from a
        # previous test; real study jobs run in separate Python processes.
        torch._dynamo.reset()

    def test_joint_output_and_all_five_gradients(self):
        torch.manual_seed(33)
        # Two packed rows, including singleton documents and row boundaries.
        layout = (torch.tensor([0, 1, 4, 7, 9, 13, 14], dtype=torch.int32), 4)
        tensors = [torch.randn(2, h, 7, 8, requires_grad=True) for h in (4, 2, 2, 2, 2)]
        q, k, v, kp, vp = tensors
        reference = flash_joint_attention(q, k, v, kp, vp, layout, .3, reference_kernel)
        n, nl = attention_lse(q, k, v, layout, .3, reference_kernel)
        p, pl = attention_lse(q, kp, vp, layout, .3, reference_kernel)
        merged = merge_attention(n, p, nl, pl)
        torch.testing.assert_close(reference, merged, rtol=2e-6, atol=5e-7)
        adjoint = torch.randn_like(reference)
        gradients = torch.autograd.grad((reference*adjoint).sum(), tensors, retain_graph=True)
        other = torch.autograd.grad((merged*adjoint).sum(), tensors, retain_graph=True)
        for a, b in zip(gradients, other): torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        # An incorrect detached-LSE implementation must NOT pass this test.
        broken = merge_attention(n, p, nl.detach(), pl.detach())
        bad = torch.autograd.grad((broken*adjoint).sum(), q)[0]
        self.assertGreater((bad-gradients[0]).norm().item(), .1)

    def test_causal_and_document_gradient_isolation(self):
        layout = (torch.tensor([0, 3, 4, 8], dtype=torch.int32), 4)
        tensors = [torch.randn(1, h, 8, 8, requires_grad=True) for h in (4, 2, 2, 2, 2)]
        q, k, v, kp, vp = tensors
        n, nl = attention_lse(q, k, v, layout, .3, reference_kernel)
        p, pl = attention_lse(q, kp, vp, layout, .3, reference_kernel)
        out = merge_attention(n, p, nl, pl)
        gradients = torch.autograd.grad(out[:, 5].square().sum(), tensors)
        for gradient in gradients:
            self.assertEqual(gradient[:, :, :4].abs().max().item(), 0)
            self.assertEqual(gradient[:, :, 6:].abs().max().item(), 0)

    def test_compiled_cosine_zero_norms_masked_rows_and_target_routes(self):
        import deep_kv.proxy_estimators as estimators
        torch.manual_seed(18)
        prediction=torch.randn(3, 5, 8); target=torch.randn_like(prediction)
        prediction[0,0]=0; target[0,1]=0
        prediction[1]*=1e-8; target[1]*=1e-8
        valid=torch.ones(3,5,dtype=torch.bool); valid[2]=False
        for target_grad in (False,True):
            captures=[]
            for mode in ('eager','cosine'):
                a=prediction.clone().requires_grad_();b=target.clone().requires_grad_()
                with implementation(mode,backend='aot_eager'):
                    out=estimators.cosine_loss(a,b,valid,target_grad=target_grad,valid_counts=valid.sum(-1))
                    out[1].sum().backward()
                self.assertEqual(b.grad is not None,target_grad)
                self.assertEqual(a.grad[2].abs().max().item(),0)
                captures.append((out,a.grad,b.grad))
            for a,b in zip(*captures):
                if a is None:self.assertIsNone(b)
                else:torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

    def test_complete_model_objectives_buffers_and_optimizer(self):
        with patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {})):
            for checkpoint in (False, True):
                for mode in ('cosine', 'lse', 'both'):
                    captures = []
                    for candidate in ('eager', mode):
                        with implementation(candidate, backend='aot_eager'):
                            m = model('P7-simple-targetgrad', 'fa4', checkpoint)
                            optimizer = torch.optim.AdamW(m.parameters(), lr=.0003)
                            out = m(batch(), collect_target_statistics=True)
                            # Both independent paths matter: auxiliary -> deep
                            # backbone remains live, LM -> predictor stays blocked.
                            named = dict(m.named_parameters())
                            names = [n for n in named if '.w1.' in n or '.mlp.' in n]
                            grads = {}
                            for label, loss in (('lm', out['lm_sum']/out['lm_count']),
                                                ('aux', out['aux_sum']/out['aux_count'])):
                                values = torch.autograd.grad(loss, [named[n] for n in names], allow_unused=True, retain_graph=True)
                                grads.update({label+'/'+n: v.detach().clone() for n, v in zip(names, values) if v is not None})
                            self.assertTrue(any(n.startswith('aux/backbone.') and v.abs().sum()>0 for n, v in grads.items()))
                            self.assertFalse(any(n.startswith('lm/heads.') for n in grads))
                            objective(out).backward()
                            grads.update({n: p.grad.clone() for n, p in named.items() if p.grad is not None})
                            optimizer.step()
                            m.update_statistics(out['center_sums'], out['center_squares'], out['center_counts'])
                            captures.append((out, grads, copy.deepcopy(m.state_dict())))
                    for a, b in zip(*captures):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a:
                            torch.testing.assert_close(a[key], b[key], rtol=4e-4, atol=3e-6, msg=f'{mode}/{checkpoint}/{key}')

    def test_refinements_preserve_model_routes_state_and_optimizer(self):
        with patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {})):
            for checkpointed in (False, True):
                for candidate in ('fused_aux', 'entries', 'kv', 'head_batch', 'target_kernel', 'kernel_both'):
                    captures=[]
                    for mode in ('entries' if candidate in ('head_batch','target_kernel','kernel_both') else 'both', candidate):
                        with implementation(mode, backend='aot_eager'):
                            m=model('P7-simple-targetgrad','fa4',checkpointed)
                            # Nontrivial running statistics and actual clipping.
                            m.mu.fill_(.3);m.sigma2.fill_(.02)
                            optimizer=torch.optim.AdamW(m.parameters(),lr=.0003)
                            out=m(batch(),collect_target_statistics=True)
                            named=dict(m.named_parameters())
                            gradients={}
                            for label,loss in (('lm',out['lm_sum']/out['lm_count']),('aux',out['aux_sum']/out['aux_count'])):
                                values=torch.autograd.grad(loss,list(named.values()),allow_unused=True,retain_graph=True)
                                gradients.update({label+'/'+n:v.clone() for n,v in zip(named,values) if v is not None})
                            self.assertFalse(any(n.startswith('lm/heads.') and '.w1.' in n for n in gradients))
                            self.assertTrue(any(n.startswith('aux/backbone.') and v.abs().sum()>0 for n,v in gradients.items()))
                            self.assertTrue(any(n.startswith('lm/heads.') and '.k_proj.' in n for n in gradients))
                            objective(out).backward();optimizer.step()
                            m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'])
                            with torch.no_grad():
                                inference=m.hidden_states(batch())[0]
                                # Stats-only/bootstrap and disabled clipping paths.
                                statistics=m(batch(),compute_auxiliary_losses=False,collect_target_statistics=True)
                                without_clips=m(batch(),collect_target_statistics=True,collect_clip_statistics=False)
                            captures.append((out,gradients,copy.deepcopy(m.state_dict()),dict(hidden=inference),statistics,without_clips))
                    for a,b in zip(*captures):
                        self.assertEqual(a.keys(),b.keys())
                        for key in a:
                            torch.testing.assert_close(a[key],b[key],rtol=5e-4,atol=5e-6,msg=candidate+'/'+key)

    def test_fused_normalization_clipping_and_small_norm_gradients(self):
        from scripts.p7_fusion_trial import normalized_cosine
        from deep_kv.proxy_estimators import cosine_loss
        torch.manual_seed(71)
        pred=torch.randn(3,5,8);center=torch.randn_like(pred)*5
        pred[0,0]=0;center[0,1]=0;pred[1]*=1e-8;center[1]*=1e-8
        # Include points exactly at the clipping threshold.
        center[0,2]=3.;center[0,3]=-3.
        inverse=torch.ones(8);valid=torch.ones(3,5,dtype=torch.bool);valid[2]=False
        fn=torch.compile(normalized_cosine,backend='aot_eager',fullgraph=True)
        for live in (False,True):
            captures=[]
            for fused in (False,True):
                p=pred.clone().requires_grad_();c=center.clone().requires_grad_()
                outputs=(fn(p,c,inverse,valid,valid.sum(-1),3.,live) if fused else
                         cosine_loss(p,(c*inverse).clamp(-3,3),valid,target_grad=live,valid_counts=valid.sum(-1)))
                outputs[1].sum().backward();captures.append((outputs,p.grad,c.grad))
                self.assertEqual(c.grad is not None,live)
            for a,b in zip(*captures):
                if a is None:self.assertIsNone(b)
                else:torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

    def test_production_flag_saved_resume_and_legacy_compatibility(self):
        from safetensors.torch import load_file
        from tests.test_proxy_memory import config
        from tests.test_proxy_training import proxy_fixture
        from tests.test_train import invoke
        from deep_kv.proxy import resolve_proxy_settings, ProxySettings
        from deep_kv.report import comparable_config
        import train
        before=dict(pilot=dict(arm='P7-simple-targetgrad'),training={})
        after=copy.deepcopy(before);after['pilot']['proxy_compile_auxiliary']=False
        self.assertEqual(train.resume_performance_changes(before,after),{})
        after['pilot']['proxy_compile_auxiliary']=True
        with self.assertRaisesRegex(ValueError,'allow_performance_change'):
            train.resume_performance_changes(before,after)
        self.assertEqual(set(train.resume_performance_changes(before,after,True)),{'proxy_compile_auxiliary'})
        entries=copy.deepcopy(before);entries['pilot']['proxy_compile_entries']=True
        with self.assertRaisesRegex(ValueError,'allow_performance_change'):
            train.resume_performance_changes(before,entries)
        self.assertEqual(set(train.resume_performance_changes(before,entries,True)),{'proxy_compile_entries'})
        for arm in ('P7','P6-iso','P7-simple'):
            with self.assertRaisesRegex(ValueError,'validated only'):
                resolve_proxy_settings(arm,ProxySettings(compile_auxiliary=True))
            with self.assertRaisesRegex(ValueError,'validated only'):
                resolve_proxy_settings(arm,ProxySettings(compile_entries=True))
        compiler=torch.compile
        def cpu_compile(fn,**kwargs):
            return compiler(fn,**{**kwargs,'backend':'aot_eager'})
        with tempfile.TemporaryDirectory() as tmp, patch('torch.compile',side_effect=cpu_compile), patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):
            root=Path(tmp);recipe=proxy_fixture(root);config().save_pretrained(root/'model')
            recipe.update(proxy_groups=2,arm='P7-simple-targetgrad',proxy_compile_auxiliary=True,
                          proxy_lse_attention=True,proxy_compile_entries=True,attention_backend='fa4')
            invoke(root,{**recipe,'stop_after':2})
            saved=json.loads((root/'run/train_config.json').read_text())
            self.assertTrue(saved['pilot']['proxy_compile_auxiliary'])
            self.assertTrue(saved['pilot']['proxy_lse_attention'])
            self.assertTrue(saved['pilot']['proxy_compile_entries'])
            self.assertNotIn('proxy_compile_auxiliary',comparable_config(saved)['pilot'])
            self.assertNotIn('proxy_compile_entries',comparable_config(saved)['pilot'])
            invoke(root,recipe)
            invoke(root,{**recipe,'output_dir':str(root/'full')})
            resumed=load_file(root/'run/model.safetensors');full=load_file(root/'full/model.safetensors')
            self.assertEqual(resumed.keys(),full.keys())
            self.assertFalse(any('_orig_mod' in name for name in full))
            for name in full:torch.testing.assert_close(resumed[name],full[name],rtol=0,atol=0,msg=name)

    def test_production_flags_match_validated_candidates(self):
        compiler=torch.compile
        def cpu_compile(fn,**kwargs):return compiler(fn,**{**kwargs,'backend':'aot_eager'})
        with patch('torch.compile',side_effect=cpu_compile), patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{})):
            for mode in ('cosine','lse','both','entries'):
                captures=[]
                for production in (False,True):
                    settings=(dict(compile_auxiliary=mode in ('cosine','both','entries'),lse_attention=mode in ('lse','both','entries'),compile_entries=mode=='entries')
                              if production else {})
                    with implementation('eager' if production else mode,backend='aot_eager'):
                        m=model('P7-simple-targetgrad','fa4',**settings)
                        out=m(batch(),collect_target_statistics=True);objective(out).backward()
                        captures.append((out,{n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}))
                for a,b in zip(*captures):
                    self.assertEqual(a.keys(),b.keys())
                    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,msg=mode+'/'+key)


if __name__ == '__main__': unittest.main()

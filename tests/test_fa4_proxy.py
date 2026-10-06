"""Independent per-document CPU oracle for FA4 proxy plumbing; CUDA is tested separately."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from deep_kv import PROXY_ARMS
from deep_kv.model import Context
from deep_kv.proxy import ProxyModel
from tests.test_fa4_baseline import reference_kernel
from tests.test_proxy_heads import cfg, settings, batch
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


def make(arm, backend, checkpoint=False):
    return ProxyModel.from_scratch(cfg(),arm,consumer=2,deep_target=8,proxy_settings=settings(),
        attention_backend=backend,checkpoint_layers=checkpoint,checkpoint_lm=checkpoint,
        checkpoint_aux=checkpoint,lm_chunk=3)


class FA4ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        mock=patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'}))
        mock.start();self.addCleanup(mock.stop)

    def test_all_arms_active_gates_losses_gradients_statistics_and_checkpoint_replay(self):
        for arm in PROXY_ARMS:
            for checkpoint in (False,True):
                with self.subTest(arm=arm,checkpoint=checkpoint):
                    dense=make(arm,'sdpa',checkpoint);flash=make(arm,'fa4',checkpoint)
                    with torch.no_grad():
                        for head in dense.heads.values():head.alpha.normal_(0,.2)
                        if dense.family:
                            dense.mu.normal_(0,.01);dense.sigma2.uniform_(.1,1);dense.mu_initialized.fill_(True)
                    flash.load_state_dict(dense.state_dict(),strict=True)
                    outputs=[]
                    for m in (dense,flash):
                        original={k:v.clone() for k,v in m.named_buffers()}
                        with patch.object(Context,'allowed',side_effect=AssertionError('dense mask')) if m is flash else nullcontext():
                            out=m(batch(),collect_target_statistics=True,auxiliary_grad=not arm.endswith('lambda0'))
                            weight=m.auxiliary_weight(250)
                            loss=out['lm_sum']/out['lm_count']+weight*out['aux_sum']/out['aux_count'].clamp_min(1)
                            loss.backward()
                        outputs.append(out)
                        for k,v in m.named_buffers():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
                    for k in ('lm_sum','aux_sum','center_sums','center_squares','center_counts','statistics'):
                        torch.testing.assert_close(outputs[0][k],outputs[1][k],rtol=2e-4,atol=2e-5,msg=arm+'/'+k)
                    for (name,p),(_,q) in zip(dense.named_parameters(),flash.named_parameters()):
                        self.assertEqual(p.grad is None,q.grad is None,name)
                        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=3e-6,msg=arm+'/'+name)
                    for name,p in flash.heads.named_parameters():
                        self.assertIsNotNone(p.grad,name)
                        self.assertGreater(p.grad.abs().sum().item(),0,name)

    def test_active_proxy_document_isolation_and_auxiliary_routing(self):
        for arm in ('P1-block','P1-flow','P3-block','P3-flow','P4-4h','P4-iso-4h'):
            m=make(arm,'fa4');ctx=batch()
            with torch.no_grad():
                for head in m.heads.values():head.alpha.fill_(.2)
            captures=[]
            handle=m.backbone.model.embed_tokens.register_forward_hook(lambda mod,args,out:captures.append(out))
            try:
                hidden=m.hidden_states(ctx)[0]
                gradient=torch.autograd.grad(hidden[:,3:].square().sum(),captures[-1])[0]
                self.assertEqual(gradient[:,:3].abs().max().item(),0,arm)
                altered=copy.deepcopy(ctx);altered.input_ids[:,:3]=20
                torch.testing.assert_close(hidden[:,3:],m.hidden_states(altered)[0][:,3:],rtol=0,atol=0)
                m.zero_grad(set_to_none=True);out=m(ctx);out['aux_sum'].backward()
                backbone=[p.grad for p in m.backbone.parameters() if p.grad is not None]
                if arm.endswith('block') or arm.startswith('P4'):self.assertFalse(backbone)
                else:self.assertGreater(sum(g.abs().sum().item() for g in backbone),0)
            finally:handle.remove()

    def test_four_head_p4_backend_and_separate_gradient_routes(self):
        config=cfg();config.num_attention_heads=16;config.num_key_value_heads=8
        for arm in ('P4-4h','P4-iso-4h'):
            models=[ProxyModel.from_scratch(config,arm,consumer=2,deep_target=8,
                proxy_settings=settings(),attention_backend=backend,lm_chunk=3)
                for backend in ('sdpa','fa4')]
            self.assertEqual(models[0].value_groups,2)
            self.assertEqual(models[0].settings.alpha_init,1)
            for component in ('lm','aux','combined'):
                outputs=[]
                for m in models:
                    m.zero_grad(set_to_none=True)
                    with patch.object(Context,'allowed',side_effect=AssertionError('dense mask')) if m is models[1] else nullcontext():
                        out=m(batch());lm=out['lm_sum']/out['lm_count'];aux=out['aux_sum']/out['aux_count']
                        loss={'lm':lm,'aux':aux,'combined':lm+.1*aux}[component];loss.backward()
                    outputs.append(loss.detach())
                    for name,p in m.named_parameters():
                        estimator=name.startswith('heads.') and not name.endswith('.alpha')
                        active=p.grad is not None and bool(p.grad.count_nonzero())
                        if component=='aux':self.assertEqual(active,estimator,name)
                        if component=='lm' and estimator:self.assertEqual(active,'iso' not in arm,name)
                torch.testing.assert_close(*outputs,rtol=1e-5,atol=1e-6)
                for (name,p),(_,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=3e-6,msg=arm+'/'+component+'/'+name)

    def test_actual_trainer_proxy_normalization_and_exact_resume(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=proxy_fixture(root)
            for arm in ('P1-block','P3-flow','P3-lambda0','P4-4h','P4-iso-4h'):
                args={**config,'arm':arm,'attention_backend':'fa4','output_dir':str(root/arm)}
                invoke(root,{**args,'stop_after':2})
                before=load_file(root/arm/'checkpoint-2/model.safetensors')
                self.assertTrue(before['mu_initialized']);self.assertGreater(before['sigma2'].sum(),0)
                invoke(root,args);invoke(root,{**args,'output_dir':str(root/('full-'+arm))})
                resumed=load_file(root/arm/'model.safetensors');full=load_file(root/('full-'+arm)/'model.safetensors')
                for key,value in full.items():torch.testing.assert_close(resumed[key],value,rtol=0,atol=0,msg=arm+'/'+key)
                self.assertFalse(torch.equal(before['mu'],resumed['mu']))
                self.assertEqual(json.loads((root/arm/'result.json').read_text())['attention_runtime']['backend'],'fa4')
                with self.assertRaisesRegex(ValueError,'Resume configuration'):
                    invoke(root,{**args,'attention_backend':'sdpa','allow_performance_change_on_resume':True})


if __name__=='__main__':unittest.main()

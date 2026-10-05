"""FA4 baseline plumbing against independent CPU attention, not CUDA validation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from deep_kv.fa4 import document_layout, trainer_audit, load_kernel
from deep_kv.model import Context
from deep_kv.proxy import ProxyModel
from deep_kv.__main__ import jobs
from deep_kv.packing import isolated_data_collator
from deep_kv.proxy_training import ProxyTrainer
from tests.test_proxy_heads import cfg, settings, batch
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke
import train


def reference_kernel(q,k,v,*,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,causal,softmax_scale):
    # Separate attention problem for each fragment, with native GQA.
    assert causal and cu_seqlens_q.dtype == torch.int32
    assert torch.equal(cu_seqlens_q,cu_seqlens_k)
    assert max_seqlen_q == max_seqlen_k == int((cu_seqlens_q[1:]-cu_seqlens_q[:-1]).max())
    parts=[]
    for a,b in zip(cu_seqlens_q[:-1].tolist(),cu_seqlens_q[1:].tolist()):
        parts.append(F.scaled_dot_product_attention(q[a:b].transpose(0,1)[None],
            k[a:b].transpose(0,1)[None],v[a:b].transpose(0,1)[None],is_causal=True,
            enable_gqa=True,scale=softmax_scale)[0].transpose(0,1))
    return torch.cat(parts),None


def make_model(backend,checkpoint=False):
    return ProxyModel.from_scratch(cfg(),'A',consumer=2,deep_target=8,proxy_settings=settings(),
        attention_backend=backend,checkpoint_layers=checkpoint,checkpoint_lm=checkpoint,lm_chunk=3)


class FA4BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.kernel=patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-test-double'}))
        self.kernel.start();self.addCleanup(self.kernel.stop)

    def test_layout_row_boundaries_singletons_and_positions(self):
        rows=[dict(input_ids=[3,31,4,31,31,5,6,31],labels=[3,31,4,31,31,5,6,31],
                   attention_mask=[1]*8,segments=[7,7,7,8,9,10,10,10])]*2
        ctx=ProxyTrainer.context(isolated_data_collator(rows))
        cu,maximum=document_layout(ctx)
        self.assertEqual(cu.tolist(),[0,3,4,5,8,11,12,13,16])
        self.assertEqual(maximum,3)
        self.assertEqual(ctx.position_ids[0].tolist(),[0,1,2,0,0,0,1,2])
        self.assertEqual(ctx.targets()[0].tolist(),[True,True,False,False,False,True,True])
        ctx.valid[0,0]=False
        with self.assertRaisesRegex(ValueError,'fully packed'):document_layout(ctx)

    def test_dense_equivalence_gradients_checkpointing_and_no_dense_mask(self):
        ctx=batch()
        for checkpoint in (False,True):
            dense=make_model('sdpa',checkpoint);flash=make_model('fa4',checkpoint)
            for name,tensor in dense.state_dict().items():
                torch.testing.assert_close(tensor,flash.state_dict()[name],rtol=0,atol=0)
            a=dense(ctx)
            with patch.object(Context,'allowed',side_effect=AssertionError('dense mask allocated')):
                b=flash(ctx)
                (b['lm_sum']/b['lm_count']).backward()
            torch.testing.assert_close(a['lm_sum'],b['lm_sum'],rtol=1e-6,atol=1e-5)
            self.assertEqual(a['lm_count'],b['lm_count'])
            (a['lm_sum']/a['lm_count']).backward()
            for (name,p),(_,q) in zip(dense.named_parameters(),flash.named_parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6,msg=name)

    def test_no_cross_document_output_or_gradient_leakage(self):
        m=make_model('fa4');ctx=batch();captured=[]
        handle=m.backbone.model.embed_tokens.register_forward_hook(lambda mod,args,out: captured.append(out))
        try:
            output=m.hidden_states(ctx)[0];embedding=captured[-1]
            gradient=torch.autograd.grad(output[:,3:].square().sum(),embedding)[0]
            self.assertEqual(gradient[:,:3].abs().max().item(),0.)
            altered=copy.deepcopy(ctx);altered.input_ids[:,:3]=20
            torch.testing.assert_close(output[:,3:],m.hidden_states(altered)[0][:,3:],rtol=0,atol=0)
        finally:handle.remove()

    def test_actual_entry_save_resume_and_backend_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=proxy_fixture(root)
            config.update(arm='A',attention_backend='fa4',output_dir=str(root/'A'))
            invoke(root,{**config,'stop_after':2});invoke(root,config)
            invoke(root,{**config,'output_dir':str(root/'full')})
            from safetensors.torch import load_file
            resumed=load_file(root/'A/model.safetensors');full=load_file(root/'full/model.safetensors')
            for name,value in full.items():torch.testing.assert_close(resumed[name],value,rtol=0,atol=0)
            result=json.loads((root/'A/result.json').read_text())
            self.assertEqual(result['global_step'],3)
            self.assertEqual(result['attention_runtime']['backend'],'fa4')
            self.assertEqual(result['sdpa_receipts'],[])
            with self.assertRaisesRegex(ValueError,'Resume configuration'):
                invoke(root,{**config,'attention_backend':'sdpa','allow_performance_change_on_resume':True})

    def test_queue_and_scope_guards(self):
        manifest=jobs('baseline_a_fa4.b200.json',seeds=[42])
        training=[job for job in manifest['jobs'] if 'gpus' in job]
        self.assertEqual(len(training),1)
        self.assertIn('--attention_backend',training[0]['argv'])
        self.assertEqual(training[0]['gpus'],list(range(8)))
        with self.assertRaisesRegex(ValueError,'only arm A'):
            jobs('baseline_a_fa4.b200.json',arms=('A','P1-block'))
        with self.assertRaisesRegex(ValueError,'do not reuse'):
            jobs('baseline_a_fa4.b200.json',reuse_baselines={42:'/old/A'})
        with self.assertRaisesRegex(ValueError,'only for vanilla arm A'):
            ProxyModel.from_scratch(cfg(),'P1-block',consumer=2,deep_target=8,
                proxy_settings=settings(),attention_backend='fa4')
        # Dense recipes omit the default field, preserving historical resume identity.
        previous=dict(pilot=dict(arm='A'),training={})
        requested=copy.deepcopy(previous);requested['pilot']['attention_backend']='fa4'
        with self.assertRaisesRegex(ValueError,'Resume configuration'):
            train.resume_performance_changes(previous,requested,True)

    def test_forward_backward_receipt_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=make_model('fa4')
            trainer=SimpleNamespace(model=m,state=SimpleNamespace(global_step=0),
                args=SimpleNamespace(process_index=0,world_size=1,output_dir=tmp))
            with trainer_audit(trainer,'train'):
                out=m(batch());(out['lm_sum']/out['lm_count']).backward()
            receipt=json.loads((Path(tmp)/trainer._fa4_receipts[0]).read_text())
            self.assertEqual(receipt['calls'],8);self.assertEqual(receipt['query_gradient_calls'],8)
            self.assertFalse(receipt['dense_mask']);self.assertIsNone(m._fa4_observer)
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with trainer_audit(trainer,'eval'):raise RuntimeError('intentional')
            self.assertIsNone(m._fa4_observer)

    def test_missing_package_fails_without_fallback(self):
        from importlib.metadata import PackageNotFoundError
        with patch('deep_kv.fa4.version',side_effect=PackageNotFoundError('flash-attn-4')):
            with self.assertRaisesRegex(RuntimeError,'no SDPA fallback'):load_kernel()

    def test_launch_gate_checks_recipe_and_full_resume_artifacts(self):
        from scripts.validate_fa4_baseline import validate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/'fa4';old=root/'dense';arm.mkdir();old.mkdir()
            config=dict(pilot=dict(arm='A'),training=dict(logging_steps=10,learning_rate=.0003),
                        train_fingerprint='same',eval_fingerprint='same')
            (old/'train_config.json').write_text(json.dumps(config))
            config['pilot']['attention_backend']='fa4';config['training']['logging_steps']=1
            (arm/'train_config.json').write_text(json.dumps(config))
            state=dict(global_step=3,max_steps=28600,log_history=[dict(loss=12.,grad_norm=1.)]*3)
            (arm/'trainer_state.json').write_text(json.dumps(state))
            evaluation=dict(eval_lm_loss=12.)
            (arm/'eval_results.json').write_text(json.dumps(evaluation))
            result=dict(arm='A',status='stopped',global_step=3,schedule_steps=28600,input_tokens=3*1048576,
                evaluation=evaluation,sdpa_receipts=[],fa4_receipts=['fa4-train.json','fa4-eval.json'],
                attention_runtime=dict(backend='fa4',version='4.0.0b33'),
                training_cost=dict(peak_cuda_allocated_bytes=100*2**30))
            (arm/'result.json').write_text(json.dumps(result));(arm/'model.safetensors').write_bytes(b'fixture')
            checkpoint=arm/'checkpoint-3';checkpoint.mkdir()
            for name in ['optimizer.pt','scheduler.pt','model.safetensors',*[f'rng_state_{i}.pth' for i in range(8)]]:
                (checkpoint/name).write_bytes(b'fixture')
            (checkpoint/'trainer_state.json').write_text(json.dumps(state))
            for phase in ('train','eval'):
                (arm/f'fa4-{phase}.json').write_text(json.dumps(dict(phase=phase,calls=28,world_size=8,
                    kernel_dtype='torch.bfloat16',document_isolation=True,causal=True,dense_mask=False,
                    positions='reset_per_document',query_gradient_calls=28 if phase=='train' else 0)))
            validate(arm,old,3,root/'passed.json')
            self.assertEqual(json.loads((root/'passed.json').read_text())['status'],'passed')
            config['training']['learning_rate']=.001
            (arm/'train_config.json').write_text(json.dumps(config))
            with self.assertRaises(AssertionError):validate(arm,old,3,root/'bad.json')
            config['training']['learning_rate']=.0003
            (arm/'train_config.json').write_text(json.dumps(config))
            (checkpoint/'rng_state_7.pth').unlink()
            with self.assertRaises(FileNotFoundError):validate(arm,old,3,root/'bad.json')


if __name__=='__main__':unittest.main()

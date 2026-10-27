import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import DynamicCache
from efficient_llm.kv_mlp import CompressedQwen, convert, load_compressed
from efficient_llm.kv_mlp_check import reference, cache_bytes
from efficient_llm.kv_mlp_study import score
from efficient_llm.kv_placement import (apply_layout, layout_sources, LAYOUTS, summarize,
    train_affected_qkv, parameter_fingerprint, summarize_qkv)
from efficient_llm.diagnostic_data import sha256
from tests.efficient.test_kv_mlp import tiny


class PlacementTests(unittest.TestCase):
    def test_qkv_scope_preserves_forward_and_frozen_parameters(self):
        config=tiny().config;config.num_hidden_layers=8;config.layer_types=['full_attention']*8
        config.kv_sources=list(range(8));config.mlp_widths=[config.intermediate_size]*8
        model=apply_layout(CompressedQwen._from_config(config,attn_implementation='sdpa'),'uniform25')
        ids=torch.arange(1,19).reshape(2,9)%47
        with torch.no_grad():expected=model(ids,use_cache=False).logits
        self.assertEqual(train_affected_qkv(model),[2,3,6,7])
        actual={n for n,p in model.named_parameters() if p.requires_grad}
        names={f'model.layers.{i}.self_attn.{proj}.{suffix}'
               for i in (2,3,6,7)
               for proj in (('q_proj','k_proj','v_proj') if i%2==0 else ('q_proj',))
               for suffix in ('weight','bias')}
        self.assertEqual(actual,names)
        frozen=parameter_fingerprint(model,trainable=False)
        trained=parameter_fingerprint(model,trainable=True)
        logits=model(ids,use_cache=False).logits
        torch.testing.assert_close(logits,expected,rtol=0,atol=0)
        logits.square().mean().backward()
        for name,p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            else:self.assertIsNone(p.grad,name)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-3)
        optimizer.step()
        self.assertEqual(parameter_fingerprint(model,trainable=False),frozen)
        self.assertNotEqual(parameter_fingerprint(model,trainable=True),trained)
        with self.assertRaises(ValueError):train_affected_qkv(tiny())

    def test_layout_reference_gradients_cache_and_reload(self):
        config=tiny().config;config.num_hidden_layers=8;config.layer_types=['full_attention']*8
        config.kv_sources=list(range(8));config.mlp_widths=[config.intermediate_size]*8
        torch.manual_seed(41)
        dense=CompressedQwen._from_config(config,attn_implementation='sdpa').eval()
        ids=torch.arange(1,19).reshape(2,9)%47
        self.assertEqual(layout_sources(8,'uniform25'),[0,1,2,2,4,5,6,6])
        self.assertEqual(layout_sources(8,'upper25'),[0,1,2,3,4,4,6,6])
        for layout in LAYOUTS:
            model=apply_layout(deepcopy(dense),layout)
            ref=deepcopy(model)
            actual=model(ids,use_cache=False).logits;expected=reference(ref,ids)
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)
            actual.square().sum().backward();expected.square().sum().backward()
            for (name,a),(other,b) in zip(model.named_parameters(),ref.named_parameters()):
                self.assertEqual(name,other);self.assertIsNotNone(a.grad)
                torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=2e-5,msg=name)
            ga=torch.cat([p.grad.flatten() for p in model.parameters()])
            gb=torch.cat([p.grad.flatten() for p in ref.parameters()])
            self.assertLess(float((ga-gb).norm()/gb.norm()),1e-5)
            model.zero_grad(set_to_none=True);ref.zero_grad(set_to_none=True)
            same=reference(ref,ids,sdpa=True)
            direct=model(ids,use_cache=False).logits
            torch.testing.assert_close(direct,same,rtol=0,atol=0)
            direct.square().sum().backward();same.square().sum().backward()
            for (name,a),(_,b) in zip(model.named_parameters(),ref.named_parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7,msg=name)
            with torch.no_grad():
                cache=DynamicCache(config=model.config)
                parts=[model(ids[:,:4],past_key_values=cache,use_cache=True).logits]
                for t in range(4,9):parts.append(model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits)
                torch.testing.assert_close(torch.cat(parts,1),actual,rtol=1e-4,atol=1e-6)
                active=len(set(model.config.kv_sources))
                self.assertEqual(cache_bytes(cache),active*2*2*9*config.num_key_value_heads*config.head_dim*4)
                if layout=='full50':
                    torch.testing.assert_close(actual,convert(deepcopy(dense),share=True)(ids,use_cache=False).logits,rtol=0,atol=0)
                with tempfile.TemporaryDirectory() as tmp:
                    model.save_pretrained(tmp)
                    torch.testing.assert_close(load_compressed(tmp,'cpu')(ids,use_cache=False).logits,actual,rtol=0,atol=0)
            with self.assertRaises(ValueError):apply_layout(model,layout)
        with self.assertRaises(ValueError):layout_sources(6,'upper25')

    def test_summary_matches_parent_recipe_and_data(self):
        import math
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            keys=['learning_rate','lr_scheduler_type','warmup_steps','weight_decay',
                  'max_grad_norm','per_device_train_batch_size','gradient_accumulation_steps',
                  'seed','data_seed','bf16']
            base=dict(status='passed',arm='A',smoke=False,phase='new_recovery',
                steps=2048,world_size=8,input_tokens=134217728,
                sample_order_sha256_per_rank=['same']*8,probe_sha256='same',parent_steps=512,
                parent_result_sha256='dense-parent',initial_model_files={'weights':'hash'},
                kv_sources=list(range(8)),mlp_widths=[32]*8,training_seconds=10.,
                training_arguments=dict.fromkeys(keys,1),
                inference_validation=dict(ppl=math.exp(3.),per_document_nll=[3.]*8))
            baseline=root/'A.json';baseline.write_text(json.dumps(base))
            for layout in LAYOUTS:
                report=deepcopy(base);loss=3.1 if layout=='full50' else 3.02
                report.update(arm='B',phase='placement_conversion',sharing_layout=layout,
                    kv_sources=layout_sources(8,layout),
                    inference_validation=dict(ppl=math.exp(loss),per_document_nll=[loss]*8))
                (root/layout).mkdir();(root/layout/'result.json').write_text(json.dumps(report))
            summarize(root,baseline,root/'summary.json')
            result=json.loads((root/'summary.json').read_text())['comparisons']
            self.assertFalse(result['full50']['within_five_percent'])
            self.assertTrue(result['upper25']['within_five_percent'])
            self.assertEqual(result['uniform25']['cache_fraction'],.75)
            path=root/'upper25/result.json';report=json.loads(path.read_text())
            for key,value in [('parent_result_sha256','wrong'),('sample_order_sha256_per_rank',['wrong']*8)]:
                wrong=deepcopy(report);wrong[key]=value;path.write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError,'Unmatched placement'):
                    summarize(root,baseline,root/'summary.json')
            wrong=deepcopy(report);wrong['training_arguments']['seed']=2
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'Unmatched recovery recipe'):
                summarize(root,baseline,root/'summary.json')

    def test_qkv_summary_rejects_changed_frozen_weights(self):
        import math
        keys=['learning_rate','lr_scheduler_type','warmup_steps','weight_decay',
              'max_grad_norm','per_device_train_batch_size','gradient_accumulation_steps',
              'seed','data_seed','bf16']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=dict(status='passed',arm='A',smoke=False,phase='new_recovery',steps=2048,
                world_size=8,input_tokens=134217728,sample_order_sha256_per_rank=['same']*8,
                probe_sha256='same',parent_steps=512,parent_result_sha256='parent',
                initial_model_files={'weights':'same'},training_arguments=dict.fromkeys(keys,1),
                inference_validation=dict(ppl=math.exp(3.),per_document_nll=[3.]*8))
            c=deepcopy(a);c.update(arm='B',phase='placement_conversion',sharing_layout='uniform25',
                parameters=1000,kv_sources=layout_sources(8,'uniform25'),mlp_widths=[32]*8,
                validation={'0':{'per_document_nll':[4.]*8}})
            q=deepcopy(c);q.update(trainable_scope='affected_qkv',training_seconds=1.,
                adaptation=dict(trainable_parameters=100,frozen_before='same',frozen_after='same',
                    trainable_before='old',trainable_after='new'))
            for name,value in [('a',a),('c',c),('q',q)]:
                (root/(name+'.json')).write_text(json.dumps(value))
            summarize_qkv(root/'q.json',root/'c.json',root/'a.json',root/'summary.json')
            self.assertTrue(json.loads((root/'summary.json').read_text())['frozen_weights_unchanged'])
            q['adaptation']['frozen_after']='changed';(root/'q.json').write_text(json.dumps(q))
            with self.assertRaisesRegex(ValueError,'weight proof'):
                summarize_qkv(root/'q.json',root/'c.json',root/'a.json',root/'summary.json')

    def test_two_rank_dense_parent_conversion(self):
        self._two_rank_conversion(qkv=False)

    def test_two_rank_qkv_only_conversion(self):
        self._two_rank_conversion(qkv=True)

    def _two_rank_conversion(self, qkv):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=tiny();original=root/'model';model.save_pretrained(original)
            parent=root/'parent';model.save_pretrained(parent/'final')
            probe=root/'probe';probe.mkdir()
            torch.manual_seed(8)
            tokens=dict(calibration=torch.randint(0,47,(8,17)),evaluation=torch.randint(0,47,(4,17)))
            torch.save(tokens,probe/'tokens.pt')
            manifest=dict(status='passed',length=16,calibration_documents=8,evaluation_documents=4,
                tokens_sha256=sha256(probe/'tokens.pt'),evaluation_reused=True,
                recovery_from_manifest_sha256='prior-probe',
                model_files={p.name:sha256(p) for p in original.iterdir() if p.is_file()})
            (probe/'manifest.json').write_text(json.dumps(manifest))
            # Measured dense parent scores form the fixture. No optimizer state
            # is inherited: the integration under test is the new conversion phase.
            report=dict(status='passed',smoke=False,arm='A',steps=512,probe_sha256='prior-probe',
                kv_sources=model.config.kv_sources,mlp_widths=model.config.mlp_widths,
                parameters=sum(p.numel() for p in model.parameters()),
                validation={'512':score(model,tokens['evaluation'])})
            (parent/'result.json').write_text(json.dumps(report))
            before=sha256(parent/'final/model.safetensors')
            done=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node=2','-m','efficient_llm.kv_mlp_study','train',
                '--model',str(original),'--probe',str(probe),'--initial-model',str(parent/'final'),
                '--parent-result',str(parent/'result.json'),'--device','cpu','--arm','B',
                '--calibration','0','--sharing-layout','uniform25' if qkv else 'upper25',
                '--steps','2','--batch','2','--eval-every','1','--output',str(root/'fit')]
                + (['--qkv-only'] if qkv else []),capture_output=True,text=True,timeout=180)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            actual=json.loads((root/'fit/result.json').read_text())
            self.assertEqual(actual['phase'],'placement_conversion')
            self.assertEqual(actual['kv_sources'],[0,1,2,2])
            self.assertEqual(actual['initial_max_nll_difference'],0)
            self.assertEqual(actual['reloaded_max_nll_difference'],0)
            self.assertEqual(actual['unique_documents'],8)
            self.assertEqual(sha256(parent/'final/model.safetensors'),before)
            self.assertNotEqual(sha256(root/'fit/final/model.safetensors'),before)
            self.assertTrue((root/'fit/trainer/checkpoint-2/optimizer.pt').is_file())
            if qkv:
                self.assertEqual(actual['trainable_scope'],'affected_qkv')
                proof=actual['adaptation']
                self.assertEqual(proof['affected_layers'],[2,3])
                self.assertEqual(proof['frozen_before'],proof['frozen_after'])
                self.assertNotEqual(proof['trainable_before'],proof['trainable_after'])
                # Compare actual tensors, independently of the fingerprint report.
                final=load_compressed(root/'fit/final','cpu')
                parent_model=apply_layout(load_compressed(parent/'final','cpu'),'uniform25')
                trained=set(proof['trainable_names']);changed=[]
                for name,p in final.named_parameters():
                    other=dict(parent_model.named_parameters())[name]
                    if name in trained:changed.append(not torch.equal(p,other))
                    else:torch.testing.assert_close(p,other,rtol=0,atol=0,msg=name)
                self.assertTrue(any(changed))


if __name__=='__main__':unittest.main()

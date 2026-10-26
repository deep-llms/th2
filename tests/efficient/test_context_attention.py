import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,DynamicCache,StaticCache
from efficient_llm.context_attention import ContextPolicy,local_attention
from efficient_llm.routers import RouterPair
from efficient_llm.context_check import check
from efficient_llm.context_study import static_policies
from efficient_llm.diagnostic_data import sha256


def model():
    torch.manual_seed(17)
    return Qwen3ForCausalLM._from_config(Qwen3Config(vocab_size=53,hidden_size=32,
        intermediate_size=64,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=4,
        head_dim=8,attention_bias=True,attention_dropout=0.,max_position_embeddings=128),
        attn_implementation='sdpa').eval().requires_grad_(False)


class ContextTests(unittest.TestCase):
    def test_full_gate_and_static_budget(self):
        m=model();ids=torch.arange(1,25)[None,:]%53
        errors=check(m,ids)
        self.assertLess(max(errors.values()),2e-3)
        totals=torch.tensor([[1.,2.],[3.,2.]])
        costs=torch.rand(2,2,6)
        masks,scores=static_policies(costs,totals,4,2)
        for value in masks.values():self.assertEqual(value.sum().item(),4)
        self.assertEqual(masks['fixed_layers'].tolist(),[[False]*4,[True]*4])

    def test_real_two_rank_context_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'model';probe=root/'probe';probe.mkdir()
            model().save_pretrained(path);torch.manual_seed(4)
            torch.save(dict(calibration=torch.randint(0,53,(8,17)),evaluation=torch.randint(0,53,(4,17))),probe/'tokens.pt')
            (probe/'manifest.json').write_text(json.dumps(dict(status='passed',length=16,seed=42,
                calibration_documents=8,evaluation_documents=4,tokens_sha256=sha256(probe/'tokens.pt'),
                model_files={p.name:sha256(p) for p in path.iterdir() if p.is_file()})))
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 '-m','efficient_llm.context_study','--model',str(path),'--probe',str(probe),
                 '--output',str(root/'run'),'--device','cpu','--keep','2','--window','5','--sinks','2',
                 '--stride','2','--steps','8','--feature-batch','4']
            done=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            train=json.loads((root/'run/training.json').read_text())
            self.assertEqual(train['steps'],8);self.assertEqual(train['feature_examples'],64)
            for bank in ('joint','independent'):
                self.assertLess(train['final'][bank]['cross_entropy'],train['initial'][bank]['cross_entropy'])
            report=json.loads((root/'run/evaluation.json').read_text())
            self.assertEqual(report['status'],'passed');self.assertEqual(report['world_size'],2)
            methods=report['scopes']['development_2k']['methods']
            self.assertEqual(len(methods),8)
            for row in methods.values():self.assertEqual(row['scored_tokens'],64)
            for step in (4,8):
                checkpoint=root/'run/trainer'/f'checkpoint-{step}'
                self.assertTrue((checkpoint/'optimizer.pt').is_file())
                self.assertTrue((checkpoint/'scheduler.pt').is_file())

    def test_local_slices_equal_explicit_mask(self):
        m=model().model.layers[0].self_attn
        torch.manual_seed(11)
        for sinks in (0,4):
            for start,length in ((0,1),(0,19),(7,1),(7,12)):
                q=torch.randn(2,8,length,8);k=torch.randn(2,4,start+length+3,8);v=torch.randn_like(k)
                a=local_attention(m,q,k,v,start=start,window=5,sinks=sinks,chunk=3)
                qi=torch.arange(start,start+length)[:,None];ki=torch.arange(k.shape[2])[None,:]
                mask=(ki<=qi)&((ki<sinks)|(ki>=qi-4))
                b=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=m.scaling,enable_gqa=True).transpose(1,2)
                torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        with patch('efficient_llm.context_attention.sdpa_attention_forward',wraps=__import__('efficient_llm.context_attention',fromlist=['sdpa_attention_forward']).sdpa_attention_forward) as call:
            local_attention(m,torch.randn(2,8,1,8),torch.randn(2,4,100,8),torch.randn(2,4,100,8),start=99,window=5,sinks=4)
            self.assertEqual(call.call_args.args[2].shape[2],9)

    def test_teacher_identity_and_feature_alignment(self):
        m=model();ids=torch.arange(1,25)[None,:]%53
        before={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():
            dense=m(ids).logits
            with ContextPolicy(m,mode='teacher',keep=2,window=5,sinks=2,stride=3) as c:
                actual=m(ids).logits;x,costs=c.take()
            torch.testing.assert_close(actual,dense,rtol=0,atol=0)
        self.assertEqual(x.shape,(8,2,32));self.assertEqual(costs.shape,(8,2,2,6))
        self.assertTrue(torch.isfinite(costs).all());self.assertTrue((costs>=0).all())
        self.assertTrue((c.layer_totals[:,0]>0).all())
        for key,value in m.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_static_branches_match_two_branch_reference_and_cached(self):
        m=model();ids=torch.arange(1,28)[None,:]%53
        for mask in (torch.ones(2,4,dtype=torch.bool),torch.zeros(2,4,dtype=torch.bool),
                     torch.tensor([[0,1,0,1],[1,0,1,0]],dtype=torch.bool),
                     torch.tensor([[1,1,1,1],[0,0,0,0]],dtype=torch.bool)):
            with torch.no_grad(),ContextPolicy(m,mode='reference_static',keep=2,window=5,sinks=2,static_masks=mask):
                ref=m(ids).logits
            for cache_type in ('dynamic','static'):
                with torch.no_grad(),ContextPolicy(m,mode='static',keep=2,window=5,sinks=2,static_masks=mask):
                    full=m(ids,use_cache=False).logits
                    cache=DynamicCache(config=m.config) if cache_type=='dynamic' else StaticCache(config=m.config,max_cache_len=35)
                    chunks=[]
                    for start,end in ((0,11),(11,12),(12,19),(19,27)):
                        chunks.append(m(ids[:,start:end],past_key_values=cache,use_cache=True).logits)
                    torch.testing.assert_close(torch.cat(chunks,1),full,rtol=2e-5,atol=2e-5)
                torch.testing.assert_close(full,ref,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(m(ids).logits,m(ids).logits,rtol=0,atol=0)

    def test_dynamic_causality_cached_and_cleanup(self):
        m=model();ids=torch.arange(1,25)[None,:]%53
        router=RouterPair(2,32,4,2,torch.ones(2))
        for mode in ('local','oracle','router'):
            with torch.no_grad(),ContextPolicy(m,mode=mode,keep=2,window=5,sinks=2,routers=router):
                full=m(ids,use_cache=False).logits
                changed=ids.clone();changed[:,12:]=1
                torch.testing.assert_close(full[:,:12],m(changed,use_cache=False).logits[:,:12],rtol=0,atol=0)
                cache=StaticCache(config=m.config,max_cache_len=30);pieces=[]
                for t in range(24):pieces.append(m(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits)
                torch.testing.assert_close(torch.cat(pieces,1),full,rtol=3e-4,atol=2e-3)
            with torch.no_grad(),ContextPolicy(m,mode='local',window=5,sinks=2):
                with self.assertRaisesRegex(ValueError,'switch policies'):m(ids[:,:1],past_key_values=cache)
        original=m.model.layers[0].self_attn.forward
        with self.assertRaisesRegex(ValueError,'unpadded'):
            with ContextPolicy(m,mode='local'):
                m(ids,attention_mask=torch.ones_like(ids))
        self.assertEqual(m.model.layers[0].self_attn.forward,original)

    def test_cache_retains_unique_policy_identity(self):
        m=model();ids=torch.tensor([[1,2]])
        cache=StaticCache(config=m.config,max_cache_len=8)
        with torch.no_grad(),ContextPolicy(m,mode='local') as policy:
            m(ids,past_key_values=cache,use_cache=True)
            self.assertIs(cache._context_policy_owner,policy.cache_owner)
        token=cache._context_policy_owner
        del policy
        with torch.no_grad(),ContextPolicy(m,mode='local') as replacement:
            self.assertIsNot(token,replacement.cache_owner)
            with self.assertRaisesRegex(ValueError,'switch policies'):
                m(ids,past_key_values=cache,use_cache=True)


if __name__=='__main__': unittest.main()

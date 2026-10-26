"""CPU interpreter math/cache tests; compiled CUDA is a separate mandatory gate."""
import os
import subprocess
import sys
import unittest
from efficient_llm.lookup_probe import build,COLORS


class Tokenizer:
    def encode(self,text,add_special_tokens=False):
        import re
        parts=re.split('('+'|'.join(' '+c for c in COLORS)+')',text)
        ids=[]
        for part in parts:
            if part in [' '+c for c in COLORS]:ids.append(1000+COLORS.index(part[1:]))
            else:ids.extend(2000+sum(ord(c)*256**j for j,c in enumerate(part[i:i+3])) for i in range(0,len(part),3))
        return ids


class SelectiveTests(unittest.TestCase):
    def test_compact_interpreter_budgets_masks_and_cache(self):
        code='''
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,StaticCache
from efficient_llm.selective_decode import decode,decode_compact
from efficient_llm.context_attention import ContextPolicy
from efficient_llm.routers import RouterPair
torch.set_num_threads(1);torch.manual_seed(73)
for length,capacity in ((1,7),(4,37),(17,260),(259,600)):
 for sinks in (0,4):
  q=torch.randn(2,4,1,8)
  k=torch.randn(2,2,capacity,16)[...,::2];v=torch.randn(2,2,capacity,16)[...,::2]
  k[:,:,length:]=float('nan');v[:,:,length:]=float('nan')
  for keep,m in ((0,torch.zeros(2,2,dtype=torch.bool)),(2,torch.ones(2,2,dtype=torch.bool)),(1,torch.tensor([[1,0],[0,1]],dtype=torch.bool))):
   kwargs=dict(length=length,window=16,sinks=sinks,scale=8**-.5)
   actual=decode_compact(q,k,v,m,global_groups=keep,**kwargs)
   original=decode(q,k,v,m,**kwargs)
   p=torch.arange(length);allowed=m.repeat_interleave(2,1)[:,:,None,None]|((p<sinks)|(p>=length-16))[None,None,None,:]
   expected=torch.nn.functional.scaled_dot_product_attention(q,k[:,:,:length],v[:,:,:length],allowed,enable_gqa=True).transpose(1,2)
   torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
   torch.testing.assert_close(actual,original,atol=2e-6,rtol=1e-5)
bad=decode_compact(q,k,v,torch.zeros(2,2,dtype=torch.bool),global_groups=1,**kwargs)
assert torch.isnan(bad).all()
q=torch.randn(1,6,1,8);k=torch.randn(1,3,300,8);v=torch.randn_like(k)
k[:,:,4:243]=float('nan');v[:,:,4:243]=float('nan')
assert torch.isfinite(decode_compact(q,k,v,torch.zeros(1,3,dtype=torch.bool),length=259,window=16,sinks=4,scale=8**-.5,global_groups=0)).all()
model=Qwen3ForCausalLM._from_config(Qwen3Config(vocab_size=41,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8),attn_implementation='sdpa').eval().requires_grad_(False)
r=RouterPair(2,32,2,1,torch.ones(2)).eval();ids=torch.randint(0,41,(2,19))
for sinks in (0,2):
 out=[]
 for backend in ('sdpa','triton','triton_compact'):
  with torch.inference_mode(),ContextPolicy(model,mode='router',routers=r,keep=1,window=5,sinks=sinks,decode_backend=backend):
   cache=StaticCache(config=model.config,max_cache_len=24)
   model(ids[:,:11],past_key_values=cache,use_cache=True)
   out.append(torch.cat([model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits for t in range(11,19)],1))
 for actual in out[1:]:torch.testing.assert_close(actual,out[0],atol=1e-5,rtol=1e-5)
print('COMPACT_INTERPRETER_AND_CACHE_PASS')
'''
        result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'TRITON_INTERPRET':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_numerical_controls_restore_and_replay(self):
        import torch
        from efficient_llm.decode_numerics import routes,fp32_attention
        from efficient_llm.routers import RouterPair
        from efficient_llm import selective_decode
        router=RouterPair(2,32,2,1,torch.ones(2)).eval()
        x=torch.randn(1,1,32);original=router.layer_choice
        with routes(router) as seen:router.layer_choice(x,0,0)
        self.assertEqual(router.layer_choice,original)
        forced=[1-seen[0]]
        with routes(router,replay=forced) as replay:
            actual=router.layer_choice(x,0,0)
        torch.testing.assert_close(actual,forced[0])
        self.assertEqual(len(replay),1)
        decode=selective_decode.decode
        q=torch.randn(1,4,1,8);k=torch.randn(1,2,17,8);v=torch.randn_like(k)
        selected=torch.tensor([[True,False]])
        with fp32_attention():
            result=selective_decode.decode(q,k,v,selected,length=13,window=5,sinks=2,scale=8**-.5)
            v[:,:,13:]=float('nan')
            again=selective_decode.decode(q,k,v,selected,length=13,window=5,sinks=2,scale=8**-.5)
        self.assertIs(selective_decode.decode,decode)
        torch.testing.assert_close(result,again,atol=0,rtol=0)

    def test_lookup_balance_positions_and_reproducibility(self):
        a=build(Tokenizer());b=build(Tokenizer())
        self.assertEqual(len(a['rows']),128)
        for x,y in zip(a['rows'],b['rows']):
            self.assertEqual(x['tokens'].tolist(),y['tokens'].tolist())
            self.assertEqual(len(x['tokens']),x['length'])
            self.assertEqual(x['tokens'][x['target_token_position']],a['candidate_ids'][x['label']])
        for length in (2048,8192):
            for depth in ('near','10%','50%','90%'):
                rows=[r for r in a['rows'] if (r['length'],r['depth'])==(length,depth)]
                self.assertEqual([sum(r['label']==i for r in rows) for i in range(8)],[2]*8)
                for r in rows:self.assertEqual(r['length']-r['target_token_position']<=128,depth=='near')

    def test_interpreted_kernel_and_real_cache_model(self):
        code='''
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,StaticCache
from efficient_llm.selective_decode import decode
from efficient_llm.context_attention import ContextPolicy
from efficient_llm.routers import RouterPair
torch.set_num_threads(1);torch.manual_seed(2)
for length,capacity in ((1,3),(3,37),(132,260),(291,600)):
 q=torch.randn(2,4,1,8);k=torch.randn(2,2,capacity,16)[...,::2];v=torch.randn_like(k.contiguous()).repeat_interleave(2,-1)[...,::2]
 assert k.stride()==v.stride()
 k[:,:,length:]=float('nan');v[:,:,length:]=float('nan')
 for m in (torch.zeros(2,2,dtype=torch.bool),torch.ones(2,2,dtype=torch.bool),torch.tensor([[1,0],[0,1]],dtype=torch.bool)):
  actual=decode(q,k,v,m,length=length,window=16,sinks=4,scale=8**-.5)
  p=torch.arange(length);allowed=m.repeat_interleave(2,1)[:,:,None,None]|((p<4)|(p>=length-16))[None,None,None,:]
  expected=torch.nn.functional.scaled_dot_product_attention(q,k[:,:,:length],v[:,:,:length],allowed,enable_gqa=True).transpose(1,2)
  torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
model=Qwen3ForCausalLM._from_config(Qwen3Config(vocab_size=41,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8),attn_implementation='sdpa').eval().requires_grad_(False)
r=RouterPair(2,32,2,1,torch.ones(2)).eval();ids=torch.randint(0,41,(2,19));out=[]
for backend in ('sdpa','triton'):
 with torch.inference_mode(),ContextPolicy(model,mode='router',routers=r,keep=1,window=5,sinks=2,decode_backend=backend):
  cache=StaticCache(config=model.config,max_cache_len=24)
  model(ids[:,:11],past_key_values=cache,use_cache=True)
  out.append(torch.cat([model(ids[:,t:t+1],past_key_values=cache,use_cache=True).logits for t in range(11,19)],1))
torch.testing.assert_close(out[0],out[1],atol=1e-5,rtol=1e-5)
print('INTERPRETER_AND_MODEL_CACHE_PASS')
'''
        result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'TRITON_INTERPRET':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()

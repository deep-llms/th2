import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from efficient_llm.kv_mlp import CompressedQwen, convert, select_mlp, distill_loss, allocation, load_compressed
from efficient_llm.diagnostic_data import sha256


def tiny():
    torch.manual_seed(41)
    c = Qwen3Config(vocab_size=47, hidden_size=24, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0., attention_bias=True)
    return CompressedQwen._from_config(c, attn_implementation='sdpa').eval()


def independent_reference(model, ids):
    """Explicit layer algebra, independently replicated GQA and causal softmax."""
    h = model.model.embed_tokens(ids)
    positions = torch.arange(ids.shape[1])[None,:].to(ids.device)
    cos, sin = model.model.rotary_emb(h, positions)
    shared = {}
    for i, layer in enumerate(model.model.layers):
        x = layer.input_layernorm(h); a = layer.self_attn
        shape = (*x.shape[:2], -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shape)).transpose(1,2)
        # Independent explicit rotation, including source K only once.
        def rope(z):
            lo, hi = z.chunk(2,-1)
            return z*cos[:,None]+torch.cat((-hi,lo),-1)*sin[:,None]
        q = rope(q)
        if a.source_idx == i:
            k = rope(a.k_norm(a.k_proj(x).view(shape)).transpose(1,2))
            v = a.v_proj(x).view(shape).transpose(1,2)
            shared[i] = (k,v)
        k,v = shared[a.source_idx]
        k = k.repeat_interleave(a.num_key_value_groups,1)
        v = v.repeat_interleave(a.num_key_value_groups,1)
        scores = (q@k.transpose(-1,-2))*a.scaling
        mask = torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, -float('inf'))
        y = (scores.softmax(-1)@v).transpose(1,2).flatten(2)
        h = h+a.o_proj(y)
        h = h+layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h))


class KVMlpTests(unittest.TestCase):
    def test_bf16_independent_decoder_same_kernel(self):
        from efficient_llm.kv_mlp_check import reference
        model=convert(tiny(),share=True,channels=[list(range(24))]*4)
        other=deepcopy(model);ids=torch.arange(1,19).reshape(2,9)%47
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=model(ids,use_cache=False).logits
            expected=reference(other,ids,sdpa=True)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.float().square().sum().backward()
            expected.float().square().sum().backward()
        for (name,a),(other_name,b) in zip(model.named_parameters(),other.named_parameters()):
            self.assertEqual(name,other_name)
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-5,msg=name)

    def test_dense_identity_and_shared_reference_gradients(self):
        model = tiny(); ids = torch.arange(1,19).reshape(2,9)%47
        native = Qwen3ForCausalLM._from_config(deepcopy(model.config),attn_implementation='sdpa')
        native.load_state_dict(model.state_dict()); native.eval()
        torch.testing.assert_close(model(ids,use_cache=False).logits,native(ids,use_cache=False).logits,rtol=0,atol=0)
        convert(model,share=True)
        ref = deepcopy(model)
        a,b = model(ids,use_cache=False).logits,independent_reference(ref,ids)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        a.square().sum().backward();b.square().sum().backward()
        for (name,x),(other,y) in zip(model.named_parameters(),ref.named_parameters()):
            self.assertEqual(name,other)
            self.assertIsNotNone(x.grad,name)
            torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=2e-5,msg=name)
        self.assertIsNone(model.model.layers[1].self_attn.k_proj)

    def test_consumer_gradient_reaches_producer(self):
        model=convert(tiny(),share=True)
        a,b=(layer.self_attn for layer in model.model.layers[:2])
        x=torch.randn(2,6,24,requires_grad=True);y=torch.randn_like(x,requires_grad=True)
        pos=model.model.rotary_emb(x,torch.arange(6)[None]);shared={}
        a(x,pos,None,shared_kv=shared)
        output=b(y,pos,None,shared_kv=shared)[0]
        output.square().sum().backward()
        self.assertGreater(a.k_proj.weight.grad.norm().item(),0)
        self.assertGreater(a.v_proj.weight.grad.norm().item(),0)
        self.assertGreater(x.grad.norm().item(),0)

    def test_cache_chunking_capacity_and_reload(self):
        m=convert(tiny(),share=True,channels=[list(range(24))]*4)
        ids=torch.arange(1,27).reshape(2,13)%47
        with torch.no_grad():
            full=m(ids,use_cache=False).logits
            cache=DynamicCache(config=m.config)
            pieces=[]
            for start,end in [(0,5),(5,6),(6,10),(10,13)]:
                pieces.append(m(ids[:,start:end],past_key_values=cache,use_cache=True).logits)
            torch.testing.assert_close(torch.cat(pieces,1),full,rtol=2e-5,atol=2e-6)
            used=[i for i,l in enumerate(cache.layers) if l.is_initialized]
            self.assertEqual(used,[0,2])
            total=sum(l.keys.numel()+l.values.numel() for l in cache.layers if l.is_initialized)
            self.assertEqual(total,2*2*2*13*8*2)
            automatic=m(ids[:,:5],use_cache=True).past_key_values
            continuation=m(ids[:,5:],past_key_values=automatic,use_cache=True).logits
            torch.testing.assert_close(continuation,full[:,5:],rtol=2e-5,atol=2e-6)
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            loaded=load_compressed(tmp,torch.device('cpu'))
            torch.testing.assert_close(loaded(ids,use_cache=False).logits,full,rtol=0,atol=0)
        m.train();m.gradient_checkpointing_enable()
        with self.assertRaisesRegex(ValueError,'checkpointing'):m(ids,use_cache=False)

    def test_pruning_equals_masked_mlp_and_gradient(self):
        original=tiny().model.layers[0].mlp
        pruned=deepcopy(original); keep=torch.tensor([0,2,7,15,20,29])
        select_mlp(pruned,keep)
        x=torch.randn(2,5,24,requires_grad=True);y=x.detach().clone().requires_grad_(True)
        gate=original.act_fn(original.gate_proj(y))*original.up_proj(y)
        mask=torch.zeros(32);mask[keep]=1
        ref=original.down_proj(gate*mask)
        actual=pruned(x)
        torch.testing.assert_close(actual,ref,rtol=1e-5,atol=1e-7)
        actual.square().sum().backward();ref.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-7)
        for name,axis in [('gate_proj',0),('up_proj',0),('down_proj',1)]:
            torch.testing.assert_close(getattr(pruned,name).weight.grad,
                getattr(original,name).weight.grad.index_select(axis,keep),rtol=1e-5,atol=1e-7)

    def test_loss_chunks_and_equal_batch_mean(self):
        torch.manual_seed(7)
        x=torch.randn(4,9,8,requires_grad=True);w=torch.randn(19,8,requires_grad=True)
        h=torch.randn_like(x);v=torch.randn_like(w);t=torch.randint(0,19,(4,9))
        expected=.5*(F.cross_entropy(F.linear(x,w).flatten(0,1),t.flatten())+
            F.kl_div(F.linear(x,w).log_softmax(-1),F.linear(h,v).log_softmax(-1),
                     log_target=True,reduction='sum')/t.numel())
        actual=distill_loss(x,w,h,v,t,chunk=3)
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,(x,w),retain_graph=True)
        gb=torch.autograd.grad(expected,(x,w),retain_graph=True)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
        pieces=sum(distill_loss(x[i:i+1],w,h[i:i+1],v,t[i:i+1],chunk=4) for i in range(4))/4
        torch.testing.assert_close(pieces,expected)

    def test_budget_and_invalid_config(self):
        result=allocation([[1,1.1,1.2],[1,1.3,1.4]],prune_units=2)
        self.assertEqual(result,[2,0])
        with self.assertRaises(ValueError):allocation([[1,2]],prune_units=2)
        with self.assertRaises(ValueError):select_mlp(tiny().model.layers[0].mlp,[1,1])
        m=convert(tiny(),share=True)
        with self.assertRaises(ValueError):convert(m,share=True)

    def test_two_rank_trainer_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';probe=root/'probe';probe.mkdir()
            tiny().save_pretrained(model)
            torch.manual_seed(7)
            torch.save(dict(calibration=torch.randint(0,47,(24,17)),evaluation=torch.randint(0,47,(4,17))),probe/'tokens.pt')
            (probe/'manifest.json').write_text(json.dumps(dict(status='passed',length=16,
                calibration_documents=24,evaluation_documents=4,tokens_sha256=sha256(probe/'tokens.pt'),
                model_files={p.name:sha256(p) for p in model.iterdir() if p.is_file()})))
            launch=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    '-m','efficient_llm.kv_mlp_study']
            common=['--model',str(model),'--probe',str(probe),'--device','cpu','--calibration','4']
            for mode,extra in [('calibrate',['--output',str(root/'plan.json')]),
                               ('train',['--output',str(root/'D'),'--plan',str(root/'plan.json'),
                                         '--arm','D','--steps','4','--batch','2'])]:
                done=subprocess.run([*launch,mode,*common,*extra],capture_output=True,text=True,timeout=240)
                self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            report=json.loads((root/'D/result.json').read_text())
            self.assertEqual(report['steps'],4);self.assertEqual(report['input_tokens'],256)
            self.assertEqual(set(report['validation']),{'0','1','4'})
            for step in (1,4):
                p=root/f'D/trainer/checkpoint-{step}'
                for name in ['optimizer.pt','scheduler.pt','rng_state_0.pth','rng_state_1.pth']:
                    self.assertTrue((p/name).is_file(),name)


if __name__=='__main__':unittest.main()

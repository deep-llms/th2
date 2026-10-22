import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.update(OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import Qwen3Config
from bdot.model import BDOTForCausalLM, RoutedStep
from bdot.routing import RoutingConfig, quota_table
from packed_training import PackedCollator, model_inputs

torch.set_num_threads(1)


def model(**options):
    torch.manual_seed(21)
    c = Qwen3Config(vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_dropout=0., tie_word_embeddings=True, pad_token_id=None,
        rms_norm_eps=1e-6, max_position_embeddings=64)
    c._attn_implementation = 'sdpa'
    c.use_cache = False
    c.bdot = RoutingConfig(n_stem=1, n_coda=1, S=4, **options).to_dict()
    return BDOTForCausalLM(c)


def batch():
    rows = [dict(input_ids=[1,2,3,4,5,6], attention_mask=[1]*6,
                 labels=[1,2,3,4,5,6], segments=[0,0,0,1,1,1]),
            dict(input_ids=[7,8,9,3,2,1], attention_mask=[1]*6,
                 labels=[7,8,9,3,2,1], segments=[2]*6)]
    return model_inputs(PackedCollator()(rows))


def mapped_pairs(m, r):
    yield m.model.embed_tokens.weight, r.model.embed_tokens.weight
    yield m.model.norm.weight, r.model.norm.weight
    for i, layer in enumerate(m.model.layers):
        ref = r.model.layers[i]
        if not isinstance(layer, RoutedStep):
            yield from zip(layer.parameters(), ref.parameters())
        else:
            expert = m.model.bdot_experts[layer.step]
            for name in ('q_proj','q_norm','o_proj'):
                yield from zip(getattr(expert,name).parameters(),getattr(ref.self_attn,name).parameters())
            for name in ('k_proj','k_norm','v_proj'):
                yield from zip(getattr(layer,name).parameters(),getattr(ref.self_attn,name).parameters())
            yield from zip(layer.input_layernorm.parameters(),ref.input_layernorm.parameters())
            yield from zip(expert.mlp_norm.parameters(),ref.post_attention_layernorm.parameters())
            yield from zip(expert.mlp.parameters(),ref.mlp.parameters())


class ModelTests(unittest.TestCase):
    def test_routing_seed_does_not_change_parameter_initialization(self):
        a=model(seed=0); b=model(seed=91)
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,atol=0,rtol=0)
        a.progress=b.progress=.1
        # Random routing draws follow the separate seed, not the weight init RNG.
        a.eval();b.eval();a.capture_routes=b.capture_routes=True
        a.routing_override=b.routing_override='random'
        a(**batch());b(**batch())
        self.assertFalse(torch.equal(a.last_routes,b.last_routes))

    def test_native_logits_gradients_adamw(self):
        m=model(routing='fixed',curriculum='none',step_film=False)
        r=m.native_reference(); m.train(); r.train()
        a=m(**batch()); b=r(**batch())
        torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-6)
        a.loss.backward();b.loss.backward()
        for p,q in mapped_pairs(m,r):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
        for net in (m,r): torch.optim.AdamW(net.parameters(),lr=1e-3).step()
        for p,q in mapped_pairs(m,r):
            torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)

    def test_causality_and_document_isolation(self):
        m=model(curriculum='none',default_prior=0.)
        m.eval(); inputs=batch(); a=m(**inputs).logits
        modified=copy.deepcopy(inputs);modified['input_ids'][0,2:]=17
        b=m(**modified).logits
        torch.testing.assert_close(a[0,:2],b[0,:2],atol=1e-6,rtol=1e-6)
        modified=copy.deepcopy(inputs);modified['input_ids'][0,:3]=19
        b=m(**modified).logits
        torch.testing.assert_close(a[0,3:],b[0,3:],atol=1e-6,rtol=1e-6)

    def test_router_lm_gradient(self):
        m=model(curriculum='none',z_loss_coef=0.,default_prior=0.)
        m(**batch()).loss.backward()
        for layer in m.routed_steps:
            self.assertGreater(layer.router.weight.grad.abs().sum().item(),0)
            self.assertGreater(layer.router.bias.grad.abs().sum().item(),0)

    def test_quota_table_counts(self):
        for reach,expected in [(0,1),(1,34),(2,400),(7,40320)]:
            allowed,nxt=quota_table(8,reach)
            def count(state):
                choices=allowed[state].nonzero().flatten().tolist()
                return sum(count(int(nxt[state,m])) for m in choices) if choices else 1
            self.assertEqual(count(0),expected)

    def test_no_reuse_and_curriculum(self):
        m=model(reuse=False,curriculum='A');m.capture_routes=True
        for progress,reach in [(0.,0),(.1,1),(.7,1)]:
            m.progress=progress;m(**batch())
            routes=m.last_routes
            self.assertTrue((routes.sort(0).values==torch.arange(4)[:,None,None]).all())
            self.assertTrue(((routes-torch.arange(4)[:,None,None]).abs()<=reach).all())
        c=RoutingConfig(r=2,curriculum='A')
        self.assertEqual(c.phase(.1,True),(1,False,.05))
        self.assertAlmostEqual(c.phase(.15,True)[2],.025)
        self.assertEqual(c.phase(.31,True),(2,False,0.))

    def test_accumulation_loss_normalization(self):
        a=model(curriculum='B');b=copy.deepcopy(a)
        inputs=batch();count=inputs['labels'][:,1:].ne(-100).sum()
        decisions=inputs['input_ids'].numel()*a.options.S
        whole=a(**inputs,num_items_in_batch=count,bdot_num_decisions=decisions).loss
        whole.backward();parts=[]
        for row in range(2):
            part=b(**{k:v[row:row+1] for k,v in inputs.items()},
                   num_items_in_batch=count,bdot_num_decisions=decisions).loss
            part.backward();parts.append(part.detach())
        torch.testing.assert_close(whole.detach(),sum(parts),atol=1e-6,rtol=1e-6)
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is None: self.assertIsNone(q.grad)
            else: torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
        a.finish_update();b.finish_update()
        torch.testing.assert_close(a.balance_bias,b.balance_bias)

    def test_checkpoint_replay_gradients_and_counts(self):
        for curriculum,progress in [('A',.1),('B',.01)]:
            a=model(curriculum=curriculum,default_prior=0.,explore_eps_max=1.)
            b=copy.deepcopy(a);a.progress=b.progress=progress
            b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            a.capture_routes=b.capture_routes=True
            x=a(**batch());y=b(**batch())
            x.loss.backward();y.loss.backward()
            torch.testing.assert_close(x.logits,y.logits,atol=0,rtol=0)
            torch.testing.assert_close(a.last_routes,b.last_routes,atol=0,rtol=0)
            torch.testing.assert_close(a.load_counts,b.load_counts,atol=0,rtol=0)
            for p,q in zip(a.parameters(),b.parameters()):
                if p.grad is None: self.assertIsNone(q.grad)
                else: torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)

    def test_cached_logits(self):
        for routing in ('fixed','learned'):
            m=model(routing=routing,curriculum='none',default_prior=0.);m.eval()
            inputs=batch();full=m(**inputs).logits; cache=None
            for t in range(6):
                out=m(input_ids=inputs['input_ids'][:,t:t+1],
                    position_ids=inputs['position_ids'][:,t:t+1],
                    attention_mask=inputs['attention_mask'][:,:,t:t+1,:t+1],
                    use_cache=True,past_key_values=cache)
                cache=out.past_key_values
                torch.testing.assert_close(out.logits[:,0],full[:,t],atol=1e-6,rtol=1e-5)

    def test_save_load(self):
        variants=[dict(routing='fixed',curriculum='none',r=0),
                  dict(routing='random',curriculum='none')]
        variants += [dict(reuse=reuse,curriculum=c) for reuse in (True,False) for c in ('A','B')]
        for options in variants:
            with self.subTest(**options),tempfile.TemporaryDirectory() as tmp:
                m=model(**options);m.progress=.2
                m(**batch()).loss.backward()
                torch.optim.AdamW(m.parameters(),lr=.001).step()
                m.zero_grad(set_to_none=True);m.finish_update()
                m.save_pretrained(tmp)
                r=BDOTForCausalLM.from_pretrained(tmp,attn_implementation='sdpa')
                for name,value in m.state_dict().items():
                    torch.testing.assert_close(value,r.state_dict()[name],atol=0,rtol=0,msg=name)
                self.assertFalse(r.load_counts.any())
                self.assertFalse(r.metric_totals.any())
                for step in r.routed_steps:
                    if not r.options.reuse:
                        for reach in range(r.options.r+1):
                            allowed,transitions=quota_table(r.options.S,reach)
                            torch.testing.assert_close(getattr(step,f'allowed_{reach}'),allowed)
                            torch.testing.assert_close(getattr(step,f'transitions_{reach}'),transitions)
                m.eval();r.eval()
                torch.testing.assert_close(m(**batch()).logits,r(**batch()).logits,atol=0,rtol=0)
                torch.testing.assert_close(m.routing_counter,r.routing_counter)
                # Loading for further updates must also start
                # with empty accumulators, while preserving persistent beta.
                m.train();r.train();r.progress=m.progress
                a=m(**batch()).loss;b=r(**batch()).loss
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                a.backward();b.backward();m.finish_update();r.finish_update()
                torch.testing.assert_close(m.balance_bias,r.balance_bias,atol=0,rtol=0)


if __name__=='__main__': unittest.main()

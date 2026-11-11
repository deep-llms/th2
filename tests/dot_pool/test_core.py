import copy,tempfile,unittest
from dataclasses import replace
from pathlib import Path
import torch
from efficient_llm.dot_pool.model import Config,Model
from efficient_llm.dot_pool.data import e2_batch

torch.set_num_threads(2)
def tiny(arm='DOT'):
    return Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=24,expert_ff=12,middle=4,pool=8,vocab=32,candidates=4,arm=arm,label_rate=1)
class Core(unittest.TestCase):
 def test_parameter_counts(self):
    values={'E1':{'D':(37.76,37.76),'L':(25.18,37.76),'SO':(94.65,38.03),'DOT':(94.69,38.06),'SR':(94.39,37.77)},'E2':{'D':(75.53,75.53),'PL':(217.17,75.61),'SO':(218.73,77.17),'DOT':(218.81,77.26),'SR':(217.09,75.54)}}
    for s,arms in values.items():
      for a,(total,active) in arms.items():
       with torch.device('meta'):m=Model(Config.production(s,a))
       z=m.counts();self.assertLess(abs(z['nonembedding_total']/1e6/total-1),.001);self.assertLess(abs(z['nonembedding_active']/1e6/active-1),.001)
 def test_causality_and_doc_isolation(self):
    for a in ['D','L','SO','DOT','SR','PL']:
     m=Model(tiny(a)).eval();x=torch.randint(1,32,(2,6));seg=torch.tensor([[0,0,0,1,1,1]]).expand_as(x);pos=torch.tensor([[0,1,2,0,1,2]]).expand_as(x)
     torch.manual_seed(4);y=m(x,segments=seg,positions=pos);z=x.clone();z[:,2]=(z[:,2]+1)%32
     torch.manual_seed(4);yy=m(z,segments=seg,positions=pos);torch.testing.assert_close(y[:,:2],yy[:,:2]);torch.testing.assert_close(y[:,3:],yy[:,3:])
 def test_identity_routing_and_balancing(self):
    for a in ['SO','DOT']:
     m=Model(tiny(a)).eval();m.capture=True;x=torch.randint(0,32,(2,4));y=m(x);m.forced_routes=[(r['ids'],r['gates']) for r in m.routes];yy=m(x);torch.testing.assert_close(y,yy)
     for ids,gates in m.forced_routes:self.assertTrue((ids[...,0]!=ids[...,1]).all());torch.testing.assert_close(gates.sum(-1),torch.ones(2,4))
     old=m.beta.clone();m.loads[0,0]=100;m.finish_update();self.assertLess(m.beta[0,0],old[0,0]);self.assertGreater(m.beta[0,1],old[0,1]);self.assertEqual(m.loads.sum().item(),0)
 def test_taylor_microbatch(self):
    torch.manual_seed(1);m=Model(tiny());m2=copy.deepcopy(m);x=torch.randint(0,32,(4,4));lab=x.clone();lab[:,:3]=-100
    for model,slices in [(m,[slice(None)]),(m2,[slice(0,2),slice(2,4)])]:
     for sl in slices:
      out=model(x[sl],labels=lab[sl]);model.gradient_factor=4;(out['nll_sum']/4+out['balance_loss']*len(x[sl])/4).backward()
    for s in range(4):
     a=torch.cat([z['q'] for z in m.labels if z['s']==s]);b=torch.cat([z['q'] for z in m2.labels if z['s']==s]);torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    m.zero_grad();m.promote_labels();m.replay_loss().backward()
    for n,p in m.named_parameters():
     if p.grad is not None and n not in ['router','path']:self.assertEqual(p.grad.abs().sum().item(),0,n)
 def test_save_resume_pending_taylor(self):
    torch.manual_seed(8);m=Model(tiny());o=torch.optim.AdamW(m.parameter_groups(),lr=.001);x=torch.randint(0,32,(2,4));lab=x.clone();lab[:,:3]=-100
    def step(model,opt):
     opt.zero_grad();out=model(x,labels=lab);model.gradient_factor=2;(out['nll_sum']/2+out['balance_loss']+model.replay_loss()).backward();opt.step();model.finish_update();model.promote_labels();return float(out['nll_sum'])
    step(m,o)
    with tempfile.TemporaryDirectory() as td:
     p=Path(td)/'state.pt';torch.save(dict(model=m.state_dict(),optim=o.state_dict(),pending=m.pending,rng=torch.get_rng_state()),p);state=torch.load(p,weights_only=False)
     expected=[step(m,o) for _ in range(10)];m2=Model(tiny());o2=torch.optim.AdamW(m2.parameter_groups(),lr=.001);m2.load_state_dict(state['model']);o2.load_state_dict(state['optim']);m2.pending=state['pending'];torch.set_rng_state(state['rng']);actual=[step(m2,o2) for _ in range(10)]
     self.assertLess(max(abs(a-b) for a,b in zip(expected,actual)),1e-4)
 def test_hf_dense_equivalence(self):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    c=tiny('D');cfg=Qwen3Config(vocab_size=c.vocab,hidden_size=c.d,intermediate_size=c.ff,num_hidden_layers=c.stem+c.middle+c.coda,num_attention_heads=c.heads,num_key_value_heads=c.kv_heads,head_dim=c.head_dim,rms_norm_eps=1e-6,rope_theta=10000.,tie_word_embeddings=True,attention_bias=False,attention_dropout=0.,use_cache=False)
    cfg._attn_implementation='sdpa';hf=Qwen3ForCausalLM(cfg).eval();m=Model(c).eval()
    m.embed_tokens.load_state_dict(hf.model.embed_tokens.state_dict());m.norm.load_state_dict(hf.model.norm.state_dict())
    for a,b in zip(list(m.stem)+list(m.middle)+list(m.coda),hf.model.layers):a.load_state_dict(b.state_dict())
    x=torch.randint(0,c.vocab,(2,6));torch.testing.assert_close(m(x),hf(x).logits,atol=1e-5,rtol=1e-5)
 def test_eos_masks(self):
    b=e2_batch([[4,5,0,6,7,0]],'cpu');self.assertEqual(b['positions'].tolist(),[[0,1,2,0,1,2]]);self.assertEqual(b['labels'].tolist(),[[-100,5,0,-100,7,0]])
if __name__=='__main__':unittest.main()

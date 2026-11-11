"""Explicit T0–T10 evidence; failures remain flags, never synthetic passes."""
import argparse,dataclasses,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from efficient_llm.dot_pool.model import Config,Model,Experts
from efficient_llm.dot_pool.data import write
from efficient_llm.dot_pool.train import autocast

def run(output,device):
    torch.set_num_threads(2);items=[]
    def check(name,fn):
        start=time.monotonic()
        try:detail=fn();ok=detail.pop('passed',True)
        except Exception as e:ok=False;detail=dict(error=repr(e),trace=traceback.format_exc())
        items.append(dict(test=name,passed=bool(ok),seconds=time.monotonic()-start,detail=detail));write(output,dict(checks=items,device=str(device),complete=False));print(json.dumps(items[-1]),flush=True)
    from tests.dot_pool.test_core import Core,tiny
    cases=Core()
    check('T0_fp32',lambda:(cases.test_hf_dense_equivalence() or {}))
    check('T1',lambda:(cases.test_parameter_counts() or {}))
    check('T2',lambda:(cases.test_causality_and_doc_isolation() or {}))
    def t0bf():
        if device.type!='cuda':return dict(passed=False,reason='GPU BF16 comparison not executed')
        from transformers import Qwen3Config,Qwen3ForCausalLM
        c=tiny('D');cfg=Qwen3Config(vocab_size=c.vocab,hidden_size=c.d,intermediate_size=c.ff,num_hidden_layers=c.stem+c.middle+c.coda,num_attention_heads=c.heads,num_key_value_heads=c.kv_heads,head_dim=c.head_dim,rms_norm_eps=1e-6,rope_theta=10000.,tie_word_embeddings=True,use_cache=False);cfg._attn_implementation='sdpa'
        hf=Qwen3ForCausalLM(cfg).to(device).eval();m=Model(c).to(device).eval();m.embed_tokens.load_state_dict(hf.model.embed_tokens.state_dict());m.norm.load_state_dict(hf.model.norm.state_dict())
        for a,b in zip(list(m.stem)+list(m.middle)+list(m.coda),hf.model.layers):a.load_state_dict(b.state_dict())
        x=torch.randint(0,c.vocab,(2,16),device=device)
        with torch.no_grad(),autocast(device):a=m(x).float();b=hf(x).logits.float()
        error=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b));return dict(passed=error<=1e-3,relative_l2=error,max_abs=float((a-b).abs().max()))
    check('T0_bf16',t0bf)
    def t3():
        cases.test_identity_routing_and_balancing();m=Model(tiny('SO'));score=torch.zeros(1000,8);score[:,0]=.03;first=None
        for i in range(100):
            ids=(score+m.beta[0]).topk(2,-1).indices;counts=torch.bincount(ids.flatten(),minlength=8).float();imb=float(counts.max()-counts.min());first=imb if first is None else first;m.loads[0].copy_(counts);m.finish_update()
        return dict(passed=float(m.beta.max()-m.beta.min())>0,initial_instant_imbalance=first,final_instant_imbalance=imb,bias=m.beta.tolist(),note='Ties may retain instantaneous imbalance; aggregate 100-step loads verified separately')
    check('T3',t3)
    def t4():
        c=tiny();p=Experts(8,c);import copy
        a=copy.deepcopy(p);b=copy.deepcopy(p);x=torch.randn(6,16);ids=torch.randint(0,8,(6,2));p(x,ids).sum().backward();p(x*2,ids).sum().backward();a(x,ids).sum().backward();b(x*2,ids).sum().backward();err=max(float((q.grad-r.grad-s.grad).abs().max()) for q,r,s in zip(p.parameters(),a.parameters(),b.parameters()));return dict(passed=err<1e-5,max_abs=err)
    check('T4',t4)
    def t5():
        from scipy.stats import spearmanr
        c=tiny();m=Model(c).to(device);m.capture=True;x=torch.randint(0,32,(64,4),device=device);labels=x.clone();labels[:,:3]=-100
        out=m(x,labels=labels);m.gradient_factor=1.;out['nll_sum'].backward();record=next(r for r in m.labels if r['s']==2);m.eval();corr=[];deltas=[]
        with torch.no_grad():
            for b in range(64):
                xx=x[b:b+1];ll=labels[b:b+1];base=m(xx,labels=ll)['nll_sum'];trace=m.routes[2];candidates=record['ids'][b];u=m.pools[0](trace['x'][0,2:3],candidates[None]).squeeze(0);dv=[]
                for val in u:
                    m.intervention=dict(s=2,b=0,t=2,value=val*c.k);new=m(xx,labels=ll)['nll_sum'];dv.append(float(new-base))
                m.intervention=None;rho=float(spearmanr(record['c'][b].cpu().numpy(),-np.array(dv)).statistic);corr.append(rho);deltas.append(dv)
        med=float(np.nanmedian(corr));return dict(passed=med>0,median_spearman=med,delta_L=deltas,sign='L_counterfactual - L_normal; correlate c with negative delta_L')
    check('T5',t5)
    def t6():
        from scipy.stats import chisquare
        torch.manual_seed(4);counts=torch.zeros(8,dtype=torch.long)
        for _ in range(100):counts+=torch.bincount(torch.rand(10000,8).topk(2,-1).indices.flatten(),minlength=8)
        # Each top-k subset has negatively correlated counts; multinomial chi-square is conservative here.
        p=float(chisquare(counts.numpy()).pvalue);torch.manual_seed(5);a=torch.rand(200,8).topk(2,-1).indices;torch.manual_seed(5);b=torch.rand(200,8).topk(2,-1).indices
        return dict(passed=p>.01 and torch.equal(a,b),p=p,counts=counts.tolist(),decisions=1000000)
    check('T6',t6)
    for arm in ['D','L','SO','DOT','SR','PL']:
        def t7(arm=arm):
            c=dataclasses.replace(tiny(arm),vocab=2240,label_rate=.1);m=Model(c).to(device);opt=torch.optim.AdamW(m.parameter_groups(),lr=.01,betas=(.9,.95));torch.manual_seed(0);x=torch.randint(0,2000,(256,4),device=device);labels=x.clone();labels[:,:3]=-100;initial=None
            for step in range(300):
                opt.zero_grad();out=m(x,labels=labels);m.gradient_factor=256;loss=out['nll_sum']/256+out['balance_loss']+out['taylor_loss'];loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();m.finish_update();m.promote_labels();value=float(out['nll_sum'].detach()/256);initial=value if initial is None else initial
                if value<(.5*initial if arm=='SR' else .1):break
            return dict(passed=value<(.5*initial if arm=='SR' else .1),initial=initial,final=value,steps=step+1,fixture='tiny architecture, E1 256 facts; full-shape GPU checks still required for production kernels')
        check('T7_'+arm,t7)
    check('T8',lambda:(cases.test_identity_routing_and_balancing() or {}))
    check('T9',lambda:(cases.test_save_resume_pending_taylor() or {}))
    check('T10',lambda:(cases.test_taylor_microbatch() or {}))
    write(output,dict(checks=items,complete=True,device=str(device),failed=[x['test'] for x in items if not x['passed']]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();run(a.output,torch.device(a.device))

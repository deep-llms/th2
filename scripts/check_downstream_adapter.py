"""GPU gate: checkpoint adapter versus the existing packed inference path."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.bdot_harness import make_harness
from packed_training import PackedCollator, model_inputs
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise ValueError('Choose fresh output')
    reports={}
    for name,checkpoint in json.loads(Path(a.checkpoints).read_text()).items():
        lm=make_harness(checkpoint,device='cuda:0',batch_size=2,max_length=1024,
                        attention_backend='fa4',precision='bf16')
        assert int(lm.model.completed_steps)==10000 and lm.model.evaluation_policy=='target'
        before={k:v.clone() for k,v in lm.model.named_buffers() if k in
                ('balance_bias','completed_steps','routing_counter','training_totals')}
        tokens=[lm.tok_encode(t) for t in ['The city is near the sea. This is a short test.',
                                          'A large bird flew above the quiet forest.']]
        # EOS inside a request is not a segmentation signal.
        tokens[0].insert(3,lm.tokenizer.eos_token_id)
        length=max(map(len,tokens))
        ids=torch.tensor([t+[0]*(length-len(t)) for t in tokens],device='cuda')
        rows=[dict(input_ids=t,labels=t,attention_mask=[1]*length,segments=[0]*length)
              for t in ids.cpu().tolist()]
        batch=PackedCollator(attention_backend='fa4')(rows);batch.pop('labels')
        batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        lm.model.capture_routes=True
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            expected=lm.model(**model_inputs(batch)).logits
        routes=lm.model.last_routes.clone()
        actual=lm._model_call(ids)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        assert torch.equal(routes,lm.model.last_routes)
        lm.model.capture_routes=False
        requests=[(None,t[:3],t[3:]) for t in tokens]
        observed=lm._loglikelihood_tokens(requests)
        # Use the harness's exact input truncation/right-padding convention.
        n=max(len(t)-1 for t in tokens)
        score_ids=torch.tensor([t[:-1]+[0]*(n-len(t)+1) for t in tokens],device='cuda')
        with torch.inference_mode():logp=lm._model_call(score_ids).float().log_softmax(-1)
        gaps=[]
        for i,t in enumerate(tokens):
            logits=logp[i,2:len(t)-1];target=torch.tensor(t[3:],device='cuda')
            value=logits.gather(-1,target[:,None]).sum().item()
            gaps.append(abs(value-observed[i][0]))
            assert gaps[-1] < 1e-4, gaps
        for k,v in before.items():torch.testing.assert_close(dict(lm.model.named_buffers())[k],v,atol=0,rtol=0)
        reports[name]=dict(status='ok',packed_logits_max_abs=0.,routes_equal=True,
                          max_likelihood_error=max(gaps),master_dtype=str(next(lm.model.parameters()).dtype))
        del lm,actual,expected,logp,batch,before
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='ok',checkpoints=reports),indent=2)+'\n')


if __name__=='__main__':main()

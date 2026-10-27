"""Real-checkpoint gate: native logits, padding, and manual continuation likelihood."""
import argparse
import json
from pathlib import Path
import torch
from eval.kv_harness import make_harness


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise ValueError('Choose fresh output')
    reports={}
    for name,checkpoint in json.loads(Path(a.checkpoints).read_text()).items():
        lm=make_harness(checkpoint,tokenizer_path=a.tokenizer,device='cuda:0',batch_size=2,max_length=1024,precision='bf16')
        tokens=[lm.tok_encode(t) for t in ['The city is near the sea. This is a short test.',
                                          'A large bird flew above the quiet forest.']]
        tokens[0].insert(3,lm.tokenizer.eos_token_id)
        # Harness sorts by descending input length; preserve that batch shape here.
        tokens.sort(key=lambda t:(-len(t),tuple(t)))
        length=max(map(len,tokens))
        ids=torch.tensor([t+[0]*(length-len(t)) for t in tokens],device='cuda')
        with torch.inference_mode():expected=lm.model(input_ids=ids,use_cache=False).logits
        actual=lm._model_call(ids)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        requests=[(None,t[:3],t[3:]) for t in tokens]
        observed=lm._loglikelihood_tokens(requests)
        n=max(len(t)-1 for t in tokens)
        score_ids=torch.tensor([t[:-1]+[0]*(n-len(t)+1) for t in tokens],device='cuda')
        logp=lm._model_call(score_ids).float().log_softmax(-1)
        gaps=[]
        for i,t in enumerate(tokens):
            logits=logp[i,2:len(t)-1];target=torch.tensor(t[3:],device='cuda')
            value=logits.gather(-1,target[:,None]).sum().item()
            gaps.append(abs(value-observed[i][0]))
            if gaps[-1] > 1e-4:raise ValueError('Manual likelihood mismatch: '+str(gaps))
        reports[name]=dict(status='ok',native_logits_max_abs=0.,max_likelihood_error=max(gaps),
                          weight_dtype=str(next(lm.model.parameters()).dtype),kv_sources=lm.model.config.kv_sources)
        del lm,actual,expected,logp
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(dict(status='ok',checkpoints=reports),f,indent=2)


if __name__=='__main__':main()

"""Native A2 scoring of canonical, pre-tokenized likelihood requests.

No lm-eval dependency or environment changes. A separate CPU harness stage must
prepare the requests and later aggregate the returned per-request scores. Input
identity/tokenization and checkpoint receipts remain the coordinator's job.
"""
import torch
from efficient_llm.a2.model import assert_native, autocast


@torch.inference_mode()
def score_requests(model, requests, *, batch_size=8, max_length=4096):
    """Return (log-probability sum, exact greedy match) in original request order.

    Each request contains context and continuation token-ID lists. Empty context
    gets EOS0, matching the pinned GPT-NeoX harness prefix. Plain causal attention
    spans EOS inside each request; right padding never influences scored tokens.
    """
    assert_native(model)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Require native FP32 parameters with BF16 CUDA autocast')
    if batch_size < 1 or not 2 <= max_length <= min(4096,model.config.max_position_embeddings):
        raise ValueError('Invalid batch/context')
    model.eval()
    prepared=[]; truncated=0
    for index,request in enumerate(requests):
        context,continuation=request['context'],request['continuation']
        if (not isinstance(context,list) or not isinstance(continuation,list)
                or not 1 <= len(continuation) <= max_length
                or any(type(t) is not int or not 0 <= t < model.config.vocab_size
                       for t in context+continuation)):
            raise ValueError('Invalid canonical token request')
        context=context or [0]
        tokens=context+continuation
        truncated+=len(tokens)>max_length+1
        inputs=tokens[-(max_length+1):][:-1]
        prepared.append((index,inputs,continuation))
    prepared.sort(key=lambda row:-len(row[1]))
    results=[None]*len(prepared)
    for start in range(0,len(prepared),batch_size):
        batch=prepared[start:start+batch_size]
        ids=torch.zeros((len(batch),len(batch[0][1])),dtype=torch.long,device=model.device)
        for row,(_,inputs,_) in enumerate(batch):
            ids[row,:len(inputs)]=torch.tensor(inputs,device=model.device)
        with autocast(model.device):
            logits=model(input_ids=ids,use_cache=False,
                position_ids=torch.arange(ids.shape[1],device=ids.device).expand_as(ids)).logits
        for row,(index,inputs,target) in enumerate(batch):
            # Slice before FP32 log-softmax to avoid allocating padding/context
            # probabilities. The vocabulary normalization is still complete.
            selected=logits[row,len(inputs)-len(target):len(inputs)].float()
            wanted=torch.tensor(target,device=ids.device)
            logp=selected.log_softmax(-1).gather(1,wanted[:,None]).sum()
            if not torch.isfinite(logp):raise FloatingPointError('Nonfinite likelihood')
            results[index]=(float(logp),bool(torch.equal(selected.argmax(-1),wanted)))
        del logits
    return dict(scores=results,likelihood_requests=len(prepared),truncated_requests=truncated,
                precision='fp32_parameters_bf16_cuda_autocast',max_length=max_length)
